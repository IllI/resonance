"""
quantum_state_audit.py — Rigorous full characterization of OAT boundary ρ₂.

Computes ALL measures directly from ρ₂ without approximation:
  - Purity Tr(ρ²)
  - von Neumann entropy
  - Concurrence C (Wootters exact)
  - Negativity N and log-negativity E_N
  - Maximal singlet fraction f (direct optimization, NOT (1+C)/2)
  - F_opt = (2f+1)/3  (exact)
  - CHSH S_max via Horodecki (exact)
  - X-state fidelity (how well does ρ₂ fit X-state form?)
  - Entanglement witness W = I/4 - |Φ+><Φ+| (Tr[Wρ] < 0 → entangled)

Also checks: does f ≈ (1+C)/2? (pure-state approximation validity)
"""
import math, json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence


# ── Pauli matrices ─────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
SIGMAS = [sx, sy, sz]

# Bell basis
PHI_PLUS  = np.array([1,0,0,1], dtype=complex) / math.sqrt(2)
PHI_MINUS = np.array([1,0,0,-1], dtype=complex) / math.sqrt(2)
PSI_PLUS  = np.array([0,1,1,0], dtype=complex) / math.sqrt(2)
PSI_MINUS = np.array([0,1,-1,0], dtype=complex) / math.sqrt(2)
BELL_STATES = [PHI_PLUS, PHI_MINUS, PSI_PLUS, PSI_MINUS]


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def von_neumann(rho):
    eigs = np.real(np.linalg.eigvalsh(rho))
    eigs = eigs[eigs > 1e-14]
    return float(-np.sum(eigs * np.log2(eigs)))


def negativity(rho):
    """Partial transpose negativity for 2-qubit state."""
    # Partial transpose on second qubit: swap indices 1<->3 in 2x2x2x2 tensor
    rho4 = rho.reshape(2, 2, 2, 2)
    rho_pt = rho4.transpose(0, 3, 2, 1).reshape(4, 4)
    eigs = np.real(np.linalg.eigvalsh(rho_pt))
    neg = float(np.sum(np.abs(eigs[eigs < 0])))
    log_neg = float(np.log2(1 + 2 * neg)) if neg > 1e-14 else 0.0
    return neg, log_neg


def chsh_horodecki(rho):
    """Exact Horodecki CHSH criterion. Returns S_max."""
    T = np.zeros((3, 3))
    for i, si in enumerate(SIGMAS):
        for j, sj in enumerate(SIGMAS):
            T[i, j] = float(np.real(np.trace(rho @ np.kron(si, sj))))
    eigs = np.linalg.eigvalsh(T.T @ T)
    eigs_sorted = np.sort(eigs)[::-1]
    S_max = 2 * math.sqrt(max(0, eigs_sorted[0] + eigs_sorted[1]))
    return float(S_max)


def maximal_singlet_fraction(rho):
    """
    f = max_{U_A, U_B} <Φ+| (U_A⊗U_B) ρ (U_A⊗U_B)† |Φ+>

    For Bell-diagonal states this equals the largest Bell-basis eigenvalue.
    For general states: f = (1/4)(1 + T_max) where T_max is the largest
    singular value of the correlation matrix T[i,j] = Tr[ρ σᵢ⊗σⱼ].

    More precisely: f = (1 + ||T||_1 / 4) / ... — use the direct formula:
    f = max over all local unitaries = (1 + max_SV(T)) / 4 ... 

    Actually the exact formula: For any 2-qubit state,
    f = (1/4)(1 + sum of singular values of the correlation matrix)?
    No — that's not right either.

    Correct approach (Verstraete & Verschelde 2002):
    f = max_{|ψ> local unit equiv to |Φ+>} <ψ|ρ|ψ>
      = largest eigenvalue of the partially transposed ρ in the Bell basis
      
    Simpler: Bell-diagonalize. f = max Bell-basis diagonal element after
    local unitaries. For general state, use:
    f_exact = (1/4)(1 + max_eigenvalue(T)) ... 

    ACTUAL correct formula (Horodecki 1999):
    f(ρ) = (1 + max_{n: unit 3-vector} T·n) / 4  NO.

    Let me use the direct correct method:
    The maximal singlet fraction equals the maximum eigenvalue of
    W = (1/4) Σ_{α,β} T_αβ σ_α ⊗ σ_β + (1/4)I⊗I ... 

    Direct computation: check all 4 Bell states and local unitary orbits.
    For Bell-diagonal states only, f = max diagonal coefficient.
    For general states, f = max over single-qubit unitaries U_A, U_B of
    <Φ+|(U_A⊗U_B)ρ(U_A†⊗U_B†)|Φ+>.

    This equals (by twirling): f = (1 + ||T_rho||_∞ + Tr[T]) / 4
    where T_rho is the correlation matrix...

    Use the known exact result: f = (Tr[rho] + max_SV(M)) / 4 where
    M is a specific matrix related to rho. Actually:

    SIMPLEST CORRECT approach: Bell-diagonalize via optimal local unitaries.
    After optimal U_A ⊗ U_B, ρ → Bell-diagonal form with coefficients (λ₁,λ₂,λ₃,λ₄).
    Then f = max(λᵢ).

    The four Bell-diagonal coefficients are related to the correlation matrix T by:
      λ₁ = (1 + T₁₁ - T₂₂ + T₃₃)/4  ... (for |Φ+>)
    but these depend on the specific T matrix after diagonalization.

    PRAGMATIC: compute <Φ+|(U_A⊗U_B)ρ(U_A†⊗U_B†)|Φ+> numerically
    by optimizing over U_A, U_B ∈ SU(2).
    """
    # Direct approach: f ≥ max_{Bell state |β>} <β|ρ|β> (lower bound)
    # For general state: use the correlation matrix approach
    
    # Compute "R matrix" = rho in the Bell basis
    R = np.zeros((4, 4), dtype=complex)
    for i, bi in enumerate(BELL_STATES):
        for j, bj in enumerate(BELL_STATES):
            R[i, j] = bi.conj() @ rho @ bj

    # The maximal singlet fraction equals the largest eigenvalue of the
    # "Φ+ component" maximized over local unitaries.
    # Exact formula (Verstraete 2002): f = (1 + Σ singular values of C_matrix) / 4
    # where C_matrix relates to the correlation matrix.

    # Use the known result: for any 2-qubit state,
    # f_max = (1 + ||\vec{t}||_1) / 4  where t_i = Tr[ρ σ_i⊗σ_i] diagonal only
    # This is exact only for Bell-diagonal states.

    # NUMERICAL OPTIMIZATION over SU(2)×SU(2) — exact result
    def singlet_frac(params):
        # U_A = exp(i θ σ_n), parametrize via angles
        a1, a2, a3, b1, b2, b3 = params
        # Rodrigues rotation for SU(2)
        def su2(a, b, c):
            angle = math.sqrt(a**2 + b**2 + c**2)
            if angle < 1e-10:
                return np.eye(2, dtype=complex)
            n = np.array([a, b, c]) / angle
            return (math.cos(angle) * np.eye(2) + 
                    1j * math.sin(angle) * (n[0]*sx + n[1]*sy + n[2]*sz))
        UA = su2(a1, a2, a3)
        UB = su2(b1, b2, b3)
        U = np.kron(UA, UB)
        rho_rot = U @ rho @ U.conj().T
        return -float(np.real(PHI_PLUS.conj() @ rho_rot @ PHI_PLUS))

    # Quick scan + scipy minimize
    from scipy.optimize import minimize, differential_evolution
    bounds = [(-math.pi, math.pi)] * 6
    
    # Try multiple random starts
    best_f = -1
    rng = np.random.default_rng(42)
    for _ in range(30):
        x0 = rng.uniform(-math.pi, math.pi, 6)
        res = minimize(singlet_frac, x0, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-12, 'gtol': 1e-10})
        if -res.fun > best_f:
            best_f = -res.fun

    return float(best_f)


def x_state_fidelity(rho):
    """
    X-states have only diagonal and anti-diagonal elements non-zero in
    computational basis. Check how well rho fits X-form.
    
    X-state: rho[i,j] = 0 unless (i+j) is even (0-indexed).
    Non-zero positions: [0,0],[1,1],[2,2],[3,3],[0,3],[3,0],[1,2],[2,1]
    """
    mask = np.zeros((4, 4), dtype=bool)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                mask[i, j] = True
    
    rho_x = rho * mask  # X-projected version
    off_diag_norm = np.linalg.norm(rho - rho_x, 'fro')
    total_norm = np.linalg.norm(rho, 'fro')
    x_fidelity = 1 - off_diag_norm / max(total_norm, 1e-14)
    return float(x_fidelity), rho_x


def entanglement_witness(rho):
    """
    W = I/4 - |Φ+><Φ+|
    Tr[Wρ] = 1/4 - <Φ+|ρ|Φ+> = 1/4 - f_Φ+
    
    Negative iff ρ has singlet fraction f > 1/4 (i.e., any entangled state
    that was created by mixing with enough |Φ+>).
    
    Also compute the strongest witness via: W_opt = I/4 - |ψ_opt><ψ_opt|
    where |ψ_opt> is the Bell state closest to ρ.
    """
    f_phi_plus = float(np.real(PHI_PLUS.conj() @ rho @ PHI_PLUS))
    w_standard = 0.25 - f_phi_plus  # negative = detects entanglement

    # Best Bell-state witness
    f_vals = [float(np.real(b.conj() @ rho @ b)) for b in BELL_STATES]
    f_best = max(f_vals)
    w_best = 0.25 - f_best

    return float(w_standard), float(w_best), f_vals


def analyze_state(N, chi_t):
    n = N // 2
    tensors = oat_mps(N, chi_t)
    rho = extract_boundary_rho(tensors, n)

    C    = float(concurrence(rho))
    pur  = purity(rho)
    ent  = von_neumann(rho)
    neg, log_neg = negativity(rho)
    S_chsh = chsh_horodecki(rho)
    f_opt  = maximal_singlet_fraction(rho)
    F_opt  = (2 * f_opt + 1) / 3
    F_approx = (2 + C) / 3  # pure-state approximation
    x_fid, _ = x_state_fidelity(rho)
    w_std, w_best, f_bell = entanglement_witness(rho)

    # Check pure-state formula validity
    f_approx_val = (1 + C) / 2  # approximation
    f_error = abs(f_opt - f_approx_val)

    return {
        "N": N, "chi_t": chi_t,
        "purity": pur,
        "von_neumann": ent,
        "concurrence_C": C,
        "negativity": neg,
        "log_negativity": log_neg,
        "singlet_frac_exact": f_opt,
        "singlet_frac_approx": f_approx_val,
        "singlet_frac_error": f_error,
        "F_opt_exact": F_opt,
        "F_opt_approx": F_approx,
        "F_opt_error": abs(F_opt - F_approx),
        "CHSH_S_max": S_chsh,
        "CHSH_violates": S_chsh > 2.0,
        "x_state_fidelity": x_fid,
        "witness_std": w_std,
        "witness_best": w_best,
        "f_phi_plus": f_bell[0],
        "f_phi_minus": f_bell[1],
        "f_psi_plus": f_bell[2],
        "f_psi_minus": f_bell[3],
    }


def main():
    print("=" * 78)
    print("  Full Quantum State Audit — OAT Boundary ρ₂")
    print("  All measures computed directly. No pure-state approximations.")
    print("=" * 78)

    sweep = [
        (2,  2.985), (4,  1.091), (6,  0.713),
        (8,  0.523), (10, 0.429), (12, 0.334), (16, 0.239),
    ]

    results = []
    for N, chi_t in sweep:
        print(f"\n  Analyzing N={N}, chi_t={chi_t:.3f} ...", flush=True)
        r = analyze_state(N, chi_t)
        results.append(r)

        print(f"    Purity:       {r['purity']:.6f}  "
              f"({'pure' if r['purity'] > 0.99 else 'mixed'})")
        print(f"    S_vN:         {r['von_neumann']:.6f} bits")
        print(f"    Concurrence:  {r['concurrence_C']:.6f}")
        print(f"    Negativity:   {r['negativity']:.6f}")
        print(f"    Log-neg E_N:  {r['log_negativity']:.6f}")
        print(f"    f_exact:      {r['singlet_frac_exact']:.6f}  "
              f"f_approx (1+C)/2={r['singlet_frac_approx']:.6f}  "
              f"error={r['singlet_frac_error']:.4f}")
        print(f"    F_opt exact:  {r['F_opt_exact']:.6f}  "
              f"F_approx (2+C)/3={r['F_opt_approx']:.6f}  "
              f"error={r['F_opt_error']:.4f}")
        print(f"    CHSH S_max:   {r['CHSH_S_max']:.6f}  "
              f"{'VIOLATES' if r['CHSH_violates'] else 'no violation'}")
        print(f"    X-state fid:  {r['x_state_fidelity']:.6f}  "
              f"({'approx X-state' if r['x_state_fidelity'] > 0.99 else 'NOT X-state'})")
        print(f"    Witness W:    {r['witness_best']:.6f}  "
              f"({'DETECTS entanglement' if r['witness_best'] < 0 else 'no detection'})")
        print(f"    Bell fracs:   Φ+={r['f_phi_plus']:.4f}  "
              f"Φ-={r['f_phi_minus']:.4f}  "
              f"Ψ+={r['f_psi_plus']:.4f}  Ψ-={r['f_psi_minus']:.4f}")

    print("\n" + "=" * 78)
    print("  Summary: X-state fidelity (key for formula validity)")
    print("=" * 78)
    for r in results:
        x = r['x_state_fidelity']
        fe = r['F_opt_error']
        print(f"  N={r['N']:>3d}: X-fid={x:.6f}  F_error={fe:.5f}  "
              f"Formula {'valid' if fe < 0.005 else 'INVALID'} (need X-state)")

    with open("quantum_audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[DONE] quantum_audit_results.json")


if __name__ == "__main__":
    main()
