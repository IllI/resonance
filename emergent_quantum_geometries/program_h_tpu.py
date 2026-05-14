"""
program_h_tpu.py -- Finite-Size Scaling on TPU (Program H, N=4..768)
=====================================================================
Self-contained TPU script. No local package imports.
Uses exact closed-form rho2(chi_t, N) -- no MPS needed.

ROTATION METHOD: Schmidt rotation from dominant eigenvector of rho2.
  Matches oat_teleport_v2_tpu.py (established convention):
    psi_dom = dominant eigenvector of rho2 (eigh)
    M = psi_dom.reshape(2,2)          # 2x2 coefficient matrix
    U, sigma, Vh = np.linalg.svd(M)
    U_A = U.conj().T                  # U-dagger (Alice)
    U_B = Vh.conj()                   # Vh.conj() -- NOT Vh-dagger (Bob)
  Maps OAT state to Schmidt canonical form: sigma_1|00> + sigma_2|11>
  C_schmidt = 2*sigma_1*sigma_2
  (Paper Section III.B: R_z phase alignment + R_x Rabi pulse = Schmidt steps)

THREE OBSERVABLES per (N, chi_t):
  F_std   : standard Bell protocol, no rotation        [analytic: (1+T_xx)/... ]
  F_sch   : Schmidt-rotated protocol                   [Haar simulation, n=K]
  F_opt   : SU(2)xSU(2) upper bound from T-SVD        [analytic: (2*f_max+1)/3]
            f_max = (1 + sv1 + sv2 + sv3) / 4

Paper Theorems:
  [PROVED] T_yy = T_zz = 0 for all N, chi_t (Theorems 1,2)
  [PROVED] T_xx = cos^{N-2}(chi_t/2)         (Theorem 3)
  [PROVED] F_avg(R_z only) <= 2/3             (Corollary to Theorems 1-2)
  [PROVED] rho2(chi_t=pi) = I/4, F=1/2       (Theorem 4)
  [OBSERVED] F_opt^direct = 0.719 N=4         (paper Table III.C)

Upload and run:
  python3 program_h_tpu.py [--N-max 768] [--K 300] [--n-chi 100]

Saves: program_h_tpu_results.json + per-N sweep files + .zip
"""

import json, time, os, zipfile, math, cmath
import numpy as np

# ---- Try JAX, fall back to numpy ----------------------------------------
try:
    import jax
    import jax.numpy as jnp
    USE_JAX = True
    print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
except ImportError:
    USE_JAX = False
    jnp = np
    print("[DEVICE] JAX not available -- using numpy", flush=True)

# ---- Gates (numpy, for teleportation circuit) ---------------------------
I2   = np.eye(2, dtype=complex)
X    = np.array([[0,1],[1,0]], dtype=complex)
Z    = np.array([[1,0],[0,-1]], dtype=complex)
H2   = np.array([[1,1],[1,-1]], dtype=complex) / math.sqrt(2)
P0   = np.array([[1,0],[0,0]], dtype=complex)
P1   = np.array([[0,0],[0,1]], dtype=complex)

def kron3(A, B, C): return np.kron(np.kron(A, B), C)

# Standard Bell-measurement circuit (on 3-qubit system: input + resource pair)
CNOT_01  = kron3(P0, I2, I2) + kron3(P1, X, I2)
CIRCUIT  = kron3(H2, I2, I2) @ CNOT_01
CIRCUIT_DAG = CIRCUIT.conj().T
PROJ = {(m1, m2): kron3(P0 if m1==0 else P1,
                         P0 if m2==0 else P1, I2)
        for m1 in range(2) for m2 in range(2)}
UCORR = {(0,0): I2, (0,1): X, (1,0): Z, (1,1): X @ Z}

# ---- Exact closed-form rho2(chi_t, N) -----------------------------------
# Proposition 1 (paper): rho_{(iL iR),(jL jR)} =
#   (1/4) * exp(-i*chi_t*[(iL-1/2)(iR-1/2)-(jL-1/2)(jR-1/2)])
#         * cos^{N/2-1}((iR-jR)*chi_t/2)
#         * cos^{N/2-1}((iL-jL)*chi_t/2)
# Basis: 00, 01, 10, 11
IDX = [(0,0),(0,1),(1,0),(1,1)]

def rho2_exact(chi_t, N, gamma_t=0.0):
    """
    4x4 boundary density matrix. Hamming-distance dephasing included.
    gamma_t = Gamma * t (dimensionless); off-diag(d_H) decays as exp(-2*dH*gamma_t).
    """
    rho = np.zeros((4, 4), dtype=complex)
    exp_pow = max(0, N//2 - 1)          # cos exponent (0 for N=2)
    HAmm = np.array([[0,1,1,2],[1,0,2,1],[1,2,0,1],[2,1,1,0]])  # Hamming distances

    for a, (iL, iR) in enumerate(IDX):
        for b, (jL, jR) in enumerate(IDX):
            phase = -1j * chi_t * ((iL-0.5)*(iR-0.5) - (jL-0.5)*(jR-0.5))
            dR = iR - jR; dL = iL - jL
            cos_R = math.cos(dR * chi_t / 2) ** exp_pow if exp_pow > 0 else 1.0
            cos_L = math.cos(dL * chi_t / 2) ** exp_pow if exp_pow > 0 else 1.0
            decay = math.exp(-2.0 * HAmm[a, b] * gamma_t)
            rho[a, b] = 0.25 * cmath.exp(phase) * cos_R * cos_L * decay
    return rho

# ---- Entanglement diagnostics -------------------------------------------
def concurrence(rho):
    """Wootters concurrence (full formula)."""
    sy2 = np.kron(np.array([[0,-1j],[1j,0]]), np.array([[0,-1j],[1j,0]]))
    R = rho @ (sy2 @ rho.conj() @ sy2)
    eigs = np.sqrt(np.abs(np.linalg.eigvals(R).real).clip(0))
    eigs = np.sort(eigs)[::-1]
    return float(max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3]))

def compute_T_matrix(rho):
    """3x3 correlation tensor T_ij = Tr(rho sigma_i x sigma_j)."""
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    paulis = [sx, sy, sz]
    T = np.zeros((3,3))
    for i, si in enumerate(paulis):
        for j, sj in enumerate(paulis):
            T[i,j] = float(np.real(np.trace(np.kron(si, sj) @ rho)))
    return T

def f_opt_analytic(rho):
    """f_max = (1+sv1+sv2+sv3)/4 from T-matrix SVD. Returns (f_max, svs)."""
    T = compute_T_matrix(rho)
    svs = np.linalg.svd(T, compute_uv=False)
    return float((1.0 + svs.sum()) / 4.0), svs

def spectral_entropy(svs):
    s = svs / (svs.sum() + 1e-15)
    return float(-np.sum(s * np.log(s + 1e-15)))

# ---- Schmidt rotation (dominant eigenvector SVD) -------------------------
# From oat_teleport_v2_tpu.py -- the established convention.
def schmidt_rotate(rho2):
    """
    Rotate rho2 to Schmidt canonical form using dominant eigenvector.

    For dominant eigvec |psi_dom>, coefficient matrix M = psi_dom.reshape(2,2).
    SVD: M = U diag(sigma) Vh.
    Local unitaries:
      U_A = U.conj().T  (U-dagger: maps left Schmidt vectors to standard basis)
      U_B = Vh.conj()   (Vh.conj(), NOT Vh-dagger: maps right Schmidt vectors)
    Proof: (U_A x U_B)|psi_dom> = sigma_1|00> + sigma_2|11>
           See oat_teleport_v2_tpu.py lines 74-84.

    Returns: rho2_rotated, C_schmidt, U_A, U_B
    """
    eigvals, eigvecs = np.linalg.eigh(rho2)
    psi_dom = eigvecs[:, -1]          # dominant eigenvector
    M = psi_dom.reshape(2, 2)

    U, sigma, Vh = np.linalg.svd(M)

    U_A = U.conj().T                  # U-dagger
    U_B = Vh.conj()                   # Vh.conj() -- NOT Vh.conj().T

    U_loc = np.kron(U_A, U_B)
    rho2_rot = U_loc @ rho2 @ U_loc.conj().T

    C_sch = float(2.0 * abs(sigma[0]) * abs(sigma[1]))
    return rho2_rot, C_sch, U_A, U_B

# ---- Teleportation circuit (from oat_teleport_v2_tpu.py) ----------------
def teleport_fidelity(psi_A, rho2):
    """F = <psi_A|rho_B_out|psi_A>. Applies standard Bell measurement + corrections."""
    rho_total = np.kron(np.outer(psi_A, psi_A.conj()), rho2)
    rho_bell  = CIRCUIT @ rho_total @ CIRCUIT_DAG
    rho_B_out = np.zeros((2,2), dtype=complex)
    for (m1, m2), Pi in PROJ.items():
        rho_proj = Pi @ rho_bell @ Pi
        prob = float(np.trace(rho_proj).real)
        if prob < 1e-12: continue
        r = (rho_proj / prob).reshape(2,2,2,2,2,2)
        rho_q2 = np.einsum('ijkijl->kl', r)
        U = UCORR[(m1, m2)]
        rho_B_out += prob * (U @ rho_q2 @ U.conj().T)
    return float(np.clip((psi_A.conj() @ rho_B_out @ psi_A).real, 0.0, 1.0))

def haar_qubit(rng):
    u1, u2 = rng.uniform(0, 1, 2)
    th = math.acos(1.0 - 2.0*u1)
    ph = 2.0 * math.pi * u2
    return np.array([math.cos(th/2),
                     math.sin(th/2) * cmath.exp(1j*ph)], dtype=complex)

def avg_fidelity_haar(rho2, K=300, seed=42):
    """Haar-averaged teleportation fidelity via circuit simulation."""
    rng = np.random.default_rng(seed)
    fids = [teleport_fidelity(haar_qubit(rng), rho2) for _ in range(K)]
    return float(np.mean(fids)), float(np.std(fids))

# ---- Analytic F_std (from paper Theorem 3 + Corollary) ------------------
def F_std_analytic(rho):
    """
    Standard Bell protocol fidelity (no rotation), analytic.
    Paper Corollary: F_avg(R_z only) = (1 + T_xx/3) / 2 <= 2/3.
    Equivalently: F_std = (2*f_Phi+ + 1) / 3
    where f_Phi+ = (1 + T_xx - T_yy + T_zz) / 4.
    Since T_yy = T_zz = 0 (Theorems 1,2): F_std = (1 + T_xx/3) / 2.
    """
    T = compute_T_matrix(rho)
    T_xx = T[0, 0]
    return float((1.0 + T_xx / 3.0) / 2.0)

# ---- Per-N sweep ---------------------------------------------------------
def analyze_chi_t_point(chi_t, N, K=300):
    """Full analysis for one (chi_t, N). K = Haar samples for F_sch."""
    rho = rho2_exact(chi_t, N, 0.0)
    C   = concurrence(rho)
    T   = compute_T_matrix(rho)
    fm, svs = f_opt_analytic(rho)
    F_opt_val = (2.0 * fm + 1.0) / 3.0
    F_std_val = F_std_analytic(rho)

    # Schmidt rotation and teleportation fidelity
    rho_rot, C_sch, U_A, U_B = schmidt_rotate(rho)
    F_sch_val, F_sch_std = avg_fidelity_haar(rho_rot, K=K)

    rank_T = int((svs > 1e-3).sum())
    H_T    = spectral_entropy(svs)
    delta_F = F_sch_val - F_std_val

    return {
        "chi_t": float(chi_t),
        "N": N,
        "C": float(C),
        "C_schmidt": float(C_sch),
        "F_std": float(F_std_val),
        "F_sch": float(F_sch_val),
        "F_sch_std": float(F_sch_std),
        "F_opt": float(F_opt_val),
        "f_max": float(fm),
        "delta_F": float(delta_F),
        "delta_F_opt": float(F_opt_val - F_std_val),
        "T_xx": float(T[0,0]), "T_yy": float(T[1,1]), "T_zz": float(T[2,2]),
        "rank_T": rank_T,
        "H_T": float(H_T),
        "svs": svs.tolist(),
    }

def sweep_N(N, n_chi=100, K=300):
    """Full chi_t sweep for one N."""
    chi_vals = np.linspace(0.01, math.pi - 0.01, n_chi)
    sweep = []
    for chi_t in chi_vals:
        r = analyze_chi_t_point(chi_t, N, K=K)
        sweep.append(r)

    # Best chi_t* = maximizes F_sch
    best_idx = int(np.argmax([r["F_sch"] for r in sweep]))
    best     = sweep[best_idx]

    # Transport stratum width: region where F_opt > 2/3
    above = [r["chi_t"] for r in sweep if r["F_opt"] > 2.0/3.0]
    delta_chi = (max(above) - min(above)) if len(above) > 1 else 0.0

    # Verify internal null: chi_t = pi -> F = 1/2 (Theorem 4)
    rho_pi = rho2_exact(math.pi - 0.001, N, 0.0)
    F_null = F_std_analytic(rho_pi)

    peak_C = float(max(r["C"] for r in sweep))

    return {
        "N": N,
        "chi_star": float(best["chi_t"]),
        "F_std_star": best["F_std"],
        "F_sch_star": best["F_sch"],
        "F_opt_star": best["F_opt"],
        "delta_F_star": best["delta_F"],
        "f_max_star": best["f_max"],
        "C_star": best["C"],
        "C_schmidt_star": best["C_schmidt"],
        "rank_T_star": best["rank_T"],
        "H_T_star": best["H_T"],
        "delta_chi": float(delta_chi),
        "peak_concurrence": float(peak_C),
        "F_null_analytic": float(F_null),   # should be ~0.5 (Theorem 4)
        "svs_star": best["svs"],
        "sweep": sweep,
    }

# ---- Power-law fit -------------------------------------------------------
def fit_powerlaw(Ns, vals):
    """y ~ A * N^(-alpha). Returns (A, alpha, R2)."""
    log_N = np.log(np.array(Ns, dtype=float))
    log_v = np.log(np.array(vals, dtype=float) + 1e-12)
    c     = np.polyfit(log_N, log_v, 1)
    pred  = np.polyval(c, log_N)
    ss_res = np.sum((log_v - pred)**2)
    ss_tot = np.sum((log_v - log_v.mean())**2)
    R2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    return float(np.exp(c[1])), float(-c[0]), R2

# ---- Main ---------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N-max",  type=int, default=256,
                   help="Max N (doubles from 4: 4,8,16,...)")
    p.add_argument("--n-chi",  type=int, default=80,
                   help="chi_t grid points per N")
    p.add_argument("--K",      type=int, default=300,
                   help="Haar samples for F_sch per chi_t point")
    p.add_argument("--out",    default="program_h_tpu_results.json")
    args = p.parse_args()

    N_vals = []
    N = 4
    while N <= args.N_max:
        N_vals.append(N); N *= 2

    print("="*70, flush=True)
    print("Program H (TPU) -- Finite-Size Scaling of Recoverable Transport")
    print(f"N values: {N_vals}  chi_t grid: {args.n_chi}  K(Haar): {args.K}")
    print("Rotation: Schmidt (dominant eigvec SVD, oat_teleport_v2 convention)")
    print("="*70, flush=True)

    t0 = time.time()
    all_results = []

    for N in N_vals:
        t_n = time.time()
        print(f"\n--- N={N} ---", flush=True)
        result = sweep_N(N, n_chi=args.n_chi, K=args.K)
        all_results.append(result)
        print(f"  chi_t*={result['chi_star']:.4f}  "
              f"F_std={result['F_std_star']:.4f}  "
              f"F_sch={result['F_sch_star']:.4f}  "
              f"F_opt={result['F_opt_star']:.4f}  "
              f"dF={result['delta_F_star']:.4f}  "
              f"Dchi={result['delta_chi']:.3f}  "
              f"C={result['C_star']:.4f}  "
              f"F(pi)={result['F_null_analytic']:.4f}  "
              f"({time.time()-t_n:.1f}s)", flush=True)

    # Scaling table
    print("\n" + "="*70)
    print("SCALING TABLE")
    print(f"{'N':>6} {'chi*':>7} {'F_std':>7} {'F_sch':>7} {'F_opt':>7} "
          f"{'dF':>6} {'Dchi':>6} {'C_peak':>7} {'rank':>5} {'F(pi)':>7}")
    print("-"*70)
    for r in all_results:
        print(f"{r['N']:>6} {r['chi_star']:>7.4f} {r['F_std_star']:>7.4f} "
              f"{r['F_sch_star']:>7.4f} {r['F_opt_star']:>7.4f} "
              f"{r['delta_F_star']:>6.4f} {r['delta_chi']:>6.3f} "
              f"{r['peak_concurrence']:>7.4f} {r['rank_T_star']:>5} "
              f"{r['F_null_analytic']:>7.4f}")

    # Power-law fits
    Ns     = [r['N']            for r in all_results]
    dFs    = [max(1e-6, r['delta_F_star']) for r in all_results]
    F_schs = [r['F_sch_star']   for r in all_results]
    Cs     = [r['peak_concurrence'] for r in all_results]
    chis   = [r['chi_star']     for r in all_results]

    print()
    if len(Ns) >= 3:
        A, alpha, R2 = fit_powerlaw(Ns, dFs)
        print(f"dF      ~ {A:.4f} * N^(-{alpha:.4f})  R2={R2:.4f}")
        A2, a2, R2_2 = fit_powerlaw(Ns, F_schs)
        print(f"F_sch   ~ {A2:.4f} * N^(-{a2:.4f})   R2={R2_2:.4f}")
        A3, a3, R2_3 = fit_powerlaw(Ns, Cs)
        print(f"C_peak  ~ {A3:.4f} * N^(-{a3:.4f})   R2={R2_3:.4f}")
        A4, a4, R2_4 = fit_powerlaw(Ns, chis)
        print(f"chi_t*  ~ {A4:.4f} * N^(-{a4:.4f})   R2={R2_4:.4f}")
        meta = {"powerlaw_alpha_dF": alpha, "powerlaw_alpha_Fsch": a2,
                "powerlaw_alpha_C": a3,     "powerlaw_alpha_chi": a4}
        for r in all_results: r.update(meta)

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s", flush=True)

    # Save summary JSON (no sweep arrays)
    summary = [{k:v for k,v in r.items() if k != 'sweep'} for r in all_results]
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {args.out}")

    # Per-N sweep files
    for r in all_results:
        fname = f"sweep_N{r['N']}.json"
        with open(fname, 'w') as f:
            json.dump(r['sweep'], f, indent=2)

    # Zip everything
    out_zip = args.out.replace('.json', '.zip')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
        for r in all_results:
            fname = f"sweep_N{r['N']}.json"
            if os.path.exists(fname):
                zf.write(fname)
    print(f"Zipped:  {out_zip}")


if __name__ == "__main__":
    main()
