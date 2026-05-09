"""
oat_teleport_v5d_witness_observer.py
D-LinOSS Observer on Witness Decay Series

The entanglement witness Tr[W·ρ(t)] is a LINEAR functional of ρ.
Under Lindblad dephasing, it decays as a weighted sum of exponentials:

    Tr[W·ρ(t)] = Σ_k  c_k · exp(-Γ · d_H(k)² · t)

where the sum runs over coherence pathways, d_H(k) is the Hamming distance
of pathway k, and c_k = W[k] · ρ[k](0) are the weighted initial coherences.

This is EXACTLY what D-LinOSS (damped sinusoid observer) is designed for:
- Linear observable ✓
- Sum of exponentials ✓
- No Wootters nonlinearity artifact ✓

Framework discrimination is now clean:
- Lindblad:   single mode, γ_k ≈ 4Γ (Hamming-2 dominant)
- MBL:        many small γ_k (power-law from distributed Hamming distances)
- SYK:        uniform γ_k ≈ λ_L (flat density of decay rates)
- Penrose OR: bimodal γ_k (pre/post collapse)

D-LinOSS observer structure:
  State: h(t) ∈ ℝ^K  (K hidden oscillatory modes)
  Obs:   y(t) = C · h(t)  (scalar witness)
  Dyn:   dh/dt = Λ·h  where Λ = diag(-γ_k + iω_k)

The observer is fit to the witness decay and its learned Λ spectrum
is the framework fingerprint.
"""
import math, json, sys, os
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.linalg import hankel, svd as scipy_svd
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence

BELLS = [
    np.array([1,0,0,1],  dtype=complex)/math.sqrt(2),
    np.array([1,0,0,-1], dtype=complex)/math.sqrt(2),
    np.array([0,1,1,0],  dtype=complex)/math.sqrt(2),
    np.array([0,1,-1,0], dtype=complex)/math.sqrt(2),
]


# ── Witness function (linear in ρ) ─────────────────────────────────────────

def best_witness(rho0):
    """
    W_opt = I/4 - |β_opt><β_opt| where β_opt is the Bell state closest to ρ₂.
    Tr[W_opt · ρ] = 1/4 - max_k f_k  where f_k = <β_k|ρ|β_k>.
    """
    f_vals = [float(np.real(b.conj() @ rho0 @ b)) for b in BELLS]
    k_opt = np.argmax(f_vals)
    beta_opt = BELLS[k_opt]
    W_op = np.eye(4, dtype=complex)/4 - np.outer(beta_opt, beta_opt.conj())
    return W_op, k_opt, f_vals[k_opt]


def dephase(rho2, gamma_t):
    basis = [(0,0),(0,1),(1,0),(1,1)]
    dH = np.array([[abs(a-c)+abs(b-d) for c,d in basis] for a,b in basis],
                  dtype=float)
    rho = rho2 * np.exp(-gamma_t * dH**2)
    tr = float(np.trace(rho).real)
    return rho / max(tr, 1e-14)


# ── Analytic witness decay (exact for any rho) ─────────────────────────────

def witness_decay_analytic(rho0, W_op, x_grid):
    """
    Compute Tr[W·ρ(x)] analytically by expanding in coherence pathways.
    Each matrix element ρ[i,j] decays as exp(-d_H(i,j)²·x).
    Tr[W·ρ(x)] = Σ_{ij} W[j,i] · ρ[i,j](0) · exp(-d_H(i,j)²·x)
    """
    basis = [(0,0),(0,1),(1,0),(1,1)]
    dH = np.array([[abs(a-c)+abs(b-d) for c,d in basis] for a,b in basis])
    W_decay = W_op * (rho0.T)  # W[j,i] * ρ[i,j]
    decay_rates = dH**2         # rate for each element

    y = np.array([float(np.real(np.sum(W_decay * np.exp(-x * decay_rates))))
                  for x in x_grid])
    return y


# ── D-LinOSS observer: fit sum of exponentials ─────────────────────────────

def dlinoss_fit(x, y, K=6):
    """
    D-LinOSS observer: fit y(x) = Σ_k A_k · exp(-(γ_k + i·ω_k)·x)
    using matrix pencil on the analytic witness decay (which IS a sum of
    exponentials with known rates d_H²·Γ for each coherence).

    Since the signal is clean (analytic, no noise), we can use direct
    matrix pencil without truncation issues.

    Returns: (omega_k, gamma_k, A_k, recon_error)
    """
    N = len(y)
    L = N // 2
    K = min(K, L-1)

    H = hankel(y[:L], y[L-1:])
    U, s, Vh = scipy_svd(H)

    # Adaptive K: keep modes with singular value > 1% of max
    s_thresh = s[0] * 0.01
    K_eff = max(1, int(np.sum(s > s_thresh)))
    K_eff = min(K_eff, K)

    U_k = U[:, :K_eff]
    Z = np.linalg.pinv(U_k[:-1, :]) @ U_k[1:, :]
    eigs = np.linalg.eigvals(Z)

    # Filter: keep only physically meaningful eigs (|eig| ≤ 1, i.e., decaying)
    eigs = eigs[np.abs(eigs) <= 1.0 + 1e-6]
    if len(eigs) == 0:
        eigs = np.array([np.exp(-4 * (x[1]-x[0]))])

    gamma_k = np.clip(-np.real(np.log(np.abs(eigs) + 1e-15)), 0, 100)
    omega_k = np.imag(np.log(eigs + 1e-15*(eigs==0)))

    # Amplitudes via least squares
    V = np.vander(eigs, N, increasing=True).T
    A_k, _, _, _ = np.linalg.lstsq(V, y.astype(complex), rcond=None)
    A_k = np.abs(A_k)

    y_recon = np.real(V @ np.abs(A_k) * np.exp(1j * np.angle(A_k + 1e-14)))
    recon_err = float(np.sqrt(np.mean((y - y_recon)**2)))

    return omega_k, gamma_k, A_k, recon_err, K_eff


# ── Framework model fits (on witness decay) ────────────────────────────────

def model_fits(x, y, y0):
    """Fit each framework model to normalized witness decay."""
    # Normalize: witness is negative, divide by signed y0 so yn → +1 at x=0
    yn = y / (y0 - 1e-14 * (y0 == 0))  # signed normalize: yn = exp(-b*x) > 0


    results = {}

    # Lindblad: single exponential
    try:
        p, _ = curve_fit(lambda t, b: np.exp(-b*t), x, yn,
                         p0=[4.0], bounds=([0.1],[50.0]), maxfev=5000)
        yf = np.exp(-p[0]*x)
        results["Lindblad"] = {"params": p.tolist(), "b": float(p[0]),
                               "residual": float(np.sqrt(np.mean((yn-yf)**2)))}
    except Exception:
        results["Lindblad"] = {"params": [], "b": 4.0, "residual": 1.0}

    # MBL: power law
    try:
        p, _ = curve_fit(lambda t, b, a: 1/(1+b*t)**a, x, yn,
                         p0=[1.0, 1.0], bounds=([0,0.1],[20,5]), maxfev=2000)
        yf = 1/(1+p[0]*x)**p[1]
        results["MBL"] = {"params": p.tolist(),
                          "residual": float(np.sqrt(np.mean((yn-yf)**2)))}
    except Exception:
        results["MBL"] = {"params": [], "residual": 1.0}

    # SYK: sub-exponential
    try:
        p, _ = curve_fit(lambda t, b: np.exp(-b*np.sqrt(t+1e-10)), x, yn,
                         p0=[2.0], bounds=([0],[20]), maxfev=2000)
        yf = np.exp(-p[0]*np.sqrt(x+1e-10))
        results["SYK"] = {"params": p.tolist(),
                          "residual": float(np.sqrt(np.mean((yn-yf)**2)))}
    except Exception:
        results["SYK"] = {"params": [], "residual": 1.0}

    # Heisenberg: constant
    results["Heisenberg"] = {"params": [], "residual": float(np.sqrt(np.mean((yn-1)**2)))}

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  v5d: Witness Decay as D-LinOSS Observer")
    print("  Tr[W·ρ(x)] = linear observable → clean mode decomposition")
    print("="*70)

    sweep = [(2,2.985),(4,1.091),(6,0.713),(8,0.523),(12,0.334),(16,0.239)]
    N_GAMMA, GT_MAX = 100, 1.5  # 1.5 Γt covers exp(-6)≈0.002: full decay range
    x_grid = np.linspace(0, GT_MAX, N_GAMMA)
    K_OBS  = 8  # D-LinOSS hidden state dimension

    all_results = []
    print("\n  Analytic prediction: Tr[W·ρ(x)] = Σ_k c_k·exp(-d_H(k)²·x)")
    print("  For Hamming distances 0,1,2: decay rates 0, Γ, 4Γ")
    print("  Dominant mode at d_H=2: γ_k ≈ 4 (in Γ·t units)\n")

    for N, chi_t in sweep:
        n = N//2
        tensors = oat_mps(N, chi_t)
        rho0 = extract_boundary_rho(tensors, n)
        C0 = float(concurrence(rho0))

        # Best witness operator for this state
        W_op, k_opt, f_opt = best_witness(rho0)
        W0 = float(np.real(np.trace(W_op @ rho0)))  # Tr[W·ρ₀]

        # Analytic witness decay (linear — no Wootters artifact)
        y_witness = witness_decay_analytic(rho0, W_op, x_grid)

        # D-LinOSS observer fit
        om, gm, Am, recon_err, K_eff = dlinoss_fit(x_grid, y_witness, K=K_OBS)

        # Physical rate identification
        # Hamming distance structure: expect peaks at γ=0, γ=1, γ=4 (in Γ·t units per step)
        x_step = GT_MAX / N_GAMMA
        gamma_per_Gt = gm / x_step  # convert per-step to per-unit-x

        # Dominant mode
        amp_norm = Am / (Am.sum() + 1e-14)
        dom = np.argmax(amp_norm)

        # Framework model fits on analytic witness decay
        fits = model_fits(x_grid, y_witness, W0)
        winner = min(fits, key=lambda fw: fits[fw]["residual"])

        # Gamma_mb from Lindblad fit (b should be 4 for pure Markovian)
        b_lind = fits["Lindblad"]["b"]
        # In Γ·t units, b=4 means C decays as exp(-4x) where x=Γ·t
        # The WITNESS decays at the same rate (linear): b should also be 4
        Gamma_mb_ratio = b_lind / 4.0  # ratio to expected Markovian

        # Theoretical prediction for b from OAT state structure
        # Tr[W·ρ(x)] = Σ_{ij, d_H=2} W[j,i]·ρ[i,j](0)·exp(-4x)
        #             + Σ_{ij, d_H=1} W[j,i]·ρ[i,j](0)·exp(-x)
        #             + W[i,i]·ρ[i,i](0)  (diagonal, no decay)
        basis = [(0,0),(0,1),(1,0),(1,1)]
        dH_mat = np.array([[abs(a-c)+abs(b-d) for c,d in basis] for a,b in basis])
        weight_d2 = abs(np.sum(W_op.T * rho0 * (dH_mat == 2)))
        weight_d1 = abs(np.sum(W_op.T * rho0 * (dH_mat == 1)))
        weight_d0 = abs(np.sum(W_op.T * rho0 * (dH_mat == 0)))
        total = weight_d2 + weight_d1 + weight_d0 + 1e-14
        frac_d2 = float(weight_d2 / total)  # fraction from Hamming-2 coherences

        print(f"  N={N:>3d}  C₀={C0:.4f}  W₀={W0:.4f}")
        print(f"    Coherence weights: d=2:{frac_d2:.3f}  d=1:{float(weight_d1/total):.3f}  "
              f"d=0:{float(weight_d0/total):.3f}")
        print(f"    Expected b = {4*frac_d2+1*(1-frac_d2-float(weight_d0/total)):.3f}  "
              f"(weighted avg of d_H² decay rates)")
        print(f"    Lindblad fit: b={b_lind:.3f}  residual={fits['Lindblad']['residual']:.5f}")
        print(f"    MBL fit:      residual={fits['MBL']['residual']:.5f}")
        print(f"    SYK fit:      residual={fits['SYK']['residual']:.5f}")
        print(f"    Winner: {winner}  (b/4 = {Gamma_mb_ratio:.3f})")
        print(f"    D-LinOSS: K_eff={K_eff}  recon_err={recon_err:.5f}")
        print(f"    dom γ_k={gm[dom]/x_step:.3f} (per Γt)  "
              f"expected 4.0 for Markovian Hamming-2")
        print()

        all_results.append({
            "N": N, "C0": C0, "W0": W0,
            "frac_d2": frac_d2,
            "Lindblad_b": b_lind, "Lindblad_residual": fits["Lindblad"]["residual"],
            "MBL_residual": fits["MBL"]["residual"],
            "SYK_residual": fits["SYK"]["residual"],
            "winner": winner, "Gamma_mb_ratio": Gamma_mb_ratio,
            "dlinoss_K_eff": K_eff, "dlinoss_recon_err": recon_err,
            "dom_gamma_per_Gt": float(gm[dom]/x_step),
            "gamma_k": gm.tolist(), "omega_k": om.tolist(),
            "A_k": amp_norm.tolist(),
            "fit_params": {fw: fits[fw].get("params", []) for fw in fits},
        })

    print("="*70)
    print("  SUMMARY: Witness Decay Framework Discrimination")
    print("="*70)
    print(f"  {'N':>4}  {'Winner':>10}  {'b_Lind':>7}  {'b/4':>5}  "
          f"{'Lind_res':>9}  {'MBL_res':>8}  {'SYK_res':>8}  {'K_eff':>6}")
    for r in all_results:
        print(f"  {r['N']:>4}  {r['winner']:>10}  "
              f"{r['Lindblad_b']:>7.3f}  {r['Gamma_mb_ratio']:>5.3f}  "
              f"{r['Lindblad_residual']:>9.5f}  {r['MBL_residual']:>8.5f}  "
              f"{r['SYK_residual']:>8.5f}  {r['dlinoss_K_eff']:>6d}")

    print(f"\n  Key: b≈4 → Markovian (Hamming-2 dominant)")
    print(f"       b<4 → sub-dominant Hamming-1 coherences contribute")
    print(f"       b>4 → additional dephasing mechanism or Wootters artifact absent here")

    winners = [r["winner"] for r in all_results]
    print(f"\n  Framework verdict (witness, clean): ", end="")
    dom_fw = max(set(winners), key=winners.count)
    print(dom_fw)

    if dom_fw == "Lindblad":
        print("  → Markovian dephasing confirmed from LINEAR witness observable.")
        print("    b values close to 4 → Hamming-2 coherences dominate → expected.")
        print("    This is the null hypothesis: OAT dephasing is single-exponential.")
        print("    Real data deviation from b=4 → identifies Γ_mb directly.")

    with open("v5d_witness_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n[DONE] v5d_witness_results.json")

    # ── JILA experimental protocol ─────────────────────────────────────────
    print("\n"+"="*70)
    print("  JILA PROTOCOL: Witness Decay Curve Measurement")
    print("="*70)
    n4 = next(r for r in all_results if r["N"]==4)
    print(f"  N=4, W₀={n4['W0']:.4f}, frac_d2={n4['frac_d2']:.3f}")
    print(f"  1. Prepare N=4 OAT state at chi_t=1.091")
    print(f"  2. Apply Schmidt rotation R_z(θ_A)⊗R_z(θ_B)")
    print(f"  3. Insert variable delay τ (0 to T2/10 ≈ 12s)")
    print(f"  4. Measure W = I/4 - |Φ+><Φ+| on boundary pair")
    print(f"  5. Repeat 280 shots per τ value")
    print(f"  6. Fit Tr[W·ρ(τ)] = W₀·exp(-b·τ)")
    print(f"     → b = Γ_mb·4  (extracts Γ_mb directly!)")
    print(f"  7. If b≈4·Γ_phys: pure Markovian. Identify Γ_phys = J_ex·N")
    print(f"     If b≠4·Γ_phys: many-body mechanism → fit MBL/SYK model")
    print(f"  Total shots: 280 × 20 delay points = 5600 shots")
    print(f"  At 1kHz repetition rate: ~6 seconds per experimental run")


if __name__ == "__main__":
    main()
