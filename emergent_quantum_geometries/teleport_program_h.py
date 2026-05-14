"""
teleport_program_h.py — Finite-Size Scaling of Recoverable Teleportation Geometry
==================================================================================
Program H1: Does the PTM-aligned advantage ΔF = F_rot - F_std survive as N grows?

Uses the exact closed-form rho2(chi_t, N) — no MPS required for boundary pair.
TPU is needed only for the D-LinOSS corpus at scale (Program H2+).

Observables per N:
  chi_t*       : optimal coupling (maximizes F_rot)
  F_std        : naive Bell protocol (standard basis)
  F_rot        : PTM-aligned protocol (optimal SU(2)xSU(2))
  F_opt        : analytic upper bound (1+sum_svs)/4 * 2/3 + 1/3... = (2*f_max+1)/3
  delta_F      : F_rot - F_std (recoverable geometry advantage)
  svs          : singular values of D_phi @ T  (transport spectrum)
  H_T          : spectral entropy -sum(s*log(s+eps)) (richness)
  rank_T       : number of SVs above threshold (PTM rank)
  delta_chi_t  : width of region where F_rot > 2/3 (transport stratum volume)

Usage:
  python teleport_program_h.py                    # N=4..64, fast
  python teleport_program_h.py --N-max 256        # N=4..256
  python teleport_program_h.py --fit              # fit power law ΔF ~ N^(-alpha)
"""
import argparse, json, time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from ptm_features import rho2_exact, compute_T_matrix

# ── Constants ─────────────────────────────────────────────────────────────────
I2  = np.eye(2, dtype=complex)
sz  = np.array([[1,0],[0,-1]], dtype=complex)
sx  = np.array([[0,1],[1,0]],  dtype=complex)
B0  = np.array([1,0,0,1])/np.sqrt(2)
B1  = np.array([1,0,0,-1])/np.sqrt(2)
B2  = np.array([0,1,1,0])/np.sqrt(2)
B3  = np.array([0,1,-1,0])/np.sqrt(2)
BELLS = [B0,B1,B2,B3]
CORR  = [I2, sz, sx, sx@sz]
D_phi = np.diag([1., -1., 1.])   # |Phi+> metric: f = (1+Tr(D T_rot))/4

def su2(p):
    a,b,c = p
    return np.array([[np.exp(-1j*(a+c)/2)*np.cos(b/2), -np.exp(-1j*(a-c)/2)*np.sin(b/2)],
                     [np.exp( 1j*(a-c)/2)*np.sin(b/2),  np.exp( 1j*(a+c)/2)*np.cos(b/2)]])

def teleport(rho_in, rho_res, Ua, Ub):
    Ua_f = np.kron(Ua, I2); Ub_f = np.kron(I2, Ub)
    rr   = Ub_f @ (Ua_f @ rho_res @ Ua_f.conj().T) @ Ub_f.conj().T
    rho3 = np.kron(rho_in, rr)
    out  = np.zeros((2,2), dtype=complex)
    for bk, Uk in zip(BELLS, CORR):
        Pb = np.outer(bk, bk.conj()); Pk = np.kron(Pb, I2)
        rp = Pk @ rho3 @ Pk; rpr = rp.reshape(2,2,2,2,2,2)
        rB = np.einsum('ijkijl->kl', rpr)
        out += Uk @ rB @ Uk.conj().T
    return out

def haar_F(rho_res, Ua, Ub, n=150, seed=0):
    rng = np.random.default_rng(seed); fids = []
    for _ in range(n):
        psi = rng.standard_normal(2) + 1j*rng.standard_normal(2)
        psi /= np.linalg.norm(psi)
        ri = np.outer(psi, psi.conj())
        fids.append(float(np.real(np.trace(ri @ teleport(ri, rho_res, Ua, Ub)))))
    return float(np.mean(fids))

phi_p = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)

def optimal_rotations(rho, n_restarts=6):
    """Find Ua, Ub maximizing singlet fraction via DE + local polish."""
    def neg_f(p):
        U = np.kron(su2(p[:3]), su2(p[3:]))
        return -float(np.real(phi_p.conj() @ U @ rho @ U.conj().T @ phi_p))
    res = differential_evolution(neg_f, [(-np.pi,np.pi)]*6,
                                  seed=0, maxiter=400, tol=1e-10,
                                  popsize=12, mutation=(0.5,1.5))
    res2 = minimize(neg_f, res.x, method='Nelder-Mead',
                    options={'xatol':1e-10,'fatol':1e-10,'maxiter':20000})
    return su2(res2.x[:3]), su2(res2.x[3:])

def f_opt_analytic(rho):
    """f_max = (1+sigma_1+sigma_2+sigma_3)/4 from T-matrix SVD."""
    T = compute_T_matrix(rho)
    svs = np.linalg.svd(T, compute_uv=False)
    return float((1 + svs.sum())/4), svs

def spectral_entropy(svs):
    s = svs / (svs.sum() + 1e-15)
    return float(-np.sum(s * np.log(s + 1e-15)))

# ── Per-N analysis ─────────────────────────────────────────────────────────────
def analyze_N(N, n_chi_coarse=40, n_chi_fine=20, n_haar=150):
    """Full analysis for one system size."""
    print(f"\n--- N={N} ---", flush=True)

    # Step 1: coarse sweep to find chi_t* region
    chi_grid = np.linspace(0.05, np.pi - 0.05, n_chi_coarse)
    f_vals = []
    for chi_t in chi_grid:
        rho = rho2_exact(chi_t, N, 0.0)
        fm, _ = f_opt_analytic(rho)
        f_vals.append(fm)
    chi_star_coarse = chi_grid[np.argmax(f_vals)]

    # Step 2: fine sweep around chi_t*
    chi_fine = np.linspace(max(0.05, chi_star_coarse - 0.5),
                           min(np.pi-0.05, chi_star_coarse + 0.5), n_chi_fine)
    best_fm = 0; chi_star = chi_star_coarse
    for chi_t in chi_fine:
        rho = rho2_exact(chi_t, N, 0.0)
        fm, _ = f_opt_analytic(rho)
        if fm > best_fm: best_fm = fm; chi_star = chi_t

    # Step 3: evaluate all metrics at chi_t*
    rho_star = rho2_exact(chi_star, N, 0.0)
    fm, svs = f_opt_analytic(rho_star)
    F_opt = (2*fm + 1)/3
    T = compute_T_matrix(rho_star)
    rank_T = int((svs > 1e-3).sum())
    H_T = spectral_entropy(svs)

    # Optimal rotations and fidelities
    Ua_opt, Ub_opt = optimal_rotations(rho_star)
    F_std = haar_F(rho_star, I2, I2, n_haar)
    F_rot = haar_F(rho_star, Ua_opt, Ub_opt, n_haar)
    delta_F = F_rot - F_std

    # Step 4: transport stratum width (region where F_rot > 2/3)
    # Estimate chi_t_lo, chi_t_hi using F_opt analytic threshold
    n_width = 30
    chi_width = np.linspace(0.02, np.pi - 0.02, n_width)
    above = []
    for chi_t in chi_width:
        rho = rho2_exact(chi_t, N, 0.0)
        fm_w, _ = f_opt_analytic(rho)
        above.append((2*fm_w+1)/3 > 2/3)
    if any(above):
        idxs = [i for i,a in enumerate(above) if a]
        delta_chi = chi_width[idxs[-1]] - chi_width[idxs[0]]
    else:
        delta_chi = 0.0

    result = {
        "N": N, "chi_star": float(chi_star),
        "F_std": F_std, "F_rot": F_rot, "F_opt": F_opt,
        "delta_F": float(delta_F),
        "f_max": float(fm),
        "svs": svs.tolist(),
        "rank_T": rank_T,
        "H_T": float(H_T),
        "delta_chi": float(delta_chi),
    }
    print(f"  chi_t*={chi_star:.3f}  F_std={F_std:.4f}  F_rot={F_rot:.4f}  "
          f"F_opt={F_opt:.4f}  ΔF={delta_F:.4f}  Δχt={delta_chi:.3f}  H_T={H_T:.3f}")
    return result

# ── Scaling analysis ───────────────────────────────────────────────────────────
def fit_powerlaw(Ns, delta_Fs):
    """Fit ΔF ~ A * N^(-alpha). Returns (A, alpha, R^2)."""
    log_N  = np.log(Ns)
    log_dF = np.log(np.array(delta_Fs) + 1e-10)
    coeffs = np.polyfit(log_N, log_dF, 1)
    alpha  = -coeffs[0]
    A      = np.exp(coeffs[1])
    predicted = coeffs[0]*log_N + coeffs[1]
    ss_res = np.sum((log_dF - predicted)**2)
    ss_tot = np.sum((log_dF - log_dF.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return float(A), float(alpha), float(R2)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N-max", type=int, default=64,
                   help="Maximum N (power of 2 steps: 4,8,16,32,64,...)")
    p.add_argument("--fit", action="store_true", help="Fit power law")
    p.add_argument("--n-haar", type=int, default=150)
    p.add_argument("--out", default="program_h_results.json")
    args = p.parse_args()

    N_vals = []
    N = 4
    while N <= args.N_max:
        N_vals.append(N); N *= 2

    print("="*70)
    print(f"Program H — Finite-Size Scaling of Recoverable Teleportation Geometry")
    print(f"N values: {N_vals}")
    print("="*70)

    t0 = time.time()
    results = []
    for N in N_vals:
        r = analyze_N(N, n_haar=args.n_haar)
        results.append(r)

    # Print scaling table
    print("\n" + "="*70)
    print("SCALING TABLE")
    print(f"{'N':>6} {'chi_t*':>8} {'F_std':>7} {'F_rot':>7} {'F_opt':>7} "
          f"{'ΔF':>7} {'Δχt':>7} {'rank':>6} {'H_T':>6}")
    print("-"*70)
    for r in results:
        print(f"{r['N']:>6} {r['chi_star']:>8.4f} {r['F_std']:>7.4f} "
              f"{r['F_rot']:>7.4f} {r['F_opt']:>7.4f} {r['delta_F']:>7.4f} "
              f"{r['delta_chi']:>7.3f} {r['rank_T']:>6} {r['H_T']:>6.3f}")

    # Power law fit
    Ns     = [r['N']       for r in results]
    dFs    = [r['delta_F'] for r in results]
    F_opts = [r['F_opt']   for r in results]

    if args.fit or len(results) >= 3:
        A, alpha, R2 = fit_powerlaw(Ns, dFs)
        print(f"\nPower-law fit ΔF ~ {A:.4f} * N^(-{alpha:.4f})  R²={R2:.4f}")
        if alpha < 0.05:
            print("  → ΔF is approximately CONSTANT: geometry survives thermodynamically")
        elif alpha < 0.5:
            print("  → ΔF decays slowly: geometry persists but weakens")
        elif alpha < 1.0:
            print("  → ΔF decays moderately: finite-size effect")
        else:
            print("  → ΔF decays rapidly: likely finite-size only")

        A2, alpha2, R2_2 = fit_powerlaw(Ns, F_opts)
        print(f"Power-law fit F_opt ~ {A2:.4f} * N^(-{alpha2:.4f})  R²={R2_2:.4f}")
        print(f"  → F_opt → {A2*max(Ns)**(-alpha2):.4f} at N={max(Ns)}")

        for r in results:
            r['powerlaw_alpha_dF'] = alpha
            r['powerlaw_alpha_Fopt'] = alpha2

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
