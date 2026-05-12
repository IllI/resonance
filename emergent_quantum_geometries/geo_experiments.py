"""
geo_experiments.py
------------------
Implements Experiments III, IV, and partial II from the reviewer's framework:

III. Phase-winding geometry: test if theta_A_opt ≈ -arg(rho_{00,11})
IV.  Large-N concurrence scaling: N=4 to N=1024 (pure numpy, no TPU)
II.  Correlation tensor and geometric invariants of OAT boundary states

All from exact closed-form rho_2.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, differential_evolution

# Load exact formula from source
src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}
exec(compile(src, 'tpu_dlinoss_training_gen', 'exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']


# ── Core utilities ────────────────────────────────────────────────────────────

def concurrence(rho):
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy,sy)
    R = la.sqrtm(rho @ (ss @ rho.conj() @ ss))
    e = sorted(np.sqrt(np.maximum(np.real(la.eigvals(R)),0)), reverse=True)
    return max(0., e[0]-e[1]-e[2]-e[3])

def su2(p):
    c, s = np.cos(p[0]/2), np.sin(p[0]/2)
    return np.array([[c*np.exp(1j*p[1]), s*np.exp(1j*p[2])],
                     [-s*np.exp(-1j*p[2]), c*np.exp(-1j*p[1])]])

PHI_PLUS = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)

def f_singlet(params, rho):
    UA, UB = su2(params[:3]), su2(params[3:])
    psi = np.kron(UA, UB) @ PHI_PLUS
    return -np.real(psi.conj() @ rho @ psi)

def optimize_fmax(rho, n=80):
    bounds = [(-np.pi,np.pi)]*6
    res_de = differential_evolution(lambda p: f_singlet(p,rho), bounds,
                                    seed=42, maxiter=1000, tol=1e-10)
    bp = res_de.x; bf = -res_de.fun
    rng = np.random.default_rng(42)
    for _ in range(n):
        p0 = rng.uniform(-np.pi, np.pi, 6)
        r = minimize(f_singlet, p0, args=(rho,), method='Nelder-Mead',
                     options={'xatol':1e-11,'fatol':1e-11,'maxiter':50000})
        if -r.fun > bf:
            bf = -r.fun; bp = r.x
    r2 = minimize(f_singlet, bp, args=(rho,), method='Powell',
                  options={'xtol':1e-13,'ftol':1e-13,'maxiter':100000})
    return -r2.fun, r2.x


# ── Experiment III: Phase-Winding Geometry ────────────────────────────────────

def experiment_III():
    """
    Test: does theta_A_opt ≈ -arg(rho_{00,11})?
    rho_{00,11} is the key off-diagonal (singlet sector coherence).
    theta_A_opt is the optimal U_A rotation from singlet fraction optimization.
    """
    print("=" * 65)
    print("EXPERIMENT III: Phase-Winding Geometry")
    print("Test: theta_A_opt ≈ -arg(rho[0,3])?")
    print("=" * 65)
    print()
    grid = np.linspace(0.02, 6.28, 400)

    header = f"{'N':>4}  {'chi_t*':>7}  {'C':>7}  {'arg_rho03':>10}  "
    header += f"{'theta_A_naive':>14}  {'theta_A_opt':>11}  {'match?':>8}"
    print(header)
    print("-" * 65)

    for N in [2, 4, 6, 8, 12, 16, 24, 32]:
        Cs = [concurrence(oat_rho2_exact(N,t)) for t in grid]
        idx = np.argmax(Cs)
        chi_t = grid[idx]
        rho = oat_rho2_exact(N, chi_t)
        C = Cs[idx]

        # rho[0,1] = <00|rho|01> has phase exp(-i*chi*0.5) -- the main phase
        arg_01 = np.angle(rho[0,1])   # expected: -chi_t/2
        arg_03 = np.angle(rho[0,3])   # always real (=0)

        theta_A_naive = -arg_01  # phase-frame alignment prediction

        # Optimal theta_A from full SU(2) optimization
        _, params = optimize_fmax(rho, n=60)
        UA_opt = su2(params[:3])
        # Extract phase of UA_opt: dominated by phi parameter (params[1])
        # UA = [[c*e^{iphi}, s*e^{ipsi}], [-s*e^{-ipsi}, c*e^{-iphi}]]
        # The effective phase winding is primarily params[1] (phi)
        theta_A_opt = 2 * np.arctan2(abs(UA_opt[1,0]), abs(UA_opt[0,0]))
        # Effective rotation angle on Bloch sphere
        phi_A_opt = np.angle(UA_opt[0,0])

        match = "YES" if abs(theta_A_naive - phi_A_opt) < 0.3 else "~"

        print(f"  {N:2d}  {chi_t:7.4f}  {C:7.4f}  "
              f"arg01={np.degrees(arg_01):+8.2f}°  "
              f"pred={np.degrees(theta_A_naive):+8.2f}°  "
              f"opt={np.degrees(phi_A_opt):+8.2f}°  {match}")

    print()
    print("Key: if arg_rho03 → -180° and theta_A_naive → +180°,")
    print("the phase-frame correction is Berry-like geometric phase winding.")
    print()


# ── Experiment IV: Large-N Scaling (no TPU needed) ───────────────────────────

def experiment_IV():
    """
    C_peak vs N for N=4 to N=1024 using exact 4x4 formula.
    Determine if exponent -0.636 holds or converges to -1 at large N.
    """
    print("=" * 65)
    print("EXPERIMENT IV: Large-N Concurrence Scaling")
    print("C_peak ~ A * N^alpha: what is alpha at large N?")
    print("=" * 65)
    print()

    Ns = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    results = []

    # For large N, optimal chi_t* ≈ B*N^{-beta}. Use dense grid near predicted optimum.
    for N in Ns:
        # Predicted chi_t* from earlier fit: 2.26 * N^{-0.577}
        # Use FULL range 0.02-6.28 to avoid missing peaks near 2*pi
        grid = np.linspace(0.02, 6.28, 300)
        Cs = [concurrence(oat_rho2_exact(N,t)) for t in grid]
        idx = np.argmax(Cs)
        C_peak = Cs[idx]
        t_star = grid[idx]
        results.append((N, C_peak, t_star))
        print(f"  N={N:4d}  C_peak={C_peak:.6f}  chi_t*={t_star:.5f}")

    print()
    print("Power-law fits (progressive):")

    # Full fit — filter C=0 entries
    valid = [(r[0],r[1],r[2]) for r in results if r[1] > 1e-6]
    if len(valid) >= 3:
        log_N = np.log([r[0] for r in valid[1:]])
        log_C = np.log([r[1] for r in valid[1:]])
        alpha_full, lA = np.polyfit(log_N, log_C, 1)
        A_full = np.exp(lA)
        print(f"  N=4-{valid[-1][0]}: C = {A_full:.4f} * N^{alpha_full:.4f}")

        # Large-N only
        large = [(r[0],r[1]) for r in valid if r[0] >= 32]
        if len(large) >= 3:
            log_N_lg = np.log([r[0] for r in large])
            log_C_lg = np.log([r[1] for r in large])
            alpha_lg, lA_lg = np.polyfit(log_N_lg, log_C_lg, 1)
            A_lg = np.exp(lA_lg)
            print(f"  N>=32:  C = {A_lg:.4f} * N^{alpha_lg:.4f}")
            if abs(alpha_lg + 1) < 0.08:
                print("  CONVERGING TO N^{-1} [CONFIRMED]")
            elif abs(alpha_lg + 0.5) < 0.08:
                print("  CONVERGING TO N^{-1/2}")
            else:
                print(f"  ASYMPTOTIC EXPONENT: {alpha_lg:.3f}")

    # Optimal time scaling (only for valid C>0 entries, skip first)
    log_t_valid = np.log([r[2] for r in valid[1:]])
    if len(log_N) == len(log_t_valid):
        beta_full, lB = np.polyfit(log_N, log_t_valid, 1)
        B_full = np.exp(lB)
        print(f"  chi_t* scaling: {B_full:.4f} * N^{beta_full:.4f}")
    print()


# ── Experiment II (partial): Geometric Invariants ────────────────────────────

def experiment_II_partial():
    """
    Compute multiple geometric invariants of the OAT boundary state.
    Check if f_max = g(C, lambda_1, lambda_2) for some function g.
    """
    print("=" * 65)
    print("EXPERIMENT II: Geometric Invariants of OAT Boundary States")
    print("Does f_max lie on a constrained sub-manifold?")
    print("=" * 65)
    print()

    grid = np.linspace(0.02, 6.28, 400)

    print(f"{'N':>4}  {'C':>7}  {'f_max':>7}  {'negat':>7}  "
          f"{'purity':>8}  {'T_max':>7}  {'(1+C)/2':>9}  {'f_max/[(1+C)/2]':>16}")
    print("-" * 75)

    for N in [2, 4, 6, 8, 12, 16, 32]:
        Cs = [concurrence(oat_rho2_exact(N,t)) for t in grid]
        idx = np.argmax(Cs)
        chi_t = grid[idx]
        rho = oat_rho2_exact(N, chi_t)
        C = Cs[idx]
        f_max, _ = optimize_fmax(rho, n=50)

        # Negativity — correct PT_B permutation: (0,3,2,1)
        # rho_tensor[iL,iR,jL,jR] = rho[2iL+iR, 2jL+jR]
        # PT_B: result[iL,iR,jL,jR] = rho_tensor[iL,jR,jL,iR]
        # Permutation: (0,3,2,1) applied to rho_tensor
        rho_tensor = rho.reshape(2,2,2,2)
        rho_pt = rho_tensor.transpose(0,3,2,1).reshape(4,4)
        eigs_pt = np.real(la.eigvals(rho_pt))
        negativity = sum(abs(e) for e in eigs_pt if e < 0)

        # Purity
        purity = np.real(np.trace(rho @ rho))

        # Max correlation tensor element
        sx = np.array([[0,1],[1,0]], dtype=complex)
        sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        sigmas = [sx,sy,sz]
        T = [[np.real(np.trace(rho@np.kron(si,sj))) for sj in sigmas] for si in sigmas]
        T_max = max(abs(T[i][j]) for i in range(3) for j in range(3))

        ratio = f_max / ((1+C)/2) if C > 0.01 else float('nan')

        print(f"  {N:2d}  {C:7.4f}  {f_max:7.5f}  {negativity:7.4f}  "
              f"{purity:8.5f}  {T_max:7.4f}  {(1+C)/2:9.6f}  {ratio:>16.4f}")

    print()
    print("Key: if ratio = f_max/[(1+C)/2] is constant -> constrained manifold.")
    print("     if it varies -> genuinely non-X without simple reduction.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("OAT BOUNDARY STATE GEOMETRY EXPERIMENTS")
    print("=========================================")
    print()

    experiment_IV()   # Fast pure-numpy — run first
    experiment_III()  # Phase-winding test
    experiment_II_partial()  # Geometric invariants
