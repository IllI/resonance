"""
singlet_fraction_proof.py
--------------------------
Directly computes f_max = max_{U_A, U_B} <Phi+|(U_A x U_B) rho (U_A x U_B)^dag|Phi+>
from the exact OAT reduced density matrix, WITHOUT assuming X-state structure.

Tests the conjecture: f_max = (1+C)/2 for the OAT boundary-pair family.

Method:
  - Parameterize U_A in SU(2) by (theta, phi, psi); U_B same.
  - Use scipy.optimize.minimize (Nelder-Mead, 50 random restarts).
  - Compare to Wootters concurrence C computed from full 4x4 matrix.
  - Also verify the F_opt = (2+C)/3 claim directly.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, differential_evolution

# Load original formula (utf-8-sig strips BOM)
src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}
exec(compile(src, 'tpu_dlinoss_training_gen', 'exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']
import sys  # noqa — needed by exec'd source

# oat_rho2_exact is now loaded from tpu_dlinoss_training_gen.py above


# ── Wootters concurrence ─────────────────────────────────────────────────────

def wootters_concurrence(rho):
    """Full Wootters concurrence for a 4x4 density matrix."""
    sigma_y = np.array([[0, -1j],[1j, 0]])
    sysy = np.kron(sigma_y, sigma_y)
    rho_tilde = sysy @ rho.conj() @ sysy
    R = la.sqrtm(rho @ rho_tilde)
    # Eigenvalues of R in decreasing order
    eigs = sorted(np.sqrt(np.maximum(np.real(la.eigvals(R)), 0)), reverse=True)
    return max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3])


# ── SU(2) parameterization ───────────────────────────────────────────────────

def su2(params):
    """SU(2) matrix from 3 real parameters (theta, phi, psi)."""
    theta, phi, psi = params
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [c * np.exp(1j*phi),  s * np.exp(1j*psi)],
        [-s * np.exp(-1j*psi), c * np.exp(-1j*phi)]
    ])


# ── Singlet fraction objective ───────────────────────────────────────────────

PHI_PLUS = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)

def neg_singlet_fraction(params, rho):
    """Negative singlet fraction for minimization."""
    UA = su2(params[:3])
    UB = su2(params[3:])
    U = np.kron(UA, UB)
    psi = U @ PHI_PLUS
    val = np.real(psi.conj() @ rho @ psi)
    return -val


def optimize_singlet_fraction(rho, n_restarts=120):
    """Find f_max via Nelder-Mead+DE with random restarts."""
    # First try differential evolution (global optimizer)
    def obj_de(params):
        return neg_singlet_fraction(params, rho)
    bounds = [(-np.pi, np.pi)]*6
    res_de = differential_evolution(obj_de, bounds, seed=42,
                                    maxiter=2000, tol=1e-10, workers=1)
    best_f = -res_de.fun
    best_p = res_de.x

    # Then Nelder-Mead restarts from random starts + DE winner
    rng = np.random.default_rng(42)
    starts = [best_p] + [rng.uniform(-np.pi, np.pi, 6) for _ in range(n_restarts)]
    for p0 in starts:
        res = minimize(neg_singlet_fraction, p0, args=(rho,),
                       method='Nelder-Mead',
                       options={'xatol':1e-11,'fatol':1e-11,'maxiter':100000})
        if -res.fun > best_f:
            best_f = -res.fun
            best_p = res.x

    # Powell refinement on winner
    res2 = minimize(neg_singlet_fraction, best_p, args=(rho,),
                    method='Powell',
                    options={'xtol':1e-13,'ftol':1e-13,'maxiter':200000})
    return -res2.fun, res2.x


def validate_on_werner():
    """Confirm optimizer on Werner state where f_max=(1+C)/2 is exact."""
    for p in [0.3, 0.5, 0.8, 1.0]:
        rho_w = p*np.outer(PHI_PLUS, PHI_PLUS.conj()) + (1-p)*np.eye(4)/4
        C_w = max(0., (3*p-1)/2)
        f_theory = (1 + C_w)/2
        f_opt, _ = optimize_singlet_fraction(rho_w, n_restarts=40)
        print(f"  Werner p={p:.1f}: C={C_w:.4f} f_theory={f_theory:.6f} "
              f"f_opt={f_opt:.6f} err={f_opt-f_theory:+.2e}")


# ── Horodecki Bell bound ─────────────────────────────────────────────────────

def horodecki_smax(rho):
    """
    Direct Horodecki criterion: S_max = 2*sqrt(M(rho))
    where M(rho) = sum of two largest eigenvalues of T^T T,
    T_ij = Tr[rho (sigma_i x sigma_j)].
    """
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    sigmas = [sx, sy, sz]

    T = np.zeros((3,3), dtype=float)
    for i, si in enumerate(sigmas):
        for j, sj in enumerate(sigmas):
            T[i,j] = np.real(np.trace(rho @ np.kron(si, sj)))

    TtT = T.T @ T
    eigs = sorted(np.real(la.eigvals(TtT)), reverse=True)
    M = eigs[0] + eigs[1]
    return 2 * np.sqrt(M)


# ── Asymptotic concurrence analysis ─────────────────────────────────────────

def oat_peak_concurrence(N, chi_t_grid):
    """Find peak concurrence over chi_t grid."""
    best_C, best_t = 0.0, 0.0
    for chi_t in chi_t_grid:
        rho = oat_rho2_exact(N, chi_t)
        C = wootters_concurrence(rho)
        if C > best_C:
            best_C, best_t = C, chi_t
    return best_C, best_t


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    chi_t_grid = np.linspace(0.02, 6.28, 400)

    print("=" * 70)
    print("OPTIMIZER VALIDATION on Werner states (ground truth: f_max=(1+C)/2)")
    print("=" * 70)
    validate_on_werner()
    print()

    print("=" * 75)
    print("SINGLET FRACTION DIRECT OPTIMIZATION vs (1+C)/2 CONJECTURE")
    print("=" * 75)
    print()
    print(f"{'N':>4}  {'chi_t*':>7}  {'C':>8}  {'f_max':>8}  "
          f"{'(1+C)/2':>9}  {'delta':>10}  {'F_opt':>8}  {'S_max':>8}")
    print("-" * 75)

    results = []
    for N in [2, 4, 6, 8, 12, 16, 24, 32]:
        C_peak, chi_t_star = oat_peak_concurrence(N, chi_t_grid)
        rho = oat_rho2_exact(N, chi_t_star)

        f_max, _ = optimize_singlet_fraction(rho, n_restarts=60)
        f_conj   = (1 + C_peak) / 2.0
        delta    = f_max - f_conj

        F_opt_direct = (2 * f_max + 1) / 3.0
        F_opt_formula= (2 + C_peak) / 3.0
        S_max = horodecki_smax(rho)

        print(f"  {N:2d}  {chi_t_star:7.4f}  {C_peak:8.5f}  {f_max:8.6f}  "
              f"{f_conj:9.6f}  {delta:+10.2e}  {F_opt_direct:8.6f}  {S_max:8.5f}")

        results.append((N, C_peak, chi_t_star, f_max, f_conj, delta, F_opt_direct, S_max))

    print()
    print("CONJECTURE STATUS: f_max = (1+C)/2")
    deltas = [abs(r[5]) for r in results]
    print(f"  Max |delta| = {max(deltas):.2e}")
    if max(deltas) < 1e-4:
        print("  STATUS: CONFIRMED to within 1e-4 [OBSERVED]")
    else:
        print("  STATUS: VIOLATED -- conjecture fails for some N")

    print()
    print("=" * 75)
    print("ASYMPTOTIC CONCURRENCE SCALING: C_peak vs N")
    print("=" * 75)

    N_vals = [r[0] for r in results]
    C_vals = [r[1] for r in results]
    t_vals = [r[2] for r in results]

    # Log-log fit for C_peak ~ N^alpha
    log_N = np.log(N_vals[1:])  # skip N=2 (C=1 saturates)
    log_C = np.log(C_vals[1:])
    alpha, log_A = np.polyfit(log_N, log_C, 1)
    A = np.exp(log_A)
    print(f"  Power-law fit C_peak = {A:.4f} * N^{alpha:.4f}")
    print(f"  Predicted for N=64:  C={A * 64**alpha:.5f}")
    print()

    # Optimal time scaling
    log_t = np.log(t_vals[1:])
    beta, log_B = np.polyfit(log_N, log_t, 1)
    B = np.exp(log_B)
    print(f"  Optimal time fit chi_t* = {B:.4f} * N^{beta:.4f}")
    print()

    # Theoretical argument for C ~ 1/N
    print("THEORETICAL ARGUMENT for C_peak ~ 1/N:")
    print("  The dominant coherence is rho_{00,11} = (1/4)*exp(i*phi)*cos^{N/2-1}(chi_t/2)")
    print("  At optimal chi_t* ~ N^{-1} (from scaling), let x = chi_t*/2 << 1:")
    print("  cos^{N/2-1}(x) ~ exp(-(N/2-1)*x^2/2) ~ exp(-N*x^2/4)")
    print("  Maximized over x: set d/dx[x*exp(-N*x^2/4)] = 0 -> x* = sqrt(2/N)")
    print("  => C_peak ~ (1/4)*exp(-1/2) * (2/N)^{1/2} * (N/2)^{1/2}... ")
    print("  Exact large-N: C_peak ~ const * N^{-1} (more careful analysis needed)")
    print()

    print("=" * 75)
    print("BELL INEQUALITY (direct Horodecki S_max):")
    print("=" * 75)
    print()
    for r in results:
        N, C, _, _, _, _, _, Sm = r
        violates = "VIOLATES" if Sm > 2 else "classical"
        print(f"  N={N:2d}: C={C:.4f}  S_max={Sm:.5f}  -> {violates}")

    print()
    print("Note: S_max computed directly from Horodecki criterion T^T T,")
    print("NOT inferred from concurrence (which is only valid for X-states).")


if __name__ == "__main__":
    main()
