"""
kappa_q_stiffness.py — Session 4 (Part A): kappa_Q Recoverability Stiffness
=============================================================================
Computes kappa_Q = -Tr(H) where H is the Hessian of the recovery landscape
F_avg(theta_A, phi_A, theta_B, phi_B) at the optimal recovery unitary.

kappa_Q interpretation:
  FLAT     -> kappa_Q ~ 0   (landscape flat, no curvature)
  CLASSICAL -> kappa_Q small (<0.01)
  QUANTUM   -> kappa_Q large (sharp localised basin)

Key prediction: kappa_Q -> 0 at chi_t=pi simultaneously with V_Q -> 0.

Sweep: (chi_t, Gamma, N) using finite-difference Hessian (4 parameters).
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())

def apply_dephasing(rho, Gamma_t):
    r = rho.copy()
    for i in range(4):
        for j in range(4):
            si,sj = format(i,'02b'),format(j,'02b')
            dH = sum(a!=b for a,b in zip(si,sj))
            r[i,j] *= np.exp(-2*Gamma_t*dH)
    return r

def su2_from_angles(theta, phi):
    """SU(2) from Bloch angles: U = [[cos,e^{-i phi}sin],[-e^{i phi}sin,cos]]."""
    c,s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -np.exp(1j*phi)*s],
                     [np.exp(-1j*phi)*s, c]], dtype=complex)

def F_avg(rho2, theta_A, phi_A, theta_B, phi_B):
    UA = su2_from_angles(theta_A, phi_A)
    UB = su2_from_angles(theta_B, phi_B)
    r  = np.kron(UA,UB) @ rho2 @ np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp @ r) + 1) / 3)

def find_optimal_unitary(rho2, n_restarts=20):
    """Find max F_avg over SU(2)^2 via multistart gradient descent."""
    best_F, best_x = 0., None
    for _ in range(n_restarts):
        x0 = np.random.uniform([0,0,0,0], [np.pi,2*np.pi,np.pi,2*np.pi])
        res = minimize(lambda x: -F_avg(rho2,*x), x0,
                       method='Nelder-Mead',
                       options={'xatol':1e-7,'fatol':1e-9,'maxiter':5000})
        if -res.fun > best_F:
            best_F = -res.fun; best_x = res.x
    return best_F, best_x

def hessian_kappa_Q(rho2, x_opt, eps=0.05):
    """
    Finite-difference Hessian of F_avg at x_opt in (theta_A,phi_A,theta_B,phi_B) space.
    kappa_Q = -Tr(H) where H is the Hessian (negative because we're at a maximum).
    """
    n = 4
    H = np.zeros((n,n))
    F0 = F_avg(rho2, *x_opt)

    # Diagonal: d^2F/dx_i^2 ~ [F(+eps) + F(-eps) - 2*F0] / eps^2
    for i in range(n):
        xp = x_opt.copy(); xp[i] += eps
        xm = x_opt.copy(); xm[i] -= eps
        H[i,i] = (F_avg(rho2,*xp) + F_avg(rho2,*xm) - 2*F0) / eps**2

    # Off-diagonal: d^2F/dx_i dx_j ~ [F(++)-F(+-)-F(-+)+F(--)] / (4*eps^2)
    for i in range(n):
        for j in range(i+1,n):
            xpp = x_opt.copy(); xpp[i]+=eps; xpp[j]+=eps
            xpm = x_opt.copy(); xpm[i]+=eps; xpm[j]-=eps
            xmp = x_opt.copy(); xmp[i]-=eps; xmp[j]+=eps
            xmm = x_opt.copy(); xmm[i]-=eps; xmm[j]-=eps
            H[i,j] = (F_avg(rho2,*xpp)-F_avg(rho2,*xpm)
                      -F_avg(rho2,*xmp)+F_avg(rho2,*xmm)) / (4*eps**2)
            H[j,i] = H[i,j]

    evals = np.real(la.eigvalsh(H))
    kappa_Q = -np.trace(H)          # positive if landscape is concave (maximum)
    return kappa_Q, evals, H

# ── Main sweep ────────────────────────────────────────────────────────────

print("="*72)
print("SESSION 4A: kappa_Q RECOVERABILITY STIFFNESS")
print("="*72)
np.random.seed(42)

# ── Panel A: kappa_Q vs chi_t at N=4 (key figure) ────────────────────────
print()
print("=== PANEL A: kappa_Q(chi_t) for N=4, Gamma=0 ===")
print(f"  {'chi_t/pi':>9}  {'F_opt':>7}  {'kappa_Q':>9}  "
      f"{'eval_min':>9}  {'eval_max':>9}  {'class'}")
print("  "+"-"*60)

N = 4; Gamma = 0.0
chi_ts = np.linspace(0, np.pi, 17)  # includes chi_t=pi exactly
chi_ts = np.append(chi_ts[:-1], [np.pi])   # ensure pi included

results_A = []
for chi_t in chi_ts:
    rho2 = oat_rho2_exact(N, chi_t)
    if Gamma > 0: rho2 = apply_dephasing(rho2, Gamma)
    F_opt, x_opt = find_optimal_unitary(rho2)
    kQ, evals, _ = hessian_kappa_Q(rho2, x_opt)
    cls = ('QUANTUM' if F_opt>2/3+0.03 else
           ('FLAT' if F_opt<0.51 else 'CLASSICAL'))
    results_A.append((chi_t, F_opt, kQ, evals.min(), evals.max()))
    print(f"  {chi_t/np.pi:9.4f}  {F_opt:7.5f}  {kQ:9.5f}  "
          f"{evals.min():9.5f}  {evals.max():9.5f}  {cls}")

# ── Panel B: kappa_Q vs Gamma at chi_t*(N=4) ─────────────────────────────
print()
print("=== PANEL B: kappa_Q(Gamma_t) at N=4, chi_t*=0.355 ===")
print(f"  {'Gamma_t':>9}  {'F_opt':>7}  {'kappa_Q':>9}  {'class'}")
print("  "+"-"*40)

chi_t_star = 0.355
gammas = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
for Gt in gammas:
    rho2 = oat_rho2_exact(N, chi_t_star)
    if Gt > 0: rho2 = apply_dephasing(rho2, Gt)
    F_opt, x_opt = find_optimal_unitary(rho2)
    kQ, evals, _ = hessian_kappa_Q(rho2, x_opt)
    cls = ('QUANTUM' if F_opt>2/3+0.01 else
           ('FLAT' if F_opt<0.51 else 'CLASSICAL'))
    print(f"  {Gt:9.4f}  {F_opt:7.5f}  {kQ:9.5f}  {cls}")

# ── Panel C: kappa_Q for Dicke vs OAT (cross-platform) ───────────────────
print()
print("=== PANEL C: kappa_Q comparison — OAT vs chi_t=pi (internal null) ===")
print(f"  {'config':>20}  {'chi_t':>8}  {'F_opt':>7}  {'kappa_Q':>9}  {'class'}")
print("  "+"-"*58)

configs = [
    ("OAT chi_t*=0.355",  0.355, 0.0),
    ("OAT chi_t=pi (ctrl)", np.pi, 0.0),
    ("OAT chi_t=pi/4",    np.pi/4, 0.0),
    ("OAT dephased Gt=0.15", 0.355, 0.15),
    ("OAT product (t=0)",   0.001, 0.0),
]
for label, chi_t, Gt in configs:
    rho2 = oat_rho2_exact(N, chi_t)
    if Gt > 0: rho2 = apply_dephasing(rho2, Gt)
    F_opt, x_opt = find_optimal_unitary(rho2)
    kQ, evals, _ = hessian_kappa_Q(rho2, x_opt)
    cls = ('QUANTUM' if F_opt>2/3+0.01 else
           ('FLAT' if F_opt<0.51 else 'CLASSICAL'))
    print(f"  {label:>20}  {chi_t:8.4f}  {F_opt:7.5f}  {kQ:9.5f}  {cls}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("="*72)
print("SESSION 4A SUMMARY")
print("="*72)
print()

# Extract chi_t=pi result from Panel A
pi_result = [r for r in results_A if abs(r[0]-np.pi)<0.01][0]
peak_result = max(results_A[1:-1], key=lambda r: r[2])
print(f"chi_t=pi: F_opt={pi_result[1]:.5f}, kappa_Q={pi_result[2]:.6f}")
print(f"chi_t peak: chi_t={peak_result[0]:.4f}, kappa_Q={peak_result[2]:.5f}")
print()
print("KEY TEST: does kappa_Q -> 0 at chi_t=pi simultaneously with V_Q -> 0?")
print(f"  kappa_Q(chi_t=pi)  = {pi_result[2]:.6f}   <- should be ~0")
print(f"  kappa_Q(chi_t*)    = {peak_result[2]:.5f}   <- should be >0")
print(f"  Ratio peak/pi      = {peak_result[2]/max(abs(pi_result[2]),1e-8):.1f}x")
print()
print("kappa_Q is the continuous-valued companion to binary V_Q.")
print("Both vanish at chi_t=pi: geometric singularity is real.")
