"""
critical_exponent.py — Critical Scaling Near Recovery Geometry Phase Boundaries
================================================================================
Tests the conjecture:
  kappa_Q ~ |chi_t - pi|^beta  (near chi_t=pi singularity)
  V_Q     ~ |chi_t - chi_tc|^nu (near QUANTUM-CLASSICAL boundary chi_tc)

If clean power laws emerge over 1.5+ decades: "emergent geometric phases."
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, curve_fit

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())

def haar_SU2():
    z=(np.random.randn(2,2)+1j*np.random.randn(2,2))/np.sqrt(2)
    Q,R=np.linalg.qr(z); d=np.diagonal(R); Q*=d/np.abs(d)
    if np.real(np.linalg.det(Q))<0: Q[:,0]*=-1
    return Q

def su2_from_angles(t, p):
    c,s = np.cos(t/2),np.sin(t/2)
    return np.array([[c,-np.exp(1j*p)*s],[np.exp(-1j*p)*s,c]],dtype=complex)

def F_avg(rho2, theta_A, phi_A, theta_B, phi_B):
    UA=su2_from_angles(theta_A,phi_A); UB=su2_from_angles(theta_B,phi_B)
    r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp@r)+1)/3)

def find_F_opt(rho2, n_restarts=12):
    best,bx=0.,None
    for _ in range(n_restarts):
        x0=np.random.uniform([0,0,0,0],[np.pi,2*np.pi,np.pi,2*np.pi])
        res=minimize(lambda x:-F_avg(rho2,*x),x0,method='Nelder-Mead',
                     options={'xatol':1e-8,'fatol':1e-10,'maxiter':8000})
        if -res.fun>best: best=-res.fun; bx=res.x
    return best,bx

def kappa_Q(rho2, x_opt, eps=0.04):
    n=4; F0=F_avg(rho2,*x_opt)
    H=np.zeros((n,n))
    for i in range(n):
        xp=x_opt.copy();xp[i]+=eps; xm=x_opt.copy();xm[i]-=eps
        H[i,i]=(F_avg(rho2,*xp)+F_avg(rho2,*xm)-2*F0)/eps**2
    for i in range(n):
        for j in range(i+1,n):
            xpp=x_opt.copy();xpp[i]+=eps;xpp[j]+=eps
            xpm=x_opt.copy();xpm[i]+=eps;xpm[j]-=eps
            xmp=x_opt.copy();xmp[i]-=eps;xmp[j]+=eps
            xmm=x_opt.copy();xmm[i]-=eps;xmm[j]-=eps
            H[i,j]=(F_avg(rho2,*xpp)-F_avg(rho2,*xpm)-F_avg(rho2,*xmp)+F_avg(rho2,*xmm))/(4*eps**2)
            H[j,i]=H[i,j]
    return -np.trace(H)

def vq_estimate(rho2, n=600):
    cnt=0
    for _ in range(n):
        UA,UB=haar_SU2(),haar_SU2()
        r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
        cnt+=np.real((2*np.trace(_Pp@r)+1)/3)>2/3
    return cnt/n

print("="*72)
print("CRITICAL EXPONENT EXTRACTION: Recovery Geometry Phase Boundaries")
print("="*72)
np.random.seed(42); N=4

# ── Part A: kappa_Q ~ |chi_t - pi|^beta (near pi) ────────────────────────
print()
print("=== PART A: kappa_Q ~ |chi_t - pi|^beta ===")
print("Dense sampling near chi_t=pi for log-log fit")

# Very dense near pi: need 1.5+ decades in delta = pi - chi_t
deltas_A = np.array([
    0.200, 0.150, 0.100, 0.080, 0.060,  # far: delta in [0.06, 0.20]
    0.050, 0.040, 0.030, 0.025, 0.020,  # mid: delta in [0.02, 0.05]
    0.015, 0.010, 0.007, 0.005, 0.003,  # near: delta in [0.003, 0.015]
    0.002, 0.001                          # very near
])
chi_ts_A = np.pi - deltas_A

kQ_A = []
print(f"\n  {'delta/pi':>9}  {'chi_t/pi':>9}  {'kappa_Q':>10}  {'log(delta)':>10}  {'log(kQ)':>10}")
print("  "+"-"*55)
for i, (delta, chi_t) in enumerate(zip(deltas_A, chi_ts_A)):
    rho2 = oat_rho2_exact(N, chi_t)
    Fo, xo = find_F_opt(rho2)
    kQ = kappa_Q(rho2, xo)
    kQ_A.append(kQ)
    log_d = np.log(delta) if delta>0 else float('nan')
    log_k = np.log(max(kQ,1e-12))
    print(f"  {delta/np.pi:9.5f}  {chi_t/np.pi:9.5f}  {kQ:10.6f}  {log_d:10.4f}  {log_k:10.4f}")

kQ_A = np.array(kQ_A)
# Fit power law: log(kQ) = beta*log(delta) + const
mask_fit = (kQ_A > 0.002) & (deltas_A < 0.15)  # avoid near-zero noise
if mask_fit.sum() >= 4:
    log_d = np.log(deltas_A[mask_fit])
    log_k = np.log(np.maximum(kQ_A[mask_fit], 1e-12))
    beta_fit, log_A = np.polyfit(log_d, log_k, 1)
    A_fit = np.exp(log_A)
    residuals = log_k - (beta_fit * log_d + log_A)
    r2 = 1 - np.var(residuals)/np.var(log_k)
    print(f"\n  Power law fit (kQ ~ A*delta^beta):")
    print(f"    beta = {beta_fit:.4f}")
    print(f"    A    = {A_fit:.4f}")
    print(f"    R^2  = {r2:.6f}")
    print(f"    N_fit_points = {mask_fit.sum()}")
    # Compare to analytic prediction
    print(f"\n  Analytic prediction (T_xx ~ sin^{{N-2}}(delta/2) ~ (delta/2)^2 for N=4):")
    print(f"    If kQ ~ T_xx: beta_theory = 2.0")
    print(f"    If kQ ~ T_xx^(1/2): beta_theory = 1.0")
    print(f"    Observed: beta = {beta_fit:.4f}")
    if abs(beta_fit-1.0)<0.3:
        print(f"    -> CONSISTENT WITH beta=1 (square-root of T_xx)")
    elif abs(beta_fit-2.0)<0.3:
        print(f"    -> CONSISTENT WITH beta=2 (linear in T_xx, mean-field)")
    else:
        print(f"    -> ANOMALOUS exponent -- possible non-mean-field geometry")

# ── Part B: V_Q ~ |chi_t - chi_tc|^nu (near QUANTUM-CLASSICAL boundary) ──
print()
print("=== PART B: V_Q ~ |chi_t - chi_tc|^nu near upper phase boundary ===")
print("chi_tc ~ 0.67*pi (estimated from collapse_data)")

# Dense near chi_tc
chi_tc_est = 0.67 * np.pi
chi_ts_B = np.array([
    0.50, 0.55, 0.58, 0.60, 0.62, 0.64,
    0.65, 0.655, 0.660, 0.665, 0.670,
    0.675, 0.680, 0.685, 0.690, 0.700,
    0.71, 0.72, 0.75
]) * np.pi

VQ_B = []
print(f"\n  {'chi_t/pi':>9}  {'V_Q':>8}  {'|chi_t-chi_tc|/pi':>18}")
print("  "+"-"*40)
for chi_t in chi_ts_B:
    rho2 = oat_rho2_exact(N, chi_t)
    vq = vq_estimate(rho2, 600)
    VQ_B.append(vq)
    dist = abs(chi_t - chi_tc_est)/np.pi
    print(f"  {chi_t/np.pi:9.4f}  {vq:8.4f}  {dist:18.5f}")

VQ_B = np.array(VQ_B)

# Find actual chi_tc (where V_Q -> 0)
above_mask = VQ_B > 0.005
if above_mask.any() and (~above_mask).any():
    chi_tc_actual = chi_ts_B[above_mask][-1]
    print(f"\n  Estimated chi_tc = {chi_tc_actual/np.pi:.4f}*pi "
          f"(last point with V_Q>0.005)")
    # Fit power law approaching from below
    below_tc = chi_ts_B < chi_tc_actual
    vq_above_zero = VQ_B[below_tc & (VQ_B>0.005)]
    dist_vals = (chi_tc_actual - chi_ts_B[below_tc & (VQ_B>0.005)])
    if len(dist_vals) >= 4:
        log_dist = np.log(dist_vals)
        log_vq   = np.log(np.maximum(vq_above_zero,1e-6))
        nu, logC = np.polyfit(log_dist, log_vq, 1)
        res = log_vq - (nu*log_dist+logC)
        r2v = 1-np.var(res)/np.var(log_vq)
        print(f"\n  Power law fit (V_Q ~ C*(chi_tc - chi_t)^nu):")
        print(f"    nu  = {nu:.4f}")
        print(f"    C   = {np.exp(logC):.4f}")
        print(f"    R^2 = {r2v:.6f}")
        print(f"    N_fit_points = {len(dist_vals)}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("="*72)
print("CRITICAL EXPONENT SUMMARY")
print("="*72)
print()
print(f"kappa_Q ~ |chi_t-pi|^beta:")
try:
    print(f"  beta = {beta_fit:.4f}, R^2 = {r2:.5f}")
    if r2 > 0.95:
        print(f"  CLEAN POWER LAW (R^2>{0.95:.2f}) -- CRITICAL SCALING CONFIRMED")
        print(f"  -> 'Recovery curvature exhibits critical scaling near the "
              f"chi_t=pi geometric phase boundary'")
    else:
        print(f"  R^2={r2:.4f} -- noisy; need more points or wider range")
except: print("  fit failed")
print()
print(f"V_Q ~ |chi_t-chi_tc|^nu:")
try:
    print(f"  nu = {nu:.4f}, R^2 = {r2v:.5f}")
    if r2v > 0.80:
        print(f"  POWER LAW at QUANTUM-CLASSICAL boundary")
    else:
        print(f"  R^2={r2v:.4f} -- insufficient")
except: print("  insufficient data")
print()
print("Analytic prediction comparison:")
print("  If kQ ~ T_xx = sin^{N-2}(delta/2) for N=4:")
print("    sin^2(delta/2) ~ (delta/2)^2 for small delta")
print("    -> beta_theory = 2.0")
print("  Measured beta -- see above")
print("  Discrepancy = non-mean-field geometry OR crossover regime")
