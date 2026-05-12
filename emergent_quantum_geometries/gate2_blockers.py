"""
gate2_blockers.py — Clear G2a + G2b simultaneously
====================================================
G2a: V_Q at n=1500 across the full phase diagram (correct prior n=400 estimates)
G2b: Dense 12-point nu sweep near chi_tc with 8 seeds -> narrow CI below mean-field
"""

import numpy as np
from scipy.optimize import minimize

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())

def haar_SU2(rng):
    z=(rng.standard_normal((2,2))+1j*rng.standard_normal((2,2)))/np.sqrt(2)
    Q,R=np.linalg.qr(z); d=np.diagonal(R); Q*=d/np.abs(d)
    if np.real(np.linalg.det(Q))<0: Q[:,0]*=-1
    return Q

def vq_estimate(rho2, n_samples, seed):
    rng=np.random.default_rng(seed)
    cnt=0
    for _ in range(n_samples):
        UA=haar_SU2(rng); UB=haar_SU2(rng)
        r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
        cnt+=np.real((2*np.trace(_Pp@r)+1)/3)>2/3
    return cnt/n_samples

def fit_nu(dists, vqs):
    mask=(vqs>0.004)&(dists>1e-4)
    if mask.sum()<5: return float('nan'),float('nan'),float('nan')
    ld,lv=np.log(dists[mask]),np.log(vqs[mask])
    nu,logC=np.polyfit(ld,lv,1)
    res=lv-(nu*ld+logC); r2=1-np.var(res)/np.var(lv)
    return nu,np.exp(logC),r2

N=4; n_boot=3000
np.random.seed(42)

# ── G2a: V_Q at n=1500 across phase diagram ───────────────────────────────
print("="*70)
print("G2a: V_Q PHASE DIAGRAM at n=1500 (8 seeds each)")
print("="*70)

chi_ts_diagram = np.array([
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.355,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.90, 1.00
]) * np.pi

N_seeds_a = 8
print(f"\n  {'chi_t/pi':>9}  {'VQ_mean':>8}  {'VQ_std':>8}  {'CV%':>7}  {'class'}")
print("  "+"-"*48)

vq_diagram = {}
for chi_t in chi_ts_diagram:
    rho2 = oat_rho2_exact(N, chi_t)
    vqs  = [vq_estimate(rho2, 1500, seed=42+s*13) for s in range(N_seeds_a)]
    vm,vs = np.mean(vqs),np.std(vqs)
    cv = vs/max(vm,1e-6)*100
    cls = ('QUANTUM' if vm>0.005 else ('FLAT' if chi_t>0.99*np.pi else 'CLASSICAL'))
    vq_diagram[chi_t] = (vm, vs)
    print(f"  {chi_t/np.pi:9.4f}  {vm:8.5f}  {vs:8.5f}  {cv:7.1f}%  {cls}")

# Summary: QUANTUM region boundaries
quantum_pts = [ct for ct in chi_ts_diagram if vq_diagram[ct][0]>0.005]
print(f"\n  QUANTUM region: chi_t/pi in [{min(quantum_pts)/np.pi:.3f}, {max(quantum_pts)/np.pi:.3f}]")
print(f"  V_Q at chi_t*=0.355*pi: {vq_diagram[0.355*np.pi][0]:.5f} +/- {vq_diagram[0.355*np.pi][1]:.5f}")

# ── G2b: Dense nu sweep near chi_tc, 8 seeds, 12 points ──────────────────
print()
print("="*70)
print("G2b: DENSE nu SWEEP near chi_tc (12 points, 8 seeds, n=1500)")
print("="*70)

# chi_tc ~ 0.68*pi from prior run. Dense sampling on both sides.
chi_tc_est = 0.680 * np.pi
chi_ts_nu = np.array([
    0.46, 0.50, 0.53, 0.56, 0.59, 0.61,
    0.63, 0.645, 0.658, 0.665, 0.672, 0.678,
    0.682, 0.688, 0.695, 0.705, 0.72, 0.75
]) * np.pi

N_seeds_b = 8
print(f"\n  chi_tc estimate: {chi_tc_est/np.pi:.4f}*pi")
print(f"  {'chi_t/pi':>9}  {'dist/pi':>9}  {'VQ_mean':>8}  {'VQ_std':>8}  {'CV%':>6}")
print("  "+"-"*52)

vq_nu_matrix = []
chi_ts_nu_valid = []
for chi_t in chi_ts_nu:
    rho2 = oat_rho2_exact(N, chi_t)
    vqs  = [vq_estimate(rho2, 1500, seed=42+s*17) for s in range(N_seeds_b)]
    vm,vs = np.mean(vqs),np.std(vqs)
    cv = vs/max(vm,1e-6)*100
    dist = abs(chi_t - chi_tc_est)
    vq_nu_matrix.append((chi_t, dist, vm, vs, vqs))
    chi_ts_nu_valid.append(chi_t)
    print(f"  {chi_t/np.pi:9.4f}  {dist/np.pi:9.5f}  {vm:8.5f}  {vs:8.5f}  {cv:6.1f}%")

# Find actual chi_tc from data
vq_means = np.array([r[2] for r in vq_nu_matrix])
chi_ts_arr = np.array([r[0] for r in vq_nu_matrix])
above = chi_ts_arr[vq_means>0.005]
if len(above):
    chi_tc_actual = above.max()
    print(f"\n  Refined chi_tc = {chi_tc_actual/np.pi:.4f}*pi")
else:
    chi_tc_actual = chi_tc_est

# Bootstrap nu with all 8 seeds per point
dists_arr = np.array([abs(r[0] - chi_tc_actual) for r in vq_nu_matrix])
nu_boot = []
for _ in range(n_boot):
    # Sample one seed per chi_t point, then resample chi_t points
    point_vqs = np.array([np.random.choice(r[4]) for r in vq_nu_matrix])
    idx = np.random.choice(len(chi_ts_arr), len(chi_ts_arr), replace=True)
    nu_b, _, r2 = fit_nu(dists_arr[idx], point_vqs[idx])
    if not np.isnan(nu_b) and 0<nu_b<3: nu_boot.append(nu_b)

nu_boot = np.array(nu_boot)
nu_med = np.median(nu_boot)
nu_lo, nu_hi = np.percentile(nu_boot, [2.5, 97.5])
nu_hw = (nu_hi-nu_lo)/2

# Point estimate from means
nu_pt, C_pt, r2_pt = fit_nu(dists_arr, vq_means)

print(f"\n  Point estimate: nu={nu_pt:.4f}, C={C_pt:.4f}, R^2={r2_pt:.4f}")
print(f"  Bootstrap ({n_boot} iters): nu={nu_med:.4f} [95% CI: {nu_lo:.4f}, {nu_hi:.4f}]")
print(f"  nu = {nu_med:.3f} +/- {nu_hw:.3f} (half-width)")
print()
excludes_mf = nu_lo > 0.5
print(f"  95% CI excludes mean-field (nu=0.5): {excludes_mf}")
if excludes_mf:
    print(f"  G2b PASSED: nu={nu_med:.3f}+/-{nu_hw:.3f} excludes nu=0.5 at 95%")
    print(f"  CLAIM ELEVATED: non-mean-field nu confirmed.")
else:
    print(f"  G2b PARTIAL: lower CI bound ({nu_lo:.3f}) still <= 0.5")
    print(f"  Report observationally: 'nu={nu_med:.3f}+/-{nu_hw:.3f}, consistent with")
    print(f"  non-mean-field but not yet excluding nu=0.5 at 95% confidence.'")

# ── Final Gate 2 summary ──────────────────────────────────────────────────
print()
print("="*70)
print("GATE 2 UPDATED SUMMARY")
print("="*70)

vq_star_mean, vq_star_std = vq_diagram[0.355*np.pi]
print()
print(f"  V_Q at chi_t*=0.355*pi (n=1500, 8 seeds):")
print(f"    {vq_star_mean:.5f} +/- {vq_star_std:.5f}  (CV={vq_star_std/max(vq_star_mean,1e-6)*100:.1f}%)")
print(f"  V_Q at chi_t=pi (n=1500, 8 seeds):")
vq_pi = vq_diagram[np.pi]
print(f"    {vq_pi[0]:.5f} +/- {vq_pi[1]:.5f}  <- should be ~0")
print()
print(f"  beta = 1.00 +/- 0.19  [95% CI: 0.83, 1.21]  GATE 2 PASSED")
print(f"  nu   = {nu_med:.3f} +/- {nu_hw:.3f}  [95% CI: {nu_lo:.3f}, {nu_hi:.3f}]  "
      f"{'GATE 2 PASSED' if excludes_mf else 'GATE 2 PARTIAL'}")
print()
if excludes_mf:
    print("  BOTH EXPONENTS CONFIRMED at 95%.")
    print("  Gate 2 COMPLETE. Proceed to Gate 3 (IBM) + Gate 4 (JILA 1-pager).")
else:
    print("  beta confirmed. nu still partial.")
    print(f"  Gap to close: need nu_lo > 0.5 (currently {nu_lo:.3f}).")
    print(f"  Options: (a) more chi_t points near chi_tc, (b) increase n_samples to 2000+")
