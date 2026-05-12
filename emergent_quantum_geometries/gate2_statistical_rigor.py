"""
gate2_statistical_rigor.py — Gate 2: Statistical Hardening
============================================================
Kills the concern: "Could these exponents drift under sampling?"

Five tests:
  A. Bootstrap 95% CI for beta (kQ ~ |chi_t - pi|^beta)
  B. Haar convergence test for V_Q
  C. Optimizer seed independence for kappa_Q
  D. Bootstrap 95% CI for nu (V_Q ~ |chi_t - chi_tc|^nu)
  E. kappa_Q multi-seed stability at chi_t*

Reports: beta = X +/- Y, nu = X +/- Y
"""

import numpy as np
import scipy.linalg as la
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

def su2_from_angles(t,p):
    c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-np.exp(1j*p)*s],[np.exp(-1j*p)*s,c]],dtype=complex)

def F_avg(rho2,tA,pA,tB,pB):
    UA=su2_from_angles(tA,pA); UB=su2_from_angles(tB,pB)
    r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp@r)+1)/3)

def find_F_opt(rho2, n_restarts=12, seed=42):
    rng=np.random.default_rng(seed)
    best,bx=0.,None
    for _ in range(n_restarts):
        x0=rng.uniform([0,0,0,0],[np.pi,2*np.pi,np.pi,2*np.pi])
        res=minimize(lambda x:-F_avg(rho2,*x),x0,method='Nelder-Mead',
                     options={'xatol':1e-8,'fatol':1e-10,'maxiter':8000})
        if -res.fun>best: best=-res.fun; bx=res.x
    return best,bx

def kappa_Q(rho2,x_opt,eps=0.04):
    n=4; F0=F_avg(rho2,*x_opt); H=np.zeros((n,n))
    for i in range(n):
        xp=x_opt.copy();xp[i]+=eps; xm=x_opt.copy();xm[i]-=eps
        H[i,i]=(F_avg(rho2,*xp)+F_avg(rho2,*xm)-2*F0)/eps**2
    for i in range(n):
        for j in range(i+1,n):
            xpp=x_opt.copy();xpp[i]+=eps;xpp[j]+=eps
            xpm=x_opt.copy();xpm[i]+=eps;xpm[j]-=eps
            xmp=x_opt.copy();xmp[i]-=eps;xmp[j]+=eps
            xmm=x_opt.copy();xmm[i]-=eps;xmm[j]-=eps
            H[i,j]=(F_avg(rho2,*xpp)-F_avg(rho2,*xpm)
                    -F_avg(rho2,*xmp)+F_avg(rho2,*xmm))/(4*eps**2)
            H[j,i]=H[i,j]
    return -np.trace(H)

def vq_estimate(rho2, n_samples, seed):
    rng=np.random.default_rng(seed)
    cnt=0
    for _ in range(n_samples):
        UA=haar_SU2(rng); UB=haar_SU2(rng)
        r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
        cnt+=np.real((2*np.trace(_Pp@r)+1)/3)>2/3
    return cnt/n_samples

def fit_beta_from_arrays(deltas, kQs):
    mask=kQs>0.001
    if mask.sum()<4: return float('nan'),float('nan')
    ld,lk=np.log(deltas[mask]),np.log(kQs[mask])
    beta,logA=np.polyfit(ld,lk,1)
    res=lk-(beta*ld+logA); r2=1-np.var(res)/np.var(lk)
    return beta,r2

def fit_nu_from_arrays(dists, vqs):
    mask=(vqs>0.005)&(dists>0)
    if mask.sum()<4: return float('nan'),float('nan')
    ld,lv=np.log(dists[mask]),np.log(vqs[mask])
    nu,logC=np.polyfit(ld,lv,1)
    res=lv-(nu*ld+logC); r2=1-np.var(res)/np.var(lv)
    return nu,r2

print("="*72)
print("GATE 2: STATISTICAL RIGOR")
print("="*72)
N=4; chi_t_star=0.355; chi_tc=0.680*np.pi

# ── Test A: Bootstrap CI for beta ─────────────────────────────────────────
print()
print("=== TEST A: Bootstrap 95% CI for beta ===")
deltas_beta = np.array([0.100, 0.060, 0.030, 0.020, 0.012, 0.008, 0.004, 0.002])
n_seeds_beta = 6
print(f"\n  Running kappa_Q at {len(deltas_beta)} delta points x{n_seeds_beta} seeds...")

kQ_matrix = np.zeros((len(deltas_beta), n_seeds_beta))
for j, delta in enumerate(deltas_beta):
    chi_t = np.pi - delta
    rho2  = oat_rho2_exact(N, chi_t)
    for s in range(n_seeds_beta):
        _, xo = find_F_opt(rho2, n_restarts=10, seed=42+s*7)
        kQ_matrix[j, s] = kappa_Q(rho2, xo)

kQ_mean = kQ_matrix.mean(axis=1)
kQ_std  = kQ_matrix.std(axis=1)
print(f"\n  {'delta/pi':>9}  {'kQ_mean':>9}  {'kQ_std':>9}  {'CV%':>7}")
for j,d in enumerate(deltas_beta):
    cv = kQ_std[j]/max(kQ_mean[j],1e-10)*100
    print(f"  {d/np.pi:9.5f}  {kQ_mean[j]:9.6f}  {kQ_std[j]:9.6f}  {cv:7.1f}%")

# Bootstrap resampling
n_boot = 2000
beta_boot = []
for _ in range(n_boot):
    idx = np.random.choice(len(deltas_beta), len(deltas_beta), replace=True)
    kQ_b = np.array([kQ_matrix[i, np.random.randint(n_seeds_beta)] for i in idx])
    b, _ = fit_beta_from_arrays(deltas_beta[idx], kQ_b)
    if not np.isnan(b): beta_boot.append(b)
beta_boot = np.array(beta_boot)
beta_lo, beta_hi = np.percentile(beta_boot, [2.5, 97.5])
beta_med = np.median(beta_boot)
print(f"\n  Bootstrap ({n_boot} iterations):")
print(f"  beta = {beta_med:.4f}  [95% CI: {beta_lo:.4f}, {beta_hi:.4f}]")
print(f"  beta = {beta_med:.3f} +/- {(beta_hi-beta_lo)/2:.3f}  (half-width)")

# ── Test B: Haar convergence for V_Q ─────────────────────────────────────
print()
print("=== TEST B: Haar Convergence for V_Q ===")
rho2_star = oat_rho2_exact(N, chi_t_star)
n_list = [50, 100, 200, 400, 600, 800, 1000]
n_repeats = 8
print(f"\n  chi_t*={chi_t_star}, {n_repeats} repeats per n_samples")
print(f"  {'n_samples':>10}  {'VQ_mean':>9}  {'VQ_std':>9}  {'CV%':>7}  {'converged?'}")
vq_ref = None
for n in n_list:
    vqs=[vq_estimate(rho2_star, n, seed=42+k*13) for k in range(n_repeats)]
    vm,vs=np.mean(vqs),np.std(vqs)
    if n==1000: vq_ref=vm
    cv=vs/max(vm,1e-6)*100
    conv="YES" if cv<10 else "NO"
    print(f"  {n:10d}  {vm:9.4f}  {vs:9.4f}  {cv:7.1f}%  {conv}")
print(f"\n  V_Q reference (n=1000): {vq_ref:.4f}")

# ── Test C: Optimizer seed independence for kappa_Q ───────────────────────
print()
print("=== TEST C: Optimizer Seed Independence for kappa_Q ===")
rho2_star = oat_rho2_exact(N, chi_t_star)
print(f"\n  chi_t*={chi_t_star}, 10 seeds x varying n_restarts")
print(f"  {'n_restarts':>12}  {'kQ_mean':>9}  {'kQ_std':>9}  {'CV%':>7}")
for nr in [5, 8, 12, 16, 20]:
    kqs=[kappa_Q(rho2_star,find_F_opt(rho2_star,nr,seed=s*11+7)[1]) for s in range(8)]
    km,ks=np.mean(kqs),np.std(kqs)
    print(f"  {nr:12d}  {km:9.6f}  {ks:9.6f}  {ks/max(km,1e-6)*100:7.1f}%")

# ── Test D: Bootstrap CI for nu ───────────────────────────────────────────
print()
print("=== TEST D: Bootstrap 95% CI for nu ===")
chi_ts_nu = np.array([0.50,0.55,0.58,0.61,0.63,0.65,0.660,0.670]) * np.pi
dists_nu  = np.maximum(chi_tc - chi_ts_nu, 0.0)
n_seeds_nu = 6
print(f"\n  Running V_Q at {len(chi_ts_nu)} chi_t points x{n_seeds_nu} seeds...")

vq_matrix_nu = np.zeros((len(chi_ts_nu), n_seeds_nu))
for j, ct in enumerate(chi_ts_nu):
    rho2 = oat_rho2_exact(N, ct)
    for s in range(n_seeds_nu):
        vq_matrix_nu[j,s] = vq_estimate(rho2, 600, seed=42+s*17)

vq_mean_nu = vq_matrix_nu.mean(axis=1)
print(f"\n  {'chi_t/pi':>9}  {'dist/pi':>9}  {'VQ_mean':>9}  {'VQ_std':>9}")
for j in range(len(chi_ts_nu)):
    print(f"  {chi_ts_nu[j]/np.pi:9.4f}  {dists_nu[j]/np.pi:9.5f}  "
          f"{vq_mean_nu[j]:9.4f}  {vq_matrix_nu[j].std():9.4f}")

nu_boot=[]
for _ in range(n_boot):
    idx=np.random.choice(len(chi_ts_nu), len(chi_ts_nu), replace=True)
    vq_b=np.array([vq_matrix_nu[i,np.random.randint(n_seeds_nu)] for i in idx])
    nu_b,_=fit_nu_from_arrays(dists_nu[idx], vq_b)
    if not np.isnan(nu_b): nu_boot.append(nu_b)
nu_boot=np.array(nu_boot)
nu_lo,nu_hi=np.percentile(nu_boot,[2.5,97.5])
nu_med=np.median(nu_boot)
print(f"\n  Bootstrap ({n_boot} iterations):")
print(f"  nu = {nu_med:.4f}  [95% CI: {nu_lo:.4f}, {nu_hi:.4f}]")
print(f"  nu = {nu_med:.3f} +/- {(nu_hi-nu_lo)/2:.3f}")

# ── Final Gate 2 Report ───────────────────────────────────────────────────
print()
print("="*72)
print("GATE 2 REPORT")
print("="*72)
print()
print(f"  beta = {beta_med:.3f}  [95% CI: {beta_lo:.3f}, {beta_hi:.3f}]")
print(f"  nu   = {nu_med:.3f}  [95% CI: {nu_lo:.3f}, {nu_hi:.3f}]")
print()
beta_excludes_2 = beta_hi < 1.95
beta_excludes_05 = beta_lo > 0.5
nu_excludes_05 = nu_lo > 0.5
print(f"  95% CI excludes mean-field (beta=2): {beta_excludes_2}")
print(f"  95% CI excludes beta<0.5:            {beta_excludes_05}")
print(f"  95% CI excludes nu<0.5 (mean-field): {nu_excludes_05}")
print()
if beta_excludes_2 and beta_excludes_05:
    print("  GATE 2 PASSED for beta:")
    print(f"    Non-mean-field exponent confirmed at 95% confidence.")
    print(f"    beta = {beta_med:.3f} +/- {(beta_hi-beta_lo)/2:.3f}")
else:
    print("  GATE 2 PARTIAL for beta -- widen sample range or increase seeds.")
print()
if nu_excludes_05:
    print("  GATE 2 PASSED for nu:")
    print(f"    nu = {nu_med:.3f} +/- {(nu_hi-nu_lo)/2:.3f}")
else:
    print("  GATE 2 PARTIAL for nu.")
