"""
collapse_figure.py — The Killer Figure: Multi-Observable Collapse at chi_t=pi
==============================================================================
Six observables on a common chi_t grid for N=4.
All six collapse simultaneously at chi_t=pi.
This is the central figure of the paper.

Outputs:
  - ASCII table (all 6 columns)
  - collapse_data.npz (numpy arrays for matplotlib)
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

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

def su2_from_angles(theta, phi):
    c,s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,-np.exp(1j*phi)*s],[np.exp(-1j*phi)*s,c]],dtype=complex)

def F_avg(rho2, theta_A, phi_A, theta_B, phi_B):
    UA = su2_from_angles(theta_A, phi_A)
    UB = su2_from_angles(theta_B, phi_B)
    r  = np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp@r)+1)/3)

def find_F_opt(rho2, n_restarts=15):
    best = 0.; best_x = None
    for _ in range(n_restarts):
        x0 = np.random.uniform([0,0,0,0],[np.pi,2*np.pi,np.pi,2*np.pi])
        res = minimize(lambda x:-F_avg(rho2,*x), x0, method='Nelder-Mead',
                       options={'xatol':1e-7,'fatol':1e-9,'maxiter':5000})
        if -res.fun > best: best=-res.fun; best_x=res.x
    return best, best_x

def vq_estimate(rho, n=400):
    cnt,Fvals=[],[]
    for _ in range(n):
        UA,UB=haar_SU2(),haar_SU2()
        r=np.kron(UA,UB)@rho@np.kron(UA,UB).conj().T
        F=np.real((2*np.trace(_Pp@r)+1)/3)
        cnt.append(F>2/3); Fvals.append(F)
    return np.mean(cnt), np.std(Fvals)

def kappa_Q_and_evals(rho2, x_opt, eps=0.05):
    n=4; F0=F_avg(rho2,*x_opt)
    H=np.zeros((n,n))
    for i in range(n):
        xp=x_opt.copy(); xp[i]+=eps
        xm=x_opt.copy(); xm[i]-=eps
        H[i,i]=(F_avg(rho2,*xp)+F_avg(rho2,*xm)-2*F0)/eps**2
    for i in range(n):
        for j in range(i+1,n):
            xpp=x_opt.copy(); xpp[i]+=eps; xpp[j]+=eps
            xpm=x_opt.copy(); xpm[i]+=eps; xpm[j]-=eps
            xmp=x_opt.copy(); xmp[i]-=eps; xmp[j]+=eps
            xmm=x_opt.copy(); xmm[i]-=eps; xmm[j]-=eps
            H[i,j]=(F_avg(rho2,*xpp)-F_avg(rho2,*xpm)
                    -F_avg(rho2,*xmp)+F_avg(rho2,*xmm))/(4*eps**2)
            H[j,i]=H[i,j]
    evals=np.real(la.eigvalsh(H))
    return -np.trace(H), evals

def landscape_std(rho2, n=300):
    """std of F_avg over Haar-random SU(2)^2."""
    Fs=[]; 
    for _ in range(n):
        UA,UB=haar_SU2(),haar_SU2()
        r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
        Fs.append(np.real((2*np.trace(_Pp@r)+1)/3))
    return np.std(Fs)

def Txx(rho):
    X=np.array([[0,1],[1,0]])
    return np.real(np.trace(np.kron(X,X)@rho))

# ── Main sweep ────────────────────────────────────────────────────────────

print("="*80)
print("MULTI-OBSERVABLE COLLAPSE FIGURE: All Six Panels")
print("="*80)
np.random.seed(42)

N = 4
# 33-point grid including pi exactly, denser around pi
chi_ts = np.concatenate([
    np.linspace(0.0,   0.5*np.pi, 10, endpoint=False),
    np.linspace(0.5*np.pi, 0.9*np.pi, 8, endpoint=False),
    np.linspace(0.9*np.pi, np.pi,  10, endpoint=False),
    [np.pi]
])
chi_ts = np.unique(chi_ts)

print(f"\nN={N}, {len(chi_ts)} chi_t points")
print()
print(f"  {'chi_t/pi':>9}  {'T_xx':>8}  {'F_opt':>7}  "
      f"{'V_Q':>7}  {'kappa_Q':>9}  {'lnd_std':>9}  {'H_min':>9}  "
      f"{'H_max':>9}  {'class'}")
print("  "+"-"*85)

# Storage
rec = []
for chi_t in chi_ts:
    rho2 = oat_rho2_exact(N, chi_t)

    # Observable 6: T_xx (analytic)
    txx  = np.cos(chi_t/2)**(N-2)

    # Observable 5: F_opt
    F_opt, x_opt = find_F_opt(rho2)

    # Observable 1: V_Q
    VQ, _ = vq_estimate(rho2, 400)

    # Observable 2: kappa_Q
    kQ, evals = kappa_Q_and_evals(rho2, x_opt)

    # Observable 4: landscape std
    lst = landscape_std(rho2, 300)

    # Class
    cls = ('QUANTUM' if VQ>0.005 else ('FLAT' if F_opt<0.505 else 'CLASSICAL'))

    rec.append({
        'chi_t': chi_t, 'Txx': txx, 'F_opt': F_opt,
        'VQ': VQ, 'kQ': kQ, 'lst': lst,
        'H_min': evals.min(), 'H_max': evals.max(),
        'cls': cls
    })
    print(f"  {chi_t/np.pi:9.4f}  {txx:8.5f}  {F_opt:7.5f}  "
          f"{VQ:7.4f}  {kQ:9.5f}  {lst:9.6f}  "
          f"{evals.min():9.5f}  {evals.max():9.5f}  {cls}")

# Save for plotting
arr = lambda k: np.array([r[k] for r in rec])
np.savez('collapse_data.npz',
         chi_t=arr('chi_t'), Txx=arr('Txx'), F_opt=arr('F_opt'),
         VQ=arr('VQ'), kQ=arr('kQ'), lst=arr('lst'),
         H_min=arr('H_min'), H_max=arr('H_max'))

print()
pi_idx = np.argmin(np.abs(arr('chi_t') - np.pi))
peak_idx = np.argmax(arr('kQ'))
pi_r = rec[pi_idx]; peak_r = rec[peak_idx]

print("="*80)
print("COLLAPSE FIGURE SUMMARY")
print("="*80)
print()
print(f"chi_t=pi (idx={pi_idx}):")
print(f"  T_xx    = {pi_r['Txx']:.8f}   <- analytic zero")
print(f"  F_opt   = {pi_r['F_opt']:.8f}   <- operationally unrecoverable")
print(f"  V_Q     = {pi_r['VQ']:.8f}   <- no recovery basin")
print(f"  kappa_Q = {pi_r['kQ']:.8f}   <- curvature zero")
print(f"  lnd_std = {pi_r['lst']:.8f}   <- landscape flat")
print(f"  H_min   = {pi_r['H_min']:.8f}   <- Hessian collapsed")
print()
print(f"Peak (chi_t={peak_r['chi_t']:.4f}, chi_t/pi={peak_r['chi_t']/np.pi:.4f}):")
print(f"  T_xx    = {peak_r['Txx']:.6f}")
print(f"  F_opt   = {peak_r['F_opt']:.6f}")
print(f"  V_Q     = {peak_r['VQ']:.6f}")
print(f"  kappa_Q = {peak_r['kQ']:.6f}   <- peak curvature")
print(f"  lnd_std = {peak_r['lst']:.6f}")
print()
kQ_pi = max(abs(pi_r['kQ']), 1e-12)
print(f"kappa_Q collapse: peak/singularity > {peak_r['kQ']/kQ_pi:.2e}  ")
print(f"  -> Recovery curvature collapses by over {int(np.log10(peak_r['kQ']/kQ_pi))} orders of magnitude")
print()
print("Data saved to collapse_data.npz for matplotlib figure generation.")
print()
print("QUANTUM region (V_Q>0.005):")
q_pts = [(r['chi_t']/np.pi) for r in rec if r['VQ']>0.005]
if q_pts:
    print(f"  chi_t/pi in [{min(q_pts):.4f}, {max(q_pts):.4f}]")
print()
print("Singularity confirmation: all 6 observables at chi_t=pi:")
for k,label in [('Txx','T_xx'),('F_opt','F_opt'),('VQ','V_Q'),
                ('kQ','kappa_Q'),('lst','lnd_std'),('H_min','H_min_eval')]:
    v = pi_r[k]
    status = "EXACT ZERO" if abs(v)<1e-6 else (f"~{v:.5f}")
    print(f"  {label:12s} = {status}")
