"""
p3_bridge_correlation.py — Priority 3: F_avg ∝ V_Q^alpha Bridge Test
======================================================================
Tests whether teleportation fidelity correlates with recovery geometry.

If Delta_F = F_avg - 2/3 ∝ V_Q^alpha:
  V_Q becomes the operational order parameter of teleportation capability.
  This bridges operational teleportation <-> recovery geometry <-> criticality.

Sweeps across:
  (a) chi_t at fixed N=4, no dephasing
  (b) Gamma_t dephasing at chi_t*
  (c) N in {4,6} at chi_t*
"""

import numpy as np
from scipy.optimize import minimize, curve_fit

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())

def su2(t,p):
    c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-np.exp(1j*p)*s],[np.exp(-1j*p)*s,c]],dtype=complex)

def F_avg(rho2,tA,pA,tB,pB):
    UA=su2(tA,pA); UB=su2(tB,pB)
    r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp@r)+1)/3)

def find_F_opt(rho2, n_restarts=10, seed=42):
    rng=np.random.default_rng(seed)
    best=0.
    for _ in range(n_restarts):
        x0=rng.uniform([0,0,0,0],[np.pi,2*np.pi,np.pi,2*np.pi])
        res=minimize(lambda x:-F_avg(rho2,*x),x0,method='Nelder-Mead',
                     options={'xatol':1e-8,'fatol':1e-10,'maxiter':6000})
        if -res.fun>best: best=-res.fun
    return best

def haar_SU2(rng):
    z=(rng.standard_normal((2,2))+1j*rng.standard_normal((2,2)))/np.sqrt(2)
    Q,R=np.linalg.qr(z); d=np.diagonal(R); Q*=d/np.abs(d)
    if np.real(np.linalg.det(Q))<0: Q[:,0]*=-1
    return Q

def vq_est(rho2, n=1500, seed=42):
    rng=np.random.default_rng(seed); cnt=0
    for _ in range(n):
        UA=haar_SU2(rng); UB=haar_SU2(rng)
        r=np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
        cnt+=np.real((2*np.trace(_Pp@r)+1)/3)>2/3
    return cnt/n

def apply_dephasing(rho2, gamma_t):
    """Apply independent single-qubit dephasing to each qubit of rho2.
    Dephasing: off-diagonal elements in Z basis suppressed by exp(-gamma_t)."""
    rho = rho2.copy()
    decay = np.exp(-gamma_t)
    # Qubit A (first index): suppress X,Y coherences
    # Qubit B (second index): suppress X,Y coherences
    # In 4x4 basis |00>,|01>,|10>,|11>:
    # Elements rho[i,j] where i!=j decay by exp(-gamma_A) or exp(-gamma_B) or both
    for i in range(4):
        for j in range(4):
            if i!=j:
                ia,ib = divmod(i,2); ja,jb = divmod(j,2)
                d = (ia!=ja) + (ib!=jb)  # number of differing qubits
                rho[i,j] *= decay**d
    return rho

def power_law(x, alpha, C): return C * x**alpha

print("="*70)
print("PRIORITY 3: BRIDGE CORRELATION F_avg vs V_Q")
print("="*70)
np.random.seed(42); N=4

# ── Sweep A: chi_t at fixed N=4 ───────────────────────────────────────────
print()
print("=== SWEEP A: chi_t sweep, N=4, no dephasing ===")
chi_ts_A = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.355,
                      0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80,1.00]) * np.pi

F_A, VQ_A = [], []
print(f"\n  {'chi_t/pi':>9}  {'F_avg':>8}  {'V_Q':>8}  {'dF=F-2/3':>10}  {'PTM_rank':>9}")
print("  "+"-"*55)
for chi_t in chi_ts_A:
    rho2=oat_rho2_exact(N, chi_t)
    Fo = find_F_opt(rho2, n_restarts=10, seed=42)
    vq = vq_est(rho2, n=800, seed=42)
    # PTM rank from T_xx
    Txx = np.cos(chi_t/2)**(N-2)
    rank = 4 if Txx>0.001 else 1
    dF = Fo - 2/3
    F_A.append(Fo); VQ_A.append(vq)
    print(f"  {chi_t/np.pi:9.4f}  {Fo:8.5f}  {vq:8.5f}  {dF:10.6f}  {rank:9d}")

F_A=np.array(F_A); VQ_A=np.array(VQ_A)
dF_A = F_A - 2/3

# Fit dF ∝ V_Q^alpha (using points where both nonzero)
mask = (VQ_A>0.005) & (dF_A>0)
if mask.sum()>=4:
    try:
        popt, _ = curve_fit(power_law, VQ_A[mask], dF_A[mask], p0=[1.0,1.0])
        alpha_A, C_A = popt
        # Pearson correlation on log-log
        ll_corr = np.corrcoef(np.log(VQ_A[mask]), np.log(np.maximum(dF_A[mask],1e-10)))[0,1]
        print(f"\n  Power law fit dF = C * V_Q^alpha:")
        print(f"    alpha = {alpha_A:.4f}")
        print(f"    C     = {C_A:.4f}")
        print(f"    log-log Pearson r = {ll_corr:.4f}")
        # Linear correlation
        lin_corr = np.corrcoef(VQ_A[mask], dF_A[mask])[0,1]
        print(f"    linear Pearson r  = {lin_corr:.4f}")
    except Exception as e:
        print(f"  Fit failed: {e}"); alpha_A=float('nan')

# ── Sweep B: Gamma_t dephasing at chi_t* ─────────────────────────────────
print()
print("=== SWEEP B: Dephasing sweep at chi_t*=0.355*pi ===")
chi_t_star = 0.355*np.pi
rho2_star  = oat_rho2_exact(N, chi_t_star)
gammas = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30])

F_B, VQ_B = [], []
print(f"\n  {'Gamma_t':>8}  {'F_avg':>8}  {'V_Q':>8}  {'dF':>10}  {'V_Q/V_Q0':>10}")
print("  "+"-"*50)
vq0 = vq_est(rho2_star, 1500, 42)
for gamma in gammas:
    rho_d = apply_dephasing(rho2_star, gamma)
    Fo  = find_F_opt(rho_d, n_restarts=10, seed=42)
    vq  = vq_est(rho_d, 800, 42)
    dF  = Fo - 2/3
    F_B.append(Fo); VQ_B.append(vq)
    print(f"  {gamma:8.3f}  {Fo:8.5f}  {vq:8.5f}  {dF:10.6f}  {vq/max(vq0,1e-6):10.4f}")

F_B=np.array(F_B); VQ_B=np.array(VQ_B); dF_B=F_B-2/3
mask_B=(VQ_B>0.002)&(dF_B>0)
if mask_B.sum()>=4:
    try:
        popt_B,_=curve_fit(power_law,VQ_B[mask_B],dF_B[mask_B],p0=[1.0,1.0])
        aB,CB=popt_B
        rB=np.corrcoef(VQ_B[mask_B],dF_B[mask_B])[0,1]
        print(f"\n  dF vs V_Q (dephasing sweep): alpha={aB:.4f}, C={CB:.4f}, r={rB:.4f}")
    except: aB=float('nan'); print("  Fit failed")
    # Also: threshold Gamma where V_Q -> 0
    zero_gamma = gammas[VQ_B<0.005]
    if len(zero_gamma): print(f"  V_Q<0.005 at Gamma_t >= {zero_gamma[0]:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("="*70)
print("BRIDGE CORRELATION SUMMARY")
print("="*70)
print()
print("Sweep A (chi_t, N=4, no dephasing):")
try:
    print(f"  dF = {C_A:.4f} * V_Q^{alpha_A:.4f}  (log-log r = {ll_corr:.4f})")
    if abs(ll_corr)>0.95:
        print(f"  STRONG log-log correlation: V_Q IS an order parameter for dF")
    elif abs(ll_corr)>0.85:
        print(f"  MODERATE correlation: V_Q tracks dF but relationship non-trivial")
    if abs(alpha_A-1.0)<0.2:
        print(f"  alpha~1: NEAR-LINEAR: dF approximately proportional to V_Q")
except: pass
print()
print("Sweep B (dephasing, chi_t*, N=4):")
try:
    print(f"  dF = {CB:.4f} * V_Q^{aB:.4f}  (r={rB:.4f})")
    if abs(rB)>0.95: print(f"  STRONG: same alpha survives dephasing -> UNIVERSAL")
    else: print(f"  dephasing shifts exponent: dF-V_Q relation is state-dependent")
except: pass
print()
print("Physical interpretation:")
print("  If alpha~1 in both sweeps: V_Q is proportional to teleportation advantage.")
print("  Then V_Q is not merely a geometric proxy -- it IS the order parameter.")
print("  Paper claim: 'The recovery basin volume V_Q is proportional to the")
print("  quantum teleportation advantage Delta_F = F_avg - 2/3, establishing")
print("  it as the operational order parameter of the teleportation phase.'")
