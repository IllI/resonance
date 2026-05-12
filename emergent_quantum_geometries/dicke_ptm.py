"""
dicke_ptm.py — Session 3: Dicke Model PTM Extension (fixed)

Fixes:
  1. rho_boundary_pair: correct loop-based partial trace (no reshape error)
  2. omega_0=0 to isolate OAT-like interaction (remove spin precession)
  3. Compare concurrence C(t) — basis-independent, no axis confusion
  4. V_Q as primary metric

TPU note: runs locally (dim<=1280). TPU needed for Session 4 (D-LinOSS).
Deployment follows trc_tpu_bible.md: chronos-alice v4-8 us-central2-b.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import curve_fit

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

# ── Operators ─────────────────────────────────────────────────────────────

def boson_ops(n_max):
    a = np.zeros((n_max, n_max))
    for n in range(1, n_max): a[n-1,n] = np.sqrt(n)
    return a, a.T, np.diag(np.arange(n_max, dtype=float))

def collective_spin(N, axis):
    ops = {'x': np.array([[0,1],[1,0]])/2,
           'y': np.array([[0,-1j],[1j,0]])/2,
           'z': np.array([[1,0],[0,-1]])/2}
    op = ops[axis]; dim = 2**N
    J = np.zeros((dim,dim), dtype=complex)
    for i in range(N):
        s = [np.eye(2)]*N; s[i] = op
        term = s[0]
        for m in s[1:]: term = np.kron(term, m)
        J += term
    return J

def build_dicke_H(N, n_max, omega_m, g):
    """omega_0=0: pure phonon-mediated OAT (no spin precession)."""
    d_s, d_b = 2**N, n_max
    Jx = collective_spin(N, 'x')
    a, ad, Nph = boson_ops(n_max)
    H  = np.zeros((d_s*d_b, d_s*d_b), dtype=complex)
    H += omega_m * np.kron(np.eye(d_s), Nph)
    H += g       * np.kron(Jx, a + ad)
    return H


def initial_state_zpolarized(N, n_max):
    """|0>^N ⊗ |0>_phonon  (z-polarized spins, phonon vacuum).
    NOT an eigenstate of Jx^2 -> entanglement develops under H_eff~-chi_eff*Jx^2.
    """
    psi_s = np.zeros(2**N); psi_s[0] = 1.0   # |00...0>
    psi_b = np.zeros(n_max); psi_b[0] = 1.0
    return np.kron(psi_s, psi_b)

def rho_boundary_pair(psi, N, n_max):
    """
    Partial trace: trace out bulk spins 1..N-2 and all phonons.
    Boundary pair: site 0 (MSB) and site N-1 (LSB).
    Site ordering: row index = s0*2^(N-1) + ... + s_{N-1}*1
    For boundary sites (i0,iN) and bulk k in 0..2^(N-2)-1:
      row = i0*(2^(N-1)) + k*2 + iN
    """
    d_s, d_b = 2**N, n_max
    rho_full = np.outer(psi, psi.conj())
    # Trace phonons
    rho_s = np.einsum('iajb->ij',
                      rho_full.reshape(d_s, d_b, d_s, d_b))
    if N == 2:
        return rho_s / np.trace(rho_s)
    d_bulk = 2**(N-2)
    rho2 = np.zeros((4,4), dtype=complex)
    for ib in range(4):
        i0, iN = ib >> 1, ib & 1
        for jb in range(4):
            j0, jN = jb >> 1, jb & 1
            val = 0j
            for k in range(d_bulk):
                row = i0*(d_bulk*2) + k*2 + iN
                col = j0*(d_bulk*2) + k*2 + jN
                val += rho_s[row, col]
            rho2[ib, jb] = val
    return rho2 / np.trace(rho2)

def concurrence(rho):
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy,sy)
    R  = la.sqrtm(rho @ ss @ rho.conj() @ ss)
    ev = sorted(np.real(np.sqrt(np.maximum(la.eigvals(R).real,0))), reverse=True)
    return max(0., ev[0]-ev[1]-ev[2]-ev[3])

def Txx(rho): return np.real(np.trace(np.kron(np.array([[0,1],[1,0]]),
                                              np.array([[0,1],[1,0]])) @ rho))

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())
def haar_SU2():
    z=(np.random.randn(2,2)+1j*np.random.randn(2,2))/np.sqrt(2)
    Q,R=np.linalg.qr(z); d=np.diagonal(R); Q*=d/np.abs(d)
    if np.real(np.linalg.det(Q))<0: Q[:,0]*=-1
    return Q
def vq_estimate(rho, n=200):
    cnt,Fm=0,0
    for _ in range(n):
        UA,UB=haar_SU2(),haar_SU2()
        r=np.kron(UA,UB)@rho@np.kron(UA,UB).conj().T
        F=np.real((2*np.trace(_Pp@r)+1)/3)
        cnt+=F>2/3; Fm=max(Fm,F)
    return cnt/n, Fm

print("="*72)
print("SESSION 3: DICKE MODEL PTM EXTENSION (omega_0=0, basis-independent)")
print("="*72)
np.random.seed(42)

omega_m = 1.0
g_c     = 0.5   # sqrt(omega_0*omega_m)/2, here omega_0->0 so this is reference only
n_max   = 20

# ── Panel A: Concurrence C(t) — Dicke vs OAT prediction ─────────────────
print()
print("=== PANEL A: Concurrence C(t) — Dicke vs OAT (chi_eff=g²/omega_m) ===")

for N in [4, 6]:
    print(f"\n  N={N}:")
    print(f"  {'g':>6}  {'chi_eff':>9}  {'t':>5}  "
          f"{'C(Dicke)':>10}  {'C(OAT)':>8}  {'|diff|':>8}")
    print("  "+"-"*52)
    for g in [0.05, 0.10, 0.20]:
        chi_eff = g**2 / omega_m
        H       = build_dicke_H(N, n_max, omega_m, g)
        evals, evecs = la.eigh(H)
        psi0    = initial_state_zpolarized(N, n_max)
        coeffs  = evecs.conj().T @ psi0
        for t in [1.0, 2.0, 5.0, 10.0]:
            phases = np.exp(-1j*evals*t)
            psit   = evecs@(phases*coeffs)
            rho2   = rho_boundary_pair(psit, N, n_max)
            CD     = concurrence(rho2)
            # OAT comparison: same chi_eff*t as effective coupling
            rho2_oat = oat_rho2_exact(N, chi_eff*t)
            CA       = concurrence(rho2_oat)
            diff     = abs(CD - CA)
            print(f"  {g:6.3f}  {chi_eff:9.5f}  {t:5.1f}  "
                  f"{CD:10.6f}  {CA:8.6f}  {diff:8.5f}")

# ── Panel B: chi_eff fit from C(t) curve ─────────────────────────────────
print()
print("=== PANEL B: Fitted chi_eff vs Theoretical g²/omega_m ===")
print(f"  {'N':>3}  {'g':>6}  {'chi_theory':>11}  {'chi_fit':>9}  "
      f"{'ratio':>7}  {'match?':>7}")
print("  "+"-"*52)

t_grid = np.linspace(0.5, 30.0, 50)   # longer range: chi_eff small, need chi_eff*t~1

for N in [4, 6]:
    for g in [0.05, 0.10, 0.20, 0.40]:
        chi_theory = g**2 / omega_m
        H = build_dicke_H(N, n_max, omega_m, g)
        evals, evecs = la.eigh(H)
        psi0   = initial_state_zpolarized(N, n_max)
        coeffs = evecs.conj().T @ psi0
        C_data = []
        for t in t_grid:
            psit = evecs@(np.exp(-1j*evals*t)*coeffs)
            rho2 = rho_boundary_pair(psit, N, n_max)
            C_data.append(concurrence(rho2))
        C_data = np.array(C_data)
        # Fit: C(t) ~ C_peak * |cos^{N-2}(chi_fit*t/2)|
        # Use C_peak from OAT at chi_theory
        C_peak = max(C_data) if max(C_data)>0 else 0.01
        def model(t, chi):
            return C_peak * np.abs(np.cos(chi*t/2))**(N-2)
        try:
            popt,_ = curve_fit(model, t_grid, C_data,
                               p0=[chi_theory], bounds=(0,1.0), maxfev=5000)
            chi_fit = popt[0]
            ratio   = chi_fit/chi_theory if chi_theory>0 else 0
            match   = "YES" if 0.7<ratio<1.4 else "NO"
        except:
            chi_fit, ratio, match = float('nan'), float('nan'), "FAIL"
        print(f"  {N:3d}  {g:6.3f}  {chi_theory:11.6f}  {chi_fit:9.6f}  "
              f"{ratio:7.3f}  {match:>7}")

# ── Panel C: V_Q and T_xx for Dicke boundary pair ───────────────────────
print()
print("=== PANEL C: V_Q and T_xx for Dicke Boundary Pair (N=4, t=5.0) ===")
print(f"  {'g':>6}  {'g/g_c':>6}  {'chi_eff':>8}  {'C':>8}  "
      f"{'T_xx':>8}  {'V_Q':>8}  {'F_max':>8}  {'class'}")
print("  "+"-"*68)

N, t = 4, 5.0
for g in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    chi_eff = g**2/omega_m
    if g == 0.0:
        rho2 = np.eye(4)/4; CD=0.0; Td=0.0; VQ=0.0; Fm=2/3
    else:
        H = build_dicke_H(N, n_max, omega_m, g)
        evals,evecs = la.eigh(H)
        psi0 = initial_state_zpolarized(N, n_max)
        coeffs = evecs.conj().T @ psi0
        psit = evecs@(np.exp(-1j*evals*t)*coeffs)
        rho2 = rho_boundary_pair(psit, N, n_max)
        CD = concurrence(rho2); Td = Txx(rho2)
        VQ, Fm = vq_estimate(rho2, 200)
    cls = 'QUANTUM' if VQ>0.01 else ('FLAT' if Fm<0.51 else 'CLASSICAL')
    print(f"  {g:6.3f}  {g/g_c:6.3f}  {chi_eff:8.5f}  {CD:8.5f}  "
          f"{Td:8.5f}  {VQ:8.4f}  {Fm:8.4f}  {cls}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("="*72)
print("SESSION 3 SUMMARY")
print("="*72)
print()
print("Framework generalisation test:")
print("  Panel A: C(Dicke) vs C(OAT,chi_eff) -- do they track?")
print("  Panel B: fitted chi_eff / g^2/omega_m -- ratio~1 = OAT limit holds")
print("  Panel C: V_Q>0 for Dicke boundary pair = teleportation resource present")
print()
print("Connection to Rey 2026 (arXiv:2602.06114):")
print("  If ratio~1 at g<<g_c: 2-mode squeezing IS the OAT boundary resource")
print("  If ratio!=1 near g_c: phonon sector modifies boundary entanglement")
print("  Collapses/revivals in <Jz(t)> correspond to C(t) oscillations")
print()
print("TPU deployment note (per trc_tpu_bible.md):")
print("  Session 3 fits locally (dim<=1280, runtime<60s)")
print("  Session 4 (D-LinOSS V_Q training) -> chronos-alice v4-8 us-central2-b")
print("  Upload: gcloud compute tpus tpu-vm scp dicke_ptm.py")
print("          chronos-alice-node:~/dicke_ptm.py")
print("          --project=time-emission --zone=us-central2-b")
