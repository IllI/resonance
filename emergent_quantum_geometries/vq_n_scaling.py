"""
vq_n_scaling.py — Session 2: N-Scaling of V_Q
===============================================
Fixes the Panel D bug from Session 1: uses concurrence-maximizing chi_t*(N),
NOT T_xx-maximizing chi_t (which finds the product state at 2pi).

Sweeps:
  N    in {2, 4, 6, 8, 12, 16}
  chi_t = chi_t*(N) from concurrence peak
  Gamma in {0, Gamma_1, 3*Gamma_1, 10*Gamma_1}

Outputs:
  - Peak V_Q vs N
  - Gamma_t*(N) phase boundary
  - V_Q at chi_t=pi (control) vs chi_t* (signal)
  - D-LinOSS training table
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi, _phi.conj())

GAMMA_1 = 1/118.0   # JILA Sr-87, T2=118s

def haar_SU2():
    z = (np.random.randn(2,2) + 1j*np.random.randn(2,2)) / np.sqrt(2)
    Q, R = np.linalg.qr(z)
    d = np.diagonal(R); Q *= d/np.abs(d)
    if np.real(np.linalg.det(Q)) < 0: Q[:,0] *= -1
    return Q

def vq_estimate(rho, n_samples=300):
    count = 0; Fvals = []
    for _ in range(n_samples):
        UA, UB = haar_SU2(), haar_SU2()
        r = np.kron(UA,UB) @ rho @ np.kron(UA,UB).conj().T
        F = np.real((2*np.trace(_Pp@r)+1)/3)
        Fvals.append(F); count += F > 2/3
    return count/n_samples, np.max(Fvals)

def apply_dephasing(rho, Gamma_t):
    r = rho.copy()
    for i in range(4):
        for j in range(4):
            si,sj = format(i,'02b'), format(j,'02b')
            dH = sum(a!=b for a,b in zip(si,sj))
            r[i,j] *= np.exp(-2*Gamma_t*dH)
    return r

def concurrence(rho):
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy,sy)
    R  = la.sqrtm(rho @ ss @ rho.conj() @ ss)
    eigs = sorted(np.real(np.sqrt(np.maximum(la.eigvals(R).real, 0))), reverse=True)
    return max(0., eigs[0]-eigs[1]-eigs[2]-eigs[3])

def find_chi_t_star_concurrence(N, n_grid=400):
    """Find chi_t maximizing concurrence — the true entanglement peak."""
    grid = np.linspace(0.01, 2*np.pi-0.01, n_grid)
    best_t, best_C = 0., 0.
    for t in grid:
        C = concurrence(oat_rho2_exact(N, t))
        if C > best_C:
            best_C = C; best_t = t
    return best_t, best_C

def gamma_boundary(rho0, threshold=0.005, n_samples=150):
    """Find Gamma_t where V_Q drops below threshold."""
    for Gamma_t in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        r = apply_dephasing(rho0, Gamma_t) if Gamma_t > 0 else rho0
        VQ, _ = vq_estimate(r, n_samples)
        if VQ < threshold:
            return Gamma_t
    return ">0.50"


print("="*72)
print("SESSION 2: N-SCALING OF V_Q (concurrence-based chi_t*)")
print("="*72)
np.random.seed(42)

# ── Table 1: chi_t*(N) from concurrence peak ─────────────────────────────
print()
print("=== TABLE 1: chi_t*(N) from Concurrence Peak ===")
print(f"  {'N':>4}  {'chi_t*':>8}  {'C_peak':>8}  {'T_xx*':>8}  {'notes'}")
print("-"*55)

chi_t_stars = {}
for N in [2, 4, 6, 8, 12, 16]:
    chi_t, C = find_chi_t_star_concurrence(N)
    chi_t_stars[N] = chi_t
    Txx = np.cos(chi_t/2)**(N-2)
    note = "chi_t=pi exception" if N == 2 else ""
    print(f"  {N:4d}  {chi_t:8.4f}  {C:8.5f}  {Txx:8.5f}  {note}")

# ── Table 2: V_Q at chi_t* vs chi_t=pi (signal vs control) ──────────────
print()
print("=== TABLE 2: V_Q Signal vs Control ===")
print(f"  {'N':>4}  {'chi_t*':>8}  {'V_Q(chi_t*)':>13}  "
      f"{'V_Q(pi)':>9}  {'Ratio':>8}  {'Gamma_t*':>10}")
print("-"*65)

results = {}
for N in [2, 4, 6, 8, 12, 16]:
    chi_t = chi_t_stars[N]
    rho_star = oat_rho2_exact(N, chi_t)
    rho_pi   = oat_rho2_exact(N, np.pi)
    VQ_star, Fmax_star = vq_estimate(rho_star, 400)
    VQ_pi,   _         = vq_estimate(rho_pi,   200)
    ratio = VQ_star/max(VQ_pi, 1e-6) if VQ_pi > 0 else float('inf')
    gb = gamma_boundary(rho_star) if VQ_star > 0.005 else "N/A"
    results[N] = (chi_t, VQ_star, Fmax_star, VQ_pi, gb)
    ratio_str = f"{ratio:8.1f}" if ratio < 1e5 else "   inf"
    print(f"  {N:4d}  {chi_t:8.4f}  {VQ_star:13.4f}  "
          f"{VQ_pi:9.4f}  {ratio_str}  {str(gb):>10}")

# ── Table 3: V_Q vs Gamma at chi_t*(N) ───────────────────────────────────
print()
print("=== TABLE 3: V_Q vs Gamma_t at chi_t*(N) ===")
gammas = [0.0, GAMMA_1*1, GAMMA_1*3, GAMMA_1*10, 0.05, 0.10, 0.15, 0.20]
glabels = ['0', 'Γ₁', '3Γ₁', '10Γ₁', '0.05', '0.10', '0.15', '0.20']

# Header
hdr = f"  {'N/chi_t*':>12}"
for gl in glabels: hdr += f"  {gl:>7}"
print(hdr)
print("  " + "-"*12 + ("  -------")*len(gammas))

for N in [4, 6, 8, 12, 16]:
    chi_t = chi_t_stars[N]
    rho0  = oat_rho2_exact(N, chi_t)
    row   = f"  {N}/{chi_t:.3f}    "
    for Gamma_t in gammas:
        r = apply_dephasing(rho0, Gamma_t) if Gamma_t > 0 else rho0
        VQ, _ = vq_estimate(r, 150)
        sym = '█' if VQ > 0.01 else ('▒' if VQ > 0.001 else '░')
        row += f"  {sym}{VQ:.4f}"
    print(row)

# ── Table 4: D-LinOSS training features ──────────────────────────────────
print()
print("=== TABLE 4: D-LinOSS Training Feature Set ===")
print("Input: V_Q(chi_t, Gamma_t, N) | Label: {FLAT, CLASSICAL, QUANTUM}")
print(f"  {'N':>4}  {'chi_t':>8}  {'Gamma_t':>8}  {'V_Q':>8}  "
      f"{'F_max':>8}  {'Label':>10}")
print("-"*60)

for N in [4, 8, 16]:
    chi_t = chi_t_stars[N]
    for (Gamma_t, lbl_override) in [
        (0.0, None), (0.05, None), (0.15, None),
        (np.pi, None)  # chi_t=pi control (embed as Gamma_t=999 flag)
    ]:
        if Gamma_t == np.pi:
            rho = oat_rho2_exact(N, np.pi)
            g_display = "pi_ctrl"
            Gamma_display = np.pi
        else:
            rho = oat_rho2_exact(N, chi_t)
            if Gamma_t > 0: rho = apply_dephasing(rho, Gamma_t)
            g_display = f"{Gamma_t:.3f}"
            Gamma_display = Gamma_t
        VQ, Fmax = vq_estimate(rho, 200)
        if lbl_override:
            label = lbl_override
        elif VQ > 0.01:
            label = 'QUANTUM'
        elif Fmax < 0.51:
            label = 'FLAT'
        else:
            label = 'CLASSICAL'
        print(f"  {N:4d}  {chi_t if Gamma_t != np.pi else np.pi:8.4f}  "
              f"{g_display:>8}  {VQ:8.4f}  {Fmax:8.4f}  {label:>10}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("="*72)
print("SESSION 2 SUMMARY")
print("="*72)
print()
print("chi_t*(N) from concurrence (corrects Session 1 Panel D bug):")
for N, (chi_t, VQ_star, Fmax, VQ_pi, gb) in results.items():
    arrow = " ← detectable" if VQ_star > 0.01 else ""
    print(f"  N={N:2d}: chi_t*={chi_t:.4f}, V_Q={VQ_star:.4f}, "
          f"Gamma_t*={str(gb):>6}{arrow}")
print()
print("V_Q signal/control ratio (chi_t* vs chi_t=pi):")
for N, (chi_t, VQ_star, _, VQ_pi, _) in results.items():
    if VQ_pi == 0:
        ratio_str = "inf (V_Q(pi)=0, control perfect)"
    else:
        ratio_str = f"{VQ_star/VQ_pi:.1f}x"
    print(f"  N={N:2d}: {ratio_str}")
print()
print("D-LinOSS training set: V_Q(chi_t, Gamma_t, N) grid ready.")
print("Prediction: V_Q decay ~ exp(-4*Gamma*t) -- to be confirmed in Session 4.")
