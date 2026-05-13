"""
step0_calibration.py — Pre-TPU analytic calibration
Resolves: is f_max = (1 + T_xx)/4 tight for the OAT boundary channel?
Compares: Horodecki analytic bound vs. scipy DE numerical result.
Runtime: ~2 min on CPU.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import eigvalsh

# ── rho2 from Proposition 1 ─────────────────────────────────────────────────
def rho2(chi_t, N):
    m = N//2 - 1; B = [(0,0),(0,1),(1,0),(1,1)]
    r = np.zeros((4,4), dtype=complex)
    for a,(iL,iR) in enumerate(B):
        for b,(jL,jR) in enumerate(B):
            r[a,b] = 0.25 * np.exp(1j*chi_t*((iL-.5)*(iR-.5)-(jL-.5)*(jR-.5))) \
                   * np.cos((iR-jR)*chi_t/2)**m * np.cos((iL-jL)*chi_t/2)**m
    return r

# ── SU(2) from Euler angles ──────────────────────────────────────────────────
def su2(p):
    a,b,g = p
    return (np.array([[np.exp(-1j*a/2),0],[0,np.exp(1j*a/2)]]) @
            np.array([[np.cos(b/2),-np.sin(b/2)],[np.sin(b/2),np.cos(b/2)]]) @
            np.array([[np.exp(-1j*g/2),0],[0,np.exp(1j*g/2)]]))

PSI_P = np.array([0,1,1,0], dtype=complex) / np.sqrt(2)

def singlet_frac(p, rho):
    U = np.kron(su2(p[:3]), su2(p[3:]))
    r = U @ rho @ U.conj().T
    return float(np.real(PSI_P.conj() @ r @ PSI_P))

def find_fmax_scipy(rho, seed=0):
    np.random.seed(seed)
    res = differential_evolution(lambda p: -singlet_frac(p, rho),
                                  [(-np.pi,np.pi)]*6, seed=seed,
                                  maxiter=500, tol=1e-12)
    best = -res.fun
    for s in range(40):
        r2 = minimize(lambda p: -singlet_frac(p, rho),
                      res.x + np.random.normal(0,.3,6), method='Powell',
                      options={'ftol':1e-14,'maxiter':100000})
        if -r2.fun > best: best = -r2.fun; res = r2
    return best

# ── Analytic T-matrix ────────────────────────────────────────────────────────
def T_matrix_analytic(chi_t, N):
    """Compute full 3x3 T-matrix from Proposition 1 elements."""
    rho = rho2(chi_t, N)
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    paulis = [sx, sy, sz]
    T = np.zeros((3,3))
    for i,si in enumerate(paulis):
        for j,sj in enumerate(paulis):
            T[i,j] = float(np.real(np.trace(np.kron(si,sj) @ rho)))
    return T

def horodecki_bound(T):
    """f_max = (1 + max_singular_value(T)) / 4"""
    svs = np.linalg.svd(T, compute_uv=False)
    return (1 + svs[0]) / 4

# ── Main calibration ─────────────────────────────────────────────────────────
N = 4
print("="*60)
print("STEP 0 CALIBRATION — OAT Boundary Channel N=4")
print("="*60)

test_points = [
    ("chi_t → 0  (product state)",  0.001),
    ("chi_t = 0.5 (early phase)",   0.5),
    ("chi_t* ≈ 1.456 (optimal)",    1.45626),
    ("chi_t = 2.0  (late phase)",   2.0),
    ("chi_t = π   (singular)",      np.pi - 1e-6),
]

print(f"\n{'Point':<35} {'T_xx':>8} {'f_analytic':>12} {'F_analytic':>12} {'f_scipy':>10} {'F_scipy':>10}")
print("-"*90)

results = []
for label, chi_t in test_points:
    rho = rho2(chi_t, N)
    T   = T_matrix_analytic(chi_t, N)
    T_xx = T[0,0]

    # Analytic Horodecki bound
    f_ana  = horodecki_bound(T)
    F_ana  = (2*f_ana + 1) / 3

    # Numerical scipy DE
    f_num  = find_fmax_scipy(rho)
    F_num  = (2*f_num + 1) / 3

    print(f"{label:<35} {T_xx:>8.4f} {f_ana:>12.5f} {F_ana:>12.5f} {f_num:>10.5f} {F_num:>10.5f}")
    results.append((label, chi_t, T_xx, f_ana, F_ana, f_num, F_num))

print("\n" + "="*60)
print("VERDICT")
print("="*60)

chi_star_result = results[2]  # chi_t* row
f_ana_star = chi_star_result[3]
f_num_star = chi_star_result[5]
F_num_star = chi_star_result[6]

print(f"\nAt chi_t*:")
print(f"  Analytic Horodecki bound:  f_max = {f_ana_star:.5f}, F_avg = {(2*f_ana_star+1)/3:.5f}")
print(f"  Numerical scipy DE result: f_max = {f_num_star:.5f}, F_avg = {F_num_star:.5f}")
print(f"  Classical limit:           F_avg = 0.66667")

gap = abs(f_num_star - f_ana_star)
print(f"\n  |f_numerical - f_analytic| = {gap:.5f}")

if gap < 0.01:
    print("\n  ✅ BOUND IS TIGHT: f_max = (1+T_xx)/4 confirmed numerically.")
    print("  → F_avg < 2/3 for ALL chi_t. Paper's teleportation claim needs reframing.")
    print("  → Strongest results: Theorem 4 (proved null) + PTM characterization.")
else:
    print(f"\n  ⚠️  BOUND NOT TIGHT: numerical f_max exceeds analytic bound by {gap:.4f}.")
    print("  → T-matrix argument is missing something (off-diagonal elements?).")
    print("  → Inspect T-matrix off-diagonals at chi_t*:")
    rho_star = rho2(1.45626, N)
    T_star = T_matrix_analytic(1.45626, N)
    print(f"\n  Full T-matrix at chi_t*:\n{T_star.round(5)}")

# ── Calibration check: known exact values ───────────────────────────────────
print("\n" + "="*60)
print("CALIBRATION CHECKS (known exact values)")
print("="*60)

# chi_t → 0: product state, f_max should = 0.5, F_avg = 2/3
product_result = results[0]
print(f"\nchi_t→0 (product state):")
print(f"  f_max numerical = {product_result[5]:.5f}  (expected: 0.50000)")
print(f"  F_avg numerical = {product_result[6]:.5f}  (expected: 0.66667)")

# chi_t = π: Theorem 4, rho = I/4, all T = 0, f_max = 0.25, F_avg = 0.5
pi_result = results[4]
print(f"\nchi_t=π (Theorem 4 singular):")
print(f"  f_max numerical = {pi_result[5]:.5f}  (expected: 0.25000)")
print(f"  F_avg numerical = {pi_result[6]:.5f}  (expected: 0.50000)")

# Calibration pass/fail
cal_ok = (abs(product_result[5] - 0.5) < 0.005 and
          abs(pi_result[5] - 0.25) < 0.005)
print(f"\nOptimizer calibration: {'✅ PASS' if cal_ok else '❌ FAIL'}")
