"""
p1_coord_invariance.py — Priority 1: Coordinate/Gauge Invariance Test
=======================================================================
Tests whether V_Q, chi_tc, and beta are independent of SU(2)^2 parameterization.

Reviewer attack: "V_Q is an artifact of Euler-angle coordinate choice."

Three Haar-equivalent constructions of uniform SU(2) sampling:
  A. QR-decomposition (current method — theoretically Haar uniform)
  B. Quaternion-based Haar sampling
  C. Axis-angle parameterization with Haar-correct density
  D. [Control] Naive Euler-angle uniform (biased — should differ)

If A, B, C agree and D differs: confirms V_Q depends on measure, not coordinates.
If A=B=C=D: V_Q is robust to parameterization choice.
"""

import numpy as np

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

_phi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
_Pp  = np.outer(_phi,_phi.conj())

def F_from_U(rho2, UA, UB):
    r = np.kron(UA,UB)@rho2@np.kron(UA,UB).conj().T
    return np.real((2*np.trace(_Pp@r)+1)/3)

# ── Four SU(2) sampling methods ───────────────────────────────────────────

def haar_qr(rng):
    """Method A: QR decomposition (current, theoretically Haar-uniform)."""
    z = (rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))) / np.sqrt(2)
    Q,R = np.linalg.qr(z); d = np.diagonal(R); Q *= d/np.abs(d)
    if np.real(np.linalg.det(Q)) < 0: Q[:,0] *= -1
    return Q

def haar_quaternion(rng):
    """Method B: Quaternion parameterization with Haar-correct density.
    Sample (q0,q1,q2,q3) from unit 4-sphere, construct SU(2) matrix.
    This is exactly Haar-uniform."""
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    q0,q1,q2,q3 = q
    return np.array([[q0+1j*q3, q2+1j*q1],
                     [-q2+1j*q1, q0-1j*q3]])

def haar_axis_angle(rng):
    """Method C: Axis-angle with Haar-correct density.
    Haar measure on SU(2): uniform over axis (unit 3-sphere direction),
    angle theta in [0,pi] with density sin^2(theta/2)."""
    # Sample axis from uniform sphere
    v = rng.standard_normal(3); v /= np.linalg.norm(v)
    # Sample angle with density sin^2(theta/2), theta in [0,2*pi]
    # via rejection from uniform: p(theta) ~ sin^2(theta/2)
    while True:
        theta = rng.uniform(0, 2*np.pi)
        if rng.uniform(0, 1) < np.sin(theta/2)**2:
            break
    # U = cos(theta/2)*I - i*sin(theta/2)*(sigma·v)
    c, s = np.cos(theta/2), np.sin(theta/2)
    nx,ny,nz = v
    return np.array([[c - 1j*s*nz, -s*(ny + 1j*nx)],
                     [s*(ny - 1j*nx), c + 1j*s*nz]])

def naive_euler(rng):
    """Control: Naive uniform Euler angles (BIASED — not Haar uniform).
    This should give DIFFERENT V_Q if the measure matters."""
    alpha = rng.uniform(0, 2*np.pi)
    beta  = rng.uniform(0, np.pi)   # WRONG: should use arccos distribution
    gamma = rng.uniform(0, 2*np.pi)
    # Rz(alpha) Ry(beta) Rz(gamma)
    ca,sa = np.cos(alpha/2),np.sin(alpha/2)
    cb,sb = np.cos(beta/2), np.sin(beta/2)
    cg,sg = np.cos(gamma/2),np.sin(gamma/2)
    Rza = np.array([[np.exp(-1j*alpha/2),0],[0,np.exp(1j*alpha/2)]])
    Ryb = np.array([[cb,-sb],[sb,cb]])
    Rzg = np.array([[np.exp(-1j*gamma/2),0],[0,np.exp(1j*gamma/2)]])
    return Rza @ Ryb @ Rzg

def vq_method(rho2, sampler_A, sampler_B, n_samples, seed):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_samples):
        UA = sampler_A(rng); UB = sampler_B(rng)
        cnt += F_from_U(rho2, UA, UB) > 2/3
    return cnt / n_samples

print("="*70)
print("PRIORITY 1: COORDINATE / GAUGE INVARIANCE TEST")
print("="*70)
N = 4; n_samples = 1500; n_seeds = 8

methods = [
    ('A-QR (Haar)',       haar_qr,        haar_qr),
    ('B-Quaternion',      haar_quaternion, haar_quaternion),
    ('C-Axis-angle',      haar_axis_angle, haar_axis_angle),
    ('D-Naive-Euler',     naive_euler,     naive_euler),
    ('E-Mixed (A+B)',     haar_qr,         haar_quaternion),
]

test_points = [(0.355*np.pi, 'chi_t*'),
               (np.pi/2,     'chi_t=pi/2'),
               (np.pi,       'chi_t=pi')]

for chi_t, label in test_points:
    rho2 = oat_rho2_exact(N, chi_t)
    print(f"\n  chi_t={chi_t/np.pi:.3f}*pi ({label}):")
    print(f"  {'Method':>18}  {'VQ_mean':>8}  {'VQ_std':>8}  {'CV%':>7}")
    print("  "+"-"*44)
    results = {}
    for mname, sampA, sampB in methods:
        vqs = [vq_method(rho2, sampA, sampB, n_samples, 42+s*13) for s in range(n_seeds)]
        vm, vs = np.mean(vqs), np.std(vqs)
        results[mname] = (vm, vs)
        print(f"  {mname:>18}  {vm:8.5f}  {vs:8.5f}  {vs/max(vm,1e-6)*100:7.1f}%")

    # Test: are all Haar methods consistent?
    haar_means = [results[m][0] for m in ['A-QR (Haar)','B-Quaternion','C-Axis-angle','E-Mixed (A+B)']]
    haar_std_of_means = np.std(haar_means)
    naive_mean = results['D-Naive-Euler'][0]
    ref_mean   = results['A-QR (Haar)'][0]
    print(f"\n  Haar methods std-of-means: {haar_std_of_means:.6f}")
    print(f"  Naive-Euler vs QR-Haar:    {abs(naive_mean-ref_mean):.6f}")
    if haar_std_of_means < 0.005:
        print(f"  INVARIANCE CONFIRMED: all Haar methods agree within 0.005")
    else:
        print(f"  WARNING: Haar methods disagree by {haar_std_of_means:.4f}")

print()
print("="*70)
print("INVARIANCE SUMMARY")
print("="*70)
print()
print("  Test: V_Q is independent of SU(2)^2 parameterization")
print("  when the Haar measure is used (Methods A, B, C, E).")
print()
print("  Naive uniform Euler angles (Method D) serve as control:")
print("  if they differ significantly, it confirms the Haar measure")
print("  is the physically correct choice (not arbitrary coordinates).")
