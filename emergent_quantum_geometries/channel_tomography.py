"""
channel_tomography.py — Teleportation Channel Reconstruction Experiment
======================================================================
Implements the reviewer's critical request: transition from state-property
analysis to OPERATIONAL channel characterization.

Protocol:
  1. Generate rho_2(N, chi_t, Gamma_loc) using exact formula
  2. Apply constrained pre-rotation (Rz only — JILA-realizable)
  3. Run explicit Bell measurement: CNOT + H + Z-basis readout
  4. Apply Pauli feedforward corrections
  5. Reconstruct Pauli Transfer Matrix from 6 cardinal Bloch states
  6. Extract: F_avg, F_e, anisotropy, diamond distance bound

Key distinction: uses only Rz gates (NOT arbitrary SU(2)xSU(2))
This is the experimentally defensible claim.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar, minimize

# ── Load exact formula ────────────────────────────────────────────────────────
src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}
exec(compile(src, 'tpu_dlinoss_training_gen', 'exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

# ── Pauli matrices ────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def Rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)


# ── Dephasing on rho_2 ───────────────────────────────────────────────────────
def apply_dephasing(rho, Gamma_t):
    """Exact Lindblad dephasing: rho[i,j] *= exp(-2*Gamma*dH(i,j)*t)"""
    spin = np.array([1,0,0,-1])  # M_total for |uu>,|ud>,|du>,|dd>
    r = rho.copy()
    for i in range(4):
        for j in range(4):
            dH = abs(spin[i] != spin[j]) + abs(spin[i]//2 != spin[j]//2)
            # Hamming distance
            si = format(i, '02b'); sj = format(j, '02b')
            dH = sum(a != b for a, b in zip(si, sj))
            r[i,j] *= np.exp(-2 * Gamma_t * dH)
    return r


# ── Core teleportation protocol ───────────────────────────────────────────────

def teleport_channel_ptm(rho_AB, UA=None, UB=None):
    """
    Reconstruct the Pauli Transfer Matrix of the teleportation channel.

    Protocol: standard Bennett et al. with Pauli feedforward.
    Bell measurement = H(a) + CNOT(a->A) + Z-basis (a,A)
    Feedforward: (0,0)->I, (0,1)->Z, (1,0)->X, (1,1)->XZ

    Args:
        rho_AB: 4x4 resource density matrix (qubits A, B)
        UA, UB: optional 2x2 pre-rotations on resource pair

    Returns:
        T: 4x4 Pauli Transfer Matrix (real)
        F_avg: average teleportation fidelity
        F_e: entanglement fidelity
    """
    # Apply pre-rotations to resource
    if UA is not None or UB is not None:
        Ua = UA if UA is not None else I2
        Ub = UB if UB is not None else I2
        U = np.kron(Ua, Ub)
        rho_AB = U @ rho_AB @ U.conj().T

    # Bell states (a, A ordering)
    B = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]],
                 dtype=complex) / np.sqrt(2)
    Pi = [np.outer(B[m], B[m].conj()) for m in range(4)]

    # Feedforward corrections (Bob)
    U_fb = [I2, Z, X, X @ Z]

    # PTM: T[i,j] = Tr[sigma_i * E(sigma_j/2)] * 2
    T = np.zeros((4, 4))

    for j in range(4):
        rho_in = PAULIS[j] / 2  # traceless for j>0; identity/2 for j=0

        # 3-qubit state: a(input) x A(resource) x B(resource)
        rho_tot = np.kron(rho_in, rho_AB)  # 8x8

        rho_out = np.zeros((2, 2), dtype=complex)

        for m in range(4):
            # Projector on Bell state m acting on first 2 qubits (a, A)
            Pi_full = np.kron(Pi[m], I2)  # 8x8

            # Post-measurement state (unnormalized)
            post = Pi_full @ rho_tot @ Pi_full.conj().T

            # Trace out qubits a and A (first two), keep B
            post_t = post.reshape(2, 2, 2, 2, 2, 2)
            # rho_B[i2,j2] = sum_{i0,i1} post_t[i0,i1,i2, i0,i1,j2]
            rho_B = np.einsum('ijaijb->ab', post_t)

            # Apply feedforward correction
            rho_out += U_fb[m] @ rho_B @ U_fb[m].conj().T

        # PTM column j
        for i in range(4):
            T[i, j] = 2 * np.real(np.trace(PAULIS[i] @ rho_out))

    # T[0,0] = 1 by trace preservation
    F_avg = (np.trace(T[1:, 1:]) / 3 + 1) / 2   # = (Tr(T_3x3)/3 + 1)/2
    # Standard qubit formula: F_avg = (d*F_e + 1)/(d+1) with d=2
    # => F_e = (3*F_avg - 1)/2
    F_e = (3 * F_avg - 1) / 2

    return T, float(np.real(F_avg)), float(np.real(F_e))


def optimize_rz_only(rho_AB, n_angles=36):
    """
    Optimize F_avg over Rz(theta_A) x Rz(theta_B) — JILA-realizable.
    These are virtual Z rotations: software-only, no extra pulses.
    """
    best = {'F_avg': 0, 'F_e': 0, 'theta_A': 0, 'theta_B': 0, 'T': None}
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    for tA in angles:
        for tB in angles:
            _, F_avg, F_e = teleport_channel_ptm(rho_AB, Rz(tA), Rz(tB))
            if F_avg > best['F_avg']:
                best.update({'F_avg': F_avg, 'F_e': F_e,
                             'theta_A': tA, 'theta_B': tB})

    # Refine with scipy
    def neg_favg(params):
        _, f, _ = teleport_channel_ptm(rho_AB, Rz(params[0]), Rz(params[1]))
        return -f

    res = minimize(neg_favg, [best['theta_A'], best['theta_B']],
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000})
    if -res.fun > best['F_avg']:
        best['theta_A'], best['theta_B'] = res.x
        best['F_avg'] = -res.fun
        T, _, Fe = teleport_channel_ptm(rho_AB, Rz(res.x[0]), Rz(res.x[1]))
        best['F_e'] = Fe
        best['T'] = T

    if best['T'] is None:
        T, _, Fe = teleport_channel_ptm(rho_AB, Rz(best['theta_A']), Rz(best['theta_B']))
        best['T'] = T

    return best


def ptm_anisotropy(T):
    """Measure how far the PTM deviates from isotropic (depolarizing) channel."""
    Tv = np.diag(T)[1:]       # [T_xx, T_yy, T_zz]
    iso = np.mean(Tv)          # = (2*F_avg - 1) for isotropic
    return float(np.std(Tv))  # 0 = isotropic, large = anisotropic


def diamond_distance_lower_bound(T):
    """
    Lower bound on diamond distance to classical channel (F_avg=2/3, T_iso=0).
    Classical PTM: diag(1, 0, 0, 0).
    Diamond distance: max over input state of trace-norm of (E - E_class)(rho).
    Lower bound = max_i |T_ii - T_class_ii| / 2.
    """
    T_class = np.diag([1, 0, 0, 0])
    return float(np.max(np.abs(np.diag(T) - np.diag(T_class))) / 2)


# ── Concurrence (standalone) ─────────────────────────────────────────────────
def concurrence(rho):
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy, sy)
    R = la.sqrtm(rho @ (ss @ rho.conj() @ ss))
    e = sorted(np.sqrt(np.maximum(np.real(la.eigvals(R)), 0)), reverse=True)
    return max(0., e[0]-e[1]-e[2]-e[3])


# ── Main experiment ──────────────────────────────────────────────────────────

def run_channel_tomography():
    print("=" * 72)
    print("TELEPORTATION CHANNEL RECONSTRUCTION EXPERIMENT")
    print("Constraint: Rz(theta_A) x Rz(theta_B) only (JILA-realizable)")
    print("=" * 72)
    print()

    grid = np.linspace(0.02, 6.28, 300)

    # ── Sweep 1: N dependence (noiseless) ────────────────────────────────────
    print("--- Sweep 1: N dependence (Gamma=0) ---")
    print(f"{'N':>4}  {'chi_t*':>7}  {'C':>6}  {'F_avg(Rz)':>10}  "
          f"{'F_e(Rz)':>8}  {'tA(deg)':>8}  {'aniso':>6}  "
          f"{'d_diam':>7}  {'classical?':>10}")
    print("-" * 72)

    results = []
    for N in [2, 4, 6, 8, 12, 16, 24, 32]:
        Cs = [concurrence(oat_rho2_exact(N, t)) for t in grid]
        idx = np.argmax(Cs)
        chi_t = grid[idx]
        rho = oat_rho2_exact(N, chi_t)
        C = Cs[idx]

        best = optimize_rz_only(rho, n_angles=48)
        T = best['T']
        aniso = ptm_anisotropy(T)
        d_diam = diamond_distance_lower_bound(T)
        F_avg = best['F_avg']
        F_e = best['F_e']
        classical = "YES" if F_avg <= 2/3 else "NO"

        print(f"  {N:2d}  {chi_t:7.4f}  {C:6.4f}  {F_avg:10.6f}  "
              f"{F_e:8.6f}  {np.degrees(best['theta_A']):8.2f}  "
              f"{aniso:6.4f}  {d_diam:7.4f}  {classical:>10}")
        results.append((N, chi_t, C, F_avg, F_e, aniso, d_diam))

    print()

    # ── Sweep 2: Dephasing robustness (N=8, 16) ──────────────────────────────
    print("--- Sweep 2: Dephasing robustness ---")
    print(f"{'N':>4}  {'Gamma*t':>8}  {'F_avg(Rz)':>10}  {'F_e':>8}  "
          f"{'classical?':>10}  {'note':>20}")
    print("-" * 55)

    for N in [8, 16]:
        # Use chi_t* from noiseless case
        Cs = [concurrence(oat_rho2_exact(N, t)) for t in grid]
        chi_t = grid[np.argmax(Cs)]

        for Gamma_t in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            rho = apply_dephasing(oat_rho2_exact(N, chi_t), Gamma_t)
            best = optimize_rz_only(rho, n_angles=36)
            F_avg = best['F_avg']
            F_e = best['F_e']
            classical = "YES" if F_avg <= 2/3 else "NO"
            note = "quantum adv." if F_avg > 2/3 else "LOST"
            print(f"  {N:2d}  {Gamma_t:8.3f}  {F_avg:10.6f}  {F_e:8.6f}  "
                  f"{classical:>10}  {note:>20}")
        print()

    # ── PTM for N=4, noiseless ────────────────────────────────────────────────
    print("--- PTM for N=4 (noiseless, optimal Rz rotation) ---")
    N = 4
    Cs = [concurrence(oat_rho2_exact(N, t)) for t in grid]
    chi_t = grid[np.argmax(Cs)]
    rho = oat_rho2_exact(N, chi_t)
    best = optimize_rz_only(rho, n_angles=72)
    T = best['T']

    print(f"Resource: N={N}, chi_t*={chi_t:.4f}")
    print(f"Optimal: theta_A={np.degrees(best['theta_A']):.2f}° "
          f"theta_B={np.degrees(best['theta_B']):.2f}°")
    print(f"F_avg = {best['F_avg']:.6f}  (classical limit: 0.667)")
    print(f"F_e   = {best['F_e']:.6f}  (classical limit: 0.500)")
    print()
    print("Pauli Transfer Matrix T[i,j] = Tr[sigma_i E(sigma_j)]:")
    labels = ['I', 'X', 'Y', 'Z']
    print(f"      {'  '.join(f'{l:>8}' for l in labels)}")
    for i, row in enumerate(T):
        print(f"  {labels[i]}  {'  '.join(f'{v:+8.4f}' for v in row)}")
    print()
    print(f"PTM diagonal (T_xx, T_yy, T_zz): "
          f"{T[1,1]:.4f}, {T[2,2]:.4f}, {T[3,3]:.4f}")
    print(f"Anisotropy: {ptm_anisotropy(T):.4f} "
          f"(0=isotropic depolarizing, large=phase bias)")
    print(f"Diamond distance lower bound from classical: {diamond_distance_lower_bound(T):.4f}")
    print()
    print("CONCLUSION:")
    if best['F_avg'] > 2/3:
        print(f"  F_avg = {best['F_avg']:.4f} > 2/3 = 0.6667 [QUANTUM ADVANTAGE CONFIRMED]")
        print(f"  Under Rz-only (JILA-realizable) gates.")
        print(f"  Claim: OAT boundary state is an operational teleportation resource")
        print(f"  under experimentally realizable controls.")
    else:
        print(f"  F_avg = {best['F_avg']:.4f} <= 2/3 [quantum advantage NOT demonstrated]")


if __name__ == '__main__':
    run_channel_tomography()
