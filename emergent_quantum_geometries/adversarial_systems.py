"""
adversarial_systems.py
======================
Stress-test the D-LinOSS fingerprint framework against systems designed to
expose whether K_eff/gamma_dom captures genuinely quantum dynamical structure
or merely generic spectral morphology.

Seven adversarial classes:
  1. Quasiperiodic Floquet      -- quasi-periodic but not quantum scrambling
  2. Damped classical oscillator -- single-mode decay + oscillation (classical)
  3. Random telegraph noise      -- non-smooth, non-quantum stochastic
  4. Integrable XXZ (W=0)        -- quantum but NOT thermal / not MBL
  5. Cavity-QED decay (Jaynes-Cummings + Lindblad)
  6. Chaotic spin chain (TFIM + disorder, not SYK)
  7. Classical Duffing oscillator -- chaotic but classical

If D-LinOSS genuinely separates quantum universality classes, these systems
should produce fingerprints DIFFERENT from their nearest quantum lookalike.
If D-LinOSS only reads spectral morphology, systems 2,3,7 will clone OAT/SYK/MBL.
"""

import numpy as np
from scipy.linalg import expm
import sys
import os

# ── Import D-LinOSS decomposer from training gen ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tpu_dlinoss_training_gen import dlinoss_decompose

RNG = np.random.default_rng(42)
N_TAU = 80
TAU_MAX = 200.0
TAU = np.linspace(0.01, TAU_MAX, N_TAU)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Quasiperiodic Floquet system
#    y(t) = sum_k A_k * cos(omega_k * t)  with omega_k = k + phi*(k^2)
#    (incommensurate frequencies -- never decays, never scrambles)
# ═════════════════════════════════════════════════════════════════════════════
def floquet_quasiperiodic(tau, n_modes=6, phi=0.6180339887):
    """Golden-ratio frequency spacing → quasiperiodic, no decay."""
    y = np.zeros(len(tau))
    for k in range(1, n_modes + 1):
        omega_k = k + phi * k**2
        A_k = 1.0 / k
        y += A_k * np.cos(omega_k * tau)
    return y / y[0]

# ═════════════════════════════════════════════════════════════════════════════
# 2. Damped classical oscillator
#    y(t) = e^{-gamma*t} * cos(omega*t)
#    Nearest OAT lookalike -- but classical.
# ═════════════════════════════════════════════════════════════════════════════
def damped_oscillator(tau, gamma=0.04, omega=0.5):
    return np.exp(-gamma * tau) * np.cos(omega * tau)

# ═════════════════════════════════════════════════════════════════════════════
# 3. Random telegraph noise (RTN)
#    Poisson-rate switching between +1/-1; averaged correlation function
#    C(t) = e^{-2*lambda*t}  (mimics Lindblad)
#    But the trajectory is non-smooth and non-exponential at short times.
# ═════════════════════════════════════════════════════════════════════════════
def random_telegraph_noise(tau, lam=0.05, n_real=500):
    """Disorder-averaged two-time correlation of RTN process."""
    dt = tau[1] - tau[0]
    n_steps = len(tau)
    corr = np.zeros(n_steps)
    for _ in range(n_real):
        # Simulate RTN: flip with rate lam per unit time
        state = RNG.choice([-1.0, 1.0])
        traj = np.zeros(n_steps)
        traj[0] = state
        for i in range(1, n_steps):
            if RNG.random() < lam * dt:
                state = -state
            traj[i] = state
        corr += traj[0] * traj  # two-time correlation
    corr /= n_real
    return corr

# ═════════════════════════════════════════════════════════════════════════════
# 4. Integrable XXZ (W=0, no disorder)
#    Quantum, but integrable -- should NOT look like MBL.
#    Imbalance I(t) for clean XXZ decays to zero (ergodic).
# ═════════════════════════════════════════════════════════════════════════════
def integrable_xxz(tau, L=8, J=1.0, Delta=0.5):
    """Staggered imbalance I(t) for clean (W=0) XXZ via exact diag."""
    dim = 2**L
    # Build XXZ Hamiltonian in full Hilbert space
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(L - 1):
        for op_pair in [(sx, sx), (sy, sy)]:
            o1, o2 = op_pair
            left = np.eye(2**i, dtype=complex)
            right = np.eye(2**(L - i - 2), dtype=complex)
            H += J * np.kron(np.kron(left, np.kron(o1, o2)), right)
        left = np.eye(2**i, dtype=complex)
        right = np.eye(2**(L - i - 2), dtype=complex)
        H += J * Delta * np.kron(np.kron(left, np.kron(sz, sz)), right)

    # Neel state initial condition
    neel = np.zeros(dim)
    neel[int('10' * (L // 2), 2)] = 1.0

    # Staggered magnetization operator
    Iz = np.zeros(dim, dtype=complex)
    for i in range(L):
        sign = (-1)**i
        eye_l = np.eye(2**i)
        eye_r = np.eye(2**(L - i - 1))
        sz_i = np.kron(np.kron(eye_l, sz), eye_r)
        Iz += sign * np.diag(sz_i)
    Iz_norm = neel @ Iz  # normalization

    # Time evolution via matrix exponentiation (small L only)
    vals, vecs = np.linalg.eigh(H)
    c0 = vecs.conj().T @ neel

    y = np.zeros(len(tau))
    for t_idx, t in enumerate(tau):
        ct = c0 * np.exp(-1j * vals * t)
        psi_t = vecs @ ct
        y[t_idx] = np.real(psi_t.conj() @ np.diag(Iz) @ psi_t) / Iz_norm
    return y

# ═════════════════════════════════════════════════════════════════════════════
# 5. Cavity-QED decay (Jaynes-Cummings + photon loss)
#    H = g*(a sigma+ + a† sigma-)  +  Lindblad: kappa * a decay
#    Nearest quantum-optical lookalike to OAT but different universality
# ═════════════════════════════════════════════════════════════════════════════
def cavity_qed_decay(tau, g=1.0, kappa=0.1, n_photon_max=8):
    """
    Jaynes-Cummings + cavity decay.
    Observable: sigma_z (atom inversion) starting from |e,0>.
    """
    dim = 2 * (n_photon_max + 1)  # |atom> x |photon>
    # Basis: (|g,0>, |g,1>,...,|g,n>, |e,0>,...,|e,n>)
    # Annihilation op on photon sector
    a_phot = np.diag(np.sqrt(np.arange(1, n_photon_max + 1)), 1)
    a_phot = a_phot[:n_photon_max + 1, :n_photon_max + 1]

    # Atom operators
    sp = np.array([[0, 1], [0, 0]])  # sigma+: |g>→|e>
    sm = sp.T
    sz_atom = np.array([[1, 0], [0, -1]]) * 0.5

    H = g * (np.kron(sp, a_phot) + np.kron(sm, a_phot.T))

    # Lindblad: L = sqrt(kappa) * I ⊗ a
    L_op = np.sqrt(kappa) * np.kron(np.eye(2), a_phot)
    L_dag_L = L_op.conj().T @ L_op

    # Vectorized Lindblad superoperator
    I_dim = np.eye(dim)
    L_super = np.kron(L_op.conj(), L_op) \
              - 0.5 * np.kron(I_dim, L_dag_L) \
              - 0.5 * np.kron(L_dag_L.T, I_dim) \
              - 1j * np.kron(I_dim, H) + 1j * np.kron(H.T, I_dim)

    # Initial state: |e, 0>
    psi0 = np.zeros(dim)
    psi0[n_photon_max + 1] = 1.0  # |e, 0>
    rho0 = np.outer(psi0, psi0).flatten()

    # Observable: sigma_z ⊗ I_photon
    sz_obs = np.kron(sz_atom, np.eye(n_photon_max + 1))

    vals, vecs = np.linalg.eig(L_super)
    c0 = np.linalg.solve(vecs, rho0)

    y = np.zeros(len(tau))
    for t_idx, t in enumerate(tau):
        rho_t = vecs @ (c0 * np.exp(vals * t))
        rho_mat = rho_t.reshape(dim, dim)
        y[t_idx] = np.real(np.trace(sz_obs @ rho_mat))
    return y

# ═════════════════════════════════════════════════════════════════════════════
# 6. Chaotic spin chain (TFIM + longitudinal field -- quantum chaos, not SYK)
#    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i - g sum_i Z_i
#    Goldenfeld et al. parameters: h=0.9045, g=0.809 → level statistics ETH
# ═════════════════════════════════════════════════════════════════════════════
def chaotic_spin_chain(tau, L=8, J=1.0, h=0.9045, g=0.809):
    """TFIM + longitudinal field: ETH/chaotic but NOT SYK. Spectral form factor proxy."""
    dim = 2**L
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(L - 1):
        left = np.eye(2**i, dtype=complex)
        right = np.eye(2**(L - i - 2), dtype=complex)
        H += -J * np.kron(np.kron(left, np.kron(sz, sz)), right)
    for i in range(L):
        left = np.eye(2**i, dtype=complex)
        right = np.eye(2**(L - i - 1), dtype=complex)
        H += -h * np.kron(np.kron(left, sx), right)
        H += -g * np.kron(np.kron(left, sz), right)

    # Spectral form factor proxy: K(tau) = |Tr[e^{-iHtau}]|^2 / dim
    vals = np.linalg.eigvalsh(H)
    y = np.zeros(len(tau))
    for t_idx, t in enumerate(tau):
        phases = np.exp(-1j * vals * t)
        y[t_idx] = np.abs(np.sum(phases))**2 / dim
    return y

# ═════════════════════════════════════════════════════════════════════════════
# 7. Classical Duffing oscillator (chaotic but classical)
#    x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
#    Use x(t) as the observable.
# ═════════════════════════════════════════════════════════════════════════════
def duffing_oscillator(tau, delta=0.2, alpha=-1.0, beta=1.0, gamma_d=0.3, omega_d=1.2):
    """Classical Duffing oscillator via RK4. Chaotic regime."""
    dt = tau[1] - tau[0]

    def rhs(t, state):
        x, v = state
        dxdt = v
        dvdt = -delta * v - alpha * x - beta * x**3 + gamma_d * np.cos(omega_d * t)
        return np.array([dxdt, dvdt])

    state = np.array([0.5, 0.0])
    y = np.zeros(len(tau))
    t = tau[0]
    for i, t_target in enumerate(tau):
        while t < t_target - 1e-10:
            k1 = rhs(t, state)
            k2 = rhs(t + dt / 2, state + dt / 2 * k1)
            k3 = rhs(t + dt / 2, state + dt / 2 * k2)
            k4 = rhs(t + dt, state + dt * k3)
            state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        y[i] = state[0]
    return y

# ═════════════════════════════════════════════════════════════════════════════
# Main: generate fingerprints for all adversarial systems
# ═════════════════════════════════════════════════════════════════════════════
def run_adversarial_suite():
    print("=" * 65)
    print("D-LinOSS ADVERSARIAL STRESS TEST")
    print("=" * 65)
    print(f"{'System':<30} {'K_eff':>5} {'gamma_dom':>10} {'omega_dom':>10} {'sigma1/2':>9}")
    print("-" * 65)

    systems = [
        ("Floquet Quasiperiodic",  floquet_quasiperiodic(TAU),         "EXPECTED: K>1, gamma~0 (no decay)"),
        ("Damped Oscillator (classical)", damped_oscillator(TAU),       "EXPECTED: K=1, gamma=0.04, omega=0.5"),
        ("Random Telegraph Noise", random_telegraph_noise(TAU),         "EXPECTED: K=1, gamma=0.10 (exp decay)"),
        ("Integrable XXZ (W=0)",   integrable_xxz(TAU, L=8),            "EXPECTED: K>1, gamma~0 (oscillatory)"),
        ("Cavity-QED Decay",       cavity_qed_decay(TAU),               "EXPECTED: K=1-2, gamma=kappa/2"),
        ("Chaotic Spin Chain",     chaotic_spin_chain(TAU, L=8),        "EXPECTED: K>1, complex spectrum"),
        ("Duffing Oscillator",     duffing_oscillator(TAU),             "EXPECTED: K>1, gamma~0 (chaotic classical)"),
    ]

    records = {}
    for name, y, expectation in systems:
        try:
            result = dlinoss_decompose(y, TAU)
            ke = result['K_eff']
            gk = result.get('gamma_k', [0])
            gdom = float(np.max(np.abs(gk))) if len(gk) else 0
            wk = result.get('omega_k', [0])
            wdom = float(np.abs(wk[np.argmax(np.abs(gk))])) if len(wk) else 0
            sk = result.get('sigma_k', [1, 0.01])
            sr = float(sk[0] / sk[1]) if len(sk) > 1 and sk[1] > 0 else float('inf')
            print(f"  {name:<28} {ke:>5} {gdom:>10.4f} {wdom:>10.4f} {sr:>9.2f}")
            print(f"    {expectation}")
            records[name] = {'K_eff': ke, 'gamma_dom': gdom, 'omega_dom': wdom, 'sigma_ratio': sr}
        except Exception as e:
            print(f"  {name:<28} ERROR: {e}")

    print("\n" + "=" * 65)
    print("KEY DIAGNOSTIC: Do classical/stochastic systems look like quantum ones?")
    print("=" * 65)

    # Compare RTN vs OAT
    oat_sig = {'K_eff': 1, 'gamma_dom': 0.04, 'omega_dom': 0.0}
    rtn = records.get("Random Telegraph Noise", {})
    syk_lookalike = records.get("Chaotic Spin Chain", {})

    print(f"\n  OAT (quantum Lindblad):       K=1, gamma=0.04, omega=0")
    print(f"  RTN (classical stochastic):   K={rtn.get('K_eff','?')}, gamma={rtn.get('gamma_dom', '?'):.4f}, omega={rtn.get('omega_dom','?'):.4f}")
    print(f"  ---> {'SAME FINGERPRINT (spectral morphology only!)' if rtn.get('K_eff')==1 else 'DIFFERENT (quantum structure detected)'}")

    print(f"\n  Chaotic spin chain (ETH):     K={syk_lookalike.get('K_eff','?')}, gamma={syk_lookalike.get('gamma_dom','?'):.4f}")
    print(f"  SYK4 (scrambling):            K=6, gamma~0.1-0.4")
    same = syk_lookalike.get('K_eff', 0) == 6
    print(f"  ---> {'CANNOT DISTINGUISH (K_eff saturates -- artifact)' if same else 'DISTINGUISHABLE'}")

    print()
    return records


if __name__ == '__main__':
    records = run_adversarial_suite()
    out = 'dlinoss_training/adversarial_results.npz'
    np.savez(out, **{k: v for k, v in records.items()})
    print(f"Saved to {out}")
