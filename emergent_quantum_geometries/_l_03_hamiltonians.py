# _l_03_hamiltonians.py  —  Model builders, time evolution, noise channels


def build_xxz(N, W=0.0, J=1.0, Delta=1.0, seed=0):
    """XXZ + disorder: H = J Σ(XX+YY)/2 + Delta·ZZ + W·hi·Zi."""
    rng = np.random.default_rng(seed)
    disorder = rng.uniform(-W, W, N)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        for i in range(N):
            bit = (s >> i) & 1
            H[s, s] += disorder[i] * (1 - 2 * bit)
        for i in range(N - 1):
            bi  = (s >> i) & 1
            bi1 = (s >> (i + 1)) & 1
            H[s, s] += J * Delta * (1 - 2 * bi) * (1 - 2 * bi1)
            if bi != bi1:
                s2 = s ^ (1 << i) ^ (1 << (i + 1))
                H[s, s2] += J
    return H


def build_ising(N, J=1.0, hx=1.0, hz=0.0):
    """Transverse Ising: H = -J·ZZ - hx·X - hz·Z."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        for i in range(N - 1):
            bi  = (s >> i) & 1
            bi1 = (s >> (i + 1)) & 1
            H[s, s] -= J * (1 - 2 * bi) * (1 - 2 * bi1)
        for i in range(N):
            bi = (s >> i) & 1
            H[s, s] -= hz * (1 - 2 * bi)
            s2 = s ^ (1 << i)
            H[s, s2] -= hx
    return H


def build_oat(N, chi=None):
    """One-axis twisting: H = chi·Jz^2."""
    if chi is None:
        chi = 1.0 / N
    Jz = sum(kron_site(_sz / 2, i, N) for i in range(N))
    return float(chi) * (Jz @ Jz)


MODEL_BUILDERS = {
    "XXZ":              lambda N, seed: build_xxz(N, W=0.0, seed=seed),
    "DisorderedXXZ_W1": lambda N, seed: build_xxz(N, W=1.0, seed=seed),
    "DisorderedXXZ_W3": lambda N, seed: build_xxz(N, W=3.0, seed=seed),
    "OAT":              lambda N, seed: build_oat(N),
    "Ising":            lambda N, seed: build_ising(N, hx=1.0, hz=0.0),
    "TiltedIsing":      lambda N, seed: build_ising(N, hx=0.9045, hz=0.809),
}


def evolve_rho(rho, U):
    """rho -> U rho U†."""
    return U @ rho @ U.conj().T


def make_U(H, dt):
    """Compute time-evolution operator expm(-i H dt)."""
    return la.expm(-1j * H * dt)


def apply_noise(rho, N, params, dt):
    """Apply Lindblad noise (T1, T2 dephasing) per site."""
    g1 = params.get("T1_rate", 0.0) * dt   # amplitude damping rate
    g2 = params.get("T2_rate", 0.0) * dt   # dephasing rate
    sz_ops = [kron_site(_sz, s, N) for s in range(N)]
    raise_ops = [kron_site(np.array([[0, 1], [0, 0]], dtype=complex), s, N)
                 for s in range(N)]

    for site in range(N):
        # T1 amplitude damping
        if g1 > 0:
            p = min(g1, 1.0)
            K0 = np.eye(2 ** N, dtype=complex)
            K0 += (np.sqrt(1 - p) - 1) * raise_ops[site].conj().T @ raise_ops[site]
            K1 = np.sqrt(p) * raise_ops[site]
            rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
        # T2 dephasing
        if g2 > 0:
            p = min(g2 / 2, 0.5)
            K0 = np.sqrt(1 - p) * np.eye(2 ** N, dtype=complex)
            K1 = np.sqrt(p) * sz_ops[site]
            rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    # Coherent drift (rotation on site 0)
    drift_amp  = params.get("drift_amp", 0.0)
    drift_freq = params.get("drift_freq", 0.1)
    t_cur      = params.get("_t_cur", 0.0)
    if drift_amp > 0:
        angle = drift_amp * np.sin(drift_freq * t_cur) * dt
        Ud = la.expm(-1j * angle * sz_ops[0])
        rho = Ud @ rho @ Ud.conj().T

    return rho


def apply_clifford(rho, N, cliff_idx, site):
    """Apply Clifford gate CLIFFORDS[cliff_idx] to qubit `site`."""
    U_full = kron_site(CLIFFORDS[cliff_idx], site, N)
    return U_full @ rho @ U_full.conj().T
