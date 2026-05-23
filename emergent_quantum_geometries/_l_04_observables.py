# _l_04_observables.py  —  Observable computation, D-LinOSS latent state


def compute_observables(rho, N, rho_prev=None, dt=0.1, rng=None):
    """
    Compute the 12-dim observable vector X_t from current density matrix.
    Returns np.ndarray of shape (N_OBS,).
    """
    rho4 = partial_trace_boundary(rho, N)
    T    = compute_T_matrix(rho4)
    sv   = np.linalg.svd(T, compute_uv=False)

    ptm_sv1 = float(sv[0])
    ptm_sv2 = float(sv[1]) if len(sv) > 1 else 0.0
    ptm_sv3 = float(sv[2]) if len(sv) > 2 else 0.0
    ptm_H   = spectral_entropy_T(T)

    EE  = von_neumann(rho4)
    MI  = mutual_info(rho4)
    OS  = operator_spreading(rho, N)
    H_T = ptm_H  # transport entropy ≈ spectral entropy of PTM

    # Effective diffusion: rough estimate from EE growth slope
    if rho_prev is not None:
        rho4_prev = partial_trace_boundary(rho_prev, N)
        dEE = von_neumann(rho4) - von_neumann(rho4_prev)
        D_eff = float(dEE / (dt + 1e-12))
    else:
        D_eff = 0.0

    # Scrambling front velocity: change in OS
    front_v = 0.0
    if rho_prev is not None:
        OS_prev = operator_spreading(rho_prev, N)
        front_v = float((OS - OS_prev) / (dt + 1e-12))

    # Noise-response slope: estimated from dSV1/dt
    noise_slope = 0.0
    if rho_prev is not None:
        T_prev = compute_T_matrix(partial_trace_boundary(rho_prev, N))
        sv_prev = np.linalg.svd(T_prev, compute_uv=False)
        noise_slope = float((sv[0] - sv_prev[0]) / (dt + 1e-12))

    # Basis-invariant residual: apply random Clifford and measure PTM shift
    if rng is None:
        rng = np.random.default_rng(42)
    c_idx = int(rng.integers(1, N_CLIFF))  # skip identity
    rho_sc = apply_clifford(rho, N, c_idx, 0)
    T_sc   = compute_T_matrix(partial_trace_boundary(rho_sc, N))
    basis_invar = float(np.linalg.norm(T - T_sc, "fro"))

    return np.array([
        ptm_sv1, ptm_sv2, ptm_sv3, ptm_H,
        EE, MI, OS, H_T,
        D_eff, front_v, noise_slope, basis_invar,
    ])


# ── D-LinOSS MATRIX PENCIL DECOMPOSITION ───────────────────────────────────
# Borrowed from tpu_dlinoss_training_gen.py (Hua & Sarkar 1990).
# No modifications to the core math.

def dlinoss_decompose(y, t, K_max=8, noise_threshold=1e-3):
    """
    Matrix pencil method: extract (gamma_k, omega_k, A_k) from time series y(t).
    Returns dict with dominant mode parameters and BIC-selected rank.
    """
    M = len(y)
    if M < 4:
        return {"K_eff": 1, "gamma_k": np.array([0.0]),
                "omega_k": np.array([0.0]), "A_k": np.array([1.0]),
                "sigma_ratio": 1.0}

    L_p = max(2, M // 3)
    Y   = np.array([[y[i + j] for j in range(L_p + 1)] for i in range(M - L_p)])
    Y1  = Y[:, :-1]
    Y2  = Y[:, 1:]

    U1, s1, Vh1 = la.svd(Y1, full_matrices=False)
    s_norm = s1 / (s1[0] + 1e-30)
    K_eff  = max(1, int(np.sum(s_norm > noise_threshold)))
    K_eff  = min(K_eff, K_max, len(s1))

    # BIC rank selection
    bic_best = (1, np.inf)
    for k in range(1, K_eff + 1):
        Y1_k = U1[:, :k] @ np.diag(s1[:k]) @ Vh1[:k, :]
        rss  = np.sum((Y1 - Y1_k) ** 2)
        bic  = M * np.log(rss / (M * L_p) + 1e-30) + k * np.log(M)
        if bic < bic_best[1]:
            bic_best = (k, bic)
    K_bic = bic_best[0]

    U1_k  = U1[:, :K_eff]
    s1_k  = s1[:K_eff]
    Vh1_k = Vh1[:K_eff, :]
    Z_mat = np.diag(1.0 / (s1_k + 1e-30)) @ (U1_k.conj().T @ Y2) @ Vh1_k.conj().T
    poles, _ = la.eig(Z_mat)

    dt_val  = float(t[1] - t[0]) if len(t) > 1 else 1.0
    gamma_k = -np.log(np.abs(poles) + 1e-30) / dt_val
    omega_k =  np.angle(poles) / dt_val

    Z_cols  = np.array([poles ** n for n in range(M)])
    A_c, _, _, _ = np.linalg.lstsq(Z_cols, y.astype(complex), rcond=None)
    A_k = np.abs(A_c)

    sigma_ratio = float(s1[0] / s1[1]) if len(s1) > 1 and s1[1] > 1e-15 else 1e6

    return {
        "K_eff": K_eff, "K_bic": K_bic,
        "gamma_k": gamma_k.real, "omega_k": omega_k.real,
        "A_k": A_k, "sigma_ratio": sigma_ratio,
    }


def compute_latent_state(obs_history, t_history):
    """
    Compute 36-dim latent state z_t from observable time-series history.
    For each of N_OBS channels: extract (gamma_dom, omega_dom, sigma_ratio).
    """
    obs_history = np.array(obs_history)   # shape (T, N_OBS)
    t_history   = np.array(t_history)
    z = np.zeros(N_OBS * 3)
    for i in range(N_OBS):
        y = obs_history[:, i]
        if len(y) < 4 or np.std(y) < 1e-12:
            z[i*3], z[i*3+1], z[i*3+2] = 0.0, 0.0, 1.0
            continue
        res = dlinoss_decompose(y, t_history)
        gk = res["gamma_k"]
        ok = res["omega_k"]
        dom = int(np.argmax(res["A_k"])) if len(res["A_k"]) > 0 else 0
        z[i*3]     = float(gk[dom]) if dom < len(gk) else 0.0
        z[i*3+1]   = float(ok[dom]) if dom < len(ok) else 0.0
        z[i*3+2]   = float(res["sigma_ratio"])
    return z
