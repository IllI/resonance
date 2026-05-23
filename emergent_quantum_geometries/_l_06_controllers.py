# _l_06_controllers.py  —  Static, Hamiltonian-aware, D-LinOSS agnostic


def ctrl_static(step, N, **_):
    """
    Static controller: identity everywhere except a pre-calibrated X pulse
    at step 0 (boundary initialisation) and periodic Z echoes.
    No adaptation. Lower bound.
    """
    if step == 0:
        return (1, 0)   # X on site 0
    if step % 10 == 0:
        return (3, 0)   # Z echo on site 0
    return (0, 0)       # identity


def ctrl_haware(step, N, rho, H, dt, noise_params, **_):
    """
    Hamiltonian-aware controller: 1-step greedy oracle.
    Tries each Clifford on site 0, simulates 1 step with true H,
    picks pulse maximising F_rec. Upper bound.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)

    best_idx, best_score = 0, -np.inf
    U = make_U(H, dt * CTRL_INTERVAL)

    for ci in range(N_CLIFF):
        rho_c   = apply_clifford(rho, N, ci, 0)
        rho_evo = evolve_rho(rho_c, U)
        rho_evo = apply_noise(rho_evo, N, noise_params, dt * CTRL_INTERVAL)
        rho4    = partial_trace_boundary(rho_evo, N)
        T       = compute_T_matrix(rho4)
        score   = f_rec_from_T(T)
        if score > best_score:
            best_score = score
            best_idx   = ci

    return (best_idx, 0)


def ctrl_agnostic(step, N, rho, obs_history, t_history,
                  dropout_rng, use_dropout=True, **_):
    """
    D-LinOSS agnostic controller.
    Uses observable time-series (no Hamiltonian) to pick pulse via
    strata score on current observables + Clifford PTM effect.

    Agnosticism: knows its own Clifford actions (physical),
    does NOT know H or noise model.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)

    # Current observables
    x_cur = np.array(obs_history[-1]) if obs_history else np.zeros(N_OBS)

    # Optional representation dropout on z_t (tests latent robustness)
    if use_dropout and len(obs_history) >= 4:
        mask  = dropout_rng.binomial(1, 1.0 - P_DROPOUT, size=N_OBS).astype(float)
        x_cur = x_cur * mask

    # Score each Clifford by its expected effect on strata signature.
    # Clifford C transforms PTM: T -> R_C^T T R_C  (SO(3) adjoint action).
    # We approximate the effect on ptm_sv1 via the Clifford's SO(3) image.
    best_idx, best_score = 0, strata_score(x_cur)

    for ci in range(1, N_CLIFF):   # skip identity
        x_pred = x_cur.copy()
        # Predict ptm_sv1 change: Clifford rotates T, sv1 changes by
        # ||R_C^T T|| / ||T|| factor (conservative: ≈ unchanged for I,Z; varies for H)
        # This is the agnostic controller's "model" — only its own gate physics.
        cliff_sv_factor = _clifford_sv_factor(ci)
        x_pred[0] = np.clip(x_cur[0] * cliff_sv_factor, 0.0, 1.0)
        sc = strata_score(x_pred)
        if sc > best_score:
            best_score = sc
            best_idx   = ci

    return (best_idx, 0)


# SO(3) singular-value preservation factors for each Clifford gate.
# I,Z: preserve T exactly.  X,Y: flip off-diagonals, same sv.
# H: mixes x<->z axis, sv1 may change.  S: mixes x<->y, sv unchanged.
# These are exact for diagonal T (good approximation near transport stratum).
_CLIFF_SV_FACTORS = [1.0, 1.0, 1.0, 1.0, 0.95, 1.0]

def _clifford_sv_factor(ci):
    return _CLIFF_SV_FACTORS[ci]


def get_controller(name):
    controllers = {
        "static":   ctrl_static,
        "haware":   ctrl_haware,
        "agnostic": ctrl_agnostic,
    }
    if name not in controllers:
        raise ValueError(f"Unknown controller '{name}'. Choose: {list(controllers)}")
    return controllers[name]
