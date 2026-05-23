# _l_05_strata.py  —  Transport strata inference (pre-registered, fresh)
#
# NOTE: NOT borrowed from dlinoss_blind_transport.py — those thresholds were
# tuned for OAT/PTM data. These are derived fresh from transport observables
# and locked before first run.


def infer_transport_strata(x_obs):
    """
    Classify a 12-dim observable vector as TRANSPORT or not.
    ALL thresholds must be satisfied simultaneously (pre-registered).

    Returns: bool (True = TRANSPORT stratum)
    """
    ptm_sv1   = x_obs[OBS_NAMES.index("ptm_sv1")]
    ptm_H     = x_obs[OBS_NAMES.index("ptm_H")]
    EE        = x_obs[OBS_NAMES.index("EE")]
    MI        = x_obs[OBS_NAMES.index("MI")]
    basis_inv = x_obs[OBS_NAMES.index("basis_invar")]

    return (
        ptm_sv1   >= TH_SV1_MIN    and
        ptm_H     >= TH_SPEC_H_MIN and
        EE        >= TH_EE_MIN     and
        MI        >= TH_MI_MIN     and
        basis_inv <= TH_BI_MAX
    )


def strata_score(x_obs):
    """
    Soft score in [0,1] for how deep into the TRANSPORT stratum x_obs sits.
    Used by the agnostic controller for gradient-free pulse selection.
    """
    ptm_sv1   = x_obs[OBS_NAMES.index("ptm_sv1")]
    ptm_H     = x_obs[OBS_NAMES.index("ptm_H")]
    EE        = x_obs[OBS_NAMES.index("EE")]
    MI        = x_obs[OBS_NAMES.index("MI")]
    basis_inv = x_obs[OBS_NAMES.index("basis_invar")]

    # Margin to threshold, clipped and normalised
    s1 = np.clip((ptm_sv1   - TH_SV1_MIN)    / (1.0  - TH_SV1_MIN    + 1e-9), 0, 1)
    s2 = np.clip((ptm_H     - TH_SPEC_H_MIN) / (np.log(3) - TH_SPEC_H_MIN + 1e-9), 0, 1)
    s3 = np.clip((EE        - TH_EE_MIN)     / (2.0  - TH_EE_MIN     + 1e-9), 0, 1)
    s4 = np.clip((MI        - TH_MI_MIN)     / (2.0  - TH_MI_MIN     + 1e-9), 0, 1)
    s5 = np.clip((TH_BI_MAX - basis_inv)     / (TH_BI_MAX             + 1e-9), 0, 1)

    return float(np.mean([s1, s2, s3, s4, s5]))


# ── DIFFUSION MAP (borrowed from dlinoss_blind_transport.py, unchanged) ─────

def diffusion_map(X, n_components=3, alpha=0.5):
    """Diffusion map embedding of observable matrix X (n_pts x n_features)."""
    from scipy.spatial.distance import cdist
    from scipy.linalg import eigh
    D2  = cdist(X, X) ** 2
    eps = float(np.median(D2[D2 > 0])) if (D2 > 0).any() else 1.0
    K   = np.exp(-D2 / eps)
    d   = K.sum(axis=1)
    K   = K / np.outer(d ** alpha, d ** alpha)
    P   = K / K.sum(axis=1, keepdims=True)
    vals, vecs = eigh(P)
    idx  = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    emb = vecs[:, 1:n_components + 1] * (vals[1:n_components + 1] ** 2)
    return emb, vals[1:n_components + 1]
