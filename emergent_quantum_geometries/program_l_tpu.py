#!/usr/bin/env python3
"""
program_l_tpu.py  —  Program L: Representation-Agnostic Adaptive Recovery
=========================================================================
Tests whether a D-LinOSS controller using only transport observables can
stabilize recoverable quantum transport under noise without knowing the
Hamiltonian or coordinate system.

Three controllers compared on identical noise trajectories:
  Static   — fixed pre-calibrated pulses (lower bound)
  HAware   — full H + noise model access   (upper bound)
  Agnostic — observable-only D-LinOSS      (test)

Pre-registered thresholds FROZEN before first run.
"""
import argparse
import json
import os
import time
from functools import partial
import numpy as np
import scipy.linalg as la

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:  # Local smoke tests should still run without JAX installed.
    jax = None
    jnp = None
    HAS_JAX = False

# ── PRE-REGISTERED THRESHOLDS (FROZEN) ─────────────────────────────────────
TH_F_REC      = 2 / 3   # classical communication limit (Programs H–K)
TH_SV1_MIN    = 0.30    # PTM leading singular value floor
TH_SPEC_H_MIN = 0.40    # spectral entropy floor
TH_EE_MIN     = 0.10    # entanglement entropy floor
TH_MI_MIN     = 0.03    # mutual information floor (Program K pre-reg)
TH_BI_MAX     = 0.25    # basis-invariant residual ceiling
P_DROPOUT     = 0.30    # representation dropout probability
CTRL_INTERVAL = 5       # timesteps between control decisions

# ── OBSERVABLE VECTOR (12-dim, locked) ─────────────────────────────────────
OBS_NAMES = [
    "ptm_sv1", "ptm_sv2", "ptm_sv3", "ptm_H",
    "EE", "MI", "OS", "H_T",
    "D_eff", "front_v", "noise_slope", "basis_invar",
]
N_OBS = len(OBS_NAMES)  # 12

# ── CLIFFORD LIBRARY (6 key single-qubit gates) ─────────────────────────────
_I      = np.eye(2, dtype=complex)
_X      = np.array([[0, 1], [1, 0]], dtype=complex)
_Y      = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z      = np.array([[1, 0], [0, -1]], dtype=complex)
_Hg     = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S      = np.array([[1, 0], [0, 1j]], dtype=complex)
CLIFFORDS      = [_I, _X, _Y, _Z, _Hg, _S]
CLIFFORD_NAMES = ["I", "X", "Y", "Z", "H", "S"]
N_CLIFF = len(CLIFFORDS)


def _jax_device_summary():
    if not HAS_JAX:
        return {"available": False, "backend": "none", "devices": []}
    devices = jax.devices()
    return {
        "available": True,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "has_tpu": any(getattr(d, "platform", "") == "tpu" for d in devices),
    }


def select_backend(requested="auto", require_tpu=False):
    """Choose the numerical backend and fail fast if a TPU run is required."""
    if requested == "numpy":
        if require_tpu:
            raise RuntimeError("--require-tpu cannot be combined with --backend numpy")
        return "numpy", _jax_device_summary()

    info = _jax_device_summary()
    if not info["available"]:
        if requested == "jax" or require_tpu:
            raise RuntimeError("JAX is not available, but a JAX/TPU backend was requested")
        return "numpy", info

    if require_tpu and not info.get("has_tpu", False):
        raise RuntimeError(f"JAX is available but no TPU device was found: {info}")

    if requested == "jax" or info.get("has_tpu", False):
        return "jax", info
    return "numpy", info
# _l_02_operators.py  —  Pauli helpers, PTM, entropies, F_rec

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [_sx, _sy, _sz]


def kron_site(op, site, N):
    """Embed 2x2 `op` at qubit index `site` in N-qubit Hilbert space."""
    ops = [np.eye(2, dtype=complex)] * N
    ops[site] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def partial_trace_boundary(rho, N):
    """Trace out all but sites 0 and N-1; return 4x4 boundary density matrix."""
    dim = 2 ** N
    keep = [0, N - 1]
    env_sites = [s for s in range(N) if s not in keep]
    n_env = len(env_sites)
    rho4 = np.zeros((4, 4), dtype=complex)
    for ia in range(2):
        for ib in range(2):
            for ja in range(2):
                for jb in range(2):
                    val = 0.0 + 0j
                    for e in range(2 ** n_env):
                        # build full basis index for bra and ket
                        def full_idx(a_bit, b_bit, env_bits):
                            idx = 0
                            ep = 0
                            for q in range(N):
                                if q == 0:
                                    b = a_bit
                                elif q == N - 1:
                                    b = b_bit
                                else:
                                    b = (env_bits >> (n_env - 1 - ep)) & 1
                                    ep += 1
                                idx = idx * 2 + b
                            return idx
                        bra = full_idx(ia, ib, e)
                        ket = full_idx(ja, jb, e)
                        val += rho[bra, ket]
                    rho4[ia * 2 + ib, ja * 2 + jb] = val
    return rho4


def von_neumann(rho):
    """Von Neumann entropy in bits."""
    ev = np.real(np.linalg.eigvalsh(rho))
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def mutual_info(rho4):
    """I(A:B) for 4x4 two-qubit state."""
    rA = np.array([[rho4[0, 0] + rho4[1, 1], rho4[0, 2] + rho4[1, 3]],
                   [rho4[2, 0] + rho4[3, 1], rho4[2, 2] + rho4[3, 3]]])
    rB = np.array([[rho4[0, 0] + rho4[2, 2], rho4[0, 1] + rho4[2, 3]],
                   [rho4[1, 0] + rho4[3, 2], rho4[1, 1] + rho4[3, 3]]])
    return max(0.0, von_neumann(rA) + von_neumann(rB) - von_neumann(rho4))


def compute_T_matrix(rho4):
    """3x3 PTM correlation matrix T_ij = Tr[(si x sj) rho4]."""
    T = np.zeros((3, 3))
    for i, si in enumerate(PAULIS):
        for j, sj in enumerate(PAULIS):
            T[i, j] = np.real(np.trace(np.kron(si, sj) @ rho4))
    return T


def f_rec_from_T(T):
    """Max recoverable fidelity F_rec = (1 + sv_max(T)) / 2."""
    sv = np.linalg.svd(T, compute_uv=False)
    return float((1.0 + sv[0]) / 2.0)


def spectral_entropy_T(T):
    """Normalised spectral entropy of |T| singular values."""
    sv = np.linalg.svd(T, compute_uv=False)
    sv = sv / (sv.sum() + 1e-15)
    sv = sv[sv > 1e-12]
    return float(-np.sum(sv * np.log(sv + 1e-15)))


def operator_spreading(rho, N, site=0):
    """OTOC proxy: <Z_0(t) Z_{N//2}> as scrambling measure."""
    mid = N // 2
    Z0  = kron_site(_sz, site, N)
    Zm  = kron_site(_sz, mid, N)
    val = np.real(np.trace(Z0 @ Zm @ rho))
    return float(val)
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


# ── Per-site operator cache (built once per N, reused every step) ────────────
_SZ_OPS_CACHE:    dict = {}
_RAISE_OPS_CACHE: dict = {}

def _get_sz_ops(N):
    if N not in _SZ_OPS_CACHE:
        _SZ_OPS_CACHE[N] = [kron_site(_sz, s, N) for s in range(N)]
    return _SZ_OPS_CACHE[N]

def _get_raise_ops(N):
    _raise = np.array([[0, 1], [0, 0]], dtype=complex)
    if N not in _RAISE_OPS_CACHE:
        _RAISE_OPS_CACHE[N] = [kron_site(_raise, s, N) for s in range(N)]
    return _RAISE_OPS_CACHE[N]


def apply_noise(rho, N, params, dt):
    """Apply Lindblad noise (T1, T2 dephasing) per site.
    Operator lists are cached at module level — built only once per N.
    """
    g1 = params.get("T1_rate", 0.0) * dt   # amplitude damping rate
    g2 = params.get("T2_rate", 0.0) * dt   # dephasing rate
    sz_ops    = _get_sz_ops(N)
    raise_ops = _get_raise_ops(N)

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


# ── ADVERSARIAL POISONING (PROGRAM_L_PLAN.md §Adversarial Representation) ────

ADVERSARIAL_MODES = ["none", "ee_scramble", "mi_inject", "ptm_phase", "channel_perm"]

def apply_adversarial_poisoning(obs_history, mode, rng):
    """
    Apply adversarial poisoning to the observable history seen by the
    agnostic controller. Modifies a *copy* — real history unchanged.

    Modes (PROGRAM_L_PLAN.md §Adversarial Representation Poisoning):
      ee_scramble  — shuffle EE time-series, preserve marginals
      mi_inject    — inject synthetic MI fronts
      ptm_phase    — randomise PTM SV phases (zero out correlations)
      channel_perm — randomly permute channel order
    """
    if not obs_history or mode == "none":
        return obs_history
    arr = np.array(obs_history, dtype=float)   # (T, N_OBS)
    if mode == "ee_scramble":
        ee_idx = OBS_NAMES.index("EE")
        shuffled = arr[:, ee_idx].copy()
        rng.shuffle(shuffled)
        arr = arr.copy(); arr[:, ee_idx] = shuffled
    elif mode == "mi_inject":
        mi_idx = OBS_NAMES.index("MI")
        arr = arr.copy()
        arr[:, mi_idx] = np.abs(rng.standard_normal(len(arr))) * 0.1
    elif mode == "ptm_phase":
        arr = arr.copy()
        for idx in [0, 1, 2]:   # ptm_sv1, ptm_sv2, ptm_sv3
            arr[:, idx] = np.abs(arr[:, idx]) * rng.choice([-1, 1], size=len(arr))
    elif mode == "channel_perm":
        perm = rng.permutation(N_OBS)
        arr = arr[:, perm]
    return arr.tolist()


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


_HAWARE_EIGH_CACHE: dict = {}   # model_key -> (evals, evecs)

def ctrl_haware(step, N, psi=None, H=None, dt=None, noise_params=None,
                evals=None, evecs=None, rng=None, **_):
    """
    Hamiltonian-aware controller: 1-step greedy oracle (pure-state).
    Tries each Clifford on site 0, applies eigh-based U + quantum-jump noise,
    picks pulse maximising F_rec. Upper bound.  O(2^N) per Clifford trial.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)

    if rng is None:
        rng = np.random.default_rng(step)

    best_idx, best_score = 0, -np.inf
    dt_ctrl = dt * CTRL_INTERVAL

    for ci in range(N_CLIFF):
        psi_c   = apply_clifford_psi(psi, N, ci, 0)
        psi_evo = apply_U_psi(psi_c, evals, evecs, dt_ctrl)
        psi_evo = apply_noise_psi(psi_evo, N, noise_params, dt_ctrl, rng)
        rho4    = partial_trace_boundary_psi(psi_evo, N)
        T       = compute_T_matrix(rho4)
        score   = f_rec_from_T(T)
        if score > best_score:
            best_score = score
            best_idx   = ci

    return (best_idx, 0)



# ── Program N3 controller constants (pre-registered, frozen before run) ────────
#
# GRAD_ALPHA_BASE: maximum gradient-following weight.
#   Unchanged from N2 = 0.05. Gated by susceptibility — see α_eff below.
GRAD_ALPHA_BASE = 0.05
#
# SUSCEPTIBILITY_SCALE: sensitivity threshold for gradient activation.
#   α_eff = GRAD_ALPHA_BASE * tanh(susceptibility / SUSCEPTIBILITY_SCALE)
SUSCEPTIBILITY_SCALE = 0.3
#
# LAMBDA_SPARSE: intervention sparsity penalty in ACTIVE transport regimes.
#   N2 value = 0.02. Kept for non-stable-basin steps.
LAMBDA_SPARSE = 0.02
#
# LAMBDA_STABLE: higher sparsity penalty enforced in STABLE BASINS.
#   N3 addition: fixes OAT over-firing from N2 (10 gates, tau_a < tau_s).
#   In a stable basin, controller must clear 0.07 improvement to fire any gate.
#   OAT at D_eff≈0 will not clear this bar. W3 in active regime is NOT stable
#   basin → gets LAMBDA_SPARSE=0.02, restoring gradient-following gain.
LAMBDA_STABLE = 0.07
#
# Stable basin detection thresholds (pre-registered, CLI-overridable):
#   STABLE_BASIN_D_EFF    — max |D_eff| to classify as stable
#   STABLE_BASIN_DX_NORM  — max ||Δx|| to classify as stable
#   STABLE_BASIN_SUSCEPT  — max susceptibility to classify as stable
STABLE_BASIN_D_EFF   = 0.08
STABLE_BASIN_DX_NORM = 0.05
STABLE_BASIN_SUSCEPT = 0.15


def ctrl_agnostic(step, N, rho, obs_history, t_history,
                  dropout_rng, use_dropout=True, **_):
    """
    D-LinOSS agnostic controller v4 (N3) — regime-conditioned λ.

    Architecture:  z_t = f(x_{0:t}, u_{0:t}, Δx_{0:t})

    Score function:
        score(u) = strata_score(x_pred)
                 + α_eff(t)    * cos(Δx_pred, Δx_obs)   ← susceptibility-gated
                 - lambda_eff(t)                          ← regime-conditioned penalty

    Complementary gate (N3 innovation):
        high susceptibility  →  α_eff↑  + lambda_eff=LAMBDA_SPARSE  (W3 regime)
        low susceptibility   →  α_eff≈0 + lambda_eff=LAMBDA_STABLE  (OAT/stable)

    Stable basin criteria (all three must hold):
        |D_eff|       < STABLE_BASIN_D_EFF
        ||Δx||        < STABLE_BASIN_DX_NORM
        susceptibility < STABLE_BASIN_SUSCEPT

    Perturbation model calibrated from N1 response Jacobian:
        high-response channels: D_eff, noise_slope, basis_invar, front_v
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)

    x_cur = np.array(obs_history[-1]) if obs_history else np.zeros(N_OBS)

    # Response gradient: Δx = x_t - x_{t-1}
    if len(obs_history) >= 2:
        dx_recent = np.array(obs_history[-1]) - np.array(obs_history[-2])
    else:
        dx_recent = np.zeros(N_OBS)
    dx_norm = float(np.linalg.norm(dx_recent))

    # OBS_NAMES indices: D_eff=8, noise_slope=10
    d_eff_mag   = abs(float(x_cur[8]))    if len(x_cur) > 8  else 0.0
    noise_s_mag = abs(float(x_cur[10]))   if len(x_cur) > 10 else 0.0
    susceptibility = dx_norm * (1.0 + d_eff_mag + noise_s_mag)

    # Dynamic α_eff — susceptibility-gated gradient weight.
    alpha_eff = GRAD_ALPHA_BASE * float(np.tanh(susceptibility / SUSCEPTIBILITY_SCALE))

    # Regime-conditioned λ (N3 core change).
    # Stable basin: D_eff small AND dx_norm small AND susceptibility low.
    # λ=LAMBDA_STABLE forces strong restraint (fixes OAT).
    # Active regime: λ=LAMBDA_SPARSE allows gradient-guided intervention (restores W3).
    is_stable_basin = (
        d_eff_mag     < STABLE_BASIN_D_EFF
        and dx_norm   < STABLE_BASIN_DX_NORM
        and susceptibility < STABLE_BASIN_SUSCEPT
    )
    lambda_eff = LAMBDA_STABLE if is_stable_basin else LAMBDA_SPARSE

    # Optional representation dropout
    if use_dropout and len(obs_history) >= 4:
        mask  = dropout_rng.binomial(1, 1.0 - P_DROPOUT, size=N_OBS).astype(float)
        x_cur = x_cur * mask

    # Identity baseline — no sparsity penalty for doing nothing
    best_idx, best_score = 0, strata_score(x_cur)

    for ci in range(1, N_CLIFF):   # skip identity
        x_pred     = _predict_obs_after_cliff(x_cur, ci)
        delta_pred = x_pred - x_cur
        dp_norm    = float(np.linalg.norm(delta_pred))

        # Gradient alignment
        if dx_norm > 1e-8 and dp_norm > 1e-8:
            alignment = float(np.dot(delta_pred, dx_recent) / (dp_norm * dx_norm))
        else:
            alignment = 0.0

        # Combined score: strata + gradient - regime-conditioned penalty
        sc = strata_score(x_pred) + alpha_eff * alignment - lambda_eff
        if sc > best_score:
            best_score = sc
            best_idx   = ci

    return (best_idx, 0)



# Per-Clifford predicted observable perturbation vectors.
# v3: calibrated from Program N1 response Jacobian.
#
# High-response channels (from Jacobian, descending |Δx|):
#   D_eff(8)=0.626, noise_slope(10)=0.465, basis_invar(11)=0.396,
#   front_v(9)=0.334, OS(6)=0.243, EE(4)=0.127
#
# Old model targeted: ptm_sv1, ptm_H, EE, MI, basis_invar (low-Jacobian channels).
# New model targets the channels the system actually responds through.
#
# Physics rationale per gate:
#   X  : echo-type flip — suppresses noise_slope (refocusing), slows D_eff
#   Y  : X + axis-rotation — also disrupts basis_invar
#   Z  : dephasing axis — strongest noise_slope suppression, stabilises basis_invar
#   H  : x<->z swap — large basis_invar change, increases front_v (spreading)
#   S  : quarter-turn — moderate basis_invar, mild noise_slope suppression
_CLIFF_OBS_DELTAS = [
    # I: no change
    {},
    # X: echo refocusing — suppresses noise_slope, reduces D_eff (coherence echo)
    {"noise_slope": -0.08, "D_eff": -0.05, "front_v": -0.03},
    # Y: echo + axis-mix — like X but also perturbs basis_invar
    {"noise_slope": -0.06, "D_eff": -0.04, "basis_invar": -0.05, "front_v": -0.02},
    # Z: dephasing suppression — strongest noise_slope reduction, stabilises basis_invar
    {"noise_slope": -0.10, "basis_invar": -0.06, "D_eff": -0.03},
    # H: basis swap — large basis_invar perturbation, increases front_v (operator spreading)
    {"basis_invar": +0.08, "front_v": +0.06, "D_eff": +0.03},
    # S: quarter-turn — moderate basis_invar, mild noise_slope
    {"basis_invar": -0.04, "noise_slope": -0.04, "front_v": -0.01},
]


def _predict_obs_after_cliff(x_cur, ci):
    """Return predicted observable vector after applying Clifford ci."""
    x_pred = x_cur.copy()
    for obs_name, delta in _CLIFF_OBS_DELTAS[ci].items():
        idx = OBS_NAMES.index(obs_name)
        x_pred[idx] = np.clip(x_pred[idx] + delta, 0.0, 1.0)
    return x_pred




def ctrl_dd_xy8(step, N, **_):
    """
    Dynamical decoupling -- XY8 schedule (engineering baseline).
    8-pulse cycle: X Y X Y Y X Y X on site 0.
    No adaptation, no Hamiltonian, no observables.
    Best known fixed-schedule DD for combined T1/T2 noise.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)
    _XY8 = [1, 2, 1, 2, 2, 1, 2, 1]
    cycle = (step // CTRL_INTERVAL) % 8
    return (_XY8[cycle], 0)


def ctrl_dd_cpmg(step, N, **_):
    """
    Dynamical decoupling -- CPMG schedule (engineering baseline).
    Alternating X echoes at each control interval.
    Simpler than XY8; optimised for T2 dephasing noise.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)
    cycle = (step // CTRL_INTERVAL) % 2
    return (1 if cycle == 0 else 0, 0)



def ctrl_random(step, N, seed=None, **_):
    """
    Stochastic control null -- selects uniformly from all Clifford gates.
    Scientific role: if agnostic > random, the adaptive structure itself causes the gain.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)
    rng = np.random.default_rng(seed if seed is not None else step * 7919)
    return (int(rng.integers(0, N_CLIFF)), 0)


def get_controller(name):
    controllers = {
        "static":   ctrl_static,
        "haware":   ctrl_haware,
        "agnostic": ctrl_agnostic,
        "dd_xy8":   ctrl_dd_xy8,
        "dd_cpmg":  ctrl_dd_cpmg,
        "random":   ctrl_random,
    }
    if name not in controllers:
        raise ValueError(f"Unknown controller '{name}'. Choose: {list(controllers)}")
    return controllers[name]
# ── Pure-state helpers (O(2^N), match Program K pattern) ─────────────────────

def partial_trace_boundary_psi(psi, N):
    """Compute 4×4 boundary RDM from pure state psi.
    Sites 0 and N-1 kept; inner sites traced.  O(2^N).
    """
    psi_r = psi.reshape([2] * N)
    rho4  = np.zeros((4, 4), dtype=complex)
    for ia in range(2):
        for ib in range(2):
            bra = psi_r[ia, ..., ib].reshape(-1)
            for ja in range(2):
                for jb in range(2):
                    ket = psi_r[ja, ..., jb].reshape(-1)
                    rho4[ia*2+ib, ja*2+jb] = np.dot(bra, ket.conj())
    return rho4


def operator_spreading_psi(psi, N, site=0):
    """<Z_site Z_{N//2}> from pure state.  Diagonal — O(2^N)."""
    diag = operator_spreading_diag(N, site)
    return float(np.real(np.dot(psi.conj(), diag * psi)))


def operator_spreading_diag(N, site=0):
    """Precompute the diagonal for <Z_site Z_{N//2}>."""
    mid  = N // 2
    idx  = np.arange(2 ** N)
    b0   = (idx >> (N - 1 - site)) & 1
    bm   = (idx >> (N - 1 - mid))  & 1
    return (1 - 2*b0) * (1 - 2*bm)     # (-1)^b0 * (-1)^bm


def apply_clifford_psi(psi, N, cliff_idx, site):
    """Apply Clifford gate to site on pure state.  O(2^N) matmul."""
    U = kron_site(CLIFFORDS[cliff_idx], site, N)
    return U @ psi


if HAS_JAX:
    @partial(jax.jit, static_argnames=("N",))
    def partial_trace_boundary_psi_jax(psi, N):
        """Boundary RDM using tensor reshape/contraction on the active JAX device."""
        psi_r = jnp.reshape(psi, (2,) * N)
        axes = (0, N - 1) + tuple(range(1, N - 1))
        kept_env = jnp.reshape(jnp.transpose(psi_r, axes), (4, -1))
        return kept_env @ jnp.conj(kept_env.T)


    @partial(jax.jit, static_argnames=("N", "site"))
    def apply_site_gate_jax(psi, gate, N, site):
        """Apply a 2x2 gate as a tensor contraction, avoiding dense kron matrices."""
        psi_r = jnp.reshape(psi, (2,) * N)
        moved = jnp.moveaxis(psi_r, site, 0)
        flat = jnp.reshape(moved, (2, -1))
        updated = gate @ flat
        restored = jnp.moveaxis(jnp.reshape(updated, (2,) + (2,) * (N - 1)), 0, site)
        return jnp.reshape(restored, (-1,))


    @jax.jit
    def apply_U_psi_jax(psi, evals, evecs, dt):
        """Exact unitary evolution via cached eigensystem on the active JAX device."""
        c = jnp.conj(evecs.T) @ psi
        c = c * jnp.exp(-1j * evals * dt)
        return evecs @ c


    @jax.jit
    def operator_spreading_psi_jax(psi, diag):
        return jnp.real(jnp.vdot(psi, diag * psi))
else:
    def partial_trace_boundary_psi_jax(*_args, **_kwargs):
        raise RuntimeError("JAX backend requested but JAX is not available")

    def apply_site_gate_jax(*_args, **_kwargs):
        raise RuntimeError("JAX backend requested but JAX is not available")

    def apply_U_psi_jax(*_args, **_kwargs):
        raise RuntimeError("JAX backend requested but JAX is not available")

    def operator_spreading_psi_jax(*_args, **_kwargs):
        raise RuntimeError("JAX backend requested but JAX is not available")


def apply_noise_psi(psi, N, params, dt, rng):
    """Quantum-trajectory (stochastic jump) noise.
    Equivalent to Lindblad in expectation; O(N * 2^N) per step.
    """
    g1 = params.get("T1_rate", 0.0) * dt
    g2 = params.get("T2_rate", 0.0) * dt
    psi_r = psi.reshape([2] * N).copy()

    for site in range(N):
        # T1 amplitude damping: jump |1>→|0> with prob ≈ g1
        if g1 > 0:
            p = min(g1, 1.0)
            idx1 = [slice(None)] * N; idx1[site] = 1
            idx0 = [slice(None)] * N; idx0[site] = 0
            if rng.random() < p:
                psi_r[tuple(idx0)] = psi_r[tuple(idx1)]
                psi_r[tuple(idx1)] = 0.0
                n = np.linalg.norm(psi_r)
                if n > 1e-12: psi_r /= n
            else:
                psi_r[tuple(idx1)] *= np.sqrt(max(1 - p, 0))
                n = np.linalg.norm(psi_r)
                if n > 1e-12: psi_r /= n

        # T2 dephasing: Z jump with prob ≈ g2/2
        if g2 > 0:
            p = min(g2 / 2, 0.5)
            if rng.random() < p:
                idx1 = [slice(None)] * N; idx1[site] = 1
                psi_r[tuple(idx1)] *= -1.0   # Z|1> = -|1>

    # Coherent drift: exp(-i*angle*Z) on site 0
    drift_amp  = params.get("drift_amp",  0.0)
    drift_freq = params.get("drift_freq", 0.1)
    t_cur      = params.get("_t_cur",     0.0)
    if drift_amp > 0:
        angle = drift_amp * np.sin(drift_freq * t_cur) * dt
        idx0 = [slice(None)] * N; idx0[0] = 0
        idx1 = [slice(None)] * N; idx1[0] = 1
        psi_r[tuple(idx0)] *= np.exp(-1j * angle)
        psi_r[tuple(idx1)] *= np.exp( 1j * angle)

    return psi_r.reshape(-1)


# ── eigh-based U cache (exact, no expm) ───────────────────────────────────────
_EIGH_CACHE: dict = {}   # model_key -> (evals, evecs)

def _get_eigh(H, model_key):
    if model_key not in _EIGH_CACHE:
        _EIGH_CACHE[model_key] = np.linalg.eigh(H)
    return _EIGH_CACHE[model_key]

def apply_U_psi(psi, evals, evecs, dt):
    """Exact unitary evolution via eigh: O(2^N * 2^N) but no expm."""
    c = evecs.conj().T @ psi           # project
    c *= np.exp(-1j * evals * dt)      # phase
    return evecs @ c                   # back


def ctrl_haware_device(step, N, psi_dev, dt, noise_params,
                       evals_dev, evecs_dev, cliffords_dev, rng):
    """
    JAX-device version of the Hamiltonian-aware greedy controller.
    Evolution and boundary contractions stay on device; stochastic jumps use
    the existing NumPy trajectory routine to preserve the registered noise path.
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)

    best_idx, best_score = 0, -np.inf
    dt_ctrl = dt * CTRL_INTERVAL

    for ci in range(N_CLIFF):
        psi_c = apply_site_gate_jax(psi_dev, cliffords_dev[ci], N, 0)
        psi_evo = apply_U_psi_jax(psi_c, evals_dev, evecs_dev, dt_ctrl)
        psi_evo_np = np.asarray(jax.device_get(psi_evo))
        psi_evo_np = apply_noise_psi(psi_evo_np, N, noise_params, dt_ctrl, rng)
        psi_evo = jnp.asarray(psi_evo_np)
        rho4 = np.asarray(jax.device_get(partial_trace_boundary_psi_jax(psi_evo, N)))
        T = compute_T_matrix(rho4)
        score = f_rec_from_T(T)
        if score > best_score:
            best_score = score
            best_idx = ci

    return (best_idx, 0)


# ── Trajectory simulator — pure-state, eigh, O(2^N) per step ─────────────────

def simulate_trajectory(model_name, N, controller_name, noise_params,
                        T_max, n_steps, seed, use_dropout=True,
                        backend="numpy", adversarial="none",
                        store_obs=False):
    """
    Simulate one noisy trajectory under one controller.
    Pure-state propagation (Schrödinger picture) + quantum-trajectory noise.
    eigh called once per model; per-step cost is O(2^{2N}) matmuls (fast).
    """
    rng   = np.random.default_rng(seed)
    d_rng = np.random.default_rng(seed + 10000)
    H     = MODEL_BUILDERS[model_name](N, seed)
    dt    = T_max / n_steps
    dim   = 2 ** N

    # Diagonalize once — reused across all steps (Program K pattern)
    model_key = (model_name, N, seed)
    evals, evecs = _get_eigh(H, model_key)

    # Initial pure state: Néel-like
    neel_idx = sum(1 << i for i in range(0, N, 2))
    psi = np.zeros(dim, dtype=complex)
    psi[neel_idx] = 1.0
    use_jax = (backend == "jax")
    if use_jax:
        psi_dev = jnp.asarray(psi)
        evals_dev = jnp.asarray(evals)
        evecs_dev = jnp.asarray(evecs)
        cliffords_dev = [jnp.asarray(g) for g in CLIFFORDS]
        os_diag_dev = jnp.asarray(operator_spreading_diag(N))
    else:
        psi_dev = None
        evals_dev = None
        evecs_dev = None
        cliffords_dev = None
        os_diag_dev = None

    obs_history  = []
    t_history    = []
    f_rec_series = []
    in_transport = []
    ctrl_actions = []   # (cliff_idx, site) at each step
    psi_prev     = None
    psi_prev_dev = None
    adv_rng      = np.random.default_rng(seed + 99999)   # separate rng for poisoning

    controller = get_controller(controller_name)

    for step in range(n_steps):
        t_cur = step * dt
        noise_params["_t_cur"] = t_cur

        # Controller decision — pure-state throughout, no density matrix built
        if controller_name == "static":
            ci, site = controller(step, N)
        elif controller_name == "haware":
            if use_jax:
                ci, site = ctrl_haware_device(
                    step, N, psi_dev, dt, noise_params,
                    evals_dev, evecs_dev, cliffords_dev, rng
                )
            else:
                ci, site = controller(step, N, psi=psi, H=H, dt=dt,
                                      noise_params=noise_params,
                                      evals=evals, evecs=evecs, rng=rng)
        else:  # agnostic — only uses obs_history, no rho needed
            # Apply adversarial poisoning to the history the controller sees
            poisoned_history = (
                apply_adversarial_poisoning(obs_history, adversarial, adv_rng)
                if adversarial != "none" else obs_history
            )
            ci, site = controller(step, N, rho=None,
                                  obs_history=poisoned_history,
                                  t_history=t_history,
                                  dropout_rng=d_rng,
                                  use_dropout=use_dropout)

        ctrl_actions.append((int(ci), int(site)))

        if use_jax:
            # Apply Clifford as a rank-N tensor update, then evolve on device.
            if ci != 0:
                psi_dev = apply_site_gate_jax(psi_dev, cliffords_dev[ci], N, site)

            psi_dev = apply_U_psi_jax(psi_dev, evals_dev, evecs_dev, dt)

            # Preserve the pre-registered stochastic noise implementation.
            psi = np.asarray(jax.device_get(psi_dev))
            psi = apply_noise_psi(psi, N, noise_params, dt, rng)
            psi_dev = jnp.asarray(psi)

            rho4 = np.asarray(jax.device_get(partial_trace_boundary_psi_jax(psi_dev, N)))
            psi_prev_rho4 = (
                np.asarray(jax.device_get(partial_trace_boundary_psi_jax(psi_prev_dev, N)))
                if psi_prev_dev is not None else None
            )
        else:
            # Apply Clifford pulse
            if ci != 0:
                psi = apply_clifford_psi(psi, N, ci, site)

            # Time-evolve (exact, no expm)
            psi = apply_U_psi(psi, evals, evecs, dt)

            # Quantum-trajectory noise
            psi = apply_noise_psi(psi, N, noise_params, dt, rng)

            # Observables from pure state
            rho4 = partial_trace_boundary_psi(psi, N)
            psi_prev_rho4 = partial_trace_boundary_psi(psi_prev, N) if psi_prev is not None else None

        T_mat    = compute_T_matrix(rho4)
        sv       = np.linalg.svd(T_mat, compute_uv=False)
        ptm_sv1  = float(sv[0])
        ptm_sv2  = float(sv[1]) if len(sv) > 1 else 0.0
        ptm_sv3  = float(sv[2]) if len(sv) > 2 else 0.0
        ptm_H    = spectral_entropy_T(T_mat)
        EE       = von_neumann(rho4)
        MI       = mutual_info(rho4)
        OS       = (float(jax.device_get(operator_spreading_psi_jax(psi_dev, os_diag_dev)))
                    if use_jax else operator_spreading_psi(psi, N))
        H_T      = ptm_H

        D_eff    = float((EE - von_neumann(psi_prev_rho4)) / (dt + 1e-12)) \
                   if psi_prev_rho4 is not None else 0.0
        front_v  = 0.0
        if psi_prev is not None:
            OS_prev = (
                float(jax.device_get(operator_spreading_psi_jax(psi_prev_dev, os_diag_dev)))
                if use_jax else operator_spreading_psi(psi_prev, N)
            )
            front_v = float((OS - OS_prev) / (dt + 1e-12))

        noise_slope = 0.0
        if psi_prev_rho4 is not None:
            T_prev = compute_T_matrix(psi_prev_rho4)
            sv_prev = np.linalg.svd(T_prev, compute_uv=False)
            noise_slope = float((sv[0] - sv_prev[0]) / (dt + 1e-12))

        c_idx    = int(rng.integers(1, N_CLIFF))
        if use_jax:
            psi_sc = apply_site_gate_jax(psi_dev, cliffords_dev[c_idx], N, 0)
            rho4_sc = np.asarray(jax.device_get(partial_trace_boundary_psi_jax(psi_sc, N)))
        else:
            psi_sc   = apply_clifford_psi(psi, N, c_idx, 0)
            rho4_sc  = partial_trace_boundary_psi(psi_sc, N)
        T_sc     = compute_T_matrix(rho4_sc)
        basis_invar = float(np.linalg.norm(T_mat - T_sc, "fro"))

        x_obs = np.array([
            ptm_sv1, ptm_sv2, ptm_sv3, ptm_H,
            EE, MI, OS, H_T,
            D_eff, front_v, noise_slope, basis_invar,
        ])
        obs_history.append(x_obs.tolist())
        t_history.append(t_cur)

        f_rec = f_rec_from_T(T_mat)
        f_rec_series.append(float(f_rec))
        in_transport.append(int(infer_transport_strata(x_obs)))

        psi_prev = psi.copy()
        psi_prev_dev = psi_dev if use_jax else None

    above = [i for i, f in enumerate(f_rec_series) if f > TH_F_REC]
    tau_transport = float(above[-1] * dt) if above else 0.0

    rec = {
        "model":         model_name,
        "controller":    controller_name,
        "N":             N,
        "seed":          seed,
        "tau_transport": tau_transport,
        "f_rec_series":  f_rec_series,
        "in_transport":  in_transport,
        "ctrl_actions":  ctrl_actions,
        "T_max":         T_max,
        "n_steps":       n_steps,
        "backend":       backend,
        "adversarial":   adversarial,
        "noise_params":  {k: v for k, v in noise_params.items()
                          if not k.startswith("_")},
    }
    if store_obs:
        rec["obs_history"] = obs_history
    return rec


def sample_noise_params(rng):
    """Sample noise parameters from pre-registered ranges."""
    return {
        "T1_rate":    float(rng.uniform(0.005, 0.05)),
        "T2_rate":    float(rng.uniform(0.005, 0.10)),
        "drift_amp":  float(rng.uniform(0.0,   0.05)),
        "drift_freq": float(rng.uniform(0.05,  0.5)),
    }


def run_experiment(args):
    models      = args.models
    controllers = getattr(args, "controllers", ["static", "haware", "agnostic"])
    results     = []
    t0          = time.time()
    backend, backend_info = select_backend(args.backend, args.require_tpu)

    print("=" * 60)
    print("Program L — Representation-Agnostic Adaptive Recovery")
    print("=" * 60)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Models: {models}")
    print(f"Controllers: {controllers}")
    print(f"Bootstrap samples: {args.B}")
    print(f"Backend: {backend}  JAX: {backend_info}")

    rng_noise = np.random.default_rng(args.seed)

    for model in models:
        print(f"\n── Model: {model} ──")
        for b in range(args.B):
            noise_params = sample_noise_params(rng_noise)
            seed_b = args.seed + b * 1000
            for ctrl in controllers:
                t1 = time.time()
                rec = simulate_trajectory(
                    model_name=model,
                    N=args.N,
                    controller_name=ctrl,
                    noise_params=noise_params.copy(),
                    T_max=args.T_max,
                    n_steps=args.n_steps,
                    seed=seed_b,
                    use_dropout=(ctrl == "agnostic"),
                    backend=backend,
                    adversarial=getattr(args, "adversarial", "none"),
                    store_obs=getattr(args, "store_obs", False),
                )
                results.append(rec)
                dt_run = time.time() - t1
                print(f"  [{ctrl:8s}] B={b}  tau={rec['tau_transport']:.3f}  "
                      f"F_final={rec['f_rec_series'][-1]:.3f}  ({dt_run:.1f}s)")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GAIN SUMMARY  (gain = (agnostic-static)/(haware-static))")
    print("=" * 60)
    summary = {}
    for model in models:
        recs = [r for r in results if r["model"] == model]
        taus = {c: [r["tau_transport"] for r in recs if r["controller"] == c]
                for c in controllers}
        tau_s = float(np.mean(taus["static"]))   if taus["static"]   else 0.0
        tau_h = float(np.mean(taus["haware"]))   if taus["haware"]   else 0.0
        tau_a = float(np.mean(taus["agnostic"])) if taus["agnostic"] else 0.0
        denom = tau_h - tau_s
        gain  = float((tau_a - tau_s) / denom) if abs(denom) > 1e-9 else float("nan")
        summary[model] = {
            "tau_static": tau_s, "tau_haware": tau_h,
            "tau_agnostic": tau_a, "gain": gain,
        }
        print(f"  {model:20s}  tau_s={tau_s:.3f}  tau_h={tau_h:.3f}  "
              f"tau_a={tau_a:.3f}  gain={gain:.3f}")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_l_N{args.N}_results.json")
    # Strip f_rec_series always. Strip obs_history unless --store-obs.
    _strip = {"f_rec_series"} if getattr(args, "store_obs", False) else {"f_rec_series", "obs_history"}
    with open(out_path, "w") as f:
        json.dump({
            "args":         vars(args),
            "backend":      backend,
            "backend_info": backend_info,
            "summary":      summary,
            "results":      [{k: v for k, v in r.items() if k not in _strip}
                             for r in results],
            "runtime_s":    round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time()-t0:.1f}s total)")
    return summary


def main():
    p = argparse.ArgumentParser(description="Program L: Agnostic Transport Recovery")
    p.add_argument("--N",       type=int,   default=8)
    p.add_argument("--T-max",   type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int,   default=30,  dest="n_steps")
    p.add_argument("--B",       type=int,   default=3,
                   help="Bootstrap samples per model")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--models",  nargs="+",
                   default=["XXZ", "Ising"],
                   choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--out-dir", default="program_l_results", dest="out_dir")
    p.add_argument("--backend", choices=["auto", "numpy", "jax"], default="auto",
                   help="auto uses JAX when a TPU is visible; otherwise NumPy")
    p.add_argument("--require-tpu", action="store_true",
                   help="Fail fast unless JAX can see a TPU device")
    p.add_argument("--store-obs", action="store_true", dest="store_obs",
                   help="Save obs_history in results JSON (large — for diagnostic runs)")
    p.add_argument("--adversarial", default="none",
                   choices=ADVERSARIAL_MODES,
                   help="Adversarial poisoning mode applied to agnostic controller view")
    p.add_argument("--controllers", nargs="+",
                   default=["static", "haware", "agnostic"],
                   choices=["static", "haware", "agnostic", "dd_xy8", "dd_cpmg", "random"],
                   help="Controllers to run (default: static haware agnostic)")

    # ── N3 pre-registered stable-basin thresholds (CLI-overridable) ────────────
    p.add_argument("--lambda-stable", type=float, default=LAMBDA_STABLE,
                   dest="lambda_stable",
                   help="Sparsity penalty in stable basins (N3, default 0.07)")
    p.add_argument("--stable-basin-d-eff", type=float, default=STABLE_BASIN_D_EFF,
                   dest="stable_basin_d_eff",
                   help="Max |D_eff| for stable-basin classification (default 0.08)")
    p.add_argument("--stable-basin-dx-norm", type=float, default=STABLE_BASIN_DX_NORM,
                   dest="stable_basin_dx_norm",
                   help="Max ||Δx|| for stable-basin classification (default 0.05)")
    p.add_argument("--stable-basin-suscept", type=float, default=STABLE_BASIN_SUSCEPT,
                   dest="stable_basin_suscept",
                   help="Max susceptibility for stable-basin classification (default 0.15)")

    p.add_argument("--smoke-test", action="store_true",
                   help="Quick validation: N=6, 2 models, B=1, 15 steps")
    args = p.parse_args()

    if args.smoke_test:
        args.N       = 6
        args.n_steps = 15
        args.B       = 1
        args.T_max   = 4.0
        args.models  = ["XXZ", "Ising"]
        print("SMOKE TEST MODE")

    run_experiment(args)


if __name__ == "__main__":
    main()
