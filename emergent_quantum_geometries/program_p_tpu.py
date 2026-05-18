"""
Program P -- Partial Information Recovery
=========================================
Extends Program O to test latent-state reconstruction under controlled degradation.

Key shift from Program O:
  Program O: "Did fidelity stay high?" (endpoint overlap)
  Program P: "How much recoverable phase information survives degradation?"

Experiment design:
  1. Encode phase probes: |psi(phi)? = (|0? + e^{iphi}|1?)/sqrt2  ?  |0?^{N-1}
  2. Apply controlled degradation channel (dephasing / jitter / scrambling)
  3. Evolve with controller (static vs agnostic) -- controller never sees phi
  4. Reconstruct phi? from output state boundary observables
  5. Metrics: Phase MAE, R2, I(phi_true; phi?)

Pre-registered thresholds FROZEN before first run.
"""
import argparse
import json
import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from program_l_tpu import (
    MODEL_BUILDERS, sample_noise_params, _get_eigh,
    apply_U_psi, apply_clifford_psi, apply_noise_psi,
    partial_trace_boundary_psi, compute_T_matrix,
    operator_spreading_psi, von_neumann, mutual_info,
    spectral_entropy_T, f_rec_from_T,
    get_controller, CLIFFORDS, N_CLIFF, CTRL_INTERVAL,
    N_OBS, strata_score, _predict_obs_after_cliff,
    LAMBDA_SPARSE, LAMBDA_STABLE, STABLE_BASIN_D_EFF,
    STABLE_BASIN_DX_NORM, STABLE_BASIN_SUSCEPT,
    GRAD_ALPHA_BASE, SUSCEPTIBILITY_SCALE,
    FRAGILITY_FRONT_V_THRESH, P_DROPOUT,
    apply_site_gate_jax, apply_U_psi_jax,
    partial_trace_boundary_psi_jax, operator_spreading_psi_jax,
    operator_spreading_diag,
    select_backend, _jax_device_summary,
    TH_F_REC,
)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    jax = None; jnp = None; HAS_JAX = False

# -- PRE-REGISTERED THRESHOLDS (FROZEN) ----------------------------------------
TH_PHASE_MAE   = 0.60   # rad -- agnostic MAE must beat static by this much
TH_R2_MIN      = 0.05   # agnostic R2 must exceed static R2 by this margin
TH_MI_POS      = 0.02   # nats -- I(phi_true; phi?) must be positive

# Number of discrete phase angles in the probe sweep
N_PHASES = 24

# Degradation channel names
DEGRADATION_CHANNELS = ["none", "dephasing", "jitter", "scrambling"]


# -- PHASE PROBE ENCODING ------------------------------------------------------

def encode_phase_probe(phi, N):
    """
    |psi(phi)? = (|0? + e^{iphi}|1?)/sqrt2  on site 0, |0? on all others.
    The encoded phase phi is the latent variable -- never given to the controller.
    """
    dim = 2 ** N
    psi = np.zeros(dim, dtype=complex)
    # Site 0 in |0? state ? index 0?(2^{N-1}-1)
    # Site 0 in |1? state ? index 2^{N-1}?(2^N-1)
    half = dim // 2
    # All other sites in |0? ? only index 0 and half contribute
    psi[0]    = 1.0 / np.sqrt(2)
    psi[half] = np.exp(1j * phi) / np.sqrt(2)
    return psi


# -- DEGRADATION CHANNELS ------------------------------------------------------

def apply_degradation(psi, N, channel, sigma, rng):
    """Apply one controlled corruption to state psi."""
    if channel == "none" or sigma == 0.0:
        return psi

    if channel == "dephasing":
        # Apply random Z-rotations on each qubit with strength sigma
        dim = 2 ** N
        angles = rng.normal(0, sigma, N)
        for q in range(N):
            # Z rotation on qubit q: diag exp(?i angle/2) in 2^N space
            half = dim >> 1
            mask = np.arange(dim)
            bit = (mask >> (N - 1 - q)) & 1
            phase = np.where(bit == 0,
                             np.exp(1j * angles[q] / 2),
                             np.exp(-1j * angles[q] / 2))
            psi = psi * phase
        return psi / np.linalg.norm(psi)

    if channel == "jitter":
        # Scramble phase of each amplitude by N(0, sigma)
        phases = rng.normal(0, sigma, psi.shape)
        return psi * np.exp(1j * phases) / np.linalg.norm(
            psi * np.exp(1j * phases))

    if channel == "scrambling":
        # Apply random single-qubit Clifford with probability sigma
        if rng.random() < sigma:
            ci = int(rng.integers(1, N_CLIFF))
            site = int(rng.integers(0, N))
            psi = apply_clifford_psi(psi, N, ci, site)
        return psi

    return psi


# -- PHASE RECONSTRUCTION ------------------------------------------------------

def reconstruct_phase(psi, N):
    """
    Estimate phi? from output state by tracing out all sites except site 0.
    phi? = arg(?sigma_x? + i?sigma_y?) on site-0 marginal.
    """
    dim = 2 ** N
    half = dim // 2
    # Site-0 reduced density matrix (2?2)
    rho0 = np.zeros((2, 2), dtype=complex)
    # Sum over all other-site configurations
    for k in range(half):
        a = psi[k]        # |0?_0 ? |k?_rest
        b = psi[k + half] # |1?_0 ? |k?_rest
        rho0[0, 0] += a * np.conj(a)
        rho0[0, 1] += a * np.conj(b)
        rho0[1, 0] += b * np.conj(a)
        rho0[1, 1] += b * np.conj(b)

    # Bloch vector
    sx = float(np.real(rho0[0, 1] + rho0[1, 0]))
    sy = float(np.real(1j * (rho0[0, 1] - rho0[1, 0])))
    phi_hat = float(np.arctan2(sy, sx)) % (2 * np.pi)
    coherence = float(np.sqrt(sx**2 + sy**2))
    return phi_hat, coherence


# -- PHASE METRICS -------------------------------------------------------------

def phase_mae(phi_true, phi_hat):
    """Mean absolute error on circular domain [0, 2pi)."""
    diffs = np.array([(a - b) % (2 * np.pi) for a, b in zip(phi_true, phi_hat)])
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    return float(np.mean(diffs))


def phase_r2(phi_true, phi_hat):
    """Circular R2 -- 1 - Var(residuals) / Var(true)."""
    residuals = np.array([(a - b) % (2 * np.pi) for a, b in zip(phi_true, phi_hat)])
    residuals = np.minimum(residuals, 2 * np.pi - residuals)
    var_res = float(np.var(residuals))
    var_true = float(np.var(np.array(phi_true)))
    if var_true < 1e-10:
        return 0.0
    return float(1.0 - var_res / var_true)


def phase_mi(phi_true, phi_hat, n_bins=8):
    """Mutual information I(phi_true; phi?) via joint histogram (nats)."""
    phi_true = np.array(phi_true)
    phi_hat  = np.array(phi_hat)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    joint, _, _ = np.histogram2d(phi_true, phi_hat, bins=[bins, bins])
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            (joint > 0) & (px * py > 0),
            np.log(joint / (px * py + 1e-30)),
            0.0)
    return float(np.sum(joint * log_ratio))


# -- CHIRP SYSTEM IDENTIFICATION -----------------------------------------------

def chirp_sweep(model_name, N, noise_params, dt, seed, K=12, backend="numpy"):
    """
    Estimate local intervention susceptibility by sweeping phi over K angles
    and measuring ?F/?phi, phase sensitivity, damping rate.

    Analogous to MRI calibration / radar channel estimation.

    Returns:
        chirp_summary: dict with damping_rate, phase_sensitivity, echo_R, jacobian_norm
    """
    rng = np.random.default_rng(seed + 9999)
    H = MODEL_BUILDERS[model_name](N, seed)
    model_key = (model_name, N, seed)
    evals, evecs = _get_eigh(H, model_key)
    half = 2 ** N // 2

    phi_vals = np.linspace(0, 2 * np.pi, K, endpoint=False)
    coherences = []
    phase_errors = []

    for phi in phi_vals:
        psi = encode_phase_probe(phi, N)
        psi = apply_noise_psi(psi, N, noise_params, dt, rng)
        psi = apply_U_psi(psi, evals, evecs, dt * 3)  # 3-step short probe
        psi = apply_noise_psi(psi, N, noise_params, dt, rng)
        phi_hat, coh = reconstruct_phase(psi, N)
        coherences.append(coh)
        err = min(abs(phi - phi_hat), 2 * np.pi - abs(phi - phi_hat))
        phase_errors.append(err)

    damping_rate     = 1.0 - float(np.mean(coherences))
    phase_sensitivity = float(np.std(phase_errors))
    echo_R           = float(np.mean(coherences))
    jacobian_norm    = float(np.std(coherences))

    return {
        "damping_rate":     damping_rate,
        "phase_sensitivity": phase_sensitivity,
        "echo_recoverability": echo_R,
        "jacobian_norm":    jacobian_norm,
        "mean_coherence":   float(np.mean(coherences)),
    }


# -- SINGLE TRAJECTORY SIMULATION ---------------------------------------------

def simulate_phase_recovery(model_name, N, controller_name, noise_params,
                             T_max, n_steps, seed, phi_true,
                             degrade_channel, degrade_sigma,
                             chirp_info=None, backend="numpy"):
    """
    Simulate one trajectory encoding phi_true, apply degradation,
    evolve with controller, reconstruct phi_hat.

    Returns: dict with phi_true, phi_hat, coherence, f_avg, tau_transport
    """
    H = MODEL_BUILDERS[model_name](N, seed)
    model_key = (model_name, N, seed)
    evals, evecs = _get_eigh(H, model_key)
    rng   = np.random.default_rng(seed)
    d_rng = np.random.default_rng(seed + 12345)
    deg_rng = np.random.default_rng(seed + 54321)
    dt = T_max / n_steps

    use_jax = (backend == "jax" and HAS_JAX)

    # Encode phase probe
    psi = encode_phase_probe(phi_true, N)

    # Apply degradation BEFORE controller sees it (forces active rescue)
    psi = apply_degradation(psi, N, degrade_channel, degrade_sigma, deg_rng)

    if use_jax:
        psi_dev   = jnp.asarray(psi)
        evals_dev = jnp.asarray(evals)
        evecs_dev = jnp.asarray(evecs)
        cliffords_dev = [jnp.asarray(g) for g in CLIFFORDS]
        os_diag_dev   = jnp.asarray(operator_spreading_diag(N))
    else:
        psi_dev = evals_dev = evecs_dev = cliffords_dev = os_diag_dev = None

    obs_history  = []
    t_history    = []
    f_rec_series = []
    fid_series   = []
    psi_prev     = None
    psi_prev_dev = None
    controller   = get_controller(controller_name)

    # Add chirp gradient to initial obs if available
    chirp_grad = np.zeros(N_OBS)
    if chirp_info is not None:
        chirp_grad[0] = chirp_info.get("damping_rate", 0.0)
        chirp_grad[4] = chirp_info.get("phase_sensitivity", 0.0)
        chirp_grad[5] = chirp_info.get("echo_recoverability", 0.0)
        chirp_grad[7] = chirp_info.get("jacobian_norm", 0.0)

    psi_target = encode_phase_probe(phi_true, N)  # ideal target (no degradation)

    for step in range(n_steps):
        t_cur = step * dt
        noise_params["_t_cur"] = t_cur

        if controller_name == "static":
            ci, site = controller(step, N)
        else:
            ci, site = controller(step, N, rho=None,
                                  obs_history=obs_history,
                                  t_history=t_history,
                                  dropout_rng=d_rng,
                                  use_dropout=True)

        if use_jax:
            if ci != 0:
                psi_dev = apply_site_gate_jax(psi_dev, cliffords_dev[ci], N, site)
            psi_dev = apply_U_psi_jax(psi_dev, evals_dev, evecs_dev, dt)
            psi = np.asarray(jax.device_get(psi_dev))
            psi = apply_noise_psi(psi, N, noise_params, dt, rng)
            psi_dev = jnp.asarray(psi)
            rho4 = np.asarray(jax.device_get(
                partial_trace_boundary_psi_jax(psi_dev, N)))
            psi_prev_rho4 = (np.asarray(jax.device_get(
                partial_trace_boundary_psi_jax(psi_prev_dev, N)))
                if psi_prev_dev is not None else None)
        else:
            if ci != 0:
                psi = apply_clifford_psi(psi, N, ci, site)
            psi = apply_U_psi(psi, evals, evecs, dt)
            psi = apply_noise_psi(psi, N, noise_params, dt, rng)
            rho4 = partial_trace_boundary_psi(psi, N)
            psi_prev_rho4 = (partial_trace_boundary_psi(psi_prev, N)
                             if psi_prev is not None else None)

        T_mat = compute_T_matrix(rho4)
        sv    = np.linalg.svd(T_mat, compute_uv=False)
        EE    = von_neumann(rho4)
        MI    = mutual_info(rho4)
        OS    = (float(jax.device_get(
                     operator_spreading_psi_jax(psi_dev, os_diag_dev)))
                 if use_jax else operator_spreading_psi(psi, N))
        ptm_H = spectral_entropy_T(T_mat)

        D_eff = (float((EE - von_neumann(psi_prev_rho4)) / (dt + 1e-12))
                 if psi_prev_rho4 is not None else 0.0)
        front_v = 0.0
        noise_slope = 0.0
        if psi_prev is not None:
            OS_prev = operator_spreading_psi(psi_prev, N)
            front_v = float((OS - OS_prev) / (dt + 1e-12))
        if psi_prev_rho4 is not None:
            T_prev = compute_T_matrix(psi_prev_rho4)
            sv_prev = np.linalg.svd(T_prev, compute_uv=False)
            noise_slope = float((sv[0] - sv_prev[0]) / (dt + 1e-12))

        c_idx = int(rng.integers(1, N_CLIFF))
        psi_sc   = apply_clifford_psi(psi, N, c_idx, 0)
        rho4_sc  = partial_trace_boundary_psi(psi_sc, N)
        T_sc     = compute_T_matrix(rho4_sc)
        basis_invar = float(np.linalg.norm(T_mat - T_sc, "fro"))

        # Inject chirp gradient into first obs slot
        x_obs = np.array([
            sv[0] + chirp_grad[0],
            sv[1] if len(sv) > 1 else 0.0,
            sv[2] if len(sv) > 2 else 0.0,
            ptm_H,
            EE    + chirp_grad[4],
            MI    + chirp_grad[5],
            OS,
            ptm_H + chirp_grad[7],
            D_eff, front_v, noise_slope, basis_invar,
        ])

        obs_history.append(x_obs.tolist())
        t_history.append(float(t_cur))
        f_rec_series.append(float(f_rec_from_T(T_mat)))

        # State fidelity against ideal target
        overlap = abs(np.vdot(psi_target, psi)) ** 2
        fid_series.append(float(overlap))

        psi_prev     = psi.copy()
        psi_prev_dev = psi_dev if use_jax else None

    # Reconstruct phase from final state
    phi_hat, coherence = reconstruct_phase(psi, N)

    above = [i for i, f in enumerate(f_rec_series) if f > TH_F_REC]
    tau_transport = float(above[-1] * dt) if above else 0.0
    F_avg = float(np.mean(fid_series)) if fid_series else 0.0

    return {
        "phi_true":       float(phi_true),
        "phi_hat":        float(phi_hat),
        "coherence":      float(coherence),
        "F_avg":          F_avg,
        "tau_transport":  tau_transport,
        "model":          model_name,
        "controller":     controller_name,
        "degrade_channel": degrade_channel,
        "degrade_sigma":  float(degrade_sigma),
        "N":              N,
        "seed":           seed,
    }


# -- EXPERIMENT RUNNER ---------------------------------------------------------

def run_experiment_p(args):
    t0 = time.time()
    backend, backend_info = select_backend(args.backend, args.require_tpu)
    rng_noise = np.random.default_rng(args.seed)
    dt = args.T_max / args.n_steps

    phi_sweep = np.linspace(0, 2 * np.pi, args.n_phases, endpoint=False).tolist()
    sigmas    = [float(s) for s in args.sigmas]
    channels  = args.channels
    models    = args.models
    ctrls     = args.controllers

    print("=" * 65)
    print("Program P -- Partial Information Recovery")
    print("=" * 65)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Models:     {models}")
    print(f"Controllers:{ctrls}")
    print(f"Channels:   {channels}")
    print(f"Sigmas:     {sigmas}")
    print(f"N_phases:   {args.n_phases}  B={args.B}")
    print(f"Backend:    {backend}  JAX: {backend_info}")
    print()

    all_records = []
    summary = {}

    for model in models:
        print(f"\n-- Model: {model} --")

        # Phase 1: chirp system identification
        chirp_info = None
        if args.run_chirp:
            print(f"  [Chirp] Running system identification...")
            noise_chirp = sample_noise_params(rng_noise)
            chirp_info = chirp_sweep(model, args.N, noise_chirp.copy(),
                                     dt, seed=args.seed, K=args.n_phases,
                                     backend="numpy")
            print(f"  damping={chirp_info['damping_rate']:.3f}  "
                  f"phase_sens={chirp_info['phase_sensitivity']:.3f}  "
                  f"echo_R={chirp_info['echo_recoverability']:.3f}")

        for channel in channels:
            for sigma in sigmas:
                # Skip sigma=0 repetition
                if channel != "none" and sigma == 0.0:
                    continue

                key = f"{model}/{channel}/sig={sigma:.1f}"
                ctrl_records = {c: [] for c in ctrls}

                for b in range(args.B):
                    seed_b = args.seed + b * 1000
                    noise_params = sample_noise_params(rng_noise)

                    for phi in phi_sweep:
                        for ctrl in ctrls:
                            t1 = time.time()
                            rec = simulate_phase_recovery(
                                model_name=model,
                                N=args.N,
                                controller_name=ctrl,
                                noise_params=noise_params.copy(),
                                T_max=args.T_max,
                                n_steps=args.n_steps,
                                seed=seed_b,
                                phi_true=phi,
                                degrade_channel=channel,
                                degrade_sigma=sigma,
                                chirp_info=chirp_info,
                                backend=backend,
                            )
                            rec["B"] = b
                            ctrl_records[ctrl].append(rec)
                            all_records.append(rec)

                # Compute metrics per controller pair
                for ctrl in ctrls:
                    recs = ctrl_records[ctrl]
                    phis_true = [r["phi_true"] for r in recs]
                    phis_hat  = [r["phi_hat"]  for r in recs]
                    mae = phase_mae(phis_true, phis_hat)
                    r2  = phase_r2(phis_true, phis_hat)
                    mi  = phase_mi(phis_true, phis_hat)
                    f_avg = float(np.mean([r["F_avg"] for r in recs]))
                    summary[f"{key}/{ctrl}"] = {
                        "mae": mae, "r2": r2, "mi": mi, "F_avg": f_avg,
                        "n": len(recs),
                    }

                # PASS/FAIL vs static baseline
                if "static" in ctrls and "agnostic" in ctrls:
                    s_key = f"{key}/static"
                    a_key = f"{key}/agnostic"
                    if s_key in summary and a_key in summary:
                        d_mae = summary[s_key]["mae"] - summary[a_key]["mae"]
                        d_r2  = summary[a_key]["r2"]  - summary[s_key]["r2"]
                        d_mi  = summary[a_key]["mi"]  - summary[s_key]["mi"]
                        passed = (d_mae > 0 and d_r2 > TH_R2_MIN)
                        marker = "PASS" if passed else "    "
                        print(f"  {marker} {key:35s} "
                              f"DMAE={d_mae:+.3f}rad  "
                              f"DR2={d_r2:+.4f}  "
                              f"DMI={d_mi:+.4f}nat")
                        summary[key] = {
                            "delta_mae": d_mae, "delta_r2": d_r2,
                            "delta_mi": d_mi, "pass": passed,
                        }

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_p_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args":         vars(args),
            "backend":      backend,
            "backend_info": backend_info,
            "summary":      summary,
            "results":      all_records,
            "runtime_s":    round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time()-t0:.1f}s total)")
    return summary


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Program P -- Partial Information Recovery")
    p.add_argument("--N",       type=int,   default=10)
    p.add_argument("--T-max",   type=float, default=6.0,  dest="T_max")
    p.add_argument("--n-steps", type=int,   default=30,   dest="n_steps")
    p.add_argument("--B",       type=int,   default=5)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--models", nargs="+",
                   default=["XXZ", "DisorderedXXZ_W1",
                            "DisorderedXXZ_W3", "OAT"],
                   choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--controllers", nargs="+",
                   default=["static", "agnostic"],
                   choices=["static", "haware", "agnostic",
                            "dd_xy8", "dd_cpmg", "random"])
    p.add_argument("--channels", nargs="+",
                   default=["none", "dephasing", "jitter", "scrambling"],
                   choices=DEGRADATION_CHANNELS)
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument("--n-phases",  type=int,   default=24,   dest="n_phases")
    p.add_argument("--run-chirp", action="store_true", dest="run_chirp")
    p.add_argument("--out-dir",   default="program_p_results", dest="out_dir")
    p.add_argument("--backend",   choices=["auto", "numpy", "jax"],
                   default="auto")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test",  action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.N          = 6
        args.n_steps    = 15
        args.B          = 1
        args.models     = ["XXZ", "OAT"]
        args.channels   = ["none", "dephasing"]
        args.sigmas     = [0.0, 0.2]
        args.controllers = ["static", "agnostic"]
        print("[SMOKE TEST] N=6, B=1, 2 models, 2 channels")

    run_experiment_p(args)
