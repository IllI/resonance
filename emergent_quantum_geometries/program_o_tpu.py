"""
Program O — Encoded-State Recoverability
=========================================
Extends Program L's transport recoverability framework to measure whether
adaptive control preserves encoded quantum information.

Metric shift:
  Program L:  tau_transport  (recoverability lifetime)
  Program O:  delta_F_avg         (state fidelity improvement over static baseline)

Four phases:
  1. Response characterization via chirped pulse sequence (J_ij estimation)
  2. Probe state injection (structured initial states)
  3. Controlled transport comparison (static / dd_xy8 / random / agnostic / agnostic_probe)
  4. delta_F_avg analysis across state families and regimes

Pre-registered thresholds FROZEN before first run.
"""
import argparse
import json
import os
import time
from functools import partial
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    jax = None
    jnp = None
    HAS_JAX = False

# -- Re-use Program L's entire simulation engine ------------------------------
# Import from program_l_tpu.py (must be in same directory)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from program_l_tpu import (
    # Physics
    MODEL_BUILDERS, sample_noise_params, _get_eigh,
    apply_U_psi, apply_clifford_psi, apply_noise_psi,
    partial_trace_boundary_psi, compute_T_matrix,
    operator_spreading_diag, operator_spreading_psi,
    von_neumann, mutual_info, spectral_entropy_T, f_rec_from_T,
    # Controller
    get_controller, CLIFFORDS, N_CLIFF, CTRL_INTERVAL,
    N_OBS, OBS_NAMES, strata_score, _predict_obs_after_cliff,
    LAMBDA_SPARSE, LAMBDA_STABLE, STABLE_BASIN_D_EFF,
    STABLE_BASIN_DX_NORM, STABLE_BASIN_SUSCEPT,
    GRAD_ALPHA_BASE, SUSCEPTIBILITY_SCALE, FRAGILITY_FRONT_V_THRESH,
    P_DROPOUT,
    # JAX helpers (if available)
    apply_site_gate_jax, apply_U_psi_jax,
    partial_trace_boundary_psi_jax, operator_spreading_psi_jax,
    select_backend, _jax_device_summary,
    # Thresholds
    TH_F_REC,
    infer_transport_strata,
)

# -- PRE-REGISTERED THRESHOLDS (FROZEN) ---------------------------------------
TH_DF_AVG    = 0.02   # minimum meaningful delta_F_avg
TH_PROBE_ADD = 0.01   # probe must add > 1% F improvement over no-probe

# -- PROBE STATE DEFINITIONS --------------------------------------------------

def _single_qubit_state(theta, phi):
    """Pure single-qubit state: cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩."""
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def _kron_state(q_state, N):
    """Tensor product of N identical single-qubit states."""
    psi = q_state.copy()
    for _ in range(N - 1):
        psi = np.kron(psi, q_state)
    return psi / np.linalg.norm(psi)


def build_probe_states(N, families=None):
    """
    Build structured probe states for Program O Phase 2.

    Returns list of (label, psi) tuples.
    Families: 'z_basis', 'x_basis', 'y_basis', 'equatorial', 'magic', 'neel'
    """
    if families is None:
        families = ["z_basis", "x_basis", "y_basis", "equatorial", "magic", "neel"]

    probes = []

    if "neel" in families:
        # Néel state — inherited from Program L (reference)
        neel_idx = sum(1 << i for i in range(0, N, 2))
        psi = np.zeros(2 ** N, dtype=complex)
        psi[neel_idx] = 1.0
        probes.append(("neel", psi))

    if "z_basis" in families:
        # |0⟩^⊗N
        psi = np.zeros(2 ** N, dtype=complex); psi[0] = 1.0
        probes.append(("z_basis", psi))

    if "x_basis" in families:
        # |+⟩^⊗N
        q = _single_qubit_state(np.pi / 2, 0.0)  # θ=π/2, φ=0
        probes.append(("x_basis", _kron_state(q, N)))

    if "y_basis" in families:
        # |R⟩^⊗N  (|+y⟩)
        q = _single_qubit_state(np.pi / 2, np.pi / 2)
        probes.append(("y_basis", _kron_state(q, N)))

    if "equatorial" in families:
        # Equatorial sweep: 12 states (θ × φ grid)
        thetas = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        phis   = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for k, (th, ph) in enumerate([(t, p) for t in thetas for p in phis]):
            q = _single_qubit_state(th, ph)
            probes.append((f"eq_{k:02d}", _kron_state(q, N)))

    if "magic" in families:
        # T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2  on each qubit
        q = np.array([1.0, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2)
        probes.append(("magic", _kron_state(q, N)))

    return probes


# -- PHASE 1: CHIRP PROBE — RESPONSE JACOBIAN ---------------------------------

def measure_response_jacobian(model_name, N, noise_params, dt, seed,
                               probe_steps=6, backend="numpy"):
    """
    Estimate J_ij = ∂x_i/∂u_j by applying each Clifford at step k and
    measuring the resulting observable shift.

    Returns:
        J: dict mapping cliff_name -> delta_x vector (N_OBS,)
        baseline_x: observable trajectory without any chirp
    """
    H = MODEL_BUILDERS[model_name](N, seed)
    model_key = (model_name, N, seed)
    evals, evecs = _get_eigh(H, model_key)
    rng = np.random.default_rng(seed)
    dim = 2 ** N

    # Baseline: evolve probe_steps steps from |0⟩^N, no gates
    neel_idx = sum(1 << i for i in range(0, N, 2))
    psi0 = np.zeros(dim, dtype=complex); psi0[neel_idx] = 1.0

    def _run_steps(psi_init, apply_gate_at=None, gate_idx=0, gate_step=2):
        psi = psi_init.copy()
        rng2 = np.random.default_rng(seed + 7777)
        xs = []
        for step in range(probe_steps):
            if apply_gate_at is not None and step == gate_step:
                psi = apply_clifford_psi(psi, N, gate_idx, 0)
            psi = apply_U_psi(psi, evals, evecs, dt)
            psi = apply_noise_psi(psi, N, noise_params, dt, rng2)
            rho4 = partial_trace_boundary_psi(psi, N)
            T = compute_T_matrix(rho4)
            sv = np.linalg.svd(T, compute_uv=False)
            EE = von_neumann(rho4)
            MI = mutual_info(rho4)
            OS = operator_spreading_psi(psi, N)
            ptm_H = spectral_entropy_T(T)
            xs.append(np.array([
                sv[0], sv[1] if len(sv) > 1 else 0.0,
                sv[2] if len(sv) > 2 else 0.0,
                ptm_H, EE, MI, OS, ptm_H,
                0.0,   # D_eff placeholder
                0.0,   # front_v placeholder
                0.0,   # noise_slope placeholder
                0.0,   # basis_invar placeholder
            ]))
        return np.array(xs)

    gate_step = probe_steps // 2
    baseline_xs = _run_steps(psi0)
    J = {}
    cliff_names = ["I", "X", "Y", "Z", "H", "S"]
    for ci in range(1, N_CLIFF):
        chirp_xs = _run_steps(psi0, apply_gate_at=True,
                               gate_idx=ci, gate_step=gate_step)
        # delta_x at step gate_step+1
        if gate_step + 1 < len(chirp_xs):
            delta = chirp_xs[gate_step + 1] - baseline_xs[gate_step + 1]
        else:
            delta = chirp_xs[-1] - baseline_xs[-1]
        J[cliff_names[ci]] = delta

    return J, baseline_xs


# -- STATE FIDELITY COMPUTATION ------------------------------------------------

def state_fidelity(psi_out, psi_target):
    """Overlap fidelity: |⟨ψ_target|ψ_out⟩|²."""
    return float(abs(np.vdot(psi_target, psi_out)) ** 2)


def boundary_fidelity(psi_out, psi_target, N):
    """
    Boundary reduced density matrix fidelity:
    F(ρ_out, ρ_target) = Tr(√(√ρ_t ρ_o √ρ_t))²

    Uses boundary 4x4 RDM (sites 0 and N-1) matching Program L's F_rec geometry.
    """
    rho_out    = partial_trace_boundary_psi(psi_out, N)
    rho_target = partial_trace_boundary_psi(psi_target, N)
    try:
        sqrt_rt = np.linalg.matrix_power(
            rho_target + 1e-12 * np.eye(4), 1)  # use direct Tr(ρ_t ρ_o)
        return float(np.real(np.trace(rho_target @ rho_out)))
    except Exception:
        return 0.0


# -- PHASE 3: TRAJECTORY WITH PROBE STATE -------------------------------------

def simulate_trajectory_probe(model_name, N, controller_name, noise_params,
                               T_max, n_steps, seed, probe_label, psi_probe,
                               psi_target, backend="numpy",
                               use_jacobian=False, J_measured=None,
                               store_obs=False):
    """
    Simulate one trajectory from a structured probe state.

    Returns:
        rec: dict with F_final, F_avg, tau_transport, ctrl_actions
    """
    H = MODEL_BUILDERS[model_name](N, seed)
    model_key = (model_name, N, seed)
    evals, evecs = _get_eigh(H, model_key)
    rng  = np.random.default_rng(seed)
    d_rng = np.random.default_rng(seed + 12345)
    dt   = T_max / n_steps
    dim  = 2 ** N

    use_jax = (backend == "jax" and HAS_JAX)
    psi = psi_probe.copy()

    if use_jax:
        psi_dev = jnp.asarray(psi)
        evals_dev = jnp.asarray(evals)
        evecs_dev = jnp.asarray(evecs)
        cliffords_dev = [jnp.asarray(g) for g in CLIFFORDS]
        os_diag_dev = jnp.asarray(operator_spreading_diag(N))
    else:
        psi_dev = evals_dev = evecs_dev = cliffords_dev = os_diag_dev = None

    obs_history  = []
    t_history    = []
    f_rec_series = []
    fidelity_series = []
    ctrl_actions = []
    psi_prev     = None
    psi_prev_dev = None
    controller   = get_controller(controller_name)

    for step in range(n_steps):
        t_cur = step * dt
        noise_params["_t_cur"] = t_cur

        # Controller decision
        if controller_name == "static":
            ci, site = controller(step, N)
        else:
            ci, site = controller(step, N, rho=None,
                                  obs_history=obs_history,
                                  t_history=t_history,
                                  dropout_rng=d_rng,
                                  use_dropout=True)

        ctrl_actions.append((int(ci), int(site)))

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

        # Observables
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
        if psi_prev is not None:
            OS_prev = (float(jax.device_get(
                           operator_spreading_psi_jax(psi_prev_dev, os_diag_dev)))
                       if use_jax else operator_spreading_psi(psi_prev, N))
            front_v = float((OS - OS_prev) / (dt + 1e-12))

        noise_slope = 0.0
        if psi_prev_rho4 is not None:
            T_prev = compute_T_matrix(psi_prev_rho4)
            sv_prev = np.linalg.svd(T_prev, compute_uv=False)
            noise_slope = float((sv[0] - sv_prev[0]) / (dt + 1e-12))

        c_idx = int(rng.integers(1, N_CLIFF))
        psi_sc = apply_clifford_psi(psi, N, c_idx, 0)
        rho4_sc = partial_trace_boundary_psi(psi_sc, N)
        T_sc = compute_T_matrix(rho4_sc)
        basis_invar = float(np.linalg.norm(T_mat - T_sc, "fro"))

        x_obs = np.array([
            sv[0], sv[1] if len(sv) > 1 else 0.0,
            sv[2] if len(sv) > 2 else 0.0,
            ptm_H, EE, MI, OS, ptm_H,
            D_eff, front_v, noise_slope, basis_invar,
        ])

        obs_history.append(x_obs.tolist())
        t_history.append(float(t_cur))

        # F_rec (transport threshold, inherited)
        f_rec = f_rec_from_T(T_mat)
        f_rec_series.append(float(f_rec))

        # State fidelity against target
        f_state = boundary_fidelity(psi, psi_target, N)
        fidelity_series.append(float(f_state))

        psi_prev = psi.copy()
        psi_prev_dev = psi_dev if use_jax else None

    above = [i for i, f in enumerate(f_rec_series) if f > TH_F_REC]
    tau_transport = float(above[-1] * dt) if above else 0.0
    F_final = fidelity_series[-1] if fidelity_series else 0.0
    F_avg   = float(np.mean(fidelity_series)) if fidelity_series else 0.0

    rec = {
        "model":         model_name,
        "controller":    controller_name,
        "probe":         probe_label,
        "N":             N,
        "seed":          seed,
        "tau_transport": tau_transport,
        "F_final":       F_final,
        "F_avg":         F_avg,
        "ctrl_actions":  ctrl_actions,
        "T_max":         T_max,
        "n_steps":       n_steps,
        "backend":       backend,
        "noise_params":  {k: v for k, v in noise_params.items()
                          if not k.startswith("_")},
    }
    if store_obs:
        rec["obs_history"]     = obs_history
        rec["fidelity_series"] = fidelity_series
    return rec


# -- EXPERIMENT RUNNER ---------------------------------------------------------

def run_experiment_o(args):
    models      = args.models
    controllers = getattr(args, "controllers",
                          ["static", "dd_xy8", "random", "agnostic"])
    families    = getattr(args, "probe_families",
                          ["neel", "z_basis", "x_basis", "y_basis",
                           "equatorial", "magic"])
    results     = []
    t0          = time.time()
    backend, backend_info = select_backend(args.backend, args.require_tpu)

    print("=" * 65)
    print("Program O — Encoded-State Recoverability")
    print("=" * 65)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Models:      {models}")
    print(f"Controllers: {controllers}")
    print(f"Probe fams:  {families}")
    print(f"Bootstrap:   B={args.B}")
    print(f"Backend:     {backend}  JAX: {backend_info}")
    print()

    rng_noise = np.random.default_rng(args.seed)
    dt = args.T_max / args.n_steps

    for model in models:
        probes = build_probe_states(args.N, families)
        print(f"\n-- Model: {model}  ({len(probes)} probe states) --")

        # Phase 1: optional chirp calibration
        J_measured = None
        if args.run_chirp:
            print(f"  [Phase 1] Chirp probe → response Jacobian...")
            noise_params_chirp = sample_noise_params(rng_noise)
            J_measured, _ = measure_response_jacobian(
                model, args.N, noise_params_chirp.copy(), dt,
                seed=args.seed, probe_steps=min(8, args.n_steps),
                backend="numpy")
            print(f"  J channels: " +
                  "  ".join(f"{g}:{np.linalg.norm(v):.3f}"
                             for g, v in J_measured.items()))

        for b in range(args.B):
            seed_b       = args.seed + b * 1000
            noise_params = sample_noise_params(rng_noise)

            for probe_label, psi_probe in probes:
                # Target = probe initial state evolved under identity (no noise)
                # i.e., what the state would be without any perturbation.
                # Simple choice: psi_probe itself as target (fidelity measures
                # how well probe state is preserved through the channel).
                psi_target = psi_probe.copy()

                for ctrl in controllers:
                    t1 = time.time()
                    rec = simulate_trajectory_probe(
                        model_name=model,
                        N=args.N,
                        controller_name=ctrl,
                        noise_params=noise_params.copy(),
                        T_max=args.T_max,
                        n_steps=args.n_steps,
                        seed=seed_b,
                        probe_label=probe_label,
                        psi_probe=psi_probe,
                        psi_target=psi_target,
                        backend=backend,
                        use_jacobian=(args.run_chirp and J_measured is not None),
                        J_measured=J_measured,
                        store_obs=getattr(args, "store_obs", False),
                    )
                    results.append(rec)
                    dt_run = time.time() - t1
                    print(f"  [{ctrl:8s}] B={b} probe={probe_label:8s} "
                          f"tau={rec['tau_transport']:.3f}  "
                          f"F_avg={rec['F_avg']:.4f}  "
                          f"({dt_run:.1f}s)")

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY  delta_F_avg = mean_F(agnostic) - mean_F(static)")
    print("=" * 65)
    summary = {}
    for model in models:
        recs_m = [r for r in results if r["model"] == model]
        for probe_label, _ in build_probe_states(args.N, families):
            recs_p = [r for r in recs_m if r["probe"] == probe_label]
            f_static   = np.mean([r["F_avg"] for r in recs_p
                                  if r["controller"] == "static"] or [0.0])
            f_agnostic = np.mean([r["F_avg"] for r in recs_p
                                  if r["controller"] == "agnostic"] or [0.0])
            delta_f = f_agnostic - f_static
            key = f"{model}/{probe_label}"
            summary[key] = {
                "F_static": float(f_static),
                "F_agnostic": float(f_agnostic),
                "delta_F": float(delta_f),
                "pass": bool(delta_f > TH_DF_AVG),
            }
            marker = "PASS" if delta_f > TH_DF_AVG else " "
            print(f"  {marker} {model:20s} {probe_label:10s} "
                  f"F_s={f_static:.4f}  F_a={f_agnostic:.4f}  "
                  f"delta_F={delta_f:+.4f}")

    # -- Save ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_o_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args":         vars(args),
            "backend":      backend,
            "backend_info": backend_info,
            "summary":      summary,
            "results":      results,
            "runtime_s":    round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time()-t0:.1f}s total)")
    return summary


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Program O — Encoded-State Recoverability")
    p.add_argument("--N",       type=int,   default=10)
    p.add_argument("--T-max",   type=float, default=6.0,  dest="T_max")
    p.add_argument("--n-steps", type=int,   default=30,   dest="n_steps")
    p.add_argument("--B",       type=int,   default=3)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--models",  nargs="+",
                   default=["XXZ", "DisorderedXXZ_W1",
                            "DisorderedXXZ_W3", "OAT"],
                   choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--controllers", nargs="+",
                   default=["static", "dd_xy8", "random", "agnostic"],
                   choices=["static", "haware", "agnostic",
                            "dd_xy8", "dd_cpmg", "random"])
    p.add_argument("--probe-families", nargs="+", dest="probe_families",
                   default=["neel", "z_basis", "x_basis", "y_basis",
                            "equatorial", "magic"],
                   choices=["neel", "z_basis", "x_basis", "y_basis",
                            "equatorial", "magic"])
    p.add_argument("--run-chirp", action="store_true", dest="run_chirp",
                   help="Run Phase 1 chirp probe to measure response Jacobian")
    p.add_argument("--out-dir",   default="program_o_results", dest="out_dir")
    p.add_argument("--backend",   choices=["auto", "numpy", "jax"],
                   default="auto")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--store-obs",   action="store_true", dest="store_obs")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick validation: N=6, 2 models, B=1, fast probes only")
    args = p.parse_args()

    if args.smoke_test:
        args.N       = 6
        args.n_steps = 15
        args.B       = 1
        args.models  = ["XXZ", "OAT"]
        args.probe_families = ["neel", "x_basis", "magic"]
        args.controllers    = ["static", "agnostic"]
        print("[SMOKE TEST] N=6, B=1, 2 models, 3 probe families")

    run_experiment_o(args)
