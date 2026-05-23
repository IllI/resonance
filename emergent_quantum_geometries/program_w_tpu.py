"""
Program W -- Stability Boundary Phase Diagram
=============================================
A rigorous quantum trajectory simulation mapping the transition boundary 
between restraint-optimal and intervention-beneficial control regimes.
Fixes:
- Seed coupling across controllers (paired initial states & paired trajectories)
- Threshold scaling via velocity-based triggering
- Complete trigger activity and gate intervention reporting

Features:
- Distinguishability via trace distance of Bob's reduced density matrix under a 0.1 rad theta perturbation
- Echo-recovery gain via Alice Z-echo post-selection
- Full phase boundary classification
- Diagnostic mode for visualizing trigger instability scales
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from functools import partial

# Check JAX
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import (
    _get_eigh, CLIFFORDS, N_CLIFF,
    select_backend,
)
from program_t_tpu import (
    build_grid_hamiltonian,
    get_bob_candidates,
)
from program_v_tpu import (
    apply_gate_psi,
    apply_stochastic_dephasing,
    apply_stochastic_amplitude_damping,
    make_initial_state,
    get_causal_observables_psi,
    run_probe_stage_v,
    compute_reduced_rho_AB,
    compute_phase_sensitive_metrics,
    run_step_psi,
)

# ---------------------------------------------------------------------------
# Program W Controllers
# ---------------------------------------------------------------------------

def ctrl_static_w(step):
    return jax.lax.cond(jnp.logical_or(step == 0, step % 8 == 0), 
                        lambda: jnp.where(step == 0, jnp.array((1, 0)), jnp.array((3, 0))), 
                        lambda: jnp.array((0, 0)))

def ctrl_dd_w(step):
    seq = jnp.array([1, 1, 2, 2])
    trigger = (step % 4 == 0)
    ci = seq[(step // 4) % 4]
    return jax.lax.cond(trigger, lambda: jnp.array((ci, 0)), lambda: jnp.array((0, 0)))

def ctrl_sparse_velocity_v(step, obs_hist, cooldown, tau_thresh, S_proj):
    # Velocity-based error: O(t) - O(t-dt)
    err_vec = obs_hist[2] - obs_hist[1]
    proj_err = jnp.dot(S_proj, err_vec)
    instability = jnp.sum(proj_err ** 2)
    trigger = jnp.logical_and(instability > tau_thresh, cooldown == 0)
    trigger = jnp.logical_and(trigger, step > 2)
    return jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))

def ctrl_sparse_accel_v(step, obs_hist, cooldown, tau_thresh, S_proj):
    # Acceleration-based error (V2 original): O(t) - 2O(t-dt) + O(t-2dt)
    pred = obs_hist[1] + (obs_hist[1] - obs_hist[0])
    err_vec = obs_hist[2] - pred
    proj_err = jnp.dot(S_proj, err_vec)
    instability = jnp.sum(proj_err ** 2)
    trigger = jnp.logical_and(instability > tau_thresh, cooldown == 0)
    trigger = jnp.logical_and(trigger, step > 2)
    return jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))

# ---------------------------------------------------------------------------
# JAX Trajectory Runner
# ---------------------------------------------------------------------------

def run_trajectory_w(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                     ctrl_name, tau_thresh, S_proj, bob_site, Haar_c0, Haar_c1, rng_key):
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    obs0 = get_causal_observables_psi(psi0, N)
    
    obs_hist0 = jnp.stack([obs0, obs0, obs0], axis=0)
    carry0 = (psi0, obs_hist0, jnp.int32(0), jnp.int32(0), rng_key)

    def scan_step(carry, step_idx):
        psi, obs_hist, cooldown, n_gates, key = carry
        key_next, _, key_step = jax.random.split(key, 3)
        
        if ctrl_name == "static":
            gate_info = ctrl_static_w(step_idx)
        elif ctrl_name == "dd":
            gate_info = ctrl_dd_w(step_idx)
        elif ctrl_name == "sparse_velocity":
            gate_info = ctrl_sparse_velocity_v(step_idx, obs_hist, cooldown, tau_thresh, S_proj)
        elif ctrl_name == "sparse_accel":
            gate_info = ctrl_sparse_accel_v(step_idx, obs_hist, cooldown, tau_thresh, S_proj)
        else:
            gate_info = jnp.array((0, 0))
            
        ci = gate_info[0]
        
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        
        cooldown_next = jax.lax.cond(ci > 0, lambda: jnp.int32(4), lambda: jnp.maximum(0, cooldown - 1))
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        
        obs_curr = get_causal_observables_psi(psi_next, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr], axis=0)
        
        return (psi_next, obs_hist_next, cooldown_next, n_gates_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, steps)
    
    psi_final, _, _, n_gates_total, _ = final_carry
    return psi_final, n_gates_total

# ---------------------------------------------------------------------------
# Metric Utilities & Probe
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(2, 4))
def run_probe_stage_w(evals, evecs, N, dt, bob_site):
    dim = 2 ** N
    eps_angles = jnp.array([jnp.pi/8, jnp.pi/4, jnp.pi/2], dtype=jnp.float32)
    
    def evolve_psi(psi, t):
        p_e = evecs.conj().T @ psi
        p_e = p_e * jnp.exp(-1j * evals * t)
        return evecs @ p_e

    # Prepare superposition reference state on Alice sites 0 and 1
    psi_ref = jnp.zeros(dim, dtype=jnp.complex64).at[0].set(1.0)
    H_gate = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)
    psi_ref = apply_gate_psi(psi_ref, H_gate, 0, N)
    psi_ref = apply_gate_psi(psi_ref, H_gate, 1, N)
    
    O_ref = get_causal_observables_psi(evolve_psi(psi_ref, dt), N, readout_error=0.0)
    
    def compute_ai(site):
        def _scan_eps(carry, eps):
            rz = jnp.array([[jnp.exp(-1j*eps/2), 0],
                            [0, jnp.exp(1j*eps/2)]], dtype=jnp.complex64)
            psi_p = apply_gate_psi(psi_ref, rz, site, N)
            O_p = get_causal_observables_psi(evolve_psi(psi_p, dt), N, readout_error=0.0)
            diff = jnp.abs(O_p - O_ref) / (eps + 1e-9)
            return None, diff
        _, diffs = jax.lax.scan(_scan_eps, None, eps_angles)
        return jnp.mean(diffs, axis=0)
        
    return jnp.stack([compute_ai(0), compute_ai(1)], axis=0)

@jax.jit
def trace_distance_2x2(rho1, rho2):
    diff = rho1 - rho2
    # Trace distance for 2x2 hermitian matrix is sum of absolute eigenvalues / 2
    vals = jnp.linalg.eigvalsh(diff)
    return 0.5 * jnp.sum(jnp.abs(vals))

# ---------------------------------------------------------------------------
# Diagnostic Logic
# ---------------------------------------------------------------------------

def run_diagnostic(args):
    """
    Diagnostic mode that prints instability metrics (velocity and acceleration)
    step-by-step for a single noisy trajectory.
    """
    print("\n" + "="*80)
    print("PROGRAM W -- INSTABILITY TRIGGER DIAGNOSTIC")
    print("="*80)
    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    dt = args.T_max / args.n_steps
    
    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=args.j_std_sweep[0], seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, ("Diagnostic", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)
    
    bob_cands = get_bob_candidates(Lx, Ly)
    bob_site, bob_dist = bob_cands["medium"][0]
    
    print(f"Disorder strength: {args.j_std_sweep[0]:.2f} | T2: {args.t2_sweep[0]:.1f}")
    print(f"Bob site: {bob_site} | Distance: {bob_dist}")
    
    S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site)
    
    # Generate Haar state
    Haar_c0 = jnp.complex64(np.cos(np.pi/4))
    Haar_c1 = jnp.complex64(np.sin(np.pi/4))
    
    psi = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    obs0 = get_causal_observables_psi(psi, N)
    obs_hist = [obs0, obs0, obs0]
    
    p1 = float(1.0 - np.exp(-dt / 20.0))
    p2 = float(1.0 - np.exp(-dt / args.t2_sweep[0]))
    p_cross = args.p_cross_sweep[0]
    
    key = jax.random.PRNGKey(args.seed)
    
    print("\nStep |  Instability (Velocity)  |  Instability (Acceleration)")
    print("-" * 65)
    
    for step in range(args.n_steps):
        key, key_step = jax.random.split(key)
        # Evolve one step with NO gates (free evolution)
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, jnp.array((0,0)), key_step)
        obs_curr = get_causal_observables_psi(psi, N)
        obs_hist = obs_hist[1:] + [obs_curr]
        
        # Calculate velocity metric
        err_vel = obs_hist[2] - obs_hist[1]
        proj_vel = jnp.dot(S_proj, err_vel)
        instability_vel = float(jnp.sum(proj_vel ** 2))
        
        # Calculate accel metric
        pred = obs_hist[1] + (obs_hist[1] - obs_hist[0])
        err_acc = obs_hist[2] - pred
        proj_acc = jnp.dot(S_proj, err_acc)
        instability_acc = float(jnp.sum(proj_acc ** 2))
        
        print(f" {step:2d}  |       {instability_vel:.8f}       |         {instability_acc:.8f}  |  obs: {obs_curr}")
    print("=" * 80 + "\n")

# ---------------------------------------------------------------------------
# Main Sweep Coordinator for Program W Phase Diagram
# ---------------------------------------------------------------------------

def run_experiment_w(args):
    if args.diagnose:
        run_diagnostic(args)
        return

    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 80)
    print("Program W -- Stability Boundary Phase Diagram (JAX)")
    print("=" * 80)
    
    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    print(f"Grid: {Lx}x{Ly} = {N} qubits  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Number of Trajectories per Batch: {args.n_trajectories}  Batches: {args.B}")
    
    dt = args.T_max / args.n_steps
    T1_phys = 20.0
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    
    bob_cands = get_bob_candidates(Lx, Ly)
    summary = {}
    phase_boundary = {}
    all_records = []

    if HAS_JAX:
        run_trajectories_vmapped = jax.jit(
            jax.vmap(run_trajectory_w, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)),
            static_argnums=(2, 4, 8)
        )
    else:
        print("[ERROR] JAX is required to run Program W.")
        sys.exit(1)

    Z_gate = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

    # Main Sweeps
    for J_std in args.j_std_sweep:
        print(f"\n== Disorder Strength J_std: {J_std:.2f} ==")
        H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
        evals_np, evecs_np = _get_eigh(H, (f"GridXXZ_J{J_std:.2f}", N, args.seed))
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for T2 in args.t2_sweep:
            p2 = float(1.0 - np.exp(-dt / T2))
            print(f"  -- Dephasing T2: {T2:.1f} (p2={p2:.4f}) --")

            for p_cross in args.p_cross_sweep:
                print(f"    -- Crosstalk p_cross: {p_cross:.3f} --")

                for dist_class, bob_list in bob_cands.items():
                    if not bob_list: continue
                    bob_site, bob_dist = bob_list[len(bob_list)//2]
                    print(f"      -- Distance: {dist_class} (Bob=site {bob_site}, dist={bob_dist}) --")
                    
                    S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site)

                    # Store results for this specific physical parameter group
                    # to build the phase boundary later
                    group_results = {}

                    # Loop over controllers and sweeps
                    for ctrl_name in args.controllers:
                        is_adaptive = (ctrl_name in ["sparse_velocity", "sparse_accel"])
                        current_taus = args.tau_sweep if is_adaptive else [0.0]

                        for tau in current_taus:
                            ctrl_records = []
                            
                            # Batches
                            for b in range(args.B):
                                # Fix Bug 1: One shared, completely unified batch random generator
                                batch_seed = args.seed + b * 1000
                                b_rng = np.random.default_rng(batch_seed)
                                
                                # Generate state 1 (Haar state)
                                u = b_rng.uniform(0, 1)
                                v = b_rng.uniform(0, 1)
                                theta = np.arccos(2.0 * u - 1.0)
                                phi = 2.0 * np.pi * v
                                Haar_c0 = np.cos(theta / 2.0)
                                Haar_c1 = np.exp(1j * phi) * np.sin(theta / 2.0)
                                
                                # Generate state 2 (Perturbed state for distinguishability)
                                theta_perturbed = theta + 0.1
                                Haar_c0_p = np.cos(theta_perturbed / 2.0)
                                Haar_c1_p = np.exp(1j * phi) * np.sin(theta_perturbed / 2.0)

                                # Paired trajectory JAX keys
                                key_seq = jax.random.PRNGKey(batch_seed + 9999)
                                traj_keys = jax.random.split(key_seq, args.n_trajectories)

                                # If "free", it's sparse_velocity with infinite threshold
                                actual_ctrl = "sparse_velocity" if ctrl_name == "free" else ctrl_name
                                actual_tau = 999.0 if ctrl_name == "free" else tau

                                # Run trajectory 1 (Unperturbed state)
                                psi_finals, n_gates_batch = run_trajectories_vmapped(
                                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                    actual_ctrl, jnp.float32(actual_tau), S_proj, bob_site,
                                    jnp.complex64(Haar_c0), jnp.complex64(Haar_c1), traj_keys
                                )

                                # Run trajectory 2 (Perturbed state - sharing IDENTICAL keys!)
                                psi_finals_perturbed, _ = run_trajectories_vmapped(
                                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                    actual_ctrl, jnp.float32(actual_tau), S_proj, bob_site,
                                    jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p), traj_keys
                                )

                                # Metric 1: Entanglement Fidelity
                                rho_AB = compute_reduced_rho_AB(psi_finals, N, bob_site)
                                metrics_arr = compute_phase_sensitive_metrics(rho_AB, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1))
                                f_ent = float(metrics_arr[0])
                                mi = float(metrics_arr[1])
                                b_sim = float(metrics_arr[2])
                                avg_gates = float(np.mean(np.array(n_gates_batch)))

                                # Metric 2: Distinguishability (Trace distance of Bob's density matrices)
                                # Extract Bob's reduced density matrices
                                trace_AB = [i for i in range(N) if i != bob_site]
                                
                                def get_rho_B(psi):
                                    psi_t = psi.reshape((2,) * N).transpose([bob_site] + trace_AB).reshape(2, -1)
                                    return jnp.tensordot(psi_t, psi_t.conj(), axes=([1], [1]))
                                
                                rho_B = jnp.mean(jax.vmap(get_rho_B)(psi_finals), axis=0)
                                rho_B_p = jnp.mean(jax.vmap(get_rho_B)(psi_finals_perturbed), axis=0)
                                dist_val = float(trace_distance_2x2(rho_B, rho_B_p))

                                # Metric 3: Echo Recovery Gain
                                # Apply Alice-Z-echo gate
                                psi_echoed = jax.vmap(apply_gate_psi, in_axes=(0, None, None, None))(psi_finals, Z_gate, 0, N)
                                rho_AB_echo = compute_reduced_rho_AB(psi_echoed, N, bob_site)
                                metrics_arr_echo = compute_phase_sensitive_metrics(rho_AB_echo, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1))
                                f_ent_echo = float(metrics_arr_echo[0])
                                echo_gain = f_ent_echo - f_ent

                                rec = {
                                    "J_std": float(J_std),
                                    "T2": float(T2),
                                    "p_cross": float(p_cross),
                                    "dist_class": dist_class,
                                    "bob_site": int(bob_site),
                                    "bob_dist": float(bob_dist),
                                    "controller": ctrl_name,
                                    "tau": float(tau),
                                    "batch": b,
                                    "F_ent": f_ent,
                                    "mutual_information": mi,
                                    "bloch_similarity": b_sim,
                                    "distinguishability": dist_val,
                                    "echo_recovery_gain": echo_gain,
                                    "n_interventions": avg_gates,
                                    "trigger_rate": avg_gates / args.n_steps
                                }
                                ctrl_records.append(rec)
                                all_records.append(rec)

                            # Statistics across batches
                            f_ents = [r["F_ent"] for r in ctrl_records]
                            dists = [r["distinguishability"] for r in ctrl_records]
                            gains = [r["echo_recovery_gain"] for r in ctrl_records]
                            gates = [r["n_interventions"] for r in ctrl_records]

                            label = f"{ctrl_name}_tau{tau:.5f}" if is_adaptive else ctrl_name
                            group_results[label] = {
                                "F_ent_mean": float(np.mean(f_ents)),
                                "F_ent_stderr": float(np.std(f_ents) / np.sqrt(args.B)),
                                "distinguishability_mean": float(np.mean(dists)),
                                "distinguishability_stderr": float(np.std(dists) / np.sqrt(args.B)),
                                "echo_recovery_gain_mean": float(np.mean(gains)),
                                "echo_recovery_gain_stderr": float(np.std(gains) / np.sqrt(args.B)),
                                "n_interventions_mean": float(np.mean(gates)),
                                "trigger_rate_mean": float(np.mean(gates) / args.n_steps),
                                "n": len(ctrl_records)
                            }
                            
                            key = f"J_{J_std:.2f}/T2_{T2:.1f}/px_{p_cross:.3f}/{dist_class}/{label}"
                            summary[key] = group_results[label]

                            print(f"        [{label:26s}] F_ent={group_results[label]['F_ent_mean']:.4f}  Dist={group_results[label]['distinguishability_mean']:.4f}  EchoGain={group_results[label]['echo_recovery_gain_mean']:+.4f}  Gates={group_results[label]['n_interventions_mean']:.1f}")

                    # -------------------------------------------------------
                    # Phase Boundary Classification for this Config Group
                    # -------------------------------------------------------
                    free_perf = group_results["free"]["F_ent_mean"]
                    dd_perf = group_results["dd"]["F_ent_mean"]
                    
                    # Find best adaptive performer
                    best_adaptive_fent = -1.0
                    best_adaptive_tau = -1.0
                    best_adaptive_label = None
                    
                    for label, metrics in group_results.items():
                        if "sparse_velocity" in label and label != "free":
                            if metrics["F_ent_mean"] > best_adaptive_fent:
                                best_adaptive_fent = metrics["F_ent_mean"]
                                best_adaptive_label = label
                                # Extract tau from label
                                best_adaptive_tau = float(label.split("tau")[1])

                    # Calculate boundary proximity
                    boundary_proximity = best_adaptive_fent - free_perf
                    
                    # Determine regime type
                    if best_adaptive_fent > free_perf + 0.005 and best_adaptive_fent > dd_perf:
                        regime = "boundary_beneficial"
                    elif dd_perf > free_perf + 0.005:
                        regime = "dd_optimal" # highly unlikely
                    elif free_perf >= best_adaptive_fent - 0.005:
                        regime = "restraint_optimal"
                    else:
                        regime = "intervention_harmful"

                    group_key = f"J_{J_std:.2f}/T2_{T2:.1f}/px_{p_cross:.3f}/{dist_class}"
                    phase_boundary[group_key] = {
                        "free_F_ent": free_perf,
                        "best_adaptive_F_ent": best_adaptive_fent if best_adaptive_fent > 0 else None,
                        "best_adaptive_tau": best_adaptive_tau if best_adaptive_tau > 0 else None,
                        "dd_F_ent": dd_perf,
                        "intervention_gain": boundary_proximity,
                        "regime": regime,
                        "boundary_proximity": boundary_proximity
                    }
                    print(f"      [DECISION] Regime: {regime} | Proximity (Gain): {boundary_proximity:+.4f}")

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_w_N{N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "phase_boundary": phase_boundary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--B", type=int, default=5)
    p.add_argument("--n-trajectories", type=int, default=100, dest="n_trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--controllers", nargs="+", default=["free", "sparse_velocity", "sparse_accel", "dd"])
    
    # Sweep parameters
    p.add_argument("--t2-sweep", nargs="+", type=float, default=[4.0, 8.0, 16.0, 32.0], dest="t2_sweep")
    p.add_argument("--j-std-sweep", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.5], dest="j_std_sweep")
    p.add_argument("--p-cross-sweep", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.06], dest="p_cross_sweep")
    p.add_argument("--tau-sweep", nargs="+", type=float, default=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], dest="tau_sweep")
    
    p.add_argument("--out-dir", default="program_w_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.Lx, args.Ly = 2, 3
        args.n_steps = 10
        args.n_trajectories = 5
        args.B = 1
        args.j_std_sweep = [0.4]
        args.t2_sweep = [8.0]
        args.p_cross_sweep = [0.03]
        args.tau_sweep = [0.01, 0.1]
        args.controllers = ["free", "sparse_velocity", "dd"]
        print("[SMOKE TEST] Running lightweight validation")

    run_experiment_w(args)
