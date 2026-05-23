"""
Program W — Causal Recoverability Validation
=============================================
Final decisive experiment. Claim under test:

    "Sparse boundary-triggered adaptive interventions improve recoverability of
    transported quantum information under noise while requiring substantially
    fewer control operations than dynamical decoupling."

Key design:
- All controllers see IDENTICAL Haar states, dephasing draws, and crosstalk draws.
- `random_matched` fires the EXACT same number of gates as `sparse_velocity` per
  trajectory, but at uniformly random timesteps. This isolates causal timing.
- Single tau=0.005 (pre-characterized from phase diagram sweep).
- W3 disorder pocket only: J_std in {0.8, 1.0, 1.2}, T2 in {8, 12}.
- Primary endpoint: dist=3 (medium). dist=1 is calibration.
- Primary success criterion:
    F_echo(sparse) > F_echo(random_matched) > F_echo(free)
  under equal intervention budgets.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import _get_eigh, select_backend
from program_t_tpu import build_grid_hamiltonian, get_bob_candidates
from program_v_tpu import (
    apply_gate_psi, make_initial_state, get_causal_observables_psi,
    compute_reduced_rho_AB, compute_phase_sensitive_metrics, run_step_psi,
)
from program_w_tpu import run_probe_stage_w, trace_distance_2x2

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def ctrl_dd_w(step):
    seq = jnp.array([1, 1, 2, 2])
    trigger = (step % 4 == 0)
    ci = seq[(step // 4) % 4]
    return jax.lax.cond(trigger, lambda: jnp.array((ci, 0)), lambda: jnp.array((0, 0)))

def ctrl_sparse_velocity(step, obs_hist, cooldown, tau_thresh, S_proj):
    err_vec = obs_hist[2] - obs_hist[1]
    proj_err = jnp.dot(S_proj, err_vec)
    instability = jnp.sum(proj_err ** 2)
    trigger = jnp.logical_and(instability > tau_thresh, cooldown == 0)
    trigger = jnp.logical_and(trigger, step > 2)
    return jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))

# ---------------------------------------------------------------------------
# Trajectory Runners
# ---------------------------------------------------------------------------

def run_trajectory_sparse(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                           tau_thresh, S_proj, bob_site, Haar_c0, Haar_c1, rng_key):
    """Run sparse_velocity controller. Returns (psi_final, n_gates, fired_steps_mask)."""
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    obs0 = get_causal_observables_psi(psi0, N)
    obs_hist0 = jnp.stack([obs0, obs0, obs0], axis=0)
    carry0 = (psi0, obs_hist0, jnp.int32(0), jnp.int32(0), rng_key)

    def scan_step(carry, step_idx):
        psi, obs_hist, cooldown, n_gates, key = carry
        key_next, _, key_step = jax.random.split(key, 3)
        gate_info = ctrl_sparse_velocity(step_idx, obs_hist, cooldown, tau_thresh, S_proj)
        ci = gate_info[0]
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        cooldown_next = jax.lax.cond(ci > 0, lambda: jnp.int32(4), lambda: jnp.maximum(0, cooldown - 1))
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        obs_curr = get_causal_observables_psi(psi_next, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr], axis=0)
        fired = jnp.where(ci > 0, 1, 0)
        return (psi_next, obs_hist_next, cooldown_next, n_gates_next, key_next), fired

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, fired_mask = jax.lax.scan(scan_step, carry0, steps)
    psi_final, _, _, n_gates_total, _ = final_carry
    return psi_final, n_gates_total, fired_mask


def run_trajectory_scheduled(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                              bob_site, Haar_c0, Haar_c1, rng_key, gate_schedule):
    """
    Run with a pre-sampled boolean gate_schedule of shape (n_steps,).
    Used for both `free` (all-False schedule) and `random_matched`.
    """
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    carry0 = (psi0, rng_key)

    def scan_step(carry, args):
        psi, key = carry
        step_idx, should_fire = args
        key_next, _, key_step = jax.random.split(key, 3)
        gate_info = jax.lax.cond(
            should_fire > 0,
            lambda: jnp.array((3, 0)),
            lambda: jnp.array((0, 0))
        )
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        return (psi_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, (steps, gate_schedule))
    psi_final, _ = final_carry
    return psi_final


def run_trajectory_dd(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                      bob_site, Haar_c0, Haar_c1, rng_key):
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    carry0 = (psi0, rng_key)

    def scan_step(carry, step_idx):
        psi, key = carry
        key_next, _, key_step = jax.random.split(key, 3)
        gate_info = ctrl_dd_w(step_idx)
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        return (psi_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, steps)
    psi_final, _ = final_carry
    return psi_final

# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(psi_finals, psi_finals_perturbed, Haar_c0, Haar_c1, bob_site, N):
    """Compute F_ent, distinguishability, echo_recovery_gain for a batch."""
    Z_gate = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

    rho_AB = compute_reduced_rho_AB(psi_finals, N, bob_site)
    metrics_arr = compute_phase_sensitive_metrics(rho_AB, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1))
    f_ent = float(metrics_arr[0])

    trace_AB = [i for i in range(N) if i != bob_site]
    def get_rho_B(psi):
        psi_t = psi.reshape((2,) * N).transpose([bob_site] + trace_AB).reshape(2, -1)
        return jnp.tensordot(psi_t, psi_t.conj(), axes=([1], [1]))

    rho_B = jnp.mean(jax.vmap(get_rho_B)(psi_finals), axis=0)
    rho_B_p = jnp.mean(jax.vmap(get_rho_B)(psi_finals_perturbed), axis=0)
    dist_val = float(trace_distance_2x2(rho_B, rho_B_p))

    psi_echoed = jax.vmap(apply_gate_psi, in_axes=(0, None, None, None))(psi_finals, Z_gate, 0, N)
    rho_AB_echo = compute_reduced_rho_AB(psi_echoed, N, bob_site)
    metrics_echo = compute_phase_sensitive_metrics(rho_AB_echo, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1))
    f_echo = float(metrics_echo[0])
    echo_gain = f_echo - f_ent

    return {"F_ent": f_ent, "distinguishability": dist_val, "echo_recovery_gain": echo_gain}

# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_causal_experiment(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program W — Causal Recoverability Validation")
    print("=" * 80)

    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    dt = args.T_max / args.n_steps
    T1_phys = 20.0
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    tau = args.tau
    n_traj = args.n_trajectories

    print(f"Grid: {Lx}x{Ly}={N}  τ={tau}  n_traj={n_traj}  T_max={args.T_max}")

    bob_cands = get_bob_candidates(Lx, Ly)

    # JIT compile vmapped runners
    vmap_sparse = jax.jit(jax.vmap(
        run_trajectory_sparse,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    vmap_scheduled = jax.jit(jax.vmap(
        run_trajectory_scheduled,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, 0, 0)
    ), static_argnums=(2, 4))

    vmap_dd = jax.jit(jax.vmap(
        run_trajectory_dd,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    summary = {}
    all_records = []

    for J_std in args.j_std_sweep:
        print(f"\n== J_std={J_std:.2f} ==")
        H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
        evals_np, evecs_np = _get_eigh(H, (f"CausalXXZ_J{J_std:.2f}", N, args.seed))
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for T2 in args.t2_sweep:
            p2 = float(1.0 - np.exp(-dt / T2))
            print(f"  T2={T2:.1f} (p2={p2:.4f})")

            for dist_class, bob_list in bob_cands.items():
                if not bob_list:
                    continue
                bob_site, bob_dist = bob_list[len(bob_list) // 2]
                dist_label = "primary" if dist_class == "medium" else "calibration"
                print(f"    [{dist_label}] dist_class={dist_class} bob={bob_site} dist={bob_dist}")

                S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site)

                # --- Generate SHARED states and keys for all controllers ---
                rng = np.random.default_rng(args.seed + int(J_std * 100) + int(T2 * 10))

                # Generate ONE Haar state for all trajectories in this config
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                theta = np.arccos(2.0 * u - 1.0)
                phi = 2.0 * np.pi * v
                Haar_c0_scalar = np.cos(theta / 2.0).astype(np.complex64)
                Haar_c1_scalar = (np.exp(1j * phi) * np.sin(theta / 2.0)).astype(np.complex64)

                # Perturbed states (fixed 0.1 rad perturbation)
                theta_p = theta + 0.1
                Haar_c0_p_scalar = np.cos(theta_p / 2.0).astype(np.complex64)
                Haar_c1_p_scalar = (np.exp(1j * phi) * np.sin(theta_p / 2.0)).astype(np.complex64)

                # Shared JAX keys — ALL controllers use these EXACT keys
                master_key = jax.random.PRNGKey(args.seed + int(J_std * 1000) + int(T2 * 100))
                traj_keys = jax.random.split(master_key, n_traj)

                # ---- 1. Run sparse_velocity (reference controller) ----
                psi_sparse, n_gates_batch, fired_masks = vmap_sparse(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    jnp.float32(tau), S_proj, bob_site,
                    jnp.complex64(Haar_c0_scalar), jnp.complex64(Haar_c1_scalar), traj_keys
                )
                psi_sparse_p, _, _ = vmap_sparse(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    jnp.float32(tau), S_proj, bob_site,
                    jnp.complex64(Haar_c0_p_scalar), jnp.complex64(Haar_c1_p_scalar), traj_keys
                )
                n_gates_np = np.array(n_gates_batch)
                avg_sparse_gates = float(np.mean(n_gates_np))
                metrics_sparse = compute_metrics(
                    psi_sparse, psi_sparse_p, Haar_c0_scalar, Haar_c1_scalar, bob_site, N)

                # ---- 2. Build random_matched schedules ----
                # Each trajectory i gets exactly n_gates_np[i] gates at random steps.
                schedule_rng = np.random.default_rng(args.seed + 77777)
                matched_schedules = np.zeros((n_traj, args.n_steps), dtype=np.int32)
                for i in range(n_traj):
                    k = int(n_gates_np[i])
                    if k > 0:
                        chosen = schedule_rng.choice(args.n_steps, size=k, replace=False)
                        matched_schedules[i, chosen] = 1

                # ---- 3. Run random_matched ----
                free_schedule = np.zeros((n_traj, args.n_steps), dtype=np.int32)
                psi_random = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_scalar), jnp.complex64(Haar_c1_scalar),
                    traj_keys, jnp.array(matched_schedules)
                )
                psi_random_p = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_p_scalar), jnp.complex64(Haar_c1_p_scalar),
                    traj_keys, jnp.array(matched_schedules)
                )
                metrics_random = compute_metrics(
                    psi_random, psi_random_p, Haar_c0_scalar, Haar_c1_scalar, bob_site, N)

                # ---- 4. Run free (zero gates) ----
                psi_free = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_scalar), jnp.complex64(Haar_c1_scalar),
                    traj_keys, jnp.array(free_schedule)
                )
                psi_free_p = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_p_scalar), jnp.complex64(Haar_c1_p_scalar),
                    traj_keys, jnp.array(free_schedule)
                )
                metrics_free = compute_metrics(
                    psi_free, psi_free_p, Haar_c0_scalar, Haar_c1_scalar, bob_site, N)

                # ---- 5. Run DD (reference decoupling) ----
                psi_dd = vmap_dd(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_scalar), jnp.complex64(Haar_c1_scalar), traj_keys
                )
                psi_dd_p = vmap_dd(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    bob_site, jnp.complex64(Haar_c0_p_scalar), jnp.complex64(Haar_c1_p_scalar), traj_keys
                )
                metrics_dd = compute_metrics(
                    psi_dd, psi_dd_p, Haar_c0_scalar, Haar_c1_scalar, bob_site, N)

                # ---- Compute intervention efficiency ----
                # eta = delta_F_echo per gate fired (vs free baseline)
                eta_sparse = (metrics_sparse["echo_recovery_gain"] - metrics_free["echo_recovery_gain"]) / max(avg_sparse_gates, 1e-6)
                eta_random = (metrics_random["echo_recovery_gain"] - metrics_free["echo_recovery_gain"]) / max(avg_sparse_gates, 1e-6)

                # ---- Primary success test ----
                success = (
                    metrics_sparse["echo_recovery_gain"] > metrics_random["echo_recovery_gain"] and
                    metrics_random["echo_recovery_gain"] > metrics_free["echo_recovery_gain"]
                )

                group_key = f"J_{J_std:.2f}/T2_{T2:.1f}/{dist_class}"
                result = {
                    "J_std": float(J_std),
                    "T2": float(T2),
                    "dist_class": dist_class,
                    "dist_label": dist_label,
                    "bob_site": int(bob_site),
                    "bob_dist": float(bob_dist),
                    "n_trajectories": n_traj,
                    "avg_gates_sparse": avg_sparse_gates,
                    "avg_gates_random_matched": float(np.mean(n_gates_np)),  # same by construction
                    "avg_gates_dd": float(args.n_steps // 4),
                    "sparse_velocity": metrics_sparse,
                    "random_matched": metrics_random,
                    "free": metrics_free,
                    "dd": metrics_dd,
                    "intervention_efficiency_sparse": eta_sparse,
                    "intervention_efficiency_random": eta_random,
                    "primary_success": success,
                }
                summary[group_key] = result
                all_records.append(result)

                # Print
                print(f"      [free          ] F_ent={metrics_free['F_ent']:.4f}  "
                      f"EchoGain={metrics_free['echo_recovery_gain']:+.4f}  "
                      f"Dist={metrics_free['distinguishability']:.4f}  Gates=0")
                print(f"      [sparse_velocity] F_ent={metrics_sparse['F_ent']:.4f}  "
                      f"EchoGain={metrics_sparse['echo_recovery_gain']:+.4f}  "
                      f"Dist={metrics_sparse['distinguishability']:.4f}  "
                      f"Gates={avg_sparse_gates:.1f}  η={eta_sparse:+.5f}")
                print(f"      [random_matched ] F_ent={metrics_random['F_ent']:.4f}  "
                      f"EchoGain={metrics_random['echo_recovery_gain']:+.4f}  "
                      f"Dist={metrics_random['distinguishability']:.4f}  "
                      f"Gates={avg_sparse_gates:.1f}  η={eta_random:+.5f}")
                print(f"      [dd             ] F_ent={metrics_dd['F_ent']:.4f}  "
                      f"EchoGain={metrics_dd['echo_recovery_gain']:+.4f}  "
                      f"Dist={metrics_dd['distinguishability']:.4f}  "
                      f"Gates={args.n_steps // 4}")
                print(f"      [VERDICT] {'✓ PASS' if success else '✗ FAIL'}: "
                      f"sparse({metrics_sparse['echo_recovery_gain']:+.4f}) > "
                      f"random({metrics_random['echo_recovery_gain']:+.4f}) > "
                      f"free({metrics_free['echo_recovery_gain']:+.4f})")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "program_w_causal_validation.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "causal_recoverability_validation",
            "args": vars(args),
            "summary": summary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time() - t0:.1f}s)")

    # Print global verdict
    n_pass = sum(1 for r in all_records if r["primary_success"])
    n_medium = sum(1 for r in all_records if r["dist_class"] == "medium")
    n_medium_pass = sum(1 for r in all_records if r["primary_success"] and r["dist_class"] == "medium")
    print(f"\n{'='*80}")
    print(f"GLOBAL VERDICT: {n_pass}/{len(all_records)} configs pass (primary: {n_medium_pass}/{n_medium} medium-dist)")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program W Causal Recoverability Validation")
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=500, dest="n_trajectories")
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_w_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")

    # W3 pocket defaults
    p.add_argument("--j-std-sweep", nargs="+", type=float,
                   default=[0.8, 1.0, 1.2], dest="j_std_sweep")
    p.add_argument("--t2-sweep", nargs="+", type=float,
                   default=[8.0, 12.0], dest="t2_sweep")

    args = p.parse_args()

    if args.smoke_test:
        args.Lx, args.Ly = 2, 3
        args.n_steps = 10
        args.n_trajectories = 20
        args.j_std_sweep = [1.0]
        args.t2_sweep = [8.0]
        print("[SMOKE TEST] Lightweight causal validation")

    run_causal_experiment(args)
