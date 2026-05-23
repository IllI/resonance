"""
Program W — Sparse Geometry Preservation Phase Diagram
=======================================================
Central question: Does intervention TIMING matter beyond SPARSITY alone?

Controllers compared:
  free            : no intervention
  dd              : dense periodic (X-X-Y-Y every 4 steps)
  random_sparse   : uniform random timing, matched gate count
  geometry_sparse : quiet-window + observable trigger (new)
  matched_poisson : Poisson-distributed timing, matched mean rate

Key design:
  - Fully locked RNG (identical Haar state, noise, disorder per trajectory)
  - Distance sweep across ALL Manhattan distances on 3x4 grid
  - Fixed: T_quiet=8 steps, eps=0.005, tau=0.005, p_cross=0.03
  - Primary metric: eta_transport = delta_F_echo / mean_gates
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import _get_eigh, select_backend
from program_t_tpu import build_grid_hamiltonian
from program_v_tpu import (
    apply_gate_psi, make_initial_state, get_causal_observables_psi,
    compute_reduced_rho_AB, compute_phase_sensitive_metrics, run_step_psi,
)
from program_w_tpu import run_probe_stage_w, trace_distance_2x2

# ---------------------------------------------------------------------------
# Distance Enumeration
# ---------------------------------------------------------------------------

def enumerate_bob_distances(Lx, Ly):
    """Return {dist: [(site_idx, dist), ...]} for all sites, excluding Alice (site 0)."""
    result = {}
    ax, ay = 0, 0
    for site in range(Lx * Ly):
        if site == 0:
            continue
        sx, sy = site % Lx, site // Lx
        d = abs(sx - ax) + abs(sy - ay)
        result.setdefault(d, []).append((site, d))
    return result

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def ctrl_dd(step):
    seq = jnp.array([1, 1, 2, 2])
    trigger = (step % 4 == 0)
    ci = seq[(step // 4) % 4]
    return jax.lax.cond(trigger, lambda: jnp.array((ci, 0)), lambda: jnp.array((0, 0)))

def ctrl_geometry_sparse(step, obs_hist, steps_since_last, T_quiet, eps, S_proj):
    """Quiet-window + observable trigger. Fire only when:
       1. steps_since_last >= T_quiet (mandatory quiet period), AND
       2. projected velocity norm > eps (physical signal required).
    """
    quiet_ok = steps_since_last >= T_quiet
    err_vec = obs_hist[2] - obs_hist[1]
    proj_err = jnp.dot(S_proj, err_vec)
    instability = jnp.sum(proj_err ** 2)
    signal_ok = instability > eps
    trigger = jnp.logical_and(quiet_ok, signal_ok)
    trigger = jnp.logical_and(trigger, step > 2)
    return jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))

# ---------------------------------------------------------------------------
# Trajectory Runners
# ---------------------------------------------------------------------------

def run_trajectory_geometry(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                             T_quiet, eps, S_proj, bob_site, Haar_c0, Haar_c1, rng_key):
    """Run geometry_sparse controller. Returns (psi_final, n_gates, fired_mask, vel_norms)."""
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    obs0 = get_causal_observables_psi(psi0, N)
    obs_hist0 = jnp.stack([obs0, obs0, obs0], axis=0)
    carry0 = (psi0, obs_hist0, jnp.int32(999), jnp.int32(0), rng_key)

    def scan_step(carry, step_idx):
        psi, obs_hist, steps_since_last, n_gates, key = carry
        key_next, _, key_step = jax.random.split(key, 3)
        gate_info = ctrl_geometry_sparse(step_idx, obs_hist, steps_since_last, T_quiet, eps, S_proj)
        ci = gate_info[0]
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        steps_since_last_next = jax.lax.cond(
            ci > 0, lambda: jnp.int32(0), lambda: steps_since_last + 1
        )
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        obs_curr = get_causal_observables_psi(psi_next, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr], axis=0)
        # Track firing and observable velocity norm at this step
        fired = jnp.where(ci > 0, jnp.float32(1.0), jnp.float32(0.0))
        err_vec = obs_hist[2] - obs_hist[1]
        proj_err = jnp.dot(S_proj, err_vec)
        vel_norm = jnp.sum(proj_err ** 2)
        return (psi_next, obs_hist_next, steps_since_last_next, n_gates_next, key_next), (fired, vel_norm)

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, (fired_mask, vel_norms) = jax.lax.scan(scan_step, carry0, steps)
    psi_final, _, _, n_gates_total, _ = final_carry
    return psi_final, n_gates_total, fired_mask, vel_norms


def run_trajectory_scheduled(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                              bob_site, Haar_c0, Haar_c1, rng_key, gate_schedule):
    """Run with a pre-sampled gate_schedule of shape (n_steps,)."""
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
        gate_info = ctrl_dd(step_idx)
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        return (psi_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, steps)
    psi_final, _ = final_carry
    return psi_final

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(psi_finals, psi_finals_perturbed, Haar_c0, Haar_c1, bob_site, N):
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
    echo_gain = float(metrics_echo[0]) - f_ent

    return {"F_ent": f_ent, "distinguishability": dist_val, "echo_recovery_gain": echo_gain}

# ---------------------------------------------------------------------------
# Poisson Schedule Builder
# ---------------------------------------------------------------------------

def build_poisson_schedule(n_traj, n_steps, mean_gates, rng):
    """Build gate schedules where gates arrive as a Poisson process with given mean."""
    schedules = np.zeros((n_traj, n_steps), dtype=np.int32)
    if mean_gates < 0.5:
        return schedules
    # Inter-arrival rate: lambda = mean_gates / n_steps
    lam = mean_gates / n_steps
    for i in range(n_traj):
        t = 0
        while t < n_steps:
            # Sample inter-arrival time from geometric distribution (discrete Poisson)
            gap = int(rng.geometric(p=lam)) if lam > 0 else n_steps + 1
            t += gap
            if t < n_steps:
                schedules[i, t] = 1
    return schedules

# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_geometry_experiment(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program W — Sparse Geometry Preservation Phase Diagram")
    print("=" * 80)

    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    dt = args.T_max / args.n_steps
    T1_phys = 20.0
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    n_traj = args.n_trajectories
    T_quiet = args.T_quiet
    eps = args.eps

    print(f"Grid: {Lx}x{Ly}={N}  T_quiet={T_quiet}  eps={eps}  n_traj={n_traj}")

    bob_distances = enumerate_bob_distances(Lx, Ly)
    print(f"Available distances: {sorted(bob_distances.keys())}")

    # JIT-compile vmapped runners
    vmap_geo = jax.jit(jax.vmap(
        run_trajectory_geometry,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)
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
        evals_np, evecs_np = _get_eigh(H, (f"GeoXXZ_J{J_std:.2f}", N, args.seed))
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for T2 in args.t2_sweep:
            p2 = float(1.0 - np.exp(-dt / T2))
            print(f"  T2={T2:.1f}")

            for dist, bob_list in sorted(bob_distances.items()):
                # Use the middle site at each distance for consistency
                bob_site, bob_dist = bob_list[len(bob_list) // 2]
                print(f"    dist={dist} bob={bob_site}")

                S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site)

                # Shared state and RNG for all controllers
                rng = np.random.default_rng(args.seed + int(J_std * 100) + int(T2 * 10) + dist)
                u, v = rng.uniform(), rng.uniform()
                theta = np.arccos(2.0 * u - 1.0)
                phi = 2.0 * np.pi * v
                Haar_c0 = np.complex64(np.cos(theta / 2.0))
                Haar_c1 = np.complex64(np.exp(1j * phi) * np.sin(theta / 2.0))
                Haar_c0_p = np.complex64(np.cos((theta + 0.1) / 2.0))
                Haar_c1_p = np.complex64(np.exp(1j * phi) * np.sin((theta + 0.1) / 2.0))

                master_key = jax.random.PRNGKey(args.seed + int(J_std * 1000) + int(T2 * 100) + dist * 7)
                traj_keys = jax.random.split(master_key, n_traj)

                p_cross = 0.03

                # ---- 1. geometry_sparse ----
                psi_geo, n_gates_batch, fired_masks, vel_norms_batch = vmap_geo(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    jnp.int32(T_quiet), jnp.float32(eps), S_proj, bob_site,
                    jnp.complex64(Haar_c0), jnp.complex64(Haar_c1), traj_keys
                )
                psi_geo_p, _, _, _ = vmap_geo(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    jnp.int32(T_quiet), jnp.float32(eps), S_proj, bob_site,
                    jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p), traj_keys
                )
                n_gates_np = np.array(n_gates_batch)
                mean_gates = float(np.mean(n_gates_np))
                m_geo = compute_metrics(psi_geo, psi_geo_p, Haar_c0, Haar_c1, bob_site, N)

                # Compute ILC (Intervention Locality Correlation):
                # corr(fired_mask, vel_norm) per trajectory, then average.
                # High ILC means the controller fires when observables are actually changing.
                fired_np = np.array(fired_masks)       # (n_traj, n_steps)
                vels_np  = np.array(vel_norms_batch)   # (n_traj, n_steps)
                ilc_vals = []
                for i in range(n_traj):
                    f, v = fired_np[i], vels_np[i]
                    if f.sum() > 1 and v.std() > 1e-12:
                        ilc_vals.append(float(np.corrcoef(f, v)[0, 1]))
                ilc_geo = float(np.mean(ilc_vals)) if ilc_vals else 0.0

                # ILC for random_sparse: same velocity norms, but random firing times
                # We'll compute after building the schedules.

                # ---- 2. random_sparse (uniform, matched gate count) ----
                sched_rng = np.random.default_rng(args.seed + 11111 + dist)
                rand_schedules = np.zeros((n_traj, args.n_steps), dtype=np.int32)
                for i in range(n_traj):
                    k = int(n_gates_np[i])
                    if k > 0:
                        chosen = sched_rng.choice(args.n_steps, size=k, replace=False)
                        rand_schedules[i, chosen] = 1

                # ILC for random_sparse: correlation of random schedule with geo's velocity norms
                ilc_rand_vals = []
                for i in range(n_traj):
                    f, v = rand_schedules[i].astype(float), vels_np[i]
                    if f.sum() > 1 and v.std() > 1e-12:
                        ilc_rand_vals.append(float(np.corrcoef(f, v)[0, 1]))
                ilc_rand = float(np.mean(ilc_rand_vals)) if ilc_rand_vals else 0.0

                psi_rand = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1),
                    traj_keys, jnp.array(rand_schedules)
                )
                psi_rand_p = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p),
                    traj_keys, jnp.array(rand_schedules)
                )
                m_rand = compute_metrics(psi_rand, psi_rand_p, Haar_c0, Haar_c1, bob_site, N)

                # ---- 3. matched_poisson ----
                poisson_rng = np.random.default_rng(args.seed + 22222 + dist)
                poisson_schedules = build_poisson_schedule(n_traj, args.n_steps, mean_gates, poisson_rng)

                psi_pois = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1),
                    traj_keys, jnp.array(poisson_schedules)
                )
                psi_pois_p = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p),
                    traj_keys, jnp.array(poisson_schedules)
                )
                m_pois = compute_metrics(psi_pois, psi_pois_p, Haar_c0, Haar_c1, bob_site, N)

                # ---- 4. free ----
                free_schedule = np.zeros((n_traj, args.n_steps), dtype=np.int32)
                psi_free = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1),
                    traj_keys, jnp.array(free_schedule)
                )
                psi_free_p = vmap_scheduled(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p),
                    traj_keys, jnp.array(free_schedule)
                )
                m_free = compute_metrics(psi_free, psi_free_p, Haar_c0, Haar_c1, bob_site, N)

                # ---- 5. dd ----
                psi_dd = vmap_dd(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1), traj_keys
                )
                psi_dd_p = vmap_dd(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    bob_site, jnp.complex64(Haar_c0_p), jnp.complex64(Haar_c1_p), traj_keys
                )
                m_dd = compute_metrics(psi_dd, psi_dd_p, Haar_c0, Haar_c1, bob_site, N)

                # ---- Efficiency and ILC ----
                k = max(mean_gates, 1e-6)
                eta_geo  = (m_geo["echo_recovery_gain"]  - m_free["echo_recovery_gain"]) / k
                eta_rand = (m_rand["echo_recovery_gain"] - m_free["echo_recovery_gain"]) / k
                eta_pois = (m_pois["echo_recovery_gain"] - m_free["echo_recovery_gain"]) / k
                eta_dd   = (m_dd["echo_recovery_gain"]   - m_free["echo_recovery_gain"]) / (args.n_steps // 4)

                timing_advantage = m_geo["echo_recovery_gain"] - m_rand["echo_recovery_gain"]

                group_key = f"J_{J_std:.2f}/T2_{T2:.1f}/dist_{dist}"
                record = {
                    "J_std": float(J_std), "T2": float(T2),
                    "dist": dist, "bob_site": int(bob_site),
                    "mean_gates_geo": mean_gates,
                    "geometry_sparse": m_geo,
                    "random_sparse": m_rand,
                    "matched_poisson": m_pois,
                    "free": m_free,
                    "dd": m_dd,
                    "eta_transport": {
                        "geometry_sparse": eta_geo,
                        "random_sparse": eta_rand,
                        "matched_poisson": eta_pois,
                        "dd": eta_dd,
                    },
                    "ilc": {
                        "geometry_sparse": ilc_geo,
                        "random_sparse": ilc_rand,
                        "ilc_advantage": ilc_geo - ilc_rand,
                    },
                    "timing_advantage_echo": timing_advantage,
                    "geo_beats_rand": timing_advantage > 0,
                }
                summary[group_key] = record
                all_records.append(record)

                print(f"      [free    ] F={m_free['F_ent']:.4f} Echo={m_free['echo_recovery_gain']:+.4f} Gates=0")
                print(f"      [geo     ] F={m_geo['F_ent']:.4f}  Echo={m_geo['echo_recovery_gain']:+.4f}  Gates={mean_gates:.1f} η={eta_geo:+.5f} ILC={ilc_geo:+.3f}")
                print(f"      [rand    ] F={m_rand['F_ent']:.4f} Echo={m_rand['echo_recovery_gain']:+.4f} Gates={mean_gates:.1f} η={eta_rand:+.5f} ILC={ilc_rand:+.3f}")
                print(f"      [poisson ] F={m_pois['F_ent']:.4f} Echo={m_pois['echo_recovery_gain']:+.4f} Gates={mean_gates:.1f} η={eta_pois:+.5f}")
                print(f"      [dd      ] F={m_dd['F_ent']:.4f}  Echo={m_dd['echo_recovery_gain']:+.4f}  Gates={args.n_steps//4} η={eta_dd:+.5f}")
                print(f"      [TIMING ADV] Δecho={timing_advantage:+.4f} ILC_adv={ilc_geo-ilc_rand:+.3f} ({'✓' if timing_advantage > 0 else '✗'})")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "program_w_geometry_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "sparse_geometry_preservation_phase_diagram",
            "args": vars(args),
            "summary": summary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time() - t0:.1f}s)")

    # Global summary
    n_total = len(all_records)
    n_geo_wins = sum(1 for r in all_records if r["geo_beats_rand"])
    by_dist = {}
    for r in all_records:
        d = r["dist"]
        by_dist.setdefault(d, []).append(r["timing_advantage_echo"])
    print(f"\n{'='*80}")
    print(f"TIMING ADVANTAGE SUMMARY (geo > rand): {n_geo_wins}/{n_total}")
    print(f"{'Dist':>5}  {'Mean ΔEcho(geo-rand)':>22}  {'Mean ILC_adv':>14}  {'Win Rate':>10}")
    for d in sorted(by_dist.keys()):
        vals = by_dist[d]
        ilc_adv_vals = [r["ilc"]["ilc_advantage"] for r in all_records if r["dist"] == d]
        print(f"  {d:>3}    {np.mean(vals):>+.5f}                   {np.mean(ilc_adv_vals):>+.4f}       {sum(v>0 for v in vals)}/{len(vals)}")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=500, dest="n_trajectories")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--eps", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_w_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
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
        print("[SMOKE TEST]")

    run_geometry_experiment(args)
