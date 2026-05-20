"""
Calibration script for Program X — Hierarchical Encoded Message Recovery
Sweeps T_max and eps to locate the optimal transport window and trigger sensitivity.
"""
import argparse, json, os, sys, time
import numpy as np

try:
    import jax, jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import _get_eigh, select_backend
from program_t_tpu import build_grid_hamiltonian
from program_w_tpu import run_probe_stage_w
from program_x_tpu import (
    build_mera_gates, _make_initial_x,
    run_trajectory_x_free, run_trajectory_x_dd,
    run_trajectory_x_geo, run_trajectory_x_scheduled,
    _decode_and_F
)


def run_calibration(args):
    select_backend("jax", require_tpu=args.require_tpu)
    print("=" * 70)
    print("Calibrating Program X: Sweeping T_max and Trigger Threshold eps")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    T1 = 20.0
    T2 = 10.0  # Fixed intermediate dephasing

    alice_gates, bob_gates = build_mera_gates(depth=args.mera_depth, seed=args.seed)
    logical_bases = [
        jnp.array([1, 0, 0, 0], dtype=jnp.complex64),
        jnp.array([0, 1, 0, 0], dtype=jnp.complex64),
        jnp.array([0, 0, 1, 0], dtype=jnp.complex64),
        jnp.array([0, 0, 0, 1], dtype=jnp.complex64),
    ]
    msg_labels = ["|00>", "|01>", "|10>", "|11>"]

    # JIT compile vmapped runners
    vmap_free = jax.jit(jax.vmap(
        run_trajectory_x_free,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    vmap_dd = jax.jit(jax.vmap(
        run_trajectory_x_dd,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    vmap_sched = jax.jit(jax.vmap(
        run_trajectory_x_scheduled,
        in_axes=(None, None, None, None, None, None, None, None, None, 0, 0)
    ), static_argnums=(2, 4))

    vmap_geo = jax.jit(jax.vmap(
        run_trajectory_x_geo,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))


    # Fixed Hamiltonian
    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=1.0, seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, (f"Calib_J1.0", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)

    # Site 7 is Bob's central boundary site
    S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(args.T_max_list[0]/args.n_steps), bob_site=7)

    results = []

    for T_max in args.T_max_list:
        dt = T_max / args.n_steps
        p1 = float(1.0 - np.exp(-dt / T1))
        p2 = float(1.0 - np.exp(-dt / T2))
        
        # Recalculate S_proj for this dt
        S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site=7)

        print(f"\n--- Sweeping T_max = {T_max:.2f} (dt = {dt:.4f}) ---")

        # Let's first run Free and DD
        master_key = jax.random.PRNGKey(args.seed + int(T_max * 100))
        traj_keys = jax.random.split(master_key, args.n_trajectories)

        free_fids = []
        dd_fids = []
        
        for msg_idx, alice_logical in enumerate(logical_bases):
            psi0 = _make_initial_x(N, alice_logical, alice_gates, bob_gates)
            
            psi_free_batch = vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                       jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                                       psi0, traj_keys)
            free_fids.append(_decode_and_F(psi_free_batch, bob_gates, alice_logical, N))

            psi_dd_batch = vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                   jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                                   psi0, traj_keys)
            dd_fids.append(_decode_and_F(psi_dd_batch, bob_gates, alice_logical, N))

        avg_free = float(np.mean(free_fids))
        avg_dd = float(np.mean(dd_fids))
        print(f"  [Free] Avg F_logical = {avg_free:.4f}")
        print(f"  [DD]   Avg F_logical = {avg_dd:.4f}")

        # Now sweep eps for geometry sparse
        for eps in args.eps_list:
            geo_fids = []
            rand_fids = []
            gate_counts = []
            ilc_vals = []

            for msg_idx, alice_logical in enumerate(logical_bases):
                psi0 = _make_initial_x(N, alice_logical, alice_gates, bob_gates)

                # Geometry
                psi_geo_batch, n_gates_batch, fired_masks, vel_norms_batch = vmap_geo(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    jnp.int32(args.T_quiet), jnp.float32(eps), S_proj,
                    psi0, traj_keys)
                
                geo_fid = _decode_and_F(psi_geo_batch, bob_gates, alice_logical, N)
                geo_fids.append(geo_fid)
                mean_gates = float(jnp.mean(n_gates_batch))
                gate_counts.append(mean_gates)

                # Compute ILC
                fired_np = np.array(fired_masks)
                vels_np = np.array(vel_norms_batch)
                msg_ilc = [float(np.corrcoef(fired_np[i], vels_np[i])[0, 1])
                           for i in range(args.n_trajectories)
                           if fired_np[i].sum() > 1 and vels_np[i].std() > 1e-12]
                if msg_ilc:
                    ilc_vals.append(np.mean(msg_ilc))

                # Random sparse matched
                rng = np.random.default_rng(args.seed + msg_idx * 9999 + int(eps*10000))
                rand_schedules = np.zeros((args.n_trajectories, args.n_steps), dtype=np.int32)
                n_gates_np = np.array(n_gates_batch)
                for i in range(args.n_trajectories):
                    k = int(n_gates_np[i])
                    if k > 0:
                        chosen = rng.choice(args.n_steps, size=k, replace=False)
                        rand_schedules[i, chosen] = 1

                psi_rand_batch = vmap_sched(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                                            psi0, traj_keys, jnp.array(rand_schedules))
                rand_fid = _decode_and_F(psi_rand_batch, bob_gates, alice_logical, N)
                rand_fids.append(rand_fid)

            avg_geo = float(np.mean(geo_fids))
            avg_rand = float(np.mean(rand_fids))
            avg_gates = float(np.mean(gate_counts))
            avg_ilc = float(np.mean(ilc_vals)) if ilc_vals else 0.0
            timing_adv = avg_geo - avg_rand

            print(f"  [Geo eps={eps:.4f}] Avg F = {avg_geo:.4f} | Rand F = {avg_rand:.4f} | "
                  f"Δ = {timing_adv:+.4f} | Gates = {avg_gates:.2f} | ILC = {avg_ilc:+.3f}")

            results.append({
                "T_max": T_max,
                "eps": eps,
                "avg_free": avg_free,
                "avg_dd": avg_dd,
                "avg_geo": avg_geo,
                "avg_rand": avg_rand,
                "timing_advantage": timing_adv,
                "avg_gates": avg_gates,
                "avg_ilc": avg_ilc
            })

    out_path = os.path.join(args.out_dir, "program_x_calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T-max-list", nargs="+", type=float, default=[1.5, 2.5, 3.5, 4.5, 6.0])
    p.add_argument("--eps-list", nargs="+", type=float, default=[0.001, 0.002, 0.005, 0.010])
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--n-trajectories", type=int, default=100)
    p.add_argument("--mera-depth", type=int, default=2)
    p.add_argument("--T-quiet", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_x_results")
    p.add_argument("--require-tpu", action="store_true")
    args = p.parse_args()
    run_calibration(args)
