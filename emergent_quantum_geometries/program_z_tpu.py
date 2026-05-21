"""
Program Z — Scale-Resolved MERA Recoverability
===============================================
Core question: Does ΔF_geo-rand increase monotonically with MERA causal depth?

New vs Program Y:
  - MERA depth extended to 4 (Layer 4 adds long-range pairs)
  - ZZ pairwise correlators tracked at each MERA layer (multiscale signal)
  - multiscale_sparse controller: triggers on highest-layer instability
  - Layer-resolved ILC: ILC_L0..L3 for each MERA scale
  - Recoverability horizon: partial MERA decode fidelity by layer

Controllers: free, dd, random_sparse, local_sparse, multiscale_sparse
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
from program_v_tpu import get_causal_observables_psi, run_step_psi
from program_w_geometry import ctrl_geometry_sparse, ctrl_dd
from program_w_tpu import run_probe_stage_w

# Import shared infrastructure from Program Y
from program_y_tpu import (
    _haar_u4, apply_2q_gate, apply_gates, apply_gates_inverse,
    get_logical_fidelity, _make_initial_y, _decode_and_F,
    run_traj_free, run_traj_dd, run_traj_scheduled, run_traj_geo,
)

# ---------------------------------------------------------------------------
# Extended MERA — supports depth 1..4
# ---------------------------------------------------------------------------

def build_mera_gates_z(depth=2, seed=42):
    """
    depth=1: L1 only           (3 gates/side — nearest-neighbor)
    depth=2: L1+L2             (5 gates/side)
    depth=3: L1+L2+L3          (7 gates/side — next-nearest)
    depth=4: L1+L2+L3+L4       (9 gates/side — long-range, skip-3)
    Alice: sites 0-5. Bob: sites 6-11. Logical qubits: (0,1) and (6,7).
    """
    rng = np.random.default_rng(seed)

    def _side_gates(offset):
        gates = []
        for s in range(offset, offset + 6, 2):
            gates.append((jnp.array(_haar_u4(rng)), s, s + 1))
        if depth >= 2:
            for s in range(offset + 1, offset + 5, 2):
                gates.append((jnp.array(_haar_u4(rng)), s, s + 1))
        if depth >= 3:
            for s in range(offset, offset + 4, 2):
                gates.append((jnp.array(_haar_u4(rng)), s, s + 2))
        if depth >= 4:
            for s in [offset, offset + 1]:
                gates.append((jnp.array(_haar_u4(rng)), s, s + 3))
        return gates

    return _side_gates(0), _side_gates(6)


def get_mera_layer_pairs(depth, N=12):
    """Return list of (i,j) pairs grouped by MERA layer, across full system."""
    alice_off, bob_off = 0, 6
    layers = []
    # Layer 1: nearest-neighbor
    l1 = [(a, a+1) for a in range(alice_off, alice_off+6, 2)]
    l1 += [(b, b+1) for b in range(bob_off, bob_off+6, 2)]
    layers.append(l1)
    if depth >= 2:
        l2 = [(a, a+1) for a in range(alice_off+1, alice_off+5, 2)]
        l2 += [(b, b+1) for b in range(bob_off+1, bob_off+5, 2)]
        layers.append(l2)
    if depth >= 3:
        l3 = [(a, a+2) for a in range(alice_off, alice_off+4, 2)]
        l3 += [(b, b+2) for b in range(bob_off, bob_off+4, 2)]
        layers.append(l3)
    if depth >= 4:
        l4 = [(alice_off, alice_off+3), (alice_off+1, alice_off+4)]
        l4 += [(bob_off, bob_off+3), (bob_off+1, bob_off+4)]
        layers.append(l4)
    return layers[:depth]


def build_ZZ_sign_matrix(layer_pairs, N=12):
    """
    Precompute ZZ sign matrix for all MERA pairs.
    Returns (signs [n_pairs, 2^N], layer_slices [(start,end)...]).
    """
    all_pairs = [(i, j) for lp in layer_pairs for (i, j) in lp]
    basis = np.arange(2 ** N)
    signs = np.zeros((len(all_pairs), 2 ** N), dtype=np.float32)
    for k, (i, j) in enumerate(all_pairs):
        bi = (basis >> (N - 1 - i)) & 1
        bj = (basis >> (N - 1 - j)) & 1
        signs[k] = (1 - 2 * bi) * (1 - 2 * bj)
    slices = []
    idx = 0
    for lp in layer_pairs:
        n = len(lp)
        slices.append((idx, idx + n))
        idx += n
    return jnp.array(signs), slices


# ---------------------------------------------------------------------------
# Multiscale trajectory runner
# ---------------------------------------------------------------------------

def run_traj_multiscale_geo(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                              T_quiet, eps, S_proj, ZZ_signs, n_pairs_total,
                              psi0, rng_key):
    """
    Multiscale controller: triggers on L0 (local velocity) OR highest-layer ZZ velocity.
    Returns (psi_final, n_gates, fired_mask, vel_norms_L0, zz_vels).
    """
    obs0 = get_causal_observables_psi(psi0, N)
    obs_hist0 = jnp.stack([obs0, obs0, obs0])
    prob0 = jnp.abs(psi0) ** 2
    zz0 = jnp.dot(ZZ_signs, prob0)  # (n_pairs_total,)
    carry0 = (psi0, obs_hist0, zz0, jnp.int32(999), jnp.int32(0), rng_key)

    def step(carry, s):
        psi, obs_hist, zz_prev, ssl, n_gates, key = carry
        key, _, ks = jax.random.split(key, 3)

        # L0 trigger (same as local_sparse / Program Y geo)
        gi_local = ctrl_geometry_sparse(s, obs_hist, ssl, T_quiet, eps, S_proj)

        # Highest-layer ZZ instability signal
        prob = jnp.abs(psi) ** 2
        zz_curr = jnp.dot(ZZ_signs, prob)
        zz_vel = jnp.sum((zz_curr - zz_prev) ** 2)

        # Multiscale fires if L0 OR high-layer ZZ vel > eps
        zz_trigger = jnp.logical_and(ssl >= T_quiet, jnp.logical_and(s > 2, zz_vel > eps * 0.5))
        gi = jax.lax.cond(
            jnp.logical_or(gi_local[0] > 0, zz_trigger),
            lambda: jnp.array((3, 0)),
            lambda: jnp.array((0, 0))
        )
        ci = gi[0]
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
        ssl_next = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
        n_next = n_gates + jnp.where(ci > 0, 1, 0)
        obs_curr = get_causal_observables_psi(psi, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr])
        fired = jnp.where(ci > 0, 1.0, 0.0)
        vel_L0 = jnp.sum(jnp.dot(S_proj, obs_hist[2] - obs_hist[1]) ** 2)
        return (psi, obs_hist_next, zz_curr, ssl_next, n_next, key), (fired, vel_L0, zz_vel)

    (psi_f, _, _, _, n_total, _), (fired_mask, vel_L0, zz_vels) = jax.lax.scan(
        step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return psi_f, n_total, fired_mask, vel_L0, zz_vels


def compute_recoverability_horizon(psi_batch, bob_gates_by_depth, alice_logical, N):
    """
    For each partial decode depth ℓ, measure F_logical using first ℓ layers of Bob's gates.
    Returns dict {depth: mean_F}.
    """
    result = {}
    for ℓ, partial_gates in enumerate(bob_gates_by_depth, 1):
        vals = []
        for psi in psi_batch:
            psi_dec = apply_gates_inverse(jnp.array(psi), partial_gates, N)
            vals.append(float(get_logical_fidelity(psi_dec, alice_logical, N)))
        result[ℓ] = float(np.mean(vals))
    return result


def get_bob_gates_by_layer(bob_gates, depth):
    """Split Bob's MERA gates into cumulative layers for recoverability horizon."""
    layer_sizes = {1: 3, 2: 5, 3: 7, 4: 9}
    gates_per_layer = [3, 2, 2, 2]  # additional gates per layer
    result = []
    idx = 0
    for ℓ in range(1, depth + 1):
        idx += gates_per_layer[ℓ - 1]
        result.append(bob_gates[:idx])
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_program_z(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program Z — Scale-Resolved MERA Recoverability")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    dt = args.T_max / args.n_steps
    T1 = 20.0
    p1 = float(1.0 - np.exp(-dt / T1))
    p_cross = 0.03

    logical_bases = [
        jnp.array([1, 0, 0, 0], dtype=jnp.complex64),
        jnp.array([0, 1, 0, 0], dtype=jnp.complex64),
        jnp.array([0, 0, 1, 0], dtype=jnp.complex64),
        jnp.array([0, 0, 0, 1], dtype=jnp.complex64),
    ]
    msg_labels = ["|00>", "|01>", "|10>", "|11>"]

    # Build vmapped runners (static argnums: N=2, n_steps=4)
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_dd = jax.jit(jax.vmap(run_traj_dd,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_sched = jax.jit(jax.vmap(run_traj_scheduled,
        in_axes=(None,None,None,None,None,None,None,None,None,0,0)), static_argnums=(2,4))
    vmap_local = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))

    # Multiscale runner — ZZ_signs shape varies with depth, so separate jit per depth
    all_records = []

    for depth in args.depth_sweep:
        print(f"\n{'='*70}")
        print(f"MERA depth={depth}")
        print(f"{'='*70}")

        alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=args.seed)
        layer_pairs = get_mera_layer_pairs(depth, N)
        ZZ_signs, zz_slices = build_ZZ_sign_matrix(layer_pairs, N)
        n_pairs_total = ZZ_signs.shape[0]
        bob_gates_by_layer = get_bob_gates_by_layer(bob_gates, depth)
        print(f"  Alice gates={len(alice_gates)}  Bob gates={len(bob_gates)}  ZZ pairs={n_pairs_total}")

        n_zz = n_pairs_total  # static shape for this depth
        vmap_ms = jax.jit(jax.vmap(
            run_traj_multiscale_geo,
            in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,None,None,0)
        ), static_argnums=(2, 4, 12))

        for J_std in args.j_std_sweep:
            print(f"\n  == J_std={J_std:.2f} ==")
            H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
            evals_np, evecs_np = _get_eigh(H, (f"Z_d{depth}_J{J_std:.2f}", N, args.seed))
            evals = jnp.array(evals_np, dtype=jnp.float32)
            evecs = jnp.array(evecs_np, dtype=jnp.complex64)

            for T2 in args.t2_sweep:
                p2 = float(1.0 - np.exp(-dt / T2))
                print(f"    T2={T2:.1f}")

                S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site=7)
                master_key = jax.random.PRNGKey(
                    args.seed + depth * 100000 + int(J_std * 1000) + int(T2 * 100))
                traj_keys = jax.random.split(master_key, args.n_trajectories)

                msg_results = {}
                for msg_idx, alice_logical in enumerate(logical_bases):
                    label = msg_labels[msg_idx]
                    psi0 = _make_initial_y(N, alice_logical, alice_gates, bob_gates)

                    # free
                    F_free = _decode_and_F(
                        vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                  jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                  psi0, traj_keys),
                        bob_gates, alice_logical, N)

                    # dd
                    F_dd = _decode_and_F(
                        vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                psi0, traj_keys),
                        bob_gates, alice_logical, N)

                    # local_sparse (= Program Y geometry trigger)
                    psi_local, n_gates_local, fired_local, vel_local = vmap_local(
                        evals, evecs, N, jnp.float32(dt), args.n_steps,
                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                        jnp.int32(args.T_quiet), jnp.float32(args.eps), S_proj,
                        psi0, traj_keys)
                    F_local = _decode_and_F(psi_local, bob_gates, alice_logical, N)
                    mean_gates_local = float(jnp.mean(n_gates_local))

                    fired_local_np = np.array(fired_local)
                    vel_local_np = np.array(vel_local)
                    ilc_local_vals = [
                        float(np.corrcoef(fired_local_np[i], vel_local_np[i])[0, 1])
                        for i in range(args.n_trajectories)
                        if fired_local_np[i].sum() > 1 and vel_local_np[i].std() > 1e-12
                    ]
                    ilc_local = float(np.mean(ilc_local_vals)) if ilc_local_vals else 0.0

                    # multiscale_sparse
                    psi_ms, n_gates_ms, fired_ms, vel_ms_L0, vel_ms_zz = vmap_ms(
                        evals, evecs, N, jnp.float32(dt), args.n_steps,
                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                        jnp.int32(args.T_quiet), jnp.float32(args.eps), S_proj,
                        ZZ_signs, n_pairs_total,
                        psi0, traj_keys)
                    F_ms = _decode_and_F(psi_ms, bob_gates, alice_logical, N)
                    mean_gates_ms = float(jnp.mean(n_gates_ms))

                    fired_ms_np = np.array(fired_ms)
                    vel_ms_L0_np = np.array(vel_ms_L0)
                    vel_ms_zz_np = np.array(vel_ms_zz)
                    ilc_ms_L0_vals = [
                        float(np.corrcoef(fired_ms_np[i], vel_ms_L0_np[i])[0, 1])
                        for i in range(args.n_trajectories)
                        if fired_ms_np[i].sum() > 1 and vel_ms_L0_np[i].std() > 1e-12
                    ]
                    ilc_ms_zz_vals = [
                        float(np.corrcoef(fired_ms_np[i], vel_ms_zz_np[i])[0, 1])
                        for i in range(args.n_trajectories)
                        if fired_ms_np[i].sum() > 1 and vel_ms_zz_np[i].std() > 1e-12
                    ]
                    ilc_ms_L0 = float(np.mean(ilc_ms_L0_vals)) if ilc_ms_L0_vals else 0.0
                    ilc_ms_zz = float(np.mean(ilc_ms_zz_vals)) if ilc_ms_zz_vals else 0.0

                    # random_sparse matched to local_sparse gate count
                    rng = np.random.default_rng(args.seed + msg_idx * 7777 + depth * 31337)
                    rand_schedules = np.zeros((args.n_trajectories, args.n_steps), dtype=np.int32)
                    n_gates_np = np.array(n_gates_local)
                    for i in range(args.n_trajectories):
                        k = int(n_gates_np[i])
                        if k > 0:
                            chosen = rng.choice(args.n_steps, size=k, replace=False)
                            rand_schedules[i, chosen] = 1
                    F_rand = _decode_and_F(
                        vmap_sched(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                   jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                   psi0, traj_keys, jnp.array(rand_schedules)),
                        bob_gates, alice_logical, N)

                    # Recoverability horizon
                    rec_horizon = compute_recoverability_horizon(
                        psi_local, bob_gates_by_layer, alice_logical, N)

                    delta_F_local = F_local - F_rand
                    delta_F_ms = F_ms - F_rand

                    print(f"      {label}: free={F_free:.4f} dd={F_dd:.4f} "
                          f"rand={F_rand:.4f} local={F_local:.4f} ms={F_ms:.4f} "
                          f"ΔF_local={delta_F_local:+.4f} ΔF_ms={delta_F_ms:+.4f} "
                          f"ILC_L0={ilc_local:+.3f} ILC_ZZ={ilc_ms_zz:+.3f}")

                    msg_results[label] = {
                        "F_logical": {"free": F_free, "dd": F_dd, "random_sparse": F_rand,
                                      "local_sparse": F_local, "multiscale_sparse": F_ms},
                        "delta_F_local": delta_F_local,
                        "delta_F_multiscale": delta_F_ms,
                        "ilc": {"local_L0": ilc_local, "multiscale_L0": ilc_ms_L0,
                                "multiscale_ZZ": ilc_ms_zz},
                        "mean_gates": {"local": mean_gates_local, "multiscale": mean_gates_ms},
                        "recoverability_horizon": rec_horizon,
                        "local_beats_rand": delta_F_local > 0,
                        "ms_beats_rand": delta_F_ms > 0,
                    }

                avg_delta_F_local = float(np.mean([msg_results[l]["delta_F_local"] for l in msg_labels]))
                avg_delta_F_ms = float(np.mean([msg_results[l]["delta_F_multiscale"] for l in msg_labels]))
                avg_ilc_L0 = float(np.mean([msg_results[l]["ilc"]["local_L0"] for l in msg_labels]))
                avg_ilc_zz = float(np.mean([msg_results[l]["ilc"]["multiscale_ZZ"] for l in msg_labels]))
                local_wins = sum(msg_results[l]["local_beats_rand"] for l in msg_labels)
                ms_wins = sum(msg_results[l]["ms_beats_rand"] for l in msg_labels)

                avg_F = {ctrl: float(np.mean([msg_results[l]["F_logical"][ctrl] for l in msg_labels]))
                         for ctrl in ["free","dd","random_sparse","local_sparse","multiscale_sparse"]}
                hierarchy = avg_F["local_sparse"] > avg_F["random_sparse"] > avg_F["dd"]

                print(f"    → ΔF_local={avg_delta_F_local:+.4f}  ΔF_ms={avg_delta_F_ms:+.4f}  "
                      f"ILC_L0={avg_ilc_L0:+.3f}  ILC_ZZ={avg_ilc_zz:+.3f}  "
                      f"local_wins={local_wins}/4  ms_wins={ms_wins}/4  hier={'✓' if hierarchy else '✗'}")

                all_records.append({
                    "mera_depth": depth,
                    "J_std": float(J_std),
                    "T2": float(T2),
                    "avg_F_logical": avg_F,
                    "avg_delta_F_local": avg_delta_F_local,
                    "avg_delta_F_multiscale": avg_delta_F_ms,
                    "avg_ilc_L0": avg_ilc_L0,
                    "avg_ilc_ZZ": avg_ilc_zz,
                    "local_wins": local_wins,
                    "ms_wins": ms_wins,
                    "hierarchy": hierarchy,
                    "messages": msg_results,
                    "n_gates_alice": len(alice_gates),
                })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "program_z_summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "scale_resolved_mera_recoverability",
                   "args": vars(args), "records": all_records,
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 70)
    print("Scaling Table: ΔF_local and ΔF_ms by (depth, J_std)")
    print(f"{'d':>3} {'J':>5} {'T2':>5} {'ΔF_local':>10} {'ΔF_ms':>9} "
          f"{'ILC_L0':>8} {'ILC_ZZ':>8} {'hier':>5}")
    print("-" * 70)
    for r in all_records:
        print(f"{r['mera_depth']:>3} {r['J_std']:>5.2f} {r['T2']:>5.1f} "
              f"{r['avg_delta_F_local']:>+10.4f} {r['avg_delta_F_multiscale']:>+9.4f} "
              f"{r['avg_ilc_L0']:>+8.3f} {r['avg_ilc_ZZ']:>+8.3f} "
              f"{'✓' if r['hierarchy'] else '✗':>5}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program Z: Scale-Resolved MERA Recoverability")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=500, dest="n_trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_z_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[1, 2, 3, 4], dest="depth_sweep")
    p.add_argument("--j-std-sweep", nargs="+", type=float, default=[0.6, 0.9, 1.2], dest="j_std_sweep")
    p.add_argument("--t2-sweep", nargs="+", type=float, default=[8.0, 12.0], dest="t2_sweep")
    args = p.parse_args()
    if args.smoke_test:
        args.n_steps = 10
        args.n_trajectories = 20
        args.depth_sweep = [1, 3]
        args.j_std_sweep = [0.9]
        args.t2_sweep = [8.0]
        print("[SMOKE TEST]")
    run_program_z(args)
