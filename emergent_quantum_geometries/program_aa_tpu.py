"""
Program AA — Renormalized Observable Reconstruction
====================================================
Core question: Does predictive power migrate from physical to renormalized
observables as MERA causal depth increases?

New vs Program Z:
  - Structured logical messages (GHZ, comp basis) instead of Haar random
  - Block entanglement entropy velocity trigger (MERA-native, replaces coarse_entropy)
  - ZZ_L1 and ZZ_renorm family triggers (Layer-specific)
  - Predictive power curves: Corr(O_l(t), F_recover)
  - Causal cone overlap metric Gamma_l per depth
  - High-trajectory count at fixed sweet spot (depth=2,3,4; J=0.9; T2=12)

Controllers: free, dd, random_sparse, local_Z, ZZ_L1, ZZ_renorm, entropy_block
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
from program_w_tpu import run_probe_stage_w
from program_z_tpu import (
    build_mera_gates_z, get_mera_layer_pairs, build_ZZ_sign_matrix,
    apply_gates, apply_gates_inverse, get_logical_fidelity,
    _make_initial_y, _decode_and_F, run_traj_free, run_traj_dd,
    run_traj_scheduled, run_traj_geo, get_bob_gates_by_layer,
    compute_recoverability_horizon,
)
from program_y_tpu import _haar_u4


# ---------------------------------------------------------------------------
# Structured messages
# ---------------------------------------------------------------------------

def build_structured_messages():
    """4 structured 2-qubit logical states spanning the logical subspace."""
    s2 = float(np.sqrt(2))
    return {
        "|00>":   jnp.array([1, 0, 0, 0], dtype=jnp.complex64),
        "|11>":   jnp.array([0, 0, 0, 1], dtype=jnp.complex64),
        "|GHZ+>": jnp.array([1/s2, 0, 0, 1/s2], dtype=jnp.complex64),
        "|GHZ->": jnp.array([1/s2, 0, 0, -1/s2], dtype=jnp.complex64),
    }


# ---------------------------------------------------------------------------
# ZZ family sign matrices
# ---------------------------------------------------------------------------

def build_ZZ_family_signs(layer_pairs, N=12):
    """Returns (ZZ_L1_signs, ZZ_renorm_signs): shape (n_pairs, 2^N)."""
    basis = np.arange(2 ** N)

    def _build(pairs):
        if not pairs:
            return jnp.zeros((1, 2 ** N), dtype=jnp.float32)
        s = np.zeros((len(pairs), 2 ** N), dtype=np.float32)
        for k, (i, j) in enumerate(pairs):
            bi = (basis >> (N - 1 - i)) & 1
            bj = (basis >> (N - 1 - j)) & 1
            s[k] = (1 - 2 * bi) * (1 - 2 * bj)
        return jnp.array(s)

    l1 = layer_pairs[0] if len(layer_pairs) >= 1 else []
    renorm = [p for layer in layer_pairs[1:] for p in layer]
    return _build(l1), _build(renorm)


# ---------------------------------------------------------------------------
# Causal cone overlap metric
# ---------------------------------------------------------------------------

def compute_cone_overlap(depth, N=12):
    """
    Γ_ℓ = max gate pair span at depth ℓ / (N-1).
    Higher Γ → broader causal cone → harder to steer geometrically.
    """
    layer_pairs = get_mera_layer_pairs(depth, N)
    if not layer_pairs or not layer_pairs[-1]:
        return 0.0
    max_span = max(abs(j - i) for (i, j) in layer_pairs[-1])
    return max_span / (N - 1)


# ---------------------------------------------------------------------------
# Factory-pattern runner builders (depth-varies safely via closure)
# ---------------------------------------------------------------------------

def make_ZZ_runner(ZZ_signs, n_pairs, N, n_steps):
    """Build a vmapped JAX runner that triggers on ZZ velocity for a fixed sign matrix."""

    def _run(evals, evecs, dt, p1, p2, p_cross, T_quiet, eps, psi0, rng_key):
        prob0 = jnp.abs(psi0) ** 2
        zz0 = jnp.dot(ZZ_signs, prob0)
        carry0 = (psi0, zz0, jnp.int32(999), jnp.int32(0), rng_key)

        def step(carry, s):
            psi, zz_prev, ssl, ng, key = carry
            key, _, ks = jax.random.split(key, 3)
            prob = jnp.abs(psi) ** 2
            zz_curr = jnp.dot(ZZ_signs, prob)
            vel = jnp.sum((zz_curr - zz_prev) ** 2)
            trigger = jnp.logical_and(ssl >= T_quiet,
                                      jnp.logical_and(s > 2, vel > eps))
            gi = jax.lax.cond(trigger,
                              lambda: jnp.array((3, 0)),
                              lambda: jnp.array((0, 0)))
            ci = gi[0]
            psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
            ssl_n = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
            ng_n = ng + jnp.where(ci > 0, 1, 0)
            fired = jnp.where(ci > 0, 1.0, 0.0)
            return (psi, zz_curr, ssl_n, ng_n, key), (fired, vel)

        (psi_f, _, _, ng_f, _), (fired, vels) = jax.lax.scan(
            step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
        return psi_f, ng_f, fired, vels

    return jax.jit(jax.vmap(_run,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)))


def make_entropy_runner(block_pairs, N, n_steps):
    """Build a vmapped runner triggering on 2-site entanglement entropy velocity."""
    n_blocks = len(block_pairs)
    # Pre-capture block_pairs as Python list — unrolled at JIT trace time

    def _entropy_of_blocks(psi):
        entropies = []
        for (i, j) in block_pairs:
            psi_t = psi.reshape((2,) * N)
            psi_t = jnp.moveaxis(psi_t, [i, j], [0, 1])
            psi_2d = psi_t.reshape(4, 2 ** (N - 2))
            rho = (psi_2d @ psi_2d.conj().T).real
            evals = jnp.linalg.eigvalsh(rho)
            evals = jnp.maximum(evals, 1e-10)
            S = -jnp.sum(evals * jnp.log(evals))
            entropies.append(S)
        return jnp.stack(entropies)  # (n_blocks,)

    def _run(evals, evecs, dt, p1, p2, p_cross, T_quiet, eps_S, psi0, rng_key):
        S0 = _entropy_of_blocks(psi0)
        carry0 = (psi0, S0, jnp.int32(999), jnp.int32(0), rng_key)

        def step(carry, s):
            psi, S_prev, ssl, ng, key = carry
            key, _, ks = jax.random.split(key, 3)
            S_curr = _entropy_of_blocks(psi)
            S_vel = jnp.sum((S_curr - S_prev) ** 2)
            trigger = jnp.logical_and(ssl >= T_quiet,
                                      jnp.logical_and(s > 2, S_vel > eps_S))
            gi = jax.lax.cond(trigger,
                              lambda: jnp.array((3, 0)),
                              lambda: jnp.array((0, 0)))
            ci = gi[0]
            psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
            ssl_n = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
            ng_n = ng + jnp.where(ci > 0, 1, 0)
            fired = jnp.where(ci > 0, 1.0, 0.0)
            return (psi, S_curr, ssl_n, ng_n, key), (fired, S_vel)

        (psi_f, _, _, ng_f, _), (fired, S_vels) = jax.lax.scan(
            step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
        return psi_f, ng_f, fired, S_vels

    return jax.jit(jax.vmap(_run,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)))


# ---------------------------------------------------------------------------
# Predictive power: Corr(obs_vel, F_recover) across trajectories
# ---------------------------------------------------------------------------

def compute_predictive_power(vels_np, F_np):
    """
    vels_np: (n_traj, n_steps) — observable velocity time series
    F_np:    (n_traj,)          — final logical fidelity per trajectory
    Returns: Pearson r between mean(vel) per trajectory and F
    """
    mean_vels = vels_np.mean(axis=1)  # (n_traj,)
    if mean_vels.std() < 1e-12 or F_np.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(mean_vels, F_np)[0, 1])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_program_aa(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program AA — Renormalized Observable Reconstruction")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    J_std = args.j_std
    T2 = args.t2
    dt = args.T_max / args.n_steps
    T1 = 20.0
    p1 = float(1.0 - np.exp(-dt / T1))
    p2 = float(1.0 - np.exp(-dt / T2))
    p_cross = 0.03

    messages = build_structured_messages()

    # Shared vmapped runners (don't depend on depth)
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_dd = jax.jit(jax.vmap(run_traj_dd,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_sched = jax.jit(jax.vmap(run_traj_scheduled,
        in_axes=(None,None,None,None,None,None,None,None,None,0,0)), static_argnums=(2,4))
    vmap_local_Z = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2,4))

    print(f"\nFixed sweet spot: J_std={J_std}, T2={T2}, n_traj={args.n_trajectories}")

    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, (f"AA_J{J_std}", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)

    all_records = []

    for depth in args.depth_sweep:
        print(f"\n{'=' * 70}")
        print(f"MERA depth={depth}  Γ_ℓ={compute_cone_overlap(depth, N):.3f}")
        print(f"{'=' * 70}")

        alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=args.seed)
        layer_pairs = get_mera_layer_pairs(depth, N)
        bob_gates_by_layer = get_bob_gates_by_layer(bob_gates, depth)
        print(f"  Alice={len(alice_gates)} gates  Bob={len(bob_gates)} gates")

        # ZZ family signs for this depth
        ZZ_L1, ZZ_renorm = build_ZZ_family_signs(layer_pairs, N)
        n_L1 = ZZ_L1.shape[0]
        n_renorm = ZZ_renorm.shape[0]

        # Entropy blocks: use Layer 1 pairs for entropy trigger
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]
        eps_S = jnp.float32(args.eps_entropy)

        # Build depth-specific runners
        vmap_ZZ_L1 = make_ZZ_runner(ZZ_L1, n_L1, N, args.n_steps)
        vmap_ZZ_renorm = make_ZZ_runner(ZZ_renorm, n_renorm, N, args.n_steps)
        vmap_entropy = make_entropy_runner(ent_block_pairs, N, args.n_steps)

        # S_proj for local_Z trigger
        S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site=7)

        master_key = jax.random.PRNGKey(args.seed + depth * 999983)
        traj_keys = jax.random.split(master_key, args.n_trajectories)

        depth_results = []

        for msg_label, alice_logical in messages.items():
            psi0 = _make_initial_y(N, alice_logical, alice_gates, bob_gates)

            # --- free ---
            F_free = _decode_and_F(
                vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                          jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                          psi0, traj_keys),
                bob_gates, alice_logical, N)

            # --- dd ---
            F_dd = _decode_and_F(
                vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                        psi0, traj_keys),
                bob_gates, alice_logical, N)

            # --- local_Z ---
            psi_local, ng_local, fired_local, vel_local = vmap_local_Z(
                evals, evecs, N, jnp.float32(dt), args.n_steps,
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps), S_proj,
                psi0, traj_keys)
            F_local = _decode_and_F(psi_local, bob_gates, alice_logical, N)
            F_local_np = np.array([float(get_logical_fidelity(
                apply_gates_inverse(jnp.array(psi_local[i]), bob_gates, N),
                alice_logical, N)) for i in range(args.n_trajectories)])
            PP_local = compute_predictive_power(np.array(vel_local), F_local_np)

            # --- random_sparse (matched to local_Z gate count) ---
            rng = np.random.default_rng(args.seed ^ hash(msg_label) & 0xFFFF)
            ng_local_np = np.array(ng_local)
            rand_sched = np.zeros((args.n_trajectories, args.n_steps), dtype=np.int32)
            for i in range(args.n_trajectories):
                k = int(ng_local_np[i])
                if k > 0:
                    rand_sched[i, rng.choice(args.n_steps, size=k, replace=False)] = 1
            F_rand = _decode_and_F(
                vmap_sched(evals, evecs, N, jnp.float32(dt), args.n_steps,
                           jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                           psi0, traj_keys, jnp.array(rand_sched)),
                bob_gates, alice_logical, N)

            # --- ZZ_L1 ---
            psi_zzl1, ng_zzl1, fired_zzl1, vel_zzl1 = vmap_ZZ_L1(
                evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps),
                psi0, traj_keys)
            F_zzl1 = _decode_and_F(psi_zzl1, bob_gates, alice_logical, N)
            F_zzl1_np = np.array([float(get_logical_fidelity(
                apply_gates_inverse(jnp.array(psi_zzl1[i]), bob_gates, N),
                alice_logical, N)) for i in range(args.n_trajectories)])
            PP_zzl1 = compute_predictive_power(np.array(vel_zzl1), F_zzl1_np)

            # --- ZZ_renorm ---
            psi_zzr, ng_zzr, fired_zzr, vel_zzr = vmap_ZZ_renorm(
                evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps),
                psi0, traj_keys)
            F_zzr = _decode_and_F(psi_zzr, bob_gates, alice_logical, N)
            F_zzr_np = np.array([float(get_logical_fidelity(
                apply_gates_inverse(jnp.array(psi_zzr[i]), bob_gates, N),
                alice_logical, N)) for i in range(args.n_trajectories)])
            PP_zzr = compute_predictive_power(np.array(vel_zzr), F_zzr_np)

            # --- entropy_block ---
            psi_ent, ng_ent, fired_ent, S_vels = vmap_entropy(
                evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), eps_S,
                psi0, traj_keys)
            F_ent = _decode_and_F(psi_ent, bob_gates, alice_logical, N)
            F_ent_np = np.array([float(get_logical_fidelity(
                apply_gates_inverse(jnp.array(psi_ent[i]), bob_gates, N),
                alice_logical, N)) for i in range(args.n_trajectories)])
            PP_ent = compute_predictive_power(np.array(S_vels), F_ent_np)

            # Scale-resolved recoverability for local_Z (representative)
            rec_horizon = compute_recoverability_horizon(
                psi_local, bob_gates_by_layer, alice_logical, N)

            dF_local = F_local - F_rand
            dF_zzl1 = F_zzl1 - F_rand
            dF_zzr = F_zzr - F_rand
            dF_ent = F_ent - F_rand

            print(f"  {msg_label}: rand={F_rand:.4f} "
                  f"local={dF_local:+.4f} ZZ_L1={dF_zzl1:+.4f} "
                  f"ZZ_r={dF_zzr:+.4f} ent={dF_ent:+.4f} "
                  f"PP_loc={PP_local:+.3f} PP_ZZr={PP_zzr:+.3f}")

            depth_results.append({
                "msg": msg_label,
                "F": {"free": F_free, "dd": F_dd, "rand": F_rand,
                      "local_Z": F_local, "ZZ_L1": F_zzl1,
                      "ZZ_renorm": F_zzr, "entropy_block": F_ent},
                "dF": {"local_Z": dF_local, "ZZ_L1": dF_zzl1,
                       "ZZ_renorm": dF_zzr, "entropy_block": dF_ent},
                "pred_power": {"local_Z": PP_local, "ZZ_L1": PP_zzl1,
                               "ZZ_renorm": PP_zzr, "entropy_block": PP_ent},
                "recoverability_horizon": rec_horizon,
                "mean_gates": {"local_Z": float(jnp.mean(ng_local)),
                               "ZZ_L1": float(jnp.mean(ng_zzl1)),
                               "ZZ_renorm": float(jnp.mean(ng_zzr)),
                               "entropy_block": float(jnp.mean(ng_ent))},
            })

        # Average across messages
        controllers = ["local_Z", "ZZ_L1", "ZZ_renorm", "entropy_block"]
        avg_dF = {c: float(np.mean([r["dF"][c] for r in depth_results])) for c in controllers}
        avg_PP = {c: float(np.mean([r["pred_power"][c] for r in depth_results])) for c in controllers}
        gamma = compute_cone_overlap(depth, N)

        print(f"\n  → Γ={gamma:.3f}  ΔF: local={avg_dF['local_Z']:+.4f} "
              f"ZZ_L1={avg_dF['ZZ_L1']:+.4f} ZZ_r={avg_dF['ZZ_renorm']:+.4f} "
              f"ent={avg_dF['entropy_block']:+.4f}")
        print(f"     PP: local={avg_PP['local_Z']:+.3f} "
              f"ZZ_r={avg_PP['ZZ_renorm']:+.3f} "
              f"ent={avg_PP['entropy_block']:+.3f}")

        all_records.append({
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "avg_dF": avg_dF,
            "avg_pred_power": avg_PP,
            "messages": depth_results,
        })

    out = {
        "experiment": "renormalized_observable_reconstruction",
        "args": vars(args),
        "records": all_records,
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "program_aa_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}  ({time.time() - t0:.1f}s)")

    # Print the key graph data
    print("\n" + "=" * 70)
    print("Scale Transition Table: ΔF and Predictive Power by Depth")
    print(f"  {'d':>2} {'Γ':>6} {'ΔF_loc':>8} {'ΔF_ZZL1':>8} "
          f"{'ΔF_ZZr':>8} {'ΔF_ent':>8} {'PP_loc':>7} {'PP_ZZr':>7} {'PP_ent':>7}")
    print("-" * 70)
    for r in all_records:
        d = r["mera_depth"]
        g = r["cone_overlap_gamma"]
        dF = r["avg_dF"]
        PP = r["avg_pred_power"]
        print(f"  {d:>2} {g:>6.3f} {dF['local_Z']:>+8.4f} {dF['ZZ_L1']:>+8.4f} "
              f"{dF['ZZ_renorm']:>+8.4f} {dF['entropy_block']:>+8.4f} "
              f"{PP['local_Z']:>+7.3f} {PP['ZZ_renorm']:>+7.3f} {PP['entropy_block']:>+7.3f}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program AA: Renormalized Observable Reconstruction")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005, dest="eps_entropy")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=1000, dest="n_trajectories")
    p.add_argument("--j-std", type=float, default=0.9, dest="j_std")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_aa_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3, 4], dest="depth_sweep")
    args = p.parse_args()
    if args.smoke_test:
        args.n_steps = 10
        args.n_trajectories = 20
        args.depth_sweep = [2, 3]
        print("[SMOKE TEST]")
    run_program_aa(args)
