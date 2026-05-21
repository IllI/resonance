"""
Program Y — MERA Logical Preservation Scaling
==============================================
Core question: Does the geometry-aware controller maintain ILC and fidelity
advantage over matched-random sparse schedules as MERA encoding depth grows?

Primary metrics:
  delta_F      = F_geo - F_rand         (geometry advantage over matched random)
  ILC_adv      = ILC_geo - ILC_rand     (causal correlation advantage)
  eta          = delta_F / mean_gates   (gate efficiency)

Sweep:
  MERA depth  : 1, 2, 3       (encoding hierarchy depth — primary axis)
  J_std       : 0.6, 0.9, 1.2 (weak / critical / localized disorder)
  T2          : 8.0, 12.0

Controllers (all compared with identical RNG seeds per trajectory):
  free            — no intervention
  dd              — dense periodic (X-X-Y-Y every 4 steps)
  geometry_sparse — quiet-window + observable velocity trigger
  random_sparse   — Poisson-distributed, matched gate count (primary baseline)
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

# ---------------------------------------------------------------------------
# MERA gate library — supports depth 1, 2, 3
# ---------------------------------------------------------------------------

def _haar_u4(rng):
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    Q = Q * (np.diag(R) / np.abs(np.diag(R)))
    return Q.astype(np.complex64)


def build_mera_gates(depth=2, seed=42):
    """
    depth=1: Layer 1 only  (3 gates/side — nearest-neighbor brick wall)
    depth=2: Layer 1+2     (5 gates/side — adds offset layer)
    depth=3: Layer 1+2+3   (7 gates/side — adds next-nearest-neighbor)
    Alice: sites 0-5.  Bob: sites 6-11.  Logical qubits: (0,1) and (6,7).
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
        return gates

    return _side_gates(0), _side_gates(6)


# ---------------------------------------------------------------------------
# Gate application helpers
# ---------------------------------------------------------------------------

def apply_2q_gate(psi, U, i, j, N):
    rest = [s for s in range(N) if s != i and s != j]
    perm = [i, j] + rest
    inv_perm = [0] * N
    for k, p in enumerate(perm):
        inv_perm[p] = k
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
    psi_t = (U @ psi_t).reshape((2,) * N)
    return jnp.transpose(psi_t, inv_perm).reshape(2 ** N)


def apply_gates(psi, gates, N):
    for U, i, j in gates:
        psi = apply_2q_gate(psi, U, i, j, N)
    return psi


def apply_gates_inverse(psi, gates, N):
    for U, i, j in reversed(gates):
        psi = apply_2q_gate(psi, U.conj().T, i, j, N)
    return psi


def get_logical_fidelity(psi, alice_logical, N=12, logical_sites=(6, 7)):
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
    overlaps = jnp.dot(alice_logical.conj(), psi_t)
    return jnp.real(jnp.sum(jnp.abs(overlaps) ** 2))


def _make_initial_y(N, alice_logical, alice_gates, bob_gates):
    psi = jnp.zeros(2 ** N, dtype=jnp.complex64)
    psi = psi.at[0].set(alice_logical[0])
    psi = psi.at[2 ** (N - 2)].set(alice_logical[1])
    psi = psi.at[2 ** (N - 1)].set(alice_logical[2])
    psi = psi.at[2 ** (N - 1) + 2 ** (N - 2)].set(alice_logical[3])
    psi = apply_gates(psi, alice_gates, N)
    psi = apply_gates(psi, bob_gates, N)
    return psi / jnp.linalg.norm(psi)


def _decode_and_F(psi_batch, bob_gates, alice_logical, N):
    vals = []
    for psi in psi_batch:
        psi_dec = apply_gates_inverse(jnp.array(psi), bob_gates, N)
        vals.append(float(get_logical_fidelity(psi_dec, alice_logical, N)))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Trajectory runners
# ---------------------------------------------------------------------------

def run_traj_free(evals, evecs, N, dt, n_steps, p1, p2, p_cross, psi0, rng_key):
    def step(carry, _):
        psi, key = carry
        key, _, ks = jax.random.split(key, 3)
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, jnp.array((0, 0)), ks)
        return (psi, key), None
    (psi_f, _), _ = jax.lax.scan(step, (psi0, rng_key), jnp.arange(n_steps, dtype=jnp.int32))
    return psi_f


def run_traj_dd(evals, evecs, N, dt, n_steps, p1, p2, p_cross, psi0, rng_key):
    def step(carry, s):
        psi, key = carry
        key, _, ks = jax.random.split(key, 3)
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, ctrl_dd(s), ks)
        return (psi, key), None
    (psi_f, _), _ = jax.lax.scan(step, (psi0, rng_key), jnp.arange(n_steps, dtype=jnp.int32))
    return psi_f


def run_traj_scheduled(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                        psi0, rng_key, gate_schedule):
    def step(carry, args):
        psi, key = carry
        s, fire = args
        key, _, ks = jax.random.split(key, 3)
        gi = jax.lax.cond(fire > 0, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
        return (psi, key), None
    xs = (jnp.arange(n_steps, dtype=jnp.int32), gate_schedule)
    (psi_f, _), _ = jax.lax.scan(step, (psi0, rng_key), xs)
    return psi_f


def run_traj_geo(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                 T_quiet, eps, S_proj, psi0, rng_key):
    obs0 = get_causal_observables_psi(psi0, N)
    obs_hist0 = jnp.stack([obs0, obs0, obs0])
    carry0 = (psi0, obs_hist0, jnp.int32(999), jnp.int32(0), rng_key)

    def step(carry, s):
        psi, obs_hist, ssl, n_gates, key = carry
        key, _, ks = jax.random.split(key, 3)
        gi = ctrl_geometry_sparse(s, obs_hist, ssl, T_quiet, eps, S_proj)
        ci = gi[0]
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
        ssl_next = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
        n_next = n_gates + jnp.where(ci > 0, 1, 0)
        obs_curr = get_causal_observables_psi(psi, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr])
        fired = jnp.where(ci > 0, 1.0, 0.0)
        vel_norm = jnp.sum(jnp.dot(S_proj, obs_hist[2] - obs_hist[1]) ** 2)
        return (psi, obs_hist_next, ssl_next, n_next, key), (fired, vel_norm)

    (psi_f, _, _, n_gates_total, _), (fired_mask, vel_norms) = jax.lax.scan(
        step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return psi_f, n_gates_total, fired_mask, vel_norms


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_program_y(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program Y — MERA Logical Preservation Scaling")
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

    # Build vmapped runners once (static argnums: N=2, n_steps=4)
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2, 4))
    vmap_dd = jax.jit(jax.vmap(run_traj_dd,
        in_axes=(None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2, 4))
    vmap_sched = jax.jit(jax.vmap(run_traj_scheduled,
        in_axes=(None,None,None,None,None,None,None,None,None,0,0)),
        static_argnums=(2, 4))
    vmap_geo = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2, 4))

    all_records = []

    for depth in args.depth_sweep:
        print(f"\n{'='*70}")
        print(f"MERA depth={depth}")
        print(f"{'='*70}")
        alice_gates, bob_gates = build_mera_gates(depth=depth, seed=args.seed)
        n_alice_gates = len(alice_gates)
        print(f"  Alice gates={n_alice_gates}  Bob gates={len(bob_gates)}")

        for J_std in args.j_std_sweep:
            print(f"\n  == J_std={J_std:.2f} ==")
            H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
            evals_np, evecs_np = _get_eigh(H, (f"Y_d{depth}_J{J_std:.2f}", N, args.seed))
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
                    psi_free = vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                         jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                         psi0, traj_keys)
                    F_free = _decode_and_F(psi_free, bob_gates, alice_logical, N)

                    # dd
                    psi_dd = vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                     jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                     psi0, traj_keys)
                    F_dd = _decode_and_F(psi_dd, bob_gates, alice_logical, N)

                    # geometry_sparse
                    psi_geo, n_gates_batch, fired_masks, vel_norms_batch = vmap_geo(
                        evals, evecs, N, jnp.float32(dt), args.n_steps,
                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                        jnp.int32(args.T_quiet), jnp.float32(args.eps), S_proj,
                        psi0, traj_keys)
                    F_geo = _decode_and_F(psi_geo, bob_gates, alice_logical, N)
                    mean_gates = float(jnp.mean(n_gates_batch))

                    fired_np = np.array(fired_masks)
                    vels_np = np.array(vel_norms_batch)
                    ilc_geo_vals = [
                        float(np.corrcoef(fired_np[i], vels_np[i])[0, 1])
                        for i in range(args.n_trajectories)
                        if fired_np[i].sum() > 1 and vels_np[i].std() > 1e-12
                    ]
                    ilc_geo = float(np.mean(ilc_geo_vals)) if ilc_geo_vals else 0.0

                    # random_sparse (matched gate count)
                    rng = np.random.default_rng(args.seed + msg_idx * 7777 + depth * 31337)
                    rand_schedules = np.zeros((args.n_trajectories, args.n_steps), dtype=np.int32)
                    n_gates_np = np.array(n_gates_batch)
                    for i in range(args.n_trajectories):
                        k = int(n_gates_np[i])
                        if k > 0:
                            chosen = rng.choice(args.n_steps, size=k, replace=False)
                            rand_schedules[i, chosen] = 1

                    psi_rand = vmap_sched(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                          jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                          psi0, traj_keys, jnp.array(rand_schedules))
                    F_rand = _decode_and_F(psi_rand, bob_gates, alice_logical, N)

                    ilc_rand_vals = [
                        float(np.corrcoef(rand_schedules[i].astype(float), vels_np[i])[0, 1])
                        for i in range(args.n_trajectories)
                        if rand_schedules[i].sum() > 1 and vels_np[i].std() > 1e-12
                    ]
                    ilc_rand = float(np.mean(ilc_rand_vals)) if ilc_rand_vals else 0.0

                    delta_F = F_geo - F_rand
                    ilc_adv = ilc_geo - ilc_rand
                    eta = delta_F / max(mean_gates, 1e-6)

                    print(f"      {label}: free={F_free:.4f} dd={F_dd:.4f} "
                          f"rand={F_rand:.4f} geo={F_geo:.4f} "
                          f"ΔF={delta_F:+.4f} ILC_adv={ilc_adv:+.3f} "
                          f"gates={mean_gates:.1f}")

                    msg_results[label] = {
                        "F_logical": {"free": F_free, "dd": F_dd,
                                      "random_sparse": F_rand, "geometry_sparse": F_geo},
                        "delta_F": delta_F,
                        "ilc": {"geometry_sparse": ilc_geo, "random_sparse": ilc_rand,
                                "advantage": ilc_adv},
                        "eta": eta,
                        "mean_gates": mean_gates,
                        "geo_beats_rand": delta_F > 0,
                    }

                avg = {ctrl: float(np.mean([msg_results[lbl]["F_logical"][ctrl]
                                            for lbl in msg_labels]))
                       for ctrl in ["free", "dd", "random_sparse", "geometry_sparse"]}
                avg_delta_F = float(np.mean([msg_results[lbl]["delta_F"] for lbl in msg_labels]))
                avg_ilc_adv = float(np.mean([msg_results[lbl]["ilc"]["advantage"]
                                             for lbl in msg_labels]))
                geo_wins = sum(msg_results[lbl]["geo_beats_rand"] for lbl in msg_labels)
                hierarchy = avg["geometry_sparse"] > avg["random_sparse"] > avg["dd"]

                print(f"    → avg: ΔF={avg_delta_F:+.4f}  ILC_adv={avg_ilc_adv:+.3f}  "
                      f"geo_wins={geo_wins}/4  hierarchy={'✓' if hierarchy else '✗'}")

                all_records.append({
                    "mera_depth": depth,
                    "J_std": float(J_std),
                    "T2": float(T2),
                    "avg_F_logical": avg,
                    "avg_delta_F": avg_delta_F,
                    "avg_ilc_advantage": avg_ilc_adv,
                    "geo_wins": geo_wins,
                    "hierarchy": hierarchy,
                    "messages": msg_results,
                    "n_alice_gates": n_alice_gates,
                })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "program_y_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "mera_logical_preservation_scaling",
            "args": vars(args),
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time() - t0:.1f}s)")

    # Scaling summary table
    print("\n" + "=" * 70)
    print("Scaling Table: avg_delta_F by (depth, J_std)")
    print(f"{'depth':>6} {'J_std':>7} {'T2':>5} {'avg_ΔF':>8} {'ILC_adv':>8} {'wins':>5} {'hier':>5}")
    print("-" * 70)
    for r in all_records:
        print(f"{r['mera_depth']:>6} {r['J_std']:>7.2f} {r['T2']:>5.1f} "
              f"{r['avg_delta_F']:>+8.4f} {r['avg_ilc_advantage']:>+8.3f} "
              f"{r['geo_wins']:>5}/4 {'✓' if r['hierarchy'] else '✗':>5}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program Y: MERA Logical Preservation Scaling")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--eps", type=float, default=0.010, dest="eps")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=500, dest="n_trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_y_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[1, 2, 3], dest="depth_sweep")
    p.add_argument("--j-std-sweep", nargs="+", type=float, default=[0.6, 0.9, 1.2],
                   dest="j_std_sweep")
    p.add_argument("--t2-sweep", nargs="+", type=float, default=[8.0, 12.0], dest="t2_sweep")
    args = p.parse_args()
    if args.smoke_test:
        args.n_steps = 10
        args.n_trajectories = 20
        args.depth_sweep = [1, 2]
        args.j_std_sweep = [0.9]
        args.t2_sweep = [8.0]
        print("[SMOKE TEST]")
    run_program_y(args)
