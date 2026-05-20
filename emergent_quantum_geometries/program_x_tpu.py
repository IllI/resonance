"""
Program X — Hierarchical Encoded Message Recovery
===================================================
Shallow MERA (2-layer, 12 physical qubits) as encoding/decoding circuit.
Alice: sites 0-5.  Bob: sites 6-11.  Logical qubits: Alice(0,1), Bob(6,7).
Transport: XXZ scrambling + T1/T2 noise.  Controllers: free, dd, random_sparse, geometry_sparse.
Primary metric: F_logical = fidelity of Bob's decoded 2-qubit state with Alice's input.
Documented separately from Programs Q-W paper results.
"""
import argparse, json, os, sys, time
import numpy as np
from functools import partial

try:
    import jax, jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import _get_eigh, select_backend
from program_t_tpu import build_grid_hamiltonian
from program_v_tpu import make_initial_state, get_causal_observables_psi, run_step_psi
from program_w_geometry import ctrl_geometry_sparse, ctrl_dd, build_poisson_schedule
from program_w_tpu import run_probe_stage_w

# ---------------------------------------------------------------------------
# MERA gate library
# ---------------------------------------------------------------------------

def _haar_u4(rng):
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    Q = Q * (np.diag(R) / np.abs(np.diag(R)))
    return Q.astype(np.complex64)


def build_mera_gates(depth=2, seed=42):
    """
    Returns (alice_gates, bob_gates) where each is a list of (U_jax, i, j).
    Alice: sites 0-5 (2 layers of brick-wall).  Bob: sites 6-11.
    """
    rng = np.random.default_rng(seed)

    def _side_gates(offset, rng):
        gates = []
        # Layer 1: pairs (offset, offset+1), (offset+2, offset+3), (offset+4, offset+5)
        for s in range(offset, offset + 6, 2):
            gates.append((jnp.array(_haar_u4(rng)), s, s + 1))
        # Layer 2: pairs (offset+1, offset+2), (offset+3, offset+4)
        for s in range(offset + 1, offset + 5, 2):
            gates.append((jnp.array(_haar_u4(rng)), s, s + 1))
        if depth >= 3:
            # Layer 3: (offset, offset+2), (offset+2, offset+4)
            for s in range(offset, offset + 4, 2):
                gates.append((jnp.array(_haar_u4(rng)), s, s + 2))
        return gates

    alice_gates = _side_gates(0, rng)
    bob_gates = _side_gates(6, rng)
    return alice_gates, bob_gates


def apply_2q_gate(psi, U, i, j, N):
    """Apply 4x4 unitary U to sites (i, j) of state psi (shape 2^N)."""
    psi_t = psi.reshape((2,) * N)
    rest = [s for s in range(N) if s != i and s != j]
    perm = [i, j] + rest
    inv_perm = [0] * N
    for k, p in enumerate(perm):
        inv_perm[p] = k
    psi_t = jnp.transpose(psi_t, perm).reshape(4, 2 ** (N - 2))
    psi_t = (U @ psi_t).reshape((2,) * N)
    return jnp.transpose(psi_t, inv_perm).reshape(2 ** N)


def apply_gates(psi, gates, N):
    """Apply a list of (U, i, j) gates in order."""
    for U, i, j in gates:
        psi = apply_2q_gate(psi, U, i, j, N)
    return psi


def apply_gates_inverse(psi, gates, N):
    """Apply conjugate-transpose of gate list in reverse order."""
    for U, i, j in reversed(gates):
        psi = apply_2q_gate(psi, U.conj().T, i, j, N)
    return psi


def get_logical_fidelity(psi, alice_logical_state, N=12, logical_sites=(6, 7)):
    """F_logical = <psi_A|rho_logical|psi_A> on Bob's logical sites after decoding."""
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
    overlaps = jnp.dot(alice_logical_state.conj(), psi_t)
    return jnp.real(jnp.sum(jnp.abs(overlaps) ** 2))


# ---------------------------------------------------------------------------
# Trajectory runners
# ---------------------------------------------------------------------------

def _make_initial_x(N, alice_logical, alice_gates, bob_gates):
    """
    Prepare MERA-encoded initial state:
      - Alice's logical qubits (sites 0,1) = alice_logical (4-component vector)
      - Ancillas = |0>
      - Apply Alice's encoding gates, then Bob's (Bob starts in |0>^6)
    """
    psi = jnp.zeros(2 ** N, dtype=jnp.complex64)
    psi = psi.at[0].set(1.0)
    # Place Alice's logical state onto sites 0,1
    psi_t = psi.reshape((2,) * N)
    # Build 2-qubit state tensor for sites 0,1
    al = alice_logical.reshape(2, 2)
    # outer product with remaining |0>
    for bit in range(2, N):
        al = jnp.stack([al, jnp.zeros_like(al)], axis=-1)
    psi = jnp.zeros(2 ** N, dtype=jnp.complex64)
    # Simpler: project onto alice_logical at sites 0,1
    psi = psi.at[0].set(alice_logical[0])  # |00...0>
    psi = psi.at[2 ** (N - 2)].set(alice_logical[1])   # |01 00...0> — site 1=1
    psi = psi.at[2 ** (N - 1)].set(alice_logical[2])   # |10 00...0> — site 0=1
    psi = psi.at[2 ** (N - 1) + 2 ** (N - 2)].set(alice_logical[3])  # |11 00..>
    # Apply MERA encoding
    psi = apply_gates(psi, alice_gates, N)
    psi = apply_gates(psi, bob_gates, N)
    return psi / jnp.linalg.norm(psi)


# Runners return psi_final only — MERA decode + F_logical happen outside vmap
# so Python-side gate site indices are never JAX-traced.

def run_trajectory_x_free(evals, evecs, N, dt, n_steps, p1, p2, p_cross, psi0, rng_key):
    carry0 = (psi0, rng_key)

    def scan_step(carry, step_idx):
        psi, key = carry
        key, _, ks = jax.random.split(key, 3)
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, jnp.array((0, 0)), ks)
        return (psi, key), None

    (psi_final, _), _ = jax.lax.scan(scan_step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return psi_final


def run_trajectory_x_dd(evals, evecs, N, dt, n_steps, p1, p2, p_cross, psi0, rng_key):
    carry0 = (psi0, rng_key)

    def scan_step(carry, step_idx):
        psi, key = carry
        key, _, ks = jax.random.split(key, 3)
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, ctrl_dd(step_idx), ks)
        return (psi, key), None

    (psi_final, _), _ = jax.lax.scan(scan_step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return psi_final


def run_trajectory_x_scheduled(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                                psi0, rng_key, gate_schedule):
    carry0 = (psi0, rng_key)

    def scan_step(carry, args):
        psi, key = carry
        step_idx, should_fire = args
        key, _, ks = jax.random.split(key, 3)
        gate_info = jax.lax.cond(should_fire > 0,
                                  lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, ks)
        return (psi, key), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    (psi_final, _), _ = jax.lax.scan(scan_step, carry0, (steps, gate_schedule))
    return psi_final


def run_trajectory_x_geo(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                          T_quiet, eps, S_proj, psi0, rng_key):
    obs0 = get_causal_observables_psi(psi0, N)
    obs_hist0 = jnp.stack([obs0, obs0, obs0])
    carry0 = (psi0, obs_hist0, jnp.int32(999), jnp.int32(0), rng_key)

    def scan_step(carry, step_idx):
        psi, obs_hist, ssl, n_gates, key = carry
        key, _, ks = jax.random.split(key, 3)
        gate_info = ctrl_geometry_sparse(step_idx, obs_hist, ssl, T_quiet, eps, S_proj)
        ci = gate_info[0]
        psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, ks)
        ssl_next = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        obs_curr = get_causal_observables_psi(psi, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr])
        fired = jnp.where(ci > 0, 1.0, 0.0)
        vel_norm = jnp.sum(jnp.dot(S_proj, obs_hist[2] - obs_hist[1]) ** 2)
        return (psi, obs_hist_next, ssl_next, n_gates_next, key), (fired, vel_norm)

    (psi_final, _, _, n_gates_total, _), (fired_mask, vel_norms) = jax.lax.scan(
        scan_step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return psi_final, n_gates_total, fired_mask, vel_norms


def _decode_and_F(psi_batch, bob_gates, alice_logical, N):
    """Python-side: decode psi_batch (n_traj, 2^N) and return mean F_logical."""
    F_vals = []
    for psi in psi_batch:
        psi_dec = apply_gates_inverse(jnp.array(psi), bob_gates, N)
        F_vals.append(float(get_logical_fidelity(psi_dec, alice_logical, N)))
    return float(np.mean(F_vals))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_program_x(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program X — Hierarchical Encoded Message Recovery")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    dt = args.T_max / args.n_steps
    T1 = 20.0
    p1 = float(1.0 - np.exp(-dt / T1))
    T_quiet = args.T_quiet

    eps = args.eps



    # MERA gates (fixed, seeded)
    alice_gates, bob_gates = build_mera_gates(depth=args.mera_depth, seed=args.seed)
    print(f"MERA depth={args.mera_depth}  Alice gates={len(alice_gates)}  Bob gates={len(bob_gates)}")

    # 4 logical messages: computational basis of 2 logical qubits
    logical_bases = [
        jnp.array([1, 0, 0, 0], dtype=jnp.complex64),  # |00>
        jnp.array([0, 1, 0, 0], dtype=jnp.complex64),  # |01>
        jnp.array([0, 0, 1, 0], dtype=jnp.complex64),  # |10>
        jnp.array([0, 0, 0, 1], dtype=jnp.complex64),  # |11>
    ]
    msg_labels = ["|00>", "|01>", "|10>", "|11>"]

    # Vmapped runners — return psi_final batch; decode happens Python-side
    # Args: evals,evecs,N,dt,n_steps,p1,p2,p_cross,psi0,rng_key  (10 args, N/n_steps static)
    vmap_free = jax.jit(jax.vmap(
        run_trajectory_x_free,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    # same signature
    vmap_dd = jax.jit(jax.vmap(
        run_trajectory_x_dd,
        in_axes=(None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    # Args: ...(8 shared), psi0, rng_key, gate_schedule  (11 args)
    vmap_sched = jax.jit(jax.vmap(
        run_trajectory_x_scheduled,
        in_axes=(None, None, None, None, None, None, None, None, None, 0, 0)
    ), static_argnums=(2, 4))

    # Args: ...(8 shared), T_quiet, eps, S_proj, psi0, rng_key  (13 args)
    vmap_geo = jax.jit(jax.vmap(
        run_trajectory_x_geo,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, 0)
    ), static_argnums=(2, 4))

    all_records = []

    for J_std in args.j_std_sweep:
        print(f"\n== J_std={J_std:.2f} ==")
        H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
        evals_np, evecs_np = _get_eigh(H, (f"X_J{J_std:.2f}", N, args.seed))
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for T2 in args.t2_sweep:
            p2 = float(1.0 - np.exp(-dt / T2))
            print(f"  T2={T2:.1f}")

            S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site=7)

            master_key = jax.random.PRNGKey(args.seed + int(J_std * 1000) + int(T2 * 100))
            traj_keys = jax.random.split(master_key, args.n_trajectories)

            msg_results = {}
            for msg_idx, alice_logical in enumerate(logical_bases):
                label = msg_labels[msg_idx]

                # Prepare encoded initial state
                psi0 = _make_initial_x(N, alice_logical, alice_gates, bob_gates)

                # --- free ---
                psi_free_batch = vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                           jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                                           psi0, traj_keys)
                F_free = _decode_and_F(psi_free_batch, bob_gates, alice_logical, N)

                # --- dd ---
                psi_dd_batch = vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                       jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                                       psi0, traj_keys)
                F_dd = _decode_and_F(psi_dd_batch, bob_gates, alice_logical, N)

                # --- geometry_sparse ---
                psi_geo_batch, n_gates_batch, fired_masks, vel_norms_batch = vmap_geo(
                    evals, evecs, N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(0.03),
                    jnp.int32(T_quiet), jnp.float32(eps), S_proj,
                    psi0, traj_keys)
                F_geo = _decode_and_F(psi_geo_batch, bob_gates, alice_logical, N)
                mean_gates = float(jnp.mean(n_gates_batch))

                # ILC
                fired_np = np.array(fired_masks)
                vels_np = np.array(vel_norms_batch)
                ilc_vals = [float(np.corrcoef(fired_np[i], vels_np[i])[0, 1])
                            for i in range(args.n_trajectories)
                            if fired_np[i].sum() > 1 and vels_np[i].std() > 1e-12]
                ilc_geo = float(np.mean(ilc_vals)) if ilc_vals else 0.0

                # --- random_sparse (matched gate count) ---
                rng = np.random.default_rng(args.seed + msg_idx * 7777)
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
                F_rand = _decode_and_F(psi_rand_batch, bob_gates, alice_logical, N)

                ilc_rand_vals = [float(np.corrcoef(rand_schedules[i].astype(float), vels_np[i])[0, 1])
                                 for i in range(args.n_trajectories)
                                 if rand_schedules[i].sum() > 1 and vels_np[i].std() > 1e-12]
                ilc_rand = float(np.mean(ilc_rand_vals)) if ilc_rand_vals else 0.0

                timing_adv = F_geo - F_rand
                k = max(mean_gates, 1e-6)
                eta = (F_geo - F_free) / k

                print(f"    {label}: free={F_free:.4f} dd={F_dd:.4f} rand={F_rand:.4f} "
                      f"geo={F_geo:.4f} Δ={timing_adv:+.4f} ILC_adv={ilc_geo-ilc_rand:+.3f} "
                      f"η={eta:+.4f} gates={mean_gates:.1f}")

                msg_results[label] = {
                    "F_logical": {"free": F_free, "dd": F_dd, "random_sparse": F_rand, "geometry_sparse": F_geo},
                    "timing_advantage": timing_adv,
                    "ilc": {"geometry_sparse": ilc_geo, "random_sparse": ilc_rand, "advantage": ilc_geo - ilc_rand},
                    "eta_logical": eta,
                    "mean_gates": mean_gates,
                    "geo_beats_rand": timing_adv > 0,
                }

            # Message-averaged F_logical
            avg = {ctrl: float(np.mean([msg_results[lbl]["F_logical"][ctrl] for lbl in msg_labels]))
                   for ctrl in ["free", "dd", "random_sparse", "geometry_sparse"]}
            geo_win_rate = sum(msg_results[lbl]["geo_beats_rand"] for lbl in msg_labels)

            record = {
                "J_std": float(J_std), "T2": float(T2),
                "mera_depth": args.mera_depth,
                "avg_F_logical": avg,
                "messages": msg_results,
                "geo_wins": geo_win_rate,
                "hierarchy": avg["geometry_sparse"] > avg["random_sparse"] > avg["dd"],
            }
            all_records.append(record)
            print(f"  → avg F_logical: free={avg['free']:.4f} dd={avg['dd']:.4f} "
                  f"rand={avg['random_sparse']:.4f} geo={avg['geometry_sparse']:.4f}  "
                  f"geo_wins={geo_win_rate}/4  "
                  f"hierarchy={'✓' if record['hierarchy'] else '✗'}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_x_depth{args.mera_depth}_results.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "hierarchical_encoded_message_recovery",
                   "args": vars(args), "records": all_records,
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time() - t0:.1f}s)")

    # Summary table
    hier_ok = sum(1 for r in all_records if r["hierarchy"])
    print(f"\nHierarchy (geo>rand>DD): {hier_ok}/{len(all_records)} configs")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--eps", type=float, default=0.005, dest="eps")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")

    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=300, dest="n_trajectories")
    p.add_argument("--mera-depth", type=int, default=2, dest="mera_depth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_x_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--j-std-sweep", nargs="+", type=float, default=[0.8, 1.0, 1.2], dest="j_std_sweep")
    p.add_argument("--t2-sweep", nargs="+", type=float, default=[8.0, 12.0], dest="t2_sweep")
    args = p.parse_args()
    if args.smoke_test:
        args.n_steps, args.n_trajectories = 10, 20
        args.j_std_sweep, args.t2_sweep = [1.0], [8.0]
        print("[SMOKE TEST]")
    run_program_x(args)
