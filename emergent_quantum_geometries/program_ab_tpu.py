"""
Program AB — Semantic Recoverability Through Hierarchical Quantum Channels
==========================================================================
Core question: Does semantic shape-recovery accuracy depend on which observable
family is used for control, and does the optimal observable migrate upward in
renormalization scale as causal depth increases?

Symbolic alphabet (4 Bell states — orthonormal, maximally entangled):
  A = Phi+  = (|00>+|11>)/√2   Z2 parity even, phase+
  B = Psi+  = (|01>+|10>)/√2   Z2 parity odd,  phase+
  C = Phi-  = (|00>-|11>)/√2   Z2 parity even, phase-
  D = Psi-  = (|01>-|10>)/√2   Z2 parity odd,  phase-

Key metrics:
  - Confusion matrix C[s_true, s_pred]
  - Shape recovery accuracy Acc = mean(diag(C))   (chance = 25%)
  - Semantic MI I(s;ŝ) in bits                    (max = 2 bits)
  - O*(ℓ) = argmax controller Acc at each depth
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
    build_mera_gates_z, get_mera_layer_pairs,
    apply_gates, apply_gates_inverse,
    _make_initial_y, run_traj_free, run_traj_dd,
    run_traj_scheduled, run_traj_geo,
)
from program_aa_tpu import (
    build_ZZ_family_signs, compute_cone_overlap,
    make_ZZ_runner, make_entropy_runner,
)


# ---------------------------------------------------------------------------
# Bell-state alphabet
# ---------------------------------------------------------------------------

def build_bell_alphabet():
    s = float(np.sqrt(2))
    return {
        "A_Phi+": jnp.array([1/s, 0, 0,  1/s], dtype=jnp.complex64),
        "B_Psi+": jnp.array([0,  1/s, 1/s, 0], dtype=jnp.complex64),
        "C_Phi-": jnp.array([1/s, 0, 0, -1/s], dtype=jnp.complex64),
        "D_Psi-": jnp.array([0,  1/s,-1/s, 0], dtype=jnp.complex64),
    }


# ---------------------------------------------------------------------------
# Classification & confusion matrix
# ---------------------------------------------------------------------------

def classify_batch(psi_f_batch, bob_gates, bell_arr, N, logical_sites=(6, 7)):
    """Return (n_traj,) int array of argmax-fidelity classifications."""
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest

    def classify_one(psi_f):
        psi_dec = apply_gates_inverse(psi_f, bob_gates, N)
        psi_t = jnp.transpose(psi_dec.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
        
        # Calculate fidelity for all 4 alphabet states
        fids = jnp.stack([
            jnp.real(jnp.sum(jnp.abs(jnp.dot(bell_arr[k].conj(), psi_t)) ** 2))
            for k in range(4)
        ])
        return jnp.argmax(fids)
        
    return jax.jit(jax.vmap(classify_one))(psi_f_batch)


def confusion_matrix(classifications_by_shape, n_traj):
    """Returns (4,4) row-normalised confusion matrix."""
    C = np.zeros((4, 4))
    for s_true, clf in enumerate(classifications_by_shape):
        for s_pred in np.asarray(clf, dtype=int):
            C[s_true, s_pred] += 1
    return C / n_traj


def semantic_MI(C):
    """I(s; ŝ) in bits (uniform prior over 4 shapes)."""
    eps = 1e-10
    log_C = np.where(C > eps, np.log2(np.maximum(C, eps)), 0.0)
    H_cond = -np.mean(np.sum(C * log_C, axis=1))   # H(ŝ|s)
    return 2.0 - H_cond                              # H(s) = log2(4) = 2


def accuracy(C):
    return float(np.mean(np.diag(C)))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_program_ab(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program AB — Semantic Recoverability Through Hierarchical Channels")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    J_std = args.j_std
    T2 = args.t2
    dt = args.T_max / args.n_steps
    T1 = 20.0
    p1 = float(1 - np.exp(-dt / T1))
    p2 = float(1 - np.exp(-dt / T2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))   # (4, 4) complex64

    print(f"\nFixed sweet spot: J_std={J_std}, T2={T2}, n_traj={args.n_trajectories}")
    print(f"Alphabet: {bell_labels}")

    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, (f"AB_J{J_std}", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)

    # Shared vmapped runners
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_dd = jax.jit(jax.vmap(run_traj_dd,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2,4))
    vmap_sched = jax.jit(jax.vmap(run_traj_scheduled,
        in_axes=(None,None,None,None,None,None,None,None,None,0,0)), static_argnums=(2,4))
    vmap_local_Z = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2,4))

    path = os.path.join(args.out_dir, "program_ab_summary.json")
    all_records = []
    completed_depths = set()

    # Load existing checkpoint if available to support seamless restart
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                checkpoint = json.load(f)
                all_records = checkpoint.get("records", [])
                completed_depths = {r["mera_depth"] for r in all_records}
                print(f"Found existing checkpoint. Resuming from completed depths: {list(completed_depths)}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Starting fresh.")

    for depth in args.depth_sweep:
        if depth in completed_depths:
            print(f"\nSkipping MERA depth={depth} (already completed in checkpoint)")
            continue

        gamma = compute_cone_overlap(depth, N)
        print(f"\n{'=' * 70}")
        print(f"MERA depth={depth}  Γ={gamma:.3f}")
        print(f"{'=' * 70}")

        alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=args.seed)
        layer_pairs = get_mera_layer_pairs(depth, N)
        ZZ_L1, ZZ_renorm = build_ZZ_family_signs(layer_pairs, N)
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]

        vmap_ZZ_L1    = make_ZZ_runner(ZZ_L1, ZZ_L1.shape[0], N, args.n_steps)
        vmap_ZZ_renorm = make_ZZ_runner(ZZ_renorm, ZZ_renorm.shape[0], N, args.n_steps)
        vmap_entropy  = make_entropy_runner(ent_block_pairs, N, args.n_steps)

        S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt), bob_site=7)

        master_key = jax.random.PRNGKey(args.seed + depth * 999983)
        traj_keys = jax.random.split(master_key, args.n_trajectories)

        # Collect classifications for each controller × each input shape
        ctrl_names = ["free", "dd", "random_sparse", "local_Z",
                      "ZZ_L1", "ZZ_renorm", "entropy_block"]
        clfs = {c: [] for c in ctrl_names}   # lists of (n_traj,) arrays, one per shape

        rng = np.random.default_rng(args.seed ^ depth)

        for s_idx, (label, alice_logical) in enumerate(bell_dict.items()):
            psi0 = _make_initial_y(N, alice_logical, alice_gates, bob_gates)

            psi_free = vmap_free(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                     jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                     psi0, traj_keys)
            clfs["free"].append(classify_batch(psi_free, bob_gates, bell_arr, N))

            psi_dd = vmap_dd(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                  jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                  psi0, traj_keys)
            clfs["dd"].append(classify_batch(psi_dd, bob_gates, bell_arr, N))

            psi_local, ng_local, *_ = vmap_local_Z(
                evals, evecs, N, jnp.float32(dt), args.n_steps,
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps), S_proj,
                psi0, traj_keys)
            clfs["local_Z"].append(classify_batch(psi_local, bob_gates, bell_arr, N))

            # Matched random schedule
            ng_np = np.array(ng_local)
            rand_sched = np.zeros((args.n_trajectories, args.n_steps), dtype=np.int32)
            for i in range(args.n_trajectories):
                k = int(ng_np[i])
                if k > 0:
                    rand_sched[i, rng.choice(args.n_steps, size=k, replace=False)] = 1
            psi_rand = vmap_sched(evals, evecs, N, jnp.float32(dt), args.n_steps,
                                       jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                       psi0, traj_keys, jnp.array(rand_sched))
            clfs["random_sparse"].append(classify_batch(psi_rand, bob_gates, bell_arr, N))

            psi_zzl1, *_ = vmap_ZZ_L1(evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps), psi0, traj_keys)
            clfs["ZZ_L1"].append(classify_batch(psi_zzl1, bob_gates, bell_arr, N))

            psi_zzr, *_ = vmap_ZZ_renorm(evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps), psi0, traj_keys)
            clfs["ZZ_renorm"].append(classify_batch(psi_zzr, bob_gates, bell_arr, N))

            psi_ent, *_ = vmap_entropy(evals, evecs, jnp.float32(dt),
                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy), psi0, traj_keys)
            clfs["entropy_block"].append(classify_batch(psi_ent, bob_gates, bell_arr, N))

            print(f"  shape {label} done")

        # Build confusion matrices and compute metrics
        depth_metrics = {}
        for ctrl in ctrl_names:
            C = confusion_matrix(clfs[ctrl], args.n_trajectories)
            acc = accuracy(C)
            mi  = semantic_MI(C)
            depth_metrics[ctrl] = {"accuracy": acc, "MI_bits": mi,
                                   "confusion": C.tolist()}
            print(f"  {ctrl:>14}: Acc={acc:.3f}  MI={mi:.3f} bits")

        best_ctrl = max(ctrl_names, key=lambda c: depth_metrics[c]["accuracy"])
        print(f"  → O*(ℓ={depth}) = {best_ctrl}  (Acc={depth_metrics[best_ctrl]['accuracy']:.3f})")

        all_records.append({
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "metrics": depth_metrics,
            "O_star": best_ctrl,
        })

        # Save checkpoint incrementally
        out = {
            "experiment": "semantic_recoverability_hierarchical",
            "args": vars(args),
            "alphabet": bell_labels,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }
        os.makedirs(args.out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved checkpoint -> {path}")

    # Summary table
    print("\n" + "=" * 70)
    print("Semantic Recovery Table: Accuracy and MI by Depth and Controller")
    print(f"  {'d':>2} {'ctrl':>15} {'Acc':>7} {'MI(bits)':>10} {'O*':>14}")
    print("-" * 70)
    for r in all_records:
        d = r["mera_depth"]
        for ctrl in ctrl_names:
            m = r["metrics"][ctrl]
            star = " ←" if ctrl == r["O_star"] else ""
            print(f"  {d:>2} {ctrl:>15} {m['accuracy']:>7.3f} {m['MI_bits']:>10.3f}{star}")
        print()
    print("=" * 70)

    # O*(ℓ) table
    print("\nPredictive Observable Migration O*(ℓ):")
    for r in all_records:
        print(f"  depth={r['mera_depth']}  Γ={r['cone_overlap_gamma']:.3f}"
              f"  O*(ℓ) = {r['O_star']}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005, dest="eps_entropy")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--n-trajectories", type=int, default=1000, dest="n_trajectories")
    p.add_argument("--j-std", type=float, default=0.9, dest="j_std")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_ab_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3, 4], dest="depth_sweep")
    args = p.parse_args()
    if args.smoke_test:
        args.n_steps = 10
        args.n_trajectories = 20
        args.depth_sweep = [2, 3]
        print("[SMOKE TEST]")
    run_program_ab(args)
