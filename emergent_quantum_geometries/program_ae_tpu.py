"""
Program AE — Dynamic Logical Frame Transport & Emergent Semantic Hydrodynamics
=============================================================================
Core question: Does semantic structure survive much longer inside a co-moving
logical frame than a static logical frame, even under scrambling-induced drift?

Scientific framing:
  In a MERA-encoded scrambling channel, relational semantic distinguishability
  is preserved modulo a dynamic, dynamically drifting logical-unitary frame.
  By tracking the co-moving frame at every sequence step using the Hungarian
  algorithm, we isolate basis scrambling from actual information destruction,
  revealing that semantic structure survives deeper than raw or statically-corrected
  accuracy suggest.
"""
import argparse, json, os, sys, time
import numpy as np
from itertools import product as iproduct

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
    _make_initial_y, run_traj_free, run_traj_geo,
)
from program_aa_tpu import (
    build_ZZ_family_signs, compute_cone_overlap,
    make_entropy_runner,
)
from program_ab_tpu import build_bell_alphabet


# ---------------------------------------------------------------------------
# Geometric sequence library
# ---------------------------------------------------------------------------

def build_sequences(bell_dict):
    """
    Relational sequence library spanning alternating, symmetric, cyclic,
    and shuffled timing controls.
    """
    keys = list(bell_dict.keys())  # [A_Phi+, B_Psi+, C_Phi-, D_Psi-]
    A, B, C, D = keys
    return {
        # Alternating family (periodic structure)
        "periodic_A":      (A, B, A, B),
        "periodic_B":      (C, D, C, D),
        
        # Palindromic family (reflection structure)
        "mirrored_A":      (A, B, B, A),
        "mirrored_B":      (C, D, D, C),
        
        # Cyclic family (orbital structure)
        "cyclic_A":        (A, B, C, D),
        "cyclic_B":        (B, C, D, A),
        
        # Temporal Shuffling Controls (destroys temporal relational timing)
        "shuffled_control": (A, A, B, B),  # Shuffled periodic_A
        "cyclic_shuffled":  (A, C, B, D),  # Shuffled cyclic_A
    }


# ---------------------------------------------------------------------------
# Logical subspace projection & reinjection
# ---------------------------------------------------------------------------

def project_to_logical(psi, N, logical_sites=(6, 7)):
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
    return psi_t  # (4, 2^(N-2))


def reinject_alice(psi, new_alice_logical, alice_gates, N, alice_sites=(0, 1)):
    rest = [s for s in range(N) if s not in alice_sites]
    perm = list(alice_sites) + rest
    inv_perm = [0] * N
    for k, p in enumerate(perm):
        inv_perm[p] = k

    # Permute so Alice sites are first
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))

    env_norm = jnp.sqrt(jnp.sum(jnp.abs(psi_t) ** 2, axis=0, keepdims=True))  # (1, 2^(N-2))
    env_norm = jnp.where(env_norm < 1e-12, jnp.ones_like(env_norm), env_norm)

    env_normed = psi_t / env_norm  # (4, 2^(N-2))
    env_avg = jnp.mean(env_normed, axis=0, keepdims=True)  # (1, 2^(N-2))
    env_avg = env_avg / jnp.linalg.norm(env_avg)

    new_psi_t = new_alice_logical[:, None] * env_norm  # (4, 2^(N-2))
    new_psi = jnp.transpose(new_psi_t.reshape((2,) * N), inv_perm).reshape(2 ** N)

    # Apply Alice's encoding gates to new state
    new_psi = apply_gates(new_psi, alice_gates, N)
    return new_psi / jnp.linalg.norm(new_psi)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_sequence(psi_seq_batch, bob_gates, bell_arr, N,
                       logical_sites=(6, 7)):
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest

    def classify_one_step(psi_f):
        psi_dec = apply_gates_inverse(psi_f, bob_gates, N)
        psi_t = jnp.transpose(psi_dec.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
        fids = jnp.stack([
            jnp.real(jnp.sum(jnp.abs(jnp.dot(bell_arr[k].conj(), psi_t)) ** 2))
            for k in range(4)
        ])
        return jnp.argmax(fids)

    def classify_one_traj(psi_seq):
        return jax.vmap(classify_one_step)(psi_seq)

    return jax.jit(jax.vmap(classify_one_traj))(psi_seq_batch)  # (n_traj, T)


# ---------------------------------------------------------------------------
# Dynamic Frame Metrics
# ---------------------------------------------------------------------------

def compute_step_transition_matrices(true_seqs, pred_seqs):
    """
    Calculate individual 4x4 transition matrices for each timestep t in [0, 1, 2, 3].
    """
    n_traj, T = true_seqs.shape
    T_mats = []
    for t in range(T):
        T_t = np.zeros((4, 4))
        for i, j in zip(true_seqs[:, t], pred_seqs[:, t]):
            T_t[int(i), int(j)] += 1
        row_sums = T_t.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        T_mats.append(T_t / row_sums)
    return T_mats


def co_moving_frame_correct(pred_seqs, T_mats):
    """
    Co-moving frame alignment: align each step t independently using the
    step-specific Hungarian mapping matrix.
    """
    from scipy.optimize import linear_sum_assignment
    n_traj, T = pred_seqs.shape
    corrected = np.zeros_like(pred_seqs)
    perms = []
    for t in range(T):
        row_ind, col_ind = linear_sum_assignment(-T_mats[t])
        perm = col_ind  # perm[i] = what true symbol i maps to at step t
        perms.append(perm)
        
        # Inverse mapping
        inv_perm = np.zeros(4, dtype=int)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        corrected[:, t] = np.vectorize(lambda x: inv_perm[x])(pred_seqs[:, t])
    return corrected, perms


def compute_global_transition_matrix(true_seqs, pred_seqs):
    T_mat = np.zeros((4, 4))
    n_traj, n_steps = true_seqs.shape
    for t in range(n_steps):
        for i, j in zip(true_seqs[:, t], pred_seqs[:, t]):
            T_mat[int(i), int(j)] += 1
    row_sums = T_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return T_mat / row_sums


def frame_correct(pred_seqs, T_mat):
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-T_mat)
    perm = col_ind
    inv_perm = np.zeros(4, dtype=int)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    corrected = np.vectorize(lambda x: inv_perm[x])(pred_seqs)
    return corrected, perm


def sequence_MI(true_seqs, pred_seqs, T_steps=4):
    n_traj = true_seqs.shape[0]
    def encode_seq(seq):
        return seq[:, 0] * 64 + seq[:, 1] * 16 + seq[:, 2] * 4 + seq[:, 3]

    true_codes = encode_seq(true_seqs)
    pred_codes = encode_seq(pred_seqs)

    joint = np.zeros((256, 256))
    for tc, pc in zip(true_codes, pred_codes):
        joint[int(tc), int(pc)] += 1
    joint /= n_traj

    p_true = joint.sum(axis=1)
    p_pred = joint.sum(axis=0)

    eps = 1e-12
    H_true = -np.sum(p_true * np.log2(np.maximum(p_true, eps)))
    H_joint = -np.sum(joint * np.log2(np.maximum(joint, eps)))
    H_pred_given_true = H_joint - H_true
    H_pred = -np.sum(p_pred * np.log2(np.maximum(p_pred, eps)))
    return float(H_pred - H_pred_given_true)


def transition_preservation(true_seqs, pred_seqs):
    n_traj, T = true_seqs.shape
    preserved = 0
    total = 0
    for t in range(T - 1):
        true_diff = (true_seqs[:, t + 1] - true_seqs[:, t]) % 4
        pred_diff = (pred_seqs[:, t + 1] - pred_seqs[:, t]) % 4
        preserved += np.sum(true_diff == pred_diff)
        total += n_traj
    return float(preserved / total) if total > 0 else 0.0


def symbol_accuracy(true_seqs, pred_seqs):
    return float(np.mean(true_seqs == pred_seqs))


# ---------------------------------------------------------------------------
# Main Sweep
# ---------------------------------------------------------------------------

def run_program_ae(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 75)
    print("Program AE — Dynamic Logical Frame Transport")
    print("=" * 75)

    N = 12
    Lx, Ly = 3, 4
    J_std = args.j_std
    T2 = args.t2
    dt_step = args.T_max / args.n_steps_per_sym
    T1 = 20.0
    p1 = float(1 - np.exp(-dt_step / T1))
    p2 = float(1 - np.exp(-dt_step / T2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))  # (4, 4)
    sequences = build_sequences(bell_dict)

    print(f"\nFixed sweep parameters: J_std={J_std}, T2={T2}, n_traj={args.n_trajectories}")
    print(f"Sequences: {list(sequences.keys())}")
    print(f"Controllers: free, local_Z, entropy_block")

    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, (f"AE_J{J_std}", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)

    # Shared JAX runner functions
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2, 4))
    vmap_local_Z = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2, 4))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "program_ae_summary.json")
    all_records = []
    completed_depths = set()

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                ckpt = json.load(f)
                all_records = ckpt.get("records", [])
                completed_depths = {r["mera_depth"] for r in all_records}
                print(f"Checkpoint found. Completed depths: {list(completed_depths)}")
        except Exception as e:
            print(f"Checkpoint load failed ({e}). Starting fresh.")

    for depth in args.depth_sweep:
        if depth in completed_depths:
            print(f"\nSkipping depth={depth} (checkpoint)")
            continue

        gamma = compute_cone_overlap(depth, N)
        print(f"\n{'=' * 75}")
        print(f"MERA depth={depth}  Gamma={gamma:.3f}")
        print(f"{'=' * 75}")

        alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=args.seed)
        layer_pairs = get_mera_layer_pairs(depth, N)
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]

        # Define reinjection closure capturing concrete alice_gates and N statically
        def reinject_fixed(psi, new_alice_logical):
            return reinject_alice(psi, new_alice_logical, alice_gates, N)
        vmap_reinject = jax.jit(jax.vmap(reinject_fixed, in_axes=(0, None)))

        vmap_entropy = make_entropy_runner(ent_block_pairs, N, args.n_steps_per_sym)
        S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt_step), bob_site=7)

        master_key = jax.random.PRNGKey(args.seed + depth * 999983)
        traj_keys = jax.random.split(master_key, args.n_trajectories)

        ctrl_names = ["free", "local_Z", "entropy_block"]
        depth_metrics = {}

        for ctrl in ctrl_names:
            print(f"\n  Controller: {ctrl}")

            true_labels_all = []
            pred_labels_all = []

            for seq_name, seq_labels in sequences.items():
                true_idx = np.array([bell_labels.index(lbl) for lbl in seq_labels])
                true_traj = np.tile(true_idx[None, :], (args.n_trajectories, 1))  # (n_traj, 4)

                # Option A: Memory-preserving Correlated Evolution
                alice_logical_0 = bell_dict[seq_labels[0]]
                psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                psi_batch = jnp.tile(psi0_0[None, :], (args.n_trajectories, 1))

                psi_finals_by_sym = []

                for sym_t in range(4):
                    if sym_t > 0:
                        new_logical = bell_dict[seq_labels[sym_t]]
                        psi_batch = vmap_reinject(psi_batch, new_logical)

                    if ctrl == "free":
                        psi_batch = vmap_free(
                            evals, evecs, N, jnp.float32(dt_step),
                            args.n_steps_per_sym,
                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                            psi_batch[0], traj_keys)

                    elif ctrl == "local_Z":
                        psi_batch, *_ = vmap_local_Z(
                            evals, evecs, N, jnp.float32(dt_step),
                            args.n_steps_per_sym,
                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                            jnp.int32(args.T_quiet), jnp.float32(args.eps),
                            S_proj, psi_batch[0], traj_keys)

                    elif ctrl == "entropy_block":
                        psi_batch, *_ = vmap_entropy(
                            evals, evecs, jnp.float32(dt_step),
                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                            jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy),
                            psi_batch[0], traj_keys)

                    psi_finals_by_sym.append(psi_batch)

                # Stack: (n_traj, T=4, 2^N)
                psi_seq_batch = jnp.stack(psi_finals_by_sym, axis=1)

                # Classify all symbols
                pred_traj = np.asarray(
                    classify_sequence(psi_seq_batch, bob_gates, bell_arr, N))

                true_labels_all.append(true_traj)
                pred_labels_all.append(pred_traj)

                acc = symbol_accuracy(true_traj, pred_traj)
                print(f"    seq={seq_name}: raw_acc={acc:.3f}")

            # Stack all sequences for global metrics
            true_all = np.vstack(true_labels_all)  # (n_traj * n_seq, 4)
            pred_all = np.vstack(pred_labels_all)

            # 1. Global static corrected accuracy
            T_mat = compute_global_transition_matrix(true_all, pred_all)
            pred_static, static_map = frame_correct(pred_all, T_mat)

            # 2. Dynamic co-moving frame corrected accuracy
            T_step_mats = compute_step_transition_matrices(true_all, pred_all)
            pred_comove, step_maps = co_moving_frame_correct(pred_all, T_step_mats)

            raw_acc     = symbol_accuracy(true_all, pred_all)
            static_acc  = symbol_accuracy(true_all, pred_static)
            comove_acc  = symbol_accuracy(true_all, pred_comove)
            seq_mi      = sequence_MI(true_all, pred_all)
            trans_pres  = transition_preservation(true_all, pred_all)

            # Frame Drift Rates between steps
            frame_drift_rates = []
            for t in range(1, 4):
                drift = float(np.mean(step_maps[t] != step_maps[t-1]))
                frame_drift_rates.append(drift)
            mean_drift_rate = float(np.mean(frame_drift_rates)) if frame_drift_rates else 0.0

            print(f"    -> raw_acc={raw_acc:.3f}  frame_acc={static_acc:.3f}  comove_acc={comove_acc:.3f}"
                  f"  seq_MI={seq_mi:.3f} bits  trans_pres={trans_pres:.3f}  drift_rate={mean_drift_rate:.3f}")
            print(f"       frame_map={static_map.tolist()}")
            print(f"       step_maps={[m.tolist() for m in step_maps]}")

            depth_metrics[ctrl] = {
                "raw_accuracy":              raw_acc,
                "frame_corrected_accuracy":  static_acc,
                "comove_accuracy":           comove_acc,
                "sequence_MI_bits":          seq_mi,
                "transition_preservation":   trans_pres,
                "frame_drift_rate":          mean_drift_rate,
                "step_permutations":         [m.tolist() for m in step_maps],
                "transition_matrix":         T_mat.tolist(),
                "frame_map":                 static_map.tolist(),
            }

        # O*(l) by comove_accuracy
        best_ctrl = max(ctrl_names,
                        key=lambda c: depth_metrics[c]["comove_accuracy"])
        print(f"\n  -> O*(l={depth}) = {best_ctrl}"
              f"  (comove_acc={depth_metrics[best_ctrl]['comove_accuracy']:.3f})")

        all_records.append({
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "metrics": depth_metrics,
            "O_star": best_ctrl,
        })

        # Save checkpoint
        out = {
            "experiment": "dynamic_logical_frame_transport",
            "args": vars(args),
            "sequences": {k: list(v) for k, v in sequences.items()},
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Checkpoint saved -> {path}")

    # Summary table
    print("\n" + "=" * 85)
    print("Dynamic Logical Frame Transport Summary (Program AE)")
    print(f"  {'d':>2} {'Gamma':>6} {'ctrl':>15} {'raw':>6} {'frame':>6} {'comove':>6}"
          f" {'seq_MI':>8} {'trans':>7} {'drift':>7}")
    print("-" * 85)
    for r in all_records:
        d = r["mera_depth"]
        g = r["cone_overlap_gamma"]
        for ctrl in ["free", "local_Z", "entropy_block"]:
            m = r["metrics"][ctrl]
            star = " <-" if ctrl == r["O_star"] else ""
            print(f"  {d:>2} {g:>6.3f} {ctrl:>15}"
                  f" {m['raw_accuracy']:>6.3f} {m['frame_corrected_accuracy']:>6.3f} {m['comove_accuracy']:>6.3f}"
                  f" {m['sequence_MI_bits']:>8.3f} {m['transition_preservation']:>7.3f}"
                  f" {m['frame_drift_rate']:>7.3f}{star}")
    print("=" * 85)

    print("\nO*(l) Migration (co-moving frame accuracy):")
    for r in all_records:
        print(f"  depth={r['mera_depth']}  Gamma={r['cone_overlap_gamma']:.3f}"
              f"  O*(l) = {r['O_star']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program AE: Dynamic Logical Frame Transport")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--n-steps-per-sym", type=int, default=40, dest="n_steps_per_sym")
    p.add_argument("--n-trajectories", type=int, default=1000, dest="n_trajectories")
    p.add_argument("--j-std", type=float, default=0.9, dest="j_std")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005, dest="eps_entropy")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_ae_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3, 4],
                   dest="depth_sweep")
    args = p.parse_args()

    if args.smoke_test:
        args.n_steps_per_sym = 10
        args.n_trajectories = 30
        args.depth_sweep = [2]
        print("[SMOKE TEST]")

    run_program_ae(args)
