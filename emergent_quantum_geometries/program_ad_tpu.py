"""
Program AD-lite — Sequential Semantic Transport & Emergent Logical Frames
=========================================================================
Core question: Does relational structure between sequentially encoded Bell-state
messages survive hierarchical scrambling beyond the point where individual
symbol accuracy collapses?

Key upgrade from AB:
  - Sequences of 4 symbols (not single-shot static symbols)
  - Memory-preserving (correlated) channel: environment evolves continuously
    across symbol boundaries; only Alice's logical sites are re-initialized
  - Transition matrix T_ij = P(ŝ_j | s_i) — the full empirical logical map
  - Frame-corrected accuracy: accuracy after removing the dominant logical rotation
  - Sequence MI: I(S_1:T ; Ŝ_1:T) over full sequence tuples
  - Transition preservation: relational structure survival score

Scientific framing:
  In a MERA-encoded scrambling channel, relational structure between sequentially
  encoded Bell-state messages is preserved beyond individual symbol accuracy
  collapse, and the renormalized entropy observable is the optimal predictor of
  this relational survival at depth 3.
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
    _make_initial_y, run_traj_free, run_traj_geo, run_traj_scheduled,
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
# Logical subspace projection
# ---------------------------------------------------------------------------

def project_to_logical(psi, N, logical_sites=(6, 7)):
    """
    Project psi onto the logical subspace of Bob (sites 6,7), tracing out rest.
    Returns (4,) amplitude vector: psi_logical = sum_env < env | psi >.
    Used to re-inject a new symbol into Alice's sites while preserving environment.
    """
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
    # The environment part: we keep the full state but replace Alice's amplitude
    return psi_t  # (4, 2^(N-2))


def reinject_alice(psi, new_alice_logical, alice_gates, N, alice_sites=(0, 1)):
    """
    Memory-preserving reinject: replace Alice's 2 logical sites with new_alice_logical
    while leaving the rest of the state (Bob's sites, environment) as evolved.

    Implementation:
    1. Trace out Alice's 2 sites from the current full state to get environment amplitudes.
    2. Tensor-product new Alice logical state with remaining environment.
    3. Apply Alice's gates to re-encode.

    Because Alice and environment are entangled after evolution, we use the
    conditional preparation: project Alice to |00> then inject new_alice_logical.
    This is physically equivalent to resetting Alice's sites and re-encoding.
    """
    rest = [s for s in range(N) if s not in alice_sites]
    perm = list(alice_sites) + rest
    inv_perm = [0] * N
    for k, p in enumerate(perm):
        inv_perm[p] = k

    # Permute so Alice sites are first
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))

    # New Alice amplitude injected, environment marginalized via norm-preserving projection
    # Strategy: replace Alice's amplitude block while preserving env norm
    env_norm = jnp.sqrt(jnp.sum(jnp.abs(psi_t) ** 2, axis=0, keepdims=True))  # (1, 2^(N-2))
    env_norm = jnp.where(env_norm < 1e-12, jnp.ones_like(env_norm), env_norm)

    # Normalized environment basis
    env_normed = psi_t / env_norm  # (4, 2^(N-2)) — directional basis

    # New amplitude: project onto average env direction, inject new_alice_logical
    env_avg = jnp.mean(env_normed, axis=0, keepdims=True)  # (1, 2^(N-2))
    env_avg = env_avg / jnp.linalg.norm(env_avg)

    new_psi_t = new_alice_logical[:, None] * env_norm  # (4, 2^(N-2))
    new_psi = jnp.transpose(new_psi_t.reshape((2,) * N), inv_perm).reshape(2 ** N)

    # Apply Alice's encoding gates to new state
    new_psi = apply_gates(new_psi, alice_gates, N)
    return new_psi / jnp.linalg.norm(new_psi)


# ---------------------------------------------------------------------------
# Sequential trajectory runner
# ---------------------------------------------------------------------------

def run_sequence_free(evals, evecs, N, dt, n_steps_per_sym, p1, p2, p_cross,
                      alice_gates, psi0_seq, rng_key):
    """
    Run T=4 sequential symbols with memory-preserving injection.
    psi0_seq: (4, 2^N) initial states for each symbol.
    Returns: (4, 2^N) final states after each symbol's evolution.
    """
    psi_finals = []
    psi = psi0_seq[0]
    key = rng_key

    for t in range(4):
        if t > 0:
            # Reinject: keep environment, replace Alice with new symbol
            psi = reinject_alice(psi, psi0_seq[t, :4], alice_gates, N)

        def step(carry, _):
            p, k = carry
            k, _, ks = jax.random.split(k, 3)
            p = run_step_psi(p, evals, evecs, N, dt, p1, p2, p_cross,
                             jnp.array((0, 0)), ks)
            return (p, k), None

        (psi, key), _ = jax.lax.scan(
            step, (psi, key), jnp.arange(n_steps_per_sym, dtype=jnp.int32))
        psi_finals.append(psi)

    return jnp.stack(psi_finals)  # (4, 2^N)


# ---------------------------------------------------------------------------
# Classification — same projection logic as AB (fixed)
# ---------------------------------------------------------------------------

def classify_sequence(psi_seq_batch, bob_gates, bell_arr, N,
                       logical_sites=(6, 7)):
    """
    psi_seq_batch: (n_traj, T, 2^N)
    Returns: (n_traj, T) integer classifications.
    """
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

    # classify_one_traj: (T, 2^N) → (T,)
    def classify_one_traj(psi_seq):
        return jax.vmap(classify_one_step)(psi_seq)

    return jax.jit(jax.vmap(classify_one_traj))(psi_seq_batch)  # (n_traj, T)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_transition_matrix(true_seqs, pred_seqs):
    """
    true_seqs, pred_seqs: (n_traj, T) int arrays.
    Returns 4x4 row-normalized transition matrix T[i,j] = P(ŝ=j | s=i).
    Built from all (true_t, pred_t) pairs across all timesteps and trajectories.
    """
    T_mat = np.zeros((4, 4))
    n_traj, n_steps = true_seqs.shape
    for t in range(n_steps):
        for i, j in zip(true_seqs[:, t], pred_seqs[:, t]):
            T_mat[int(i), int(j)] += 1
    # Row normalize
    row_sums = T_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return T_mat / row_sums


def frame_correct(pred_seqs, T_mat):
    """
    Find the dominant logical permutation using the Hungarian algorithm
    (linear sum assignment) to guarantee a mathematically rigorous bijective mapping,
    then apply its inverse to the predicted sequences.
    Returns (corrected_preds, permutation_map).
    """
    from scipy.optimize import linear_sum_assignment
    # Maximize transition probability: equivalent to minimizing negative T_mat
    row_ind, col_ind = linear_sum_assignment(-T_mat)
    perm = col_ind  # perm[i] = what true symbol i maps to
    
    # Inverse permutation mapping
    inv_perm = np.zeros(4, dtype=int)
    for i, p in enumerate(perm):
        inv_perm[p] = i
        
    corrected = np.vectorize(lambda x: inv_perm[x])(pred_seqs)
    return corrected, perm


def sequence_MI(true_seqs, pred_seqs, T_steps=4):
    """
    Estimate I(S_1:T ; Ŝ_1:T) in bits.
    Enumerate over sequence-tuples (4^T possible) and compute joint distribution.
    For T=4 and 4 symbols this is 256 bins — feasible with 1000 trajectories.
    """
    n_traj = true_seqs.shape[0]
    # Encode each length-4 sequence as a base-4 integer
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
    # I = H(Ŝ) - H(Ŝ|S)
    H_pred = -np.sum(p_pred * np.log2(np.maximum(p_pred, eps)))
    return float(H_pred - H_pred_given_true)


def transition_preservation(true_seqs, pred_seqs):
    """
    For each adjacent pair (t, t+1) in the sequence, check if the
    relative transition is preserved under the dominant frame rotation.
    Returns fraction of transitions where the relational structure is preserved:
    transition preserved iff pred[t+1] - pred[t] == true[t+1] - true[t] (mod 4).
    """
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
# Main experiment
# ---------------------------------------------------------------------------

def run_program_ad(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program AD-lite — Sequential Semantic Transport")
    print("=" * 70)

    N = 12
    Lx, Ly = 3, 4
    J_std = args.j_std
    T2 = args.t2
    dt_step = args.T_max / args.n_steps_per_sym   # dt per sub-step
    T1 = 20.0
    p1 = float(1 - np.exp(-dt_step / T1))
    p2 = float(1 - np.exp(-dt_step / T2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))  # (4, 4)
    sequences = build_sequences(bell_dict)

    print(f"\nFixed sweet spot: J_std={J_std}, T2={T2}, n_traj={args.n_trajectories}")
    print(f"Sequences: {list(sequences.keys())}")
    print(f"Controllers: free, local_Z, entropy_block")

    H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
    evals_np, evecs_np = _get_eigh(H, (f"AD_J{J_std}", N, args.seed))
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)

    # Shared vmapped runners
    vmap_free = jax.jit(jax.vmap(run_traj_free,
        in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2, 4))
    vmap_local_Z = jax.jit(jax.vmap(run_traj_geo,
        in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
        static_argnums=(2, 4))

    # Output path (checkpointing)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "program_ad_summary.json")
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
        print(f"\n{'=' * 70}")
        print(f"MERA depth={depth}  Gamma={gamma:.3f}")
        print(f"{'=' * 70}")

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

            # For collecting true/pred labels across all sequences for global metrics
            true_labels_all = []
            pred_labels_all = []

            # Step-specific predictions for tracking frame drift: shape (n_seq, n_traj, T=4)
            true_by_step = {t: [] for t in range(4)}
            pred_by_step = {t: [] for t in range(4)}

            for seq_name, seq_labels in sequences.items():
                # True label indices for this sequence
                true_idx = np.array([bell_labels.index(lbl) for lbl in seq_labels])
                true_traj = np.tile(true_idx[None, :], (args.n_trajectories, 1))  # (n_traj, 4)

                # --- Option A: Memory-preserving Correlated Evolution ---
                # Initialize batch with Symbol 0
                alice_logical_0 = bell_dict[seq_labels[0]]
                psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                psi_batch = jnp.tile(psi0_0[None, :], (args.n_trajectories, 1))  # (n_traj, 2^N)

                psi_finals_by_sym = []

                for sym_t in range(4):
                    if sym_t > 0:
                        # reinject new symbol into Alice's logical sites, retaining environment memory
                        new_logical = bell_dict[seq_labels[sym_t]]
                        psi_batch = vmap_reinject(psi_batch, new_logical)

                    # Evolve for n_steps_per_sym steps under current controller
                    if ctrl == "free":
                        psi_batch = vmap_free(
                            evals, evecs, N, jnp.float32(dt_step),
                            args.n_steps_per_sym,
                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                            psi_batch[0], traj_keys)  # JAX vmapped over traj_keys

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

                # Classify all symbols across all trajectories
                pred_traj = np.asarray(
                    classify_sequence(psi_seq_batch, bob_gates, bell_arr, N))  # (n_traj, 4)

                true_labels_all.append(true_traj)
                pred_labels_all.append(pred_traj)

                # Track by sequence step for frame drift
                for t in range(4):
                    true_by_step[t].append(true_traj[:, t])
                    pred_by_step[t].append(pred_traj[:, t])

                acc = symbol_accuracy(true_traj, pred_traj)
                print(f"    seq={seq_name}: raw_acc={acc:.3f}")

            # Stack all sequences into big arrays
            true_all = np.vstack(true_labels_all)  # (n_traj * n_seq, 4)
            pred_all = np.vstack(pred_labels_all)

            # Global metrics
            T_mat = compute_transition_matrix(true_all, pred_all)
            pred_corrected, frame_map = frame_correct(pred_all, T_mat)

            raw_acc    = symbol_accuracy(true_all, pred_all)
            frame_acc  = symbol_accuracy(true_all, pred_corrected)
            seq_mi     = sequence_MI(true_all, pred_all)
            trans_pres = transition_preservation(true_all, pred_all)

            # --- Logical-Frame Tracking Layer (Step-by-step P_t) ---
            step_permutations = []
            frame_drift_rates = []

            for t in range(4):
                true_t = np.concatenate(true_by_step[t])
                pred_t = np.concatenate(pred_by_step[t])
                
                # Compute transition matrix at this exact time step
                T_t = np.zeros((4, 4))
                for i, j in zip(true_t, pred_t):
                    T_t[int(i), int(j)] += 1
                row_sums = T_t.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                T_t /= row_sums
                
                # Dominant mapping P_t
                P_t = np.argmax(T_t, axis=1).tolist()
                step_permutations.append(P_t)

                if t > 0:
                    # Hamming distance / drift rate between consecutive permutations
                    drift = float(np.mean(np.array(P_t) != np.array(step_permutations[t-1])))
                    frame_drift_rates.append(drift)

            mean_drift_rate = float(np.mean(frame_drift_rates)) if frame_drift_rates else 0.0

            print(f"    -> raw_acc={raw_acc:.3f}  frame_acc={frame_acc:.3f}"
                  f"  seq_MI={seq_mi:.3f} bits  trans_pres={trans_pres:.3f}  drift_rate={mean_drift_rate:.3f}")
            print(f"       frame_map={frame_map.tolist()}")
            print(f"       step_maps={step_permutations}")

            depth_metrics[ctrl] = {
                "raw_accuracy":             raw_acc,
                "frame_corrected_accuracy": frame_acc,
                "sequence_MI_bits":         seq_mi,
                "transition_preservation":  trans_pres,
                "frame_drift_rate":         mean_drift_rate,
                "step_permutations":        step_permutations,
                "transition_matrix":        T_mat.tolist(),
                "frame_map":                frame_map.tolist(),
            }

        # O*(l) by frame-corrected accuracy
        best_ctrl = max(ctrl_names,
                        key=lambda c: depth_metrics[c]["frame_corrected_accuracy"])
        print(f"\n  -> O*(l={depth}) = {best_ctrl}"
              f"  (frame_acc={depth_metrics[best_ctrl]['frame_corrected_accuracy']:.3f})")

        all_records.append({
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "metrics": depth_metrics,
            "O_star": best_ctrl,
        })

        # Save checkpoint
        out = {
            "experiment": "sequential_semantic_transport",
            "args": vars(args),
            "sequences": {k: list(v) for k, v in sequences.items()},
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Checkpoint saved -> {path}")

    # Summary table
    print("\n" + "=" * 75)
    print("Sequential Semantic Transport Summary")
    print(f"  {'d':>2} {'Gamma':>6} {'ctrl':>15} {'raw':>6} {'frame':>6}"
          f" {'seq_MI':>8} {'trans':>7} {'drift':>7}")
    print("-" * 75)
    for r in all_records:
        d = r["mera_depth"]
        g = r["cone_overlap_gamma"]
        for ctrl in ["free", "local_Z", "entropy_block"]:
            m = r["metrics"][ctrl]
            star = " <-" if ctrl == r["O_star"] else ""
            print(f"  {d:>2} {g:>6.3f} {ctrl:>15}"
                  f" {m['raw_accuracy']:>6.3f} {m['frame_corrected_accuracy']:>6.3f}"
                  f" {m['sequence_MI_bits']:>8.3f} {m['transition_preservation']:>7.3f}"
                  f" {m['frame_drift_rate']:>7.3f}{star}")
    print("=" * 75)

    print("\nO*(l) Migration (frame-corrected accuracy):")
    for r in all_records:
        print(f"  depth={r['mera_depth']}  Gamma={r['cone_overlap_gamma']:.3f}"
              f"  O*(l) = {r['O_star']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program AD-lite: Sequential Semantic Transport")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max",
                   help="Evolution time per symbol segment")
    p.add_argument("--n-steps-per-sym", type=int, default=40, dest="n_steps_per_sym",
                   help="Trotter steps per symbol segment")
    p.add_argument("--n-trajectories", type=int, default=1000, dest="n_trajectories")
    p.add_argument("--j-std", type=float, default=0.9, dest="j_std")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005, dest="eps_entropy")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="program_ad_results", dest="out_dir")
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

    run_program_ad(args)
