"""
Program AF — Predictive Logical Frame Hydrodynamics
===================================================
Core question: Can we predict the future logical frame evolution strongly enough
to preserve semantic transport proactively rather than retrospectively?

Scientific framing:
  In a MERA-encoded scrambling channel, we represent the logical frame as a
  continuous unitary operator U_t in U(4) obtained via SVD-based Orthogonal Procrustes
  alignment. By training a Vector Autoregressive (VAR) model on a training trajectory
  ensemble, we proactively forecast the next frame U_{t+1} using macroscopic environment
  block entropy and ZZ correlation velocities as features.
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
# Dynamic Sequence Library
# ---------------------------------------------------------------------------

def build_sequences_len(bell_dict, seq_len):
    keys = list(bell_dict.keys())
    A, B, C, D = keys
    
    # Periodic
    periodic_A = ([A, B] * (seq_len // 2 + 1))[:seq_len]
    periodic_B = ([C, D] * (seq_len // 2 + 1))[:seq_len]
    
    # Mirrored
    mirrored_A = []
    for _ in range(seq_len // 4 + 1):
        mirrored_A += [A, B, B, A]
    mirrored_A = mirrored_A[:seq_len]
    
    mirrored_B = []
    for _ in range(seq_len // 4 + 1):
        mirrored_B += [C, D, D, C]
    mirrored_B = mirrored_B[:seq_len]
    
    # Cyclic
    cyclic_A = ([A, B, C, D] * (seq_len // 4 + 1))[:seq_len]
    cyclic_B = ([B, C, D, A] * (seq_len // 4 + 1))[:seq_len]
    
    # Shuffled control
    shuffled = ([A, A, B, B] * (seq_len // 4 + 1))[:seq_len]
    cyclic_shuffled = ([A, C, B, D] * (seq_len // 4 + 1))[:seq_len]
    
    return {
        "periodic_A": tuple(periodic_A),
        "periodic_B": tuple(periodic_B),
        "mirrored_A": tuple(mirrored_A),
        "mirrored_B": tuple(mirrored_B),
        "cyclic_A": tuple(cyclic_A),
        "cyclic_B": tuple(cyclic_B),
        "shuffled_control": tuple(shuffled),
        "cyclic_shuffled": tuple(cyclic_shuffled),
    }


# ---------------------------------------------------------------------------
# Memory-preserving Alice Injection
# ---------------------------------------------------------------------------

def reinject_alice(psi, new_alice_logical, alice_gates, N, alice_sites=(0, 1)):
    rest = [s for s in range(N) if s not in alice_sites]
    perm = list(alice_sites) + rest
    inv_perm = [0] * N
    for k, p in enumerate(perm):
        inv_perm[p] = k

    # Permute so Alice sites are first
    psi_t = jnp.transpose(psi.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))

    env_norm = jnp.sqrt(jnp.sum(jnp.abs(psi_t) ** 2, axis=0, keepdims=True))
    env_norm = jnp.where(env_norm < 1e-12, jnp.ones_like(env_norm), env_norm)

    env_normed = psi_t / env_norm
    env_avg = jnp.mean(env_normed, axis=0, keepdims=True)
    env_avg = env_avg / jnp.linalg.norm(env_avg)

    new_psi_t = new_alice_logical[:, None] * env_norm
    new_psi = jnp.transpose(new_psi_t.reshape((2,) * N), inv_perm).reshape(2 ** N)

    new_psi = apply_gates(new_psi, alice_gates, N)
    return new_psi / jnp.linalg.norm(new_psi)


# ---------------------------------------------------------------------------
# Continuous Procrustes State Extraction & Unitary Alignment
# ---------------------------------------------------------------------------

def extract_logical_density_matrices(psi_seq_batch, bob_gates, bell_arr, N,
                                     logical_sites=(6, 7)):
    """
    Extract 4x4 density matrices in the Bell basis for each sequence step.
    psi_seq_batch: (n_traj, T, 2^N)
    Returns: (n_traj, T, 4, 4) complex density matrices.
    """
    rest = [s for s in range(N) if s not in logical_sites]
    perm = list(logical_sites) + rest

    def step_density(psi_f):
        psi_dec = apply_gates_inverse(psi_f, bob_gates, N)
        psi_t = jnp.transpose(psi_dec.reshape((2,) * N), perm).reshape(4, 2 ** (N - 2))
        
        # Compute projection coefficients in Bell basis: (4, 2^(N-2))
        coeffs = jnp.stack([
            jnp.dot(bell_arr[k].conj(), psi_t) for k in range(4)
        ])
        # Outer product sum over environmental degrees of freedom: (4, 4)
        rho = jnp.dot(coeffs, coeffs.conj().T)
        # Normalize trace
        tr = jnp.trace(rho)
        rho_normed = jnp.where(jnp.abs(tr) < 1e-12, rho, rho / tr)
        return rho_normed

    def traj_densities(psi_seq):
        return jax.vmap(step_density)(psi_seq)

    return jax.jit(jax.vmap(traj_densities))(psi_seq_batch)


def compute_procrustes_unitary(true_labels, rhos_batch):
    """
    Find optimal U_t mapping average evolved Bob density states to the injected Bell bases.
    true_labels: (n_traj, T)
    rhos_batch: (n_traj, T, 4, 4)
    Returns: (T, 4, 4) complex unitary matrices.
    """
    n_traj, T, _, _ = rhos_batch.shape
    optimal_U = []

    for t in range(T):
        # 1. Compute average density matrix for each injected symbol i
        v_i = []
        for i in range(4):
            idx = np.where(true_labels[:, t] == i)[0]
            if len(idx) == 0:
                # Fallback to identity if symbol is unrepresented
                v_i.append(np.eye(4)[:, i])
                continue
            rho_avg = np.mean(rhos_batch[idx, t], axis=0)
            # Dominant eigenvector represents average state
            evals, evecs = np.linalg.eigh(rho_avg)
            dominant_evec = evecs[:, -1]
            v_i.append(dominant_evec)
        
        A_t = np.stack(v_i, axis=1) # (4, 4)
        
        # 2. Orthogonal Procrustes SVD solver to map A_t to Identity
        # Minimize || U_t A_t - I ||_F^2
        U, S, Vh = np.linalg.svd(A_t.conj().T)
        optimal_U_t = Vh.conj().T @ U.conj().T
        optimal_U.append(optimal_U_t)

    return np.stack(optimal_U, axis=0) # (T, 4, 4)


def apply_unitary_decoder(rhos_batch, U_mats):
    """
    Decode each step t using the given unitary frame U_t.
    rhos_batch: (n_traj, T, 4, 4)
    U_mats: (T, 4, 4) or (n_traj, T, 4, 4)
    Returns: (n_traj, T) integer classifications.
    """
    n_traj, T, _, _ = rhos_batch.shape
    decoded = np.zeros((n_traj, T), dtype=int)
    
    for t in range(T):
        U_t = U_mats[t] if len(U_mats.shape) == 3 else U_mats[:, t]
        
        for m in range(n_traj):
            rho = rhos_batch[m, t]
            U_t_m = U_t[m] if len(U_t.shape) == 4 else U_t
            
            # Rotated density matrix
            rho_rot = U_t_m @ rho @ U_t_m.conj().T
            # Maximize diagonal element (overlap with Bell basis)
            decoded[m, t] = int(np.argmax(np.real(np.diag(rho_rot))))
            
    return decoded


# ---------------------------------------------------------------------------
# Predictive Frame VAR Forecaster
# ---------------------------------------------------------------------------

def train_predictive_forecaster(train_true, train_rhos, ent_velocities):
    """
    Train a Vector Autoregressive (VAR) forecaster predicting U_{t+1} from U_t and O_t.
    ent_velocities: (n_traj, T) macroscopic block entropy velocities.
    Returns: (W_U, W_O, C) regression weights.
    """
    T = train_rhos.shape[1]
    
    # 1. Compute retrospectively optimal frame U_t for the training set
    U_train = compute_procrustes_unitary(train_true, train_rhos) # (T, 4, 4)
    
    # Flatten frames to real components: (T, 32)
    U_flat = np.stack([
        np.concatenate([u.real.flatten(), u.imag.flatten()]) for u in U_train
    ], axis=0)
    
    # Average entropy velocity at step t
    O_avg = np.mean(ent_velocities, axis=0) # (T,)
    
    # Form regression dataset for transitions t -> t+1
    X_U, X_O, Y = [], [], []
    for t in range(T - 1):
        X_U.append(U_flat[t])
        X_O.append([O_avg[t]])
        Y.append(U_flat[t+1])
        
    X_U = np.array(X_U)
    X_O = np.array(X_O)
    Y = np.array(Y)
    
    # Design matrix: [U_flat, O_avg, 1]
    X = np.hstack([X_U, X_O, np.ones((X_U.shape[0], 1))])
    
    # Fit least-squares weights
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None) # (34, 32)
    
    W_U = W[:32, :]   # (32, 32)
    W_O = W[32:33, :] # (1, 32)
    C = W[33, :]      # (32,)
    
    return W_U, W_O, C, U_train


def forecast_comoving_frames(test_rhos, test_ent_vel, W_U, W_O, C, U0):
    """
    Proactively forecast future U_{t+1} out-of-sample using trained weights.
    Projects forecast back onto U(4) via SVD to ensure physical unitariness.
    """
    T = test_rhos.shape[1]
    predicted_U = [U0] # Start with verified initial frame U0
    
    # Average test entropy velocity
    O_avg = np.mean(test_ent_vel, axis=0)
    
    for t in range(T - 1):
        prev_U = predicted_U[-1]
        u_flat = np.concatenate([prev_U.real.flatten(), prev_U.imag.flatten()])
        
        # 1. Forecast next step real components
        u_next_flat = u_flat @ W_U + O_avg[t] * W_O[0] + C
        
        # Reconstruct complex matrix
        u_real = u_next_flat[:16].reshape(4, 4)
        u_imag = u_next_flat[16:].reshape(4, 4)
        U_est = u_real + 1j * u_imag
        
        # 2. Project onto physical unitary space via SVD
        V, S, W_proj = np.linalg.svd(U_est)
        U_physical = V @ W_proj
        predicted_U.append(U_physical)
        
    return np.stack(predicted_U, axis=0)


# ---------------------------------------------------------------------------
# General Metrics
# ---------------------------------------------------------------------------

def sequence_MI(true_seqs, pred_seqs, T_steps=4):
    n_traj = true_seqs.shape[0]
    # General base-4 sequence encoder
    def encode_seq(seq):
        code = np.zeros(n_traj, dtype=int)
        for t in range(min(T_steps, 4)): # Cap joint MI to T=4 to prevent bin sparsity
            code += seq[:, t] * (4 ** (3 - t))
        return code

    true_codes = encode_seq(true_seqs)
    pred_codes = encode_seq(pred_seqs)
    
    n_bins = 4 ** min(T_steps, 4)
    joint = np.zeros((n_bins, n_bins))
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


# ---------------------------------------------------------------------------
# Master Running Logic
# ---------------------------------------------------------------------------

_EIGH_CACHE = {}

def get_cached_eigh(H, cache_key):
    if cache_key in _EIGH_CACHE:
        return _EIGH_CACHE[cache_key]
    print("  [CPU/MKL] Diagonalizing Hamiltonian via multi-threaded scipy.linalg.eigh (64-core parallel)...")
    import scipy.linalg as la
    evals_np, evecs_np = la.eigh(H)
    evals = jnp.array(evals_np, dtype=jnp.float32)
    evecs = jnp.array(evecs_np, dtype=jnp.complex64)
    _EIGH_CACHE[cache_key] = (evals, evecs)
    return evals, evecs

def run_program_af(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program AF — Predictive Logical Frame Hydrodynamics")
    print("=" * 80)

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
    
    # Multi-seed Suite
    seeds = [11, 29, 47, 83, 101] if not args.smoke_test else [42]
    seq_lengths = [4, 8] if not args.smoke_test else [4]

    print(f"\nDisorder Seed Suite: {seeds}")
    print(f"Sequence Lengths to test: {seq_lengths}")
    print(f"Trajectories per run: {args.n_trajectories}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_af_summary.json")
    all_records = []
    completed_depths = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                loaded = json.load(f)
                all_records = loaded.get("records", [])
                completed_depths = [r["mera_depth"] for r in all_records]
                print(f"Loaded existing checkpoint from {summary_path}. Completed depths: {completed_depths}")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), starting fresh.")

    for depth in args.depth_sweep:
        if depth in completed_depths:
            print(f"Skipping MERA depth={depth} as it is already completed in checkpoint.")
            continue
        gamma = compute_cone_overlap(depth, N)
        print(f"\n{'=' * 80}")
        print(f"MERA depth={depth}  Gamma={gamma:.3f}")
        print(f"{'=' * 80}")

        # Shared results across seeds
        depth_results = {ctrl: {sl: [] for sl in seq_lengths} for ctrl in ["free", "local_Z", "entropy_block"]}

        for seed in seeds:
            print(f"\n--- Running Seed: {seed} ---")
            H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=seed)
            evals, evecs = get_cached_eigh(H, (seed, J_std))

            vmap_free = jax.jit(jax.vmap(run_traj_free,
                in_axes=(None,None,None,None,None,None,None,None,None,0)), static_argnums=(2, 4))
            vmap_local_Z = jax.jit(jax.vmap(run_traj_geo,
                in_axes=(None,None,None,None,None,None,None,None,None,None,None,None,0)),
                static_argnums=(2, 4))

            alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=seed)
            layer_pairs = get_mera_layer_pairs(depth, N)
            ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]

            def reinject_fixed(psi, new_alice_logical):
                return reinject_alice(psi, new_alice_logical, alice_gates, N)
            vmap_reinject = jax.jit(jax.vmap(reinject_fixed, in_axes=(0, None)))

            vmap_entropy = make_entropy_runner(ent_block_pairs, N, args.n_steps_per_sym)
            S_proj = run_probe_stage_w(evals, evecs, N, jnp.float32(dt_step), bob_site=7)

            traj_keys = jax.random.split(jax.random.PRNGKey(seed), args.n_trajectories)

            for sl in seq_lengths:
                print(f"\n  Sequence Length: {sl}")
                sequences = build_sequences_len(bell_dict, sl)

                for ctrl in ["free", "local_Z", "entropy_block"]:
                    true_labels_all = []
                    pred_raw_all = []
                    rhos_all = []
                    ent_vel_all = []

                    for seq_name, seq_labels in sequences.items():
                        true_idx = np.array([bell_labels.index(lbl) for lbl in seq_labels])
                        true_traj = np.tile(true_idx[None, :], (args.n_trajectories, 1))

                        # Option A correlated injection
                        alice_logical_0 = bell_dict[seq_labels[0]]
                        psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                        psi_batch = jnp.tile(psi0_0[None, :], (args.n_trajectories, 1))

                        psi_finals_by_sym = []
                        entropies_by_sym = []

                        for sym_t in range(sl):
                            if sym_t > 0:
                                new_logical = bell_dict[seq_labels[sym_t]]
                                psi_batch = vmap_reinject(psi_batch, new_logical)

                            # Macroscopic feature: entropy tracking
                            # Compute single entropy over the active block pairs for each trajectory
                            def single_entropy(psi):
                                i, j = ent_block_pairs[0]
                                psi_t = psi.reshape((2,) * N)
                                psi_t = jnp.moveaxis(psi_t, [i, j], [0, 1])
                                psi_2d = psi_t.reshape(4, 2 ** (N - 2))
                                rho = (psi_2d @ psi_2d.conj().T).real
                                evals = jnp.linalg.eigvalsh(rho)
                                evals = jnp.maximum(evals, 1e-10)
                                return -jnp.sum(evals * jnp.log(evals))
                            
                            s2 = jax.vmap(single_entropy)(psi_batch) # (n_traj,)
                            entropies_by_sym.append(s2)

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

                        psi_seq_batch = jnp.stack(psi_finals_by_sym, axis=1)
                        
                        # 1. Continuous state extraction (Tomography)
                        rhos_seq = np.asarray(
                            extract_logical_density_matrices(psi_seq_batch, bob_gates, bell_arr, N))

                        # Raw classification
                        pred_raw = np.zeros((args.n_trajectories, sl), dtype=int)
                        for t in range(sl):
                            pred_raw[:, t] = np.argmax(np.real(np.diagonal(rhos_seq[:, t], axis1=1, axis2=2)), axis=1)

                        true_labels_all.append(true_traj)
                        pred_raw_all.append(pred_raw)
                        rhos_all.append(rhos_seq)

                        # Stack to construct trajectory-specific entropy vectors of shape (n_traj, sl)
                        ent_traj = np.stack([np.asarray(e) for e in entropies_by_sym], axis=1)
                        ent_vel = np.gradient(ent_traj, axis=1) # (n_traj, sl)
                        ent_vel_all.append(ent_vel)

                    true_all = np.vstack(true_labels_all)
                    pred_all = np.vstack(pred_raw_all)
                    rhos_all = np.vstack(rhos_all)
                    ent_vel_all = np.vstack(ent_vel_all)

                    # Split trajectories: 50% Train, 50% Test for out-of-sample forecasting
                    n_total = true_all.shape[0]
                    split = n_total // 2
                    
                    train_true, test_true = true_all[:split], true_all[split:]
                    train_pred, test_pred = pred_all[:split], pred_all[split:]
                    train_rhos, test_rhos = rhos_all[:split], rhos_all[split:]
                    train_ent_vel, test_ent_vel = ent_vel_all[:split], ent_vel_all[split:]

                    # A. Baseline Raw Accuracy (on Test set)
                    raw_acc = float(np.mean(test_true == test_pred))

                    # B. Retrospective optimal frame corrected (Test set)
                    U_retrospective = compute_procrustes_unitary(test_true, test_rhos)
                    pred_retro = apply_unitary_decoder(test_rhos, U_retrospective)
                    retro_acc = float(np.mean(test_true == pred_retro))

                    # C. Predictive Frame Forecast (Out-of-sample)
                    W_U, W_O, C, U_train_optimal = train_predictive_forecaster(train_true, train_rhos, train_ent_vel)
                    predicted_U = forecast_comoving_frames(test_rhos, test_ent_vel, W_U, W_O, C, U_train_optimal[0])
                    
                    pred_forecast = apply_unitary_decoder(test_rhos, predicted_U)
                    forecast_acc = float(np.mean(test_true == pred_forecast))

                    # MI and Transition Preservation (on Forecast set)
                    seq_mi = sequence_MI(test_true, pred_forecast, T_steps=sl)
                    trans_pres = transition_preservation(test_true, pred_forecast)

                    # Unified Semantic Transport Fidelity (STF)
                    # STF = 0.4*MI + 0.3*Trans + 0.3*Forecast_Acc
                    stf = 0.4 * seq_mi + 0.3 * trans_pres + 0.3 * forecast_acc

                    # Frame Drift Velocity (Average deviation of predicted unitaries)
                    drift_vel = 0.0
                    for t in range(1, sl):
                        drift_vel += np.linalg.norm(predicted_U[t] - predicted_U[t-1], 'fro')
                    drift_vel /= (sl - 1)

                    print(f"    {ctrl:>15} -> Raw:{raw_acc:.3f}  Retro:{retro_acc:.3f}  Forecast:{forecast_acc:.3f}"
                          f"  MI:{seq_mi:.3f}  STF:{stf:.3f}  Drift:{drift_vel:.3f}")

                    depth_results[ctrl][sl].append({
                        "raw_acc":       raw_acc,
                        "retro_acc":     retro_acc,
                        "forecast_acc":  forecast_acc,
                        "seq_mi":        seq_mi,
                        "trans_pres":    trans_pres,
                        "stf":           stf,
                        "drift_vel":     drift_vel
                    })

        # Process ensemble statistics (95% Confidence Intervals)
        record = {
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "sequence_lengths": {}
        }

        for sl in seq_lengths:
            record["sequence_lengths"][sl] = {}
            for ctrl in ["free", "local_Z", "entropy_block"]:
                metrics = ["raw_acc", "retro_acc", "forecast_acc", "seq_mi", "trans_pres", "stf", "drift_vel"]
                record["sequence_lengths"][sl][ctrl] = {}
                
                for m in metrics:
                    vals = [res[m] for res in depth_results[ctrl][sl]]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    # 95% Confidence interval
                    ci = float(1.96 * std / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
                    
                    record["sequence_lengths"][sl][ctrl][m] = {
                        "mean": round(mean, 4),
                        "ci95": round(ci, 4)
                    }

        all_records.append(record)

        # Checkpoint
        out = {
            "experiment": "predictive_logical_frame_hydrodynamics",
            "seed_suite": seeds,
            "sequence_lengths": seq_lengths,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1)
        }
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Checkpoint saved -> {summary_path}")

    # Summary master comparative print
    print("\n" + "=" * 105)
    print("Predictive Logical Frame Hydrodynamics Summary (Program AF)")
    print(f"  {'d':>2} {'len':>3} {'ctrl':>15} {'raw':>12} {'retro':>12} {'forecast':>12}"
          f" {'seq_MI':>12} {'STF':>12} {'drift':>10}")
    print("-" * 105)
    for r in all_records:
        d = r["mera_depth"]
        for sl in seq_lengths:
            for ctrl in ["free", "local_Z", "entropy_block"]:
                m = r["sequence_lengths"][str(sl)][ctrl]
                raw_str = f"{m['raw_acc']['mean']:.3f}±{m['raw_acc']['ci95']:.3f}"
                ret_str = f"{m['retro_acc']['mean']:.3f}±{m['retro_acc']['ci95']:.3f}"
                for_str = f"{m['forecast_acc']['mean']:.3f}±{m['forecast_acc']['ci95']:.3f}"
                mi_str  = f"{m['seq_mi']['mean']:.3f}±{m['seq_mi']['ci95']:.3f}"
                stf_str = f"{m['stf']['mean']:.3f}±{m['stf']['ci95']:.3f}"
                dr_str  = f"{m['drift_vel']['mean']:.3f}"
                print(f"  {d:>2} {sl:>3} {ctrl:>15} {raw_str:>12} {ret_str:>12} {for_str:>12}"
                      f" {mi_str:>12} {stf_str:>12} {dr_str:>10}")
    print("=" * 105)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program AF: Predictive Logical Frame Hydrodynamics")
    p.add_argument("--T-max", type=float, default=4.5, dest="T_max")
    p.add_argument("--n-steps-per-sym", type=int, default=40, dest="n_steps_per_sym")
    p.add_argument("--n-trajectories", type=int, default=1000, dest="n_trajectories")
    p.add_argument("--j-std", type=float, default=0.9, dest="j_std")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005, dest="eps_entropy")
    p.add_argument("--T-quiet", type=int, default=8, dest="T_quiet")
    p.add_argument("--out-dir", default="program_af_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3, 4],
                   dest="depth_sweep")
    args = p.parse_args()

    if args.smoke_test:
        args.n_steps_per_sym = 10
        args.n_trajectories = 40
        args.depth_sweep = [2]
        print("[SMOKE TEST]")

    run_program_af(args)
