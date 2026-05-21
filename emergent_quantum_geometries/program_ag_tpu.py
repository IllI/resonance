"""
Program AG — Latent Semantic Flow Fields
=======================================
Merging symbolic reconstruction (Track A) and latent semantic geometry (Track B)
to characterize the continuous, coordinate-independent geometry of scrambling-induced
frame transport and its stabilization under environmental block entropy cooling.
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
from program_y_tpu import apply_2q_gate
from program_aa_tpu import (
    build_ZZ_family_signs, compute_cone_overlap,
    make_entropy_runner,
)
from program_ab_tpu import build_bell_alphabet
from program_af_tpu import reinject_alice, extract_logical_density_matrices, compute_procrustes_unitary, apply_unitary_decoder


# ---------------------------------------------------------------------------
# SU(4) Gell-Mann Generators & Bloch Projection
# ---------------------------------------------------------------------------

def get_gell_mann_matrices():
    """
    Generates the 15 standard Gell-Mann matrices for SU(4) normalized such that
    Tr(lambda_i lambda_j) = 2 delta_{ij}.
    """
    lambdas = []
    # 6 Symmetric ones (sigma_x like)
    for j in range(4):
        for k in range(j + 1, 4):
            mat = np.zeros((4, 4), dtype=complex)
            mat[j, k] = 1.0
            mat[k, j] = 1.0
            lambdas.append(mat)
    # 6 Antisymmetric ones (sigma_y like)
    for j in range(4):
        for k in range(j + 1, 4):
            mat = np.zeros((4, 4), dtype=complex)
            mat[j, k] = -1j
            mat[k, j] = 1j
            lambdas.append(mat)
    # 3 Diagonal ones (sigma_z like)
    d1 = np.diag([1.0, -1.0, 0.0, 0.0])
    lambdas.append(d1)
    
    d2 = np.diag([1.0, 1.0, -2.0, 0.0]) / np.sqrt(3)
    lambdas.append(d2)
    
    d3 = np.diag([1.0, 1.0, 1.0, -3.0]) / np.sqrt(6)
    lambdas.append(d3)
    
    return np.stack(lambdas, axis=0) # (15, 4, 4)


def density_to_bloch(rho, lambdas):
    """
    Map complex density matrices of shape (..., 4, 4) to real Bloch coordinates of shape (..., 15).
    z^{(k)} = 0.5 * Tr(rho * lambda_k)
    """
    # Trace(A B) = sum(A_ij B_ji) = sum(A * B^T)
    # Gell-Mann matrices are Hermitian, so lambda_k^T = lambda_k^*
    lambdas_T = jnp.transpose(lambdas, (0, 2, 1))
    return 0.5 * jnp.real(jnp.einsum('...ij,kij->...k', rho, lambdas_T))


# ---------------------------------------------------------------------------
# Compositional Motifs with ECC Properties
# ---------------------------------------------------------------------------

def build_compositional_motifs(bell_dict, seq_len):
    keys = list(bell_dict.keys())
    A, B, C, D = keys
    
    # 1. Periodic: ABAB...
    periodic = ([A, B] * (seq_len // 2 + 1))[:seq_len]
    # 2. Mirrored (ECC-like symmetrical parity): ABBA...
    mirrored = []
    for _ in range(seq_len // 4 + 1):
        mirrored += [A, B, B, A]
    mirrored = mirrored[:seq_len]
    # 3. Cyclic: ABCD...
    cyclic = ([A, B, C, D] * (seq_len // 4 + 1))[:seq_len]
    
    return {
        "periodic": tuple(periodic),
        "mirrored": tuple(mirrored),
        "cyclic": tuple(cyclic)
    }


# ---------------------------------------------------------------------------
# Trajectory Geometry Metrics (Track B)
# ---------------------------------------------------------------------------

def compute_trajectory_geometry(z_coords):
    """
    z_coords: (n_traj, T, 15) continuous real Bloch coordinates.
    Returns geometric metrics:
      - Latent Diffusion (D_semantic): mean(||z_t+1 - z_t||^2)
      - Curvature: mean(||z_t+1 - 2*z_t + z_t-1||)
      - Mean separation: average distance between trajectory pairs
    """
    z_coords = jnp.array(z_coords)
    n_traj, T, d = z_coords.shape
    
    # 1. Diffusion
    diffs = z_coords[:, 1:] - z_coords[:, :-1]
    latent_diff = float(jnp.mean(jnp.sum(diffs ** 2, axis=2)))
    
    # 2. Curvature (t from 1 to T-2)
    if T > 2:
        curv = z_coords[:, 2:] - 2 * z_coords[:, 1:-1] + z_coords[:, :-2]
        latent_curv = float(jnp.mean(jnp.sqrt(jnp.sum(curv ** 2, axis=2))))
    else:
        latent_curv = 0.0
        
    # 3. Trajectory Separation (mean pairwise distance across ensemble)
    dists = jnp.sqrt(jnp.sum((z_coords[:, None, :, :] - z_coords[None, :, :, :]) ** 2, axis=-1)) # (n_traj, n_traj, T)
    mask = 1.0 - jnp.eye(n_traj)[:, :, None]
    dists = dists * mask
    step_sep = jnp.sum(dists, axis=(0, 1)) / (n_traj * (n_traj - 1)) # (T,)
    mean_sep = float(jnp.mean(step_sep))
    
    return latent_diff, latent_curv, mean_sep


# ---------------------------------------------------------------------------
# Koopman Operator Learning (Fully JAX / TPU Accelerated)
# ---------------------------------------------------------------------------

def train_koopman_operator(z_train):
    """
    Fit a Koopman linear model in latent space: z_{t+1} = z_t * K^T + C
    z_train: (n_traj, T, 15)
    Returns K (15, 15) and C (15,)
    """
    z_train = jnp.array(z_train)
    n_traj, T, d = z_train.shape
    X = z_train[:, :-1, :].reshape(-1, d)
    Y = z_train[:, 1:, :].reshape(-1, d)
    Design = jnp.hstack([X, jnp.ones((X.shape[0], 1))])
    W, _, _, _ = jnp.linalg.lstsq(Design, Y, rcond=None)
    K = W[:d, :].T
    C = W[d, :]
    return K, C


def evaluate_koopman_predictor(z_test, K, C):
    """
    Measure prediction error under learned Koopman operator.
    Returns mean-squared error (MSE).
    """
    z_test = jnp.array(z_test)
    n_traj, T, d = z_test.shape
    X = z_test[:, :-1, :].reshape(-1, d)
    Y = z_test[:, 1:, :].reshape(-1, d)
    Y_pred = X @ K.T + C
    mse = float(jnp.mean(jnp.sum((Y - Y_pred) ** 2, axis=1)))
    return mse


# ---------------------------------------------------------------------------
# JAX-Accelerated Procrustes & Predictive Frame VAR Forecaster
# ---------------------------------------------------------------------------

def compute_procrustes_unitary_jax(true_labels, rhos_batch):
    true_labels = jnp.array(true_labels)
    rhos_batch = jnp.array(rhos_batch)
    n_traj, T, _, _ = rhos_batch.shape
    
    def solve_t(labels_t, rhos_t):
        def get_avg(i):
            mask = (labels_t == i)[:, None, None]
            sum_rho = jnp.sum(jnp.where(mask, rhos_t, 0.0), axis=0)
            count = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1e-10)
            rho_avg = sum_rho / count
            evals, evecs = jnp.linalg.eigh(rho_avg)
            return evecs[:, -1]
        
        v_i = jax.vmap(get_avg)(jnp.arange(4)) # (4, 4)
        A_t = v_i.T # (4, 4)
        U, S, Vh = jnp.linalg.svd(jnp.conj(A_t).T)
        return jnp.conj(Vh).T @ jnp.conj(U).T

    labels_T = true_labels.T # (T, n_traj)
    rhos_T = jnp.transpose(rhos_batch, (1, 0, 2, 3)) # (T, n_traj, 4, 4)
    return jax.vmap(solve_t)(labels_T, rhos_T)


def apply_unitary_decoder_jax(rhos_batch, U_mats):
    rhos_batch = jnp.array(rhos_batch)
    U_mats = jnp.array(U_mats)
    n_traj, T, _, _ = rhos_batch.shape
    if len(U_mats.shape) == 3:
        U_mats = jnp.tile(U_mats[None, ...], (n_traj, 1, 1, 1))
    rho_rot = jnp.einsum('...ij,...jk,...lk->...il', U_mats, rhos_batch, jnp.conj(U_mats))
    diags = jnp.real(jnp.diagonal(rho_rot, axis1=-2, axis2=-1))
    return np.asarray(jnp.argmax(diags, axis=-1))


def train_predictive_forecaster_jax(train_true, train_rhos, ent_velocities):
    train_true = jnp.array(train_true)
    train_rhos = jnp.array(train_rhos)
    ent_velocities = jnp.array(ent_velocities)
    
    T = train_rhos.shape[1]
    U_train = compute_procrustes_unitary_jax(train_true, train_rhos)
    U_flat = jnp.stack([
        jnp.concatenate([u.real.flatten(), u.imag.flatten()]) for u in U_train
    ], axis=0)
    O_avg = jnp.mean(ent_velocities, axis=0)
    
    X_U = U_flat[:-1]
    X_O = O_avg[:-1, None]
    Y = U_flat[1:]
    X = jnp.hstack([X_U, X_O, jnp.ones((X_U.shape[0], 1))])
    
    W, _, _, _ = jnp.linalg.lstsq(X, Y, rcond=None)
    W_U = W[:32, :]
    W_O = W[32:33, :]
    C = W[33, :]
    return W_U, W_O, C, U_train


def forecast_comoving_frames_jax(test_rhos, test_ent_vel, W_U, W_O, C, U0):
    test_rhos = jnp.array(test_rhos)
    test_ent_vel = jnp.array(test_ent_vel)
    W_U = jnp.array(W_U)
    W_O = jnp.array(W_O)
    C = jnp.array(C)
    U0 = jnp.array(U0)
    
    T = test_rhos.shape[1]
    O_avg = jnp.mean(test_ent_vel, axis=0)
    
    def scan_fn(prev_U, o_val):
        u_flat = jnp.concatenate([prev_U.real.flatten(), prev_U.imag.flatten()])
        u_next_flat = u_flat @ W_U + o_val * W_O[0] + C
        u_real = u_next_flat[:16].reshape(4, 4)
        u_imag = u_next_flat[16:].reshape(4, 4)
        U_est = u_real + 1j * u_imag
        
        V, S, Vh = jnp.linalg.svd(U_est)
        U_physical = V @ Vh
        return U_physical, U_physical
    
    _, predicted_Us = jax.lax.scan(scan_fn, U0, O_avg[:-1])
    return jnp.concatenate([U0[None, ...], predicted_Us], axis=0)


# ---------------------------------------------------------------------------
# Master Running Logic
# ---------------------------------------------------------------------------

_EIGH_CACHE = {}

def get_cached_eigh(H, cache_key):
    if cache_key in _EIGH_CACHE:
        return _EIGH_CACHE[cache_key]
    print("  [JAX/TPU] Diagonalizing Hamiltonian via JAX/XLA on TPU...")
    evals_j, evecs_j = jnp.linalg.eigh(jnp.array(H))
    _EIGH_CACHE[cache_key] = (evals_j, evecs_j)
    return evals_j, evecs_j


# ---------------------------------------------------------------------------
# Global JAX JIT Compilations (Preventing loop recompilation bottlenecks)
# ---------------------------------------------------------------------------

vmap_free_global = jax.jit(
    jax.vmap(run_traj_free, in_axes=(None, None, None, None, None, None, None, None, 0, 0)),
    static_argnums=(2, 4)
)

vmap_local_Z_global = jax.jit(
    jax.vmap(run_traj_geo, in_axes=(None, None, None, None, None, None, None, None, None, None, None, 0, 0)),
    static_argnums=(2, 4)
)

def reinject_alice_split(psi, new_alice_logical, alice_matrices, alice_indices, N):
    alice_sites = (0, 1)
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

    for idx, (i, j) in enumerate(alice_indices):
        U = alice_matrices[idx]
        new_psi = apply_2q_gate(new_psi, U, i, j, N)
    return new_psi / jnp.linalg.norm(new_psi)

vmap_reinject_global = jax.jit(
    jax.vmap(reinject_alice_split, in_axes=(0, None, None, None, None)),
    static_argnums=(3, 4)
)

def compute_block_entropy(psi, i, j):
    # N is implicitly 12 for this sweep
    N = 12
    psi_t = psi.reshape((2,) * N)
    psi_t = jnp.moveaxis(psi_t, [i, j], [0, 1])
    psi_2d = psi_t.reshape(4, 2 ** (N - 2))
    rho = (psi_2d @ psi_2d.conj().T).real
    evs = jnp.linalg.eigvalsh(rho)
    evs = jnp.maximum(evs, 1e-10)
    return -jnp.sum(evs * jnp.log(evs))

vmap_block_entropy_global = jax.jit(
    jax.vmap(compute_block_entropy, in_axes=(0, None, None)),
    static_argnums=(1, 2)
)


def run_program_ag(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program AG — Latent Semantic Flow Fields")
    print("=" * 80)

    N = 12
    Lx, Ly = 3, 4
    dt_step = args.T_max / args.n_steps_per_sym
    T1 = 20.0
    p1 = float(1 - np.exp(-dt_step / T1))
    p2 = float(1 - np.exp(-dt_step / args.t2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))
    lambdas = get_gell_mann_matrices()
    lambdas_j = jnp.array(lambdas)

    # Replicated Ensembles setup
    n_ensembles = args.n_ensembles
    n_traj_per_ens = args.n_traj_per_ensemble
    seeds = [11, 29, 47, 83, 101] if not args.smoke_test else [42]
    seq_lengths = [8] if not args.smoke_test else [4]

    print(f"\nDisorder Seed Suite: {seeds}")
    print(f"Replicated Ensembles: {n_ensembles} runs of {n_traj_per_ens} trajectories each")
    print(f"Sequence Lengths to test: {seq_lengths}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_ag_summary.json")
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

        layer_pairs = get_mera_layer_pairs(depth, N)
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]
        vmap_entropy = make_entropy_runner(ent_block_pairs, N, args.n_steps_per_sym)

        # Shared results dictionary
        depth_results = {
            ctrl: {
                sl: {
                    "raw_acc": [],
                    "retro_acc": [],
                    "forecast_acc": [],
                    "latent_diff": [],
                    "latent_curv": [],
                    "mean_sep": [],
                    "koopman_err": [],
                    # Adversarial controls
                    "adv_shuffled_acc": [],
                    "adv_random_acc": [],
                    "adv_decorrelated_acc": []
                } for sl in seq_lengths
            } for ctrl in ["free", "local_Z", "entropy_block"]
        }

        for seed in seeds:
            print(f"\n--- Running Seed: {seed} ---")
            
            # Out-of-distribution Hamiltonian standard deviations
            J_train_val = args.j_std_train
            J_test_val = args.j_std_test

            H_train = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_train_val, seed=seed)
            H_test = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_test_val, seed=seed + 1000) # OOD seed

            evals_train, evecs_train = get_cached_eigh(H_train, (seed, J_train_val))
            evals_test, evecs_test = get_cached_eigh(H_test, (seed + 1000, J_test_val))

            alice_gates = tuple(build_mera_gates_z(depth=depth, seed=seed)[0])
            bob_gates = tuple(build_mera_gates_z(depth=depth, seed=seed)[1])
            S_proj_train = run_probe_stage_w(evals_train, evecs_train, N, jnp.float32(dt_step), bob_site=7)
            S_proj_test = run_probe_stage_w(evals_test, evecs_test, N, jnp.float32(dt_step), bob_site=7)

            # Execution loop over sequence lengths
            for sl in seq_lengths:
                print(f"\n  Sequence Length: {sl}")
                sequences = build_compositional_motifs(bell_dict, sl)

                for ctrl in ["free", "local_Z", "entropy_block"]:
                    
                    # We run E replicated ensembles
                    ens_raw_acc = []
                    ens_retro_acc = []
                    ens_forecast_acc = []
                    ens_latent_diff = []
                    ens_latent_curv = []
                    ens_mean_sep = []
                    ens_koopman_err = []
                    
                    ens_adv_shuffle = []
                    ens_adv_random = []
                    ens_adv_decorr = []

                    for ens_idx in range(n_ensembles):
                        ens_seed = seed + 100 * ens_idx
                        traj_keys = jax.random.split(jax.random.PRNGKey(ens_seed), n_traj_per_ens)

                        # We collect both training data and OOD test data
                        def run_ensemble_on_hamiltonian(evals_h, evecs_h, S_proj_h):
                            true_labels_all = []
                            pred_raw_all = []
                            rhos_all = []
                            ent_vel_all = []
                            
                            for seq_name, seq_labels in sequences.items():
                                true_idx = np.array([bell_labels.index(lbl) for lbl in seq_labels])
                                true_traj = np.tile(true_idx[None, :], (n_traj_per_ens, 1))

                                alice_logical_0 = bell_dict[seq_labels[0]]
                                psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                                psi_batch = jnp.tile(psi0_0[None, :], (n_traj_per_ens, 1))

                                psi_finals = []
                                entropies = []

                                alice_matrices = jnp.stack([g[0] for g in alice_gates])
                                alice_indices = tuple((int(g[1]), int(g[2])) for g in alice_gates)

                                for sym_t in range(sl):
                                    if sym_t > 0:
                                        new_logical = bell_dict[seq_labels[sym_t]]
                                        psi_batch = vmap_reinject_global(psi_batch, new_logical, alice_matrices, alice_indices, N)

                                    # Block entropy check
                                    i_site, j_site = ent_block_pairs[0]
                                    entropies.append(vmap_block_entropy_global(psi_batch, i_site, j_site))

                                    if ctrl == "free":
                                        psi_batch = vmap_free_global(
                                            evals_h, evecs_h, N, jnp.float32(dt_step),
                                            args.n_steps_per_sym,
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                            psi_batch, traj_keys)

                                    elif ctrl == "local_Z":
                                        psi_batch, *_ = vmap_local_Z_global(
                                            evals_h, evecs_h, N, jnp.float32(dt_step),
                                            args.n_steps_per_sym,
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                            jnp.int32(args.T_quiet), jnp.float32(args.eps),
                                            S_proj_h, psi_batch, traj_keys)

                                    elif ctrl == "entropy_block":
                                        psi_batch, *_ = vmap_entropy(
                                            evals_h, evecs_h, jnp.float32(dt_step),
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                            jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy),
                                            psi_batch, traj_keys)

                                    psi_finals.append(psi_batch)

                                psi_seq_batch = jnp.stack(psi_finals, axis=1)
                                rhos_seq = np.asarray(extract_logical_density_matrices(psi_seq_batch, bob_gates, bell_arr, N))

                                pred_raw = np.zeros((n_traj_per_ens, sl), dtype=int)
                                for t in range(sl):
                                    pred_raw[:, t] = np.argmax(np.real(np.diagonal(rhos_seq[:, t], axis1=1, axis2=2)), axis=1)

                                true_labels_all.append(true_traj)
                                pred_raw_all.append(pred_raw)
                                rhos_all.append(rhos_seq)

                                ent_traj = np.stack([np.asarray(e) for e in entropies], axis=1)
                                ent_vel_all.append(np.gradient(ent_traj, axis=1))

                            return (np.vstack(true_labels_all), np.vstack(pred_raw_all), 
                                    np.vstack(rhos_all), np.vstack(ent_vel_all))

                        # 1. Run Train dataset on standard J_std
                        train_true, train_pred, train_rhos, train_ent_vel = run_ensemble_on_hamiltonian(evals_train, evecs_train, S_proj_train)
                        # 2. Run Test dataset on OOD J_std + unique disorder configurations
                        test_true, test_pred, test_rhos, test_ent_vel = run_ensemble_on_hamiltonian(evals_test, evecs_test, S_proj_test)

                        # Extract Continuous Bloch vectors (Track B)
                        z_train = np.asarray(density_to_bloch(jnp.array(train_rhos), lambdas_j))
                        z_test = np.asarray(density_to_bloch(jnp.array(test_rhos), lambdas_j))

                        # Calculate continuous flow statistics on OOD Test set
                        diff, curv, sep = compute_trajectory_geometry(z_test)
                        ens_latent_diff.append(diff)
                        ens_latent_curv.append(curv)
                        ens_mean_sep.append(sep)

                        # Koopman learning & OOD Generalization
                        K, C = train_koopman_operator(z_train)
                        koopman_err = evaluate_koopman_predictor(z_test, K, C)
                        ens_koopman_err.append(koopman_err)

                        # Track A - Symbolic reconstruction accuracies on Test set
                        raw_acc = float(np.mean(test_true == test_pred))
                        ens_raw_acc.append(raw_acc)

                        # Retrospective alignment (TPU Eigh & SVD)
                        U_retro = compute_procrustes_unitary_jax(test_true, test_rhos)
                        pred_retro = apply_unitary_decoder_jax(test_rhos, U_retro)
                        retro_acc = float(np.mean(test_true == pred_retro))
                        ens_retro_acc.append(retro_acc)

                        # Online forecast frame decoding (TPU VAR & SVD Frame Projection)
                        W_U, W_O, C_var, U_train_optimal = train_predictive_forecaster_jax(train_true, train_rhos, train_ent_vel)
                        predicted_U = forecast_comoving_frames_jax(test_rhos, test_ent_vel, W_U, W_O, C_var, U_train_optimal[0])
                        pred_forecast = apply_unitary_decoder_jax(test_rhos, predicted_U)
                        forecast_acc = float(np.mean(test_true == pred_forecast))
                        ens_forecast_acc.append(forecast_acc)

                        # --- ADVERSARIAL CONTROLS ---
                        # 1. Shuffled time steps
                        shuffled_idx = np.random.permutation(sl)
                        pred_shuffled = pred_forecast[:, shuffled_idx]
                        ens_adv_shuffle.append(float(np.mean(test_true == pred_shuffled)))

                        # 2. Randomized frame alignment (using completely random SU(4) alignments via TPU QR)
                        U_random = []
                        key_qr = jax.random.PRNGKey(ens_seed + 999)
                        for _ in range(sl):
                            key_qr, subkey = jax.random.split(key_qr)
                            g = jax.random.normal(subkey, (4, 4, 2))
                            z_mat = g[..., 0] + 1j * g[..., 1]
                            q, r = jnp.linalg.qr(z_mat)
                            U_random.append(q)
                        pred_random = apply_unitary_decoder_jax(test_rhos, jnp.stack(U_random, axis=0))
                        ens_adv_random.append(float(np.mean(test_true == pred_random)))

                        # 3. Decorrelated trajectory indices (shuffle trajectory order independently per timestep)
                        pred_decorr = np.zeros_like(pred_forecast)
                        for t in range(sl):
                            pred_decorr[:, t] = np.random.permutation(pred_forecast[:, t])
                        ens_adv_decorr.append(float(np.mean(test_true == pred_decorr)))

                    # Update shared sweeps results
                    depth_results[ctrl][sl]["raw_acc"].append(ens_raw_acc)
                    depth_results[ctrl][sl]["retro_acc"].append(ens_retro_acc)
                    depth_results[ctrl][sl]["forecast_acc"].append(ens_forecast_acc)
                    depth_results[ctrl][sl]["latent_diff"].append(ens_latent_diff)
                    depth_results[ctrl][sl]["latent_curv"].append(ens_latent_curv)
                    depth_results[ctrl][sl]["mean_sep"].append(ens_mean_sep)
                    depth_results[ctrl][sl]["koopman_err"].append(ens_koopman_err)
                    
                    depth_results[ctrl][sl]["adv_shuffled_acc"].append(ens_adv_shuffle)
                    depth_results[ctrl][sl]["adv_random_acc"].append(ens_adv_random)
                    depth_results[ctrl][sl]["adv_decorrelated_acc"].append(ens_adv_decorr)

                    print(f"    {ctrl:>15} -> Diff:{np.mean(ens_latent_diff):.4f}  Curv:{np.mean(ens_latent_curv):.4f}"
                          f"  Sep:{np.mean(ens_mean_sep):.4f}  K-Err:{np.mean(ens_koopman_err):.4f}"
                          f"  Forecast-Acc:{np.mean(ens_forecast_acc):.4f}  Shuffle-Ctrl:{np.mean(ens_adv_shuffle):.3f}")

        # Compute final ensemble statistics across seeds & realizations
        record = {
            "mera_depth": depth,
            "cone_overlap_gamma": gamma,
            "sequence_lengths": {}
        }
        
        for sl in seq_lengths:
            record["sequence_lengths"][str(sl)] = {}
            for ctrl in ["free", "local_Z", "entropy_block"]:
                record["sequence_lengths"][str(sl)][ctrl] = {}
                metrics = [
                    "raw_acc", "retro_acc", "forecast_acc", 
                    "latent_diff", "latent_curv", "mean_sep", "koopman_err",
                    "adv_shuffled_acc", "adv_random_acc", "adv_decorrelated_acc"
                ]
                for m in metrics:
                    # Flatten metrics over all seeds & all replicated realizations
                    all_vals = np.concatenate(depth_results[ctrl][sl][m])
                    mean = float(np.mean(all_vals))
                    std = float(np.std(all_vals))
                    # 95% Confidence interval
                    ci = float(1.96 * std / np.sqrt(len(all_vals))) if len(all_vals) > 1 else 0.0
                    record["sequence_lengths"][sl][ctrl][m] = {
                        "mean": round(mean, 5),
                        "ci95": round(ci, 5)
                    }

        all_records.append(record)

        # Checkpoint save
        out = {
            "experiment": "latent_semantic_flow_fields",
            "seed_suite": seeds,
            "sequence_lengths": seq_lengths,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1)
        }
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Checkpoint saved -> {summary_path}")

    # Summary master comparative printout
    print("\n" + "=" * 135)
    print("Latent Semantic Flow Fields Summary Table (Program AG)")
    print(f"  {'d':>2} {'len':>3} {'ctrl':>15} {'diffusion':>14} {'curvature':>14} {'separation':>14} {'K-error (OOD)':>14} {'forecast_acc':>14} {'shuff_ctrl':>12}")
    print("-" * 135)
    for r in all_records:
        d = r["mera_depth"]
        for sl in seq_lengths:
            for ctrl in ["free", "local_Z", "entropy_block"]:
                m = r["sequence_lengths"][str(sl)][ctrl]
                diff_str = f"{m['latent_diff']['mean']:.4f}±{m['latent_diff']['ci95']:.4f}"
                curv_str = f"{m['latent_curv']['mean']:.4f}±{m['latent_curv']['ci95']:.4f}"
                sep_str = f"{m['mean_sep']['mean']:.4f}±{m['mean_sep']['ci95']:.4f}"
                ke_str = f"{m['koopman_err']['mean']:.4f}±{m['koopman_err']['ci95']:.4f}"
                for_str = f"{m['forecast_acc']['mean']:.3f}±{m['forecast_acc']['ci95']:.3f}"
                shf_str = f"{m['adv_shuffled_acc']['mean']:.3f}"
                print(f"  {d:>2} {sl:>3} {ctrl:>15} {diff_str:>14} {curv_str:>14} {sep_str:>14} {ke_str:>14} {for_str:>14} {shf_str:>12}")
    print("=" * 135 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Program AG: Latent Semantic Flow Fields")
    p.add_argument("--T-max", type=float, default=4.5)
    p.add_argument("--n-steps-per-sym", type=int, default=40)
    p.add_argument("--n-ensembles", type=int, default=5, dest="n_ensembles")
    p.add_argument("--n-traj-per-ensemble", type=int, default=200, dest="n_traj_per_ensemble")
    p.add_argument("--j-std-train", type=float, default=0.9, dest="j_std_train")
    p.add_argument("--j-std-test", type=float, default=1.1, dest="j_std_test")
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--eps-entropy", type=float, default=0.0005)
    p.add_argument("--T-quiet", type=int, default=8)
    p.add_argument("--out-dir", default="program_ag_results")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3, 4])
    args = p.parse_args()

    if args.smoke_test:
        args.n_steps_per_sym = 10
        args.n_ensembles = 2
        args.n_traj_per_ensemble = 30
        args.depth_sweep = [2]
        print("[SMOKE TEST]")

    run_program_ag(args)
