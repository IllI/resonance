"""
Program AH — Semantic Normal Modes
=======================================
Extending latent semantic geometry with Dynamic Mode Decomposition (DMD), 
recurrent sequence decoders, Koopman trajectory rollouts, and a suite 
of ablation controls to confirm the reality of entropy-stabilized transport.
"""
import argparse, json, os, sys, time

try:
    import jax, jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import select_backend
from program_t_tpu import build_grid_hamiltonian
from program_w_tpu import run_probe_stage_w
from program_z_tpu import build_mera_gates_z, get_mera_layer_pairs, _make_initial_y
from program_aa_tpu import compute_cone_overlap, make_entropy_runner
from program_ab_tpu import build_bell_alphabet
from program_af_tpu import extract_logical_density_matrices
from program_ag_tpu import (
    get_gell_mann_matrices, density_to_bloch, build_compositional_motifs,
    get_cached_eigh, vmap_free_global, vmap_local_Z_global, vmap_reinject_global, vmap_block_entropy_global
)

# ---------------------------------------------------------------------------
# Continuous Geometry, DMD & Koopman Core
# ---------------------------------------------------------------------------

def compute_dmd_and_gap(z_trajs, dt):
    """
    Apply Dynamic Mode Decomposition to a batch of continuous-time Bloch trajectories.
    z_trajs: (n_traj, L, 15)
    Returns: K operator, eigenvalues, growth rates, frequencies, spectral gap
    """
    n_traj, L, d = z_trajs.shape
    X = z_trajs[:, :-1, :].reshape(-1, d).T  # (15, n_traj * (L-1))
    Y = z_trajs[:, 1:, :].reshape(-1, d).T   # (15, n_traj * (L-1))
    
    K = Y @ jnp.linalg.pinv(X)
    
    import numpy as np
    # Use numpy on the CPU for the one-off eigendecomposition as TPU lacks the MLIR translation rule
    evals, evecs = np.linalg.eig(np.array(K, dtype=np.complex64))
    
    # Sort eigenvalues by magnitude
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    
    Omega = np.log(evals + 1e-12j) / dt
    growth_rates = np.real(Omega)
    frequencies = np.imag(Omega) / (2 * np.pi)
    
    gap = np.abs(evals[0]) - np.abs(evals[1]) if len(evals) > 1 else 0.0
    return K, jnp.array(evals), jnp.array(growth_rates), jnp.array(frequencies), float(gap)

def compute_flow_compressibility(z_trajs):
    """
    Measure effective dimension of the transport flow via Shannon Participation Ratio.
    """
    n_traj, L, d = z_trajs.shape
    Z_mat = z_trajs.reshape(-1, d)
    U, S, Vh = jnp.linalg.svd(Z_mat, full_matrices=False)
    
    S_norm = S / (jnp.sum(S) + 1e-12)
    entropy = -jnp.sum(S_norm * jnp.log(S_norm + 1e-12))
    d_eff = jnp.exp(entropy)
    return d_eff

def rollout_koopman(z_init, K, L):
    """
    Predict future latent coordinates using only the learned linear Koopman operator.
    z_init: (B, 15)
    K: (15, 15)
    """
    def scan_fn(z, _):
        z_next = z @ K.T
        return z_next, z_next
    
    _, rollout = jax.lax.scan(scan_fn, z_init, None, length=L-1)
    rollout = jnp.transpose(rollout, (1, 0, 2))
    return jnp.concatenate([z_init[:, None, :], rollout], axis=1)

# ---------------------------------------------------------------------------
# Lightweight Plain-JAX Recurrent Latent Decoder
# ---------------------------------------------------------------------------

def init_rnn_params(d_in, d_hidden, d_out, key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W_hh": jax.random.orthogonal(k1, d_hidden),
        "W_xh": jax.random.normal(k2, (d_in, d_hidden)) / jnp.sqrt(d_in),
        "b_h": jnp.zeros(d_hidden),
        "W_hy": jax.random.normal(k3, (d_hidden, d_out)) / jnp.sqrt(d_hidden),
        "b_y": jnp.zeros(d_out)
    }

def rnn_forward(params, x_seq):
    def scan_fn(h, x):
        h_next = jnp.tanh(h @ params["W_hh"] + x @ params["W_xh"] + params["b_h"])
        y = h_next @ params["W_hy"] + params["b_y"]
        return h_next, y
    h0 = jnp.zeros(params["W_hh"].shape[0])
    _, y_seq = jax.lax.scan(scan_fn, h0, x_seq)
    return y_seq

def rnn_loss(params, x_batch, y_batch):
    y_pred_logits = jax.vmap(rnn_forward, in_axes=(None, 0))(params, x_batch)
    log_probs = jax.nn.log_softmax(y_pred_logits, axis=-1)
    one_hot_targets = jax.nn.one_hot(y_batch, 4)
    return -jnp.sum(one_hot_targets * log_probs) / y_batch.size

@jax.jit
def rnn_train_step(params, opt_state, x_batch, y_batch):
    loss, grads = jax.value_and_grad(rnn_loss)(params, x_batch, y_batch)
    
    beta1, beta2, lr, eps = 0.9, 0.999, 1e-2, 1e-8
    t = opt_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2), opt_state["v"], grads)
    
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** t), v)
    
    new_params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, m_hat, v_hat)
    
    return new_params, {"t": t, "m": m, "v": v}, loss

@jax.jit
def _train_loop(params, opt_state, x_train_j, y_train_j, epochs):
    def body_fn(i, carry):
        p, os = carry
        new_p, new_os, _ = rnn_train_step(p, os, x_train_j, y_train_j)
        return new_p, new_os
    return jax.lax.fori_loop(0, epochs, body_fn, (params, opt_state))

def train_rnn_decoder(x_train, y_train, epochs=300):
    key = jax.random.PRNGKey(42)
    d_in = x_train.shape[-1]
    d_hidden = 16
    d_out = 4
    
    params = init_rnn_params(d_in, d_hidden, d_out, key)
    opt_state = {
        "t": 0,
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params)
    }
    
    x_train_j = jnp.array(x_train)
    y_train_j = jnp.array(y_train)
    
    params, opt_state = _train_loop(params, opt_state, x_train_j, y_train_j, epochs)
        
    return params

def predict_rnn(params, x_test):
    y_pred_logits = jax.vmap(rnn_forward, in_axes=(None, 0))(params, jnp.array(x_test))
    return jnp.argmax(y_pred_logits, axis=-1)

# ---------------------------------------------------------------------------
# Master Execution & Ablation Loop
# ---------------------------------------------------------------------------

def run_program_ah(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program AH — Semantic Normal Modes")
    print("=" * 80)

    N = 12
    Lx, Ly = 3, 4
    import math
    dt_step = args.T_max / args.n_steps_per_sym
    T1 = 20.0
    p1 = float(1 - math.exp(-dt_step / T1))
    p2 = float(1 - math.exp(-dt_step / args.t2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))
    lambdas_j = jnp.array(get_gell_mann_matrices())

    # Generate fixed random embedding projection for Ablation 4
    key_emb = jax.random.PRNGKey(999)
    random_embedding_proj = jax.random.normal(key_emb, (16, 15))

    n_ensembles = args.n_ensembles
    n_traj_per_ens = args.n_traj_per_ensemble
    seeds = [11, 29, 47, 83, 101] if not args.smoke_test else [42]
    seq_lengths = args.sequence_lengths

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_ah_summary.json")
    all_records = []

    for depth in args.depth_sweep:
        gamma = float(compute_cone_overlap(depth, N))
        print(f"\n{'=' * 80}\nMERA depth={depth}  Gamma={gamma:.3f}\n{'=' * 80}")

        layer_pairs = get_mera_layer_pairs(depth, N)
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]
        vmap_entropy = make_entropy_runner(ent_block_pairs, N, args.n_steps_per_sym)

        # Pre-compile randomized entropy cooling function (Ablation 2)
        rand_block_pairs = [(2, 3)]  # Deliberately off-cone
        vmap_rand_entropy = make_entropy_runner(rand_block_pairs, N, args.n_steps_per_sym)

        depth_results = {
            ctrl: {
                sl: {
                    "koopman_acc": [],
                    "spectral_gap": [],
                    "flow_dim": [],
                    "ablation_shuffled": [],
                    "ablation_rand_entropy": [],
                    "ablation_decorr": [],
                    "ablation_rand_emb": [],
                    "ablation_phase_noise": []
                } for sl in seq_lengths
            } for ctrl in ["free", "entropy_block"]
        }

        for seed in seeds:
            print(f"\n--- Running Seed: {seed} ---")
            
            H_train = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=args.j_std_train, seed=seed)
            H_test = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=args.j_std_test, seed=seed + 1000)

            evals_train, evecs_train = get_cached_eigh(H_train, (seed, args.j_std_train))
            evals_test, evecs_test = get_cached_eigh(H_test, (seed + 1000, args.j_std_test))
            
            # Ablation 5: Perturbed Hamiltonian phases
            key_phase = jax.random.PRNGKey(seed + 123)
            phase_noise = jax.random.normal(key_phase, evals_test.shape) * 0.1
            evals_test_noisy = evals_test + phase_noise

            alice_gates = tuple(build_mera_gates_z(depth=depth, seed=seed)[0])
            bob_gates = tuple(build_mera_gates_z(depth=depth, seed=seed)[1])
            S_proj_train = run_probe_stage_w(evals_train, evecs_train, N, jnp.float32(dt_step), bob_site=7)
            S_proj_test = run_probe_stage_w(evals_test, evecs_test, N, jnp.float32(dt_step), bob_site=7)

            for sl in seq_lengths:
                print(f"\n  Sequence Length: {sl}")
                sequences = build_compositional_motifs(bell_dict, sl)

                for ctrl in ["free", "entropy_block"]:
                    ens_seeds = jnp.array([seed + 100 * i for i in range(args.n_ensembles)])
                    traj_keys_batch = jax.vmap(lambda s: jax.random.split(jax.random.PRNGKey(s), args.n_traj_per_ensemble))(ens_seeds)

                    def run_ensembles_parallel(evals_h, evecs_h, use_rand_entropy=False):
                        true_labels_all = []
                        rhos_all = []
                        
                        for seq_name, seq_labels in sequences.items():
                            true_idx = jnp.array([bell_labels.index(lbl) for lbl in seq_labels])
                            true_traj = jnp.tile(true_idx[None, None, :], (args.n_ensembles, args.n_traj_per_ensemble, 1))

                            alice_logical_0 = bell_dict[seq_labels[0]]
                            psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                            psi_batch = jnp.tile(psi0_0[None, None, :], (args.n_ensembles, args.n_traj_per_ensemble, 1))

                            alice_matrices = jnp.stack([g[0] for g in alice_gates])
                            alice_indices = tuple((int(g[1]), int(g[2])) for g in alice_gates)

                            pmap_reinject = jax.pmap(vmap_reinject_global, in_axes=(0, None, None, None, None), static_broadcasted_argnums=(3, 4))
                            pmap_free = jax.pmap(vmap_free_global, in_axes=(None, None, None, None, None, None, None, None, 0, 0), static_broadcasted_argnums=(2, 4))
                            pmap_entropy = jax.pmap(vmap_entropy, in_axes=(None, None, None, None, None, None, None, None, 0, 0))
                            pmap_rand_entropy = jax.pmap(vmap_rand_entropy, in_axes=(None, None, None, None, None, None, None, None, 0, 0))

                            psi_finals = []
                            for sym_t in range(sl):
                                if sym_t > 0:
                                    new_logical = bell_dict[seq_labels[sym_t]]
                                    psi_batch = pmap_reinject(psi_batch, new_logical, alice_matrices, alice_indices, N)

                                if ctrl == "free":
                                    psi_batch = pmap_free(
                                        evals_h, evecs_h, N, jnp.float32(dt_step),
                                        args.n_steps_per_sym,
                                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                        psi_batch, traj_keys_batch)
                                elif ctrl == "entropy_block":
                                    if use_rand_entropy:
                                        psi_batch, *_ = pmap_rand_entropy(
                                            evals_h, evecs_h, jnp.float32(dt_step),
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                            jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy),
                                            psi_batch, traj_keys_batch)
                                    else:
                                        psi_batch, *_ = pmap_entropy(
                                            evals_h, evecs_h, jnp.float32(dt_step),
                                            jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                            jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy),
                                            psi_batch, traj_keys_batch)
                                psi_finals.append(psi_batch)

                            psi_seq_batch = jnp.stack(psi_finals, axis=2)
                            
                            vmap_extract = jax.vmap(extract_logical_density_matrices, in_axes=(0, None, None, None))
                            rhos_seq_batch = vmap_extract(psi_seq_batch, bob_gates, bell_arr, N)

                            true_labels_all.append(true_traj)
                            rhos_all.append(rhos_seq_batch)

                        return jnp.concatenate(true_labels_all, axis=1), jnp.concatenate(rhos_all, axis=1)

                    print(f"  [JAX/TPU] Running {args.n_ensembles} ensembles in parallel via pmap...")
                    test_true_batch, test_rhos_batch = run_ensembles_parallel(evals_test_noisy, evecs_test)
                    
                    test_true_batch_rand_ent, test_rhos_batch_rand_ent = run_ensembles_parallel(evals_test, evecs_test, use_rand_entropy=True)
                    test_true_batch_noisy, test_rhos_batch_noisy = run_ensembles_parallel(evals_test_noisy, evecs_test)
                    train_true_batch, train_rhos_batch = run_ensembles_parallel(evals_train, evecs_train)

                    for ens_idx in range(args.n_ensembles):
                        test_true = test_true_batch[ens_idx]
                        test_rhos = test_rhos_batch[ens_idx]
                        train_true = train_true_batch[ens_idx]
                        train_rhos = train_rhos_batch[ens_idx]
                        test_rhos_rand_ent = test_rhos_batch_rand_ent[ens_idx]
                        test_true_noisy = test_true_batch_noisy[ens_idx]
                        test_rhos_noisy = test_rhos_batch_noisy[ens_idx]
                        
                        # Extract continuous Bloch vectors
                        z_train = density_to_bloch(jnp.array(train_rhos), lambdas_j)
                        z_test = density_to_bloch(jnp.array(test_rhos), lambdas_j)

                        # 1. DMD & Compressibility
                        K, evals_k, growth, freqs, gap = compute_dmd_and_gap(z_test, dt_step * args.n_steps_per_sym)
                        d_eff = compute_flow_compressibility(z_test)
                        
                        depth_results[ctrl][sl]["spectral_gap"].append(float(gap))
                        depth_results[ctrl][sl]["flow_dim"].append(float(d_eff))

                        # 2. Train RNN Decoder & Base Koopman Accuracy
                        rnn_params = train_rnn_decoder(z_train, train_true)
                        
                        # Rollout Koopman & Decode
                        z_test_hat = rollout_koopman(jnp.array(z_test[:, 0, :]), K, sl)
                        pred_koopman = predict_rnn(rnn_params, z_test_hat)
                        base_acc = float(jnp.mean(test_true == pred_koopman))
                        depth_results[ctrl][sl]["koopman_acc"].append(base_acc)

                        # --- Ablation 1: Shuffled Latent Order ---
                        key_shuf = jax.random.PRNGKey(seed + 456)
                        def permute_seq(z, key):
                            return z[jax.random.permutation(key, sl), :]
                        shuf_keys = jax.random.split(key_shuf, z_test_hat.shape[0])
                        z_shuffled = jax.vmap(permute_seq)(z_test_hat, shuf_keys)
                        depth_results[ctrl][sl]["ablation_shuffled"].append(float(jnp.mean(test_true == predict_rnn(rnn_params, z_shuffled))))

                        # --- Ablation 2: Randomized Entropy Controller ---
                        if ctrl == "entropy_block":
                            rhos_rand_ent = test_rhos_rand_ent
                            z_rand_ent = density_to_bloch(jnp.array(rhos_rand_ent), lambdas_j)
                            depth_results[ctrl][sl]["ablation_rand_entropy"].append(float(jnp.mean(test_true == predict_rnn(rnn_params, z_rand_ent))))
                        else:
                            depth_results[ctrl][sl]["ablation_rand_entropy"].append(base_acc)

                        # --- Ablation 3: Destroy Temporal Correlations ---
                        key_decorr = jax.random.PRNGKey(seed + 789)
                        train_true_decorr = jax.random.permutation(key_decorr, train_true.flatten()).reshape(train_true.shape)
                        rnn_params_decorr = train_rnn_decoder(z_train, train_true_decorr)
                        depth_results[ctrl][sl]["ablation_decorr"].append(float(jnp.mean(test_true == predict_rnn(rnn_params_decorr, z_test_hat))))

                        # --- Ablation 4: Random Embeddings ---
                        def embed_random(rhos):
                            flat_rhos = rhos.reshape(-1, 16)
                            return jnp.real((flat_rhos @ random_embedding_proj).reshape(rhos.shape[0], rhos.shape[1], 15))
                        z_train_emb = embed_random(train_rhos)
                        z_test_emb = embed_random(test_rhos)
                        K_emb, _, _, _, _ = compute_dmd_and_gap(z_test_emb, dt_step * args.n_steps_per_sym)
                        rnn_params_emb = train_rnn_decoder(z_train_emb, train_true)
                        z_test_hat_emb = rollout_koopman(jnp.array(z_test_emb[:, 0, :]), K_emb, sl)
                        depth_results[ctrl][sl]["ablation_rand_emb"].append(float(jnp.mean(test_true == predict_rnn(rnn_params_emb, z_test_hat_emb))))

                        # --- Ablation 5: Random Hamiltonian Phases ---
                        test_rhos_noisy = test_rhos_noisy
                        z_noisy = density_to_bloch(jnp.array(test_rhos_noisy), lambdas_j)
                        K_noisy, _, _, _, _ = compute_dmd_and_gap(z_noisy, dt_step * args.n_steps_per_sym)
                        z_test_hat_noisy = rollout_koopman(jnp.array(z_noisy[:, 0, :]), K_noisy, sl)
                        depth_results[ctrl][sl]["ablation_phase_noise"].append(float(jnp.mean(test_true_noisy == predict_rnn(rnn_params, z_test_hat_noisy))))

                    print(f"    {ctrl:>15} -> Gap:{float(jnp.mean(jnp.array(depth_results[ctrl][sl]['spectral_gap']))):.4f} "
                          f"  d_eff:{float(jnp.mean(jnp.array(depth_results[ctrl][sl]['flow_dim']))):.2f} "
                          f"  Acc:{float(jnp.mean(jnp.array(depth_results[ctrl][sl]['koopman_acc']))):.3f} "
                          f"  Abl-Shuffle:{float(jnp.mean(jnp.array(depth_results[ctrl][sl]['ablation_shuffled']))):.3f}")

        # Compile final depth record
        record = {"mera_depth": depth, "gamma": gamma, "seq_lengths": {}}
        for sl in seq_lengths:
            record["seq_lengths"][str(sl)] = {}
            for ctrl in ["free", "entropy_block"]:
                record["seq_lengths"][str(sl)][ctrl] = {}
                for metric, vals in depth_results[ctrl][sl].items():
                    vals_j = jnp.array(vals)
                    mean_val = float(jnp.mean(vals_j))
                    std_val = float(jnp.std(vals_j))
                    ci95_val = float(1.96 * std_val / jnp.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    record["seq_lengths"][str(sl)][ctrl][metric] = {
                        "mean": mean_val,
                        "ci95": ci95_val
                    }
        all_records.append(record)

        out = {
            "experiment": "semantic_normal_modes",
            "seed_suite": seeds,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1)
        }
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)

    # Summary master printout
    print("\n" + "=" * 125)
    print(f"  {'d':>2} {'L':>3} {'ctrl':>15} {'Gap':>9} {'d_eff':>9} {'Acc':>8} {'Abl-Shuf':>9} {'Abl-RandEnt':>12} {'Abl-Decorr':>11} {'Abl-Emb':>8} {'Abl-Phase':>10}")
    print("-" * 125)
    for r in all_records:
        d = r["mera_depth"]
        for sl in seq_lengths:
            for ctrl in ["free", "entropy_block"]:
                m = r["seq_lengths"][str(sl)][ctrl]
                print(f"  {d:>2} {sl:>3} {ctrl:>15} {m['spectral_gap']['mean']:>9.4f} {m['flow_dim']['mean']:>9.2f} {m['koopman_acc']['mean']:>8.3f} "
                      f"{m['ablation_shuffled']['mean']:>9.3f} {m['ablation_rand_entropy']['mean']:>12.3f} {m['ablation_decorr']['mean']:>11.3f} "
                      f"{m['ablation_rand_emb']['mean']:>8.3f} {m['ablation_phase_noise']['mean']:>10.3f}")
    print("=" * 125 + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T-max", type=float, default=4.5)
    p.add_argument("--n-steps-per-sym", type=int, default=40)
    p.add_argument("--n-ensembles", type=int, default=8)
    p.add_argument("--n-traj-per-ensemble", type=int, default=60)
    p.add_argument("--j-std-train", type=float, default=0.9)
    p.add_argument("--j-std-test", type=float, default=1.1)
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps-entropy", type=float, default=0.0005)
    p.add_argument("--T-quiet", type=int, default=8)
    p.add_argument("--out-dir", default="program_ah_results")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--sequence-lengths", nargs="+", type=int, default=[16, 32])
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[2, 3])
    args = p.parse_args()

    if args.smoke_test:
        args.n_steps_per_sym = 10
        args.n_ensembles = 8
        args.n_traj_per_ensemble = 60
        args.sequence_lengths = [8]
        args.depth_sweep = [2]
        print("[SMOKE TEST]")

    run_program_ah(args)
