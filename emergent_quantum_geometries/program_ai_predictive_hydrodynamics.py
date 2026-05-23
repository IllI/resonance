"""
Program AI — Predictive Semantic Transport via Koopman-Guided D-LinOSS
=======================================================================
Splits D-LinOSS into a Semantic Geometry Encoder (Layer A) and a
Dynamical Flow Observer (Layer B) using a Koopman-RNN residual hybrid.
Replaces reactive triggering with a predictive forecast stabilization
mechanism.
"""
import argparse, json, os, sys, time
import numpy as np

try:
    import jax, jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import select_backend
from program_t_tpu import build_grid_hamiltonian
from program_v_tpu import run_step_psi
from program_z_tpu import build_mera_gates_z, get_mera_layer_pairs, _make_initial_y, apply_gates_inverse
from program_aa_tpu import compute_cone_overlap
from program_ab_tpu import build_bell_alphabet
from program_af_tpu import extract_logical_density_matrices
from program_ag_tpu import get_gell_mann_matrices, density_to_bloch, build_compositional_motifs, get_cached_eigh, vmap_free_global, vmap_reinject_global

# ---------------------------------------------------------------------------
# Layer A: Semantic Geometry Diagnostics
# ---------------------------------------------------------------------------

def compute_lyapunov_rate(z_coords):
    """
    Measure semantic chaos lambda = (1/t) * log(||delta z_t|| / ||delta z_0||)
    z_coords: (n_traj, T, 15)
    """
    diffs = jnp.sqrt(jnp.sum((z_coords[:, 1:] - z_coords[:, :-1])**2, axis=-1) + 1e-12)
    # Average across trajectories
    mean_diffs = jnp.mean(diffs, axis=0) # (T-1,)
    # lambda_t = log(diff_t / diff_0) / t
    t_arr = jnp.arange(1, mean_diffs.shape[0] + 1)
    rates = jnp.log(mean_diffs / (mean_diffs[0] + 1e-12)) / t_arr
    return float(jnp.mean(rates[rates > -jnp.inf]))

def compute_latent_ricci(z_coords):
    """
    Proxy for manifold contraction/expansion (volume change rate).
    """
    diffs = z_coords[:, 1:] - z_coords[:, :-1]
    # contraction = Trace(Covariance(diffs))
    var = jnp.var(diffs, axis=0) # (T-1, 15)
    trace_var = jnp.sum(var, axis=-1) # (T-1,)
    return float(jnp.mean(trace_var))

def compute_transport_coherence(z_coords):
    """
    Autocorrelation decay persistence time.
    """
    z_cent = z_coords - jnp.mean(z_coords, axis=1, keepdims=True)
    n_traj, T, d = z_cent.shape
    # Compute dot product between z_t and z_{t+k}
    cov = jnp.einsum('nti,nti->nt', z_cent[:, :-1], z_cent[:, 1:])
    var = jnp.einsum('nti,nti->nt', z_cent[:, :-1], z_cent[:, :-1])
    autocorr = jnp.mean(cov / (var + 1e-12))
    return float(autocorr)

# ---------------------------------------------------------------------------
# Layer B: Koopman-RNN Hybrid Flow Observer
# ---------------------------------------------------------------------------

def fit_koopman_operator(z_train):
    """
    Fit linear backbone K for z_{t+1} = z_t K^T
    z_train: (B, T, d)
    """
    n_traj, T, d = z_train.shape
    X = z_train[:, :-1, :].reshape(-1, d)
    Y = z_train[:, 1:, :].reshape(-1, d)
    K, _, _, _ = jnp.linalg.lstsq(X, Y, rcond=None)
    return K.T  # (d, d)

def init_rnn_params(d_in, d_hidden, key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W_hh": jax.random.orthogonal(k1, d_hidden),
        "W_xh": jax.random.normal(k2, (d_in, d_hidden)) / jnp.sqrt(d_in),
        "b_h": jnp.zeros(d_hidden),
        "W_hr": jax.random.normal(k3, (d_hidden, d_in)) / jnp.sqrt(d_hidden),
        "b_r": jnp.zeros(d_in)
    }

def rnn_forward(params, h, x):
    h_next = jnp.tanh(h @ params["W_hh"] + x @ params["W_xh"] + params["b_h"])
    r = h_next @ params["W_hr"] + params["b_r"]
    return h_next, r

def rnn_loss(params, z_seq, K):
    # z_seq: (T, d)
    # Predict z_{t+1} = K z_t + RNN(z_t)
    def scan_fn(h, z_t):
        h_next, r_t = rnn_forward(params, h, z_t)
        z_next_pred = z_t @ K.T + r_t
        return h_next, z_next_pred
    
    h0 = jnp.zeros(params["W_hh"].shape[0])
    _, z_preds = jax.lax.scan(scan_fn, h0, z_seq[:-1])
    return jnp.mean((z_preds - z_seq[1:])**2)

@jax.jit
def rnn_train_step(params, opt_state, z_batch, K):
    loss, grads = jax.value_and_grad(lambda p: jnp.mean(jax.vmap(rnn_loss, in_axes=(None, 0, None))(p, z_batch, K)))(params)
    
    beta1, beta2, lr, eps = 0.9, 0.999, 1e-2, 1e-8
    t = opt_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2), opt_state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** t), v)
    new_params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, m_hat, v_hat)
    
    return new_params, {"t": t, "m": m, "v": v}, loss

def train_residual_rnn(z_train, K, epochs=300):
    key = jax.random.PRNGKey(42)
    d_in = z_train.shape[-1]
    d_hidden = 16
    
    params = init_rnn_params(d_in, d_hidden, key)
    opt_state = {
        "t": 0,
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params)
    }
    
    z_batch = jnp.array(z_train)
    def body_fn(i, carry):
        p, os = carry
        new_p, new_os, _ = rnn_train_step(p, os, z_batch, K)
        return new_p, new_os
    
    params, opt_state = jax.lax.fori_loop(0, epochs, body_fn, (params, opt_state))
    return params

# ---------------------------------------------------------------------------
# Predictive JAX Physics Runner
# ---------------------------------------------------------------------------

def extract_single_rho(psi, bob_gates, N):
    psi_dec = apply_gates_inverse(psi, bob_gates, N)
    psi_2d = psi_dec.reshape(4, 2**(N-2))
    return (psi_2d @ psi_2d.conj().T).real

def compute_block_entropy(psi, i, j, N):
    psi_t = psi.reshape((2,) * N)
    psi_t = jnp.moveaxis(psi_t, [i, j], [0, 1])
    psi_2d = psi_t.reshape(4, 2 ** (N - 2))
    rho = (psi_2d @ psi_2d.conj().T).real
    evs = jnp.linalg.eigvalsh(rho)
    evs = jnp.maximum(evs, 1e-10)
    return -jnp.sum(evs * jnp.log(evs))

def make_predictive_runner(block_pairs, N, n_steps, bob_gates, lambdas_j, d_hidden):
    """
    Builds the closed-loop predictive controller.
    """
    def _run(evals, evecs, dt, p1, p2, p_cross, T_quiet, eps_predictive, psi0, rng_key, K, rnn_params):
        
        S0 = compute_block_entropy(psi0, block_pairs[0][0], block_pairs[0][1], N)
        rho0 = extract_single_rho(psi0, bob_gates, N)
        z0 = density_to_bloch(rho0, lambdas_j)
        h0 = jnp.zeros(d_hidden)
        
        carry0 = (psi0, S0, jnp.int32(999), jnp.int32(0), rng_key, z0, h0)
        
        def step(carry, s):
            psi, S_prev, ssl, ng, key, z_prev, h_prev = carry
            key, _, ks = jax.random.split(key, 3)
            
            # Forecast
            h_next, r_pred = rnn_forward(rnn_params, h_prev, z_prev)
            z_pred = z_prev @ K.T + r_pred
            
            # Curvature spike prediction
            # kappa = ||z_pred - z_prev||
            pred_drift = jnp.sqrt(jnp.sum((z_pred - z_prev)**2))
            
            trigger = jnp.logical_and(ssl >= T_quiet,
                                      jnp.logical_and(s > 2, pred_drift > eps_predictive))
            
            gi = jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))
            ci = gi[0]
            
            psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
            
            ssl_n = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
            ng_n = ng + jnp.where(ci > 0, 1, 0)
            fired = jnp.where(ci > 0, 1.0, 0.0)
            
            # Extract state for next forecast step
            # Note: For full rigor, extraction happens after step.
            rho_curr = extract_single_rho(psi, bob_gates, N)
            z_curr = density_to_bloch(rho_curr, lambdas_j)
            S_curr = compute_block_entropy(psi, block_pairs[0][0], block_pairs[0][1], N)
            
            return (psi, S_curr, ssl_n, ng_n, key, z_curr, h_next), (fired, pred_drift)
            
        (psi_f, _, _, ng_f, _, _, _), (fired, drifts) = jax.lax.scan(step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
        return psi_f, ng_f, fired, drifts
        
    return jax.jit(jax.vmap(_run, in_axes=(None, None, None, None, None, None, None, None, 0, 0, None, None)))

def make_reactive_runner(block_pairs, N, n_steps):
    """
    Standard reactive entropy runner.
    """
    def _run(evals, evecs, dt, p1, p2, p_cross, T_quiet, eps_S, psi0, rng_key):
        S0 = compute_block_entropy(psi0, block_pairs[0][0], block_pairs[0][1], N)
        carry0 = (psi0, S0, jnp.int32(999), jnp.int32(0), rng_key)

        def step(carry, s):
            psi, S_prev, ssl, ng, key = carry
            key, _, ks = jax.random.split(key, 3)
            S_curr = compute_block_entropy(psi, block_pairs[0][0], block_pairs[0][1], N)
            S_vel = jnp.sum((S_curr - S_prev) ** 2)
            trigger = jnp.logical_and(ssl >= T_quiet,
                                      jnp.logical_and(s > 2, S_vel > eps_S))
            gi = jax.lax.cond(trigger, lambda: jnp.array((3, 0)), lambda: jnp.array((0, 0)))
            ci = gi[0]
            psi = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gi, ks)
            ssl_n = jax.lax.cond(ci > 0, lambda: jnp.int32(0), lambda: ssl + 1)
            ng_n = ng + jnp.where(ci > 0, 1, 0)
            fired = jnp.where(ci > 0, 1.0, 0.0)
            return (psi, S_curr, ssl_n, ng_n, key), (fired, S_vel)

        (psi_f, _, _, ng_f, _), (fired, S_vels) = jax.lax.scan(step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
        return psi_f, ng_f, fired, S_vels

    return jax.jit(jax.vmap(_run, in_axes=(None, None, None, None, None, None, None, None, 0, 0)))


# ---------------------------------------------------------------------------
# Master Execution
# ---------------------------------------------------------------------------

def run_program_ai(args):
    t0 = time.time()
    select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 80)
    print("Program AI — Predictive Semantic Transport")
    print("=" * 80)

    N = 12
    Lx, Ly = 3, 4
    dt_step = args.T_max / args.n_steps_per_sym
    p1 = float(1 - np.exp(-dt_step / 20.0))
    p2 = float(1 - np.exp(-dt_step / args.t2))
    p_cross = 0.03

    bell_dict = build_bell_alphabet()
    bell_labels = list(bell_dict.keys())
    bell_arr = jnp.stack(list(bell_dict.values()))
    lambdas_j = jnp.array(get_gell_mann_matrices())

    seeds = [11, 29, 47, 83, 101] if not args.smoke_test else [42]
    seq_lengths = args.sequence_lengths

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_ai_summary.json")
    all_records = []

    for depth in args.depth_sweep:
        gamma = float(compute_cone_overlap(depth, N))
        print(f"\n{'=' * 80}\nMERA depth={depth}  Gamma={gamma:.3f}\n{'=' * 80}")

        layer_pairs = get_mera_layer_pairs(depth, N)
        ent_block_pairs = layer_pairs[0] if layer_pairs else [(0, 1)]

        depth_results = {
            ctrl: {
                sl: {
                    "latent_diff": [],
                    "lyapunov": [],
                    "ricci_contraction": [],
                    "transport_coherence": [],
                    "mean_gates": []
                } for sl in seq_lengths
            } for ctrl in ["free", "reactive_entropy", "predictive_entropy"]
        }

        for seed in seeds:
            print(f"\n--- Running Seed: {seed} ---")
            
            H_train = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=args.j_std_train, seed=seed)
            H_test = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=args.j_std_test, seed=seed + 1000)

            evals_train, evecs_train = get_cached_eigh(H_train, (seed, args.j_std_train))
            evals_test, evecs_test = get_cached_eigh(H_test, (seed + 1000, args.j_std_test))

            alice_gates, bob_gates = build_mera_gates_z(depth=depth, seed=seed)
            bob_gates_j = jnp.array([g[0] for g in bob_gates])
            
            # Setup JIT runners
            pmap_reinject = jax.pmap(vmap_reinject_global, in_axes=(0, None, None, None, None), static_broadcasted_argnums=(3, 4))
            pmap_free = jax.pmap(vmap_free_global, in_axes=(None, None, None, None, None, None, None, None, 0, 0), static_broadcasted_argnums=(2, 4))
            
            vmap_reactive = make_reactive_runner(ent_block_pairs, N, args.n_steps_per_sym)
            pmap_reactive = jax.pmap(vmap_reactive, in_axes=(None, None, None, None, None, None, None, None, 0, 0))
            
            d_hidden = 16
            vmap_predictive = make_predictive_runner(ent_block_pairs, N, args.n_steps_per_sym, tuple(bob_gates), lambdas_j, d_hidden)
            pmap_predictive = jax.pmap(vmap_predictive, in_axes=(None, None, None, None, None, None, None, None, 0, 0, None, None))

            for sl in seq_lengths:
                print(f"\n  Sequence Length: {sl}")
                sequences = build_compositional_motifs(bell_dict, sl)
                ens_seeds = jnp.array([seed + 100 * i for i in range(args.n_ensembles)])
                traj_keys_batch = jax.vmap(lambda s: jax.random.split(jax.random.PRNGKey(s), args.n_traj_per_ensemble))(ens_seeds)

                def run_ensembles_parallel(evals_h, evecs_h, ctrl_type="free", K=None, rnn_params=None):
                    rhos_all = []
                    gates_all = []
                    
                    for seq_name, seq_labels in sequences.items():
                        alice_logical_0 = bell_dict[seq_labels[0]]
                        psi0_0 = _make_initial_y(N, alice_logical_0, alice_gates, bob_gates)
                        psi_batch = jnp.tile(psi0_0[None, None, :], (args.n_ensembles, args.n_traj_per_ensemble, 1))

                        alice_matrices = jnp.stack([g[0] for g in alice_gates])
                        alice_indices = tuple((int(g[1]), int(g[2])) for g in alice_gates)

                        psi_finals = []
                        total_gates = []
                        
                        for sym_t in range(sl):
                            if sym_t > 0:
                                new_logical = bell_dict[seq_labels[sym_t]]
                                psi_batch = pmap_reinject(psi_batch, new_logical, alice_matrices, alice_indices, N)

                            if ctrl_type == "free":
                                psi_batch = pmap_free(
                                    evals_h, evecs_h, N, jnp.float32(dt_step),
                                    args.n_steps_per_sym,
                                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                    psi_batch, traj_keys_batch)
                                ng = jnp.zeros_like(psi_batch[:, :, 0], dtype=jnp.int32)
                            elif ctrl_type == "reactive_entropy":
                                psi_batch, ng, *_ = pmap_reactive(
                                    evals_h, evecs_h, jnp.float32(dt_step),
                                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                    jnp.int32(args.T_quiet), jnp.float32(args.eps_entropy),
                                    psi_batch, traj_keys_batch)
                            elif ctrl_type == "predictive_entropy":
                                psi_batch, ng, *_ = pmap_predictive(
                                    evals_h, evecs_h, jnp.float32(dt_step),
                                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                    jnp.int32(args.T_quiet), jnp.float32(args.eps_predictive),
                                    psi_batch, traj_keys_batch, K, rnn_params)
                                    
                            psi_finals.append(psi_batch)
                            total_gates.append(ng)

                        psi_seq_batch = jnp.stack(psi_finals, axis=2)
                        vmap_extract = jax.vmap(extract_logical_density_matrices, in_axes=(0, None, None, None))
                        rhos_seq_batch = vmap_extract(psi_seq_batch, tuple(bob_gates), bell_arr, N)
                        
                        rhos_all.append(rhos_seq_batch)
                        gates_all.append(jnp.sum(jnp.stack(total_gates, axis=2), axis=2))

                    return jnp.concatenate(rhos_all, axis=1), jnp.concatenate(gates_all, axis=1)

                # --- PASS 1: Calibration (Free Dynamics) ---
                print(f"  [Pass 1] Calibrating Koopman-RNN Hybrid on Train set...")
                train_rhos, _ = run_ensembles_parallel(evals_train, evecs_train, ctrl_type="free")
                z_train_ens = jax.vmap(lambda rhos: density_to_bloch(rhos, lambdas_j))(train_rhos)
                # Flatten ensemble and trajectories
                z_train = z_train_ens.reshape(-1, sl, 15)
                
                K = fit_koopman_operator(z_train)
                rnn_params = train_residual_rnn(z_train, K, epochs=200)

                # --- PASS 2: Evaluation ---
                for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
                    print(f"  [Pass 2] Executing {ctrl} on Test set...")
                    test_rhos, test_gates = run_ensembles_parallel(
                        evals_test, evecs_test, ctrl_type=ctrl, K=K, rnn_params=rnn_params)
                        
                    for ens_idx in range(args.n_ensembles):
                        z_test = density_to_bloch(test_rhos[ens_idx], lambdas_j)
                        
                        # Diagnostics
                        diffs = z_test[:, 1:] - z_test[:, :-1]
                        latent_diff = float(jnp.mean(jnp.sum(diffs ** 2, axis=2)))
                        lyapunov = compute_lyapunov_rate(z_test)
                        ricci = compute_latent_ricci(z_test)
                        coherence = compute_transport_coherence(z_test)
                        
                        depth_results[ctrl][sl]["latent_diff"].append(latent_diff)
                        depth_results[ctrl][sl]["lyapunov"].append(lyapunov)
                        depth_results[ctrl][sl]["ricci_contraction"].append(ricci)
                        depth_results[ctrl][sl]["transport_coherence"].append(coherence)
                        depth_results[ctrl][sl]["mean_gates"].append(float(jnp.mean(test_gates[ens_idx]).real))

                    print(f"    {ctrl:>18} -> Diff:{float(np.mean(depth_results[ctrl][sl]['latent_diff'])):.4f} "
                          f"  Lyap:{float(np.mean(depth_results[ctrl][sl]['lyapunov'])):.4f} "
                          f"  Ricci:{float(np.mean(depth_results[ctrl][sl]['ricci_contraction'])):.4f} "
                          f"  Coherence:{float(np.mean(depth_results[ctrl][sl]['transport_coherence'])):.3f}")

        # Compile final depth record
        record = {"mera_depth": depth, "gamma": gamma, "seq_lengths": {}}
        for sl in seq_lengths:
            record["seq_lengths"][str(sl)] = {}
            for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
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
            "experiment": "predictive_semantic_transport",
            "seed_suite": seeds,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1)
        }
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)

    # Summary master printout
    print("\n" + "=" * 120)
    print(f"  {'d':>2} {'L':>3} {'ctrl':>18} {'Diff':>9} {'Lyap':>9} {'Ricci':>9} {'Coherence':>10} {'Gates':>8}")
    print("-" * 120)
    for r in all_records:
        d = r["mera_depth"]
        for sl in seq_lengths:
            for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
                m = r["seq_lengths"][str(sl)][ctrl]
                print(f"  {d:>2} {sl:>3} {ctrl:>18} {m['latent_diff']['mean']:>9.4f} {m['lyapunov']['mean']:>9.4f} "
                      f"{m['ricci_contraction']['mean']:>9.4f} {m['transport_coherence']['mean']:>10.3f} {m['mean_gates']['mean']:>8.1f}")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T-max", type=float, default=4.5)
    p.add_argument("--n-steps-per-sym", type=int, default=40)
    p.add_argument("--n-ensembles", type=int, default=8)
    p.add_argument("--n-traj-per-ensemble", type=int, default=30)
    p.add_argument("--j-std-train", type=float, default=0.9)
    p.add_argument("--j-std-test", type=float, default=1.1)
    p.add_argument("--t2", type=float, default=12.0)
    p.add_argument("--eps-entropy", type=float, default=0.0005)
    p.add_argument("--eps-predictive", type=float, default=0.015)
    p.add_argument("--T-quiet", type=int, default=8)
    p.add_argument("--out-dir", default="program_ai_results")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--sequence-lengths", nargs="+", type=int, default=[16, 32])
    p.add_argument("--depth-sweep", nargs="+", type=int, default=[3, 4])
    args = p.parse_args()

    if args.smoke_test:
        args.n_steps_per_sym = 10
        args.n_ensembles = 2
        args.n_traj_per_ensemble = 15
        args.sequence_lengths = [8]
        args.depth_sweep = [3]
        print("[SMOKE TEST]")

    run_program_ai(args)
