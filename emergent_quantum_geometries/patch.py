import os

with open('program_aj_predictive_horizon.py', 'r') as f:
    content = f.read()

target = """                print(f"  [Pass 1] Calibrating Koopman-RNN Hybrid on Train set...")
                train_rhos, _ = run_ensembles_parallel(evals_train, evecs_train, ctrl_type="free")
                z_train_ens = jax.vmap(lambda rhos: density_to_bloch(rhos, lambdas_j))(train_rhos)
                z_train = z_train_ens.reshape(-1, sl, 15)
                
                K = fit_koopman_operator(z_train)
                rnn_params = train_residual_rnn(z_train, K, epochs=200)

                for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
                    k_list = args.k_horizons if ctrl == "predictive_entropy" else [1]
                    for k_horiz in k_list:
                        ctrl_name = ctrl if ctrl != "predictive_entropy" else f"predictive_k{k_horiz}"
                        if ctrl_name not in depth_results:
                            for sl2 in seq_lengths:
                                depth_results[ctrl_name] = {sl2: {"latent_diff": [], "lyapunov": [], "ricci_contraction": [], "transport_coherence": [], "mean_gates": []} for sl2 in seq_lengths}

                        print(f"  [Pass 2] Executing {ctrl_name} on Test set...")
                        test_rhos, test_gates = run_ensembles_parallel(
                            evals_test, evecs_test, ctrl_type=ctrl, k_horiz=k_horiz, K=K, rnn_params=rnn_params)
                            
                        z_test_all = []
                        for ens_idx in range(args.n_ensembles):
                            z_test = density_to_bloch(test_rhos[ens_idx], lambdas_j)
                            
                            diffs = z_test[:, 1:] - z_test[:, :-1]
                            latent_diff = float(jnp.mean(jnp.sum(diffs ** 2, axis=2)))
                            lyapunov = compute_lyapunov_rate(z_test)
                            ricci = compute_latent_ricci(z_test)
                            coherence = compute_transport_coherence(z_test)
                            
                            depth_results[ctrl_name][sl]["latent_diff"].append(latent_diff)
                            depth_results[ctrl_name][sl]["lyapunov"].append(lyapunov)
                            depth_results[ctrl_name][sl]["ricci_contraction"].append(ricci)
                            depth_results[ctrl_name][sl]["transport_coherence"].append(coherence)
                            depth_results[ctrl_name][sl]["mean_gates"].append(float(jnp.mean(test_gates[ens_idx]).real))
                            
                            z_test_all.append(z_test)
                        
                        try:
                            npz_path = os.path.join(args.out_dir, f"traj_d{depth}_s{seed}_L{sl}_{ctrl_name}.npz")
                            np.savez_compressed(npz_path, z_coords=np.array(z_test_all), K_matrix=np.array(K), gates=np.array(test_gates))
                        except Exception as e:
                            print(f"Failed to export npz: {e}")"""

replacement = """                # Initialize structures if not already there
                for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
                    k_list = args.k_horizons if ctrl == "predictive_entropy" else [1]
                    for k_horiz in k_list:
                        ctrl_name = ctrl if ctrl != "predictive_entropy" else f"predictive_k{k_horiz}"
                        if ctrl_name not in depth_results:
                            for sl2 in seq_lengths:
                                depth_results[ctrl_name] = {sl2: {"latent_diff": [], "lyapunov": [], "ricci_contraction": [], "transport_coherence": [], "mean_gates": []} for sl2 in seq_lengths}

                # Check which controllers need to run for this sequence length
                to_run = []
                for ctrl in ["free", "reactive_entropy", "predictive_entropy"]:
                    k_list = args.k_horizons if ctrl == "predictive_entropy" else [1]
                    for k_horiz in k_list:
                        ctrl_name = ctrl if ctrl != "predictive_entropy" else f"predictive_k{k_horiz}"
                        npz_path = os.path.join(args.out_dir, f"traj_d{depth}_s{seed}_L{sl}_{ctrl_name}.npz")
                        
                        if os.path.exists(npz_path):
                            print(f"  [RESUME] Loading cached data for {ctrl_name}...")
                            try:
                                data = np.load(npz_path)
                                z_test_all = jnp.array(data["z_coords"])
                                test_gates = jnp.array(data["gates"])
                                
                                for ens_idx in range(args.n_ensembles):
                                    z_test = z_test_all[ens_idx]
                                    diffs = z_test[:, 1:] - z_test[:, :-1]
                                    latent_diff = float(jnp.mean(jnp.sum(diffs ** 2, axis=2)))
                                    lyapunov = compute_lyapunov_rate(jnp.expand_dims(z_test, 0))
                                    ricci = compute_latent_ricci(jnp.expand_dims(z_test, 0))
                                    coherence = compute_transport_coherence(jnp.expand_dims(z_test, 0))
                                    
                                    depth_results[ctrl_name][sl]["latent_diff"].append(latent_diff)
                                    depth_results[ctrl_name][sl]["lyapunov"].append(lyapunov)
                                    depth_results[ctrl_name][sl]["ricci_contraction"].append(ricci)
                                    depth_results[ctrl_name][sl]["transport_coherence"].append(coherence)
                                    depth_results[ctrl_name][sl]["mean_gates"].append(float(jnp.mean(test_gates[ens_idx]).real))
                                continue
                            except Exception as e:
                                print(f"  [ERROR] Failed to read {npz_path}, will re-run: {e}")
                                to_run.append((ctrl, k_horiz, ctrl_name, npz_path))
                        else:
                            to_run.append((ctrl, k_horiz, ctrl_name, npz_path))

                if len(to_run) > 0:
                    print(f"  [Pass 1] Calibrating Koopman-RNN Hybrid on Train set...")
                    train_rhos, _ = run_ensembles_parallel(evals_train, evecs_train, ctrl_type="free")
                    z_train_ens = jax.vmap(lambda rhos: density_to_bloch(rhos, lambdas_j))(train_rhos)
                    z_train = z_train_ens.reshape(-1, sl, 15)
                    
                    K = fit_koopman_operator(z_train)
                    rnn_params = train_residual_rnn(z_train, K, epochs=200)

                    for ctrl, k_horiz, ctrl_name, npz_path in to_run:
                        print(f"  [Pass 2] Executing {ctrl_name} on Test set...")
                        test_rhos, test_gates = run_ensembles_parallel(
                            evals_test, evecs_test, ctrl_type=ctrl, k_horiz=k_horiz, K=K, rnn_params=rnn_params)
                            
                        z_test_all = []
                        for ens_idx in range(args.n_ensembles):
                            z_test = density_to_bloch(test_rhos[ens_idx], lambdas_j)
                            
                            diffs = z_test[:, 1:] - z_test[:, :-1]
                            latent_diff = float(jnp.mean(jnp.sum(diffs ** 2, axis=2)))
                            lyapunov = compute_lyapunov_rate(jnp.expand_dims(z_test, 0))
                            ricci = compute_latent_ricci(jnp.expand_dims(z_test, 0))
                            coherence = compute_transport_coherence(jnp.expand_dims(z_test, 0))
                            
                            depth_results[ctrl_name][sl]["latent_diff"].append(latent_diff)
                            depth_results[ctrl_name][sl]["lyapunov"].append(lyapunov)
                            depth_results[ctrl_name][sl]["ricci_contraction"].append(ricci)
                            depth_results[ctrl_name][sl]["transport_coherence"].append(coherence)
                            depth_results[ctrl_name][sl]["mean_gates"].append(float(jnp.mean(test_gates[ens_idx]).real))
                            
                            z_test_all.append(z_test)
                        
                        try:
                            np.savez_compressed(npz_path, z_coords=np.array(z_test_all), K_matrix=np.array(K), gates=np.array(test_gates))
                        except Exception as e:
                            print(f"Failed to export npz: {e}")"""

if target in content:
    with open('program_aj_predictive_horizon.py', 'w') as f:
        f.write(content.replace(target, replacement))
    print('replaced')
else:
    print('target not found')
