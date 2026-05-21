"""
Program V2 -- Adaptive Intervention Threshold Scan
===================================================
3x4 grid geometry with Alice (0, 1) and Bob (site_dist classes).
Maps out the transition boundary between restraint-optimal and intervention-beneficial regimes.
Sweeps:
- dephasing rates (T2)
- disorder strengths (J_std)
- vector adaptive thresholds (tau_thresh)
- transport distance classes (short, mid, long)
Baselines include 'free' (true passive coherent transport), 'static', 'dd', and 'vector_adaptive'.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from functools import partial

# Check for JAX
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import (
    _get_eigh, CLIFFORDS, N_CLIFF,
    select_backend,
)
from program_t_tpu import (
    build_grid_hamiltonian,
    get_bob_candidates,
)
from program_v_tpu import (
    apply_gate_psi,
    apply_stochastic_dephasing,
    apply_stochastic_amplitude_damping,
    make_initial_state,
    get_causal_observables_psi,
    run_probe_stage_v,
    run_trajectory_v,
    compute_reduced_rho_AB,
    compute_phase_sensitive_metrics,
)

# ---------------------------------------------------------------------------
# Main Sweep Coordinator for Phase Diagram
# ---------------------------------------------------------------------------

def run_experiment_v2(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 70)
    print("Program V2 -- Adaptive Intervention Threshold Scan (JAX)")
    print("=" * 70)
    
    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    print(f"Grid: {Lx}x{Ly} = {N} qubits  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Number of Trajectories per Batch: {args.n_trajectories}")
    
    dt = args.T_max / args.n_steps
    T1_phys = 20.0
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    p_cross = 0.03
    
    bob_cands = get_bob_candidates(Lx, Ly)
    summary = {}
    all_records = []

    if HAS_JAX:
        run_trajectories_vmapped = jax.jit(
            jax.vmap(run_trajectory_v, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)),
            static_argnums=(2, 4, 8)
        )
    else:
        print("[ERROR] JAX is required to run Program V2.")
        sys.exit(1)

    # 1. Sweep over Disorder Strengths (J_std)
    for J_std in args.j_std_sweep:
        print(f"\n== Disorder Strength J_std: {J_std:.2f} ==")
        H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=J_std, seed=args.seed)
        evals_np, evecs_np = _get_eigh(H, (f"GridXXZ_J{J_std:.2f}", N, args.seed))
        
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        # 2. Sweep over Dephasing Rates (T2)
        for T2 in args.t2_sweep:
            p2 = float(1.0 - np.exp(-dt / T2))
            print(f"  -- Dephasing T2: {T2:.1f} (p2={p2:.4f}) --")

            # 3. Sweep over Transport Distance Classes
            for dist_class, bob_list in bob_cands.items():
                if not bob_list: continue
                bob_site, bob_dist = bob_list[len(bob_list)//2]
                print(f"    -- Distance: {dist_class} (Bob=site {bob_site}, dist={bob_dist}) --")
                
                # Probe stage is JIT-compiled and runs entirely in JAX
                S_proj = run_probe_stage_v(evals, evecs, N, jnp.float32(dt), bob_site)

                # 4. Sweep over Controllers
                for ctrl_name in args.controllers:
                    # If adaptive, we sweep thresholds; otherwise, run once
                    current_taus = args.tau_sweep if ctrl_name == "vector_adaptive" else [0.0]
                    
                    for tau in current_taus:
                        ctrl_records = []
                        
                        for b in range(args.B):
                            seed_b = args.seed + b * 1000 + hash(ctrl_name) % 5000
                            b_rng = np.random.default_rng(seed_b)
                            
                            # Strictly OOD random Haar State Generator
                            u = b_rng.uniform(0, 1)
                            v = b_rng.uniform(0, 1)
                            theta = np.arccos(2.0 * u - 1.0)
                            phi = 2.0 * np.pi * v
                            Haar_c0 = np.cos(theta / 2.0)
                            Haar_c1 = np.exp(1j * phi) * np.sin(theta / 2.0)
                            
                            traj_keys = jax.random.split(jax.random.PRNGKey(seed_b), args.n_trajectories)
                            
                            # Map 'free' to adaptive with large threshold to avoid code duplication
                            actual_ctrl = "vector_adaptive" if ctrl_name == "free" else ctrl_name
                            actual_tau = 999.0 if ctrl_name == "free" else tau
                            
                            psi_finals, n_gates_batch = run_trajectories_vmapped(
                                evals, evecs, N, jnp.float32(dt), args.n_steps,
                                jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                                actual_ctrl, jnp.float32(actual_tau), S_proj, bob_site,
                                jnp.complex64(Haar_c0), jnp.complex64(Haar_c1), traj_keys
                            )
                            
                            rho_AB = compute_reduced_rho_AB(psi_finals, N, bob_site)
                            metrics_arr = compute_phase_sensitive_metrics(rho_AB, jnp.complex64(Haar_c0), jnp.complex64(Haar_c1))
                            
                            f_ent, mi, b_sim = float(metrics_arr[0]), float(metrics_arr[1]), float(metrics_arr[2])
                            avg_gates = float(np.mean(np.array(n_gates_batch)))
                            
                            rec = {
                                "J_std": float(J_std),
                                "T2": float(T2),
                                "dist_class": dist_class,
                                "bob_site": int(bob_site),
                                "bob_dist": float(bob_dist),
                                "controller": ctrl_name,
                                "tau": float(tau),
                                "batch": b,
                                "F_ent": f_ent,
                                "mutual_information": mi,
                                "bloch_similarity": b_sim,
                                "n_interventions": avg_gates,
                                "efficiency": float(f_ent / (avg_gates + 1e-9))
                            }
                            ctrl_records.append(rec)
                            all_records.append(rec)
                            
                        # Compute averages over batches
                        avg_fent = np.mean([r["F_ent"] for r in ctrl_records])
                        avg_mi = np.mean([r["mutual_information"] for r in ctrl_records])
                        avg_bloch = np.mean([r["bloch_similarity"] for r in ctrl_records])
                        avg_gates_all = np.mean([r["n_interventions"] for r in ctrl_records])
                        
                        label = f"{ctrl_name}_tau{tau:.4f}" if ctrl_name == "vector_adaptive" else ctrl_name
                        key = f"J_{J_std:.2f}/T2_{T2:.1f}/{dist_class}/{label}"
                        summary[key] = {
                            "F_ent": float(avg_fent),
                            "mutual_information": float(avg_mi),
                            "bloch_similarity": float(avg_bloch),
                            "n_interventions": float(avg_gates_all),
                            "n": len(ctrl_records)
                        }
                        print(f"      [{label:22s}] F_ent={avg_fent:.4f}  MutInfo={avg_mi:.4f}  BlochSim={avg_bloch:.4f}  Gates={avg_gates_all:.1f}")
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HEARTBEAT: Completed {label} for J={J_std:.2f}, T2={T2:.1f}")

    # Save output dataset
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_v2_N{N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--B", type=int, default=3)
    p.add_argument("--n-trajectories", type=int, default=100, dest="n_trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["GridXXZ_hetero"])
    p.add_argument("--controllers", nargs="+", default=["free", "static", "dd", "vector_adaptive"])
    
    # Grid search sweep arrays
    p.add_argument("--t2-sweep", nargs="+", type=float, default=[8.0, 16.0, 32.0], dest="t2_sweep")
    p.add_argument("--j-std-sweep", nargs="+", type=float, default=[0.2, 0.6, 1.0], dest="j_std_sweep")
    p.add_argument("--tau-sweep", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.03, 0.06, 0.12, 0.25, 0.50, 1.0], dest="tau_sweep")
    
    p.add_argument("--out-dir", default="program_v2_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.Lx, args.Ly = 2, 3
        args.n_steps = 12
        args.n_trajectories = 10
        args.B = 1
        args.controllers = ["free", "vector_adaptive"]
        args.t2_sweep = [12.0]
        args.j_std_sweep = [0.4]
        args.tau_sweep = [0.01, 0.50]
        print("[SMOKE TEST] Running lightweight sweep")

    run_experiment_v2(args)
