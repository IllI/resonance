"""
Program P2 -- Continuous Phase Tracking (TPU-native)
=========================================================
Reconstructs an evolving phase trajectory phi(t) = w0*t + alpha*t^2 
under strong scrambling and noise.

Features:
  - Continuous trajectory tracking (phi_hat(t) for all t)
  - True Probe-Conditioned Adaptive Controller in pure JAX
  - Ultra-fast jax.lax.scan + vmap execution
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from program_l_tpu import (
    MODEL_BUILDERS, sample_noise_params, _get_eigh,
    apply_clifford_psi, CLIFFORDS, N_CLIFF,
    select_backend, _jax_device_summary,
)

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Pure-JAX helpers
# ---------------------------------------------------------------------------

def _apply_U_psi_jax(psi, evals, evecs, dt):
    psi_e = evecs.conj().T @ psi
    psi_e = psi_e * jnp.exp(-1j * evals * dt)
    return evecs @ psi_e

def _apply_site_gate_static_site(psi, gate_2x2, N, site_val):
    dim = 2 ** N
    stride = 2 ** (N - 1 - site_val)
    psi = psi.reshape(-1, 2 * stride)
    lo = psi[:, :stride]
    hi = psi[:, stride:]
    lo_new = gate_2x2[0, 0] * lo + gate_2x2[0, 1] * hi
    hi_new = gate_2x2[1, 0] * lo + gate_2x2[1, 1] * hi
    psi = jnp.concatenate([lo_new, hi_new], axis=1)
    return psi.reshape(dim)

def _apply_site_gate_jax(psi, gate_2x2, N, site):
    branches = [
        lambda p, g, s=s_val: _apply_site_gate_static_site(p, g, N, s)
        for s_val in range(N)
    ]
    return jax.lax.switch(site, branches, psi, gate_2x2)

def _apply_noise_jax(psi, N, noise_row, dt):
    gamma = jnp.clip(noise_row[0] * dt, 0.0, 0.5)
    dim = 2 ** N
    idx = jnp.arange(dim)
    n_ones = jnp.zeros(dim, dtype=jnp.float32)
    for q in range(N):
        n_ones = n_ones + ((idx >> (N - 1 - q)) & 1).astype(jnp.float32)
    damp = jnp.power((1.0 - gamma), n_ones).astype(jnp.complex64)
    psi = psi * damp
    norm = jnp.linalg.norm(psi)
    return jnp.where(norm > 1e-12, psi / norm, psi)

def _reconstruct_phase_jax(psi, N):
    dim = 2 ** N
    half = dim // 2
    a = psi[:half]
    b = psi[half:]
    rho01 = jnp.sum(a * jnp.conj(b))
    sx = jnp.real(rho01 + jnp.conj(rho01))
    sy = jnp.real(1j * (rho01 - jnp.conj(rho01)))
    phi_hat = jnp.arctan2(sy, sx) % (2 * jnp.pi)
    coherence = jnp.sqrt(sx ** 2 + sy ** 2)
    return phi_hat, coherence

# ---------------------------------------------------------------------------
# JAX Scan Body
# ---------------------------------------------------------------------------

def _make_scan_step(evals, evecs, cliffords_stack, N, dt,
                    noise_seq, ctrl_name, w0, alpha):
    """
    carry = (psi, prev_coherence, best_ci, max_gain)
    """
    @jax.jit
    def step_fn(carry, step_idx):
        psi, prev_coh, best_ci, max_gain = carry
        
        # 1. Controller Logic
        if ctrl_name == "static":
            ci = (1 + (step_idx % (N_CLIFF - 1))).astype(jnp.int32)
            site = (step_idx % N).astype(jnp.int32)
            
        elif ctrl_name == "agnostic":
            # Systematic sweep proxy
            ci = (1 + ((step_idx + N // 2) % (N_CLIFF - 1))).astype(jnp.int32)
            site = ((step_idx + 1) % N).astype(jnp.int32)
            
        elif ctrl_name == "probe_conditioned":
            # True Adaptive JAX Controller!
            # Probe phase: steps 0 to N_CLIFF-2. Test each Clifford.
            is_probe = step_idx < (N_CLIFF - 1)
            test_ci  = (step_idx + 1).astype(jnp.int32)
            
            # Use the best_ci during transport, pulsing every 2 steps
            transport_ci = jnp.where(step_idx % 2 == 0, best_ci, 0)
            
            ci = jnp.where(is_probe, test_ci, transport_ci).astype(jnp.int32)
            site = jnp.int32(0) # Focus equalization on Qubit 0

        else:
            ci = jnp.int32(0)
            site = jnp.int32(0)

        # 2. Apply Controller Gate
        gate = cliffords_stack[ci]
        psi = jax.lax.cond(
            ci > 0,
            lambda p: _apply_site_gate_jax(p, gate, N, site),
            lambda p: p,
            psi,
        )

        # 3. Evolution & Noise
        psi = _apply_U_psi_jax(psi, evals, evecs, dt)
        noise_row = noise_seq[step_idx]
        psi = _apply_noise_jax(psi, N, noise_row, dt)

        # 4. Continuous Phase Injection (Chirp)
        # phi(t) = w0*t + alpha*t^2 => delta_phi = (w0 + 2*alpha*t)*dt
        t_curr = step_idx * dt
        d_phi = (w0 + 2.0 * alpha * t_curr) * dt
        
        # Apply d_phi to Qubit 0 (Z rotation: e^{-i d_phi/2} on |0>, e^{i d_phi/2} on |1>)
        # Since we just want relative phase, we can apply e^{i d_phi} to |1> half.
        dim = 2 ** N
        half = dim // 2
        psi_0 = psi[:half]
        psi_1 = psi[half:] * jnp.exp(1j * d_phi)
        psi = jnp.concatenate([psi_0, psi_1], axis=0)

        # 5. Continuous Reconstruction
        phi_hat, curr_coh = _reconstruct_phase_jax(psi, N)
        
        # 6. Adaptive State Update
        gain = curr_coh - prev_coh
        
        # If in probe phase and gain is better, update best_ci
        is_probe = step_idx < (N_CLIFF - 1)
        update = jnp.logical_and(is_probe, gain > max_gain)
        
        new_best_ci = jnp.where(update, ci, best_ci)
        new_max_gain = jnp.where(update, gain, max_gain)

        new_carry = (psi, curr_coh, new_best_ci, new_max_gain)
        
        # Output is the trajectory snapshot at t_curr
        out_state = jnp.stack([phi_hat, curr_coh])
        return new_carry, out_state

    return step_fn

# ---------------------------------------------------------------------------
# Single trajectory: vmap-able
# ---------------------------------------------------------------------------

def _run_one_trajectory(phi_0, psi0_pre_degrade, evals, evecs, cliffords_stack,
                        N, dt, n_steps, noise_seq, ctrl_name, w0, alpha):
    
    step_fn = _make_scan_step(evals, evecs, cliffords_stack, N, dt,
                              noise_seq, ctrl_name, w0, alpha)
    
    # Initialize adaptive carry
    _, initial_coh = _reconstruct_phase_jax(psi0_pre_degrade, N)
    carry0 = (psi0_pre_degrade, initial_coh, jnp.int32(0), jnp.float32(-999.0))
    
    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, traj_out = jax.lax.scan(step_fn, carry0, steps)
    
    phi_hat_traj = traj_out[:, 0]
    coh_traj     = traj_out[:, 1]
    best_ci      = final_carry[2]
    
    return phi_hat_traj, coh_traj, best_ci

# Vectorize over the phi_0 sweep dimension
_run_batch = jax.jit(jax.vmap(
    _run_one_trajectory,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None),
), static_argnums=(5, 7, 9))

# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def trajectory_r2(y_true_traj, y_pred_traj):
    """Compute R2 for continuous circular trajectories."""
    # Unroll phase jumps
    y_true_unrolled = np.unwrap(y_true_traj)
    y_pred_unrolled = np.unwrap(y_pred_traj)
    
    residuals = y_true_unrolled - y_pred_unrolled
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true_unrolled - np.mean(y_true_unrolled))**2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def run_experiment_p2(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 65)
    print("Program P2 -- Continuous Phase Tracking (TPU-native)")
    print("=" * 65)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"w0={args.w0} alpha={args.alpha} (Chirp Injection)")
    
    dt = args.T_max / args.n_steps
    phi_sweep = np.linspace(0, 2 * np.pi, args.n_phases, endpoint=False)
    phi_sweep_j = jnp.array(phi_sweep, dtype=jnp.float32)
    rng = np.random.default_rng(args.seed)

    cliffords_np = np.stack([np.eye(2, dtype=np.complex64)] +
                            [np.array(g, dtype=np.complex64) for g in CLIFFORDS[1:]], axis=0)
    cliffords_stack = jnp.array(cliffords_np)

    all_records = []
    summary = {}

    for model in args.models:
        print(f"\n-- Model: {model} --")
        H = MODEL_BUILDERS[model](args.N, args.seed)
        evals_np, evecs_np = _get_eigh(H, (model, args.N, args.seed))
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for channel in args.channels:
            for sigma in [float(s) for s in args.sigmas]:
                if channel != "none" and sigma == 0.0: continue

                key = f"{model}/{channel}/sig={sigma:.1f}"
                ctrl_records = {c: [] for c in args.controllers}

                for b in range(args.B):
                    seed_b = args.seed + b * 1000
                    noise_params = sample_noise_params(rng)
                    deg_rng = np.random.default_rng(seed_b + 54321)
                    noise_rng = np.random.default_rng(seed_b)

                    # Noise sequence
                    noise_seq_np = np.zeros((args.n_steps, 8), dtype=np.float32)
                    gamma = float(noise_params.get("gamma", 0.01))
                    kappa = float(noise_params.get("kappa", 0.01))
                    noise_seq_np[:, 0] = noise_rng.normal(gamma, 0.005, args.n_steps).clip(0)
                    noise_seq_np[:, 1] = noise_rng.normal(kappa, 0.005, args.n_steps).clip(0)
                    noise_seq_j = jnp.array(noise_seq_np)

                    # Pre-encode phi_0
                    psi0_batch = []
                    for phi in phi_sweep:
                        dim, half = 2 ** args.N, 2 ** args.N // 2
                        psi = np.zeros(dim, dtype=np.complex64)
                        psi[0] = 1.0 / np.sqrt(2)
                        psi[half] = np.exp(1j * float(phi)) / np.sqrt(2)

                        # CPU degradation
                        if channel == "dephasing" and sigma > 0:
                            angles = deg_rng.normal(0, sigma, args.N)
                            phase = np.ones(dim, dtype=np.complex64)
                            for q in range(args.N):
                                idx = np.arange(dim)
                                bit = (idx >> (args.N - 1 - q)) & 1
                                ph = np.where(bit == 0, np.exp(1j * angles[q]/2), np.exp(-1j * angles[q]/2))
                                phase *= ph
                            psi = psi * phase
                            psi /= np.linalg.norm(psi)
                        elif channel == "scrambling" and sigma > 0:
                            if deg_rng.random() < sigma:
                                ci = int(deg_rng.integers(1, N_CLIFF))
                                site = int(deg_rng.integers(0, args.N))
                                psi = apply_clifford_psi(psi, args.N, ci, site)
                                
                        psi0_batch.append(psi)

                    psi0_batch_j = jnp.array(np.stack(psi0_batch, axis=0))

                    for ctrl in args.controllers:
                        t_run = time.time()
                        phi_hat_trajs, coh_trajs, best_cis = _run_batch(
                            phi_sweep_j, psi0_batch_j, evals, evecs, cliffords_stack,
                            args.N, jnp.float32(dt), args.n_steps, noise_seq_j, ctrl,
                            jnp.float32(args.w0), jnp.float32(args.alpha)
                        )
                        
                        phi_hat_trajs = np.array(jax.device_get(phi_hat_trajs))
                        best_cis = np.array(jax.device_get(best_cis))
                        elapsed_ms = (time.time() - t_run) * 1000

                        # Calculate ideal true trajectory
                        t_arr = np.arange(args.n_steps) * dt
                        
                        for i, phi_0 in enumerate(phi_sweep):
                            phi_true_traj = (phi_0 + args.w0 * t_arr + args.alpha * t_arr**2) % (2*np.pi)
                            r2_traj = trajectory_r2(phi_true_traj, phi_hat_trajs[i])
                            
                            rec = {
                                "phi_0": float(phi_0),
                                "r2_traj": float(r2_traj),
                                "best_ci": int(best_cis[i]),
                                "model": model,
                                "controller": ctrl,
                                "channel": channel,
                                "sigma": sigma,
                                "B": b,
                            }
                            ctrl_records[ctrl].append(rec)
                            all_records.append(rec)

                        print(f"    [{ctrl:17s}] batch {b}: {args.n_phases} trajectories in {elapsed_ms:.0f}ms")

                # Metrics
                for ctrl in args.controllers:
                    recs = ctrl_records[ctrl]
                    mean_r2 = float(np.mean([r["r2_traj"] for r in recs]))
                    summary[f"{key}/{ctrl}"] = {"r2_traj": mean_r2, "n": len(recs)}

                s_key = f"{key}/static"
                p_key = f"{key}/probe_conditioned"
                if s_key in summary and p_key in summary:
                    d_r2 = summary[p_key]["r2_traj"] - summary[s_key]["r2_traj"]
                    passed = (d_r2 > 0.1)
                    print(f"  {'PASS' if passed else '    '} {key:35s} DR2_traj={d_r2:+.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_p2_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--T-max", type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=30, dest="n_steps")
    p.add_argument("--w0", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["XXZ", "DisorderedXXZ_W3", "OAT"])
    p.add_argument("--controllers", nargs="+", default=["static", "agnostic", "probe_conditioned"])
    p.add_argument("--channels", nargs="+", default=["none", "dephasing", "scrambling"])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--n-phases", type=int, default=12, dest="n_phases")
    p.add_argument("--out-dir", default="program_p2_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.N = 6
        args.n_steps = 15
        args.B = 1
        args.models = ["OAT"]
        args.channels = ["scrambling"]
        args.sigmas = [0.5]
        args.n_phases = 4
        print("[SMOKE TEST] N=6, B=1")

    run_experiment_p2(args)
