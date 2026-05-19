"""
Program Q -- Blind Information Recovery (TPU-native)
=========================================================
Encodes a hidden random quantum state, subjects it to strong
scrambling and noise, and evaluates endpoint fidelity F = |<psi_ideal|psi>|^2.
The adaptive sparse controller intervenes only when observable instability
exceeds a threshold, without ever accessing the target state or decoder.
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

def _get_coherence(psi, N):
    dim = 2 ** N
    half = dim // 2
    a = psi[:half]
    b = psi[half:]
    rho01 = jnp.sum(a * jnp.conj(b))
    sx = jnp.real(rho01 + jnp.conj(rho01))
    sy = jnp.real(1j * (rho01 - jnp.conj(rho01)))
    return jnp.sqrt(sx ** 2 + sy ** 2)

# ---------------------------------------------------------------------------
# JAX Scan Body
# ---------------------------------------------------------------------------

def _make_scan_step(evals, evecs, cliffords_stack, N, dt,
                    noise_seq, random_cis, ctrl_name, tau_thresh, N_probe=15):
    """
    carry = (psi, sma_buf, best_ci, max_gain, cooldown)
    """
    SMA_LEN = 5

    @jax.jit
    def step_fn(carry, step_idx):
        psi, sma_buf, best_ci, max_gain, cooldown = carry
        
        coh_before = _get_coherence(psi, N)
        sma = jnp.mean(sma_buf)
        
        # Instability metric: sudden drop in coherence below the moving average
        instability = sma - coh_before
        trigger = jnp.logical_and(instability > tau_thresh, cooldown == 0)

        site = jnp.int32(0) # Targeted intervention on Q0
        
        # 1. Controller Logic
        if ctrl_name == "static":
            ci = jnp.int32(0)
            
        elif ctrl_name == "periodic_sweep":
            # Deterministic Floquet-style scanning (from Program P2)
            ci = (1 + ((step_idx + N // 2) % (N_CLIFF - 1))).astype(jnp.int32)
            site = ((step_idx + 1) % N).astype(jnp.int32)
            
        elif ctrl_name == "random_sparse":
            # Fire at roughly the same rate as the adaptive trigger (e.g. 10% prob)
            # Use the pre-sampled random_cis array to determine both firing and CI
            val = random_cis[step_idx]
            # random_cis is randomly sampled from 0 to N_CLIFF-1.
            # If we want 10% sparse firing, we can say if val > 0 and step % 10 == 0
            # A simpler way: we'll just fire sparsely using step parity for baseline randomness
            fire = jnp.logical_and(val > 0, step_idx % 8 == 0)
            ci = jnp.where(fire, val, 0).astype(jnp.int32)
            
        elif ctrl_name == "adaptive_sparse":
            # Probe Phase (Estimate Response Jacobian)
            is_probe = step_idx < N_probe
            test_ci  = (step_idx % (N_CLIFF - 1) + 1).astype(jnp.int32)
            
            # Transport Phase (Sparse Predictive Stabilization)
            is_transport = step_idx >= N_probe
            
            transport_ci = jnp.where(trigger, best_ci, 0)
            ci = jnp.where(is_probe, test_ci, transport_ci).astype(jnp.int32)
            
        else:
            ci = jnp.int32(0)

        # 2. Apply Controller Gate
        gate = cliffords_stack[ci]
        psi = jax.lax.cond(
            ci > 0,
            lambda p: _apply_site_gate_jax(p, gate, N, site),
            lambda p: p,
            psi,
        )

        # Evaluate Probe Gain (Instantaneous Coherence Response)
        coh_after = _get_coherence(psi, N)
        gain = coh_after - coh_before
        
        if ctrl_name == "adaptive_sparse":
            is_probe = step_idx < N_probe
            update_probe = jnp.logical_and(is_probe, gain > max_gain)
            best_ci = jnp.where(update_probe, ci, best_ci)
            max_gain = jnp.where(update_probe, gain, max_gain)
            
            # Cooldown management
            fired_transport = jnp.logical_and(is_transport, trigger)
            cooldown = jnp.where(fired_transport, 3, jnp.maximum(0, cooldown - 1))
            
        # 3. Evolution & Noise
        psi = _apply_U_psi_jax(psi, evals, evecs, dt)
        noise_row = noise_seq[step_idx]
        psi = _apply_noise_jax(psi, N, noise_row, dt)

        # 4. Update Moving Average Buffer
        final_coh = _get_coherence(psi, N)
        new_sma_buf = jnp.concatenate([sma_buf[1:], jnp.array([final_coh])])

        new_carry = (psi, new_sma_buf, best_ci, max_gain, cooldown)
        return new_carry, None # No trajectory output, purely blind transport

    return step_fn

# ---------------------------------------------------------------------------
# Single trajectory: vmap-able
# ---------------------------------------------------------------------------

def _run_one_trajectory(psi0, psi_ideal_T, evals, evecs, cliffords_stack,
                        N, dt, n_steps, noise_seq, random_cis, ctrl_name, tau_thresh):
    
    step_fn = _make_scan_step(evals, evecs, cliffords_stack, N, dt,
                              noise_seq, random_cis, ctrl_name, tau_thresh)
    
    # Initialize adaptive carry
    initial_coh = _get_coherence(psi0, N)
    sma_buf0 = jnp.full((5,), initial_coh, dtype=jnp.float32)
    carry0 = (psi0, sma_buf0, jnp.int32(0), jnp.float32(-999.0), jnp.int32(0))
    
    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(step_fn, carry0, steps)
    
    psi_final = final_carry[0]
    best_ci_found = final_carry[2]
    
    # Frozen Blind Decoder Fidelity
    fidelity = jnp.abs(jnp.vdot(psi_ideal_T, psi_final))**2
    
    return fidelity, best_ci_found

# Vectorize over the initial state batch dimension
_run_batch = jax.jit(jax.vmap(
    _run_one_trajectory,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None),
), static_argnums=(5, 7, 10))

# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def get_random_bloch_state(N, theta, phi):
    dim = 2 ** N
    half = dim // 2
    psi = np.zeros(dim, dtype=np.complex64)
    psi[0] = np.cos(theta / 2.0)
    psi[half] = np.exp(1j * phi) * np.sin(theta / 2.0)
    return psi

def run_experiment_q(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 65)
    print("Program Q -- Blind Information Recovery (TPU-native)")
    print("=" * 65)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"tau={args.tau_thresh} (Instability Threshold)")
    
    dt = args.T_max / args.n_steps
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
                    
                    random_cis_np = noise_rng.integers(0, N_CLIFF, size=args.n_steps).astype(np.int32)
                    random_cis_j = jnp.array(random_cis_np)

                    # Generate Batch of Hidden Target States (Tier 2 - Full Bloch Sphere)
                    psi0_batch = []
                    psi_ideal_batch = []
                    
                    for _ in range(args.n_states):
                        theta = deg_rng.uniform(0, np.pi)
                        phi = deg_rng.uniform(0, 2 * np.pi)
                        
                        # Ideal Base State
                        psi0 = get_random_bloch_state(args.N, theta, phi)
                        # Compute Psi(T) under pure base Hamiltonian for the Blind Decoder
                        psi_ideal_e = evecs_np.conj().T @ psi0
                        psi_ideal_e = psi_ideal_e * np.exp(-1j * evals_np * args.T_max)
                        psi_ideal_T = evecs_np @ psi_ideal_e
                        
                        # Now degrade psi0 for the noisy transport
                        psi_noisy = psi0.copy()
                        
                        # Pre-transport scrambling/dephasing
                        if channel == "dephasing" and sigma > 0:
                            angles = deg_rng.normal(0, sigma, args.N)
                            phase = np.ones(2**args.N, dtype=np.complex64)
                            for q in range(args.N):
                                idx = np.arange(2**args.N)
                                bit = (idx >> (args.N - 1 - q)) & 1
                                ph = np.where(bit == 0, np.exp(1j * angles[q]/2), np.exp(-1j * angles[q]/2))
                                phase *= ph
                            psi_noisy = psi_noisy * phase
                            psi_noisy /= np.linalg.norm(psi_noisy)
                        elif channel == "scrambling" and sigma > 0:
                            if deg_rng.random() < sigma:
                                ci = int(deg_rng.integers(1, N_CLIFF))
                                site = int(deg_rng.integers(0, args.N))
                                psi_noisy = apply_clifford_psi(psi_noisy, args.N, ci, site)
                                
                        psi0_batch.append(psi_noisy)
                        psi_ideal_batch.append(psi_ideal_T)

                    psi0_batch_j = jnp.array(np.stack(psi0_batch, axis=0))
                    psi_ideal_batch_j = jnp.array(np.stack(psi_ideal_batch, axis=0))

                    for ctrl in args.controllers:
                        t_run = time.time()
                        fidelities, best_cis = _run_batch(
                            psi0_batch_j, psi_ideal_batch_j, evals, evecs, cliffords_stack,
                            args.N, jnp.float32(dt), args.n_steps, noise_seq_j, random_cis_j, ctrl,
                            jnp.float32(args.tau_thresh)
                        )
                        
                        fidelities = np.array(jax.device_get(fidelities))
                        elapsed_ms = (time.time() - t_run) * 1000
                        
                        for i in range(args.n_states):
                            rec = {
                                "fidelity": float(fidelities[i]),
                                "best_ci": int(best_cis[i]) if best_cis is not None else 0,
                                "model": model,
                                "controller": ctrl,
                                "channel": channel,
                                "sigma": sigma,
                                "B": b,
                            }
                            ctrl_records[ctrl].append(rec)
                            all_records.append(rec)

                        print(f"    [{ctrl:17s}] batch {b}: {args.n_states} states in {elapsed_ms:.0f}ms")

                # Metrics
                for ctrl in args.controllers:
                    recs = ctrl_records[ctrl]
                    mean_f = float(np.mean([r["fidelity"] for r in recs]))
                    summary[f"{key}/{ctrl}"] = {"fidelity": mean_f, "n": len(recs)}

                s_key = f"{key}/static"
                p_key = f"{key}/adaptive_sparse"
                if s_key in summary and p_key in summary:
                    d_f = summary[p_key]["fidelity"] - summary[s_key]["fidelity"]
                    passed = (d_f > 0.05)
                    print(f"  {'PASS' if passed else '    '} {key:35s} DFidelity={d_f:+.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_q_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--T-max", type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=30, dest="n_steps")
    p.add_argument("--tau-thresh", type=float, default=0.08, dest="tau_thresh")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["XXZ", "DisorderedXXZ_W3", "OAT"])
    p.add_argument("--controllers", nargs="+", default=["static", "periodic_sweep", "random_sparse", "adaptive_sparse"])
    p.add_argument("--channels", nargs="+", default=["none", "dephasing", "scrambling"])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--n-states", type=int, default=12, dest="n_states")
    p.add_argument("--out-dir", default="program_q_results", dest="out_dir")
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
        args.n_states = 4
        print("[SMOKE TEST] N=6, B=1")

    run_experiment_q(args)
