"""
Program S1 -- Predictive Echo Recovery (TPU-native)
=========================================================
Tests explicit recoverability of paired hidden states under scrambling.
Introduces a Probe stage for susceptibility estimation, and an Echo
recovery phase at the end to measure coherent reconstruction.
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
from functools import partial

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
    idx = jnp.arange(dim)
    vals = []
    # 1. Local X expectation values for all qubits
    for i in range(N):
        stride = 2 ** (N - 1 - i)
        psi_reshaped = psi.reshape(-1, 2 * stride)
        lo = psi_reshaped[:, :stride]
        hi = psi_reshaped[:, stride:]
        x_exp = 2.0 * jnp.real(jnp.sum(lo * jnp.conj(hi)))
        vals.append(x_exp)
    # 2. Local Z expectation values for all qubits
    for i in range(N):
        bit_mask = ((idx >> (N - 1 - i)) & 1).astype(jnp.float32)
        z_diag = 1.0 - 2.0 * bit_mask
        z_exp = jnp.real(jnp.sum(psi * jnp.conj(psi) * z_diag))
        vals.append(z_exp)
    return jnp.array(vals)

# ---------------------------------------------------------------------------
# Phase 1: Sparse Probe Tomography
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,))
def _estimate_probe_susceptibility(evals, evecs, N, dt):
    dim = 2 ** N
    psi_ref = jnp.zeros(dim, dtype=jnp.complex64)
    psi_ref = psi_ref.at[0].set(1.0)
    
    eps = np.pi / 8.0
    rx = jnp.array([[np.cos(eps/2), -1j*np.sin(eps/2)], [-1j*np.sin(eps/2), np.cos(eps/2)]], dtype=jnp.complex64)
    ry = jnp.array([[np.cos(eps/2), -np.sin(eps/2)], [np.sin(eps/2), np.cos(eps/2)]], dtype=jnp.complex64)
    rz = jnp.array([[np.exp(-1j*eps/2), 0], [0, np.exp(1j*eps/2)]], dtype=jnp.complex64)
    
    psi_I = _apply_U_psi_jax(psi_ref, evals, evecs, dt)
    O_I = _get_coherence(psi_I, N)
    
    S_vals = []
    for gate in [rx, ry, rz]:
        p = _apply_site_gate_jax(psi_ref, gate, N, jnp.int32(0))
        p_t = _apply_U_psi_jax(p, evals, evecs, dt)
        O_P = _get_coherence(p_t, N)
        S_vals.append(jnp.max(jnp.abs(O_P - O_I)))
        
    return jnp.max(jnp.array(S_vals))

# ---------------------------------------------------------------------------
# JAX Scan Body
# ---------------------------------------------------------------------------

def _make_scan_step(evals, evecs, rz_gate, cliffords_stack, N, dt,
                    noise_seq, scramble_seq, random_triggers, is_agnostic, tau_adapted):
    """
    carry = (psi, obs_hist, cooldown, n_interventions)
    obs_hist: [O_{t-2}, O_{t-1}, O_{t}]
    """
    @jax.jit
    def step_fn(carry, step_idx):
        psi, obs_hist, cooldown, n_interventions = carry
        
        # 1. Apply Dynamic Scrambling Channel step (if any)
        scramble_ci = scramble_seq[step_idx, 0]
        scramble_site = scramble_seq[step_idx, 1]
        
        gate_scr = cliffords_stack[scramble_ci]
        psi = jax.lax.cond(
            scramble_ci > 0,
            lambda p: _apply_site_gate_jax(p, gate_scr, N, scramble_site),
            lambda p: p,
            psi,
        )
        
        coh_before = _get_coherence(psi, N)
        
        # Predictability Breakdown Forecast (Max single-qubit coordinate error)
        pred = obs_hist[2] + (obs_hist[2] - obs_hist[1])
        err = jnp.max(jnp.abs(coh_before - pred))
        
        trigger = jnp.logical_and(err > tau_adapted, cooldown == 0)
        # Avoid triggering in the first 2 steps while buffer fills
        trigger = jnp.logical_and(trigger, step_idx > 2)

        site = jnp.int32(0) # Targeted intervention on Q0
        
        # 2. Controller Logic
        fire = jax.lax.cond(
            is_agnostic,
            lambda _: trigger,
            lambda _: jnp.bool_(False),
            None
        )
            
        # 3. Apply Constrained Intervention Gate (Rz(pi/8))
        psi = jax.lax.cond(
            fire,
            lambda p: _apply_site_gate_jax(p, rz_gate, N, site),
            lambda p: p,
            psi,
        )

        n_interventions = n_interventions + jnp.where(fire, 1, 0)
        cooldown = jnp.where(fire, 4, jnp.maximum(0, cooldown - 1))
            
        # 4. Evolution & Noise
        psi = _apply_U_psi_jax(psi, evals, evecs, dt)
        noise_row = noise_seq[step_idx]
        psi = _apply_noise_jax(psi, N, noise_row, dt)

        # 5. Update History Buffer
        coh_after = _get_coherence(psi, N)
        new_obs_hist = jnp.stack([obs_hist[1], obs_hist[2], coh_after], axis=0)

        new_carry = (psi, new_obs_hist, cooldown, n_interventions)
        return new_carry, None

    return step_fn

# ---------------------------------------------------------------------------
# Single trajectory: vmap-able
# ---------------------------------------------------------------------------

def _run_one_trajectory(psi0, psi_ideal_T, evals, evecs, rz_gate, cliffords_stack,
                        N, dt, n_steps, noise_seq, scramble_seq, random_triggers, 
                        is_agnostic, tau_adapted, apply_echo):
    
    step_fn = _make_scan_step(evals, evecs, rz_gate, cliffords_stack, N, dt,
                              noise_seq, scramble_seq, random_triggers, is_agnostic, tau_adapted)
    
    initial_coh = _get_coherence(psi0, N)
    obs_hist0 = jnp.stack([initial_coh, initial_coh, initial_coh], axis=0)
    carry0 = (psi0, obs_hist0, jnp.int32(0), jnp.int32(0))
    
    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(step_fn, carry0, steps)
    
    psi_final = final_carry[0]
    n_interv = final_carry[3]
    
    # ----------------------------------------------------
    # Phase 3: Echo Recovery (Inverse Ideal Evolution)
    # ----------------------------------------------------
    psi_echoed = jax.lax.cond(
        apply_echo,
        lambda p: _apply_U_psi_jax(p, evals, evecs, -dt * n_steps),
        lambda p: p,
        psi_final
    )
    
    # Endpoint Fidelity
    fidelity = jnp.abs(jnp.vdot(psi_ideal_T, psi_final))**2
    
    # Echo Recoverability Metrics (against original psi0)
    F_recover = jnp.abs(jnp.vdot(psi0, psi_echoed))**2
    
    return psi_final, psi_echoed, fidelity, F_recover, n_interv

_run_batch = jax.jit(jax.vmap(
    _run_one_trajectory,
    in_axes=(0, 0, None, None, None, None, None, None, None, 0, 0, 0, None, None, None),
), static_argnums=(6, 8, 12, 14))

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

def apply_perturbation(psi, N, epsilon, rng):
    ry = np.array([[np.cos(epsilon/2), -np.sin(epsilon/2)],
                   [np.sin(epsilon/2), np.cos(epsilon/2)]], dtype=np.complex64)
    return _apply_site_gate_static_site(psi, ry, N, 0)

def run_experiment_s1(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 65)
    print("Program S1 -- Predictive Echo Recovery (TPU)")
    print("=" * 65)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"tau={args.tau_thresh} alpha={args.probe_alpha}")
    
    dt = args.T_max / args.n_steps
    rng = np.random.default_rng(args.seed)

    # Constrained Rz(pi/8) gate
    theta_gate = np.pi / 8.0
    rz_np = np.array([[np.exp(-1j * theta_gate / 2), 0],
                      [0, np.exp(1j * theta_gate / 2)]], dtype=np.complex64)
    rz_stack = jnp.array(rz_np)

    # JAX Clifford stack
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
                
                # Phase 1: Estimate probe susceptibility once per Hamiltonian
                S_probe = float(jax.device_get(_estimate_probe_susceptibility(evals, evecs, args.N, jnp.float32(dt))))

                for b in range(args.B):
                    seed_b = args.seed + b * 1000
                    noise_params = sample_noise_params(rng)
                    deg_rng = np.random.default_rng(seed_b + 54321)
                    noise_rng = np.random.default_rng(seed_b)

                    # Generate Batch of Paired Hidden States
                    psi0_A_list = []
                    psi0_B_list = []
                    ideal_A_list = []
                    ideal_B_list = []
                    
                    for _ in range(args.n_pairs):
                        theta = deg_rng.uniform(0, np.pi)
                        phi = deg_rng.uniform(0, 2 * np.pi)
                        
                        psi_base = get_random_bloch_state(args.N, theta, phi)
                        psi_pert = apply_perturbation(psi_base, args.N, args.epsilon, deg_rng)
                        
                        phase = np.ones(2**args.N, dtype=np.complex64)
                        if channel == "dephasing" and sigma > 0:
                            angles = deg_rng.normal(0, sigma, args.N)
                            for q in range(args.N):
                                idx = np.arange(2**args.N)
                                bit = (idx >> (args.N - 1 - q)) & 1
                                ph = np.where(bit == 0, np.exp(1j * angles[q]/2), np.exp(-1j * angles[q]/2))
                                phase *= ph
                                
                        def apply_pre_noise(psi):
                            p = psi.copy()
                            if channel == "dephasing" and sigma > 0:
                                p = p * phase
                                p /= np.linalg.norm(p)
                            return p

                        psi0_A_list.append(apply_pre_noise(psi_base))
                        psi0_B_list.append(apply_pre_noise(psi_pert))

                        def get_ideal(psi):
                            p_e = evecs_np.conj().T @ psi
                            p_e = p_e * np.exp(-1j * evals_np * args.T_max)
                            return evecs_np @ p_e
                            
                        ideal_A_list.append(get_ideal(psi_base))
                        ideal_B_list.append(get_ideal(psi_pert))

                    psi0_batch = np.array(psi0_A_list + psi0_B_list)
                    ideal_batch = np.array(ideal_A_list + ideal_B_list)
                    
                    psi0_batch_j = jnp.array(psi0_batch)
                    ideal_batch_j = jnp.array(ideal_batch)

                    gamma = float(noise_params.get("gamma", 0.01))
                    kappa = float(noise_params.get("kappa", 0.01))
                    
                    base_noise_seq = np.zeros((args.n_pairs, args.n_steps, 8), dtype=np.float32)
                    base_scramble_seq = np.zeros((args.n_pairs, args.n_steps, 2), dtype=np.int32)
                    
                    for i in range(args.n_pairs):
                        base_noise_seq[i, :, 0] = noise_rng.normal(gamma, 0.005, args.n_steps).clip(0)
                        base_noise_seq[i, :, 1] = noise_rng.normal(kappa, 0.005, args.n_steps).clip(0)
                        
                        if channel == "scrambling" and sigma > 0:
                            for t in range(args.n_steps):
                                if noise_rng.random() < sigma * 0.15:
                                    ci = int(noise_rng.integers(1, N_CLIFF))
                                    site = int(noise_rng.integers(0, args.N))
                                    base_scramble_seq[i, t, 0] = ci
                                    base_scramble_seq[i, t, 1] = site
                    
                    matched_noise_seq = np.concatenate([base_noise_seq, base_noise_seq], axis=0)
                    matched_scramble_seq = np.concatenate([base_scramble_seq, base_scramble_seq], axis=0)
                    
                    noise_seq_j = jnp.array(matched_noise_seq)
                    scramble_seq_j = jnp.array(matched_scramble_seq)
                    
                    base_random_trig = noise_rng.random((args.n_pairs, args.n_steps)).astype(np.float32)
                    matched_random_trig = np.concatenate([base_random_trig, base_random_trig], axis=0)
                    random_trig_j = jnp.array(matched_random_trig)

                    D_FS_0 = []
                    for i in range(args.n_pairs):
                        overlap = np.abs(np.vdot(psi0_A_list[i], psi0_B_list[i]))
                        overlap = np.clip(overlap, 0.0, 1.0)
                        D_FS_0.append(np.arccos(overlap))

                    for ctrl in args.controllers:
                        t_run = time.time()
                        
                        is_agnostic = ("agnostic" in ctrl)
                        apply_echo = ("echo" in ctrl)
                        
                        # Phase 2: Probe Conditioning
                        if "probe" in ctrl:
                            tau_adapted = args.tau_thresh * (1.0 + args.probe_alpha * S_probe)
                        else:
                            tau_adapted = args.tau_thresh
                            
                        psi_final_batch, psi_echoed_batch, fidelities, F_recovers, n_intervs = _run_batch(
                            psi0_batch_j, ideal_batch_j, evals, evecs, rz_stack, cliffords_stack,
                            args.N, jnp.float32(dt), args.n_steps, noise_seq_j, scramble_seq_j, random_trig_j, 
                            is_agnostic, jnp.float32(tau_adapted), apply_echo
                        )
                        
                        psi_final_batch = np.array(jax.device_get(psi_final_batch))
                        psi_echoed_batch = np.array(jax.device_get(psi_echoed_batch))
                        fidelities = np.array(jax.device_get(fidelities))
                        F_recovers = np.array(jax.device_get(F_recovers))
                        n_intervs = np.array(jax.device_get(n_intervs))
                        elapsed_ms = (time.time() - t_run) * 1000
                        
                        batch_D_FS_T = []
                        batch_I_pres = []
                        
                        for i in range(args.n_pairs):
                            psiA = psi_final_batch[i]
                            psiB = psi_final_batch[i + args.n_pairs]
                            overlap = np.abs(np.vdot(psiA, psiB))
                            D_FS_T = np.arccos(np.clip(overlap, 0.0, 1.0))
                            
                            psiA_rec = psi_echoed_batch[i]
                            psiB_rec = psi_echoed_batch[i + args.n_pairs]
                            overlap_rec = np.abs(np.vdot(psiA_rec, psiB_rec))
                            D_FS_rec = np.arccos(np.clip(overlap_rec, 0.0, 1.0))
                            
                            d0 = D_FS_0[i]
                            I_pres = D_FS_T / d0 if d0 > 1e-6 else 0.0
                            I_pres_rec = D_FS_rec / d0 if d0 > 1e-6 else 0.0
                            
                            mean_fid = (fidelities[i] + fidelities[i + args.n_pairs]) / 2.0
                            mean_F_rec = (F_recovers[i] + F_recovers[i + args.n_pairs]) / 2.0
                            mean_n_int = (n_intervs[i] + n_intervs[i + args.n_pairs]) / 2.0
                            
                            rec = {
                                "D_FS_0": float(d0),
                                "D_FS_T": float(D_FS_T),
                                "D_FS_rec": float(D_FS_rec),
                                "I_preserve": float(I_pres),
                                "I_preserve_rec": float(I_pres_rec),
                                "fidelity": float(mean_fid),
                                "F_recover": float(mean_F_rec),
                                "n_interventions": float(mean_n_int),
                                "model": model,
                                "controller": ctrl,
                                "channel": channel,
                                "sigma": sigma,
                                "B": b,
                                "S_probe": float(S_probe),
                                "tau_adapted": float(tau_adapted),
                            }
                            ctrl_records[ctrl].append(rec)
                            all_records.append(rec)

                        print(f"    [{ctrl:22s}] batch {b}: {args.n_pairs} pairs in {elapsed_ms:.0f}ms")

                # Metrics
                s_key = f"{key}/static_echo"
                p_key = f"{key}/probe_agnostic_echo"
                
                for ctrl in args.controllers:
                    recs = ctrl_records[ctrl]
                    mean_f = float(np.mean([r["fidelity"] for r in recs]))
                    mean_Frec = float(np.mean([r["F_recover"] for r in recs]))
                    mean_I = float(np.mean([r["I_preserve"] for r in recs]))
                    mean_Irec = float(np.mean([r["I_preserve_rec"] for r in recs]))
                    mean_n = float(np.mean([r["n_interventions"] for r in recs]))
                    summary[f"{key}/{ctrl}"] = {
                        "fidelity": mean_f,
                        "F_recover": mean_Frec,
                        "I_preserve": mean_I,
                        "I_preserve_rec": mean_Irec,
                        "n_interv": mean_n,
                        "tau_adapted": float(recs[0]["tau_adapted"]) if recs else 0.0,
                        "n": len(recs)
                    }

                if s_key in summary and p_key in summary:
                    d_Frec = summary[p_key]["F_recover"] - summary[s_key]["F_recover"]
                    d_Irec = summary[p_key]["I_preserve_rec"] - summary[s_key]["I_preserve_rec"]
                    passed = (d_Frec > 0.0)
                    print(f"  {'PASS' if passed else '    '} {key:35s} D_Frec={d_Frec:+.3f} D_Irec={d_Irec:+.3f} n_int={summary[p_key]['n_interv']:.1f} tau={summary[p_key]['tau_adapted']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_s1_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--T-max", type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=30, dest="n_steps")
    p.add_argument("--tau-thresh", type=float, default=0.08, dest="tau_thresh")
    p.add_argument("--probe-alpha", type=float, default=1.0, dest="probe_alpha")
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["XXZ", "DisorderedXXZ_W3", "OAT"])
    p.add_argument("--controllers", nargs="+", default=["static", "static_echo", "agnostic_echo", "probe_agnostic_echo"])
    p.add_argument("--channels", nargs="+", default=["none", "dephasing", "scrambling"])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--n-pairs", type=int, default=12, dest="n_pairs")
    p.add_argument("--out-dir", default="program_s1_results", dest="out_dir")
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
        args.n_pairs = 4
        print("[SMOKE TEST] N=6, B=1")

    run_experiment_s1(args)
