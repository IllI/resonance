"""
Program U -- Hardware-Realistic Sparse Predictive Recovery
===========================================================
CPTP Kraus channels (amplitude damping T1, pure dephasing T2).
Correlated spectator crosstalk noise and assignment readout errors.
Simultaneous reporting of:
  - Endpoint Fidelity
  - Trace Distance (Distinguishability)
  - Echo Recovery (Reversibility)
  - Intervention Efficiency
Models: XXZ, W3, OAT, Ising null.
Controllers: static, DD, random_sparse, sparse_predictive.
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

# Ensure basic utilities can be loaded from parent files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import (
    _get_eigh, CLIFFORDS, N_CLIFF,
    select_backend, kron_site,
)

# ---------------------------------------------------------------------------
# Hamiltonian Builders
# ---------------------------------------------------------------------------

def build_xxz(N, W=0.0, J=1.0, Delta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    disorder = rng.uniform(-W, W, N) if W > 0 else np.zeros(N)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex64)
    for s in range(dim):
        for i in range(N):
            bit = (s >> i) & 1
            H[s, s] += disorder[i] * (1 - 2 * bit)
        for i in range(N - 1):
            bi  = (s >> i) & 1
            bi1 = (s >> (i + 1)) & 1
            H[s, s] += J * Delta * (1 - 2 * bi) * (1 - 2 * bi1)
            if bi != bi1:
                s2 = s ^ (1 << i) ^ (1 << (i + 1))
                H[s, s2] += J
    return H

def build_oat(N, chi=1.0):
    _sz = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    Jz = np.zeros((2**N, 2**N), dtype=np.complex64)
    for i in range(N):
        Jz += kron_site(_sz / 2.0, i, N)
    return chi * (Jz @ Jz)

def build_ising_null(N, J=1.0, hx=0.5, hz=0.0):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex64)
    for s in range(dim):
        for i in range(N - 1):
            bi  = (s >> i) & 1
            bi1 = (s >> (i + 1)) & 1
            H[s, s] -= J * (1 - 2 * bi) * (1 - 2 * bi1)
        for i in range(N):
            bi = (s >> i) & 1
            H[s, s] -= hz * (1 - 2 * bi)
            s2 = s ^ (1 << i)
            H[s, s2] -= hx
    return H

MODEL_BUILDERS_U = {
    "XXZ":              lambda N, seed: build_xxz(N, W=0.0, seed=seed),
    "W3":               lambda N, seed: build_xxz(N, W=3.0, seed=seed),
    "OAT":              lambda N, seed: build_oat(N, chi=1.0 / N),
    "Ising":            lambda N, seed: build_ising_null(N, J=1.0, hx=1.0, hz=0.0),
}

# ---------------------------------------------------------------------------
# CPTP Kraus Channel Application in JAX
# ---------------------------------------------------------------------------

def apply_gate_rho(rho, gate_2x2, qubit, N):
    """Apply a 2x2 operator to the row-like index 'qubit' of a density matrix rho."""
    # rho shape is (2,) * (2 * N)
    axes = list(range(2 * N))
    axes.remove(qubit)
    axes = [qubit] + axes
    rho_transposed = rho.transpose(axes)
    rho_new = jnp.tensordot(gate_2x2, rho_transposed, axes=([1], [0]))
    
    # Transpose back
    inv_axes = []
    curr = 1
    for i in range(2 * N):
        if i == qubit:
            inv_axes.append(0)
        else:
            inv_axes.append(curr)
            curr += 1
    return rho_new.transpose(inv_axes)

def apply_kraus_rho(rho, K_list, qubit, N):
    """Apply a CPTP map defined by Kraus operators K_list to qubit."""
    out = jnp.zeros_like(rho)
    for K in K_list:
        # Left action on row
        r_temp = apply_gate_rho(rho, K, qubit, N)
        # Right action (corresponds to K^* left action on column index)
        r_temp = apply_gate_rho(r_temp, K.conj(), qubit + N, N)
        out = out + r_temp
    return out

# ---------------------------------------------------------------------------
# Physical Noise Channels Generator
# ---------------------------------------------------------------------------

def get_amplitude_damping_kraus(p):
    K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - p)]], dtype=jnp.complex64)
    K1 = jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=jnp.complex64)
    return [K0, K1]

def get_dephasing_kraus(p):
    K0 = jnp.sqrt(1.0 - p) * jnp.eye(2, dtype=jnp.complex64)
    K1 = jnp.sqrt(p) * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
    return [K0, K1]

# ---------------------------------------------------------------------------
# Trace Distance Metric
# ---------------------------------------------------------------------------

def compute_trace_distance(rho_A, rho_B, N):
    """Compute the trace distance D(rho_A, rho_B) = 1/2 Tr|rho_A - rho_B|."""
    dim = 2 ** N
    diff = (rho_A - rho_B).reshape(dim, dim)
    # Trace distance is 1/2 sum of absolute eigenvalues of the hermitian difference
    evals = jnp.linalg.eigvalsh(diff)
    return 0.5 * jnp.sum(jnp.abs(evals))

# ---------------------------------------------------------------------------
# Single Step Dynamics & Noise
# ---------------------------------------------------------------------------

def run_step_rho(rho, evals, evecs, N, dt, p1, p2, p_cross, gate_info):
    """
    Evolve rho by dt under H, apply T1 & T2 noise, and spectator crosstalk.
    gate_info is (ci, site) indicating which gate is applied.
    """
    dim = 2 ** N
    # 1. Coherent Evolution: e^{-i H dt} rho e^{i H dt}
    rho_flat = rho.reshape(dim, dim)
    rho_e = evecs.conj().T @ rho_flat @ evecs
    # exp(-1j * (E_i - E_j) * dt)
    evals_diff = evals[:, None] - evals[None, :]
    rho_e = rho_e * jnp.exp(-1j * evals_diff * dt)
    rho_flat = evecs @ rho_e @ evecs.conj().T
    rho = rho_flat.reshape((2,) * (2 * N))

    # 2. Apply T1 (Amplitude Damping) and T2 (Dephasing) Kraus channels
    K_t1 = get_amplitude_damping_kraus(p1)
    K_t2 = get_dephasing_kraus(p2)
    for q in range(N):
        rho = apply_kraus_rho(rho, K_t1, q, N)
        rho = apply_kraus_rho(rho, K_t2, q, N)

    # 3. Apply Controller / DD Gate
    ci, site = gate_info
    cliffords_np = [np.eye(2, dtype=np.complex64)] + \
                   [np.array(g, dtype=np.complex64) for g in CLIFFORDS[1:]]
    cliffords_j = jnp.array(cliffords_np)

    def apply_control(r):
        r = apply_gate_rho(r, cliffords_j[ci], site, N)
        r = apply_gate_rho(r, cliffords_j[ci].conj(), site + N, N)
        # Apply Spectator Crosstalk to neighbors
        K_cross = get_dephasing_kraus(p_cross)
        if site > 0:
            r = apply_kraus_rho(r, K_cross, site - 1, N)
        if site < N - 1:
            r = apply_kraus_rho(r, K_cross, site + 1, N)
        return r

    rho = jax.lax.cond(ci > 0, apply_control, lambda r: r, rho)
    return rho

# ---------------------------------------------------------------------------
# Local Expectation Values (Causal Observers with Readout Noise)
# ---------------------------------------------------------------------------

def get_causal_observables(rho, N, readout_error=0.02):
    """
    Compute local X and Z expectations for Alice's qubits (0 and 1).
    Applies hardware assignment readout error.
    """
    dim = 2 ** N
    rho_flat = rho.reshape(dim, dim)
    
    _sx = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    _sz = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    obs = []
    # Row and Column matrices for projection trace
    for q in [0, 1]:
        X_op = kron_site(_sx, q, N)
        Z_op = kron_site(_sz, q, N)
        
        val_x = jnp.real(jnp.trace(X_op @ rho_flat))
        val_z = jnp.real(jnp.trace(Z_op @ rho_flat))
        
        # Apply assignment readout error: val -> val * (1 - 2*epsilon)
        obs.append(val_x * (1.0 - 2.0 * readout_error))
        obs.append(val_z * (1.0 - 2.0 * readout_error))
        
    return jnp.array(obs)

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def ctrl_static_u(step, **_):
    if step == 0: return (1, 0)
    if step % 8 == 0: return (3, 0)
    return (0, 0)

def ctrl_dd_u(step, **_):
    seq = [1, 1, 2, 2] # X, X, Y, Y
    if step % 4 == 0: return (seq[(step // 4) % 4], 0)
    return (0, 0)

def ctrl_random_sparse_u(step, rng_key, **_):
    # Generates a random decision with ~15% gate fire probability
    key1, key2, key3 = jax.random.split(rng_key, 3)
    fire = jax.random.uniform(key1) < 0.15
    ci = jax.random.randint(key2, (), 1, N_CLIFF)
    site = jax.random.randint(key3, (), 0, 2) # Alice's sites (0 or 1)
    return jax.lax.cond(fire, lambda: (ci, site), lambda: (0, 0))

def ctrl_sparse_predictive_u(step, obs_hist, cooldown, tau, **_):
    """
    Predictive Sparse controller using local Alice-region X/Z expectations.
    """
    pred = obs_hist[1] + (obs_hist[1] - obs_hist[0])
    err = jnp.max(jnp.abs(obs_hist[2] - pred))
    
    trigger = jnp.logical_and(err > tau, cooldown == 0)
    trigger = jnp.logical_and(trigger, step > 2)
    
    return jax.lax.cond(trigger, lambda: (3, 0), lambda: (0, 0))

# ---------------------------------------------------------------------------
# Unified Trajectory Evolution in JAX
# ---------------------------------------------------------------------------

def run_trajectory_u(rho0, evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                     ctrl_name, tau, rng_key):
    """
    Runs dynamic CPTP density matrix evolution.
    """
    dim = 2 ** N
    initial_obs = get_causal_observables(rho0, N)
    
    # Pack initial carry: (rho, obs_hist, cooldown, n_gates, key)
    # obs_hist is a buffer of size 3: [O_{t-2}, O_{t-1}, O_{t}]
    obs_hist0 = jnp.stack([initial_obs, initial_obs, initial_obs], axis=0)
    carry0 = (rho0, obs_hist0, 0, 0, rng_key)

    def scan_step(carry, step_idx):
        rho, obs_hist, cooldown, n_gates, key = carry
        
        # Split JAX key for stochastic controller choices
        key_next, key_ctrl = jax.random.split(key)
        
        # 1. Controller Decisions
        if ctrl_name == "static":
            ci, site = ctrl_static_u(step_idx)
        elif ctrl_name == "dd":
            ci, site = ctrl_dd_u(step_idx)
        elif ctrl_name == "random_sparse":
            ci, site = ctrl_random_sparse_u(step_idx, key_ctrl)
        elif ctrl_name == "sparse_predictive":
            ci, site = ctrl_sparse_predictive_u(step_idx, obs_hist, cooldown, tau)
        else:
            ci, site = 0, 0
            
        gate_info = (ci, site)
        
        # 2. Run Step
        rho_next = run_step_rho(rho, evals, evecs, N, dt, p1, p2, p_cross, gate_info)
        
        # 3. Update carry
        cooldown_next = jax.lax.cond(ci > 0, lambda: 4, lambda: jnp.maximum(0, cooldown - 1))
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        
        obs_curr = get_causal_observables(rho_next, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr], axis=0)
        
        return (rho_next, obs_hist_next, cooldown_next, n_gates_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, steps)
    
    rho_final, _, _, n_gates_total, _ = final_carry
    return rho_final, n_gates_total

# ---------------------------------------------------------------------------
# Main Sweep Coordinator
# ---------------------------------------------------------------------------

def run_experiment_u(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 70)
    print("Program U -- Hardware-Realistic Sparse Predictive Recovery")
    print("=" * 70)
    print(f"Backend: {backend_info.get('backend', 'numpy')}  Devices: {backend_info.get('devices')}")
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}  B={args.B}")
    
    dt = args.T_max / args.n_steps
    
    # CPTP Probability conversions from standard relaxation times
    # We define standard physical constants matching superconducting hardware
    T1_phys = 15.0  # T1 relaxation time (steps scale)
    T2_phys = 10.0  # T2 pure dephasing time
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    p2 = float(1.0 - np.exp(-dt / T2_phys))
    p_cross = 0.04 # spectator crosstalk rate
    
    print(f"Physical Noise Parameters: T1={T1_phys} (p1={p1:.4f}), T2={T2_phys} (p2={p2:.4f}), Crosstalk={p_cross:.2f}")

    rng = np.random.default_rng(args.seed)
    jax_key = jax.random.PRNGKey(args.seed) if HAS_JAX else None
    
    summary = {}
    all_records = []

    # Compile the trajectory loop for swift deployment
    if HAS_JAX:
        run_trajectory_jit = jax.jit(run_trajectory_u, static_argnums=(4, 9))
    else:
        print("[ERROR] JAX is required to execute density matrix sweeps in Program U.")
        sys.exit(1)

    for model_name in args.models:
        print(f"\n== Model: {model_name} ==")
        H = MODEL_BUILDERS_U[model_name](args.N, args.seed)
        evals_np, evecs_np = _get_eigh(H, (model_name, args.N, args.seed))
        
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for ctrl_name in args.controllers:
            ctrl_records = []
            
            for b in range(args.B):
                seed_b = args.seed + b * 1000
                b_rng = np.random.default_rng(seed_b)
                b_key = jax.random.PRNGKey(seed_b)
                
                # Sample a pair of close initial states A and B to measure Distinguishability
                theta = float(b_rng.uniform(0, np.pi))
                phi1 = float(b_rng.uniform(0, 2*np.pi))
                phi2 = phi1 + 0.15 # close phase perturbation
                
                psi_A_1q = np.array([np.cos(theta/2), np.exp(1j*phi1)*np.sin(theta/2)], dtype=np.complex64)
                psi_B_1q = np.array([np.cos(theta/2), np.exp(1j*phi2)*np.sin(theta/2)], dtype=np.complex64)
                
                dim = 2 ** args.N
                
                def make_rho(psi_1q):
                    psi = np.zeros(dim, dtype=np.complex64)
                    psi[0] = psi_1q[0]
                    psi[dim // 2] = psi_1q[1]
                    psi /= np.linalg.norm(psi)
                    return np.outer(psi, psi.conj()).reshape((2,) * (2 * args.N))
                
                rho_A0 = jnp.array(make_rho(psi_A_1q))
                rho_B0 = jnp.array(make_rho(psi_B_1q))
                
                # Evolve Trajectory A
                rho_Af, n_gates = run_trajectory_jit(
                    rho_A0, evals, evecs, args.N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    ctrl_name, jnp.float32(args.tau_thresh), b_key
                )
                
                # Evolve Trajectory B
                rho_Bf, _ = run_trajectory_jit(
                    rho_B0, evals, evecs, args.N, jnp.float32(dt), args.n_steps,
                    jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                    ctrl_name, jnp.float32(args.tau_thresh), b_key
                )
                
                # --- Metrics Computations ---
                # 1. Distinguishability (Trace Distance)
                D_initial = float(compute_trace_distance(rho_A0, rho_B0, args.N))
                D_final = float(compute_trace_distance(rho_Af, rho_Bf, args.N))
                preserve_index = D_final / (D_initial + 1e-12)
                
                # 2. Endpoint Fidelity: relative to noiseless coherent evolution of Trajectory A
                # Noiseless final density matrix
                evals_diff = evals_np[:, None] - evals_np[None, :]
                rho_A0_flat = make_rho(psi_A_1q).reshape(dim, dim)
                rho_ideal_e = evecs_np.conj().T @ rho_A0_flat @ evecs_np
                rho_ideal_e = rho_ideal_e * np.exp(-1j * evals_diff * args.T_max)
                rho_ideal = evecs_np @ rho_ideal_e @ evecs_np.conj().T
                
                rho_Af_flat = np.array(rho_Af).reshape(dim, dim)
                fidelity = float(np.real(np.trace(rho_ideal @ rho_Af_flat)))
                
                # 3. Echo Recovery: Apply ideal inverse coherent evolution to rho_Af
                rho_echoed_e = evecs_np.conj().T @ rho_Af_flat @ evecs_np
                rho_echoed_e = rho_echoed_e * np.exp(1j * evals_diff * args.T_max) # positive phase for inverse
                rho_echoed = evecs_np @ rho_echoed_e @ evecs_np.conj().T
                
                F_recover = float(np.real(np.trace(rho_A0_flat @ rho_echoed)))
                
                rec = {
                    "model": model_name,
                    "controller": ctrl_name,
                    "batch": b,
                    "endpoint_fidelity": fidelity,
                    "trace_distance_initial": D_initial,
                    "trace_distance_final": D_final,
                    "preserve_index": preserve_index,
                    "F_recover": F_recover,
                    "n_interventions": int(n_gates),
                }
                ctrl_records.append(rec)
                all_records.append(rec)
                
            # Summary Metrics for this model/controller pair
            fid_avg = np.mean([r["endpoint_fidelity"] for r in ctrl_records])
            pi_avg = np.mean([r["preserve_index"] for r in ctrl_records])
            f_rec_avg = np.mean([r["F_recover"] for r in ctrl_records])
            n_gates_avg = np.mean([r["n_interventions"] for r in ctrl_records])
            
            key = f"{model_name}/{ctrl_name}"
            summary[key] = {
                "endpoint_fidelity": float(fid_avg),
                "preserve_index": float(pi_avg),
                "F_recover": float(f_rec_avg),
                "n_interventions": float(n_gates_avg),
                "n": len(ctrl_records)
            }
            
            print(f"    [{ctrl_name:20s}] Average: Fid={fid_avg:.4f}  PresIndex={pi_avg:.4f}  F_rec={f_rec_avg:.4f}  Gates={n_gates_avg:.1f}")

    # Save Results
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_u_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1)
        }, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--T-max", type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=30, dest="n_steps")
    p.add_argument("--tau-thresh", type=float, default=0.08, dest="tau_thresh")
    p.add_argument("--B", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["XXZ", "W3", "OAT", "Ising"])
    p.add_argument("--controllers", nargs="+", default=["static", "dd", "random_sparse", "sparse_predictive"])
    p.add_argument("--out-dir", default="program_u_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.N = 4
        args.n_steps = 10
        args.B = 1
        args.models = ["XXZ", "Ising"]
        args.controllers = ["static", "sparse_predictive"]
        print("[SMOKE TEST] Running with N=4, B=1")

    run_experiment_u(args)
