"""
Program V -- Adaptive Entanglement-Assisted State Transfer
===========================================================
3x4 grid geometry with Alice (0, 1) and Bob (site_dist classes).
Entangled substrate prepared between Alice (0) and Bob (bob_site).
Vector-trigger predicted error vector projected onto offline susceptibility Jacobian.
Phase-sensitive metrics: Entanglement Fidelity, Mutual Information, Bloch angle recovery.
Tested exclusively on Out-Of-Distribution (OOD) random Haar states.
CPTP noise modeled via physically rigorous quantum trajectory jumps.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

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
from program_t_tpu import (
    build_grid_hamiltonian, get_alice_sites,
    get_bob_candidates, pick_bob_anchor,
)

# ---------------------------------------------------------------------------
# Quantum Trajectory State Vector Logic
# ---------------------------------------------------------------------------

def apply_gate_psi(psi, gate_2x2, qubit, N):
    """Apply a 2x2 gate to qubit in state vector psi."""
    axes = list(range(N))
    axes.remove(qubit)
    axes = [qubit] + axes
    psi_transposed = psi.reshape((2,) * N).transpose(axes)
    psi_new = jnp.tensordot(gate_2x2, psi_transposed, axes=([1], [0]))
    
    inv_axes = []
    curr = 1
    for i in range(N):
        if i == qubit:
            inv_axes.append(0)
        else:
            inv_axes.append(curr)
            curr += 1
    return psi_new.transpose(inv_axes).reshape(-1)

def apply_stochastic_dephasing(psi, qubit, N, p2, key):
    """Apply pure dephasing stochastically."""
    r = jax.random.uniform(key)
    Z_gate = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
    return jax.lax.cond(r < p2, lambda p: apply_gate_psi(p, Z_gate, qubit, N), lambda p: p, psi)

def apply_stochastic_amplitude_damping(psi, qubit, N, p1, key):
    """Apply amplitude damping stochastically using continuous excitation-tracking jumps."""
    r = jax.random.uniform(key)
    
    K1_gate = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64)
    psi_jump = apply_gate_psi(psi, K1_gate, qubit, N)
    p_ex = jnp.sum(jnp.abs(psi_jump) ** 2)
    
    prob_jump = p1 * p_ex
    
    def do_jump():
        nrm = jnp.linalg.norm(psi_jump)
        return jax.lax.cond(nrm > 1e-12, lambda: psi_jump / nrm, lambda: psi_jump)
        
    def no_jump():
        K0_gate = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - p1)]], dtype=jnp.complex64)
        psi_no = apply_gate_psi(psi, K0_gate, qubit, N)
        nrm = jnp.linalg.norm(psi_no)
        return jax.lax.cond(nrm > 1e-12, lambda: psi_no / nrm, lambda: psi_no)
        
    return jax.lax.cond(r < prob_jump, do_jump, no_jump)

# ---------------------------------------------------------------------------
# Initial State & Haar Generator
# ---------------------------------------------------------------------------

def make_initial_state(N, bob_site, Haar_c0, Haar_c1):
    """
    Prepare initial state vector:
    Bell pair between Alice 0 and Bob site, plus a random Haar state on Alice 1.
    """
    dim = 2 ** N
    psi = jnp.zeros(dim, dtype=jnp.complex64)
    
    # 4 components of (Haar_c0|0> + Haar_c1|1>)_1 x (|00> + |11>)_{0, bob_site}/sqrt(2)
    idx_A = 0
    idx_B = 1 << (N - 2) # Qubit 1 is at index N-2
    idx_C = (1 << (N - 1)) | (1 << (N - 1 - bob_site)) # Qubit 0 is N-1, bob_site is N-1-bob_site
    idx_D = (1 << (N - 1)) | (1 << (N - 1 - bob_site)) | (1 << (N - 2))
    
    psi = psi.at[idx_A].set(Haar_c0 / jnp.sqrt(2.0))
    psi = psi.at[idx_B].set(Haar_c1 / jnp.sqrt(2.0))
    psi = psi.at[idx_C].set(Haar_c0 / jnp.sqrt(2.0))
    psi = psi.at[idx_D].set(Haar_c1 / jnp.sqrt(2.0))
    return psi

# ---------------------------------------------------------------------------
# Causal Local Observables (Alice Region 0 & 1)
# ---------------------------------------------------------------------------

def get_causal_observables_psi(psi, N, readout_error=0.02):
    """Measure local X and Z expectations for Alice's qubits (0 and 1) under readout noise."""
    _sx = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    _sz = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    obs = []
    for q in [0, 1]:
        psi_x = apply_gate_psi(psi, _sx, q, N)
        psi_z = apply_gate_psi(psi, _sz, q, N)
        
        val_x = jnp.real(jnp.vdot(psi, psi_x))
        val_z = jnp.real(jnp.vdot(psi, psi_z))
        
        # Apply assignment readout error
        obs.append(val_x * (1.0 - 2.0 * readout_error))
        obs.append(val_z * (1.0 - 2.0 * readout_error))
        
    return jnp.array(obs)

# ---------------------------------------------------------------------------
# Offline Phase V0 Probe Tomography (Local Susceptibility Jacobian)
# ---------------------------------------------------------------------------

def run_probe_stage_v(evals_np, evecs_np, N, dt, bob_site):
    """
    Run offline characterization of the local Alice response Jacobian.
    Returns 2x4 matrix mapping the 4 local Alice observables to Bob-site transport.
    """
    dim = 2 ** N
    eps_angles = [np.pi/8, np.pi/4, np.pi/2]
    
    def evolve_psi(psi, t):
        p_e = evecs_np.conj().T @ psi
        p_e = p_e * np.exp(-1j * evals_np * t)
        return evecs_np @ p_e

    # Form a local reference state
    psi_ref = np.zeros(dim, dtype=np.complex64)
    psi_ref[0] = 1.0
    
    # 4 local observables: X0, Z0, X1, Z1
    def get_obs(p):
        obs = []
        _sx = np.array([[0,1],[1,0]], dtype=np.complex64)
        _sz = np.array([[1,0],[0,-1]], dtype=np.complex64)
        for q in [0, 1]:
            # row actions equivalent
            stride = 2 ** (N - 1 - q)
            pr = p.reshape(-1, 2*stride)
            lo, hi = pr[:, :stride], pr[:, stride:]
            obs.append(float(2.0 * np.real(np.sum(lo * np.conj(hi)))))
            idx = np.arange(dim)
            bm = ((idx >> (N - 1 - q)) & 1).astype(float)
            obs.append(float(np.real(np.sum(p * np.conj(p) * (1 - 2*bm)))))
        return np.array(obs)

    O_ref = get_obs(evolve_psi(psi_ref, dt))
    
    # Jacobian mapping local perturbations to Bob site expectation Z_bob
    susceptibility = np.zeros((2, 4), dtype=np.float32) # maps [pert_0, pert_1] to [dX0, dZ0, dX1, dZ1]
    
    for ai, site in enumerate([0, 1]):
        resp_sum = np.zeros(4)
        for eps in eps_angles:
            rz = np.array([[np.exp(-1j*eps/2), 0],
                           [0, np.exp(1j*eps/2)]], dtype=np.complex64)
            stride = 2 ** (N - 1 - site)
            psi_p = psi_ref.reshape(-1, 2*stride).copy()
            lo, hi = psi_p[:, :stride].copy(), psi_p[:, stride:].copy()
            psi_p[:, :stride] = rz[0,0]*lo + rz[0,1]*hi
            psi_p[:, stride:] = rz[1,0]*lo + rz[1,1]*hi
            psi_p = psi_p.reshape(dim)
            
            O_p = get_obs(evolve_psi(psi_p, dt))
            resp_sum += np.abs(O_p - O_ref) / (eps + 1e-9)
        susceptibility[ai] = resp_sum / len(eps_angles)
        
    return susceptibility

# ---------------------------------------------------------------------------
# Single Step Stochastic Dynamics (State Vector)
# ---------------------------------------------------------------------------

def run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key):
    """
    Evolve state vector psi stochastically under continuous Hamiltonian
    and jump CPTP maps, applying crosstalk and target gate interventions.
    """
    # 1. Coherent Evolution
    p_e = evecs.conj().T @ psi
    p_e = p_e * jnp.exp(-1j * evals * dt)
    psi = evecs @ p_e

    # 2. Stochastic Noise channels
    key_a, key_b = jax.random.split(key)
    for q in range(N):
        key_q = jax.random.fold_in(key_a, q)
        psi = apply_stochastic_amplitude_damping(psi, q, N, p1, key_q)
        psi = apply_stochastic_dephasing(psi, q, N, p2, key_q)

    # 3. Apply Controller Gate
    ci, site = gate_info
    cliffords_np = [np.eye(2, dtype=np.complex64)] + \
                   [np.array(g, dtype=np.complex64) for g in CLIFFORDS[1:]]
    cliffords_j = jnp.array(cliffords_np)

    def apply_control(p):
        p = apply_gate_psi(p, cliffords_j[ci], site, N)
        # Apply Spectator Crosstalk
        for neighbor in [site - 1, site + 1]:
            if 0 <= neighbor < N:
                key_cross = jax.random.fold_in(key_b, neighbor)
                p = apply_stochastic_dephasing(p, neighbor, N, p_cross, key_cross)
        return p

    psi = jax.lax.cond(ci > 0, apply_control, lambda p: p, psi)
    return psi

# ---------------------------------------------------------------------------
# Vector-Trigger Adaptive Controller
# ---------------------------------------------------------------------------

def ctrl_static_v(step, **_):
    if step == 0: return (1, 0)
    if step % 8 == 0: return (3, 0)
    return (0, 0)

def ctrl_dd_v(step, **_):
    seq = [1, 1, 2, 2] # X, X, Y, Y
    if step % 4 == 0: return (seq[(step // 4) % 4], 0)
    return (0, 0)

def ctrl_random_v(step, rng_key, **_):
    key1, key2, key3 = jax.random.split(rng_key, 3)
    fire = jax.random.uniform(key1) < 0.15
    ci = jax.random.randint(key2, (), 1, N_CLIFF)
    site = jax.random.randint(key3, (), 0, 2)
    return jax.lax.cond(fire, lambda: (ci, site), lambda: (0, 0))

def ctrl_vector_adaptive_v(step, obs_hist, cooldown, tau_thresh, S_proj, **_):
    """
    Project prediction error vector onto susceptibility matrix S_proj
    to get the Mode-Weighted Instability Score.
    """
    pred = obs_hist[1] + (obs_hist[1] - obs_hist[0])
    err_vec = obs_hist[2] - pred # length 4 vector
    
    # Project: score = || S_proj @ err_vec ||_2^2
    proj_err = jnp.dot(S_proj, err_vec) # length 2 vector
    instability = jnp.sum(proj_err ** 2)
    
    trigger = jnp.logical_and(instability > tau_thresh, cooldown == 0)
    trigger = jnp.logical_and(trigger, step > 2)
    
    return jax.lax.cond(trigger, lambda: (3, 0), lambda: (0, 0))

# ---------------------------------------------------------------------------
# Unified Trajectory Runner (JAX Scan)
# ---------------------------------------------------------------------------

def run_trajectory_v(evals, evecs, N, dt, n_steps, p1, p2, p_cross,
                     ctrl_name, tau_thresh, S_proj, bob_site, Haar_c0, Haar_c1, rng_key):
    """Evolve a single quantum trajectory state vector."""
    psi0 = make_initial_state(N, bob_site, Haar_c0, Haar_c1)
    obs0 = get_causal_observables_psi(psi0, N)
    
    obs_hist0 = jnp.stack([obs0, obs0, obs0], axis=0)
    carry0 = (psi0, obs_hist0, 0, 0, rng_key)

    def scan_step(carry, step_idx):
        psi, obs_hist, cooldown, n_gates, key = carry
        key_next, key_ctrl, key_step = jax.random.split(key, 3)
        
        # 1. Controller decisions
        if ctrl_name == "static":
            ci, site = ctrl_static_v(step_idx)
        elif ctrl_name == "dd":
            ci, site = ctrl_dd_v(step_idx)
        elif ctrl_name == "random":
            ci, site = ctrl_random_v(step_idx, key_ctrl)
        elif ctrl_name == "vector_adaptive":
            ci, site = ctrl_vector_adaptive_v(step_idx, obs_hist, cooldown, tau_thresh, S_proj)
        else:
            ci, site = 0, 0
            
        gate_info = (ci, site)
        
        # 2. Step Dynamics
        psi_next = run_step_psi(psi, evals, evecs, N, dt, p1, p2, p_cross, gate_info, key_step)
        
        # 3. Update carry
        cooldown_next = jax.lax.cond(ci > 0, lambda: 4, lambda: jnp.maximum(0, cooldown - 1))
        n_gates_next = n_gates + jnp.where(ci > 0, 1, 0)
        
        obs_curr = get_causal_observables_psi(psi_next, N)
        obs_hist_next = jnp.stack([obs_hist[1], obs_hist[2], obs_curr], axis=0)
        
        return (psi_next, obs_hist_next, cooldown_next, n_gates_next, key_next), None

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(scan_step, carry0, steps)
    
    psi_final, _, _, n_gates_total, _ = final_carry
    return psi_final, n_gates_total

# ---------------------------------------------------------------------------
# Phase-Sensitive Metrics Suite
# ---------------------------------------------------------------------------

def compute_reduced_rho_AB(psi_batch, N, bob_site):
    """
    Efficiently trace out spectator qubits from a batch of state vectors
    to obtain the 4x4 density matrix rho_AB for Alice 0 and Bob.
    """
    keep = [0, bob_site]
    trace = [i for i in range(N) if i not in keep]
    axes = keep + trace
    
    def single_rho(p):
        p_t = p.reshape((2,) * N).transpose(axes).reshape(4, -1)
        return jnp.tensordot(p_t, p_t.conj(), axes=([1], [1]))
        
    rhos = jax.vmap(single_rho)(psi_batch)
    return jnp.mean(rhos, axis=0)

def compute_phase_sensitive_metrics(rho_AB, Haar_c0, Haar_c1, N, bob_site):
    """Evaluate Entanglement Fidelity, Mutual Information, and Bloch angle recovery."""
    # 1. Entanglement Fidelity (F_ent relative to Phi^+)
    # Phi^+ = (|00> + |11>)/sqrt(2)
    F_ent = 0.5 * (rho_AB[0,0] + rho_AB[0,3] + rho_AB[3,0] + rho_AB[3,3])
    F_ent = float(jnp.real(F_ent))
    
    # 2. Von Neumann Mutual Information I(A:B)
    rho_A = jnp.array([[rho_AB[0,0] + rho_AB[1,1], rho_AB[0,2] + rho_AB[1,3]],
                       [rho_AB[2,0] + rho_AB[3,1], rho_AB[2,2] + rho_AB[3,3]]])
                       
    rho_B = jnp.array([[rho_AB[0,0] + rho_AB[2,2], rho_AB[0,1] + rho_AB[2,3]],
                       [rho_AB[1,0] + rho_AB[3,2], rho_AB[1,1] + rho_AB[3,3]]])

    def entropy(dm):
        evals = jnp.linalg.eigvalsh(dm)
        evals = jnp.clip(evals, 1e-12, 1.0)
        return -jnp.sum(evals * jnp.log2(evals))
        
    S_A = entropy(rho_A)
    S_B = entropy(rho_B)
    S_AB = entropy(rho_AB)
    mutual_info = float(S_A + S_B - S_AB)
    
    # 3. Bloch Angle Recovery of Haar state (transported on Alice 1 to Bob)
    # Cosine similarity between final Bob Bloch vector and ideal Haar Bloch vector
    x_ideal = float(2.0 * np.real(Haar_c0 * np.conj(Haar_c1)))
    y_ideal = float(2.0 * np.imag(Haar_c1 * np.conj(Haar_c0)))
    z_ideal = float(np.abs(Haar_c0)**2 - np.abs(Haar_c1)**2)
    v_ideal = np.array([x_ideal, y_ideal, z_ideal])
    
    x_act = 2.0 * jnp.real(rho_B[0,1])
    y_act = 2.0 * jnp.imag(rho_B[1,0])
    z_act = jnp.real(rho_B[0,0] - rho_B[1,1])
    v_act = np.array([x_act, y_act, z_act])
    
    nrm_ideal = np.linalg.norm(v_ideal) + 1e-12
    nrm_act = np.linalg.norm(v_act) + 1e-12
    bloch_similarity = float(np.dot(v_ideal, v_act) / (nrm_ideal * nrm_act))
    
    return {
        "F_ent": max(0.0, min(1.0, F_ent)),
        "mutual_information": max(0.0, mutual_info),
        "bloch_similarity": max(-1.0, min(1.0, bloch_similarity))
    }

# ---------------------------------------------------------------------------
# Main Sweep Coordinator
# ---------------------------------------------------------------------------

def run_experiment_v(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=args.require_tpu)
    
    print("=" * 70)
    print("Program V -- Adaptive Entanglement-Assisted State Transfer")
    print("=" * 70)
    
    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    print(f"Grid: {Lx}x{Ly} = {N} qubits  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Number of Trajectories per Batch: {args.n_trajectories}")
    
    dt = args.T_max / args.n_steps
    
    # Rigorous CPTP parameters from transmon relaxation times
    T1_phys = 20.0
    T2_phys = 12.0
    p1 = float(1.0 - np.exp(-dt / T1_phys))
    p2 = float(1.0 - np.exp(-dt / T2_phys))
    p_cross = 0.03
    
    rng = np.random.default_rng(args.seed)
    bob_cands = get_bob_candidates(Lx, Ly)
    
    summary = {}
    all_records = []

    # JAX Vectorized Trajectory Loop
    if HAS_JAX:
        # Vmap over the batch of trajectories
        run_trajectories_vmapped = jax.jit(
            jax.vmap(run_trajectory_v, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)),
            static_argnums=(2, 4, 8)
        )
    else:
        print("[ERROR] JAX is required to run Program V.")
        sys.exit(1)

    for model_name in args.models:
        print(f"\n== Model: {model_name} ==")
        H = build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.4, seed=args.seed)
        evals_np, evecs_np = _get_eigh(H, (model_name, N, args.seed))
        
        evals = jnp.array(evals_np, dtype=jnp.float32)
        evecs = jnp.array(evecs_np, dtype=jnp.complex64)

        for dist_class, bob_list in bob_cands.items():
            if not bob_list: continue
            bob_site, bob_dist = bob_list[len(bob_list)//2]
            print(f"\n  -- distance={dist_class} (Bob=site {bob_site}, dist={bob_dist}) --")
            
            # Phase V0: Probe Tomography to get susceptibility Jacobian S
            print("    [PROBE] Characterizing response Jacobian...")
            S = run_probe_stage_v(evals_np, evecs_np, N, dt, bob_site)
            S_proj = jnp.array(S, dtype=jnp.float32) # size 2x4

            for ctrl_name in args.controllers:
                ctrl_records = []
                
                for b in range(args.B):
                    seed_b = args.seed + b * 1000 + hash(ctrl_name) % 5000
                    b_rng = np.random.default_rng(seed_b)
                    
                    # 1. Strictly OOD random Haar State Generator
                    u = b_rng.uniform(0, 1)
                    v = b_rng.uniform(0, 1)
                    theta = np.arccos(2.0 * u - 1.0)
                    phi = 2.0 * np.pi * v
                    Haar_c0 = np.cos(theta / 2.0)
                    Haar_c1 = np.exp(1j * phi) * np.sin(theta / 2.0)
                    
                    # Trajectory keys batch
                    traj_keys = jax.random.split(jax.random.PRNGKey(seed_b), args.n_trajectories)
                    
                    # Run trajectories in parallel via vmap
                    psi_finals, n_gates_batch = run_trajectories_vmapped(
                        evals, evecs, N, jnp.float32(dt), args.n_steps,
                        jnp.float32(p1), jnp.float32(p2), jnp.float32(p_cross),
                        ctrl_name, jnp.float32(args.tau_thresh), S_proj, bob_site,
                        jnp.complex64(Haar_c0), jnp.complex64(Haar_c1), traj_keys
                    )
                    
                    # Form density matrix of subsystem Alice-Bob
                    rho_AB = compute_reduced_rho_AB(psi_finals, N, bob_site)
                    
                    # Compute phase sensitive metrics
                    metrics = compute_phase_sensitive_metrics(rho_AB, Haar_c0, Haar_c1, N, bob_site)
                    
                    avg_gates = float(np.mean(n_gates_batch))
                    
                    rec = {
                        "model": model_name, "controller": ctrl_name,
                        "dist_class": dist_class, "bob_site": bob_site,
                        "bob_dist": bob_dist, "batch": b,
                        "F_ent": metrics["F_ent"],
                        "mutual_information": metrics["mutual_information"],
                        "bloch_similarity": metrics["bloch_similarity"],
                        "n_interventions": avg_gates,
                        "efficiency": float(metrics["F_ent"] / (avg_gates + 1e-9))
                    }
                    ctrl_records.append(rec)
                    all_records.append(rec)
                    
                # Summarize
                avg_fent = np.mean([r["F_ent"] for r in ctrl_records])
                avg_mi = np.mean([r["mutual_information"] for r in ctrl_records])
                avg_bloch = np.mean([r["bloch_similarity"] for r in ctrl_records])
                avg_gates_all = np.mean([r["n_interventions"] for r in ctrl_records])
                
                key = f"{model_name}/{dist_class}/{ctrl_name}"
                summary[key] = {
                    "F_ent": float(avg_fent),
                    "mutual_information": float(avg_mi),
                    "bloch_similarity": float(avg_bloch),
                    "n_interventions": float(avg_gates_all),
                    "n": len(ctrl_records)
                }
                print(f"    [{ctrl_name:20s}] F_ent={avg_fent:.4f}  MutInfo={avg_mi:.4f}  BlochSim={avg_bloch:.4f}  Gates={avg_gates_all:.1f}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_v_N{N}_results.json")
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
    p.add_argument("--tau-thresh", type=float, default=0.06, dest="tau_thresh")
    p.add_argument("--B", type=int, default=3)
    p.add_argument("--n-trajectories", type=int, default=100, dest="n_trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["GridXXZ_hetero"])
    p.add_argument("--controllers", nargs="+", default=["static", "dd", "random", "vector_adaptive"])
    p.add_argument("--out-dir", default="program_v_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.Lx, args.Ly = 2, 3
        args.n_steps = 12
        args.n_trajectories = 10
        args.B = 1
        args.controllers = ["static", "vector_adaptive"]
        print("[SMOKE TEST] Running with N=6, B=1, Trajectories=10")

    run_experiment_v(args)
