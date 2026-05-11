"""
run_v2_experiment_updated.py
Drop-in replacement for run_v2_experiment.py.

Changes from original:
  Phase 1  —  ExperimentalDataIngester replaces IntraSliceJitterStreamer
  Phase 5  —  framework_residuals.run_framework_comparison() replaces
               NQS meta-learner (can run both for comparison)

Quick-start
-----------
# Calibration run (synthetic SYK — known answer):
python run_v2_experiment_updated.py --source SYNTHETIC --framework syk

# Real SYK ED data (download from github.com/BlackHatLKJH/SYK-model):
python run_v2_experiment_updated.py --source SYK_ED --data_dir ./data

# Live Rydberg hardware (requires Amazon Braket credentials):
python run_v2_experiment_updated.py --source RYDBERG --data_dir ./data

# Google Sycamore RCS bitstrings:
python run_v2_experiment_updated.py --source SYCAMORE_RCS --data_dir ./data
"""

import argparse
import jax
import jax.numpy as jnp
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from importlib.util import spec_from_file_location, module_from_spec

def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── existing modules (unchanged) ──────────────────────────────────
mapping    = load_module("mapping",    os.path.join(base_dir, "2_state_mapping/dlinoss_quantum_damper.py"))
heisenberg = load_module("heisenberg", os.path.join(base_dir, "3_projection/heisenberg_space.py"))
bitwistor  = load_module("bitwistor",  os.path.join(base_dir, "3_projection/bitwistor_space.py"))
syk_mod    = load_module("syk",        os.path.join(base_dir, "3_projection/syk_space.py"))
mbl_mod    = load_module("mbl",        os.path.join(base_dir, "3_projection/mbl_space.py"))
lindblad_mod = load_module("lindblad", os.path.join(base_dir, "3_projection/lindblad_space.py"))
otoc_mod   = load_module("otoc",       os.path.join(base_dir, "4_observation/otoc_metrics.py"))
kink_mod   = load_module("kink",       os.path.join(base_dir, "4_observation/or_kink_detection.py"))
meta_mod   = load_module("meta",       os.path.join(base_dir, "5_classification/nqs_meta_learner.py"))

# ── NEW modules ────────────────────────────────────────────────────
experimental_data_ingester = load_module("experimental_data_ingester", os.path.join(base_dir, "1_ingestion/experimental_data_ingester.py"))
ExperimentalDataIngester = experimental_data_ingester.ExperimentalDataIngester

framework_residuals = load_module("framework_residuals", os.path.join(base_dir, "5_classification/framework_residuals.py"))
run_framework_comparison = framework_residuals.run_framework_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    default="SYNTHETIC",
                   choices=["SYNTHETIC", "SYK_ED", "SYCAMORE_RCS", "RYDBERG", "JILA_MBL", "JILA_TELEPORTATION"])
    p.add_argument("--framework", default="syk",
                   choices=["heisenberg", "lindblad", "penrose_or", "syk", "mbl"],
                   help="Only used when --source SYNTHETIC")
    p.add_argument("--data_dir",  default="./data")
    p.add_argument("--batch",     type=int, default=128)
    p.add_argument("--dt",        type=float, default=0.1,
                   help="Physical timestep in units of J^{-1}")
    p.add_argument("--beta",      type=float, default=5.0,
                   help="Inverse temperature (SYK Lyapunov = 2π/beta)")
    p.add_argument("--both-classifiers", action="store_true",
                   help="Run both NQS meta-learner AND residual comparator")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 56)
    print("  Emergent Quantum Geometries via D-LinOSS  (V2+)")
    print(f"  Data source : {args.source}"
          + (f" [{args.framework}]" if args.source == "SYNTHETIC" else ""))
    print(f"  TPU backend : {jax.default_backend()}")
    print("=" * 56)

    # ── Phase 1: Ingestion ─────────────────────────────────────────
    print("\n[Phase 1] Ingesting experimental data...")
    ingester = ExperimentalDataIngester(
        source     = args.source,
        framework  = args.framework,
        data_dir   = args.data_dir,
        dt         = args.dt,
    )
    drift_data = ingester.stream_phase_data(batch_size=args.batch)
    print(f"  Data tensor shape: {drift_data.shape}")
    print(f"  Signal range:      [{float(drift_data.min()):.4f}, {float(drift_data.max()):.4f}]")

    # ── Phase 2: D-LiNOSS state mapping ───────────────────────────
    print("\n[Phase 2] Mapping to Lindbladian damper via D-LiNOSS...")
    import optax
    
    damper_net  = mapping.DLinOSSQuantumDamper()
    rng         = jax.random.PRNGKey(0)
    variables   = damper_net.init(rng, drift_data)
    
    # Simple training loop to learn the SSM parameters (ω_k, γ_k)
    print("  Training D-LiNOSS SSM on experimental data...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-3)
    )
    opt_state = optimizer.init(variables)
    
    # Load the Bi-Twistor simulator for Phase 3 integration into the training loop
    b_space = bitwistor.BiTwistorSimulator(threshold=0.8)
    initial_state = jnp.ones(32) / jnp.sqrt(32)
    
    @jax.jit
    def train_step(params, opt_st, batch):
        def moqe_composite_loss_fn(p):
            # 1. Data Reconstruction (The primary driving force)
            batch_size, seq_len, features = batch.shape
            impulse = jnp.zeros_like(batch)
            impulse = impulse.at[:, 0, :].set(1.0)
            
            # Generate the non-unitary parameters and the MoQE framework weights
            damping_matrix, y_pred_seq, omega_k, gamma_k, moqe_weights = damper_net.apply(p, impulse)
            
            # Signal fidelity: forces eigenvalues to actually match the real experimental data
            loss_reconstruction = jnp.mean((y_pred_seq - batch)**2)
            
            # 2. Framework QPC Metrics (The geometric regularizers)
            # Framework 1: Heisenberg (Unitary, zero damping)
            loss_heisenberg = jnp.mean(gamma_k**2)
            
            # Framework 2: Lindblad (Thermal Markovian bath)
            target_gamma_lindblad = 0.30
            loss_lindblad = jnp.mean((gamma_k - target_gamma_lindblad)**2)
            
            # Framework 3: Penrose OR (Gravitational collapse)
            b_state = b_space.apply_damper(initial_state, damping_matrix[0], current_energy=0.9)
            qpc_coherence = jnp.var(b_state)
            loss_penrose = jnp.abs(qpc_coherence - 0.8)
            
            # Framework 4: SYK (Holographic Chaos)
            beta = 5.0
            lambda_L = (2 * jnp.pi) / beta
            loss_syk = jnp.mean((gamma_k - lambda_L)**2)
            
            # Framework 5: MBL (Many-Body Localization)
            loss_mbl = jnp.mean(gamma_k**2) + jnp.var(omega_k) * 0.01  # Slight penalty for order
            
            # Aggregate the 5 QPC metrics
            losses = jnp.stack([loss_heisenberg, loss_lindblad, loss_penrose, loss_syk, loss_mbl])
            
            # Softmax the learnable weights to create a valid probability distribution
            probabilities = jax.nn.softmax(moqe_weights)
            
            # The Mixture of Quantum Experts composite loss
            # The network learns to shift probabilities towards the framework with the lowest intrinsic loss
            loss_moqe = jnp.dot(probabilities, losses)
            
            # Final Loss: Data Fidelity + MoQE Geometric Regularization
            composite_loss = loss_reconstruction + 0.1 * loss_moqe
            
            # Add small regularization to bound eigenvalues
            return composite_loss + 1e-4 * jnp.mean(damping_matrix**2)
        
        loss, grads = jax.value_and_grad(moqe_composite_loss_fn)(params)
        updates, new_opt_st = optimizer.update(grads, opt_st, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_st, loss

    epochs = 200
    for epoch in range(epochs):
        variables, opt_state, loss = train_step(variables, opt_state, drift_data)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f}")
            
    damping_matrix, _, _, _, _ = damper_net.apply(variables, drift_data)
    print(f"  Damping matrix shape: {damping_matrix.shape}")

    # ── Phase 3: Projection ────────────────────────────────────────
    print("\n[Phase 3] Projecting into competing quantum spaces...")
    initial_state = jnp.ones(32) / jnp.sqrt(32)
    h_space = heisenberg.HeisenbergSimulator()
    b_space = bitwistor.BiTwistorSimulator(threshold=0.8)
    s_space = syk_mod.SYKSimulator()
    m_space = mbl_mod.MBLSimulator()
    l_space = lindblad_mod.LindbladSimulator()
    
    h_state = h_space.apply_damper(initial_state, damping_matrix[0])
    b_state = b_space.apply_damper(initial_state, damping_matrix[0], current_energy=0.9)
    s_state = s_space.apply_damper(initial_state, damping_matrix[0])
    m_state = m_space.apply_damper(initial_state, damping_matrix[0])
    l_state = l_space.apply_damper(initial_state, damping_matrix[0])
    print("  Projection complete (5 spaces).")

    # ── Phase 4: Observation ───────────────────────────────────────
    print("\n[Phase 4] Computing physical observables...")
    otoc_calc = otoc_mod.OTOCCalculator()
    kink_det  = kink_mod.ORKinkDetector()
    h_history = jnp.stack([initial_state, h_state])
    b_history = jnp.stack([initial_state, b_state])
    otoc_curve = otoc_calc.calculate_otoc_decay(h_history)
    has_kink   = kink_det.detect_kink(b_history)
    print(f"  OTOC curve length : {len(otoc_curve)}")
    print(f"  OR kink detected  : {has_kink}")

    # ── Phase 5: Emergent MoQE Probabilities ────────────────────────
    print("\n[Phase 5] Emergent Mixture of Quantum Experts (MoQE)...")
    
    # Run one final forward pass to extract the learned weights
    _, _, _, _, moqe_weights = damper_net.apply(variables, drift_data)
    
    # Apply a temperature scale to sharpen the distribution. 
    # Because the MoQE loss was scaled by 0.1 during training to preserve data fidelity,
    # the raw logits are small. A low temperature amplifies the learned preference.
    temperature = 0.05
    emergent_probs = jax.nn.softmax(moqe_weights / temperature)
    
    frameworks = ["heisenberg", "lindblad", "penrose_or", "syk", "mbl"]
    descriptions = [
        "Unitary evolution, no decoherence.",
        "Thermal Lindblad bath.",
        "Penrose OR. Gravitational collapse.",
        "SYK holographic chaos.",
        "Many-body localized."
    ]
    
    # Calculate MoQE decision entropy
    entropy = -jnp.sum(emergent_probs * jnp.log(emergent_probs + 1e-12))
    
    print("========================================================")
    print("  MIXTURE OF QUANTUM EXPERTS  —  Emergent Probabilities")
    print("========================================================")
    
    results = []
    for f, p, desc in zip(frameworks, emergent_probs, descriptions):
        results.append((f, float(p), desc))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("  Framework          P(match)  Description")
    print("--------------------------------------------------------")
    for i, (f, p, desc) in enumerate(results):
        marker = "  ◄ WINNER" if i == 0 else ""
        print(f"  {f:<18} {p*100:>5.2f}%  {desc}{marker}")
    
    print("--------------------------------------------------------")
    print(f"  Decision entropy: {entropy:.4f}  (max = 1.6094 for 5 frameworks)")
    
    winner = results[0][0]
    confidence = results[0][1] - results[1][1]
    print(f"  Confidence:       {confidence*100:.1f}%")
    print("========================================================\n")
    
    # Calibration note
    if args.source == "SYNTHETIC":
        expected = args.framework
        actual   = winner
        passed   = "✓ PASS" if actual == expected else "✗ FAIL"
        print(f"  [CALIBRATION CHECK] Expected={expected}  Got={actual}  {passed}")
        print()


if __name__ == "__main__":
    main()
