import jax
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from importlib.util import spec_from_file_location, module_from_spec

def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the local modules manually since they don't have __init__.py files yet
ingestion = load_module("ingestion", os.path.join(base_dir, "1_ingestion/tpu_jitter_stream.py"))
mapping = load_module("mapping", os.path.join(base_dir, "2_state_mapping/dlinoss_quantum_damper.py"))
heisenberg = load_module("heisenberg", os.path.join(base_dir, "3_projection/heisenberg_space.py"))
bitwistor = load_module("bitwistor", os.path.join(base_dir, "3_projection/bitwistor_space.py"))
otoc = load_module("otoc", os.path.join(base_dir, "4_observation/otoc_metrics.py"))
kink = load_module("kink", os.path.join(base_dir, "4_observation/or_kink_detection.py"))
meta = load_module("meta", os.path.join(base_dir, "5_classification/nqs_meta_learner.py"))

def main():
    print("==================================================")
    print("Emergent Quantum Geometries via D-LinOSS (V2)")
    print("Architecture: Single-TPU Intra-Slice Drift (v6e-8)")
    print("==================================================")
    
    # 1. Ingestion
    print("\n[Phase 1] Initializing Intra-Slice Ingestion...")
    streamer = ingestion.IntraSliceJitterStreamer()
    drift_data = streamer.stream_phase_data(batch_size=128)
    print(f"Captured drift topology: {drift_data.shape}")
    
    # 2. State Mapping
    print("\n[Phase 2] Mapping stochasticity to Lindbladian Damper...")
    damper_net = mapping.DLinOSSQuantumDamper()
    # Dummy variables initialization for Flax
    rng = jax.random.PRNGKey(0)
    variables = damper_net.init(rng, drift_data)
    damping_matrix = damper_net.apply(variables, drift_data)
    print(f"Generated damping matrix: {damping_matrix.shape}")
    
    # 3. Projection
    print("\n[Phase 3] Projecting into competing quantum spaces...")
    h_space = heisenberg.HeisenbergSimulator()
    b_space = bitwistor.BiTwistorSimulator(threshold=0.8)
    
    # Initial state (vacuum / all-zero)
    initial_state = jax.numpy.ones(32) / jax.numpy.sqrt(32)
    
    # We apply the damper (taking the first batch item for simplicity)
    h_state = h_space.apply_damper(initial_state, damping_matrix[0])
    b_state = b_space.apply_damper(initial_state, damping_matrix[0], current_energy=0.9)
    print("Projection complete.")
    
    # 4. Observation
    print("\n[Phase 4] Calculating physical observables...")
    otoc_calc = otoc.OTOCCalculator()
    kink_det = kink.ORKinkDetector()
    
    h_history = jax.numpy.stack([initial_state, h_state])
    b_history = jax.numpy.stack([initial_state, b_state])
    
    otoc_curve = otoc_calc.calculate_otoc_decay(h_history)
    has_kink = kink_det.detect_kink(b_history)
    print(f"OTOC Curve length: {len(otoc_curve)}")
    print(f"Penrose Kink detected: {has_kink}")
    
    # 5. Classification
    print("\n[Phase 5] NQS Meta-Learner Classification...")
    meta_net = meta.NQSMetaLearner()
    meta_vars = meta_net.init(rng, otoc_curve, jax.numpy.array([has_kink]))
    
    probabilities = meta_net.apply(meta_vars, otoc_curve, jax.numpy.array([has_kink]))
    entropy = meta.calculate_entropy(probabilities)
    
    print("\n================== RESULTS =======================")
    print(f"Probability (Heisenberg Unitary): {probabilities[0]:.4f}")
    print(f"Probability (Bi-Twistor OR Collapse): {probabilities[1]:.4f}")
    print(f"Decision Entropy: {entropy:.4f}")
    
    if probabilities[0] > probabilities[1]:
        print("CONCLUSION: Macroscopic data acts as standard thermal bath (Scrambling wins)")
    else:
        print("CONCLUSION: Macroscopic data exhibits projective collapse (Penrose wins)")
    print("==================================================")

if __name__ == "__main__":
    main()
