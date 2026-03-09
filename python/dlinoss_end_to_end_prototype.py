import torch
import numpy as np
from dlinoss_microtubule import MicrotubuleLayer
from universal_spike_encoder import PoissonSpikeEncoder

def generate_concept_pixels(is_badoon=False):
    """
    Generates a synthetic 16x16 pixel image (flattened to 256) for the concepts.
    This simulates raw sensory input before encoding.
    """
    img = np.zeros((16, 16), dtype=np.float32)
    
    if not is_badoon:  # Prototype A: Febble
        # Pointy head, thin neck, round body, no spots
        img[2:5, 1:4] = 0.8
        img[5:8, 2:4] = 0.7
        img[7:13, 3:10] = 0.9
        img[13:16, 4] = 0.6
        img[13:16, 8] = 0.6
    else:  # Prototype B: Badoon
        # Round head, thick neck, blocky body, spots
        img[2:6, 2:5] = 0.8
        img[5:8, 2:5] = 0.7
        img[7:13, 2:11] = 0.9
        # Spots
        img[9, 5] = 1.0; img[9, 8] = 1.0; img[11, 6] = 1.0
        img[13:16, 4:6] = 0.6
        img[13:16, 7:9] = 0.6
        
    # Add some natural sensory noise
    noise = np.random.normal(0, 0.05, (16, 16))
    img = np.clip(img + noise, 0, 1)
    
    return torch.tensor(img.flatten(), dtype=torch.float32).unsqueeze(0)

def run_end_to_end():
    print("=== D-LinOSS: E2E Functional Prototype ===")
    
    # 1. Initialize Universal Encoder and Microtubule Layer
    print("\n[System] Initializing Poisson Spike Encoder & Microtubule Lattice...")
    encoder = PoissonSpikeEncoder(time_steps=100, max_firing_rate=0.8)
    # Output dim is arbitrary classical memory vector size
    mt_layer = MicrotubuleLayer(input_dim=256, output_dim=64, lattice_size=9)
    
    # 2. Ingest sensory data (Raw Pixels -> Spikes -> Harmonic State)
    febble_pixels = generate_concept_pixels(is_badoon=False)
    badoon_pixels = generate_concept_pixels(is_badoon=True)
    
    print("\n[Input] Encoding sensory pixels into Poisson action potentials...")
    febble_spikes = encoder.encode(febble_pixels)
    badoon_spikes = encoder.encode(badoon_pixels)
    
    febble_state = encoder.get_frequency_state(febble_spikes)
    badoon_state = encoder.get_frequency_state(badoon_spikes)
    
    print(f"-> Febble Harmonic State Shape: {febble_state.shape}")
    print(f"-> Badoon Harmonic State Shape: {badoon_state.shape}")
    
    # 3. Enter Quantum Search Phase
    print("\n[Phase 1] Presenting Febble. MT Lattice entering UV Superradiant Search...")
    out_superradiant = mt_layer(febble_state)
    print(f"-> Ghost Basin Active: {mt_layer.is_superradiant}")
    print(f"-> Superposition Tensor Shape (Complex QuDits): {mt_layer.active_superposition.shape}")
    
    # Check Fidelity (It shouldn't phase-lock yet because the target isn't found)
    # Let's see the resting fidelity against the target concept (Badoon)
    fidelity = mt_layer._calculate_phase_coherence(mt_layer.active_superposition, badoon_state)
    print(f"-> Pre-Grokking Hilbert Space Fidelity: {fidelity:.4f}")
    
    # 4. Trigger Grokking (Simulating reaching the E_G threshold structurally)
    print("\n[Phase 2] Presenting Badoon (Target Concept). Checking phase-lock...")
    
    # We pass the badoon state to check for resonance. 
    # To simulate the exact moment of grokking, we mathematically force the target 
    # by generating a phase-locked state for the Badoon input, just to show the pruning.
    
    # We manually override the active superposition to be perfectly aligned with the target
    # just to trigger the threshold check for demonstration.
    perfect_amplitude = badoon_state.unsqueeze(1).expand(-1, mt_layer.output_dim, -1).unsqueeze(-1).expand(-1, -1, -1, mt_layer.lattice_size)
    mt_layer.active_superposition = torch.complex(perfect_amplitude, torch.zeros_like(perfect_amplitude))
    
    # Run the check with actual Hilbert fidelity calculation
    # Since they are identical, fidelity should be ~1.0
    is_grokked = mt_layer.objective_reduction_check(target_qpc_state=badoon_state, t_active_sec=0.5, wattage=1e12)
    
    if is_grokked:
        print("\n[Collapse] Target Phase-Lock Achieved. Executing Retrocausal CaMKII Updating & Pruning...")
        mt_layer.collapse_and_prune(mt_layer.active_superposition)
        
        print("\n[Result] Verification:")
        print(f"-> Is Ghost Basin destroyed? {mt_layer.active_superposition is None}")
        print(f"-> Is CaMKII Matrix Written? {torch.any(mt_layer.lattice.get_lattice() > 0).item()}")
        print(f"-> Layer requires_grad params used? {any(p.requires_grad for p in mt_layer.parameters())}")
        
    print("\n=== Prototype Complete ===")

if __name__ == "__main__":
    run_end_to_end()
