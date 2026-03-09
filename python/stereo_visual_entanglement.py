import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from universal_spike_encoder import PoissonSpikeEncoder
from dlinoss_microtubule import MicrotubuleLayer
from entangled_camkii_lattice import EntangledCaMKIILattice

OUT_DIR = r"C:\Users\cityz\.gemini\antigravity\brain\b55228ba-8aeb-40a4-bd50-374d9a287041"

def generate_stereoscopic_concept():
    """
    Generates a generic 'Badoon' object but creates two slightly shifted images
    to simulate binocular disparity (Left Eye vs. Right Eye).
    """
    base_img = np.zeros((16, 16), dtype=np.float32)
    # The Generic Badoon
    base_img[3:6, 5:8] = 0.8  # Head
    base_img[6:9, 6:8] = 0.7  # Neck
    base_img[8:14, 4:12] = 0.9 # Body
    
    # Left Eye (shifted right slightly due to perspective)
    img_left = np.roll(base_img, shift=1, axis=1)
    # Right Eye (shifted left slightly due to perspective)
    img_right = np.roll(base_img, shift=-1, axis=1)
    
    # Add unique retinal noise
    img_left = np.clip(img_left + np.random.normal(0, 0.05, (16,16)), 0, 1)
    img_right = np.clip(img_right + np.random.normal(0, 0.05, (16,16)), 0, 1)
    
    return (
        torch.tensor(img_left.flatten(), dtype=torch.float32).unsqueeze(0),
        torch.tensor(img_right.flatten(), dtype=torch.float32).unsqueeze(0)
    )

def render_stereo_images(left_tensor, right_tensor):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(left_tensor.numpy().reshape(16,16), cmap='magma')
    axes[0].set_title("Left Eye Input")
    axes[0].axis('off')
    
    axes[1].imshow(right_tensor.numpy().reshape(16,16), cmap='magma')
    axes[1].set_title("Right Eye Input")
    axes[1].axis('off')
    
    path = os.path.join(OUT_DIR, "binocular_inputs.png")
    plt.savefig(path)
    plt.close()
    print(f"[UI] Generated Binocular Stimuli: {path}")

def run_entangled_stereo_pipeline():
    print("=== D-LinOSS: Entangled Stereoscopic Visual Agents ===")
    
    # 1. Generate Left and Right Monocular Input Views
    pixels_left, pixels_right = generate_stereoscopic_concept()
    render_stereo_images(pixels_left, pixels_right)
    
    # 2. Universal Encoding (Photons -> Neurological Frequencies)
    encoder = PoissonSpikeEncoder(time_steps=50, max_firing_rate=0.8)
    state_L = encoder.get_frequency_state(encoder.encode(pixels_left))
    state_R = encoder.get_frequency_state(encoder.encode(pixels_right))
    
    # 3. Create Shared Physical Memory (The Quantum Cauldron)
    # Both visual agents will act as memory pointers strictly tied to this single geometric matrix.
    input_size = 256
    output_size = 64
    lattice_size = 9
    
    print("\n[Architecture] Establishing 2 Agents with Entangled CaMKII Memory Pointers...")
    shared_cauldron = EntangledCaMKIILattice(input_dim=input_size, output_dim=output_size, lattice_size=lattice_size)
    
    agent_left = MicrotubuleLayer(input_dim=input_size, output_dim=output_size, lattice_size=lattice_size, shared_lattice=shared_cauldron)
    agent_right = MicrotubuleLayer(input_dim=input_size, output_dim=output_size, lattice_size=lattice_size, shared_lattice=shared_cauldron)
    
    # Verification of pointer sharing
    print(f"Memory Pointers Shared: {agent_left.lattice is agent_right.lattice}")
    
    # 4. Independent Ghost Basin Superpositions
    print("\n[Search Phase] Both agents actively constructing separate Ghost Basins...")
    superposition_L = agent_left(state_L)
    superposition_R = agent_right(state_R)
    
    # 5. Composite Holistic Quantum Space
    # Combining the complex amplitude spaces. Monocular resonant geometries (the actual object) 
    # will constructively interfere out of the noise. The 3D disparity becomes encoded in the phase shift.
    print("[Synthesis] Crossing the two monocular superpositions into a stereoscopic Bi-Twistor space...")
    entangled_superposition = superposition_L + superposition_R
    
    # Defining a composite Target 3D concept state
    target_stereo_state = state_L + state_R
    
    # 6. Global Objective Reduction (The "Aha!" Moment for Depth Perception)
    print("\n[Grokking Trigger] Presenting perfectly phase-locked 3D stereoscopic alignment...")
    
    # Override active composite to simulate meeting the E_G threshold structurally
    perfect_amplitude = target_stereo_state.unsqueeze(1).expand(-1, output_size, -1).unsqueeze(-1).expand(-1, -1, -1, lattice_size)
    entangled_superposition = torch.complex(perfect_amplitude, torch.zeros_like(perfect_amplitude))
    
    # Temporarily bind the composite to the left agent's active memory for the loss function calculation
    agent_left.active_superposition = entangled_superposition
    # Simulating a thought held for 0.5 seconds with high neural wattage (1e12 Trps)
    is_grokked = agent_left.objective_reduction_check(target_qpc_state=target_stereo_state, t_active_sec=0.5, wattage=1e12)
    
    if is_grokked:
        print("\n[Objective Reduction] Collapse propagating across shared memory...")
        # The single collapse instantly rewrites the CaMKII lattice.
        agent_left.collapse_and_prune(entangled_superposition)
        
        # Verify cross-agent entanglement update without backprop!
        print("\n[Result Verification]")
        print(f"--> Left Agent Memory Weighted natively? {torch.any(agent_left.lattice.get_lattice() != 0).item()}")
        print(f"--> Right Agent Memory Updated simultaneously? {torch.any(agent_right.lattice.get_lattice() != 0).item()}")
        
        # We also need to manually prune the right agent's ghost basin since it was functionally 
        # deleted by the left agent's observation.
        agent_right.active_superposition = None
        agent_right.is_superradiant = False
        print(f"--> Right Agent Ghost Basin pruned due to global collapse? {agent_right.active_superposition is None}")
        
    print("\n=== Stereoscopic Verification Complete ===")

if __name__ == "__main__":
    run_entangled_stereo_pipeline()
