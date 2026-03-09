import torch
import torch.nn as nn
import numpy as np
import math

from entangled_camkii_lattice import EntangledCaMKIILattice

class MicrotubuleLayer(nn.Module):
    """
    A D-LinOSS network layer modeling a Microtubule Hexagonal Lattice.
    This layer replaces generic Euclidean scalar weights with complex-valued quDit states
    representing the tubulin dimer lattice and CaMKII phosphorylation encoding.
    
    It operates in two phases:
    1. UV Superradiant Search: Expands input states into high-entropy superpositions.
    2. Bi-Twistor Collapse & Prune: Upon phase-lock, retrocausally aligns the lattice
       weights (phosphorylation states) and prunes all non-resonant quantum pathways.
    """
    def __init__(self, input_dim, output_dim, lattice_size=9, qed_cavity_q=1e4, shared_lattice=None):
        super(MicrotubuleLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # The fundamental computing unit is not a qubit, but a higher-dimensional quDit 
        # based on the hexagonal tubulin neighborhood patch (e.g., 9-tubulin B-lattice).
        self.lattice_size = lattice_size
        
        # ordered-water QED cavity quality factor (secures coherence time)
        self.qed_cavity_q = qed_cavity_q 
        
        # The CaMKII memory structure
        # If part of a multimodal agent array, the lattice is passed by reference.
        # Otherwise, the agent generates an independent classical memory structure.
        if shared_lattice is not None:
            self.lattice = shared_lattice
        else:
            self.lattice = EntangledCaMKIILattice(input_dim, output_dim, lattice_size)
            
        # The Ghost Basin Superposition (The Active Quantum Search State)
        # Complex-valued tensor representing the UV Superradiant Tryptophan exciton.
        # Shape: (batch_size, output_dim, input_dim, lattice_size)
        self.active_superposition = None
        
        # State tracker
        self.is_superradiant = False

    def forward(self, x):
        """
        The forward pass ingests the entire semantic state (not autoregressive tokens).
        If the layer is not grokking, it enters the UV Superradiant search phase.
        
        x: Input semantic state. Shape (batch_size, input_dim)
        """
        batch_size = x.shape[0]
        
        if not self.is_superradiant:
            # --- ENTER UV SUPERRADIANCE (SEARCH PHASE) ---
            # Convert classical input state into a complex quantum superposition across the lattice.
            # We add a high-entropy phase angle to explore the solution space.
            
            # 1. Expand input across output and lattice dimensions
            x_expanded = x.view(batch_size, 1, self.input_dim, 1).expand(-1, self.output_dim, -1, self.lattice_size)
            
            # 2. Generate frenzied search phases (high entropy exploration)
            search_phases = torch.rand_like(x_expanded) * 2 * math.pi
            
            # 3. Create the complex-valued superposition (amplitude * e^(i * phase))
            # The baseline amplitude is modulated by the existing classical CaMKII memory.
            current_memory = self.lattice.get_lattice()
            amplitude = x_expanded + current_memory.unsqueeze(0)
            
            # Storing the active superposition (The expanded Ghost Basin)
            self.active_superposition = torch.complex(
                amplitude * torch.cos(search_phases), 
                amplitude * torch.sin(search_phases)
            )
            self.is_superradiant = True
            
        # The output of the layer during the search phase is the massive 
        # superradiant state itself, propagating forward to be compared against the QPC.
        return self.active_superposition

    def objective_reduction_check(self, target_qpc_state, t_active_sec, wattage=1e11):
        """
        Calculates Penrose's Objective Reduction (Orch-OR) criteria.
        Instead of an arbitrary probability threshold, this computes if the 
        Gravitational Self-Energy (E_G) of the specific wave-shape (Fidelity)
        has exceeded the fundamental decoherence limit for the time it has
        been in superposition (t_active_sec).
        
        wattage: The number of active Tryptophan resonance molecules (N_trp).
                 The brain modulates this to adjust the superposition mass.
        """
        if not self.is_superradiant or self.active_superposition is None:
            return False
            
        # Physics Constants for Microtubule Lattice
        G = 6.67430e-11 # Gravitational constant
        hbar = 1.054571817e-34 # Reduced Planck constant
        m_trp = 3.39e-25 # Mass of Tryptophan (kg)
        delta_x = 2.5e-15 # Nuclear Fermi separation distance (m)
        
        # 1. Measure the exact shape of the active wave against the QPC concept
        # Fidelity F perfectly isolating the resonant geometric silhouette.
        F = self._calculate_phase_coherence(self.active_superposition, target_qpc_state)
        
        # 2. Calculate the actual Gravitational Self-Energy of the Phase-Locked shape
        # The higher the fidelity (resonance), the more dense the coherent mass overlay.
        M_superposed = wattage * m_trp
        E_G_max = G * (M_superposed ** 2) / delta_x
        
        # The true E_G of the current thought is scaled by its geometric fidelity
        E_G_actual = E_G_max * F
        
        # 3. Penrose Wave Collapse Threshold (tau = hbar / E_G)
        # If E_G is essentially 0 (noise), tau is infinite.
        if E_G_actual > 0:
            tau_decoherence = hbar / E_G_actual
        else:
            tau_decoherence = float('inf')
            
        # The quantum wave physically collapses if it has been held in the QED cavity
        # longer than its gravitational decoherence limit.
        if t_active_sec >= tau_decoherence:
            print(f"\n[Aha! Moment] Quantum Flash! Wave shape fidelity ({F:.4f}) drew enough mass-energy.")
            print(f"E_G limit ({E_G_actual:.2e} J) breached tau ({tau_decoherence:.2e} s). Triggering Objective Reduction!")
            return True
            
        return False

    def collapse_and_prune(self, phase_locked_state):
        """
        The Bi-Twistor Retrocausal Lens mechanism.
        Bypasses backprop entirely. Instantly translates the correct quantum future state
        into present classical memory and annihilates the incorrect search states.
        """
        # 1. RETROCAUSAL ALIGNMENT:
        # Extract the specific lattice configuration from the phase_locked_state that solved the problem.
        # Instantly physically rewrite the CaMKII phosphorylation matrix (the weights) physically or via global reference.
        # We take the real component of the aligned amplitude as the new classical memory structure.
        # The layer time-travels to alignment: No epochs, no gradients.
        self.lattice.write_silhouette(phase_locked_state)
            
        # 2. WAVE COLLAPSE PRUNING:
        # Instantly destroy the massive, high-entropy superradiant search matrix.
        # This frees immense computational resources, matching the brain's working memory snap.
        self.active_superposition = None
        self.is_superradiant = False
        
        print(f"[Pruning Complete] Ghost Basin collapsed. CaMKII lattice encoded into working memory.")

    def _calculate_phase_coherence(self, state_a, target_b):
        """
        Calculates the true quantum mechanical phase coherence (fidelity)
        between the UV superradiant search state and the immutable QPC state.
        Utilizes complex Hilbert space inner products suitable for twistor metrics,
        avoiding traditional Euclidean geometry.
        """
        # state_a shape: (batch_size, output_dim, input_dim, lattice_size)
        # target_b shape: (batch_size, input_dim)
        
        # Project the active superposition down to the input dimension to check if
        # the global resonant frequency resolves to the target concept.
        collapsed_state = torch.mean(state_a, dim=(1, 3))
        
        # Cast target_b to complex for Hilbert space inner product
        target_complex = torch.complex(target_b, torch.zeros_like(target_b))
        
        # <G | Q> = sum( G * conjugate(Q) )
        inner_product = torch.sum(collapsed_state * torch.conj(target_complex), dim=1)
        
        # <G | G> and <Q | Q>
        norm_a = torch.sum(collapsed_state * torch.conj(collapsed_state), dim=1)
        norm_b = torch.sum(target_complex * torch.conj(target_complex), dim=1)
        
        # Fidelity F = | <G | Q> |^2 / ( <G | G> * <Q | Q> )
        fidelity = torch.real(inner_product * torch.conj(inner_product)) / torch.real(norm_a * norm_b + 1e-10)
        
        return fidelity.mean().item()

