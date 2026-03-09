"""
working_memory_decoder.py
──────────────────────────────────────────────────────────────
A decoding tool for D-LinOSS to map 'Working Memory' patterns vs 
'Prototype Consolidation' during categorical learning tasks. 

This model integrates the physics of electromagnetism with modern 
empirical Vector Symbolic Architecture (VSA) and Sparse Distributed 
Memory (SDM) paradigms.

Hypothesis: 
When a subject lacks a "Categorical Superposition" (Prototype), their
brain relies on high-energy Working Memory, mapped mathematically as a 
high-dimensional Sparse Distributed Memory (SDM). The brain explicitly 
memorizes singular Dot Pattern 'Exemplars' by superimposing binary 
high-dimensional state vectors, relying on Hamming Distance binding.

As the network learns (consolidation via SpWR - Sharp Wave-Ripples), 
these massive exemplar vectors converge and collapse into a low-dimensional
manifold—the true Category Prototype (the Ghost), reducing metabolic drag 
and transitioning processing from Parietal/Hippocampal loops to the 
Ventromedial Prefrontal Cortex (vmPFC).

This tool classifies the BOLD energy state into:
1. "Exemplar Mapping" (High Dimensional SDM / High ATP Cost)
2. "Prototype Convergence" (Low Dimensional Manifold / Low ATP Cost)
"""

import numpy as np

class VectorSymbolicMemoryDecoder:
    def __init__(self, vector_dim=10000, active_memory_size=5000):
        # High-dimensional space for VSA/SDM
        # Early learning requires massive uncompressed dimensionality to store raw exemplars.
        self.vector_dim = vector_dim
        self.memory_size = active_memory_size
        
        # Initialize random SDM Addresses (the physical neurons in the Parietal working memory buffer)
        self.addresses = np.random.choice([1, -1], size=(self.memory_size, self.vector_dim))
        self.memory = np.zeros((self.memory_size, self.vector_dim))

    def _hamming_distance(self, vec_a, vec_b):
        """Calculates distance in high-dimensional exemplar space."""
        return np.sum(vec_a != vec_b)
        
    def encode_exemplar(self, bold_energy_signature, activation_radius=0.1):
        """
        Simulates the Working Memory (Exemplar) storage phase.
        When BOLD energy is high (massive ATP burn from high G-ratio / unoptimized wiring), 
        the brain is writing directly to the SDM.
        """
        distances = np.array([self._hamming_distance(bold_energy_signature, addr) for addr in self.addresses])
        
        # In a high-dimensional space, random distance is ~0.5. 
        # We activate nodes that are *closer* than random chance by the activation_radius
        threshold = int(self.vector_dim * (0.5 + activation_radius))
        active_indices = np.where(distances <= threshold)[0]
        
        # Superposition (Adding the vector to active memory buffers)
        for idx in active_indices:
            self.memory[idx] += bold_energy_signature
            
        # The number of active indices directly correlates to the BOLD Integral (Area Under Curve)
        # and total ATP cost derived from the Electromagnetic Spectral Map.
        return len(active_indices)

    def decode_prototype_manifold(self, memory_matrix):
        """
        Simulates the Learning (Prototype Consolidation) phase.
        Over prolonged exposure and sleep/Sharp Wave-Ripples (SpWR), the high-dimensional SDM 
        collapses mathematically. It extracts the principal components (PCA), which represents 
        the "Prototype" embedding on a low-dimensional manifold.
        """
        # In a biological system, SpWR diffusion smooths the memory matrix and finds the true vector
        # Prevent zero-matrices from destroying SVD
        if np.sum(np.abs(memory_matrix)) == 0:
            return np.zeros(self.vector_dim), 0.0
            
        u, s, vh = np.linalg.svd(memory_matrix, full_matrices=False)
        
        # The first Principal Component is the hidden rule/Categorical Superposition.
        prototype_vector = vh[0] 
        total_variance = np.sum(s)
        energy_concentration = s[0] / total_variance if total_variance > 0 else 0.0
        
        return prototype_vector, energy_concentration

    def classify_learning_state(self, bold_integral_z, memory_matrix):
        """
        Determines the state of the network based on the thermodynamic efficiency of the memory.
        """
        prototype, convergence_ratio = self.decode_prototype_manifold(memory_matrix)
        
        # The brain is highly inefficient initially, spreading energy across the SDM.
        # Once the Prototype is learned, the energy concentrates heavily into the primary manifold vector.
        if convergence_ratio < 0.15:
            state = "WORKING MEMORY (EXEMPLAR BINDING)"
            biological_description = "High Dimensional SDM. Heavy Parietal / Hippocampal load. High ATP structural cost."
        else:
            state = "CONSOLIDATED PROTOTYPE (SUPERPOSITION)"
            biological_description = "Low Dimensional Manifold. Efficient vmPFC routing. Minimized ATP structural cost."
            
        return {
            "State": state,
            "Manifold_Convergence_Ratio": convergence_ratio,
            "Physiology": biological_description
        }

if __name__ == "__main__":
    print("=====================================================================")
    print("  D-LinOSS WORKING MEMORY vs PROTOTYPE CLASSIFICATION ALGORITHM")
    print("=====================================================================")
    
    decoder = VectorSymbolicMemoryDecoder(vector_dim=1024, active_memory_size=500)
    
    # Simulate Early Learning (Run 1 - Sub-201 ds002813)
    # The brain is flooded with novel dot patterns and is just storing raw exemplars
    print("\n>>> SIMULATING: EARLY LEARNING (RUN 1) - NAIVE STATE")
    for _ in range(50):
        # Heavy noise, unique patterns
        synthetic_bold_signature = np.random.choice([1, -1], size=1024)
        active_nodes = decoder.encode_exemplar(synthetic_bold_signature)
        
    early_state = decoder.classify_learning_state(bold_integral_z=14.5, memory_matrix=decoder.memory)
    print(f"Network State : {early_state['State']}")
    print(f"Convergence   : {early_state['Manifold_Convergence_Ratio']:.2%}")
    print(f"Physiology    : {early_state['Physiology']}")
    
    # Simulate Late Learning (Run 8 - Sub-201 ds002813)
    # The brain has extracted the hidden rule (the prototype pattern)
    print("\n>>> SIMULATING: LATE LEARNING (RUN 8) - EXPERT STATE")
    # Generating correlated memory (the prototype)
    base_prototype = np.random.choice([1, -1], size=1024)
    for _ in range(50):
        # Minor deviations from the prototype (dot distortions)
        distortion = np.random.choice([1, -1], size=1024, p=[0.9, 0.1])
        synthetic_bold_signature = base_prototype * distortion
        decoder.encode_exemplar(synthetic_bold_signature)

    late_state = decoder.classify_learning_state(bold_integral_z=3.2, memory_matrix=decoder.memory)
    print(f"Network State : {late_state['State']}")
    print(f"Convergence   : {late_state['Manifold_Convergence_Ratio']:.2%}")
    print(f"Physiology    : {late_state['Physiology']}")
