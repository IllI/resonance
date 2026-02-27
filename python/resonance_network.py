import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try to import PyTorch, but handle if it's not available (since we are in an agent env)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal Mock for numpy-only execution if needed, 
    # but the user "repo" is PyTorch based, so we write the PyTorch code 
    # and fallback/explain if missing.
    pass

@dataclass
class NAAMConfig:
    n_neurons: int
    n_astrocytes: int
    embedding_dim: int = 64
    beta: float = 1.0  # Inverse temperature for Dense Memory
    astrocyte_decay: float = 0.95
    gliotransmission_strength: float = 0.5
    device: str = "cpu"

class NeuronAstrocyteAssociativeMemory(nn.Module if HAS_TORCH else object):
    """
    Implementation of the Neuron-Astrocyte Associative Memory (NAAM).
    Based on the architecture from 'kozleo/naam' and PNAS 2025 paper.
    
    Key Mechanism:
    - Neurons: Fast processing, Dense Associative Memory (Modern Hopfield).
    - Astrocytes: Slow integration, modulating the 'Attention' / energy landscape.
    - Tripartite Synapse: The interaction strength (beta) is locally controlled by astrocytes.
    """
    def __init__(self, config: NAAMConfig):
        super().__init__()
        self.config = config
        
        if not HAS_TORCH:
            print("WARNING: PyTorch not found. NAAM running in NumPy fallback mode.")
            self.build_numpy(config)
            return
            
        # --- PyTorch Implementation ---
        
        # 1. Neural Synaptic Weights (The "Memories")
        # Stores patterns X_mu
        self.memory_keys = nn.Parameter(torch.randn(config.n_astrocytes, config.n_neurons) * 0.01)
        self.memory_values = nn.Parameter(torch.randn(config.n_astrocytes, config.embedding_dim) * 0.01)
        
        # 2. Astrocyte State Vector (Calcium levels)
        # One astrocyte per memory pattern (simplification for dense memory)
        self.register_buffer('astrocyte_calcium', torch.zeros(config.n_astrocytes))
        
        # 3. Layer Norms
        self.ln = nn.LayerNorm(config.n_neurons)

    def build_numpy(self, config):
        """NumPy fallback initialization."""
        self.memory_keys = np.random.randn(config.n_astrocytes, config.n_neurons) * 0.01
        self.memory_values = np.random.randn(config.n_astrocytes, config.embedding_dim) * 0.01
        self.astrocyte_calcium = np.zeros(config.n_astrocytes)

    def forward(self, x: 'torch.Tensor', update_astrocytes: bool = True):
        """
        Associative Retrieval step.
        
        Args:
            x: (Batch, N_Neurons) Input firing pattern.
            update_astrocytes: Whether to update glial state.
            
        Returns:
            retrieved: (Batch, Embedding_Dim) The recalled information.
            synaptic_energy: (Batch) Energy of the current state.
        """
        if not HAS_TORCH:
            return self.forward_numpy(x, update_astrocytes)
            
        # Normalize input
        x = self.ln(x)
        
        # 1. Compute similarity (Overlap) with stored memory patterns
        # similarity = beta * (K @ x.T)
        # We allow Astrocytes to modulate the 'beta' per pattern
        
        # Astrocyte Modulation: "Gliotransmission"
        # High calcium -> Enhance synaptic reliability (Higher Beta)
        # Low calcium -> Depress/Noise (Lower Beta)
        local_beta = self.config.beta * (1.0 + self.config.gliotransmission_strength * self.astrocyte_calcium)
        
        # Similarity: (Batch, Patterns)
        overlap = F.linear(x, self.memory_keys) 
        
        # Apply Astrocyte Modulation
        # Patterns supported by active astrocytes are "sharper"
        modulated_overlap = overlap * local_beta.unsqueeze(0)
        
        # 2. Dense Associative Activation (Softmax / Exp)
        # This is the "Modern Hopfield" / Attention mechanism
        attention_weights = F.softmax(modulated_overlap, dim=-1)
        
        # 3. Retrieve Memory
        retrieved = F.linear(attention_weights, self.memory_values.t())
        
        # 4. Astrocyte Update Logic (Slow Dynamics)
        if update_astrocytes:
            with torch.no_grad():
                # Astrocytes sense the "Energy" (activity) of their domain
                # If a pattern is heavily recalled, astrocyte calcium rises
                current_activity = attention_weights.mean(dim=0) # Average over batch
                
                # Decay + Influx
                dt = 0.1
                decay = self.config.astrocyte_decay
                self.astrocyte_calcium = (1 - dt)*self.astrocyte_calcium + dt*current_activity
                
                # Homeostatic Scaling (optional, to prevent runaway)
                # self.astrocyte_calcium = torch.clamp(self.astrocyte_calcium, 0, 2.0)
        
        # Energy Calculation (Lyapunov function)
        # E = -lse(beta * overlap)
        energy = -torch.logsumexp(modulated_overlap, dim=-1)
        
        return retrieved, energy, attention_weights

    def forward_numpy(self, x: np.ndarray, update_astrocytes: bool = True):
        """NumPy implementation of the forward pass."""
        # Normalize
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-8
        x_norm = (x - mean) / std
        
        # Beta
        local_beta = self.config.beta * (1.0 + self.config.gliotransmission_strength * self.astrocyte_calcium)
        
        # Overlap
        overlap = x_norm @ self.memory_keys.T
        modulated_overlap = overlap * local_beta[None, :]
        
        # Softmax
        # Stable softmax
        m_max = np.max(modulated_overlap, axis=1, keepdims=True)
        exp_overlap = np.exp(modulated_overlap - m_max)
        attention_weights = exp_overlap / np.sum(exp_overlap, axis=1, keepdims=True)
        
        # Retrieve
        retrieved = attention_weights @ self.memory_values
        
        # Update
        if update_astrocytes:
            current_activity = np.mean(attention_weights, axis=0)
            dt = 0.1
            self.astrocyte_calcium = (1 - dt)*self.astrocyte_calcium + dt*current_activity
            
        # Energy
        energy = -np.log(np.sum(exp_overlap, axis=1)) - m_max.squeeze()
        
        return retrieved, energy, attention_weights


# ============================================================================
#  Data Integration: Connecting fMRI to NAAM
# ============================================================================

class BiofieldIntegrator:
    """
    Integrates the D-LinOSS fMRI analysis with the NAAM architecture.
    """
    def __init__(self, n_voxels: int, n_patterns: int = 100):
        self.config = NAAMConfig( n_neurons=n_voxels, n_astrocytes=n_patterns)
        self.naam = NeuronAstrocyteAssociativeMemory(self.config)
        self.history = []
        
    def process_bold_stream(self, processed_bold: np.ndarray, gnr_map: np.ndarray):
        """
        Process a stream of 'Denoised' BOLD data (neuronal component)
        modulated by the 'Astrocyte Field'.
        
        Args:
           processed_bold: (Batch/Time, N_Voxels) - from AstrocyteDeconvolver
           gnr_map: (N_Voxels,) - Glia-to-Neuron Ratio map
        """
        if HAS_TORCH:
            x = torch.tensor(processed_bold, dtype=torch.float32)
            # We can use the GNR map to initialize the astrocyte bias if we wanted
            # But NAAM learns the bias dynamically.
        else:
            x = processed_bold
            
        retrieved, energy, attention = self.naam.forward(x)
        
        # Store results
        if HAS_TORCH:
            self.history.append({
                'energy': energy.detach().numpy(),
                'attention': attention.detach().numpy(),
                'calcium': self.naam.astrocyte_calcium.clone().detach().numpy()
            })
        else:
             self.history.append({
                'energy': energy,
                'attention': attention,
                'calcium': self.naam.astrocyte_calcium.copy()
            })
            
        return self.history[-1]

    def check_grokking_state(self):
        """
        Analyze the NAAM energy landscape to detect Grokking.
        Grokking in NAAM = Deep Energy Minima + High Astrocyte Stability.
        """
        if not self.history:
            return 0.0
            
        latest = self.history[-1]
        
        # Deep energy valley = strong recognition
        avg_energy = np.mean(latest['energy'])
        
        # Astrocyte consensus (Entropy of attention)
        # If attention is sharp (low entropy), the network has 'grokked' the pattern
        att = np.mean(latest['attention'], axis=0)
        entropy = -np.sum(att * np.log(att + 1e-10))
        
        return {
            'system_energy': float(avg_energy),
            'recognition_entropy': float(entropy),
            'learning_state': 'GROK' if entropy < 2.0 else 'SEARCH'
        }
