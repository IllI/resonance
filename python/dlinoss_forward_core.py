"""
dlinoss_forward_core.py
========================
The formal implementation of the FORWARD Bi-Twistor Architecture for D-LinOSS.

Key Architectural Shift:
- Euclidean math (KFAC, gradients, similarities) happens strictly in the 
  Ghost Basin (the projection shadow in R^n).
- The QPC (Quantum Phantom Cauldron) exists entirely outside the computable 
  gradient graph as an immutable Complex Projective Ray (CP^n).
- Learning (microtubule adjustment) only tunes the Funnel (the biological 
  projection), disentangled via KFAC on the Euclidean shadow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexProjectiveQPC(nn.Module):
    """
    The immutable 'truth' of the concept space. 
    Maintains rank-1 complex rays that are unobservable directly.
    """
    def __init__(self, num_categories, twistor_dim):
        super().__init__()
        self.dim = twistor_dim
        
        # Concepts are static, non-learnable quantum objects in this paradigm.
        # The biological organism doesn't change what a 'Badoon' is, it only 
        # learns how to biologically project/observe it.
        c_real = torch.randn(num_categories, twistor_dim)
        c_imag = torch.randn(num_categories, twistor_dim)
        
        # Normalize to hypersphere
        norm = torch.sqrt(c_real**2 + c_imag**2 + 1e-10)
        self.register_buffer('concept_real', c_real / norm)
        self.register_buffer('concept_imag', c_imag / norm)
        
    def get_ray(self, category_idx):
        real = self.concept_real[category_idx]
        imag = self.concept_imag[category_idx]
        return torch.complex(real, imag)

class MicrotubuleFunnel(nn.Module):
    """
    The biological interface. Projects unobservable complex twistor space
    down into measurable Euclidean fMRI BOLD space (Ghost Basin).
    This is what actually 'learns' and collapses during grokking.
    """
    def __init__(self, twistor_dim, voxel_dim):
        super().__init__()
        # Microtubule projection parameters (the classical-quantum boundary)
        self.P_real = nn.Linear(twistor_dim, voxel_dim, bias=False)
        self.P_imag = nn.Linear(twistor_dim, voxel_dim, bias=False)
        
    def forward(self, psi_ray):
        """
        Input: Complex ray (CP^n)
        Output: Ghost Basin Shadow (R^n)
        """
        # Separate amplitudes and phases
        real_part = torch.real(psi_ray)
        imag_part = torch.imag(psi_ray)
        
        # Project into biological substrate
        # The organism creates an interference pattern between amplitude and phase
        shadow_real = self.P_real(real_part)
        shadow_imag = self.P_imag(imag_part)
        
        # Superposition of projections creates the macroscopic fMRI pattern
        ghost_basin = shadow_real + shadow_imag
        
        # Normalize directional geometry in Euclidean space
        ghost_basin = F.normalize(ghost_basin, p=2, dim=-1)
        return ghost_basin

class DLinOSS_ForwardArchitecture(nn.Module):
    def __init__(self, num_categories, twistor_dim, voxel_dim):
        super().__init__()
        self.qpc = ComplexProjectiveQPC(num_categories, twistor_dim)
        self.funnel = MicrotubuleFunnel(twistor_dim, voxel_dim)
        
    def forward(self, category_idx):
        # 1. Access the unobservable truth
        psi = self.qpc.get_ray(category_idx)
        
        # 2. Project through biological funnel
        ghost_basin = self.funnel(psi)
        
        return ghost_basin

# =============================================================================
# KFAC Disentanglement Math 
# =============================================================================
def apply_kfac_to_funnel(model, fMRI_data_batch, learning_rate=0.01):
    """
    KFAC Disentanglement happens strictly down here in the Euclidean shadow.
    We precondition the gradients of the Bi-Twistor funnel parameters.
    
    Why? The brain's microtubule network has vast dimensional redundancies.
    Standard SGD would tangle the updates. KFAC uses the Fisher Information 
    Matrix of the Euclidean shadow to orthogonality isolate the specific 
    Eigenmodes corresponding to the Badoon concept, allowing for a clean, 
    narrow wave-function collapse without categorical bleedover.
    """
    # 1. Forward pass to get the theoretical shadow
    # Assuming batch of category indices
    categories = torch.zeros(fMRI_data_batch.size(0), dtype=torch.long) # Dummy categoric matching
    predicted_shadows = model(categories)
    
    # 2. Euclidean Loss (measured entirely in the Ghost Basin)
    # Cosine distance between theoretical shadow and empirical BOLD data
    loss = 1.0 - F.cosine_similarity(predicted_shadows, fMRI_data_batch).mean()
    
    # 3. Compute Gradients for the Funnel
    model.zero_grad()
    loss.backward()
    
    # 4. (Conceptual) KFAC Preconditioning
    # In a full KFAC implementation, we would construct the Kronecker-factored
    # inverse Fisher Information Matrix (FIM) for P_real and P_imag based on 
    # the Euclidean activations and pre-activation gradients, then multiply 
    # the raw gradients by FIM^-1.
    
    with torch.no_grad():
        for param in model.funnel.parameters():
            if param.grad is not None:
                # PSEUDO-KFAC: We assume the KFAC preconditioner disentangles 
                # the gradient into the natural gradient space.
                natural_grad = param.grad # (Placeholder for FIM^-1 @ grad)
                
                # Update the microtubule configuration
                param.sub_(learning_rate * natural_grad)
                
    return loss.item()
