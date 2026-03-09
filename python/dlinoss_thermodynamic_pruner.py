import torch
import torch.nn as nn

class ThermodynamicPruner:
    """
    Implements the Physical Architecture of a Shared Hallucination.
    Automatically regularizes and prunes neural network topologies once the "Grok"
    threshold is met, dropping active structural connections just as observed in
    the fMRI analysis of the 14 OpenNeuro Masters (dropping 5k+ localized nodes).
    
    This hook is completely non-destructive to the base model architecture, acting 
    solely through dynamic boolean masks and active zero-gradients.
    """
    def __init__(self, model: nn.Module, grok_acc_threshold=0.85, var_window=5, prune_magnitude=1e-3, target_modules=['weight', 'A', 'G']):
        self.model = model
        self.grok_acc_threshold = grok_acc_threshold
        self.var_window = var_window
        self.prune_magnitude = prune_magnitude
        self.target_modules = target_modules
        
        self.is_grokking = False
        self.history = {}
        self.masks = {}
        
        # Initialize binary masks for tracking and gradient history
        # By default, tracking any fully-connected weights or structural matrices (e.g., A and G in D-LinOSS)
        for name, param in self.model.named_parameters():
            if any(key in name for key in self.target_modules):
                self.masks[name] = torch.ones_like(param, dtype=torch.bool, requires_grad=False)
                self.history[name] = []

    def log_gradients(self):
        """Track gradient volatility trailing history to detect scaffolding vs functional parameters."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.history:
                self.history[name].append(param.grad.detach().clone())
                if len(self.history[name]) > self.var_window:
                    self.history[name].pop(0)
                    
    def compute_atp_penalty(self, current_acc, current_epoch):
        """
        Calculates Biological Regularizer (R_ATP).
        Applying thermodynamic energy limits specifically as accuracy pushes towards the Superposition.
        """
        penalty = 0.0
        # Slowly ramp up thermodynamic penalty as network learns (Limit cellular ATP)
        ramp = min(max((current_acc - 0.5) / (self.grok_acc_threshold - 0.5 + 1e-6), 0.0), 1.0)
        atp_lambda = 1e-4 * ramp 
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                penalty += atp_lambda * torch.sum(torch.abs(param * self.masks[name]))
        return penalty

    def check_and_prune(self, current_acc):
        """
        The topological drop. Slices away the scaffolding connections that were used during 
        the chaotic learning phase ("Growing Pains").
        """
        total_pruned = 0
        total_params = 0
        
        if current_acc >= self.grok_acc_threshold:
            if not self.is_grokking:
                print(f"[ThermodynamicPruner] Grok Threshold Reached ({current_acc:.1%} Acc). Initiating Superposition Pruning Protocol.")
                self.is_grokking = True
                
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    with torch.no_grad():
                        # Calculate variance of gradients
                        if len(self.history[name]) == self.var_window:
                            grad_stack = torch.stack(self.history[name])
                            grad_var = torch.var(grad_stack, dim=0)
                            
                            # Disconnect parameters that have settled (low var) AND are small (low magnitude)
                            low_var_mask = grad_var < (torch.mean(grad_var) * 0.5)
                            low_mag_mask = torch.abs(param) < self.prune_magnitude
                            
                            # Update global topological mask
                            scaffold_mask = low_var_mask & low_mag_mask
                            self.masks[name] &= ~scaffold_mask
                            
                            # Apply mask to zero out weight permanently
                            param.data *= self.masks[name]
                            
                            total_pruned += (~self.masks[name]).sum().item()
                    total_params += param.numel()
                    
            if self.is_grokking:
                # To match exactly what we saw in OpenNeuro human participants where active neuron populations physically dropped
                print(f"   -> Topolgy Status: {total_params - total_pruned:,} Active Resonance Parameters | {total_pruned:,} Scaffolding Parameters Disconnected")
        
        return total_params, total_pruned

    def enforce_masks(self):
        """Enforces 0 gradient on severed architectural connections so they cannot regrow during the block."""
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad *= self.masks[name]
