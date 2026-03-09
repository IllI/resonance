import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# A PyTorch hook/callback module to handle Thermodynamic Pruning dynamically
class ThermodynamicPruner:
    """
    Implements the Physical Architecture of a Shared Hallucination.
    Automatically regularizes and prunes D-LinOSS matrices once the "Grok"
    threshold is met (mimicking True Masters resting at 89% precision).
    """
    def __init__(self, model, grok_acc_threshold=0.85, var_window=5, prune_magnitude=1e-3):
        self.model = model
        self.grok_acc_threshold = grok_acc_threshold
        self.var_window = var_window
        self.prune_magnitude = prune_magnitude
        
        self.is_grokking = False
        self.history = {}
        self.masks = {}
        
        # Initialize binary masks for tracking and gradient history
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'A' in name or 'G' in name:
                self.masks[name] = torch.ones_like(param, dtype=torch.bool, requires_grad=False)
                self.history[name] = []

    def log_gradients(self):
        """Track gradient volatility trailing history to detect scaffolding."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.history:
                self.history[name].append(param.grad.detach().clone())
                if len(self.history[name]) > self.var_window:
                    self.history[name].pop(0)
                    
    def compute_atp_penalty(self, current_acc, current_epoch):
        """
        Calculates Biological Regularizer (R_ATP).
        Only strongly engages as accuracy starts nearing the Grok peak.
        """
        penalty = 0.0
        # Slowly ramp up thermodynamic penalty as network learns (Limit cellular ATP)
        ramp = min(max((current_acc - 0.5) / (self.grok_acc_threshold - 0.5 + 1e-6), 0.0), 1.0)
        atp_lambda = 1e-4 * ramp 
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Apply frequency/resonance penalty logic here. Right now, applying a generic L1 norm for scale
                penalty += atp_lambda * torch.sum(torch.abs(param * self.masks[name]))
        return penalty

    def check_and_prune(self, current_acc):
        """
        The topological drop. Slices away the scaffolding used during the chaotic learning phase.
        """
        total_pruned = 0
        total_params = 0
        
        if current_acc >= self.grok_acc_threshold:
            if not self.is_grokking:
                print(f"[ThermodynamicPruner] Grok Threshold Reached ({current_acc:.1%} Acc). Initiating Structural Pruning.")
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
                print(f"   -> Topolgy Status: {total_params - total_pruned:,} Active Parameters | {total_pruned:,} Disconnected Parameters")
        
        return total_params, total_pruned

    def enforce_masks(self):
        """Enforces 0 gradient on severed architectural connections."""
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad *= self.masks[name]


# --- DUMMY SIMULATION OF D-LINOSS GROKKING ---
class SimpleDLinOSS(nn.Module):
    def __init__(self, in_dim=50, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1) # Simulate linear recurrence
        self.fc2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = h + torch.matmul(h, self.A)
        return self.fc2(h)

def simulate_training_loop():
    print("==========================================================================")
    print(" Zero-Shot D-LinOSS: Thermodynamic Grokking Topology Pruning")
    print("==========================================================================")
    
    # 1. Setup Phase
    model = SimpleDLinOSS(in_dim=50, hidden_dim=4096)
    pruner = ThermodynamicPruner(model, grok_acc_threshold=0.88, var_window=5, prune_magnitude=0.015)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Generate fake experimental blocks (noisy features vs categorical prototypes)
    X_train = torch.randn(1000, 50) 
    # Force an underlying arbitrary "Prototype" rule
    y_train = (X_train.sum(dim=1) > 0).long()
    
    # Add noise to simulate chaotic "Growing Pains" exemplars
    X_train += torch.randn_like(X_train) * 0.5
    
    epochs = 40
    batch_size = 50
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle
        indices = torch.randperm(X_train.size(0))
        
        for i in range(0, 1000, batch_size):
            batch_idx = indices[i:i+batch_size]
            inputs, labels = X_train[batch_idx], y_train[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Base classification loss
            loss = criterion(outputs, labels)
            
            # Simulated Accuracy (Trailing proxy)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            current_acc = correct / total
            
            # + Biological R_ATP Regularizer (Starts heavily penalizing at high acc)
            atp_penalty = pruner.compute_atp_penalty(current_acc, epoch)
            loss += atp_penalty
            
            loss.backward()
            
            # Track gradients to distinguish meaningful topology from scaffolding
            pruner.log_gradients()
            
            # Enforce severed connections
            pruner.enforce_masks()
            
            optimizer.step()
            total_loss += loss.item()
            
        epoch_acc = correct / total
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:02d} | Loss: {total_loss/20:.3f} | Accuracy: {epoch_acc:.1%}")
            
        # The Critical Step: Prune the Physical Architecture if we hit Grokking Phase
        pruner.check_and_prune(epoch_acc)

if __name__ == "__main__":
    simulate_training_loop()
