import torch
import torch.nn as nn

class EntangledCaMKIILattice(nn.Module):
    """
    A globally shared quantum memory matrix modeled after the hexagonal CaMKII
    phosphorylation patterns on biological microtubules.
    
    Instead of individual D-LinOSS agents having isolated weights, this class 
    acts as the entangled 'Quantum Cauldron'. Multiple agents (e.g., Audio, Video)
    can point to this exact exact memory instance. When Objective Reduction 
    (wave collapse) triggers in ANY agent, this universal memory structure is 
    updated instantly for ALL agents.
    """
    def __init__(self, input_dim, output_dim, lattice_size=9):
        super(EntangledCaMKIILattice, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lattice_size = lattice_size
        
        # The true classical memory trace.
        # It is structurally geometric and intransigent to standard gradients.
        self.matrix = nn.Parameter(
            torch.zeros(output_dim, input_dim, lattice_size), 
            requires_grad=False
        )
        
    def write_silhouette(self, collapsed_wave_state):
        """
        The Biological 'Quantum Flash'.
        Once a superradiant thought achieves Objective Reduction, its geometric 
        silhouette is physically written into the CaMKII lattice.
        
        Because multiple agents point to this class, calling this method 
        instantly propagates the memory to all sensors.
        """
        # Average the real components over the batch vector to get the global 
        # structural logic trace.
        aligned_memory = torch.real(collapsed_wave_state).mean(dim=0)
        
        with torch.no_grad():
            self.matrix.copy_(aligned_memory)
            
    def get_lattice(self):
        """Returns the current state of the entangled network geometry."""
        return self.matrix
