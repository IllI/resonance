import torch
import numpy as np

class PoissonSpikeEncoder:
    """
    A universal signal encoder that takes any continuous input array (e.g. image pixels or audio waves)
    and converts it into a Poisson-distributed sequence of physiological spikes.
    This replaces traditional scalar numerical arrays with dense, temporal frequency spikes 
    (mimicking neural action potentials).
    """
    def __init__(self, time_steps=100, max_firing_rate=1.0):
        self.time_steps = time_steps
        self.max_firing_rate = max_firing_rate
        
    def encode(self, x):
        """
        Converts the input tensor x into a spike train.
        
        Args:
            x (torch.Tensor): A batch of continuous input values. Expected shape: (batch_size, num_features)
                              The input values should be scaled between 0 and 1.
                              
        Returns:
            spike_train (torch.Tensor): Binary tensor of shape (batch_size, time_steps, num_features) 
                                        where 1 indicates a spike at that time step.
        """
        # Ensure inputs are scaled to probabilities using max firing rate bounded by dt (1 time step)
        # We assume x is normalized between 0 and 1.
        rate = x * self.max_firing_rate
        
        # We need essentially to do a sequence of Bernoulli trials where p = rate
        # Shape: (batch_size, time_steps, num_features)
        
        # Expand x to the time dimension
        batch_size, num_features = x.shape
        rate_expanded = rate.unsqueeze(1).expand(batch_size, self.time_steps, num_features)
        
        # Generate random thresholds between 0 and 1 uniformly
        random_thresholds = torch.rand_like(rate_expanded)
        
        # A spike occurs if the random threshold is less than the firing rate
        spike_train = (random_thresholds < rate_expanded).float()
        
        return spike_train
        
    def get_frequency_state(self, spike_train):
        """
        Compresses a temporal spike train back into our 'frequency state' semantic representation.
        This provides a smooth harmonic state vector (Phase & Amplitude) that acts 
        as the input state for the MicrotubuleLayer.
        """
        # Sum spikes over time to get raw firing counts
        spike_counts = torch.sum(spike_train, dim=1)
        
        # Normalize into a frequency measure (0.0 to 1.0)
        frequency_state = spike_counts / float(self.time_steps)
        
        return frequency_state

# Example standalone use
if __name__ == "__main__":
    print("--- Testing Universal Poisson Spike Encoder ---")
    encoder = PoissonSpikeEncoder(time_steps=50, max_firing_rate=0.8)
    
    # Dummy image/audio input mapping (Shape 1 x 10)
    # Bright/loud pixels have ~1.0, dark/quiet pixels have ~0.0
    dummy_input = torch.tensor([[0.9, 0.1, 0.5, 0.8, 0.0, 1.0, 0.3, 0.7, 0.2, 0.9]])
    
    print("\nOriginal Input Intensities:")
    print(dummy_input)
    
    # Encode as discrete spikes over time
    spikes = encoder.encode(dummy_input)
    print(f"\nEncoded Spike Train Shape: {spikes.shape}")
    
    # Showing first feature (intensity 0.9) over all 50 timesteps
    print(f"Feature 0 Spikes (High prob): {spikes[0, :, 0].int().tolist()}")
    
    # Showing second feature (intensity 0.1) over all 50 timesteps
    print(f"Feature 1 Spikes (Low prob) : {spikes[0, :, 1].int().tolist()}")
    
    # Compress spike space into the "Frequency State" output
    freq_state = encoder.get_frequency_state(spikes)
    print("\nRecovered Frequency State (Input fed to D-LinOSS):")
    print(freq_state)
