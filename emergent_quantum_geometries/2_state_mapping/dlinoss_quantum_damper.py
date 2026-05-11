"""
Phase 2: State Mapping
The D-LinOSS network acts as a Quantum Damper. It translates the 1/f noise 
topologies from the hardware into a dynamic, non-unitary damping matrix (Lindblad dissipator).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

class DLinOSSQuantumDamper(nn.Module):
    """
    A continuous-time diagonal State-Space Model (SSM) that maps stochastic hardware
    time-series into non-unitary damping parameters (ω_k, γ_k).
    """
    hidden_dim: int = 32
    out_dim: int = 32

    @nn.compact
    def __call__(self, phase_stream: jax.Array) -> jax.Array:
        """
        Args:
            phase_stream: Time-series input of shape (batch, seq_len, features)
        Returns:
            damping_matrix: Shape (batch, out_dim, out_dim)
        """
        batch_size, seq_len, features = phase_stream.shape

        # SSM Parameters
        # Lambda is the complex diagonal state transition matrix: exp(-gamma*dt + i*omega*dt)
        lambda_re = self.param('lambda_re', nn.initializers.normal(stddev=0.1), (self.hidden_dim,))
        lambda_im = self.param('lambda_im', nn.initializers.normal(stddev=0.5), (self.hidden_dim,))
        
        # Ensure lambda magnitude < 1 (stable damping)
        lam = jnp.exp(-jnp.abs(lambda_re) + 1j * lambda_im)
        
        B = self.param('B', nn.initializers.normal(stddev=0.1), (features, self.hidden_dim), jnp.complex64)
        C = self.param('C', nn.initializers.normal(stddev=0.1), (self.hidden_dim, self.out_dim), jnp.complex64)
        
        def scan_fn(state, x_t):
            # x_t is (batch, features)
            # state is (batch, hidden_dim)
            new_state = state * lam + jnp.dot(x_t.astype(jnp.complex64), B)
            y_t = jnp.dot(new_state, C)
            return new_state, y_t

        init_state = jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.complex64)
        
        # Scan over the sequence dimension (axis=1)
        # Transpose phase_stream to (seq_len, batch, features) for scan
        x_seq = jnp.transpose(phase_stream, (1, 0, 2))
        final_state, y_seq = jax.lax.scan(scan_fn, init_state, x_seq)
        
        # y_seq is (seq_len, batch, out_dim). Transpose back to (batch, seq_len, out_dim)
        y_seq = jnp.transpose(y_seq, (1, 0, 2))
        
        # Simulated sequence reconstruction output for training
        y_pred_seq = jnp.real(y_seq)
        
        # Return a simulated damping matrix for Phase 3 legacy compatibility
        # Real part of final pooled output -> square matrix
        pooled = jnp.mean(jnp.real(y_seq), axis=1) # (batch, out_dim)
        damping_vector = nn.Dense(self.out_dim * self.out_dim)(pooled)
        damping_matrix = damping_vector.reshape((batch_size, self.out_dim, self.out_dim))
        
        # MoQE: Learnable framework probability weights (unnormalized logits)
        moqe_weights = self.param('moqe_weights', nn.initializers.zeros, (5,))
        
        # Extract physical parameters for QPC metrics
        dt = 0.1
        omega_k = jnp.angle(lam) / dt
        gamma_k = -jnp.log(jnp.abs(lam)) / dt
        
        return damping_matrix, y_pred_seq, omega_k, gamma_k, moqe_weights
