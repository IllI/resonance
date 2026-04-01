"""
D-LinOSS PyTorch Port
======================
Faithful PyTorch implementation of the Damped Linear Oscillatory State-Space
Model from "Learning to Dissipate Energy in Oscillatory State-Space Models"
(Walker, Berman, Rusch, 2025) — arXiv:2505.12171v1.

Reference: https://github.com/jaredbmit/damped-linoss (JAX/Equinox)

This port replicates the exact mathematical structure:
    z'(t) = -A x(t) - G z(t) + B u(t)
    x'(t) = z(t)
    y(t)  = C x(t) + D u(t)

Discretized via IMEX (implicit G, explicit A):
    w_{k+1} = M w_k + F u_{k+1}
    y_{k+1} = H w_k

Where the recurrent matrix M (per oscillator i) is:
    M_i = [[1/S_i,           -Δt·a_i/S_i    ],
            [Δt/S_i,    1 - Δt²·a_i/S_i ]]
    S_i = 1 + Δt·g_i   (Schur complement)

Eigenvalues of M_i:
    λ± = (tr ± √(tr² - 4·det)) / 2
    tr(M_i) = 1/S + 1 - Δt²A/S = (1 + S - Δt²A) / S
    det(M_i) = 1/S
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
#  Parallel scan via prefix-sum (associative scan)
# ============================================================================

def _binary_op_2x2(Ai: torch.Tensor, bi: torch.Tensor,
                   Aj: torch.Tensor, bj: torch.Tensor):
    """
    Binary operator for associative scan of 2x2 block-diagonal linear recurrence.

    Each "matrix" is stored as a flat vector [a, b, c, d] representing:
        [[a, b],
         [c, d]]

    Matrix multiplication: M_j @ M_i   (note: j acts on i, left-multiply)
    Vector update: b_new = M_j @ b_i + b_j

    Args:
        Ai: (L, 4P) flattened matrices at position i
        bi: (L, 2P) flattened bias at position i
        Aj: (L, 4P) flattened matrices at position j
        bj: (L, 2P) flattened bias at position j

    Returns:
        A_new: (L, 4P) combined matrix
        b_new: (L, 2P) combined bias
    """
    N = Ai.shape[-1] // 4

    iA = Ai[..., 0*N:1*N]
    iB = Ai[..., 1*N:2*N]
    iC = Ai[..., 2*N:3*N]
    iD = Ai[..., 3*N:4*N]

    jA = Aj[..., 0*N:1*N]
    jB = Aj[..., 1*N:2*N]
    jC = Aj[..., 2*N:3*N]
    jD = Aj[..., 3*N:4*N]

    # 2x2 matrix multiply (element-wise for diagonal blocks)
    A_new = jA * iA + jB * iC
    B_new = jA * iB + jB * iD
    C_new = jC * iA + jD * iC
    D_new = jC * iB + jD * iD
    M_new = torch.cat([A_new, B_new, C_new, D_new], dim=-1)

    # Bias update: M_j @ b_i + b_j
    bi1, bi2 = bi[..., :N], bi[..., N:]
    new_b1 = jA * bi1 + jB * bi2
    new_b2 = jC * bi1 + jD * bi2
    b_new = torch.cat([new_b1, new_b2], dim=-1) + bj

    return M_new, b_new


def parallel_scan(M_elements: torch.Tensor, F_elements: torch.Tensor):
    """
    Compute parallel prefix scan for the linear recurrence:
        w_{k+1} = M_k @ w_k + F_k

    Sequential implementation (autograd-safe, no in-place ops).
    For production on GPU, replace with a CUDA custom op using Blelloch pattern.

    Args:
        M_elements: (L, 4P) per-step transition matrices (flattened 2x2 blocks)
        F_elements: (L, 2P) per-step forcing vectors

    Returns:
        M_out: (L, 4P) cumulative transition matrices
        w_out: (L, 2P) cumulative state vectors
    """
    L = M_elements.shape[0]

    if L == 1:
        return M_elements, F_elements

    # Accumulate as lists to avoid in-place tensor indexing (breaks autograd)
    M_list = [M_elements[0]]
    w_list = [F_elements[0]]

    for k in range(1, L):
        m_new, w_new = _binary_op_2x2(
            M_list[-1], w_list[-1],
            M_elements[k], F_elements[k]
        )
        M_list.append(m_new)
        w_list.append(w_new)

    return torch.stack(M_list), torch.stack(w_list)


# ============================================================================
#  Gated Linear Unit (GLU)
# ============================================================================

class GLU(nn.Module):
    """Gated Linear Unit: GLU(x) = W1(x) ⊙ σ(W2(x))"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim)
        self.w2 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.sigmoid(self.w2(x))


# ============================================================================
#  D-LinOSS Layer (DampedIMEX1 — Primary Variant from Paper §3.1)
# ============================================================================

class DLinOSSLayer(nn.Module):
    """
    Damped Linear Oscillatory State-Space Layer (DampedIMEX1 variant).

    This is the PRIMARY variant from the paper, using:
        - Implicit treatment of damping G (→ unconditional stability for G ≥ 0)
        - Explicit treatment of stiffness A (→ oscillatory dynamics preserved)

    Continuous-time ODE:
        z'(t) = -A·x(t) - G·z(t) + B·u(t)    [velocity/momentum]
        x'(t) = z(t)                            [position]
        y(t)  = C·x(t) + D·u(t)                [observation]

    IMEX Discretization:
        z_{k+1} = z_k + Δt·(-A·x_k - G·z_{k+1} + B·u_{k+1})
        x_{k+1} = x_k + Δt·z_{k+1}

    Solving for z_{k+1} (implicit G):
        (I + Δt·G)·z_{k+1} = z_k - Δt·A·x_k + Δt·B·u_{k+1}
        S = I + Δt·G   (Schur complement)

    Recurrence matrix M_i (per oscillator i):
        [[1/S_i,          -Δt·a_i/S_i    ],
         [Δt/S_i,    1 - Δt²·a_i/S_i ]]

    Learnable parameters: A_diag (P,), G_diag (P,), dt_raw (P,), B (P,H,2), C (H,P,2), D (H,)
    where B, C are stored as (real, imag) pairs.

    Stability: Guaranteed when eigenvalues of M lie inside unit disk.
    The soft projection ensures:
        G ≥ 0
        A_low ≤ A ≤ A_high
    where:
        A_low  = (2 + Δt·G - 2√(1 + Δt·G)) / Δt²
        A_high = (2 + Δt·G + 2√(1 + Δt·G)) / Δt²
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.01,
        theta_max: float = math.pi,
        dt_std: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Initialize via ring (eigenvalue-first) or uniform (parameter-first)
        if initialization == "ring":
            A_init, G_init, dt_init = self._ring_init(
                state_dim, r_min, r_max, theta_min, theta_max, dt_std
            )
        else:
            A_init, G_init, dt_init = self._uniform_init(state_dim, dt_std)

        self.A_diag = nn.Parameter(A_init)
        self.G_diag = nn.Parameter(G_init)
        self.dt_raw = nn.Parameter(dt_init)

        # Complex B, C stored as real tensors with last dim = 2
        # B: (P, H, 2) — maps hidden→state (complex)
        # C: (H, P, 2) — maps state→hidden (complex)
        B_std = 1.0 / math.sqrt(hidden_dim)
        C_std = 1.0 / math.sqrt(state_dim)
        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2).uniform_(-B_std, B_std))
        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2).uniform_(-C_std, C_std))
        self.D = nn.Parameter(torch.randn(hidden_dim))

    @staticmethod
    def _ring_init(P: int, r_min: float, r_max: float,
                   theta_min: float, theta_max: float,
                   dt_std: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize (A, G, Δt) by sampling eigenvalues in an annular ring
        and inverting the eigenvalue-to-parameter map.

        From Proposition 3.4 of the paper:
            The map (G, A, Δt) ↦ (λ₁, λ₂) is bijective.
            Given target λ = r·e^{iθ}, we can solve for (A, G).

        For the DampedIMEX1 recurrence matrix:
            M_i = [[1/S,        -Δt·a/S  ],
                   [Δt/S,   1 - Δt²·a/S]]
            S = 1 + Δt·g

        Eigenvalues:
            tr(M) = 1/S + 1 - Δt²a/S = (1 + S - Δt²a)/S
            det(M) = 1/S
            λ± = (tr ± √(tr² - 4det)) / 2

        Inverse map (given λ₁ = r·e^{iθ}, λ₂ = r·e^{-iθ}):
            det(M) = λ₁·λ₂ = r²   →  S = 1/r²  →  g = (1/r² - 1)/Δt
            tr(M) = λ₁ + λ₂ = 2r·cos(θ)
            From tr = (1 + S - Δt²a)/S:
                a = (1 + S - S·tr) / Δt²
                a = (1 + 1/r² - 2cos(θ)/r) / Δt²
        """
        # Sample Δt
        dt_raw = torch.randn(P) * dt_std
        dt = torch.sigmoid(dt_raw)

        # Sample eigenvalues uniformly in annular ring
        # |λ| ∈ [r_min, r_max], arg(λ) ∈ [θ_min, θ_max]
        # Uniform in area: r = √(U·(r_max² - r_min²) + r_min²)
        r = torch.sqrt(torch.rand(P) * (r_max**2 - r_min**2) + r_min**2)
        theta = torch.rand(P) * (theta_max - theta_min) + theta_min

        # Inverse map: (λ₁, λ₂, Δt) → (A, G)
        # det(M) = r² → S = 1/r² → g = (S - 1)/Δt = (1/r² - 1)/Δt
        S = 1.0 / (r * r)
        G_init = (S - 1.0) / dt.clamp(min=1e-6)

        # tr(M) = 2r·cos(θ) → a = (1 + S - S·2r·cos(θ)) / Δt²
        tr = 2 * r * torch.cos(theta)
        A_init = (1.0 + S - S * tr) / dt.pow(2).clamp(min=1e-12)

        # Verify stability: |λ| < 1
        # Since r ∈ [r_min, r_max] ⊂ [0, 1], this is guaranteed by construction.

        return A_init.float(), G_init.float(), dt_raw.float()

    @staticmethod
    def _uniform_init(P: int, dt_std: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniform initialization with rejection sampling in valid region.

        Valid region (for complex eigenvalues = oscillatory regime):
            G ≥ 0
            (G - Δt·A)² < 4A
        """
        A_vals, G_vals, dt_vals = [], [], []
        batch_size = 512

        while len(A_vals) < P:
            A = torch.rand(batch_size) * 10.0 + 0.1
            G = torch.rand(batch_size) * 2.0
            dt_raw = torch.randn(batch_size) * dt_std
            dt = torch.sigmoid(dt_raw)

            # Validity check
            disc = (G - dt * A).pow(2) - 4 * A
            valid = (G >= 0) & (disc < 0)

            A_vals.extend(A[valid].tolist())
            G_vals.extend(G[valid].tolist())
            dt_vals.extend(dt_raw[valid].tolist())

        return (
            torch.tensor(A_vals[:P]),
            torch.tensor(G_vals[:P]),
            torch.tensor(dt_vals[:P])
        )

    def _soft_project(self, A: torch.Tensor, G: torch.Tensor, dt_raw: torch.Tensor):
        """
        Soft projection to valid stability region.

        Ensures eigenvalues of M lie strictly inside the unit disk.

        Steps:
            1. dt = σ(dt_raw) ∈ (0,1)
            2. G = ReLU(G) ≥ 0
            3. A clamped to [A_low, A_high] where:
                A_low  = (2 + Δt·G - 2√(1 + Δt·G)) / Δt²
                A_high = (2 + Δt·G + 2√(1 + Δt·G)) / Δt²

        Mathematical Derivation:
            The eigenvalues of M are complex when the discriminant < 0:
                (tr² - 4det) < 0
            With tr = (1 + S - Δt²A)/S and det = 1/S, S = 1 + ΔtG.

            The bound |λ|² < 1 ⟺ det(M) < 1 ⟺ 1/S < 1 ⟺ S > 1 ⟺ ΔtG > 0.
            This is automatically satisfied by G ≥ 0, Δt > 0.

            For the discriminant condition (complex eigenvalues → oscillation):
                tr² < 4det
                ((1 + S - Δt²A)/S)² < 4/S
                (1 + S - Δt²A)² < 4S
                Let u = Δt²A:  (1 + S - u)² < 4S
                Solving: u ∈ (1 + S - 2√S, 1 + S + 2√S)
                i.e., Δt²A ∈ (2 + ΔtG - 2√(1+ΔtG), 2 + ΔtG + 2√(1+ΔtG))
        """
        dt = torch.sigmoid(dt_raw)
        # Use softplus (not ReLU) to ensure G > 0 strictly.
        # G = 0 gives det(M) = 1/S = 1/(1+0) = 1 → eigenvalues ON unit circle
        # G > 0 gives det(M) = 1/S < 1 → eigenvalues strictly INSIDE
        G = F.softplus(G)

        dtG = dt * G
        sqrt_term = torch.sqrt(1.0 + dtG)
        dt2 = dt.pow(2).clamp(min=1e-6)

        A_low = (2.0 + dtG - 2.0 * sqrt_term) / dt2
        A_high = (2.0 + dtG + 2.0 * sqrt_term) / dt2

        # Soft clamping via ReLU: clip(A, A_low, A_high)
        A = A_low + F.relu(A - A_low) - F.relu(A - A_high)

        return A, G, dt

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence through D-LinOSS.

        Args:
            input_sequence: (L, H) float tensor — L timesteps, H features

        Returns:
            output: (L, H) float tensor — transformed sequence

        Pipeline:
            1. Materialize B, C as complex tensors
            2. Project A, G, dt to stable region
            3. Compute Bu = B @ u for each timestep  → (L, P) complex
            4. Construct M elements and F elements
            5. Parallel scan to compute all x_k
            6. Output: y_k = Re(C @ x_k) + D · u_k
        """
        L, H = input_sequence.shape
        P = self.state_dim

        # 1. Extract real and imaginary parts of B and C
        # self.B: (P, H, 2), self.C: (H, P, 2)
        B_re = self.B[..., 0]
        B_im = self.B[..., 1]
        C_re = self.C[..., 0]
        C_im = self.C[..., 1]

        # 2. Soft projection for stability
        A, G, dt = self._soft_project(self.A_diag, self.G_diag, self.dt_raw)

        # 3. Input projection: Bu_k = B @ u_k
        # Since input is real: Bu_re = B_re @ u, Bu_im = B_im @ u
        Bu_re = input_sequence @ B_re.T
        Bu_im = input_sequence @ B_im.T

        # 4. Construct recurrence elements
        S = 1.0 + dt * G
        M11 = 1.0 / S
        M12 = -dt * A / S
        M21 = dt / S
        M22 = 1.0 - dt.pow(2) * A / S

        M = torch.cat([M11, M12, M21, M22])
        M_elements = M.unsqueeze(0).repeat(L, 1)

        # Forcing terms (Real and Imaginary treated separately)
        dt_S = dt / S
        dt2_S = dt.pow(2) / S
        
        F_re = torch.cat([dt_S.unsqueeze(0) * Bu_re, dt2_S.unsqueeze(0) * Bu_re], dim=-1)
        F_im = torch.cat([dt_S.unsqueeze(0) * Bu_im, dt2_S.unsqueeze(0) * Bu_im], dim=-1)

        # 5. Parallel scan (all real arithmetic)
        _, xs_re = parallel_scan(M_elements, F_re)
        _, xs_im = parallel_scan(M_elements, F_im)

        # Extract position component (x): indices P:2P
        ys_re = xs_re[:, P:]
        ys_im = xs_im[:, P:]

        # 6. Output projection: y = Re(C @ (ys_re + i*ys_im)) + D*u
        # Re((C_re + i*C_im) @ (ys_re + i*ys_im)) = C_re @ ys_re - C_im @ ys_im
        output = (ys_re @ C_re.T - ys_im @ C_im.T) + self.D.unsqueeze(0) * input_sequence

        return output

    def get_eigenvalues(self) -> torch.Tensor:
        """
        Compute eigenvalues of the discretized recurrence matrix M.

        For each oscillator i:
            tr(M_i) = 1/S_i + 1 - Δt²·a_i/S_i
            det(M_i) = 1/S_i
            λ± = (tr ± √(tr² - 4·det)) / 2

        Returns:
            eigenvalues: (P, 2) complex tensor — conjugate pairs
        """
        with torch.no_grad():
            A, G, dt = self._soft_project(self.A_diag, self.G_diag, self.dt_raw)
            S = 1.0 + dt * G
            tr = 1.0 / S + 1.0 - dt.pow(2) * A / S
            det = 1.0 / S

            disc = tr.pow(2) - 4 * det
            
            # Complex64 operations deadlock the DirectML backend graph.
            # We explicitly bypass this by computing the roots natively on CPU.
            disc = disc.cpu()
            tr_c = tr.cpu().to(torch.complex64)
            sqrt_disc = torch.sqrt(disc.to(torch.complex64))

            lam_plus = (tr_c + sqrt_disc) / 2
            lam_minus = (tr_c - sqrt_disc) / 2

            return torch.stack([lam_plus, lam_minus], dim=-1).to(tr.device)

    def get_spectral_radius(self) -> float:
        """Maximum |λ| across all oscillators. Should be < 1 for stability."""
        eigs = self.get_eigenvalues()
        return eigs.abs().max().item()


# ============================================================================
#  D-LinOSS Block (with BatchNorm, GELU, Dropout, GLU, Residual)
# ============================================================================

class DLinOSSBlock(nn.Module):
    """
    One D-LinOSS block:
        x → LayerNorm → SSM → GELU → Dropout → GLU → Dropout → (+skip)

    This matches the reference LinOSSBlock exactly.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        drop_rate: float = 0.1,
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.01,
        theta_max: float = math.pi,
        dt_std: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ssm = DLinOSSLayer(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            initialization=initialization,
            r_min=r_min, r_max=r_max,
            theta_min=theta_min, theta_max=theta_max,
            dt_std=dt_std,
        )
        self.glu = GLU(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (L, H) input sequence
        Returns:
            (L, H) output with residual connection
        """
        skip = x
        x = self.norm(x)
        x = self.ssm(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.glu(x)
        x = self.drop(x)
        return skip + x


# ============================================================================
#  Full D-LinOSS Model
# ============================================================================

class DLinOSSModel(nn.Module):
    """
    Full D-LinOSS model for sequence-to-sequence or classification.

    Architecture:
        Linear Encoder → [DLinOSSBlock × N] → Linear Decoder

    For classification: output = softmax(mean(hidden, dim=0))
    For regression: output = decoder(hidden[::output_step])
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_blocks: int = 4,
        classification: bool = False,
        output_step: int = 1,
        drop_rate: float = 0.1,
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.01,
        theta_max: float = math.pi,
        dt_std: float = 0.1,
    ):
        super().__init__()
        self.classification = classification
        self.output_step = output_step

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DLinOSSBlock(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                drop_rate=drop_rate,
                initialization=initialization,
                r_min=r_min, r_max=r_max,
                theta_min=theta_min, theta_max=theta_max,
                dt_std=dt_std,
            )
            for _ in range(num_blocks)
        ])
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (L, input_dim) or (B, L, input_dim) input sequence

        Returns:
            For classification: (output_dim,) or (B, output_dim) probabilities
            For regression: (L', output_dim) or (B, L', output_dim) predictions
        """
        batched = x.dim() == 3
        if batched:
            # Process each batch element
            return torch.stack([self._forward_single(x[i]) for i in range(x.shape[0])])
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process single (L, input_dim) sequence."""
        x = self.encoder(x)  # (L, hidden_dim)

        for block in self.blocks:
            x = block(x)

        if self.classification:
            x = x.mean(dim=0)
            x = self.decoder(x)
            x = F.softmax(x, dim=-1)
        else:
            x = x[self.output_step - 1::self.output_step]
            x = self.decoder(x)

        return x

    def get_all_eigenvalues(self):
        """Get eigenvalues from all blocks for spectral analysis."""
        eigs = []
        for i, block in enumerate(self.blocks):
            block_eigs = block.ssm.get_eigenvalues()
            eigs.append(block_eigs)
        return eigs

    def spectral_summary(self) -> dict:
        """
        Summarize the spectral properties of the model.

        Returns dict with:
            - max_spectral_radius: max |λ| across all blocks (should be < 1)
            - mean_magnitude: average |λ|
            - mean_frequency: average arg(λ)/π (0 = DC, 1 = Nyquist)
        """
        all_eigs = self.get_all_eigenvalues()
        all_flat = torch.cat([e.flatten() for e in all_eigs])

        return {
            'max_spectral_radius': all_flat.abs().max().item(),
            'mean_magnitude': all_flat.abs().mean().item(),
            'mean_frequency': (all_flat.angle().abs() / math.pi).mean().item(),
            'n_oscillators_per_block': self.blocks[0].ssm.state_dim,
            'n_blocks': len(self.blocks),
        }
