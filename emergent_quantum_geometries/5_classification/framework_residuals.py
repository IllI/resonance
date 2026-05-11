"""
framework_residuals.py
Direct comparison of D-LiNOSS learned parameters (ω_k, γ_k)
against known framework signatures.

This replaces the NQS meta-learner's black-box classification
with an interpretable, physics-grounded residual score.

Usage
-----
from framework_residuals import FrameworkComparator, extract_dlinoss_params

# After D-LiNOSS has run:
params = extract_dlinoss_params(damper_net, variables, dt=0.1)
comp   = FrameworkComparator(dt=0.1, beta=5.0)
scores = comp.score_all(params)
comp.report(scores)
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple

# ═══════════════════════════════════════════════════════════════════
#  DATA CLASS:  learned D-LiNOSS parameters
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DLinOSSParams:
    """
    Learned parameters extracted from a trained D-LiNOSS instance.

    omega_k : np.ndarray  shape (K,)  — oscillation frequencies [rad / time_unit]
    gamma_k : np.ndarray  shape (K,)  — damping rates [1 / time_unit]
    amp_k   : np.ndarray  shape (K,)  — mode amplitudes (from C matrix)
    dt      : float                   — physical timestep used during training
    K       : int                     — number of modes
    """
    omega_k : np.ndarray
    gamma_k : np.ndarray
    amp_k   : np.ndarray
    dt      : float

    @property
    def K(self):
        return len(self.omega_k)

    @property
    def quality_factors(self):
        """Q = ω/γ — high Q = weakly damped, low Q = overdamped."""
        return self.omega_k / (self.gamma_k + 1e-12)


# ═══════════════════════════════════════════════════════════════════
#  EXTRACTOR:  pull (ω_k, γ_k) out of a trained DLinOSSQuantumDamper
# ═══════════════════════════════════════════════════════════════════

def extract_dlinoss_params(damper_net, variables, dt: float) -> DLinOSSParams:
    """
    Extract the learned diagonal SSM eigenvalues from DLinOSSQuantumDamper.

    The DLinOSS A matrix has complex diagonal entries:
        λ_k = exp(i ω_k dt - γ_k dt)

    So:
        ω_k = angle(λ_k) / dt
        γ_k = -log(|λ_k|) / dt

    Parameters
    ----------
    damper_net  : DLinOSSQuantumDamper instance
    variables   : Flax variable dict from damper_net.init(...)
    dt          : physical timestep

    Returns
    -------
    DLinOSSParams
    """
    # Navigate Flax param tree to find the diagonal A (lambda) parameter.
    # Adjust the key path if your DLinOSS uses a different naming convention.
    params = variables.get("params", variables)

    # Try common naming patterns
    lambda_candidates = [
        params.get("lambda_re"),         # separate real/imag
        params.get("Lambda"),
        params.get("A_diag"),
        _deep_get(params, ["dlinoss", "lambda"]),
        _deep_get(params, ["layers_0", "lambda_re"]),
    ]
    lambda_raw = next((x for x in lambda_candidates if x is not None), None)

    if lambda_raw is None:
        # Fallback: extract from the full damping matrix via SVD
        # shape (batch, T, 32, 32) → average over batch/time, then SVD
        raise ValueError(
            "Cannot locate lambda parameter in DLinOSS variable tree.\n"
            "Keys found: " + str(list(params.keys())) + "\n"
            "Adjust extract_dlinoss_params() key path to match your model."
        )

    lam = np.array(lambda_raw)

    # Handle separate real/imag storage
    if "lambda_im" in params:
        lam = lam + 1j * np.array(params["lambda_im"])

    # If stored as angle + log_mag separately
    if lam.dtype in (np.float32, np.float64, jnp.float32):
        # Assume stored as (angle, log_magnitude) pairs interleaved
        K     = lam.size // 2
        theta = lam[:K]
        logr  = lam[K:]
        lam   = np.exp(logr + 1j * theta)

    # Now extract ω and γ
    omega_k = np.angle(lam) / dt          # rad / time_unit
    gamma_k = -np.log(np.abs(lam)) / dt   # 1 / time_unit
    gamma_k = np.maximum(gamma_k, 0.0)    # numerical safety

    # Amplitude proxy: magnitude of C (output coupling)
    C = params.get("C", params.get("output_matrix", np.ones_like(omega_k)))
    amp_k = np.abs(np.array(C)).ravel()[: len(omega_k)]

    return DLinOSSParams(omega_k=omega_k, gamma_k=gamma_k, amp_k=amp_k, dt=dt)


def extract_from_damping_matrix(damping_matrix: np.ndarray, dt: float) -> DLinOSSParams:
    """
    Alternative extractor when you only have the output damping_matrix
    (shape: batch, T, 32, 32) rather than the raw Flax params.

    Strategy: average over batch & time, then take the eigendecomposition.
    The eigenvalues are the effective (ω_k, γ_k) modes in the data.
    """
    # Average over batch dim → (32, 32)
    if damping_matrix.ndim == 3:
        M = np.mean(np.array(damping_matrix), axis=0)
    elif damping_matrix.ndim == 4:
        M = np.mean(np.array(damping_matrix), axis=(0, 1))
    else:
        M = np.array(damping_matrix)

    # Symmetrise (Lindbladian damping matrices should be Hermitian-ish)
    M_sym = 0.5 * (M + M.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M_sym)    # real eigenvalues for sym matrix

    # Map eigenvalues to (ω, γ): treat as oscillator decay rates
    # Negative eigenvalues → damping; positive → amplification (unphysical, clip)
    gamma_k = np.maximum(-eigvals, 0.0)
    # Frequencies: use index-proportional proxy (no phase info from sym matrix)
    K       = len(gamma_k)
    omega_k = np.pi * np.arange(1, K + 1) / (K + 1)
    amp_k   = np.abs(eigvecs).mean(axis=0)

    return DLinOSSParams(omega_k=omega_k, gamma_k=gamma_k, amp_k=amp_k, dt=dt)


# ═══════════════════════════════════════════════════════════════════
#  FRAMEWORK SIGNATURES
#  Each function returns the PREDICTED (ω_k, γ_k) for that framework
#  given the physical parameters of the experiment.
# ═══════════════════════════════════════════════════════════════════

def signature_heisenberg(K: int, dt: float, J: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unitary XXZ / Heisenberg spin chain.
    ω_k = 2J |cos(π k / (K+1))|   (tight-binding band)
    γ_k = 0  (no damping — unitary evolution)
    """
    k       = np.arange(1, K + 1)
    omega_k = 2 * J * np.abs(np.cos(np.pi * k / (K + 1)))
    gamma_k = np.zeros(K)
    return omega_k, gamma_k


def signature_lindblad(K: int, dt: float, J: float = 1.0, gamma: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lindblad thermal bath with uniform dephasing.
    ω_k = Heisenberg dispersion (bath doesn't shift frequencies)
    γ_k = γ  (uniform across all modes)
    """
    omega_k, _ = signature_heisenberg(K, dt, J)
    gamma_k    = gamma * np.ones(K)
    return omega_k, gamma_k


def signature_penrose_or(K: int, dt: float, J: float = 1.0, E_G_scale: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Penrose Objective Reduction.
    ω_k = same as Heisenberg (geometry of spacetime, not Hamiltonian)
    γ_k = E_G,k / ħ  — non-uniform, follows gravitational self-energy
          of each mode. Modeled as exponential distribution with
          scale E_G_scale (in units of J, the coupling energy).

    Key distinguishing feature: γ_k is HETEROGENEOUS and correlated
    with mode mass/energy, not uniform like Lindblad.
    """
    rng     = np.random.default_rng(0)
    omega_k, _ = signature_heisenberg(K, dt, J)
    # E_G,k ∝ m_k^2 / separation ∝ mode_energy^2 — exponential proxy
    gamma_k = rng.exponential(scale=E_G_scale, size=K)
    # Sort descending (high-energy modes collapse faster)
    gamma_k = np.sort(gamma_k)[::-1]
    return omega_k, gamma_k


def signature_syk(K: int, dt: float, beta: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    SYK holographic scrambler.
    ω_k ≈ flat (SYK has a flat spectral density at low energy)
    γ_k = λ_L = 2π / β  for ALL modes  (maximal chaos, universal rate)

    This is the most distinctive signature: all γ_k are equal AND
    equal to the Lyapunov exponent 2π k_B T / ħ.
    """
    k       = np.arange(1, K + 1)
    omega_k = np.pi * k / (K + 1)      # roughly flat density of states
    lam_L   = 2 * np.pi / beta         # Lyapunov exponent
    gamma_k = lam_L * np.ones(K)       # all modes decay at Lyapunov rate
    return omega_k, gamma_k


def signature_mbl(K: int, dt: float, J: float = 1.0, W: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Many-Body Localized phase.
    ω_k = disordered (random frequencies, disorder width W)
    γ_k ≈ 0 (no thermalization, no damping)

    Key feature: near-zero γ_k with disordered ω_k.
    OTOC grows logarithmically due to slow dephasing.
    """
    rng     = np.random.default_rng(1)
    omega_k = rng.uniform(J - W/2, J + W/2, K)   # disordered frequencies
    gamma_k = rng.exponential(0.01, K)             # near-zero damping
    return omega_k, gamma_k


# ═══════════════════════════════════════════════════════════════════
#  COMPARATOR
# ═══════════════════════════════════════════════════════════════════

class FrameworkComparator:
    """
    Scores a learned DLinOSSParams against each known framework signature.

    Residual = weighted L2 distance in (ω, γ) parameter space.
    Lower residual = better match.

    The comparison is done in a normalised space so that ω and γ
    contribute equally regardless of their absolute magnitudes.
    """

    FRAMEWORKS = ["heisenberg", "lindblad", "penrose_or", "syk", "mbl"]

    def __init__(self, dt: float = 0.1, J: float = 1.0, beta: float = 5.0,
                 gamma_lindblad: float = 0.3, E_G_scale: float = 0.2,
                 disorder_W: float = 5.0):
        self.dt             = dt
        self.J              = J
        self.beta           = beta
        self.gamma_lindblad = gamma_lindblad
        self.E_G_scale      = E_G_scale
        self.disorder_W     = disorder_W

        # Physical interpretations for the report
        self._descriptions = {
            "heisenberg": "Unitary evolution, no decoherence. γ_k = 0.",
            "lindblad":   f"Thermal Lindblad bath. γ_k = {gamma_lindblad:.2f} (uniform).",
            "penrose_or": "Penrose OR. γ_k non-uniform ~ E_G,k/ħ (exponential dist).",
            "syk":        f"SYK holographic chaos. γ_k = λ_L = 2π/β = {2*np.pi/beta:.3f}.",
            "mbl":        "Many-body localized. γ_k ≈ 0, ω_k disordered.",
        }

    def _get_signature(self, name: str, K: int) -> Tuple[np.ndarray, np.ndarray]:
        sigs = {
            "heisenberg": lambda: signature_heisenberg(K, self.dt, self.J),
            "lindblad":   lambda: signature_lindblad(K, self.dt, self.J, self.gamma_lindblad),
            "penrose_or": lambda: signature_penrose_or(K, self.dt, self.J, self.E_G_scale),
            "syk":        lambda: signature_syk(K, self.dt, self.beta),
            "mbl":        lambda: signature_mbl(K, self.dt, self.J, self.disorder_W),
        }
        return sigs[name]()

    def score_all(self, params: DLinOSSParams, omega_weight: float = 0.4) -> Dict[str, float]:
        """
        Compute residual scores for all frameworks.

        Returns dict  {framework_name: residual_score}
        Lower = better match. Scores are normalised to [0, 1] range.

        omega_weight controls how much frequency vs damping rate matters.
        Default 0.4/0.6 upweights γ because framework discrimination
        is primarily in the damping structure, not frequency.
        """
        K = params.K
        scores = {}

        # Normalise learned params once
        omega_scale = np.std(params.omega_k) + 1e-8
        gamma_scale = np.std(params.gamma_k) + 1e-8

        omega_learned = params.omega_k / omega_scale
        gamma_learned = params.gamma_k / gamma_scale

        for name in self.FRAMEWORKS:
            omega_pred, gamma_pred = self._get_signature(name, K)

            # Sort both by ω so modes align
            learned_order  = np.argsort(params.omega_k)
            pred_order     = np.argsort(omega_pred)

            o_l = omega_learned[learned_order]
            g_l = gamma_learned[learned_order]
            o_p = (omega_pred[pred_order]  / omega_scale)
            g_p = (gamma_pred[pred_order]  / gamma_scale)

            omega_resid = np.mean((o_l - o_p) ** 2)
            gamma_resid = np.mean((g_l - g_p) ** 2)

            scores[name] = (
                omega_weight * omega_resid
                + (1 - omega_weight) * gamma_resid
            )

        # Normalise to [0, 1]
        max_s = max(scores.values()) + 1e-12
        return {k: v / max_s for k, v in scores.items()}

    def probability_scores(self, params: DLinOSSParams) -> Dict[str, float]:
        """
        Convert residuals to pseudo-probabilities via softmax of -residual.
        Directly comparable to the NQS meta-learner's output probabilities.
        """
        residuals = self.score_all(params)
        neg_res   = np.array([-v for v in residuals.values()])
        exp_r     = np.exp(neg_res - neg_res.max())   # numerically stable softmax
        probs     = exp_r / exp_r.sum()
        return dict(zip(residuals.keys(), probs))

    def entropy(self, params: DLinOSSParams) -> float:
        """Shannon entropy of the probability scores. 0 = certain, ln(5) ≈ 1.6 = uniform."""
        probs = list(self.probability_scores(params).values())
        return -sum(p * np.log(p + 1e-12) for p in probs)

    def winner(self, params: DLinOSSParams) -> str:
        """Return the framework with the highest probability."""
        probs = self.probability_scores(params)
        return max(probs, key=probs.get)

    def report(self, params: DLinOSSParams):
        """Print a human-readable comparison report."""
        probs = self.probability_scores(params)
        H     = self.entropy(params)
        win   = self.winner(params)

        print("=" * 56)
        print("  FRAMEWORK RESIDUAL COMPARATOR  —  D-LiNOSS Report")
        print("=" * 56)
        print(f"  Modes (K): {params.K}    dt: {params.dt}    "
              f"Q_mean: {np.mean(params.quality_factors):.2f}")
        print(f"  Learned γ range: [{params.gamma_k.min():.4f}, {params.gamma_k.max():.4f}]")
        print(f"  Learned ω range: [{params.omega_k.min():.4f}, {params.omega_k.max():.4f}]")
        print("-" * 56)
        print(f"  {'Framework':<16}  {'P(match)':>9}  Description")
        print("-" * 56)
        for name in sorted(probs, key=probs.get, reverse=True):
            marker = "  ◄ WINNER" if name == win else ""
            print(f"  {name:<16}  {probs[name]:>8.2%}  "
                  f"{self._descriptions[name][:34]}{marker}")
        print("-" * 56)
        print(f"  Decision entropy: {H:.4f}  "
              f"(max = {np.log(len(probs)):.4f} for {len(probs)} frameworks)")
        print(f"  Confidence:       {(1 - H / np.log(len(probs))):.1%}")
        print("=" * 56)

        # Specific interpretive notes
        print()
        if win == "heisenberg":
            print("  ▸ Data consistent with closed, unitary quantum system.")
            print("    All modes undamped — no bath, no collapse.")
        elif win == "lindblad":
            print("  ▸ Data consistent with standard thermal decoherence.")
            print("    Uniform γ_k suggests Markovian bath coupling.")
        elif win == "penrose_or":
            print("  ▸ Data consistent with Penrose OR signature.")
            print("    Heterogeneous γ_k matching E_G/ħ distribution.")
            print("    Check OTOC for kink at collapse time τ_OR.")
        elif win == "syk":
            print("  ▸ Data consistent with maximal holographic chaos.")
            print(f"    γ_k ≈ 2π/β = {2*np.pi/self.beta:.3f} — saturates chaos bound.")
            print("    Expected OTOC: F(t) ~ exp(-λ_L t).")
        elif win == "mbl":
            print("  ▸ Data consistent with Many-Body Localized phase.")
            print("    Near-zero γ_k, disordered ω_k — no thermalization.")
            print("    Expected OTOC: logarithmic growth, not exponential.")
        print()


# ═══════════════════════════════════════════════════════════════════
#  UPDATED run_v2 INTEGRATION  (drop-in for Phase 5)
# ═══════════════════════════════════════════════════════════════════

def run_framework_comparison(damper_net, variables, damping_matrix, dt: float = 0.1,
                              beta: float = 5.0):
    """
    Full replacement for the NQS meta-learner classification step.

    Parameters
    ----------
    damper_net     : DLinOSSQuantumDamper  (Flax module)
    variables      : Flax variable dict from damper_net.init(...)
    damping_matrix : output of damper_net.apply(...)  shape (B, T, 32, 32)
    dt             : physical timestep
    beta           : inverse temperature (for SYK signature)

    Returns
    -------
    dict with keys:
        probabilities   : {framework: float}  (comparable to current output)
        winner          : str
        entropy         : float
        params          : DLinOSSParams
    """
    # Try to extract from Flax params first; fall back to matrix SVD
    try:
        params = extract_dlinoss_params(damper_net, variables, dt)
    except ValueError:
        print("[WARNING] Falling back to damping matrix SVD extraction")
        params = extract_from_damping_matrix(np.array(damping_matrix), dt)

    comp   = FrameworkComparator(dt=dt, beta=beta)
    probs  = comp.probability_scores(params)
    H      = comp.entropy(params)
    win    = comp.winner(params)

    comp.report(params)

    # Return in same format as current meta_net.apply(...)
    return {
        "probabilities": probs,
        "winner":        win,
        "entropy":       H,
        "params":        params,
    }


# ═══════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════

def _deep_get(d, keys):
    """Navigate nested dict; return None if any key is missing."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d
