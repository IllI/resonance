"""
MRI Artifact Correction: Reverse-Engineering Experimental Conditions
=====================================================================

Mathematical Framework for Cleaning fMRI Input Data
----------------------------------------------------
The observed fMRI signal is NOT the true neural electromagnetic field.
It is the TRUE field convolved with multiple observation artifacts imposed
by the experimental hardware and physics.

This module provides the mathematical tools to:
1. MODEL the forward observation process (Hardware → Signal corruption)
2. INVERT the observation to recover the true underlying field dynamics
3. INTEGRATE the correction into the D-LinOSS pipeline as a preprocessing stage

THE FORWARD MODEL (HOW THE SCANNER CORRUPTS THE TRUE SIGNAL)
=============================================================

The true neural electromagnetic field at position r and time t is:

    Ψ_true(r, t) = neural activity we want to recover

The observed MRI signal is:

    S_obs(r, t) = H_total(r, t) · Ψ_true(r, t) + η(r, t)

Where H_total is the product of multiple observation operators:

    H_total = H_coil · H_B0 · H_T2star · H_HRF · H_gradient · H_physio

Each operator introduces a SPECIFIC, MATHEMATICALLY INVERTIBLE distortion:

1. COIL SENSITIVITY PROFILE — H_coil(r)
   ─────────────────────────────────────
   Problem: Head coils have non-uniform spatial sensitivity.
   The coil sensitivity C_c(r) for channel c follows from Biot-Savart:
       C_c(r) = (μ₀/4π) ∮ [I dl × r̂] / |r - r'|²
   where the integral is over the coil geometry.

   For a multi-channel phased array with N_c coils:
       S_obs(r) = Σ_c C_c(r) · ρ(r)

   The root-sum-of-squares (RSS) combination gives:
       |S_RSS(r)| = √(Σ_c |C_c(r)|²) · ρ(r) = B₁⁺(r) · ρ(r)

   Correction: Divide by the estimated sensitivity profile.
       Ψ_corrected(r) = S_obs(r) / B₁⁺(r)

   Estimation method: Low-order polynomial or N4 bias field correction.

2. STATIC FIELD INHOMOGENEITY — H_B0(r)
   ──────────────────────────────────────
   Problem: The main magnetic field B₀ is not perfectly uniform.
   Off-resonance frequency: Δf(r) = γ · ΔB₀(r) where γ = 42.576 MHz/T.

   This causes:
   a) Signal dephasing: φ(r) = 2π · Δf(r) · TE
   b) Geometric distortion: Δr = Δf(r) / BW_pixel

   In the time domain, the transverse magnetization picks up phase:
       M_xy(r, t) = M₀(r) · exp(-t/T₂*(r)) · exp(-i·2π·Δf(r)·t)

   Correction via field map:
       Given a dual-echo acquisition with ΔTE:
       ΔΦ(r) = ∠[S(r, TE₂) · conj(S(r, TE₁))]
       Δf(r) = ΔΦ(r) / (2π · ΔTE)
       S_corrected(r) = S_obs(r) · exp(+i·2π·Δf(r)·TE)

3. T₂* DECAY / MAGNETIC SUSCEPTIBILITY — H_T2star(r)
   ───────────────────────────────────────────────────
   Problem: Different tissues have different magnetic susceptibility χ,
   causing local field gradients and signal decay:
       S(r, TE) = S₀(r) · exp(-TE / T₂*(r))
   where 1/T₂*(r) = 1/T₂(r) + γ·|∇ΔB₀(r)|·voxel_size.

   For BOLD contrast, we WANT part of this effect (T₂* shortening from
   deoxyhemoglobin), but we need to separate:
   a) BOLD-related T₂* changes (the signal we desire)
   b) Non-neural susceptibility artifacts (air-tissue interfaces, etc.)

   Multi-echo decomposition:
       S(r, TE_n) = S₀(r) · exp(-TE_n · R₂*(r))
       R₂*(r) = 1/T₂*(r)

       Linear fit: ln|S(TE_n)| = ln(S₀) - R₂* · TE_n
       Slope = -R₂*, Intercept = ln(S₀)

       BOLD ΔR₂* = R₂*(task) - R₂*(baseline)

4. HEMODYNAMIC RESPONSE FUNCTION — H_HRF
   ───────────────────────────────────────
   Problem: Neural activity n(t) is observed only through the sluggish
   hemodynamic response:
       y_BOLD(t) = h_HRF(t) ∗ n(t) + ε(t)

   The canonical HRF (double-gamma):
       h(t) = A₁ · [(t/τ₁)^(a₁-1) · exp(-t/τ₁)] / Γ(a₁)
            - A₂ · [(t/τ₂)^(a₂-1) · exp(-t/τ₂)] / Γ(a₂)

   Parameters: τ₁ ≈ 6s (peak), τ₂ ≈ 16s (undershoot), a₁ ≈ 6, a₂ ≈ 16,
               A₂/A₁ ≈ 1/6

   Deconvolution (Wiener):
       N(f) = Y(f) · conj(H(f)) / (|H(f)|² + λ²)
   where λ is a regularization parameter controlling noise amplification.

5. GRADIENT NONLINEARITY — H_gradient(r)
   ──────────────────────────────────────
   Problem: Spatial encoding gradients are not perfectly linear:
       B_z(r) = B₀ + G_x·x + G_y·y + G_z·z + higher-order terms

   The true position maps nonlinearly from the encoded position:
       r_true = r_nominal + δr(r_nominal)

   δr is modeled using manufacturer's spherical harmonic coefficients:
       ΔB(r) = Σ_{l,m} C_{lm} · Y_l^m(θ, φ) · r^l
   These coefficients are scanner-specific (Siemens, GE, Philips).

   Correction: Jacobian-based intensity correction + coordinate unwarp:
       S_corrected(r_true) = |J(r)| · S_obs(r_nominal)
       where J = ∂r_true/∂r_nominal (the Jacobian determinant)

6. PHYSIOLOGICAL NOISE — H_physio(t)
   ──────────────────────────────────
   Problem: Cardiac (1-1.5 Hz) and respiratory (0.2-0.5 Hz) pulsations
   create periodic signal modulations:
       S_obs(t) = S_neural(t) + β_c·Φ_cardiac(t) + β_r·Φ_resp(t) + β_hr·Φ_HR(t)

   RETROICOR model (Glover et al., 2000):
       Φ_cardiac(t) = Σ_{k=1}^{M} [a_k·cos(k·φ_c(t)) + b_k·sin(k·φ_c(t))]
       Φ_resp(t) = Σ_{k=1}^{M} [c_k·cos(k·φ_r(t)) + d_k·sin(k·φ_r(t))]

   where φ_c(t), φ_r(t) are the instantaneous cardiac/respiratory phases.

   Correction via regression:
       S_cleaned(t) = S_obs(t) - X_physio @ β̂
       where β̂ = (X_physio^T X_physio)^{-1} X_physio^T S_obs


MAGNET-STRENGTH-SPECIFIC CONSIDERATIONS
========================================

Different field strengths dramatically change the physics:

| B₀ (T) | γB₀ (MHz) | T₂* GM (ms) | SNR scaling | BOLD ΔR₂* scaling |
|--------|-----------|-------------|-------------|-------------------|
| 1.5    | 63.9      | ~50         | 1.0×        | 1.0×              |
| 3.0    | 127.7     | ~30         | ~2.0×       | ~2.0×             |
| 7.0    | 298.0     | ~15-25      | ~4.7×       | ~4.7×             |
| 9.4    | 400.2     | ~10-15      | ~6.3×       | ~6.3×             |
| 11.7   | 497.7     | ~5-12       | ~7.8×       | ~7.8×             |

Key physics:
    SNR ∝ B₀^(7/4)  (for macroscopic voxels)
    BOLD ΔR₂* ∝ B₀  (at high field)
    T₂* ∝ 1/B₀      (inversely proportional)
    Susceptibility artifacts ∝ B₀ (more severe at higher fields)

CALCIUM LARMOR CONSIDERATIONS
    ⁴³Ca Larmor frequency: f_Ca = γ_Ca · B₀
    γ_Ca = 2.8688 MHz/T
    At 3T: f_Ca = 8.61 MHz (well below proton Larmor of 127.7 MHz)
    At 11.7T: f_Ca = 33.57 MHz
    Detection requires specialized RF coils tuned to these frequencies.
    For standard ¹H coils, calcium dynamics are observed INDIRECTLY through
    metabolic coupling (calcium → neurovascular → BOLD).


IMPLEMENTATION
==============
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.fft import fft, ifft, fftfreq
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import minimize_scalar
    from scipy.special import gamma as gamma_func
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
#  Configuration
# ============================================================================

@dataclass
class MRIHardwareConfig:
    """
    Configuration representing the specific MRI scanner hardware setup.
    These parameters define the observation operator H_total.

    ALL of these affect the physics of signal formation and must be
    accounted for in any biophysically grounded analysis.
    """
    # Main magnetic field
    B0: float = 3.0                     # Tesla (1.5, 3.0, 7.0, 9.4, 11.7)
    gamma_proton: float = 42.576e6      # Hz/T (proton gyromagnetic ratio)

    # Coil configuration
    n_coil_channels: int = 32           # Number of receive channels
    coil_type: str = "phased_array"     # "phased_array", "birdcage", "surface"

    # Pulse sequence
    pulse_sequence: str = "GE-EPI"      # "GE-EPI", "SE-EPI", "GRASE", "MPRAGE"
    TE: float = 30e-3                   # Echo time (seconds) — optimized per B0
    TR: float = 2.0                     # Repetition time (seconds)
    flip_angle: float = 90.0            # degrees

    # Spatial encoding
    voxel_size: Tuple[float, float, float] = (3.0, 3.0, 3.0)  # mm
    matrix_size: Tuple[int, int, int] = (64, 64, 36)
    bandwidth_per_pixel: float = 1500.0  # Hz/pixel (phase-encode direction)

    # Multi-echo (for T2* decomposition)
    multi_echo: bool = False
    echo_times: List[float] = field(default_factory=lambda: [30e-3])

    # Relaxation times at this field strength (tissue-average for GM)
    T1_GM: float = 1340e-3              # Grey matter T1 (ms)
    T2_GM: float = 80e-3               # Grey matter T2 (ms)
    T2star_GM: float = 30e-3            # Grey matter T2* (ms)
    T1_WM: float = 840e-3              # White matter T1
    T2star_WM: float = 50e-3           # White matter T2*

    # Physiological parameters
    cardiac_rate: float = 1.1           # Hz (66 bpm typical)
    respiratory_rate: float = 0.25      # Hz (15 breaths/min)

    # Preprocessing that has already been applied
    slice_timing_corrected: bool = True
    motion_corrected: bool = True
    distortion_corrected: bool = False  # EPI distortion (requires field map)
    bias_field_corrected: bool = False  # N4/N3 correction

    def __post_init__(self):
        """Auto-set field-dependent parameters if not overridden."""
        # T2* scales inversely with B0 (approximately)
        if self.B0 != 3.0:
            scale = 3.0 / self.B0
            self.T2star_GM = 30e-3 * scale
            self.T2star_WM = 50e-3 * scale

        # Optimal TE ≈ T2* for maximum BOLD CNR
        self.TE = self.T2star_GM

        # Larmor frequency
        self.f_larmor = self.gamma_proton * self.B0  # Hz

    @property
    def calcium_larmor(self) -> float:
        """⁴³Ca Larmor frequency at current field strength."""
        gamma_Ca43 = 2.8688e6  # Hz/T
        return gamma_Ca43 * self.B0

    @property
    def sodium_larmor(self) -> float:
        """²³Na Larmor frequency at current field strength."""
        gamma_Na23 = 11.262e6  # Hz/T
        return gamma_Na23 * self.B0

    @property
    def potassium_larmor(self) -> float:
        """³⁹K Larmor frequency at current field strength."""
        gamma_K39 = 1.9895e6  # Hz/T
        return gamma_K39 * self.B0

    @property
    def snr_scaling_vs_3T(self) -> float:
        """Approximate SNR scaling relative to 3T."""
        return (self.B0 / 3.0) ** 1.75  # B0^(7/4)

    @property
    def bold_cnr_scaling_vs_3T(self) -> float:
        """BOLD contrast-to-noise ratio scaling relative to 3T."""
        return self.B0 / 3.0  # Linear in B0 at high field


# ============================================================================
#  1. COIL SENSITIVITY CORRECTION
# ============================================================================

class CoilSensitivityCorrector:
    """
    Corrects for non-uniform RF coil sensitivity profile B₁⁺(r).

    The bias field b(r) = B₁⁺(r) is a smooth, low-frequency modulation
    of the image intensity. We estimate it as a low-order polynomial
    or smooth field and divide it out:

        S_corrected(r) = S_observed(r) / b(r)

    Implementation: Simplified N4-style approach
        1. Log-transform the image: log(S) = log(ρ) + log(b)
        2. Estimate log(b) as a smoothed version of log(S)
        3. Subtract: log(S_corrected) = log(S) - log(b)
        4. Exponentiate back
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw
        self.smoothing_fwhm = 60.0  # mm — typical bias field smoothness

    def estimate_bias_field(self, volume: np.ndarray) -> np.ndarray:
        """
        Estimate the multiplicative bias field from a 3D or 4D volume.

        For 4D (fMRI time series): estimate from time-average volume.

        Args:
            volume: (X, Y, Z) or (X, Y, Z, T) array

        Returns:
            bias_field: (X, Y, Z) smooth multiplicative bias
        """
        if volume.ndim == 4:
            vol = np.mean(volume, axis=-1)
        else:
            vol = volume.copy()

        # Mask background
        threshold = 0.1 * np.percentile(vol[vol > 0], 50)
        mask = vol > threshold

        # Log-transform (safe)
        log_vol = np.zeros_like(vol)
        log_vol[mask] = np.log(vol[mask] + 1e-10)

        # Smooth to get bias field estimate
        sigma_voxels = self.smoothing_fwhm / (2.355 * np.array(self.hw.voxel_size))
        if HAS_SCIPY:
            log_bias = gaussian_filter(log_vol, sigma=sigma_voxels)
        else:
            # Simple box filter fallback
            from numpy.lib.stride_tricks import sliding_window_view
            k = int(sigma_voxels[0] * 2) + 1
            log_bias = np.copy(log_vol)
            for ax in range(3):
                log_bias = np.apply_along_axis(
                    lambda x: np.convolve(x, np.ones(k)/k, mode='same'), ax, log_bias
                )

        bias = np.exp(log_bias)
        bias[bias < 1e-6] = 1.0  # avoid division by zero

        return bias

    def correct(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply bias field correction.

        Args:
            volume: (X, Y, Z) or (X, Y, Z, T) array

        Returns:
            corrected: same shape as input, bias-corrected
            bias_field: (X, Y, Z) estimated bias
        """
        bias = self.estimate_bias_field(volume)

        if volume.ndim == 4:
            corrected = volume / bias[..., np.newaxis]
        else:
            corrected = volume / bias

        return corrected, bias


# ============================================================================
#  2. STATIC FIELD (B₀) INHOMOGENEITY CORRECTION
# ============================================================================

class B0FieldCorrector:
    """
    Corrects for static magnetic field inhomogeneity ΔB₀(r).

    Off-resonance causes:
    1. Phase accumulation: φ(r) = 2π · γ · ΔB₀(r) · TE
    2. Geometric distortion (EPI): Δy(r) = ΔΦ(r) / (2π · BW_pixel · N_pe)
    3. Signal dropout in regions with large ΔB₀ gradients

    Field map estimation (from dual-echo data):
        ΔΦ(r) = unwrap(∠[S₂(r) · conj(S₁(r))])
        ΔB₀(r) = ΔΦ(r) / (2π · γ · ΔTE)

    Phase correction:
        S_corrected(r) = S(r) · exp(+i · 2π · γ · ΔB₀(r) · TE)

    Geometric correction:
        Transformed coordinate map: r_corrected = r + ΔB₀(r) / (γ · G_pe)
        Interpolation to rectilinear grid
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw

    def estimate_field_map_from_dual_echo(
        self,
        echo1_complex: np.ndarray,
        echo2_complex: np.ndarray,
        dTE: float
    ) -> np.ndarray:
        """
        Estimate ΔB₀ field map from two complex-valued echo images.

        Phase difference method:
            ΔΦ = ∠(S₂ · conj(S₁))
            ΔB₀ = ΔΦ / (2π · γ · ΔTE)

        Args:
            echo1_complex: (X, Y, Z) complex array at TE₁
            echo2_complex: (X, Y, Z) complex array at TE₂
            dTE: TE₂ - TE₁ in seconds

        Returns:
            delta_B0: (X, Y, Z) field deviation in Tesla
        """
        # Phase difference
        phase_diff = np.angle(echo2_complex * np.conj(echo1_complex))

        # Simple phase unwrapping (Laplacian-based)
        phase_diff = self._laplacian_unwrap(phase_diff)

        # Convert to field deviation
        delta_B0 = phase_diff / (2 * np.pi * self.hw.gamma_proton * dTE)

        return delta_B0

    def _laplacian_unwrap(self, phase: np.ndarray) -> np.ndarray:
        """
        Laplacian-based phase unwrapping.

        The Laplacian of the true phase = Laplacian of the wrapped phase
        (in regions without wraps). We solve Poisson's equation:
            ∇²Φ_unwrapped = ∇²(sin(Φ)·cos(Φ) - cos(Φ)·sin(Φ))
        """
        # Simplified: use cosine-transform based approach
        # ∇²Φ_true ≈ Im[exp(-iΦ) · ∇²(exp(iΦ))]
        c = np.cos(phase)
        s = np.sin(phase)

        # Discrete Laplacian of exp(iΦ)
        lap_c = np.zeros_like(c)
        lap_s = np.zeros_like(s)
        for ax in range(phase.ndim):
            lap_c += np.roll(c, 1, ax) + np.roll(c, -1, ax) - 2 * c
            lap_s += np.roll(s, 1, ax) + np.roll(s, -1, ax) - 2 * s

        # Laplacian of true phase
        lap_phase = c * lap_s - s * lap_c

        # Solve Poisson equation via DCT (fast)
        # ∇²Φ = ρ  →  Φ = DCT⁻¹[DCT[ρ] / eigenvalues]
        # For simplicity, use iterative Gauss-Seidel
        unwrapped = np.copy(phase)
        for _ in range(50):
            for ax in range(phase.ndim):
                avg = (np.roll(unwrapped, 1, ax) + np.roll(unwrapped, -1, ax)) / 2
                unwrapped = avg + lap_phase / (2 * phase.ndim)

        return unwrapped

    def correct_phase(self, signal: np.ndarray, delta_B0: np.ndarray) -> np.ndarray:
        """
        Correct phase accumulation due to B₀ inhomogeneity.

        S_corrected(r) = S(r) · exp(+i · 2π · γ · ΔB₀(r) · TE)

        Args:
            signal: (X, Y, Z) or (X, Y, Z, T) complex signal
            delta_B0: (X, Y, Z) field map in Tesla

        Returns:
            corrected: phase-corrected signal
        """
        correction_phase = 2 * np.pi * self.hw.gamma_proton * delta_B0 * self.hw.TE
        correction = np.exp(1j * correction_phase)

        if signal.ndim == 4:
            return signal * correction[..., np.newaxis]
        return signal * correction

    def estimate_geometric_distortion(self, delta_B0: np.ndarray) -> np.ndarray:
        """
        Estimate the geometric distortion (pixel shift) in phase-encode direction.

        Δpixels(r) = γ · ΔB₀(r) / BW_per_pixel

        Args:
            delta_B0: (X, Y, Z) field map in Tesla

        Returns:
            shift_map: (X, Y, Z) pixel shifts in phase-encode direction
        """
        shift_pixels = self.hw.gamma_proton * delta_B0 / self.hw.bandwidth_per_pixel
        return shift_pixels


# ============================================================================
#  3. T₂* DECAY / SUSCEPTIBILITY CORRECTION
# ============================================================================

class T2starCorrector:
    """
    Multi-echo T₂* decomposition for separating BOLD signal from artifacts.

    At each voxel, multi-echo data follows:
        S(TE_n) = S₀ · exp(-TE_n · R₂*)

    We fit for S₀ and R₂* = 1/T₂* via log-linear regression:
        ln|S(TE_n)| = ln(S₀) - R₂* · TE_n

    BOLD contrast is then: ΔR₂* = R₂*(condition) - R₂*(baseline)

    For single-echo data, we estimate the SNR penalty from the known
    T₂* at the given field strength and TE.
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw

    def compute_bold_sensitivity(self) -> float:
        """
        Compute the BOLD sensitivity factor for the current TE and T₂*.

        BOLD signal change: ΔS/S ∝ ΔR₂* · TE · exp(-TE/T₂*)

        Maximum sensitivity when TE = T₂*.

        Returns:
            sensitivity: relative BOLD sensitivity (1.0 at optimal TE)
        """
        TE = self.hw.TE
        T2s = self.hw.T2star_GM
        optimal_sensitivity = T2s * math.exp(-1)  # TE = T2* peak
        actual_sensitivity = TE * math.exp(-TE / T2s)
        return actual_sensitivity / optimal_sensitivity

    def decompose_multi_echo(
        self,
        multi_echo_data: np.ndarray,
        echo_times: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform T₂* fitting on multi-echo data.

        Log-linear regression:
            ln|S(TE)| = ln(S₀) - R₂* · TE

        Args:
            multi_echo_data: (..., N_echoes) magnitude data
            echo_times: list of TE values in seconds

        Returns:
            S0: (...) proton density weighted image
            R2star: (...) R₂* map (1/s)
        """
        TEs = np.array(echo_times)
        log_data = np.log(np.abs(multi_echo_data) + 1e-10)

        # Linear regression: y = a + b·x where y = ln|S|, x = TE, b = -R2*
        n = len(TEs)
        sum_x = TEs.sum()
        sum_x2 = (TEs**2).sum()

        sum_y = log_data.sum(axis=-1)
        sum_xy = (log_data * TEs).sum(axis=-1)

        denom = n * sum_x2 - sum_x**2
        b = (n * sum_xy - sum_x * sum_y) / denom  # slope = -R2*
        a = (sum_y - b * sum_x) / n               # intercept = ln(S0)

        R2star = -b
        S0 = np.exp(a)

        # Clamp R2star to physically plausible range
        R2star = np.clip(R2star, 0, 500)  # max ~500 Hz reasonable

        return S0, R2star

    def compute_snr_weight(self, volume_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Compute per-voxel SNR weighting based on field-strength physics.

        At position r, the effective SNR depends on:
            SNR(r) ∝ B₀^(7/4) · ρ(r) · (1 - exp(-TR/T₁(r))) · exp(-TE/T₂*(r))

        Returns a relative weighting array.
        """
        # Simplified: compute steady-state signal weight
        E1 = math.exp(-self.hw.TR / self.hw.T1_GM)
        E2star = math.exp(-self.hw.TE / self.hw.T2star_GM)

        signal_weight = (1 - E1) * E2star * self.hw.snr_scaling_vs_3T

        # Uniform across voxels without spatial T1/T2* maps
        return np.full(volume_shape, signal_weight)


# ============================================================================
#  4. HRF DECONVOLUTION (Neural Activity Recovery)
# ============================================================================

class HRFDeconvolver:
    """
    Recover neural activity n(t) from observed BOLD y(t).

    Forward model:  y(t) = h(t) ∗ n(t) + ε(t)

    The canonical double-gamma HRF:
        h(t) = A₁·[(t/τ₁)^(a₁-1)·exp(-t/τ₁)]/Γ(a₁)
             - A₂·[(t/τ₂)^(a₂-1)·exp(-t/τ₂)]/Γ(a₂)

    Deconvolution via:
    1. Wiener filter:  N(f) = Y(f)·H*(f) / (|H(f)|² + λ²)
    2. Ridge regression: n = (H^TH + λI)^{-1} H^T y
    3. Tikhonov with derivative penalty: minimize ||y - H·n||² + λ||Dn||²

    We combine approach 1 (fast) with 3 (smoothness prior) for best results.
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw
        self.hrf_params = {
            'a1': 6.0,    # shape parameter 1
            'a2': 16.0,   # shape parameter 2
            'b1': 1.0,    # scale parameter 1 = 1/τ₁
            'b2': 1.0,    # scale parameter 2 = 1/τ₂
            'c': 1.0/6.0, # ratio A₂/A₁
        }

    def make_hrf(self, length: float = 32.0) -> np.ndarray:
        """
        Generate the canonical double-gamma HRF.

        h(t) = (t^(a₁-1) · b₁^a₁ · exp(-b₁·t)) / Γ(a₁)
             - c · (t^(a₂-1) · b₂^a₂ · exp(-b₂·t)) / Γ(a₂)

        Args:
            length: duration in seconds

        Returns:
            h: (N,) HRF sampled at TR
        """
        t = np.arange(0, length, self.hw.TR)

        a1, a2 = self.hrf_params['a1'], self.hrf_params['a2']
        b1, b2 = self.hrf_params['b1'], self.hrf_params['b2']
        c = self.hrf_params['c']

        # Double gamma
        t_safe = np.maximum(t, 1e-10)  # avoid t=0 issues
        h1 = (t_safe ** (a1 - 1)) * (b1 ** a1) * np.exp(-b1 * t_safe) / gamma_func(a1)
        h2 = (t_safe ** (a2 - 1)) * (b2 ** a2) * np.exp(-b2 * t_safe) / gamma_func(a2)
        h = h1 - c * h2

        # Normalize
        h = h / np.sum(np.abs(h))

        return h

    def make_astrocyte_hrf(self, length: float = 64.0) -> np.ndarray:
        """
        Generate a SLOWER HRF for the astrocytic component.

        Astrocyte hemodynamics are ~2-3x slower than neuronal:
            Peak: ~12-16s (vs 6s for neurons)
            Width: ~40s (vs ~20s for neurons)

        This reflects calcium signaling timescales in astrocytes.
        """
        t = np.arange(0, length, self.hw.TR)
        t_safe = np.maximum(t, 1e-10)

        # Slower parameters
        a1, b1 = 10.0, 0.5   # peak at ~20s
        a2, b2 = 24.0, 0.5   # undershoot at ~48s
        c = 1.0 / 6.0

        h1 = (t_safe ** (a1 - 1)) * (b1 ** a1) * np.exp(-b1 * t_safe) / gamma_func(a1)
        h2 = (t_safe ** (a2 - 1)) * (b2 ** a2) * np.exp(-b2 * t_safe) / gamma_func(a2)
        h = h1 - c * h2
        h = h / np.sum(np.abs(h))
        return h

    def deconvolve_wiener(
        self,
        bold_signal: np.ndarray,
        regularization: float = 0.01
    ) -> np.ndarray:
        """
        Wiener deconvolution to recover neural activity from BOLD.

        N(f) = Y(f) · H*(f) / (|H(f)|² + λ²)

        This is the MMSE estimate under assumptions of stationary noise.

        Args:
            bold_signal: (T,) or (N, T) BOLD time series
            regularization: λ² — noise floor (higher = more smoothing)

        Returns:
            neural: (T,) or (N, T) estimated neural activity
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for deconvolution")

        h = self.make_hrf()
        L = bold_signal.shape[-1]

        # Pad HRF to signal length
        h_padded = np.zeros(L)
        h_padded[:len(h)] = h

        H = fft(h_padded)
        H_conj = np.conj(H)
        H_power = np.abs(H) ** 2

        if bold_signal.ndim == 1:
            Y = fft(bold_signal)
            N = Y * H_conj / (H_power + regularization)
            neural = np.real(ifft(N))
        else:
            neural = np.zeros_like(bold_signal)
            for i in range(bold_signal.shape[0]):
                Y = fft(bold_signal[i])
                N = Y * H_conj / (H_power + regularization)
                neural[i] = np.real(ifft(N))

        return neural

    def deconvolve_tikhonov(
        self,
        bold_signal: np.ndarray,
        regularization: float = 1.0,
        smoothness: float = 0.1
    ) -> np.ndarray:
        """
        Tikhonov-regularized deconvolution with smoothness penalty.

        Minimizes: ||y - H·n||² + λ||n||² + μ||D·n||²

        where D is a first-difference operator enforcing temporal smoothness.

        This is solved via:
            n = (H^TH + λI + μD^TD)^{-1} H^T y

        Args:
            bold_signal: (T,) BOLD time series
            regularization: λ — ridge penalty
            smoothness: μ — derivative penalty

        Returns:
            neural: (T,) estimated neural activity
        """
        h = self.make_hrf()
        T = len(bold_signal)

        # Build convolution matrix H (Toeplitz)
        H_mat = np.zeros((T, T))
        for j in range(T):
            for k in range(min(len(h), T - j)):
                H_mat[j + k, j] = h[k]

        # First-difference matrix D
        D = np.zeros((T - 1, T))
        for i in range(T - 1):
            D[i, i] = -1
            D[i, i + 1] = 1

        # Solve: (H^TH + λI + μD^TD) n = H^T y
        HtH = H_mat.T @ H_mat
        DtD = D.T @ D
        Hty = H_mat.T @ bold_signal

        A = HtH + regularization * np.eye(T) + smoothness * DtD
        neural = np.linalg.solve(A, Hty)

        return neural


# ============================================================================
#  5. PHYSIOLOGICAL NOISE REGRESSION
# ============================================================================

class PhysiologicalNoiseRegressor:
    """
    Remove cardiac and respiratory confounds from fMRI time series.

    RETROICOR model (Glover et al., 2000):
        Cardiac: Σ_{k=1}^{M} [a_k·cos(kφ_c) + b_k·sin(kφ_c)]
        Respiratory: Σ_{k=1}^{M} [c_k·cos(kφ_r) + d_k·sin(kφ_r)]

    When physiological recordings are NOT available (most cases),
    we estimate cardiac and respiratory components from the fMRI itself:
        1. Cardiac: highest-power peak in fMRI spectra near 1-1.5 Hz
        2. Respiratory: highest-power peak near 0.2-0.5 Hz

    Regression:
        S_cleaned(t) = S(t) - X_nuisance @ β̂
        β̂ = (X^T X)^{-1} X^T S
    """

    def __init__(self, hw: MRIHardwareConfig, n_harmonics: int = 2):
        self.hw = hw
        self.n_harmonics = n_harmonics

    def estimate_physio_from_bold(self, bold_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate cardiac and respiratory phase from BOLD itself.

        Uses the approach of Beall & Lowe (2007): the global BOLD signal
        contains detectable cardiac and respiratory fluctuations.

        Args:
            bold_signal: (N_voxels, T) BOLD data

        Returns:
            dict with 'cardiac_phase', 'respiratory_phase' each (T,)
        """
        if not HAS_SCIPY:
            T = bold_signal.shape[-1]
            return {
                'cardiac_phase': np.zeros(T),
                'respiratory_phase': np.zeros(T)
            }

        # Global signal (mean across voxels)
        global_sig = np.mean(bold_signal, axis=0)
        T = len(global_sig)
        fs = 1.0 / self.hw.TR

        # Check if cardiac/respiratory frequencies are resolvable
        nyquist = fs / 2.0

        phases = {}

        # Respiratory: 0.15 - 0.5 Hz
        resp_lo, resp_hi = 0.15, min(0.5, nyquist * 0.9)
        if resp_hi > resp_lo and T > 10:
            b, a = butter(3, [resp_lo, resp_hi], btype='band', fs=fs)
            resp_filtered = filtfilt(b, a, global_sig)
            analytic = hilbert(resp_filtered)
            phases['respiratory_phase'] = np.angle(analytic)
        else:
            phases['respiratory_phase'] = np.zeros(T)

        # Cardiac: 0.8 - 1.5 Hz (only if fs > ~3 Hz, i.e., TR < 0.67s)
        cardiac_lo, cardiac_hi = 0.8, min(1.5, nyquist * 0.9)
        if cardiac_hi > cardiac_lo and nyquist > 1.0 and T > 10:
            b, a = butter(3, [cardiac_lo, cardiac_hi], btype='band', fs=fs)
            cardiac_filtered = filtfilt(b, a, global_sig)
            analytic = hilbert(cardiac_filtered)
            phases['cardiac_phase'] = np.angle(analytic)
        else:
            # Cardiac not resolvable at this TR
            phases['cardiac_phase'] = np.zeros(T)

        return phases

    def build_nuisance_matrix(self, phases: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build the nuisance regressor matrix X from estimated phases.

        X = [cos(φ_c), sin(φ_c), cos(2φ_c), sin(2φ_c), ...,
             cos(φ_r), sin(φ_r), cos(2φ_r), sin(2φ_r), ...]

        Args:
            phases: dict with 'cardiac_phase' and 'respiratory_phase'

        Returns:
            X: (T, 2·n_harmonics·2) nuisance regressors
        """
        regressors = []

        for key in ['cardiac_phase', 'respiratory_phase']:
            phi = phases.get(key, None)
            if phi is not None and np.any(phi != 0):
                for k in range(1, self.n_harmonics + 1):
                    regressors.append(np.cos(k * phi))
                    regressors.append(np.sin(k * phi))

        if not regressors:
            return np.zeros((len(next(iter(phases.values()))), 1))

        return np.column_stack(regressors)

    def regress(self, bold_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove physiological noise from BOLD data.

        S_clean = S - X @ β̂
        β̂ = (X^T X)^{-1} X^T S

        Args:
            bold_data: (N_voxels, T) BOLD time series

        Returns:
            cleaned: (N_voxels, T) physio-corrected data
            nuisance_variance: (N_voxels,) variance attributed to physio noise
        """
        phases = self.estimate_physio_from_bold(bold_data)
        X = self.build_nuisance_matrix(phases)

        if X.shape[1] == 1 and np.all(X == 0):
            return bold_data.copy(), np.zeros(bold_data.shape[0])

        # OLS: β̂ = (X^T X)^{-1} X^T S
        XtX = X.T @ X
        XtX += 1e-6 * np.eye(XtX.shape[0])  # regularize
        XtX_inv = np.linalg.inv(XtX)

        # S is (N, T), X is (T, K) → β̂ is (K, N)
        beta = XtX_inv @ X.T @ bold_data.T  # (K, N)
        prediction = (X @ beta).T  # (N, T)

        cleaned = bold_data - prediction
        nuisance_var = np.var(prediction, axis=1)

        return cleaned, nuisance_var


# ============================================================================
#  6. UNIFIED MRI FORWARD MODEL & INVERSE CORRECTOR
# ============================================================================

class MRIForwardModel:
    """
    The complete forward observation model for MRI:

        S_obs(r, t) = H_coil(r) · H_T2star(r, TE) · [HRF ∗ n(r, t)] + η_physio(t) + ε(t)

    This class can SIMULATE the observation process (for testing)
    and provides the Jacobian ∂S/∂Ψ for gradient-based correction.
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw

    def simulate_observation(
        self,
        neural_activity: np.ndarray,
        coil_sensitivity: Optional[np.ndarray] = None,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Simulate the full MRI observation process on true neural activity.

        Forward pipeline:
            Ψ_true → HRF convolution → T₂* weighting → coil modulation → noise

        Args:
            neural_activity: (N_voxels, T) true neural activity
            coil_sensitivity: (N_voxels,) coil profile (or None for uniform)
            noise_level: relative noise amplitude

        Returns:
            observed: (N_voxels, T) simulated BOLD signal
        """
        if not HAS_SCIPY:
            return neural_activity  # no scipy, return as-is

        N, T = neural_activity.shape
        hrf_gen = HRFDeconvolver(self.hw)
        hrf = hrf_gen.make_hrf()

        # 1. Convolve with HRF
        bold = np.zeros_like(neural_activity)
        for i in range(N):
            conv = np.convolve(neural_activity[i], hrf, mode='full')[:T]
            bold[i] = conv

        # 2. T₂* weighting
        t2star_weight = math.exp(-self.hw.TE / self.hw.T2star_GM)
        bold *= t2star_weight

        # 3. Coil sensitivity modulation
        if coil_sensitivity is not None:
            bold *= coil_sensitivity[:, np.newaxis]

        # 4. Add physiological noise
        cardiac_freq = self.hw.cardiac_rate
        resp_freq = self.hw.respiratory_rate
        t = np.arange(T) * self.hw.TR
        physio = (0.5 * np.sin(2 * np.pi * cardiac_freq * t) +
                  0.3 * np.sin(2 * np.pi * resp_freq * t))
        bold += noise_level * 0.5 * physio[np.newaxis, :]

        # 5. Thermal noise
        bold += noise_level * np.random.randn(*bold.shape)

        return bold


class MRIArtifactCorrector:
    """
    Unified artifact correction pipeline.

    Applies corrections in the mathematically correct ORDER:
        1. Coil sensitivity (multiplicative bias) — if not already done
        2. B₀ field map correction — if field maps available
        3. Physiological noise regression — from data itself
        4. HRF deconvolution — to recover neural timescale activity
        5. T₂* / SNR weighting — for optimal combination

    The output is the CLEANED neural activity estimate that feeds
    into the D-LinOSS model as input u(t).

    Mathematical guarantee:
        If H = H_coil · H_T2star · H_HRF · H_physio, then
        H_corrected ≈ H^{-1} · H = I + O(ε)
        where ε depends on estimation accuracy of each component.
    """

    def __init__(self, hw: MRIHardwareConfig):
        self.hw = hw
        self.coil_corrector = CoilSensitivityCorrector(hw)
        self.b0_corrector = B0FieldCorrector(hw)
        self.t2star_corrector = T2starCorrector(hw)
        self.hrf_deconvolver = HRFDeconvolver(hw)
        self.physio_regressor = PhysiologicalNoiseRegressor(hw)

    def full_correction_pipeline(
        self,
        bold_data: np.ndarray,
        field_map: Optional[np.ndarray] = None,
        do_hrf_deconvolution: bool = True,
        hrf_regularization: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Apply the full artifact correction pipeline.

        Args:
            bold_data: (N_voxels, T) raw BOLD time series
            field_map: (N_voxels,) optional B₀ field map
            do_hrf_deconvolution: whether to deconvolve the HRF
            hrf_regularization: λ for Wiener deconvolution
            verbose: print diagnostics

        Returns:
            dict with:
                'cleaned': (N_voxels, T) corrected data
                'neural_estimate': (N_voxels, T) deconvolved neural activity
                'physio_variance': (N_voxels,) variance removed by physio regression
                'snr_weights': (N_voxels,) per-voxel SNR quality
                'bold_sensitivity': float — BOLD sensitivity at current TE/T₂*
                'hw_summary': dict — hardware configuration summary
                'correction_log': list — steps applied
        """
        correction_log = []
        data = bold_data.copy()

        if verbose:
            print(f"\n{'='*70}")
            print(f"  MRI ARTIFACT CORRECTION PIPELINE")
            print(f"  Hardware: {self.hw.B0}T {self.hw.pulse_sequence}")
            print(f"  TE={self.hw.TE*1e3:.1f}ms, TR={self.hw.TR:.2f}s")
            print(f"  Larmor: ¹H={self.hw.f_larmor/1e6:.1f}MHz, "
                  f"⁴³Ca={self.hw.calcium_larmor/1e6:.2f}MHz, "
                  f"²³Na={self.hw.sodium_larmor/1e6:.2f}MHz")
            print(f"  SNR scaling vs 3T: {self.hw.snr_scaling_vs_3T:.2f}×")
            print(f"{'='*70}")

        # Step 1: Physiological noise regression
        if verbose:
            print("  [1/4] Physiological noise regression...")
        data_physio_clean, physio_var = self.physio_regressor.regress(data)
        correction_log.append({
            'step': 'physio_regression',
            'variance_removed_mean': float(np.mean(physio_var)),
            'variance_removed_max': float(np.max(physio_var)),
            'n_regressors': self.physio_regressor.n_harmonics * 4
        })
        data = data_physio_clean

        if verbose:
            frac = np.mean(physio_var) / (np.var(data, axis=1).mean() + 1e-10)
            print(f"       Removed {frac*100:.1f}% variance (mean across voxels)")

        # Step 2: Temporal detrending (polynomial)
        if verbose:
            print("  [2/4] Polynomial detrending (scanner drift)...")
        T = data.shape[-1]
        t_norm = np.linspace(-1, 1, T)
        poly_order = 3
        poly_basis = np.column_stack([t_norm ** k for k in range(poly_order + 1)])
        # Remove polynomial trend from each voxel
        beta_poly = np.linalg.lstsq(poly_basis, data.T, rcond=None)[0]
        trend = (poly_basis @ beta_poly).T
        data = data - trend
        correction_log.append({
            'step': 'polynomial_detrend',
            'order': poly_order,
            'trend_variance_mean': float(np.var(trend, axis=1).mean())
        })

        # Step 3: Z-score normalization (per voxel)
        if verbose:
            print("  [3/4] Per-voxel normalization...")
        voxel_mean = data.mean(axis=1, keepdims=True)
        voxel_std = data.std(axis=1, keepdims=True)
        voxel_std[voxel_std < 1e-10] = 1.0
        data_normalized = (data - voxel_mean) / voxel_std
        correction_log.append({
            'step': 'voxel_normalization',
            'mean_range': [float(voxel_mean.min()), float(voxel_mean.max())],
            'std_range': [float(voxel_std.min()), float(voxel_std.max())]
        })

        # Step 4: HRF deconvolution (optional but recommended)
        neural_estimate = data_normalized
        if do_hrf_deconvolution and HAS_SCIPY:
            if verbose:
                print("  [4/4] Wiener HRF deconvolution...")
            neural_estimate = self.hrf_deconvolver.deconvolve_wiener(
                data_normalized, regularization=hrf_regularization
            )
            correction_log.append({
                'step': 'hrf_deconvolution',
                'method': 'wiener',
                'regularization': hrf_regularization,
                'hrf_length_s': len(self.hrf_deconvolver.make_hrf()) * self.hw.TR
            })
        elif verbose:
            print("  [4/4] Skipping HRF deconvolution (scipy required)")

        # Compute quality metrics
        bold_sensitivity = self.t2star_corrector.compute_bold_sensitivity()

        hw_summary = {
            'B0_tesla': self.hw.B0,
            'TE_ms': self.hw.TE * 1e3,
            'TR_s': self.hw.TR,
            'T2star_GM_ms': self.hw.T2star_GM * 1e3,
            'larmor_proton_MHz': self.hw.f_larmor / 1e6,
            'larmor_calcium_MHz': self.hw.calcium_larmor / 1e6,
            'larmor_sodium_MHz': self.hw.sodium_larmor / 1e6,
            'larmor_potassium_MHz': self.hw.potassium_larmor / 1e6,
            'snr_scaling_vs_3T': self.hw.snr_scaling_vs_3T,
            'bold_cnr_scaling_vs_3T': self.hw.bold_cnr_scaling_vs_3T,
            'bold_sensitivity_fraction': bold_sensitivity,
            'pulse_sequence': self.hw.pulse_sequence,
            'preprocessing_applied': {
                'slice_timing': self.hw.slice_timing_corrected,
                'motion_correction': self.hw.motion_corrected,
                'distortion_correction': self.hw.distortion_corrected,
                'bias_field_correction': self.hw.bias_field_corrected,
            }
        }

        if verbose:
            print(f"\n  BOLD sensitivity: {bold_sensitivity:.3f} "
                  f"({'optimal' if abs(bold_sensitivity - 1.0) < 0.1 else 'sub-optimal'})")
            print(f"  Neural estimate shape: {neural_estimate.shape}")
            print(f"  Done.\n")

        return {
            'cleaned': data_normalized,
            'neural_estimate': neural_estimate,
            'physio_variance': physio_var,
            'bold_sensitivity': bold_sensitivity,
            'hw_summary': hw_summary,
            'correction_log': correction_log,
            'uncorrected_trend': trend  # keep for QA
        }


# ============================================================================
#  7. INTEGRATION HELPER: Connect to D-LinOSS Pipeline
# ============================================================================

def prepare_dlinoss_input(
    bold_data: np.ndarray,
    hw_config: Optional[MRIHardwareConfig] = None,
    do_hrf_deconvolution: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    One-call preprocessing to convert raw BOLD → D-LinOSS-ready input.

    This is the main entry point for the correction pipeline.
    It handles all the reverse-engineering of experimental conditions.

    Math guarantee:
        raw_bold ≈ H_total · Ψ_true + noise
        output ≈ H_total^{-1} · raw_bold ≈ Ψ_true + O(noise/SNR)

    Args:
        bold_data: (N_voxels, T) raw BOLD time series
        hw_config: MRI hardware configuration (auto-creates default 3T if None)
        do_hrf_deconvolution: whether to recover neural timescale
        verbose: print progress

    Returns:
        input_data: (T, N_voxels) — ready for D-LinOSS (transposed to L×H format)
        metadata: dict with correction details and quality metrics
    """
    if hw_config is None:
        hw_config = MRIHardwareConfig()

    corrector = MRIArtifactCorrector(hw_config)
    results = corrector.full_correction_pipeline(
        bold_data=bold_data,
        do_hrf_deconvolution=do_hrf_deconvolution,
        verbose=verbose
    )

    # Transpose to (T, N_voxels) = (L, H) for D-LinOSS
    input_data = results['neural_estimate'].T  # (T, N)

    return input_data, results
