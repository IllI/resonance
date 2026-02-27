#!/usr/bin/env python3
"""
Run D-LinOSS on Multi-Echo fMRI Data (ds000258)
=================================================

Pipeline Architecture (two clean stages, per project convention):

  STAGE 1: PREPROCESSING (this script handles)
  ─────────────────────────────────────────────
  Load 4 echoes (TE = 12, 28, 44, 60 ms)
      ↓
  T₂* decomposition: ln|S(TE)| = ln(S₀) - R₂*·TE
      ↓  Separates BOLD-related T₂* changes from non-BOLD artifacts
  S₀(t)  — proton-density weighted (no BOLD contrast, used as reference)
  R₂*(t) — rate map (contains the BOLD signal: ΔR₂* ∝ ΔHBO₂)
      ↓
  Select high-variance ROI voxels from R₂*(t)
      ↓
  Physiological noise regression (estimated from global signal)
      ↓
  Detrend + normalize
      ↓
  Clean (T, N_voxels) array

  STAGE 2: D-LinOSS MODEL
  ────────────────────────
  Clean signal → DLinOSSModel (PyTorch) → Oscillatory decomposition
      ↓
  Spectral analysis: eigenvalue distribution, decay rates, frequencies
      ↓
  Output: learned temporal dynamics of BOLD R₂* fluctuations

Note on HRF deconvolution: For resting-state data (ds000258), we SKIP
HRF deconvolution because there are no stimulus onsets to align to.
The D-LinOSS oscillator bank can learn the hemodynamic temporal structure
directly — each oscillator has its own decay rate and frequency, enabling
it to naturally separate fast neural fluctuations from slow hemodynamic drift.
"""

import os
import sys
import json
import time
import numpy as np

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("ERROR: nibabel required — pip install nibabel")
    sys.exit(1)

import torch
from dlinoss.dlinoss_torch import DLinOSSModel
from dlinoss.mri_artifact_correction import (
    MRIHardwareConfig,
    T2starCorrector,
    PhysiologicalNoiseRegressor,
    MRIArtifactCorrector,
)


# ============================================================================
#  Multi-Echo Loading
# ============================================================================

def load_multi_echo_subject(subject_dir: str) -> dict:
    """
    Load all 4 echo volumes for a single subject from ds000258.

    Returns dict with:
        'echoes': list of (X, Y, Z, T) arrays, one per echo
        'echo_times': list of TE values in seconds
        'TR': repetition time in seconds
        'metadata': dict of scanner parameters from JSON sidecar
    """
    func_dir = os.path.join(subject_dir, 'func')
    if not os.path.exists(func_dir):
        raise FileNotFoundError(f"No func/ directory in {subject_dir}")

    # Find all echo NIfTI files
    nii_files = sorted([f for f in os.listdir(func_dir) if f.endswith('.nii.gz')])
    json_files = sorted([f for f in os.listdir(func_dir) if f.endswith('.json')])

    if len(nii_files) < 4:
        raise ValueError(f"Expected 4 echo files, found {len(nii_files)}")

    echoes = []
    echo_times = []
    metadata = {}

    for nii_file, json_file in zip(nii_files, json_files):
        # Load JSON sidecar for echo time
        with open(os.path.join(func_dir, json_file)) as f:
            meta = json.load(f)

        te = meta['EchoTime']  # in seconds
        echo_times.append(te)

        if not metadata:
            metadata = meta

        # Load NIfTI volume (with corrupt file detection)
        nii_path = os.path.join(func_dir, nii_file)
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()  # (X, Y, Z, T) or (X, Y, Z)
        except (EOFError, Exception) as e:
            print(f"    WARNING: Corrupt/truncated file {nii_file}: {e}")
            print(f"    Skipping this subject — re-download may be needed")
            raise

        echoes.append(data)

        print(f"    Echo {len(echoes)}: TE={te*1000:.0f}ms, shape={data.shape}, "
              f"range=[{data.min():.0f}, {data.max():.0f}]")

    return {
        'echoes': echoes,
        'echo_times': echo_times,
        'TR': metadata.get('RepetitionTime', 2.47),
        'metadata': metadata,
    }


# ============================================================================
#  T₂* Decomposition (the key preprocessing step for multi-echo)
# ============================================================================

def decompose_t2star_timeseries(echoes: list, echo_times: list) -> dict:
    """
    Perform voxelwise T₂* decomposition at every timepoint.

    For each voxel at each timepoint t:
        S(TE_n, t) = S₀(t) · exp(-TE_n · R₂*(t))

    Log-linear regression over the 4 echoes gives:
        S₀(t)  — proton density weighted image (no BOLD)
        R₂*(t) — relaxation rate (CONTAINS the BOLD signal)

    The BOLD effect is a CHANGE in R₂*:
        ΔR₂* = γ · Δχ · ΔfHbO₂ · (4/3)π · B₀
    where Δχ is the susceptibility difference between oxy- and deoxyhemoglobin.

    So R₂*(t) directly tracks local blood oxygenation changes.

    Returns dict with:
        'S0': (X, Y, Z, T) proton density
        'R2star': (X, Y, Z, T) R₂* map
        'T2star': (X, Y, Z, T) T₂* = 1/R₂* map
        'mean_R2star': (X, Y, Z) time-averaged R₂*
    """
    TEs = np.array(echo_times)
    n_echoes = len(echoes)

    # Stack echoes: (X, Y, Z, T, N_echoes)
    # Handle both 3D (X,Y,Z) and 4D (X,Y,Z,T) echo volumes
    if echoes[0].ndim == 3:
        # Single volume per echo — add time dimension
        stacked = np.stack(echoes, axis=-1)  # (X, Y, Z, N_echoes)
        has_time = False
    else:
        stacked = np.stack(echoes, axis=-1)  # (X, Y, Z, T, N_echoes)
        has_time = True

    print(f"    Stacked shape: {stacked.shape}")
    print(f"    Echo times (ms): {[f'{te*1000:.0f}' for te in echo_times]}")

    # Log-linear regression: ln|S(TE)| = ln(S₀) - R₂*·TE
    # Avoid log(0) by adding small epsilon
    log_data = np.log(np.abs(stacked) + 1e-10)

    # Vectorized least-squares: y = a + b·x, x = TE, b = -R₂*
    n = len(TEs)
    sum_x = TEs.sum()
    sum_x2 = (TEs ** 2).sum()

    sum_y = log_data.sum(axis=-1)
    sum_xy = (log_data * TEs).sum(axis=-1)

    denom = n * sum_x2 - sum_x ** 2
    slope = (n * sum_xy - sum_x * sum_y) / denom  # -R₂*
    intercept = (sum_y - slope * sum_x) / n        # ln(S₀)

    R2star = np.clip(-slope, 0, 500)  # R₂* in Hz, clamped to physical range
    S0 = np.exp(np.clip(intercept, -20, 20))  # Proton density

    # T₂* = 1/R₂* (avoid division by zero)
    T2star = np.where(R2star > 0.1, 1.0 / R2star, 0.0)

    mean_R2star = R2star.mean(axis=-1) if has_time else R2star

    print(f"    R₂* range: [{R2star.min():.1f}, {R2star.max():.1f}] Hz")
    print(f"    T₂* range: [{T2star[T2star > 0].min()*1000:.1f}, "
          f"{T2star[T2star > 0].max()*1000:.1f}] ms (nonzero voxels)")
    print(f"    S₀ range:  [{S0.min():.1f}, {S0.max():.1f}]")

    return {
        'S0': S0,
        'R2star': R2star,
        'T2star': T2star,
        'mean_R2star': mean_R2star,
    }


# ============================================================================
#  ROI Selection (select informative voxels for D-LinOSS)
# ============================================================================

def select_roi_voxels(R2star_4d: np.ndarray, S0_4d: np.ndarray,
                      max_voxels: int = 500,
                      min_mean_signal: float = 100.0) -> dict:
    """
    Select voxels with highest R₂* temporal variance (= most BOLD information).

    Strategy:
    1. Mask out background (low S₀ = outside the brain)
    2. Mask out extreme R₂* (susceptibility artifacts near sinuses/ear canals)
    3. Rank remaining voxels by temporal variance of R₂* timeseries
    4. Select top-N voxels

    This gives us the voxels with the strongest BOLD-related R₂* fluctuations.
    """
    # Mean signal intensity as brain mask proxy
    mean_S0 = S0_4d.mean(axis=-1) if S0_4d.ndim == 4 else S0_4d
    brain_mask = mean_S0 > min_mean_signal

    # Mean R₂* for reasonable range check (exclude air, susceptibility dropout)
    mean_R2star = R2star_4d.mean(axis=-1)
    r2star_mask = (mean_R2star > 5) & (mean_R2star < 200)  # 5-200 Hz is brain tissue

    # Combined mask
    valid_mask = brain_mask & r2star_mask
    n_valid = valid_mask.sum()

    print(f"    Brain voxels (S₀ > {min_mean_signal}): {brain_mask.sum():,}")
    print(f"    R₂* in range (5-200 Hz): {r2star_mask.sum():,}")
    print(f"    Valid voxels: {n_valid:,}")

    # Extract R₂* timeseries for valid voxels: (N_valid, T)
    valid_coords = np.argwhere(valid_mask)
    r2star_ts = R2star_4d[valid_mask]  # (N_valid, T)

    # Rank by temporal variance
    variances = np.var(r2star_ts, axis=-1)

    # Select top-N highest variance voxels
    n_select = min(max_voxels, len(variances))
    top_idx = np.argsort(variances)[-n_select:]

    selected_ts = r2star_ts[top_idx]  # (N_select, T)
    selected_coords = valid_coords[top_idx]
    selected_vars = variances[top_idx]

    print(f"    Selected {n_select} voxels by R₂* variance")
    print(f"    Variance range: [{selected_vars.min():.2f}, {selected_vars.max():.2f}]")
    print(f"    Mean R₂* of selection: {selected_ts.mean():.1f} Hz "
          f"(T₂*≈{1000/selected_ts.mean():.0f} ms)")

    return {
        'timeseries': selected_ts,  # (N_voxels, T) — R₂* timeseries
        'coordinates': selected_coords,
        'variances': selected_vars,
        'n_total_brain': int(n_valid),
    }


# ============================================================================
#  Preprocessing: Clean the R₂* timeseries for D-LinOSS
# ============================================================================

def preprocess_r2star(r2star_ts: np.ndarray, TR: float,
                      hw: MRIHardwareConfig) -> np.ndarray:
    """
    Clean the R₂*(t) timeseries before passing to D-LinOSS model.

    Steps:
    1. Physiological noise regression (cardiac ~1Hz, respiratory ~0.3Hz)
       — estimated from global signal since we don't have physio recordings
    2. Linear detrend (remove scanner drift)
    3. Z-score normalization per voxel

    We deliberately SKIP HRF deconvolution for resting-state data because:
    - No stimulus timing to validate against
    - D-LinOSS oscillator bank can learn the hemodynamic temporal structure
    - Over-deconvolution would amplify noise in R₂* domain

    Args:
        r2star_ts: (N_voxels, T) R₂* timeseries
        TR: repetition time in seconds
        hw: hardware configuration

    Returns:
        cleaned: (T, N_voxels) preprocessed data ready for D-LinOSS
        (note: transposed to (T, N) convention for the model)
    """
    N, T = r2star_ts.shape
    print(f"\n  Preprocessing R₂* timeseries: ({N} voxels, {T} timepoints)")

    # 1. Physiological noise regression
    print(f"    Step 1: Physiological noise regression...")
    physio = PhysiologicalNoiseRegressor(hw, n_harmonics=2)
    cleaned, nuisance_var = physio.regress(r2star_ts)
    var_removed = nuisance_var.mean()
    var_total = np.var(r2star_ts, axis=1).mean()
    print(f"      Variance removed: {var_removed:.4f} "
          f"({100*var_removed/var_total:.1f}% of total)")

    # 2. Linear detrend per voxel
    print(f"    Step 2: Linear detrend...")
    t_axis = np.arange(T, dtype=np.float64)
    for i in range(N):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)

    # 3. Z-score normalization
    print(f"    Step 3: Z-score normalization...")
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    cleaned = (cleaned - mu) / sigma

    # Sanity checks
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = np.clip(cleaned, -5.0, 5.0)  # clip extreme outliers

    # Transpose to (T, N) for model input
    output = cleaned.T  # (T, N_voxels)

    print(f"    Output: {output.shape} (T, N_voxels)")
    print(f"    Range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"    Std: {output.std():.2f}")

    return output


# ============================================================================
#  D-LinOSS Model (Stage 2)
# ============================================================================

def run_dlinoss_model(clean_input: np.ndarray, TR: float,
                      state_dim: int = 32, hidden_dim: int = 64,
                      n_blocks: int = 3, n_epochs: int = 50,
                      lr: float = 1e-3) -> dict:
    """
    Run D-LinOSS on preprocessed fMRI data.

    The model learns oscillatory decomposition of the R₂* dynamics:
    - Each oscillator has eigenvalue λ = r·exp(iθ)
    - r controls decay rate (how quickly the oscillation damps)
    - θ controls frequency (what temporal scale it captures)

    For fMRI at TR=2.47s (Nyquist ≈ 0.2 Hz):
    - Oscillators with θ near 0 → ultra-slow drift (<0.01 Hz)
    - Oscillators with θ near π/10 → resting-state networks (0.01-0.1 Hz)
    - Oscillators with θ near π/2 → fast fluctuations (~0.1 Hz)

    Args:
        clean_input: (T, N_voxels) preprocessed data
        TR: repetition time
        state_dim: number of oscillators per D-LinOSS block (P)
        hidden_dim: feature dimension (H)
        n_blocks: number of D-LinOSS blocks
        n_epochs: training epochs for self-supervised reconstruction
        lr: learning rate

    Returns:
        dict with spectral analysis, model output, training loss
    """
    T, N = clean_input.shape
    print(f"\n  D-LinOSS Model")
    print(f"    Input: ({T} timepoints, {N} voxels)")
    print(f"    Architecture: {n_blocks} blocks, P={state_dim} oscillators, H={hidden_dim}")
    print(f"    Nyquist frequency: {1/(2*TR):.3f} Hz (TR={TR}s)")

    # Build model
    model = DLinOSSModel(
        input_dim=N,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=N,
        num_blocks=n_blocks,
        drop_rate=0.0,
    )

    x = torch.tensor(clean_input, dtype=torch.float32)

    # ── Forward pass (inference) ──
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"    Forward pass output: {y.shape}")
    print(f"    Output NaN: {torch.isnan(y).any().item()}")

    # ── Spectral analysis (pre-training) ──
    spectral_pre = analyze_spectrum(model, TR, label="pre-training")

    # ── Self-supervised training ──
    # Loss = reconstruction error: ||model(x) - x||²
    # This forces the oscillator bank to span the data's temporal structure
    print(f"\n    Training ({n_epochs} epochs, lr={lr})...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, x)

        if not torch.isfinite(loss):
            print(f"      Epoch {epoch}: NaN loss, stopping")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}: loss = {loss.item():.6f}")

    # ── Post-training spectral analysis ──
    spectral_post = analyze_spectrum(model, TR, label="post-training")

    # ── Final inference ──
    model.eval()
    with torch.no_grad():
        y_final = model(x)

    return {
        'output': y_final.numpy(),
        'losses': losses,
        'spectral_pre': spectral_pre,
        'spectral_post': spectral_post,
        'model_params': {
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'n_blocks': n_blocks,
            'n_epochs': n_epochs,
        },
    }


def analyze_spectrum(model, TR: float, label: str = "") -> dict:
    """Analyze the learned eigenvalue spectrum of all D-LinOSS blocks."""
    all_magnitudes = []
    all_frequencies = []
    all_decay_times = []

    for i, block in enumerate(model.blocks):
        layer = block.ssm
        with torch.no_grad():
            # Pass raw parameters to _soft_project
            A, G, dt = layer._soft_project(
                layer.A_diag, layer.G_diag, layer.dt_raw
            )

        P = A.shape[0]

        # Eigenvalues from the 2×2 recurrence matrix M
        S = 1.0 + dt * G
        det_M = 1.0 / S
        trace_M = 1.0 / S + 1.0 - dt ** 2 * A / S

        disc = trace_M ** 2 - 4 * det_M

        # Eigenvalue magnitudes: |lambda| = sqrt(det(M))
        magnitudes = torch.sqrt(det_M).numpy()

        # Eigenvalue frequencies: from imaginary part
        # For complex eigenvalues (disc < 0), theta = arccos(trace/2/|lambda|)
        # Frequency = theta / (pi * TR) Hz
        cos_theta = (trace_M / (2 * torch.sqrt(det_M))).clamp(-1, 1)
        theta = torch.acos(cos_theta)
        frequencies = (theta / (np.pi * TR)).numpy()

        # Decay time: how many TRs until amplitude drops to 1/e
        mags_safe = np.clip(magnitudes, 1e-10, 0.9999)
        decay_trs = -1.0 / np.log(mags_safe)
        decay_times = decay_trs * TR  # in seconds

        all_magnitudes.extend(magnitudes.tolist())
        all_frequencies.extend(frequencies.tolist())
        all_decay_times.extend(decay_times.tolist())

    all_magnitudes = np.array(all_magnitudes)
    all_frequencies = np.array(all_frequencies)
    all_decay_times = np.array(all_decay_times)

    print(f"\n    Spectral analysis ({label}):")
    print(f"      Max |lambda|:    {all_magnitudes.max():.6f} (stable < 1)")
    print(f"      Mean |lambda|:   {all_magnitudes.mean():.6f}")
    print(f"      Frequency range: [{all_frequencies.min():.4f}, "
          f"{all_frequencies.max():.4f}] Hz")
    print(f"      Decay time range: [{all_decay_times.min():.1f}, "
          f"{all_decay_times.max():.1f}] s")

    # Frequency bands relevant to fMRI
    ultra_slow = (all_frequencies < 0.01).sum()
    slow = ((all_frequencies >= 0.01) & (all_frequencies < 0.05)).sum()
    rsn = ((all_frequencies >= 0.05) & (all_frequencies < 0.1)).sum()
    fast = (all_frequencies >= 0.1).sum()

    print(f"      Ultra-slow (<0.01 Hz): {ultra_slow} oscillators")
    print(f"      Slow (0.01-0.05 Hz):   {slow} oscillators")
    print(f"      RSN band (0.05-0.1 Hz): {rsn} oscillators")
    print(f"      Fast (>0.1 Hz):         {fast} oscillators")

    return {
        'magnitudes': all_magnitudes.tolist(),
        'frequencies': all_frequencies.tolist(),
        'decay_times': all_decay_times.tolist(),
        'max_spectral_radius': float(all_magnitudes.max()),
        'mean_magnitude': float(all_magnitudes.mean()),
        'frequency_bands': {
            'ultra_slow': int(ultra_slow),
            'slow': int(slow),
            'rsn': int(rsn),
            'fast': int(fast),
        },
    }


# ============================================================================
#  Main Pipeline
# ============================================================================

def process_multiecho_subject(subject_dir: str, subject_name: str,
                              max_voxels: int = 500,
                              output_dir: str = None) -> dict:
    """
    Full pipeline for one multi-echo subject.

    Stage 1: Preprocessing
        Load echoes → T₂* decomposition → ROI selection → clean

    Stage 2: D-LinOSS Model
        Clean signal → model → spectral analysis

    Returns: dict with all results
    """
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  MULTI-ECHO D-LinOSS ANALYSIS: {subject_name}")
    print(f"  Path: {subject_dir}")
    print(f"{'='*70}")

    # ── Stage 1a: Load multi-echo data ──
    print(f"\n  Stage 1a: Loading multi-echo data...")
    me_data = load_multi_echo_subject(subject_dir)
    echoes = me_data['echoes']
    echo_times = me_data['echo_times']
    TR = me_data['TR']

    print(f"    TR = {TR}s")
    print(f"    Echo times = {[f'{te*1000:.0f}ms' for te in echo_times]}")
    print(f"    Volume shape = {echoes[0].shape}")

    # ── Stage 1b: T₂* decomposition ──
    print(f"\n  Stage 1b: T₂* decomposition (4-echo fit)...")
    t2_results = decompose_t2star_timeseries(echoes, echo_times)

    # ── Stage 1c: ROI selection ──
    print(f"\n  Stage 1c: ROI selection (top R₂* variance voxels)...")
    roi = select_roi_voxels(
        t2_results['R2star'],
        t2_results['S0'],
        max_voxels=max_voxels
    )

    # ── Stage 1d: Preprocess ──
    hw = MRIHardwareConfig(
        B0=me_data['metadata'].get('MagneticFieldStrength', 3.0),
        TR=TR,
        TE=echo_times[1],  # Use echo 2 (28ms, closest to T₂*)
        n_coil_channels=32,  # 32-channel head coil
        multi_echo=True,
        echo_times=echo_times,
    )

    clean_input = preprocess_r2star(roi['timeseries'], TR, hw)

    stage1_time = time.time() - t0
    print(f"\n  Stage 1 complete: {stage1_time:.1f}s")
    print(f"    Clean input shape: {clean_input.shape} (T, N_voxels)")

    # ── Stage 2: D-LinOSS ──
    print(f"\n  Stage 2: D-LinOSS model...")
    t1 = time.time()
    model_results = run_dlinoss_model(
        clean_input,
        TR=TR,
        state_dim=32,
        hidden_dim=64,
        n_blocks=3,
        n_epochs=50,
        lr=1e-3,
    )
    stage2_time = time.time() - t1

    # ── Summary ──
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  RESULTS: {subject_name}")
    print(f"  {'='*60}")
    print(f"    Total time: {total_time:.1f}s "
          f"(preprocessing: {stage1_time:.1f}s, model: {stage2_time:.1f}s)")
    print(f"    Voxels analyzed: {clean_input.shape[1]}")
    print(f"    Timepoints: {clean_input.shape[0]}")
    print(f"    Final loss: {model_results['losses'][-1]:.6f}" if model_results['losses'] else "    No training")
    print(f"    Spectral radius: {model_results['spectral_post']['max_spectral_radius']:.6f}")

    # Frequency decomposition summary
    bands = model_results['spectral_post']['frequency_bands']
    print(f"    Frequency decomposition:")
    print(f"      Ultra-slow (<0.01 Hz): {bands['ultra_slow']} oscillators — scanner drift, very slow physio")
    print(f"      Slow (0.01-0.05 Hz):   {bands['slow']} oscillators — large-scale network dynamics")
    print(f"      RSN (0.05-0.1 Hz):     {bands['rsn']} oscillators — resting-state networks")
    print(f"      Fast (>0.1 Hz):         {bands['fast']} oscillators — cardiac/respiratory aliased")

    # ── Save results ──
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'dlinoss_results',
                                  f'MultiEcho_{subject_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON summary (no large arrays)
    summary = {
        'subject': subject_name,
        'scanner': {
            'field_strength': hw.B0,
            'TR': TR,
            'echo_times_ms': [te * 1000 for te in echo_times],
            'n_channels': hw.n_coil_channels,
            'manufacturer': me_data['metadata'].get('Manufacturer', 'unknown'),
            'model': me_data['metadata'].get('ManufacturersModelName', 'unknown'),
        },
        'preprocessing': {
            'n_voxels': int(clean_input.shape[1]),
            'n_timepoints': int(clean_input.shape[0]),
            'total_brain_voxels': roi['n_total_brain'],
            'r2star_variance_range': [float(roi['variances'].min()),
                                      float(roi['variances'].max())],
        },
        'model': model_results['model_params'],
        'training': {
            'final_loss': float(model_results['losses'][-1]) if model_results['losses'] else None,
            'n_epochs_completed': len(model_results['losses']),
        },
        'spectral_pre': model_results['spectral_pre'],
        'spectral_post': model_results['spectral_post'],
        'timing': {
            'preprocessing_s': stage1_time,
            'model_s': stage2_time,
            'total_s': total_time,
        },
    }

    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n    Results saved to: {summary_path}")

    return summary


def main():
    """Run multi-echo D-LinOSS analysis on all downloaded ds000258 subjects."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000258')

    if not os.path.exists(base):
        print("ERROR: ds000258 not found. Run download_priority_datasets.py --multiecho first.")
        sys.exit(1)

    # Find all subjects
    subjects = sorted([d for d in os.listdir(base) if d.startswith('sub-')])
    print(f"\nFound {len(subjects)} multi-echo subjects in ds000258")

    # Check each has 4 echo files
    valid_subjects = []
    for subj in subjects:
        func_dir = os.path.join(base, subj, 'func')
        if os.path.exists(func_dir):
            nii_files = [f for f in os.listdir(func_dir) if f.endswith('.nii.gz')]
            if len(nii_files) >= 4:
                valid_subjects.append(subj)
                print(f"  [OK] {subj}: {len(nii_files)} echo files")
            else:
                print(f"  [SKIP] {subj}: only {len(nii_files)} files")

    if not valid_subjects:
        print("No valid multi-echo subjects found.")
        sys.exit(1)

    # Process each subject
    all_results = {}
    for subj in valid_subjects:
        subj_dir = os.path.join(base, subj)
        try:
            result = process_multiecho_subject(subj_dir, subj, max_voxels=400)
            all_results[subj] = result
        except Exception as e:
            print(f"\n  ERROR processing {subj}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-subject comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-SUBJECT COMPARISON")
        print(f"{'='*70}")
        print(f"\n  {'Subject':<15} {'Loss':>10} {'|lambda|':>10} "
              f"{'RSN osc':>10} {'Time':>8}")
        print(f"  {'-'*53}")
        for subj, r in sorted(all_results.items()):
            loss = r['training']['final_loss'] or 0
            sr = r['spectral_post']['max_spectral_radius']
            rsn = r['spectral_post']['frequency_bands']['rsn']
            t = r['timing']['total_s']
            print(f"  {subj:<15} {loss:>10.4f} {sr:>10.6f} {rsn:>10d} {t:>7.1f}s")

    print(f"\n  Analysis complete: {len(all_results)} subjects processed")
    return all_results


if __name__ == '__main__':
    main()
