#!/usr/bin/env python3
"""
D-LinOSS Visual Activation Pattern Analysis (Haxby ds000105)
==============================================================

Experiment Design:
  Subject views images of 8 visual categories presented in blocks:
    face, house, cat, bottle, scissors, shoe, chair, scrambledpix

  Block structure: 24s stimulus (12 images x 2s each) + 12s rest
  12 runs per subject, 121 volumes per run, TR = 2.5s

  The electromagnetic cascade:
    Light -> retina -> LGN -> V1 -> V2 -> V4 -> IT cortex
    Total propagation: ~50-150ms (sub-TR, smeared by hemodynamics)

  What we should see in the BOLD signal:
    1. Visual cortex activation ~4-6s after stimulus onset (HRF delay)
    2. Category-specific patterns in ventral temporal cortex
    3. Similar R2* fluctuation patterns for repeated presentations
       of the same category
    4. Habituation: potential amplitude decrease across repetitions

Pipeline:
  1. Load BOLD data + event timing for stimulus-aligned analysis
  2. Apply preprocessing (physio noise, detrend, normalize)
  3. Build stimulus-locked design matrix for each category
  4. Run D-LinOSS model — learn oscillatory decomposition
  5. Compare model oscillator activations across stimulus categories
     -> If the model learns the visual processing dynamics correctly,
        same-category blocks should produce similar oscillator patterns
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

import nibabel as nib
import torch
from dlinoss.dlinoss_torch import DLinOSSModel
from dlinoss.mri_artifact_correction import (
    MRIHardwareConfig,
    PhysiologicalNoiseRegressor,
)


# ============================================================================
#  Data Loading
# ============================================================================

def load_haxby_subject(subject_dir, max_runs=12):
    """
    Load BOLD data and event timing for a Haxby subject.

    Returns:
        bold_runs: list of (X, Y, Z, T) arrays
        events_runs: list of event dicts [{onset, duration, trial_type}, ...]
        TR: repetition time
    """
    func_dir = os.path.join(subject_dir, 'func')
    nii_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_bold.nii.gz')])
    event_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_events.tsv')])

    bold_runs = []
    events_runs = []

    for i, nii_file in enumerate(nii_files[:max_runs]):
        # Load BOLD
        img = nib.load(os.path.join(func_dir, nii_file))
        data = img.get_fdata()
        bold_runs.append(data)

        # Load events if available
        if i < len(event_files):
            events = []
            with open(os.path.join(func_dir, event_files[i])) as f:
                header = f.readline()  # skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        events.append({
                            'onset': float(parts[0]),
                            'duration': float(parts[1]),
                            'trial_type': parts[2],
                        })
            events_runs.append(events)
        else:
            events_runs.append([])

    TR = img.header.get_zooms()[-1]
    if TR < 0.1 or TR > 10:
        TR = 2.5  # fallback

    return bold_runs, events_runs, float(TR)


def load_haxby_local(data_dir, max_runs=12):
    """
    Load the local haxby2001 dataset (concatenated BOLD + labels).

    The local data has all 12 runs concatenated into one BOLD file
    with a labels.txt that marks each TR's condition.
    """
    bold_path = os.path.join(data_dir, 'bold.nii.gz')
    labels_path = os.path.join(data_dir, 'labels.txt')

    print(f"    Loading {bold_path}...")
    img = nib.load(bold_path)
    bold_4d = img.get_fdata()  # (X, Y, Z, T)
    TR = float(img.header.get_zooms()[-1])
    if TR < 0.1 or TR > 10:
        TR = 2.5

    # Load labels
    with open(labels_path) as f:
        lines = f.readlines()

    # Parse labels: format is "category run_number" e.g. "face 0", "rest 3"
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] != 'labels':
            category = parts[0]
            labels.append(category)

    print(f"    BOLD shape: {bold_4d.shape}, TR={TR}s")
    print(f"    Labels: {len(labels)} timepoints")

    return bold_4d, labels, TR


# ============================================================================
#  Preprocessing
# ============================================================================

def extract_brain_voxels(bold_4d, percentile=25):
    """
    Extract brain voxels from 4D BOLD data using intensity threshold.

    Returns:
        ts_2d: (N_voxels, T) time series of brain voxels
        brain_mask: (X, Y, Z) boolean mask
        coords: (N_voxels, 3) voxel coordinates
    """
    mean_vol = bold_4d.mean(axis=-1)
    threshold = np.percentile(mean_vol[mean_vol > 0], percentile)
    brain_mask = mean_vol > threshold

    n_brain = brain_mask.sum()
    T = bold_4d.shape[-1]

    # Extract timeseries
    ts_2d = bold_4d[brain_mask]  # (N_voxels, T)
    coords = np.argwhere(brain_mask)

    return ts_2d, brain_mask, coords


def preprocess_bold(bold_ts, TR, hw):
    """
    Preprocess BOLD timeseries for D-LinOSS.

    Steps: physio regression -> detrend -> normalize
    NO HRF deconvolution — we keep the hemodynamic response intact
    so the model can learn the full temporal dynamics.
    """
    N, T = bold_ts.shape

    # 1. Physiological noise regression
    physio = PhysiologicalNoiseRegressor(hw, n_harmonics=2)
    cleaned, _ = physio.regress(bold_ts)

    # 2. Linear detrend
    t_axis = np.arange(T, dtype=np.float64)
    for i in range(N):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)

    # 3. Z-score
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    cleaned = (cleaned - mu) / sigma

    # Sanitize
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = np.clip(cleaned, -5.0, 5.0)

    return cleaned  # (N_voxels, T)


# ============================================================================
#  Stimulus-Locked Analysis
# ============================================================================

def build_category_labels(events, T, TR):
    """
    Convert event list to per-TR category labels.

    Returns:
        labels: list of str, one per TR ('rest' for non-stimulus periods)
    """
    labels = ['rest'] * T
    for ev in events:
        onset_tr = int(ev['onset'] / TR)
        # Block duration in TRs (typically 24s / 2.5s = ~10 TRs)
        dur_trs = max(1, int(ev['duration'] / TR))
        # But events are individual stimuli at 0.5s each
        # A block of 12 stimuli spans 24s = ~10 TRs
        # Mark the TR that this stimulus falls in
        if 0 <= onset_tr < T:
            labels[onset_tr] = ev['trial_type']
    return labels


def build_block_labels(events, T, TR):
    """
    Build block-level labels (mark entire 24s blocks, not individual stimuli).
    Each block: 12 stimuli x 2s = 24s
    """
    labels = ['rest'] * T

    # Group events by consecutive same-category blocks
    if not events:
        return labels

    # Find block boundaries: consecutive events of same type
    current_type = None
    block_start = None

    for ev in events:
        if ev['trial_type'] != current_type:
            # End previous block
            if current_type and block_start is not None:
                block_end_tr = int(ev['onset'] / TR)
                for t in range(block_start, min(block_end_tr, T)):
                    labels[t] = current_type
            current_type = ev['trial_type']
            block_start = int(ev['onset'] / TR)

    # Close last block (extends ~24s from start)
    if current_type and block_start is not None:
        block_end_tr = min(block_start + int(24 / TR), T)
        for t in range(block_start, block_end_tr):
            labels[t] = current_type

    return labels


# ============================================================================
#  D-LinOSS Visual Pattern Learning
# ============================================================================

def run_visual_dlinoss(clean_input, labels, TR,
                       state_dim=32, hidden_dim=64, n_blocks=3,
                       n_epochs=80, lr=1e-3):
    """
    Train D-LinOSS on visual task data and analyze category-specific patterns.

    The model learns to reconstruct the BOLD signal through its oscillator bank.
    After training, we extract the internal oscillator states during each
    stimulus category to see if the model captures category-specific dynamics.
    """
    T, N = clean_input.shape
    print(f"\n  D-LinOSS Visual Pattern Model")
    print(f"    Input: ({T} timepoints, {N} voxels)")
    print(f"    Architecture: {n_blocks} blocks, P={state_dim}, H={hidden_dim}")

    model = DLinOSSModel(
        input_dim=N,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=N,
        num_blocks=n_blocks,
        drop_rate=0.0,
    )

    x = torch.tensor(clean_input, dtype=torch.float32)

    # ── Train ──
    print(f"\n    Self-supervised training ({n_epochs} epochs)...")
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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}: loss = {loss.item():.6f}")

    # ── Extract learned representations ──
    model.eval()
    with torch.no_grad():
        y_final = model(x)

    # Reconstruction quality: per-voxel correlation
    recon = y_final.numpy()  # (T, N)
    input_np = clean_input   # (T, N)
    correlations = []
    for v in range(N):
        r = np.corrcoef(input_np[:, v], recon[:, v])[0, 1]
        if np.isfinite(r):
            correlations.append(r)
    mean_corr = np.mean(correlations)

    print(f"\n    Final loss: {losses[-1]:.6f}")
    print(f"    Reconstruction correlation: {mean_corr:.4f}")

    # ── Category-specific activation patterns ──
    print(f"\n    Analyzing category-specific patterns...")

    # Build category masks
    categories = sorted(set(l for l in labels if l != 'rest'))
    cat_indices = {cat: [i for i, l in enumerate(labels) if l == cat]
                   for cat in categories}
    rest_indices = [i for i, l in enumerate(labels) if l == 'rest']

    print(f"    Categories: {categories}")
    print(f"    Rest TRs: {len(rest_indices)}")

    # For each category, compute mean activation pattern
    cat_patterns = {}  # category -> mean output vector
    cat_input_patterns = {}
    for cat, idx in cat_indices.items():
        if len(idx) > 0:
            cat_patterns[cat] = recon[idx].mean(axis=0)        # (N,)
            cat_input_patterns[cat] = input_np[idx].mean(axis=0)
            print(f"    {cat:15s}: {len(idx):3d} TRs, "
                  f"mean activation = {cat_patterns[cat].std():.4f}")

    # Rest pattern
    if rest_indices:
        rest_pattern = recon[rest_indices].mean(axis=0)
    else:
        rest_pattern = np.zeros(N)

    # ── Category similarity matrix ──
    # Correlation between mean activation patterns for each pair
    cat_names = sorted(cat_patterns.keys())
    n_cats = len(cat_names)
    sim_matrix = np.zeros((n_cats, n_cats))

    for i, c1 in enumerate(cat_names):
        for j, c2 in enumerate(cat_names):
            r = np.corrcoef(cat_patterns[c1], cat_patterns[c2])[0, 1]
            sim_matrix[i, j] = r if np.isfinite(r) else 0

    print(f"\n    Category similarity matrix (correlation of mean patterns):")
    header = "              " + "".join(f"{c[:6]:>8s}" for c in cat_names)
    print(f"    {header}")
    for i, c1 in enumerate(cat_names):
        row = f"    {c1:12s}  " + "".join(f"{sim_matrix[i,j]:8.3f}" for j in range(n_cats))
        print(row)

    # ── Stimulus vs Rest contrast ──
    print(f"\n    Stimulus vs Rest contrast:")
    for cat in cat_names:
        # Cohen's d: (mean_stim - mean_rest) / pooled_std
        stim_power = np.mean(recon[cat_indices[cat]] ** 2, axis=0)
        rest_power = np.mean(recon[rest_indices] ** 2, axis=0) if rest_indices else np.zeros(N)
        contrast = (stim_power - rest_power).mean()
        # Also compute from input for comparison
        input_stim = np.mean(input_np[cat_indices[cat]] ** 2, axis=0)
        input_rest = np.mean(input_np[rest_indices] ** 2, axis=0) if rest_indices else np.zeros(N)
        input_contrast = (input_stim - input_rest).mean()
        print(f"      {cat:15s}: model contrast = {contrast:+.4f}, "
              f"raw contrast = {input_contrast:+.4f}")

    # ── Temporal dynamics: category-locked time courses ──
    # Average BOLD response aligned to block onset for each category
    print(f"\n    Block-locked response (HRF-shaped BOLD response):")
    block_responses = {}
    window = int(30 / TR)  # 30 seconds post-onset

    for cat in cat_names:
        idx = cat_indices[cat]
        if len(idx) < 3:
            continue
        # Find block onsets (first TR of consecutive same-category)
        onsets = []
        for k in range(len(idx)):
            if k == 0 or idx[k] != idx[k-1] + 1:
                onsets.append(idx[k])

        # Average response aligned to onset
        responses = []
        for onset in onsets:
            if onset + window <= T:
                responses.append(recon[onset:onset+window].mean(axis=1))

        if responses:
            avg_response = np.mean(responses, axis=0)
            block_responses[cat] = avg_response
            peak_tr = np.argmax(avg_response)
            peak_time = peak_tr * TR
            print(f"      {cat:15s}: peak at {peak_time:.1f}s (TR {peak_tr}), "
                  f"amplitude = {avg_response.max():.4f}, "
                  f"n_blocks = {len(responses)}")

    # ── Spectral analysis ──
    spectral = analyze_visual_spectrum(model, TR)

    return {
        'losses': losses,
        'mean_correlation': float(mean_corr),
        'category_similarity': sim_matrix.tolist(),
        'category_names': cat_names,
        'block_responses': {k: v.tolist() for k, v in block_responses.items()},
        'spectral': spectral,
        'n_categories': n_cats,
        'n_timepoints': T,
        'n_voxels': N,
    }


def analyze_visual_spectrum(model, TR):
    """Analyze oscillator spectrum for visual task relevance."""
    all_freqs = []
    all_mags = []

    for block in model.blocks:
        layer = block.ssm
        with torch.no_grad():
            A, G, dt = layer._soft_project(
                layer.A_diag, layer.G_diag, layer.dt_raw
            )

        S = 1.0 + dt * G
        det_M = 1.0 / S
        trace_M = 1.0 / S + 1.0 - dt ** 2 * A / S

        mags = torch.sqrt(det_M).numpy()
        cos_theta = (trace_M / (2 * torch.sqrt(det_M))).clamp(-1, 1)
        freqs = (torch.acos(cos_theta) / (np.pi * TR)).numpy()

        all_mags.extend(mags.tolist())
        all_freqs.extend(freqs.tolist())

    all_mags = np.array(all_mags)
    all_freqs = np.array(all_freqs)

    # Block frequency: the stimulus blocks repeat at ~1/36s = 0.028 Hz
    # (24s stimulus + 12s rest = 36s cycle)
    block_freq = 1.0 / 36.0  # ~0.028 Hz
    near_block = np.abs(all_freqs - block_freq) < 0.01  # within 0.01 Hz

    print(f"\n    Spectral analysis:")
    print(f"      Max |lambda|: {all_mags.max():.6f}")
    print(f"      Frequency range: [{all_freqs.min():.4f}, {all_freqs.max():.4f}] Hz")
    print(f"      Block frequency: {block_freq:.4f} Hz (36s cycle)")
    print(f"      Oscillators near block freq: {near_block.sum()}")

    return {
        'max_magnitude': float(all_mags.max()),
        'frequencies': all_freqs.tolist(),
        'magnitudes': all_mags.tolist(),
        'block_frequency_hz': block_freq,
        'oscillators_near_block_freq': int(near_block.sum()),
    }


# ============================================================================
#  Main Pipeline
# ============================================================================

def process_haxby_subject(subject_dir, subject_name, max_runs=4, max_voxels=500):
    """Full pipeline for one Haxby subject."""
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  VISUAL ACTIVATION ANALYSIS: {subject_name}")
    print(f"  Dataset: Haxby ds000105 — Visual Object Recognition")
    print(f"  Path: {subject_dir}")
    print(f"{'='*70}")

    # ── Load data ──
    print(f"\n  Loading BOLD data (up to {max_runs} runs)...")
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)

    print(f"    Loaded {len(bold_runs)} runs")
    print(f"    Volume shape: {bold_runs[0].shape}")
    print(f"    TR: {TR}s")

    # ── Concatenate runs ──
    # Stack runs along time dimension
    bold_concat = np.concatenate(bold_runs, axis=-1)  # (X, Y, Z, T_total)
    T_total = bold_concat.shape[-1]
    print(f"    Concatenated: {bold_concat.shape} ({T_total} total TRs = {T_total*TR:.0f}s)")

    # Concatenate event labels
    all_labels = []
    for i, (events, bold) in enumerate(zip(events_runs, bold_runs)):
        T_run = bold.shape[-1]
        run_labels = build_block_labels(events, T_run, TR)
        all_labels.extend(run_labels)
        cats = set(l for l in run_labels if l != 'rest')
        print(f"    Run {i+1}: {T_run} TRs, categories: {cats or 'no events'}")

    # Label distribution
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\n    Label distribution across {len(all_labels)} TRs:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_labels)
        print(f"      {label:15s}: {count:4d} TRs ({pct:5.1f}%)")

    # ── Extract brain voxels ──
    print(f"\n  Extracting brain voxels...")
    ts_2d, brain_mask, coords = extract_brain_voxels(bold_concat, percentile=25)
    print(f"    Brain voxels: {ts_2d.shape[0]:,}")

    # Select top-variance voxels
    variances = np.var(ts_2d, axis=1)
    n_select = min(max_voxels, len(variances))
    top_idx = np.argsort(variances)[-n_select:]
    selected_ts = ts_2d[top_idx]
    print(f"    Selected {n_select} high-variance voxels")

    # ── Preprocess ──
    print(f"\n  Preprocessing...")
    hw = MRIHardwareConfig(B0=3.0, TR=TR)
    cleaned = preprocess_bold(selected_ts, TR, hw)  # (N, T)
    clean_input = cleaned.T  # (T, N)

    print(f"    Clean input: {clean_input.shape} (T, N_voxels)")

    preproc_time = time.time() - t0

    # ── D-LinOSS model ──
    t1 = time.time()
    results = run_visual_dlinoss(
        clean_input, all_labels, TR,
        state_dim=32, hidden_dim=64, n_blocks=3,
        n_epochs=80, lr=1e-3,
    )
    model_time = time.time() - t1
    total_time = time.time() - t0

    # ── Summary ──
    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {subject_name}")
    print(f"  {'='*60}")
    print(f"    Time: {total_time:.1f}s (preproc: {preproc_time:.1f}s, model: {model_time:.1f}s)")
    print(f"    Reconstruction correlation: {results['mean_correlation']:.4f}")
    print(f"    Final loss: {results['losses'][-1]:.6f}")
    print(f"    Categories detected: {results['n_categories']}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..',
                              'dlinoss_results', f'Visual_{subject_name}')
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'subject': subject_name,
        'dataset': 'ds000105',
        'task': 'objectviewing',
        'n_runs': len(bold_runs),
        'TR': TR,
        'n_voxels': n_select,
        'n_timepoints': T_total,
        'preprocessing_time': preproc_time,
        'model_time': model_time,
        'total_time': total_time,
        **results,
    }

    with open(os.path.join(output_dir, 'visual_analysis.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved to: {output_dir}")

    return summary


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')

    if not os.path.exists(base):
        print("ERROR: ds000105 not found.")
        sys.exit(1)

    # Find subjects with event files (sub-4, sub-5, sub-6)
    subjects = sorted([d for d in os.listdir(base) if d.startswith('sub-')])
    subjects_with_events = []
    for subj in subjects:
        func_dir = os.path.join(base, subj, 'func')
        events = [f for f in os.listdir(func_dir) if 'events' in f]
        if events:
            subjects_with_events.append(subj)

    print(f"Haxby Visual Object Recognition (ds000105)")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Subjects with event timing: {subjects_with_events}")
    print(f"  Categories: face, house, cat, bottle, scissors, shoe, chair, scrambledpix")
    print(f"  Design: block (24s stim + 12s rest), 12 runs/subject")

    # Process subjects with events
    # Start with 4 runs per subject to keep it manageable
    all_results = {}
    for subj in subjects_with_events:
        subj_dir = os.path.join(base, subj)
        try:
            result = process_haxby_subject(subj_dir, subj,
                                           max_runs=4, max_voxels=400)
            all_results[subj] = result
        except Exception as e:
            print(f"\n  ERROR processing {subj}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-subject comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-SUBJECT COMPARISON")
        print(f"{'='*70}")
        for subj, r in sorted(all_results.items()):
            print(f"\n  {subj}:")
            print(f"    Recon correlation: {r['mean_correlation']:.4f}")
            print(f"    Final loss: {r['losses'][-1]:.6f}")
            if r.get('category_similarity') and r.get('category_names'):
                # Show face-house dissimilarity (classic Haxby result)
                cats = r['category_names']
                sim = np.array(r['category_similarity'])
                if 'face' in cats and 'house' in cats:
                    fi = cats.index('face')
                    hi = cats.index('house')
                    print(f"    Face-House correlation: {sim[fi, hi]:.3f} "
                          f"(should be LOW for good category separation)")

    return all_results


if __name__ == '__main__':
    main()
