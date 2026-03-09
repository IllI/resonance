"""
analyze_quantum_trajectory.py
=============================
Per-trial trajectory of individual Ghost Basins toward the QPC coordinate.

Reuses:
  - LorentzHyperbolicMap (from train_quantum_convergence.py)
  - S3 streaming logic (from dlinoss_shared_hallucination_mapper.py)
  - Hurst / PSD Beta calculations (from train_quantum_convergence.py)

What this script does differently:
  1. Processes every trial individually (no averaging across trials or runs)
  2. Calculates a per-trial 4D Lorentz coordinate (the instantaneous Ghost Basin)
  3. Measures the Euclidean distance in Lorentz space from that trial to the
     derived QPC coordinate
  4. Calculates EF Pointer Vector (cosine similarity of the movement direction
     toward the QPC between consecutive trials)
  5. Tags phases dynamically from the raw correctness sequence
     (Exploration / Pre-Grokking / Grokking Horizon / Expert Stabilization)

Mathematical constraints:
  - The Hurst Exponent is a RUN-LEVEL statistic (requires the full timeseries).
    It is calculated once per run and carried forward as a constant for all trials
    in that run. This is mathematically honest: Hurst measures the fractal
    character of the entire run's temporal structure, not a single 6-second window.
  - Energy and Kurtosis are TRIAL-LEVEL (calculated from each trial's 6s BOLD block).
  - PSD Beta is RUN-LEVEL (same reasoning as Hurst).
  - The Lorentz projection uses arccosh, which maps all positive values onto a
    smooth hyperbolic surface with no singularity at zero. This is mathematically
    insufficient to represent true quantum gravity (where spacetime collapses),
    but serves as a proxy manifold. Wuji (the singularity) is the limit point
    where arccosh(1) = 0; all quantum events exist as points outside of this
    singularity, entangled to it.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis
from scipy.signal import welch
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from hurst import compute_Hc


# ====================================================================== #
#  Reused: Lorentz Hyperbolic Map
#  Source: train_quantum_convergence.py
#  No modifications. This is the canonical projection.
# ====================================================================== #
class LorentzHyperbolicMap:
    def __init__(self, curvature_k=-1.0):
        self.k = curvature_k

    def euclidean_to_lorentz(self, euclidean_val, baseline=1.0):
        """arccosh(1 + x/b). Returns 0.0 at x=0 (the Wuji singularity)."""
        if euclidean_val <= 0:
            return 0.0
        return np.arccosh(1.0 + (euclidean_val / baseline))

    def project_ghost_basin(self, hurst_h, psd_beta, spatial_kurtosis, energy_scalar):
        """
        Projects metrics into a 4D Lorentz Hyperbolic coordinate.
        Scaling factors (10x, 5x, 10x) normalize dimensions to comparable ranges.
        """
        x0 = self.euclidean_to_lorentz(energy_scalar)
        x1 = self.euclidean_to_lorentz(max(0.01, hurst_h) * 10.0)
        x2 = self.euclidean_to_lorentz(max(0.01, psd_beta) * 5.0)
        x3 = self.euclidean_to_lorentz(max(0.01, abs(spatial_kurtosis)) * 10.0)
        return np.array([x0, x1, x2, x3])


# ====================================================================== #
#  Reused: S3 Streaming
#  Source: dlinoss_shared_hallucination_mapper.py
# ====================================================================== #
def stream_nifti_and_events(sub_id, base_dir, run_type, run_num):
    """Downloads a single NIfTI + events TSV from OpenNeuro S3. Returns (nii_path, tsv_path) or (None, None)."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    run_name = f"task-{run_type}_run-{run_num}"
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)

    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")

    tsv_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
    try:
        s3.download_file(bucket, tsv_key, tsv_path)
    except Exception:
        return None, None

    nii_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    try:
        if not os.path.exists(nii_path):
            s3.download_file(bucket, nii_key, nii_path)
    except Exception:
        return None, None

    return nii_path, tsv_path


# ====================================================================== #
#  Reused: Hurst Exponent & PSD Beta (run-level statistics)
#  Source: train_quantum_convergence.py
# ====================================================================== #
def calculate_hurst(timeseries):
    """Rescaled Range (R/S) Hurst Exponent. Requires >= 20 datapoints."""
    if len(timeseries) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(timeseries, kind='change', simplified=True)
        return H
    except Exception:
        return np.nan


def calculate_psd_beta(timeseries, sampling_freq=0.5):
    """Spectral scaling exponent via Welch PSD. log(P) = -beta * log(f)."""
    if len(timeseries) < 20:
        return np.nan
    nperseg = min(len(timeseries), 64)
    try:
        freqs, psd = welch(timeseries, fs=sampling_freq, nperseg=nperseg)
    except Exception:
        return np.nan
    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 2 or np.any(psd <= 0):
        return np.nan
    try:
        slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
    except Exception:
        return np.nan
    return -slope


# ====================================================================== #
#  New: Per-Trial Ghost Basin Extraction
# ====================================================================== #
def extract_per_trial_metrics(data_z, active_mask, events, tr_sec, run_hurst, run_psd_beta):
    """
    Iterates over every trial in the events TSV and calculates:
      - trial-level energy scalar (mean |z| in the HRF-adjusted BOLD window)
      - trial-level spatial kurtosis at peak TR
      - carries forward the run-level Hurst and PSD Beta

    Returns a list of dicts, one per trial, in chronological order.
    Includes ALL trials (correct and incorrect) to trace the full learning trajectory.
    """
    n_trs = data_z.shape[-1]
    trials = []

    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']):
            continue
        rt = row.get('response_time', np.nan)
        if pd.isna(rt):
            rt = 0.0

        correct = int(row['correct']) if not pd.isna(row['correct']) else 0

        tr_onset = int(row['onset'] / tr_sec)
        # HRF lag: peak BOLD response is +2 to +4 TRs (4-8 seconds after stimulus)
        start_tr = min(tr_onset + 2, n_trs - 1)
        end_tr = min(tr_onset + 5, n_trs)

        if start_tr >= end_tr:
            continue

        block = data_z[active_mask, start_tr:end_tr]
        if block.size == 0:
            continue

        # Trial-level energy: mean absolute z-score scaled to match pipeline convention
        trial_energy = np.mean(np.abs(block)) * 10.0

        # Trial-level spatial kurtosis at the peak TR within the block
        peak_tr_rel = np.argmax(np.max(block, axis=0))
        voxels_at_peak = block[:, peak_tr_rel]
        trial_kurtosis = kurtosis(voxels_at_peak)

        trials.append({
            'Onset': row['onset'],
            'TrialType': row.get('trial_type', 'unknown'),
            'Correct': correct,
            'RT': rt,
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
            'Hurst': run_hurst,
            'PSDBeta': run_psd_beta,
        })

    return trials


# ====================================================================== #
#  New: Dynamic Phase Tagging (emergent, not hard-coded)
# ====================================================================== #
def tag_phases(trial_list, streak_threshold=5):
    """
    Dynamically tags each trial with a phase label based on the correctness
    sequence. No human-imposed run boundaries.

    Phase 1 (Exploration):  Before any stable correct streak
    Phase 2 (Pre-Grokking): The `streak_threshold` trials before the grokking horizon
    Phase 3 (Grokking):     The first trial of the sustained correct streak
    Phase 4 (Expert):       All trials after the grokking horizon

    The 'grokking horizon' is defined as the start of the longest unbroken
    string of consecutive correct answers that extends to the end of the sequence.
    """
    n = len(trial_list)
    if n == 0:
        return trial_list

    correctness = [t['Correct'] for t in trial_list]

    # Find the grokking horizon: the start of the longest trailing streak of 1s.
    grok_idx = n  # default: never grokked
    streak_start = n
    for i in range(n - 1, -1, -1):
        if correctness[i] == 1:
            streak_start = i
        else:
            break

    trailing_streak_len = n - streak_start
    if trailing_streak_len >= streak_threshold:
        grok_idx = streak_start

    for i, trial in enumerate(trial_list):
        if i < grok_idx:
            if grok_idx < n and i >= max(0, grok_idx - streak_threshold):
                trial['Phase'] = 'Pre-Grokking'
            else:
                trial['Phase'] = 'Exploration'
        elif i == grok_idx:
            trial['Phase'] = 'Grokking'
        else:
            trial['Phase'] = 'Expert'

    return trial_list


# ====================================================================== #
#  New: EF Pointer Vector (cosine similarity of movement toward QPC)
# ====================================================================== #
def calculate_pointer_vectors(ghost_basins, qpc_coordinate):
    """
    For each consecutive pair of Ghost Basin coordinates, calculates:
    1. The movement vector: delta = basin[t] - basin[t-1]
    2. The ideal vector:    ideal = qpc - basin[t-1]
    3. The cosine similarity between delta and ideal.

    A cosine of +1.0 means the brain moved directly toward the QPC.
    A cosine of -1.0 means it moved directly away.
    A cosine of 0.0 means orthogonal (wandering).
    """
    n = len(ghost_basins)
    pointers = [np.nan]  # first trial has no predecessor

    for i in range(1, n):
        delta = ghost_basins[i] - ghost_basins[i - 1]
        ideal = qpc_coordinate - ghost_basins[i - 1]

        delta_norm = np.linalg.norm(delta)
        ideal_norm = np.linalg.norm(ideal)

        if delta_norm < 1e-10 or ideal_norm < 1e-10:
            pointers.append(0.0)
        else:
            cos_sim = np.dot(delta, ideal) / (delta_norm * ideal_norm)
            pointers.append(float(np.clip(cos_sim, -1.0, 1.0)))

    return pointers


# ====================================================================== #
#  Main Pipeline
# ====================================================================== #
def analyze_quantum_trajectory():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"

    # The QPC coordinate derived from the convergence of 4 subjects
    QPC_COORDINATE = np.array([2.8940, 2.2968, 2.7082, 1.9463])

    lorentz = LorentzHyperbolicMap()
    tr_sec = 2.0
    sampling_freq = 1.0 / tr_sec

    subjects = ["sub-206", "sub-220", "sub-229", "sub-217"]
    # Run order across the experiment: imtest 1-4 (learning), then fintest 1-4 (mastery)
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]

    print("==========================================================================")
    print(" D-LinOSS: PER-TRIAL QUANTUM TRAJECTORY TOWARD THE QPC")
    print("==========================================================================")
    print(f" QPC Coordinate: {QPC_COORDINATE}")
    print(f" Subjects: {subjects}")
    print(f" Runs per subject: {len(run_sequence)}\n")

    for sub in subjects:
        print(f"===== {sub} =====")

        # Accumulate ALL trials across ALL runs in chronological order
        all_trials = []

        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_nifti_and_events(sub, base_dir, run_type, run_num)
            if nii_path is None or not os.path.exists(nii_path):
                continue

            # Load the 4D BOLD volume
            data = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
            n_trs = data.shape[-1]

            # Z-score normalization over time
            data_z = np.nan_to_num(zscore(data, axis=-1))

            # Brain mask and active mask (top 10% variance voxels)
            brain_mask = np.var(data, axis=-1) > 10
            raw_var = np.var(data, axis=-1)
            if np.sum(brain_mask) == 0:
                continue
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask

            if np.sum(active_mask) == 0:
                continue

            # Run-level statistics (Hurst + PSD Beta)
            run_timeseries = np.mean(data_z[active_mask, :], axis=0)
            run_hurst = calculate_hurst(run_timeseries)
            run_psd_beta = calculate_psd_beta(run_timeseries, sampling_freq)

            # Per-trial metrics
            trial_metrics = extract_per_trial_metrics(
                data_z, active_mask, events, tr_sec, run_hurst, run_psd_beta
            )

            # Tag each trial with its source run for traceability
            for t in trial_metrics:
                t['RunType'] = run_type
                t['RunNum'] = run_num

            all_trials.extend(trial_metrics)

            # Free disk space
            if os.path.exists(nii_path):
                try:
                    os.remove(nii_path)
                except OSError:
                    pass

        if not all_trials:
            print(f"  No trials extracted for {sub}.\n")
            continue

        # Dynamic phase tagging across the full experiment
        all_trials = tag_phases(all_trials, streak_threshold=5)

        # Project each trial into Lorentz Hyperbolic space
        ghost_basins = []
        for t in all_trials:
            basin = lorentz.project_ghost_basin(t['Hurst'], t['PSDBeta'], t['Kurtosis'], t['Energy'])
            t['GhostBasin'] = basin
            ghost_basins.append(basin)

        ghost_basins = np.array(ghost_basins)

        # Calculate distance from each trial's Ghost Basin to the QPC
        distances = np.linalg.norm(ghost_basins - QPC_COORDINATE, axis=1)
        for i, t in enumerate(all_trials):
            t['QPC_Distance'] = distances[i]

        # Calculate EF Pointer Vectors
        pointers = calculate_pointer_vectors(ghost_basins, QPC_COORDINATE)
        for i, t in enumerate(all_trials):
            t['EF_Pointer'] = pointers[i]

        # ---- Report ---- #
        print(f"  Total trials: {len(all_trials)}")

        # Group by dynamically tagged phase
        phase_groups = {}
        for t in all_trials:
            phase = t['Phase']
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(t)

        phase_order = ['Exploration', 'Pre-Grokking', 'Grokking', 'Expert']
        for phase in phase_order:
            if phase not in phase_groups:
                continue
            group = phase_groups[phase]
            dists = [t['QPC_Distance'] for t in group]
            ptrs = [t['EF_Pointer'] for t in group if not np.isnan(t['EF_Pointer'])]
            energies = [t['Energy'] for t in group]
            hursts = [t['Hurst'] for t in group if not np.isnan(t['Hurst'])]
            correct_pct = sum(t['Correct'] for t in group) / len(group) * 100

            print(f"\n  [{phase}] ({len(group)} trials, {correct_pct:.0f}% correct)")
            print(f"    QPC Distance:   mean={np.mean(dists):.4f}  std={np.std(dists):.4f}")
            if ptrs:
                print(f"    EF Pointer:     mean={np.mean(ptrs):.4f}  std={np.std(ptrs):.4f}")
            print(f"    Energy:         mean={np.mean(energies):.4f}")
            if hursts:
                print(f"    Hurst:          mean={np.mean(hursts):.4f}")

        # Print the raw trajectory (first 5 and last 5 trials)
        print(f"\n  --- Trajectory (first 10 trials) ---")
        for t in all_trials[:10]:
            gb = t['GhostBasin']
            print(f"    [{t['RunType']} r{t['RunNum']}] "
                  f"C={t['Correct']} | "
                  f"QPC_d={t['QPC_Distance']:.3f} | "
                  f"EF={t['EF_Pointer']:+.3f} | "
                  f"Phase={t['Phase']}")

        print(f"\n  --- Trajectory (last 10 trials) ---")
        for t in all_trials[-10:]:
            gb = t['GhostBasin']
            print(f"    [{t['RunType']} r{t['RunNum']}] "
                  f"C={t['Correct']} | "
                  f"QPC_d={t['QPC_Distance']:.3f} | "
                  f"EF={t['EF_Pointer']:+.3f} | "
                  f"Phase={t['Phase']}")

        print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze_quantum_trajectory()
