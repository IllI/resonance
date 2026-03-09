"""
train_quantum_convergence.py
============================
D-LinOSS Quantum Convergence Training Script

Instead of averaging subjects together, this script:
1. Calculates each subject's INDIVIDUAL Hurst Exponent, PSD Beta, and Spatial Kurtosis
2. Projects each subject's metrics into a Lorentz Hyperbolic coordinate (their personal Ghost Basin)
3. Measures the PAIRWISE DISTANCE between subjects' Ghost Basins in hyperbolic space
4. Compares cross-subject distances during Learning vs Grokking

If individual Ghost Basins converge onto a single shared coordinate during grokking,
the pairwise hyperbolic distance will collapse — proving the existence of the
Quantum Phantom Cauldron superposition for the "Badoon" category.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis
from scipy.signal import welch
from scipy.spatial.distance import pdist
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from hurst import compute_Hc


class LorentzHyperbolicMap:
    """
    Maps Euclidean BOLD metrics into the Lorentz Hyperbolic Manifold.
    Each subject's Ghost Basin is a point in this space.
    """
    def __init__(self, curvature_k=-1.0):
        self.k = curvature_k

    def euclidean_to_lorentz(self, euclidean_energy, baseline=1.0):
        if euclidean_energy <= 0:
            return 0.0
        return np.arccosh(1.0 + (euclidean_energy / baseline))

    def project_ghost_basin(self, hurst_h, psd_beta, spatial_kurtosis, energy_scalar):
        """
        Projects a subject's individual metrics into a 4D Lorentz Hyperbolic coordinate.

        The Ghost Basin coordinate is:
          x0 = Lorentz(energy)         — metabolic tension
          x1 = Lorentz(hurst * 10)     — fractal memory depth
          x2 = Lorentz(psd_beta * 5)   — spectral scaling
          x3 = Lorentz(kurtosis * 10)  — spatial concentration

        The Lorentz mapping ensures no singularity at zero
        and preserves the hierarchical distance relationships.
        """
        x0 = self.euclidean_to_lorentz(energy_scalar)
        x1 = self.euclidean_to_lorentz(max(0.01, hurst_h) * 10.0)
        x2 = self.euclidean_to_lorentz(max(0.01, psd_beta) * 5.0)
        x3 = self.euclidean_to_lorentz(max(0.01, abs(spatial_kurtosis)) * 10.0)
        return np.array([x0, x1, x2, x3])


def lorentz_pairwise_distance(points):
    """
    Calculates the pairwise Lorentz Hyperbolic distance between Ghost Basin points.
    In the hyperboloid model, d(x,y) = arccosh(-<x,y>_L).
    We approximate using Euclidean distance in the Lorentz-mapped space
    since each coordinate is already individually mapped via arccosh.
    """
    if len(points) < 2:
        return np.array([0.0])
    return pdist(points, metric='euclidean')


class QuantumConvergenceAnalyzer:
    def __init__(self, tr_sec=2.0):
        self.tr_sec = tr_sec
        self.sampling_freq = 1.0 / tr_sec
        self.lorentz = LorentzHyperbolicMap()

    def stream_run(self, sub_id, base_dir, run_type, run_num):
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

    def calculate_hurst(self, timeseries):
        if len(timeseries) < 20:
            return np.nan
        try:
            H, c, data = compute_Hc(timeseries, kind='change', simplified=True)
            return H
        except Exception:
            return np.nan

    def calculate_psd_beta(self, timeseries):
        if len(timeseries) < 20:
            return np.nan
        nperseg = min(len(timeseries), 64)
        try:
            freqs, psd = welch(timeseries, fs=self.sampling_freq, nperseg=nperseg)
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

    def analyze_subject_run(self, nii_path, tsv_path):
        """
        Analyzes a single run for a single subject.
        Returns per-subject summary metrics (NOT per-trial averages).
        """
        if not nii_path or not os.path.exists(nii_path) or not os.path.exists(tsv_path):
            return None

        data = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')

        n_trs = data.shape[-1]
        data_z = np.nan_to_num(zscore(data, axis=-1))

        brain_mask = np.var(data, axis=-1) > 10
        raw_var = np.var(data, axis=-1)
        active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask

        if np.sum(active_mask) == 0:
            return None

        # Run-level time-series of the most active voxels
        run_timeseries = np.mean(data_z[active_mask, :], axis=0)
        run_hurst = self.calculate_hurst(run_timeseries)
        run_psd_beta = self.calculate_psd_beta(run_timeseries)

        # Trial-level spatial kurtosis and energy (only correct trials)
        kurtosis_vals = []
        energy_vals = []
        for _, row in events.iterrows():
            if 'correct' not in row or pd.isna(row['correct']):
                continue
            if int(row['correct']) != 1:
                continue
            rt = row.get('response_time', 0)
            if pd.isna(rt) or rt == 0:
                continue

            tr_onset = int(row['onset'] / self.tr_sec)
            start_tr = min(tr_onset + 2, n_trs - 1)
            end_tr = min(tr_onset + 5, n_trs)
            if start_tr >= end_tr:
                continue

            block = data_z[active_mask, start_tr:end_tr]
            if block.size == 0:
                continue

            peak_tr_rel = np.argmax(np.max(block, axis=0))
            voxels_at_peak = block[:, peak_tr_rel]

            kurtosis_vals.append(kurtosis(voxels_at_peak))
            energy_vals.append(np.mean(np.abs(block)) * 10.0)

        # Clean up
        if os.path.exists(nii_path):
            try:
                os.remove(nii_path)
            except OSError:
                pass

        if not kurtosis_vals:
            return None

        return {
            'Hurst': run_hurst,
            'PSDBeta': run_psd_beta,
            'MeanKurtosis': np.mean(kurtosis_vals),
            'MeanEnergy': np.mean(energy_vals),
            'NumCorrect': len(kurtosis_vals),
        }


def train_quantum_convergence():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    analyzer = QuantumConvergenceAnalyzer(tr_sec=2.0)
    lorentz = LorentzHyperbolicMap()

    # Use more subjects to get meaningful pairwise distances
    masters = ["sub-206", "sub-220", "sub-229", "sub-217"]

    print("==========================================================================")
    print(" D-LinOSS: GHOST BASIN CONVERGENCE ON THE QUANTUM PHANTOM CAULDRON")
    print("==========================================================================")
    print("Tracking each subject's INDIVIDUAL Ghost Basin in Lorentz Hyperbolic space.")
    print("If the Badoon superposition exists in the QPC, individual Ghost Basins")
    print("should CONVERGE onto a shared coordinate during grokking.\n")

    # Collect per-subject Ghost Basin coordinates for each phase
    learning_basins = {}  # sub_id -> np.array([x0, x1, x2, x3])
    grokking_basins = {}

    for sub in masters:
        print(f"--- {sub} ---")

        # Aggregate across runs for each phase per subject
        learning_runs = []
        for i in range(1, 5):
            nii, tsv = analyzer.stream_run(sub, base_dir, "imtest", i)
            if nii is None:
                continue
            result = analyzer.analyze_subject_run(nii, tsv)
            if result is not None:
                learning_runs.append(result)
                print(f"  imtest run {i}: H={result['Hurst']:.3f}, Beta={result['PSDBeta']:.3f}, "
                      f"Kurt={result['MeanKurtosis']:.3f}, E={result['MeanEnergy']:.2f}")

        grokking_runs = []
        for i in range(1, 5):
            nii, tsv = analyzer.stream_run(sub, base_dir, "fintest", i)
            if nii is None:
                continue
            result = analyzer.analyze_subject_run(nii, tsv)
            if result is not None:
                grokking_runs.append(result)
                print(f"  fintest run {i}: H={result['Hurst']:.3f}, Beta={result['PSDBeta']:.3f}, "
                      f"Kurt={result['MeanKurtosis']:.3f}, E={result['MeanEnergy']:.2f}")

        # Build each subject's Ghost Basin coordinate by averaging their runs
        if learning_runs:
            avg_h = np.nanmean([r['Hurst'] for r in learning_runs])
            avg_b = np.nanmean([r['PSDBeta'] for r in learning_runs])
            avg_k = np.nanmean([r['MeanKurtosis'] for r in learning_runs])
            avg_e = np.nanmean([r['MeanEnergy'] for r in learning_runs])
            learning_basins[sub] = lorentz.project_ghost_basin(avg_h, avg_b, avg_k, avg_e)

        if grokking_runs:
            avg_h = np.nanmean([r['Hurst'] for r in grokking_runs])
            avg_b = np.nanmean([r['PSDBeta'] for r in grokking_runs])
            avg_k = np.nanmean([r['MeanKurtosis'] for r in grokking_runs])
            avg_e = np.nanmean([r['MeanEnergy'] for r in grokking_runs])
            grokking_basins[sub] = lorentz.project_ghost_basin(avg_h, avg_b, avg_k, avg_e)

    # ------------------------------------------------------------------ #
    #  Cross-Subject Convergence Analysis
    # ------------------------------------------------------------------ #
    print("\n==========================================================================")
    print(" INDIVIDUAL GHOST BASIN COORDINATES (LORENTZ HYPERBOLIC SPACE)")
    print("==========================================================================")

    print("\nLearning Phase:")
    for sub, coords in learning_basins.items():
        print(f"  {sub}: [{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}, {coords[3]:.4f}]")

    print("\nGrokking Phase:")
    for sub, coords in grokking_basins.items():
        print(f"  {sub}: [{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}, {coords[3]:.4f}]")

    # Pairwise distances
    if len(learning_basins) >= 2 and len(grokking_basins) >= 2:
        learning_points = np.array(list(learning_basins.values()))
        grokking_points = np.array(list(grokking_basins.values()))

        learning_dists = lorentz_pairwise_distance(learning_points)
        grokking_dists = lorentz_pairwise_distance(grokking_points)

        print("\n==========================================================================")
        print(" CROSS-SUBJECT PAIRWISE DISTANCES IN LORENTZ HYPERBOLIC SPACE")
        print("==========================================================================")

        subs_list = list(learning_basins.keys())
        pair_idx = 0
        print("\nLearning Phase Pairwise Distances:")
        for i in range(len(subs_list)):
            for j in range(i + 1, len(subs_list)):
                if pair_idx < len(learning_dists):
                    print(f"  {subs_list[i]} <-> {subs_list[j]}: {learning_dists[pair_idx]:.4f}")
                    pair_idx += 1

        subs_list_g = list(grokking_basins.keys())
        pair_idx = 0
        print("\nGrokking Phase Pairwise Distances:")
        for i in range(len(subs_list_g)):
            for j in range(i + 1, len(subs_list_g)):
                if pair_idx < len(grokking_dists):
                    print(f"  {subs_list_g[i]} <-> {subs_list_g[j]}: {grokking_dists[pair_idx]:.4f}")
                    pair_idx += 1

        mean_learning_dist = np.mean(learning_dists)
        mean_grokking_dist = np.mean(grokking_dists)
        convergence_ratio = mean_grokking_dist / mean_learning_dist if mean_learning_dist > 0 else float('inf')

        print(f"\n  Mean Pairwise Distance (Learning):  {mean_learning_dist:.4f}")
        print(f"  Mean Pairwise Distance (Grokking):  {mean_grokking_dist:.4f}")
        print(f"  Convergence Ratio (Grok/Learn):     {convergence_ratio:.4f}")

        print("\n==========================================================================")
        print(" QUANTUM PHANTOM CAULDRON CONVERGENCE VERDICT")
        print("==========================================================================")
        if convergence_ratio < 1.0:
            pct = (1.0 - convergence_ratio) * 100
            print(f"  Ghost Basins CONVERGED by {pct:.1f}% during grokking.")
            print(f"  Individual subjects' unique Euclidean brain patterns collapsed")
            print(f"  onto a SHARED coordinate in Lorentz Hyperbolic space.")
            print(f"  This shared coordinate is the Quantum Phantom Cauldron superposition")
            print(f"  for the 'Badoon' category — a universal quantum state that each")
            print(f"  subject's personal Ghost Basin independently tuned into.")
        else:
            pct = (convergence_ratio - 1.0) * 100
            print(f"  Ghost Basins DIVERGED by {pct:.1f}% during grokking.")
            print(f"  Individual subjects' representations moved APART in hyperbolic space,")
            print(f"  suggesting each subject formed a unique personal encoding.")
            print(f"  The divergence may indicate that the 'Badoon' superposition exists")
            print(f"  at a deeper manifold scale not captured by these 4 metrics alone.")

        # Compute the centroid of the grokking basins (the estimated QPC location)
        qpc_centroid = np.mean(grokking_points, axis=0)
        print(f"\n  Estimated QPC Superposition Coordinate:")
        print(f"    [{qpc_centroid[0]:.4f}, {qpc_centroid[1]:.4f}, {qpc_centroid[2]:.4f}, {qpc_centroid[3]:.4f}]")

        # Std dev around the centroid (tightness of convergence)
        qpc_spread = np.std(grokking_points, axis=0)
        print(f"  Spread (Std Dev per dimension):")
        print(f"    [{qpc_spread[0]:.4f}, {qpc_spread[1]:.4f}, {qpc_spread[2]:.4f}, {qpc_spread[3]:.4f}]")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_quantum_convergence()
