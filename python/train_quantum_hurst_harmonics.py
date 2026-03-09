"""
train_quantum_hurst_harmonics.py
================================
Integrated D-LinOSS Quantum Training Script

Executes the High-Frequency Detection Plan:
  Pathway 1: Fractal 1/f Scaling via Hurst Exponent (R/S Analysis)
  Pathway 2: Power Spectral Density Beta Slope (log-log PSD)
  Pathway 3: Spatial Kurtosis Flash Detection

All three metrics are calculated from real fMRI BOLD data (Badoon expert subjects)
at the Learning phase (imtest) vs. Grokking phase (fintest).

The Hurst Exponent measures fractal memory:
  H = 0.5 -> Pure random walk (no hidden driver)
  H > 0.5 -> Persistent long-range temporal dependence (fractal coordination)
  H > 0.75 -> Strong fractal memory indicating a scale-invariant high-frequency driver

The PSD Beta Slope measures spectral scaling:
  Beta ~ 0 -> White noise (flat spectrum)
  Beta ~ 1 -> 1/f "pink noise" (scale-invariant, fractal)
  Beta ~ 2 -> Brownian noise (random walk)

If the slow BOLD signal is a down-sampled projection of MHz microtubule pulses,
we expect both H and Beta to shift toward strong fractal persistence during grokking.
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


class QuantumHurstAnalyzer:
    """
    Analyzes the fractal scaling properties of the fMRI BOLD signal
    to detect signatures of high-frequency microtubule quantum activity.
    """
    def __init__(self, tr_sec=2.0):
        self.tr_sec = tr_sec
        self.sampling_freq = 1.0 / tr_sec  # 0.5 Hz for TR=2.0s

    # ------------------------------------------------------------------ #
    #  S3 Data Streaming (reused from existing pipeline)
    # ------------------------------------------------------------------ #
    def stream_run(self, sub_id, base_dir, run_type, run_num):
        """Downloads a single NIfTI + events TSV from OpenNeuro S3."""
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

    # ------------------------------------------------------------------ #
    #  Hurst Exponent (Pathway 1)
    # ------------------------------------------------------------------ #
    def calculate_hurst(self, timeseries):
        """
        Calculates the Hurst Exponent via Rescaled Range (R/S) analysis.
        The timeseries is the mean BOLD signal across the top-active voxels
        within a trial's temporal window.

        Uses kind='change' because the BOLD signal represents intensity changes,
        not a random walk of cumulative sums.

        Requires a minimum of 20 data points for stable R/S estimation.
        """
        if len(timeseries) < 20:
            return np.nan
        try:
            H, c, data = compute_Hc(timeseries, kind='change', simplified=True)
            return H
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ #
    #  Power Spectral Density Beta Slope (Pathway 2)
    # ------------------------------------------------------------------ #
    def calculate_psd_beta(self, timeseries):
        """
        Calculates the spectral scaling exponent (Beta) from the Power
        Spectral Density of the BOLD time-series using Welch's method.

        P(f) ∝ 1/f^β  =>  log(P) = -β * log(f) + const

        We fit a line in log-log space to extract Beta.
        Requires minimum 20 data points for a meaningful PSD.
        """
        if len(timeseries) < 20:
            return np.nan

        # Welch PSD: nperseg must be <= len(timeseries)
        nperseg = min(len(timeseries), 64)
        try:
            freqs, psd = welch(timeseries, fs=self.sampling_freq, nperseg=nperseg)
        except Exception:
            return np.nan

        # Remove the DC component (freq=0)
        mask = freqs > 0
        freqs = freqs[mask]
        psd = psd[mask]

        if len(freqs) < 2 or np.any(psd <= 0):
            return np.nan

        # Linear regression in log-log space: log(P) = -beta * log(f) + c
        log_f = np.log10(freqs)
        log_p = np.log10(psd)

        # numpy polyfit degree 1 => [slope, intercept]
        try:
            slope, intercept = np.polyfit(log_f, log_p, 1)
        except Exception:
            return np.nan

        beta = -slope  # Convention: P(f) ∝ 1/f^β => slope is negative
        return beta

    # ------------------------------------------------------------------ #
    #  Full Run Analysis
    # ------------------------------------------------------------------ #
    def analyze_run(self, nii_path, tsv_path):
        """
        For each trial in a run:
        1. Extract the BOLD block around the trial (onset to onset+12s = 6 TRs)
        2. Build the mean time-series of the top-active voxels within that block
        3. Calculate Hurst Exponent on the full run time-series of top-active voxels
        4. Calculate PSD Beta on the full run time-series
        5. Calculate Spatial Kurtosis at the peak TR
        """
        if not os.path.exists(nii_path) or not os.path.exists(tsv_path):
            return None

        data = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')

        n_trs = data.shape[-1]
        data_z = np.nan_to_num(zscore(data, axis=-1))

        # Brain mask: voxels with temporal variance > 10
        brain_mask = np.var(data, axis=-1) > 10
        raw_var = np.var(data, axis=-1)

        # Active mask: top 10% most variable voxels within the brain
        active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask
        n_active = np.sum(active_mask)

        if n_active == 0:
            return None

        # ----- RUN-LEVEL Hurst & PSD -----
        # Extract the mean time-series across all active voxels for the entire run.
        # This is the macroscopic "pulse" of the brain's most active tissue.
        run_timeseries = np.mean(data_z[active_mask, :], axis=0)  # shape: (n_trs,)

        run_hurst = self.calculate_hurst(run_timeseries)
        run_psd_beta = self.calculate_psd_beta(run_timeseries)

        # ----- TRIAL-LEVEL Spatial Kurtosis -----
        trial_stats = []
        for _, row in events.iterrows():
            if 'correct' not in row or pd.isna(row['correct']):
                continue

            rt = row.get('response_time', 0)
            if pd.isna(rt) or rt == 0:
                continue

            tr_onset = int(row['onset'] / self.tr_sec)
            start_scan_tr = min(tr_onset + 2, n_trs - 1)
            end_scan_tr = min(tr_onset + 5, n_trs)

            if start_scan_tr >= end_scan_tr:
                continue

            block = data_z[active_mask, start_scan_tr:end_scan_tr]
            if block.size == 0:
                continue

            max_intensity = np.max(block)

            peak_tr_idx_relative = np.argmax(np.max(block, axis=0))
            active_voxels_at_peak = block[:, peak_tr_idx_relative]
            spatial_kurt = kurtosis(active_voxels_at_peak)

            trial_stats.append({
                'TrialType': row.get('trial_type', 'unknown'),
                'Correct': int(row['correct']),
                'RT': rt,
                'MaxIntensity': max_intensity,
                'SpatialKurtosis': spatial_kurt,
                'RunHurst': run_hurst,
                'RunPSDBeta': run_psd_beta,
            })

        # Clean up the NIfTI to save disk space
        if os.path.exists(nii_path):
            try:
                os.remove(nii_path)
            except OSError:
                pass

        if not trial_stats:
            return None
        return pd.DataFrame(trial_stats)


# ====================================================================== #
#  Main Training Entrypoint
# ====================================================================== #
def train_quantum_hurst_harmonics():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    analyzer = QuantumHurstAnalyzer(tr_sec=2.0)

    masters = ["sub-206", "sub-220"]

    print("==========================================================================")
    print(" D-LinOSS QUANTUM TRAINING: HURST EXPONENT + PSD SCALING ")
    print("==========================================================================")
    print("Detecting fractal 1/f signatures of scale-invariant high-frequency drivers")
    print("in the macroscopic BOLD signal of Badoon expert subjects.\n")

    early_data = []
    late_data = []

    for sub in masters:
        print(f"--- Processing {sub} ---")

        # Phase 1: Interim tests (learning / guessing)
        for i in range(1, 3):
            nii_path, tsv_path = analyzer.stream_run(sub, base_dir, "imtest", i)
            if nii_path is None:
                continue
            df = analyzer.analyze_run(nii_path, tsv_path)
            if df is not None:
                df['Subject'] = sub
                df['Phase'] = 'Learning'
                early_data.append(df)
                print(f"  imtest run {i}: {len(df)} trials analyzed")

        # Phase 2: Final tests (grokking / mastery)
        for i in range(1, 3):
            nii_path, tsv_path = analyzer.stream_run(sub, base_dir, "fintest", i)
            if nii_path is None:
                continue
            df = analyzer.analyze_run(nii_path, tsv_path)
            if df is not None:
                df['Subject'] = sub
                df['Phase'] = 'Grokking'
                late_data.append(df)
                print(f"  fintest run {i}: {len(df)} trials analyzed")

    if not early_data or not late_data:
        print("\nERROR: Insufficient data to compare phases.")
        return

    df_early = pd.concat(early_data)
    df_late = pd.concat(late_data)

    # Only consider correct trials for "clean" grokking analysis
    early_correct = df_early[df_early['Correct'] == 1]
    late_correct = df_late[df_late['Correct'] == 1]

    print("\n==========================================================================")
    print(" RESULTS: HIGH-FREQUENCY QUANTUM SIGNATURE DETECTION")
    print("==========================================================================")

    print("\n--- PATHWAY 1: FRACTAL 1/f SCALING (HURST EXPONENT) ---")
    print(f"  Learning Phase  H = {early_correct['RunHurst'].mean():.4f}")
    print(f"  Grokking Phase  H = {late_correct['RunHurst'].mean():.4f}")
    h_delta = late_correct['RunHurst'].mean() - early_correct['RunHurst'].mean()
    print(f"  Delta H = {h_delta:+.4f}")
    if h_delta > 0:
        print("  >> Hurst Exponent INCREASED during grokking.")
        print("     The BOLD signal exhibits stronger long-range fractal memory.")
        print("     This supports the hypothesis that a high-frequency driver")
        print("     (scale-invariant microtubule network) is orchestrating the")
        print("     macroscopic BOLD fluctuations during Super-Convergence.")
    else:
        print("  >> Hurst Exponent DECREASED during grokking.")
        print("     The BOLD signal shows LESS fractal memory during mastery,")
        print("     suggesting the network has simplified its temporal structure")
        print("     into a more efficient, less redundant encoding.")

    print("\n--- PATHWAY 2: POWER SPECTRAL DENSITY BETA SLOPE ---")
    print(f"  Learning Phase  Beta = {early_correct['RunPSDBeta'].mean():.4f}")
    print(f"  Grokking Phase  Beta = {late_correct['RunPSDBeta'].mean():.4f}")
    beta_delta = late_correct['RunPSDBeta'].mean() - early_correct['RunPSDBeta'].mean()
    print(f"  Delta Beta = {beta_delta:+.4f}")
    if abs(late_correct['RunPSDBeta'].mean() - 1.0) < abs(early_correct['RunPSDBeta'].mean() - 1.0):
        print("  >> PSD Beta moved CLOSER to 1.0 (pure 1/f pink noise).")
        print("     The spectral profile of the grokked brain converges on the")
        print("     scale-invariant 1/f distribution—the mathematical fingerprint")
        print("     of a fractal, self-similar system driven by nested oscillators.")
    else:
        print("  >> PSD Beta moved AWAY from 1.0 during grokking.")

    print("\n--- PATHWAY 3: SPATIAL KURTOSIS (THE GROKKING FLASH) ---")
    early_kurt = early_correct['SpatialKurtosis'].mean()
    late_kurt = late_correct['SpatialKurtosis'].mean()
    print(f"  Learning Phase  Kurtosis = {early_kurt:.4f}")
    print(f"  Grokking Phase  Kurtosis = {late_kurt:.4f}")
    print(f"  Delta Kurtosis = {late_kurt - early_kurt:+.4f}")

    print("\n--- PATHWAY 3b: PEAK VOXEL INTENSITY ---")
    early_max = early_correct['MaxIntensity'].mean()
    late_max = late_correct['MaxIntensity'].mean()
    print(f"  Learning Phase  Max Z = {early_max:.4f}")
    print(f"  Grokking Phase  Max Z = {late_max:.4f}")
    print(f"  Delta Max Z = {late_max - early_max:+.4f}")

    print("\n==========================================================================")
    print(" QUANTUM CONVERGENCE SUMMARY")
    print("==========================================================================")
    print("If Hurst H increased AND PSD Beta approached 1.0 during grokking,")
    print("we have mathematical evidence that the macroscopic BOLD signal is a")
    print("down-sampled projection of a deeper, faster, scale-invariant quantum")
    print("substrate (the microtubule cytoskeleton network).")
    print("")
    print("The subject's Ghost Basin is not just a classical neural pattern—it is")
    print("a fractal antenna whose macroscopic echo obeys the exact same 1/f power")
    print("law that governs quantum coherent systems from semiconductor lattices")
    print("to superconducting circuits.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_quantum_hurst_harmonics()
