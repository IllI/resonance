"""
frequency_envelope_correlation.py
==================================
Honest narrow-band frequency decomposition of sub-002's 7T data.

At TR=125ms, Nyquist = 4 Hz. We can resolve:
  - Hemodynamic Band:   0.01 - 0.5 Hz  (blood flow, astrocytes - EXPECTED)
  - Neurogenic Band:    0.5  - 2.0 Hz  (autonomic, breathing artifacts - EXPECTED)
  - Anomalous Band:     2.0  - 4.0 Hz  (Should NOT exist in a metabolic signal)

For each trial we:
  1. Compute the power spectrum in a 4-second post-stimulus window
  2. Check for narrow-band spectral peaks (spikes at specific Hz)
  3. Correlate: Do Initial Dip trials show different frequency content than non-Dip trials?
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, ttest_ind, mannwhitneyu
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def stream_data(sub_id="sub-002", run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    run_name = f"task-main_run-{run_num}"
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    if not os.path.exists(tsv_path):
        try: s3.download_file(bucket, f"ds006661/{sub_id}/func/{sub_id}_{run_name}_events.tsv", tsv_path)
        except: return None, None
    if not os.path.exists(nii_path):
        try: s3.download_file(bucket, f"ds006661/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz", nii_path)
        except: return None, None
    return nii_path, tsv_path

def run_frequency_envelope_analysis():
    tr_sec = 0.125
    fs = 1.0 / tr_sec  # 8 Hz sampling rate
    nyquist = fs / 2.0  # 4 Hz
    sub = "sub-002"
    runs = ["01", "02", "03", "04"]
    
    print("=" * 80)
    print(" NARROW-BAND FREQUENCY DECOMPOSITION + ENVELOPE CORRELATION")
    print(f" Subject: {sub} | Sampling Rate: {fs} Hz | Nyquist: {nyquist} Hz")
    print("=" * 80)
    
    # Frequency bands
    hemo_band = (0.01, 0.5)   # Blood flow / astrocytes
    neuro_band = (0.5, 2.0)   # Neurogenic / autonomic
    anomalous_band = (2.0, 4.0)  # Should NOT exist in metabolic signal
    
    trial_records = []
    
    for run in runs:
        print(f"\n  Processing Run {run}...", flush=True)
        nii_path, tsv_path = stream_data(sub, run)
        if not nii_path:
            continue
            
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        n_trs = data_raw.shape[-1]
        
        # Top 15% variance voxels
        var_map = np.var(data_raw, axis=-1)
        threshold = np.percentile(var_map, 85)
        brain_mask = var_map > threshold
        active_voxels = data_raw[brain_mask]
        data_z = np.nan_to_num(zscore(active_voxels, axis=1))
        
        # Global timeseries
        global_ts = np.mean(data_z, axis=0)
        
        image_events = events[events['trial_type'] == 'image'].copy()
        
        for _, row in image_events.iterrows():
            onset_sec = row['onset']
            cat = row['image_category']
            onset_tr = int(onset_sec / tr_sec)
            
            # 4-second post-stimulus window = 32 TRs
            window_trs = 32
            pre_trs = 4
            dip_end_trs = int(1.5 / tr_sec)
            
            if onset_tr - pre_trs < 0 or onset_tr + window_trs >= n_trs:
                continue
            
            # --- ENVELOPE: Initial Dip Detection ---
            baseline = np.mean(global_ts[onset_tr - pre_trs : onset_tr])
            dip_segment = global_ts[onset_tr : onset_tr + dip_end_trs]
            dip_magnitude = np.min(dip_segment) - baseline
            has_dip = dip_magnitude < -0.05
            
            # --- FREQUENCY DECOMPOSITION ---
            # Use the 4-second post-stimulus window
            signal_window = global_ts[onset_tr : onset_tr + window_trs]
            
            # Welch PSD for smooth spectral estimate
            freqs_w, psd_w = welch(signal_window, fs=fs, nperseg=min(16, len(signal_window)), 
                                    noverlap=8, nfft=64)
            
            # Band powers
            hemo_mask = (freqs_w >= hemo_band[0]) & (freqs_w <= hemo_band[1])
            neuro_mask = (freqs_w >= neuro_band[0]) & (freqs_w <= neuro_band[1])
            anom_mask = (freqs_w >= anomalous_band[0]) & (freqs_w <= anomalous_band[1])
            
            total_power = np.sum(psd_w[freqs_w > 0]) + 1e-20
            hemo_power = np.sum(psd_w[hemo_mask]) / total_power
            neuro_power = np.sum(psd_w[neuro_mask]) / total_power
            anom_power = np.sum(psd_w[anom_mask]) / total_power
            
            # --- NARROW-BAND PEAK DETECTION ---
            # Look for sharp spectral peaks (narrow-band spikes) in the anomalous band
            # A peak = a specific frequency with disproportionate power
            anom_freqs = freqs_w[anom_mask]
            anom_psd = psd_w[anom_mask]
            
            has_narrow_peak = False
            peak_freq = np.nan
            peak_prominence = np.nan
            
            if len(anom_psd) >= 3:
                peaks, props = find_peaks(anom_psd, prominence=np.std(anom_psd) * 1.5)
                if len(peaks) > 0:
                    has_narrow_peak = True
                    best_peak = peaks[np.argmax(props['prominences'])]
                    peak_freq = anom_freqs[best_peak]
                    peak_prominence = props['prominences'][np.argmax(props['prominences'])]
            
            # --- RAW FFT for per-voxel analysis ---
            # Check if the anomalous band power is concentrated in specific voxels
            # (localized quantum event) vs spread across the brain (global noise)
            voxel_window = data_z[:, onset_tr : onset_tr + window_trs]
            
            # Sample 500 random voxels for speed
            n_sample = min(500, voxel_window.shape[0])
            sample_idx = np.random.choice(voxel_window.shape[0], n_sample, replace=False)
            voxel_anom_powers = []
            
            for vi in sample_idx:
                vf, vp = welch(voxel_window[vi], fs=fs, nperseg=min(16, window_trs), 
                               noverlap=8, nfft=64)
                v_anom_mask = (vf >= anomalous_band[0]) & (vf <= anomalous_band[1])
                v_total = np.sum(vp[vf > 0]) + 1e-20
                voxel_anom_powers.append(np.sum(vp[v_anom_mask]) / v_total)
            
            voxel_anom_powers = np.array(voxel_anom_powers)
            
            # Gini coefficient: How concentrated is the anomalous power?
            # Higher Gini = power concentrated in few voxels (localized event)
            # Lower Gini = power spread everywhere (global noise/artifact)
            sorted_powers = np.sort(voxel_anom_powers)
            n = len(sorted_powers)
            cumulative = np.cumsum(sorted_powers)
            gini = (2.0 * np.sum((np.arange(1, n+1) * sorted_powers))) / (n * np.sum(sorted_powers) + 1e-20) - (n + 1) / n
            
            trial_records.append({
                'Run': run,
                'Category': cat,
                'Dip_Magnitude': dip_magnitude,
                'Has_Dip': has_dip,
                'Hemo_Power_Frac': hemo_power,
                'Neuro_Power_Frac': neuro_power,
                'Anomalous_Power_Frac': anom_power,
                'Has_Narrow_Peak': has_narrow_peak,
                'Peak_Freq_Hz': peak_freq,
                'Peak_Prominence': peak_prominence,
                'Spatial_Gini': gini,
            })
        
        # Cleanup NIfTI
        if os.path.exists(nii_path):
            os.remove(nii_path)
    
    df = pd.DataFrame(trial_records)
    
    # ===============================================================
    # RESULTS
    # ===============================================================
    dip_trials = df[df['Has_Dip'] == True]
    nodip_trials = df[df['Has_Dip'] == False]
    
    print(f"\n\nTotal Trials: {len(df)}")
    print(f"Trials WITH Initial Dip: {len(dip_trials)}")
    print(f"Trials WITHOUT Initial Dip: {len(nodip_trials)}")
    
    print("\n" + "=" * 80)
    print(" [1] FREQUENCY BAND POWER DISTRIBUTION")
    print("=" * 80)
    
    for label, subset in [("WITH DIP (Quantum Candidate)", dip_trials), ("NO DIP (Normal)", nodip_trials)]:
        print(f"\n  {label}:")
        print(f"    Hemodynamic Band (0.01-0.5 Hz):  {subset['Hemo_Power_Frac'].mean()*100:.1f}%")
        print(f"    Neurogenic Band  (0.5-2.0 Hz):   {subset['Neuro_Power_Frac'].mean()*100:.1f}%")
        print(f"    Anomalous Band   (2.0-4.0 Hz):   {subset['Anomalous_Power_Frac'].mean()*100:.1f}%")
    
    # Statistical test: Is anomalous power different between dip and no-dip trials?
    if len(nodip_trials) > 2:
        stat, p_val = mannwhitneyu(dip_trials['Anomalous_Power_Frac'], 
                                    nodip_trials['Anomalous_Power_Frac'], 
                                    alternative='greater')
        print(f"\n  Mann-Whitney U test (Anomalous Power: Dip > No-Dip): p = {p_val:.4f}")
        if p_val < 0.05:
            print("  -> SIGNIFICANT: Dip trials carry more anomalous (2-4 Hz) power!")
        else:
            print("  -> NOT SIGNIFICANT: No difference in anomalous power between dip and no-dip trials.")
    
    print("\n" + "=" * 80)
    print(" [2] NARROW-BAND SPECTRAL PEAKS (Non-Biological Spikes)")
    print("=" * 80)
    
    dip_peaks = dip_trials[dip_trials['Has_Narrow_Peak'] == True]
    nodip_peaks = nodip_trials[nodip_trials['Has_Narrow_Peak'] == True]
    
    print(f"  DIP trials with narrow-band peak:    {len(dip_peaks)} / {len(dip_trials)} ({100*len(dip_peaks)/max(1,len(dip_trials)):.1f}%)")
    print(f"  NO-DIP trials with narrow-band peak: {len(nodip_peaks)} / {len(nodip_trials)} ({100*len(nodip_peaks)/max(1,len(nodip_trials)):.1f}%)")
    
    if len(dip_peaks) > 0:
        print(f"\n  Peak frequencies in DIP trials:")
        freq_counts = dip_peaks['Peak_Freq_Hz'].value_counts().head(5)
        for freq, count in freq_counts.items():
            print(f"    {freq:.2f} Hz: {count} occurrences")
        
    if len(nodip_peaks) > 0:
        print(f"\n  Peak frequencies in NO-DIP trials:")
        freq_counts = nodip_peaks['Peak_Freq_Hz'].value_counts().head(5)
        for freq, count in freq_counts.items():
            print(f"    {freq:.2f} Hz: {count} occurrences")
    
    print("\n" + "=" * 80)
    print(" [3] SPATIAL CONCENTRATION (Localized vs Global)")
    print("=" * 80)
    
    print(f"  Avg Gini coefficient (DIP trials):    {dip_trials['Spatial_Gini'].mean():.4f}")
    print(f"  Avg Gini coefficient (NO-DIP trials): {nodip_trials['Spatial_Gini'].mean():.4f}")
    
    if len(nodip_trials) > 2:
        stat, p_val = mannwhitneyu(dip_trials['Spatial_Gini'], 
                                    nodip_trials['Spatial_Gini'], 
                                    alternative='greater')
        print(f"  Mann-Whitney U (Gini: Dip > No-Dip): p = {p_val:.4f}")
        if p_val < 0.05:
            print("  -> SIGNIFICANT: Anomalous power is MORE CONCENTRATED in dip trials (localized event)!")
        else:
            print("  -> NOT SIGNIFICANT: Power is equally spread in both conditions (likely global noise).")
    
    print("\n" + "=" * 80)
    print(" [4] HONEST VERDICT")
    print("=" * 80)
    
    anom_dip = dip_trials['Anomalous_Power_Frac'].mean()
    anom_nodip = nodip_trials['Anomalous_Power_Frac'].mean() if len(nodip_trials) > 0 else 0
    
    if anom_dip > anom_nodip * 1.2 and len(dip_peaks) > len(nodip_peaks):
        print("  DIP trials carry more anomalous high-frequency power AND more narrow-band peaks.")
        print("  This is CONSISTENT with a non-metabolic, high-frequency event co-occurring with")
        print("  the oxygen deficit. It could be a genuine quantum-like signature, or it could be")
        print("  a physiological artifact (cardiac/respiratory aliasing into the 2-4 Hz band).")
        print("  The cardiac/respiratory physio data would need to be regressed out to confirm.")
    elif anom_dip > anom_nodip * 1.1:
        print("  Slight elevation in anomalous power during dip trials, but not convincing.")
        print("  Most likely biological noise, not a quantum signature.")
    else:
        print("  No difference in frequency content between dip and no-dip trials.")
        print("  The Initial Dip is a standard metabolic oxygen deficit, not quantum.")
        print("  The frequency profile is identical whether or not the dip occurs.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_frequency_envelope_analysis()
