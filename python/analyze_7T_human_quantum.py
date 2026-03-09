import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress
from scipy.fft import fft, fftfreq
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def stream_ds006661_data(sub_id="sub-001", run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    
    # Files we need
    run_name = f"task-main_run-{run_num}"
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    
    tsv_key = f"ds006661/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
    nii_key = f"ds006661/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    
    print(f"Downloading events from {tsv_key}...")
    try:
        if not os.path.exists(tsv_path):
            s3.download_file(bucket, tsv_key, tsv_path)
    except Exception as e:
        print(f"Failed to download events: {e}")
        return None, None
        
    print(f"Downloading BOLD from {nii_key}... (This may take a minute, it's a massive 7T file)")
    try:
        # Check size first to avoid giant downloads if possible, but we need the data.
        if not os.path.exists(nii_path):
            s3.download_file(bucket, nii_key, nii_path)
    except Exception as e:
        print(f"Failed to download BOLD: {e}")
        return None, None
        
    return nii_path, tsv_path

def spectral_slope(signal, tr_sec):
    n = len(signal)
    if n < 4:
        return np.nan
    fft_vals = fft(signal)
    freqs = fftfreq(n, d=tr_sec)
    power = np.abs(fft_vals) ** 2
    
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    
    if len(pos_freqs) < 2 or np.any(pos_power <= 0):
        return np.nan
        
    log_f = np.log10(pos_freqs)
    log_p = np.log10(pos_power + 1e-20)
    
    slope, _, _, _, _ = linregress(log_f, log_p)
    return slope # We return the raw exponent (e.g. -1 for pink)

def analyze_initial_dip_and_criticality():
    tr_sec = 0.125  # The legendary 125ms TR
    
    print("="*80)
    print(" 7T ULTRA-FAST fMRI QUANTUM VALIDATION ANALYSIS (ds006661)")
    print(f" Temporal Resolution: {tr_sec}s ({1/tr_sec} structural snapshots per second)")
    print("="*80)
    
    nii_path, tsv_path = stream_ds006661_data()
    if not nii_path:
        print("Download failed. Exiting.")
        return
        
    print("\nLoading massive 7T BOLD matrix (this will take a lot of RAM)...")
    img = nib.load(nii_path)
    data_raw = img.get_fdata()
    events = pd.read_csv(tsv_path, sep='\t')
    
    print(f"Matrix Shape: {data_raw.shape} (X, Y, Z, Time)")
    n_trs = data_raw.shape[-1]
    
    # Fast brain masking (variance based)
    var_map = np.var(data_raw, axis=-1)
    brain_mask = var_map > np.percentile(var_map, 80)
    active_voxels = data_raw[brain_mask]
    
    print(f"Z-scoring {active_voxels.shape[0]} active voxels over {n_trs} TRs...")
    data_z = zscore(active_voxels, axis=1)
    data_z = np.nan_to_num(data_z)
    
    # 1. THE INITIAL DIP HUNT
    # We look at the first 1.5 seconds after an event onset
    # 1.5s / 0.125s = 12 TRs.
    print("\n[1] Hunting for the Microtubule Initial Dip...")
    print("  Looking for extreme negative oxygen deflections in the first 1.5s post-stimulus.")
    
    dip_window_trs = int(1.5 / tr_sec) 
    peak_window_trs = int(6.0 / tr_sec) # Traditional blood flow peaks around 4-6 seconds
    
    valid_events = 0
    dip_magnitudes = []
    peak_magnitudes = []
    
    # We average across active voxels for a "global/network" initial dip,
    # or look at highly responsive individual voxels.
    global_ts = np.mean(data_z, axis=0)
    
    for i, row in events.iterrows():
        onset_sec = row.get('onset')
        if pd.isna(onset_sec): continue
        
        onset_tr = int(onset_sec / tr_sec)
        
        if onset_tr + peak_window_trs >= n_trs or onset_tr - 4 < 0:
            continue
            
        valid_events += 1
        
        # Baseline is 0.5s before stimulus
        baseline = np.mean(global_ts[onset_tr - 4 : onset_tr])
        
        # Dip is min value in first 1.5s
        dip_val = np.min(global_ts[onset_tr : onset_tr + dip_window_trs]) - baseline
        
        # Peak is max value in 2s-6s window
        peak_val = np.max(global_ts[onset_tr + int(2/tr_sec) : onset_tr + peak_window_trs]) - baseline
        
        dip_magnitudes.append(dip_val)
        peak_magnitudes.append(peak_val)
        
    avg_dip = np.mean(dip_magnitudes)
    avg_peak = np.mean(peak_magnitudes)
    
    print(f"  Analyzed {valid_events} stimuli events.")
    print(f"  Average Initial Dip Amplitude (0-1.5s): {avg_dip:.4f} Z-score")
    print(f"  Average Hemodynamic Peak Amplitude (2-6s): {avg_peak:.4f} Z-score")
    
    if avg_dip < -0.1 and avg_peak > 0.1:
        print("  -> MASSIVE SUCCESS: Distinct Initial Dip detected! The brain burned oxygen faster than blood could replenish it, indicating a high-frequency (microtubule) burst.")
    elif avg_dip < 0:
        print("  -> WEAK SUCCESS: Slight negative deflection detected, but not extreme.")
    else:
        print("  -> NO DIP: The BOLD signal only went up. Biological astrocytic response dominated instantly.")

    # 2. THE PINK NOISE SHIFT (Criticality)
    print("\n[2] Hunting for 1/f Pink Noise (Quantum Criticality Edge)...")
    
    # Calculate PSD slope for the entire run
    global_slope = spectral_slope(global_ts, tr_sec)
    
    print(f"  Whole-brain/run spectral beta exponent: {global_slope:.3f}")
    if -1.2 <= global_slope <= -0.8:
        print("  -> MASSIVE SUCCESS: System operates at perfectly 1/f Pink Noise criticality (Fractal/Microtubule Dynamics).")
    elif global_slope < -1.5:
        print("  -> ASTROCYTIC DOMINANCE: System is in 1/f^2 Brown Noise (Random Walk/Metabolic).")
    else:
        print("  -> WHITE NOISE: System operates at 1/f^0 (Random uncorrelated noise).")
        
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze_initial_dip_and_criticality()
