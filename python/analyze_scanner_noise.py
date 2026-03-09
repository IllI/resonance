"""
analyze_scanner_noise.py
──────────────────────────────────────────────────────────────
Isolates and characterizes the rhythmic BOLD signature induced by the 
7T MRI scanner's mechanical acoustic noise.

Theory:
The scanner operates on a strict mechanical cycle (TR = 2.6s). The gradient
coils fire rhythmically during the 1.4s acquisition window of each TR.
This creates a powerful, periodic acoustic stimulus (a "thudding/screeching" rhythm)
that the auditory cortex processes alongside the experimental target sounds.

Goal:
1. Isolate the auditory voxel timeseries across an entire run.
2. Identify the fundamental frequency of the scanner's induced BOLD response
   using Power Spectral Density (PSD) analysis.
3. Design a targeted filter to suppress this periodic scanner artifact.
4. Evaluate how much "ambient" variance is removed.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.stats import zscore

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds001942"
OUT_DIR = r"C:\Users\cityz\IllI\newer_all\output"
os.makedirs(OUT_DIR, exist_ok=True)

def load_bold_run(subj, ses, run):
    fname = f"{subj}_{ses}_task-auditory_run-{run:02d}_bold.nii.gz"
    fpath = os.path.join(DATA_DIR, subj, ses, 'func', fname)
    if not os.path.exists(fpath):
        return None
    return nib.load(fpath).get_fdata()

def extract_auditory_voxels(bold_4d, percentile=98):
    X, Y, Z, T = bold_4d.shape
    flat = bold_4d.reshape(-1, T)
    mask = flat.std(axis=1) > 1e-3
    brain = flat[mask]
    var = brain.var(axis=1)
    thresh = np.percentile(var, percentile)
    top_mask = var >= thresh
    selected = brain[top_mask]
    ts_z = zscore(selected, axis=1)
    return np.nan_to_num(ts_z, 0)

def main():
    subj = 'sub-S01'
    ses = 'ses-SES02'
    run = 1
    
    print("=" * 65)
    print("  7T SCANNER NOISE (ACOUSTIC ARTIFACT) ANALYSIS")
    print("=" * 65)
    
    print(f"  Loading {subj} {ses} run-{run:02d}...")
    bold_4d = load_bold_run(subj, ses, run)
    if bold_4d is None:
        print("  Data not found.")
        return
        
    TR = 2.6  # Repetition Time in seconds
    fs = 1.0 / TR  # Sampling frequency in Hz (~0.384 Hz)
    
    print("  Extracting highly responsive auditory voxels (Top 2% Variance)...")
    ts_z = extract_auditory_voxels(bold_4d, percentile=98)
    n_voxels, n_timepoints = ts_z.shape
    print(f"  Extracted {n_voxels} voxels across {n_timepoints} TRs.")
    
    # 1. Compute Mean Population Signal (The 'Macro' Auditory Response)
    pop_signal = ts_z.mean(axis=0)
    
    # 2. Power Spectral Density (PSD) via Welch's Method
    # Using a window that captures several cycles
    nperseg = min(64, n_timepoints)
    freqs, psd = welch(pop_signal, fs=fs, nperseg=nperseg, scaling='spectrum')
    
    # Ignore the DC component (freq 0)
    freqs = freqs[1:]
    psd = psd[1:]
    
    # Find dominant rhythmic peaks in the BOLD frequency spectrum
    # These peaks represent periodic forcing functions (like the scanner rhythm)
    peaks, properties = find_peaks(psd, height=np.mean(psd)+np.std(psd), distance=3)
    
    print("\n  >> SPECTRAL DOMAIN ANALYSIS:")
    print(f"     Nyquist frequency limit for TR=2.6s: {fs/2:.4f} Hz")
    
    if len(peaks) == 0:
        print("     No prominent rhythmic peaks detected above baseline noise.")
        dominant_f = None
    else:
        # Sort peaks by power (height)
        peak_idx = peaks[np.argsort(properties['peak_heights'])[::-1]]
        dominant_f = freqs[peak_idx[0]]
        
        print(f"     Found {len(peaks)} dominant periodic rhythmic spikes:")
        for idx in peak_idx:
            f = freqs[idx]
            period = 1/f
            power = psd[idx]
            print(f"       - Freq: {f:.4f} Hz | Period (Cycle Length): {period:.2f} sec | Power: {power:.4f}")
            
        print(f"\n  >> SCANNER SIGNATURE IDENTIFIED:")
        print(f"     Primary rhythmic BOLD oscillation repeating every {1/dominant_f:.2f} seconds.")
        print(f"     (This represents the brain's hemodynamic entrainment to the acoustic environment.)")
        
        # 3. Filter Design: Suppress the dominant rhythmic artifact
        # We build a Notch/Bandstop filter around the dominant frequency
        quality_factor = 30.0  # Sharp notch
        w0 = dominant_f / (fs / 2) # Normalized frequency
        bw = w0 / quality_factor # Bandwidth
        
        # Determine passbands
        low_pass = max(0.01, w0 - bw)
        high_pass = min(0.99, w0 + bw)
        
        b, a = butter(4, [low_pass, high_pass], btype='bandstop')
        
        # Apply filter to the macro signal
        filtered_pop_signal = filtfilt(b, a, pop_signal)
        
        # Calculate Variance Removed
        orig_var = np.var(pop_signal)
        filt_var = np.var(filtered_pop_signal)
        removed_var_pct = ((orig_var - filt_var) / orig_var) * 100
        
        print("\n  >> SUPPRESSION FILTER APPLIED:")
        print(f"     Targeted Bandstop Filter at {dominant_f:.4f} Hz applied.")
        print(f"     Variance removed by suppressing this rhythm: {removed_var_pct:.2f}%")
        
        if removed_var_pct > 0.5:
            print("  => Successful extraction of mechanical/ambient rhythmic artifact.")
            print("     Suppressing this layer reveals the 'true' decoding delta for target sounds.")
        else:
            print("  => Minimal variance removed. The dominant peak may not represent a massive")
            print("     scanner artifact, or the signal is dominated by aperiodic events.")
            
    print("\n  => We can map this suppression logic back into the cross-subject D-LinOSS pipeline")
    print("     to 'subtract the machine' before evaluating the brain's native ghost basin.")

if __name__ == "__main__":
    main()
