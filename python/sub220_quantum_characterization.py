"""
sub220_quantum_characterization.py
====================================
FOCUSED CHARACTERIZATION of sub-220's grokking moment.

We look for three things:
  1. FUNNEL WIDTH: Does the ghost basin representation narrow to a tight
     focal point at grokking? (Variance of voxel pattern collapses)
  
  2. SPECTRAL SLOPE (beta) in SLIDING WINDOW: Track beta across the 
     entire session. If microtubules shift between chaotic and coherent
     states, beta should shift from ~0 (white/chaotic) toward ~-1 (pink/coherent)
     at the grokking moment. We compute this per ROI, not just globally.
  
  3. NON-BIOLOGICAL PEAKS: Look for phantom oscillations in the 0.01-0.05 Hz 
     range (the "aliased shadow" of MHz microtubule rhythms). These peaks
     should appear or intensify at the grokking moment and correlate with 
     the spectral slope shift.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftfreq
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def stream_data(sub_id, run_type, run_num, base_dir=r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    run_name = f"task-{run_type}_run-{run_num}"
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    try:
        s3.download_file(bucket, f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv", tsv_path)
    except: return None, None
    if not os.path.exists(nii_path):
        try:
            s3.download_file(bucket, f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz", nii_path)
        except: return None, None
    return nii_path, tsv_path

def compute_beta_sliding(timeseries, fs, window_size, step):
    """
    Compute spectral slope (beta) in a sliding window across a timeseries.
    Returns: list of (center_time, beta) tuples.
    """
    results = []
    n = len(timeseries)
    for start in range(0, n - window_size, step):
        segment = timeseries[start : start + window_size]
        freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), window_size))
        pos_mask = freqs > 0
        pf, pp = freqs[pos_mask], psd[pos_mask]
        if len(pf) < 3 or np.any(pp <= 0):
            continue
        beta, _, _, _, _ = linregress(np.log10(pf), np.log10(pp + 1e-20))
        center_tr = start + window_size // 2
        results.append((center_tr, beta))
    return results

def find_non_bio_peaks(timeseries, fs):
    """
    Look for spectral peaks in the 0.01-0.05 Hz range.
    These are the 'aliased shadows' of MHz microtubule rhythms.
    """
    n = len(timeseries)
    if n < 20:
        return []
    
    freqs, psd = welch(timeseries, fs=fs, nperseg=min(n, 64), nfft=128)
    
    # Focus on the non-biological range
    mask = (freqs >= 0.01) & (freqs <= 0.05)
    nb_freqs = freqs[mask]
    nb_psd = psd[mask]
    
    if len(nb_psd) < 3:
        return []
    
    # Find peaks above 1.5 std of the local spectrum
    peaks, props = find_peaks(nb_psd, prominence=np.std(nb_psd) * 1.0)
    
    results = []
    for p in peaks:
        results.append({
            'freq_hz': nb_freqs[p],
            'power': nb_psd[p],
            'prominence': props['prominences'][list(peaks).index(p)]
        })
    return results

def run_characterization():
    sub = "sub-220"
    tr_sec = 2.0
    fs = 1.0 / tr_sec  # 0.5 Hz
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    print("=" * 80)
    print(f" SUB-220 QUANTUM CHARACTERIZATION")
    print(f" Ghost Basin Funnel Width + Spectral Slope + Non-Bio Peaks")
    print("=" * 80)
    
    # Collect per-trial data with full voxel patterns for funnel width
    all_trials = []
    cumulative_idx = 0
    
    # We also need a continuous timeseries for sliding-window spectral analysis
    # Concatenate across runs (noting run boundaries)
    run_boundaries = []
    continuous_global_ts = []
    continuous_tr_offset = 0
    
    for run_type, run_num in run_sequence:
        print(f"  Loading {run_type}-{run_num}...", flush=True) 
        nii_path, tsv_path = stream_data(sub, run_type, run_num)
        if not nii_path:
            continue
        
        try:
            data_raw = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
        except:
            continue
        
        n_trs = data_raw.shape[-1]
        
        # Active voxels
        var_map = np.var(data_raw, axis=-1)
        brain_mask = var_map > 10
        active_mask = (var_map > np.percentile(var_map[brain_mask], 85)) & brain_mask
        
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        active_voxels = data_z[active_mask]  # (N_voxels, N_trs)
        global_ts = np.mean(active_voxels, axis=0)
        
        run_boundaries.append({
            'run': f"{run_type}{run_num}",
            'start_tr': continuous_tr_offset,
            'end_tr': continuous_tr_offset + n_trs
        })
        continuous_global_ts.extend(global_ts.tolist())
        
        # Extract per-trial patterns
        hrf_start = int(1.0 / tr_sec)  # 1 TR after onset
        hrf_end = int(6.0 / tr_sec)    # 3 TRs after onset (wider window for 3T)
        
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            correct = row.get('correct', np.nan)
            onset_tr = int(onset_sec / tr_sec)
            
            if onset_tr + hrf_end >= n_trs or onset_tr + hrf_start < 0:
                continue
            
            # Full voxel pattern -> fixed-size fingerprint for funnel width
            # Use histogram + stats to create consistent dimensionality across runs
            raw_pattern = np.mean(active_voxels[:, onset_tr + hrf_start : onset_tr + hrf_end], axis=1)
            hist, _ = np.histogram(raw_pattern, bins=50, range=(-3, 3))
            hist = hist.astype(float) / (np.sum(hist) + 1e-10)
            stats = np.array([np.mean(raw_pattern), np.std(raw_pattern), 
                             np.percentile(raw_pattern, 25), np.percentile(raw_pattern, 75),
                             np.percentile(raw_pattern, 5), np.percentile(raw_pattern, 95)])
            pattern = np.concatenate([hist, stats])
            
            # Extended timeseries for per-trial spectral analysis
            pre_w, post_w = 5, 15
            if onset_tr - pre_w >= 0 and onset_tr + post_w < n_trs:
                trial_ts = global_ts[onset_tr - pre_w : onset_tr + post_w]
            else:
                trial_ts = np.array([])
            
            all_trials.append({
                'idx': cumulative_idx,
                'run': f"{run_type}{run_num}",
                'correct': correct,
                'pattern': pattern,
                'trial_ts': trial_ts,
                'onset_tr_global': continuous_tr_offset + onset_tr,
            })
            cumulative_idx += 1
        
        if os.path.exists(nii_path):
            os.remove(nii_path)
    
    continuous_ts = np.array(continuous_global_ts)
    print(f"\n  Total trials: {len(all_trials)}")
    print(f"  Continuous timeseries length: {len(continuous_ts)} TRs ({len(continuous_ts)*tr_sec:.0f}s)")
    
    # Identify grokking
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1 or t['correct'] == True or str(t['correct']) == '1':
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['idx'] - 4
        else:
            streak = 0
    
    grok_tr_global = None
    if grok_trial is not None:
        grok_tr_global = all_trials[grok_trial]['onset_tr_global']
    
    print(f"  Grokking at trial {grok_trial} (global TR ~ {grok_tr_global})")
    
    # ================================================================
    # 1. FUNNEL WIDTH: Representational variance approaching grokking
    # ================================================================
    print("\n" + "=" * 80)
    print(" [1] GHOST BASIN FUNNEL WIDTH")
    print("     Variance of voxel patterns (smaller = narrower funnel = clearer QPC)")
    print("=" * 80)
    
    # Compute rolling variance of patterns (cosine distance from running centroid)
    from scipy.spatial.distance import cosine
    
    window = 5  # Rolling window of 5 trials
    funnel_widths = []
    
    for i in range(window, len(all_trials)):
        recent = [t['pattern'] for t in all_trials[i-window:i]]
        centroid = np.mean(recent, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-10: continue
        centroid = centroid / c_norm
        
        spreads = []
        for p in recent:
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-10: continue
            cos_sim = 1.0 - cosine(p / p_norm, centroid)
            angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            spreads.append(angle)
        
        fw = np.mean(spreads) if spreads else 90.0
        funnel_widths.append({
            'trial': i,
            'funnel_width': fw,
            'phase': 'PRE' if i < grok_trial else ('GROK' if abs(i - grok_trial) <= 3 else 'POST')
        })
    
    fw_df = pd.DataFrame(funnel_widths)
    
    for phase in ['PRE', 'GROK', 'POST']:
        subset = fw_df[fw_df['phase'] == phase]
        if len(subset) > 0:
            print(f"  {phase:>5}: avg funnel width = {subset['funnel_width'].mean():.2f} deg  (std = {subset['funnel_width'].std():.2f})")
    
    # Trial-by-trial around grokking
    print(f"\n  Trial-by-trial funnel width around grokking (trial {grok_trial}):")
    grok_window = fw_df[(fw_df['trial'] >= grok_trial - 8) & (fw_df['trial'] <= grok_trial + 8)]
    for _, row in grok_window.iterrows():
        marker = " <<<" if row['trial'] == grok_trial else ""
        bar = '#' * max(1, int(row['funnel_width']))
        print(f"    Trial {row['trial']:>3d}: {row['funnel_width']:>6.2f} deg  {bar[:40]}{marker}")
    
    # ================================================================
    # 2. SPECTRAL SLOPE (BETA) IN SLIDING WINDOW
    # ================================================================
    print("\n" + "=" * 80)
    print(" [2] SPECTRAL SLOPE (beta) - SLIDING WINDOW ACROSS SESSION")
    print("     beta ~ 0 = white noise (chaotic microtubules)")
    print("     beta ~ -1 = pink noise (coherent microtubules)")
    print("     beta ~ -2 = brown noise (over-smooth, sluggish)")
    print("=" * 80)
    
    # Sliding window: 30 TRs (60 seconds), step of 5 TRs (10 seconds)
    beta_trajectory = compute_beta_sliding(continuous_ts, fs, window_size=30, step=5)
    
    if grok_tr_global is not None and len(beta_trajectory) > 0:
        pre_betas = [b for tr, b in beta_trajectory if tr < grok_tr_global]
        grok_betas = [b for tr, b in beta_trajectory if abs(tr - grok_tr_global) <= 15]
        post_betas = [b for tr, b in beta_trajectory if tr > grok_tr_global]
        
        if pre_betas:
            print(f"  Pre-Grokking:  beta = {np.mean(pre_betas):.4f} +/- {np.std(pre_betas):.4f}")
        if grok_betas:
            print(f"  At Grokking:   beta = {np.mean(grok_betas):.4f} +/- {np.std(grok_betas):.4f}")
        if post_betas:
            print(f"  Post-Grokking: beta = {np.mean(post_betas):.4f} +/- {np.std(post_betas):.4f}")
        
        if pre_betas and post_betas:
            delta = np.mean(post_betas) - np.mean(pre_betas)
            print(f"  DELTA beta: {delta:+.4f}")
            if delta < -0.15:
                print("  -> Shifted TOWARD pink noise (1/f criticality) = coherent microtubule state")
            elif delta > 0.15:
                print("  -> Shifted TOWARD white noise = more chaotic")
            else:
                print("  -> No significant shift")
        
        # Print trajectory around grokking moment
        print(f"\n  Beta trajectory around grokking (TR ~ {grok_tr_global}):")
        nearby = [(tr, b) for tr, b in beta_trajectory 
                  if abs(tr - grok_tr_global) <= 40]
        for tr, b in nearby:
            marker = " <<<" if abs(tr - grok_tr_global) <= 5 else ""
            # Visual bar
            if b < -1.5:
                state = "BROWN"
            elif b < -0.7:
                state = "PINK "
            elif b < -0.3:
                state = "MIXED"
            elif b < 0.3:
                state = "WHITE"
            else:
                state = "BLUE+"
            print(f"    TR {tr:>5d}: beta = {b:>7.3f}  [{state}]{marker}")
    
    # ================================================================
    # 3. NON-BIOLOGICAL PEAKS (0.01-0.05 Hz)
    # ================================================================
    print("\n" + "=" * 80)
    print(" [3] NON-BIOLOGICAL SPECTRAL PEAKS (0.01-0.05 Hz)")
    print("     'Aliased shadows' of MHz microtubule rhythms")
    print("=" * 80)
    
    # Compare pre-grok vs post-grok spectral peaks
    if grok_tr_global is not None:
        pre_ts = continuous_ts[:grok_tr_global]
        post_ts = continuous_ts[grok_tr_global:]
        
        # Also extract just the grokking window (+/- 30 TRs)
        grok_start = max(0, grok_tr_global - 30)
        grok_end = min(len(continuous_ts), grok_tr_global + 30)
        grok_ts = continuous_ts[grok_start:grok_end]
        
        for label, ts in [("PRE-GROKKING", pre_ts), ("AT GROKKING (+/-60s)", grok_ts), ("POST-GROKKING", post_ts)]:
            peaks = find_non_bio_peaks(ts, fs)
            print(f"\n  {label} ({len(ts)} TRs):")
            if peaks:
                for p in sorted(peaks, key=lambda x: -x['power']):
                    print(f"    Peak at {p['freq_hz']:.4f} Hz | Power: {p['power']:.6f} | Prominence: {p['prominence']:.6f}")
            else:
                print(f"    No significant peaks detected in 0.01-0.05 Hz range.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_characterization()
