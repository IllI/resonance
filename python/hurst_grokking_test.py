"""
hurst_grokking_test.py
=======================
FOCUSED TEST: Does the Hurst Exponent (H) and spectral slope (beta)
shift in sub-220's brain at the exact moment of grokking?

The Penrose-Hameroff prediction:
  - Microtubule restructuring changes the fractal timescale of neural activity
  - Before grokking: brain is in classical noise regime (H ~ 0.5, white noise)
  - During/after grokking: microtubules restructure, shifting the fractal
    dynamics toward 1/f criticality (H -> 0.75-1.0, beta -> -1.0)

The 3T scanner can detect this because:
  - S(t) = A_ast(t) * [1 + m * cos(2*pi*f_env*t)] + epsilon
  - The microtubule "beat frequency" f_env emerges as a shift in the PSD slope
  - Hurst Exponent H relates to beta by: H = (beta + 1) / 2

Protocol:
  1. Load sub-220's badoon experiment data (ds002813)
  2. Identify the exact grokking TR (where accuracy jumps to 100%)
  3. Compute H and beta in a SLIDING WINDOW across the entire session
  4. Test: Does H shift at the grokking moment?
  5. Compare: Does the same shift occur in non-grokking subjects?
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress
from scipy.signal import welch, detrend
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
    except: 
        return None, None
    if not os.path.exists(nii_path):
        try:
            s3.download_file(bucket, f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz", nii_path)
        except: 
            return None, None
    return nii_path, tsv_path

def compute_hurst_exponent(signal):
    """
    Compute the Hurst Exponent using Rescaled Range (R/S) analysis.
    H = 0.5: Random walk (white noise)
    H > 0.5: Persistent (trending, long-range correlations = fractal)
    H < 0.5: Anti-persistent (mean-reverting)
    H ~ 0.75-1.0: Strong 1/f criticality (Penrose prediction for quantum regime)
    """
    N = len(signal)
    if N < 10:
        return np.nan
    
    max_k = min(N // 2, 50)
    ns = []
    rs_vals = []
    
    for n in range(10, max_k + 1):
        # Divide into sub-series
        n_subseries = N // n
        if n_subseries < 1:
            continue
        rs_list = []
        for i in range(n_subseries):
            subseries = signal[i * n : (i + 1) * n]
            mean_sub = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean_sub)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(subseries, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if len(rs_list) > 0:
            ns.append(n)
            rs_vals.append(np.mean(rs_list))
    
    if len(ns) < 3:
        return np.nan
    
    log_n = np.log(ns)
    log_rs = np.log(rs_vals)
    slope, _, _, _, _ = linregress(log_n, log_rs)
    return slope

def compute_spectral_beta(signal, fs):
    """
    Compute the spectral slope beta from PSD.
    P(f) ~ f^beta
    beta ~ -1.0 = 1/f pink noise (criticality)
    beta ~ 0.0 = white noise
    beta ~ -2.0 = brown noise (overly smooth)
    """
    if len(signal) < 8:
        return np.nan
    
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 32), noverlap=None)
    
    pos_mask = freqs > 0
    pos_f = freqs[pos_mask]
    pos_p = psd[pos_mask]
    
    if len(pos_f) < 3 or np.any(pos_p <= 0):
        return np.nan
    
    log_f = np.log10(pos_f)
    log_p = np.log10(pos_p + 1e-20)
    
    slope, _, _, _, _ = linregress(log_f, log_p)
    return slope

def run_hurst_grokking_test():
    tr_sec = 2.0
    fs = 1.0 / tr_sec  # 0.5 Hz
    
    # Sub-220 = the grokker. Sub-206 = control (never grokked / different performance)
    subjects = ["sub-220", "sub-206"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    print("=" * 80)
    print(" HURST EXPONENT & SPECTRAL SLOPE AT THE GROKKING MOMENT")
    print(" Subject sub-220 (grokker) vs sub-206 (control)")
    print(" Penrose Prediction: H shifts toward 0.75-1.0 at grokking")
    print("=" * 80)
    
    for sub in subjects:
        print(f"\n{'='*80}")
        print(f" SUBJECT: {sub}")
        print(f"{'='*80}")
        
        # Collect ALL badoon trials chronologically with their BOLD timeseries
        all_trial_data = []
        cumulative_trial = 0
        
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_data(sub, run_type, run_num)
            if not nii_path:
                continue
            
            try:
                img = nib.load(nii_path)
                data_raw = img.get_fdata()
                events = pd.read_csv(tsv_path, sep='\t')
            except:
                continue
            
            n_trs = data_raw.shape[-1]
            
            # Active voxel mask
            var_map = np.var(data_raw, axis=-1)
            brain_mask = var_map > 10
            raw_var = np.var(data_raw, axis=-1)
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 85)) & brain_mask
            
            data_z = np.nan_to_num(zscore(data_raw, axis=-1))
            
            # Global timeseries from active voxels
            global_ts = np.mean(data_z[active_mask], axis=0)
            
            # Get badoon trial events
            for _, row in events.iterrows():
                if 'trial_type' in events.columns:
                    tt = row.get('trial_type', '')
                else:
                    tt = ''
                
                onset_sec = row.get('onset', np.nan)
                if pd.isna(onset_sec):
                    continue
                    
                onset_tr = int(onset_sec / tr_sec)
                
                # Get accuracy if available
                correct = row.get('correct', np.nan)
                rt = row.get('reaction_time', np.nan)
                
                # Extract a window of BOLD timeseries around the trial
                # -5 TRs to +10 TRs (10 seconds before to 20 seconds after)
                pre_window = 5
                post_window = 10
                
                if onset_tr - pre_window < 0 or onset_tr + post_window >= n_trs:
                    continue
                
                trial_bold = global_ts[onset_tr - pre_window : onset_tr + post_window]
                
                all_trial_data.append({
                    'trial_num': cumulative_trial,
                    'run': f"{run_type}{run_num}",
                    'onset_sec': onset_sec,
                    'correct': correct,
                    'rt': rt,
                    'bold_window': trial_bold,
                    'trial_type': tt,
                })
                cumulative_trial += 1
            
            # Cleanup NIfTI
            if os.path.exists(nii_path):
                os.remove(nii_path)
        
        if len(all_trial_data) == 0:
            print("  No trials found.")
            continue
        
        print(f"  Total trials extracted: {len(all_trial_data)}")
        
        # Identify the grokking moment
        # Find the first trial in a streak of >= 5 consecutive correct responses
        correct_streak = 0
        grok_trial = None
        for i, t in enumerate(all_trial_data):
            if t['correct'] == 1 or t['correct'] == True or str(t['correct']) == '1':
                correct_streak += 1
                if correct_streak >= 5 and grok_trial is None:
                    grok_trial = i - 4  # Start of the streak
            else:
                correct_streak = 0
        
        if grok_trial is None:
            print("  No grokking detected (no streak of 5+ correct).")
            grok_trial = len(all_trial_data) // 2  # Use midpoint as pseudo-reference
            print(f"  Using midpoint (trial {grok_trial}) as reference.")
        else:
            print(f"  GROKKING DETECTED at trial {grok_trial}")
        
        # Compute Hurst Exponent and Beta for each trial's BOLD window
        for t in all_trial_data:
            bold = t['bold_window']
            t['hurst'] = compute_hurst_exponent(bold)
            t['beta'] = compute_spectral_beta(bold, fs)
        
        # Split into PRE-GROKKING and POST-GROKKING
        pre_grok = [t for t in all_trial_data if t['trial_num'] < grok_trial]
        post_grok = [t for t in all_trial_data if t['trial_num'] >= grok_trial]
        
        # Also isolate the 5 trials immediately around grokking
        grok_window = [t for t in all_trial_data 
                       if abs(t['trial_num'] - grok_trial) <= 3]
        
        # Report
        print(f"\n  HURST EXPONENT (H):")
        
        pre_h = [t['hurst'] for t in pre_grok if not np.isnan(t['hurst'])]
        post_h = [t['hurst'] for t in post_grok if not np.isnan(t['hurst'])]
        grok_h = [t['hurst'] for t in grok_window if not np.isnan(t['hurst'])]
        
        if pre_h:
            print(f"    Pre-Grokking:       H = {np.mean(pre_h):.4f} +/- {np.std(pre_h):.4f}  (n={len(pre_h)})")
        if grok_h:
            print(f"    At Grokking (+/-3): H = {np.mean(grok_h):.4f} +/- {np.std(grok_h):.4f}  (n={len(grok_h)})")
        if post_h:
            print(f"    Post-Grokking:      H = {np.mean(post_h):.4f} +/- {np.std(post_h):.4f}  (n={len(post_h)})")
        
        if pre_h and post_h:
            delta_h = np.mean(post_h) - np.mean(pre_h)
            print(f"    DELTA H (Post - Pre): {delta_h:+.4f}")
            
            if delta_h > 0.05:
                print("    -> H INCREASED at grokking. Brain shifted toward PERSISTENT/FRACTAL dynamics.")
                print("       Consistent with Penrose prediction: microtubule restructuring -> 1/f criticality.")
            elif delta_h < -0.05:
                print("    -> H DECREASED at grokking. Brain became MORE RANDOM.")
                print("       INCONSISTENT with Penrose prediction.")
            else:
                print("    -> H did not shift significantly.")
        
        print(f"\n  SPECTRAL SLOPE (beta):")
        
        pre_b = [t['beta'] for t in pre_grok if not np.isnan(t['beta'])]
        post_b = [t['beta'] for t in post_grok if not np.isnan(t['beta'])]
        grok_b = [t['beta'] for t in grok_window if not np.isnan(t['beta'])]
        
        if pre_b:
            print(f"    Pre-Grokking:       beta = {np.mean(pre_b):.4f} +/- {np.std(pre_b):.4f}")
        if grok_b:
            print(f"    At Grokking (+/-3): beta = {np.mean(grok_b):.4f} +/- {np.std(grok_b):.4f}")
        if post_b:
            print(f"    Post-Grokking:      beta = {np.mean(post_b):.4f} +/- {np.std(post_b):.4f}")
        
        if pre_b and post_b:
            delta_b = np.mean(post_b) - np.mean(pre_b)
            print(f"    DELTA beta (Post - Pre): {delta_b:+.4f}")
            
            if delta_b < -0.2:  # More negative = more pink
                print("    -> beta shifted toward 1/f PINK NOISE at grokking.")
                print("       This is the Penrose signature: microtubule-driven fractal dynamics.")
            elif delta_b > 0.2:
                print("    -> beta shifted toward WHITE NOISE at grokking.")
                print("       INCONSISTENT with Penrose prediction.")
        
        # Trial-by-trial printout around grokking
        print(f"\n  TRIAL-BY-TRIAL around grokking (trial {grok_trial}):")
        print(f"    {'Trial':>6} {'Correct':>8} {'H':>8} {'beta':>8} {'Run':>10}")
        print(f"    {'-'*44}")
        
        window_start = max(0, grok_trial - 5)
        window_end = min(len(all_trial_data), grok_trial + 8)
        
        for t in all_trial_data[window_start:window_end]:
            marker = " <<<" if t['trial_num'] == grok_trial else ""
            correct_str = str(t['correct'])[:5]
            h_str = f"{t['hurst']:.4f}" if not np.isnan(t['hurst']) else "  NaN"
            b_str = f"{t['beta']:.4f}" if not np.isnan(t['beta']) else "   NaN"
            print(f"    {t['trial_num']:>6} {correct_str:>8} {h_str:>8} {b_str:>8} {t['run']:>10}{marker}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_hurst_grokking_test()
