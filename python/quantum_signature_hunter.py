"""
quantum_signature_hunter.py
============================
TRIAL-BY-TRIAL hunt for quantum signature candidates in the 7T ds006661 dataset.

For EACH individual trial, we check:
  1. INITIAL DIP: Did the BOLD signal go negative in the first 1.5s before peaking?
  2. PINK NOISE SHIFT: Did the spectral slope hit 1/f criticality (-0.7 to -1.3)?
  3. ENERGY ANOMALY: Was the HF power in the top 5% of all trials?

If 2+ of these co-occur on a single trial, we flag it as a "Quantum Candidate"
and then report EXACTLY what was happening: category, timing, context, reaction time.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress
from scipy.fft import fft, fftfreq
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def stream_data(sub_id, run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
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

def spectral_slope(signal, tr_sec):
    n = len(signal)
    if n < 6:
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
    return slope

def hf_power_fraction(signal, tr_sec):
    n = len(signal)
    if n < 4:
        return 0.0
    fft_vals = fft(signal)
    freqs = fftfreq(n, d=tr_sec)
    power = np.abs(fft_vals) ** 2
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    total = np.sum(pos_power)
    if total < 1e-10:
        return 0.0
    hf_mask = pos_freqs > 2.0  # >2 Hz is genuinely "high frequency" at 125ms TR
    return np.sum(pos_power[hf_mask]) / total

def hunt_quantum_signatures():
    tr_sec = 0.125
    subjects = ["sub-001", "sub-002", "sub-003", "sub-005", "sub-006"]
    runs = ["01", "02", "03", "04"]
    
    print("=" * 80)
    print(" QUANTUM SIGNATURE HUNTER: Trial-by-Trial 7T Analysis")
    print(" Scanning every single stimulus event for quantum fingerprints")
    print("=" * 80)
    
    all_trials = []
    
    for sub in subjects:
        for run in runs:
            print(f"\n  Loading {sub} run-{run}...", flush=True)
            nii_path, tsv_path = stream_data(sub, run)
            if not nii_path:
                print(f"    Skipped (not available).", flush=True)
                continue
                
            try:
                data_raw = nib.load(nii_path).get_fdata()
                events = pd.read_csv(tsv_path, sep='\t')
            except Exception as e:
                print(f"    Failed to load: {e}", flush=True)
                continue
                
            n_trs = data_raw.shape[-1]
            
            # Top 20% variance voxels
            var_map = np.var(data_raw, axis=-1)
            threshold = np.percentile(var_map, 80)
            brain_mask = var_map > threshold
            active_voxels = data_raw[brain_mask]
            data_z = np.nan_to_num(zscore(active_voxels, axis=1))
            
            # Global time series (mean across all active voxels)
            global_ts = np.mean(data_z, axis=0)
            
            # Get image events only
            image_events = events[events['trial_type'] == 'image'].copy()
            
            # Find the nearest fixation_change response RT for temporal context
            response_events = events[events['trial_type'] == 'response'].copy()
            
            for _, row in image_events.iterrows():
                onset_sec = row['onset']
                cat = row['image_category']
                onset_tr = int(onset_sec / tr_sec)
                
                # We need a wide window: -0.5s to +4s around stimulus
                pre_trs = int(0.5 / tr_sec)   # 4 TRs before
                post_trs = int(4.0 / tr_sec)   # 32 TRs after
                dip_end_trs = int(1.5 / tr_sec) # 12 TRs for dip window
                peak_start_trs = int(2.0 / tr_sec) # 16 TRs
                peak_end_trs = int(4.0 / tr_sec)   # 32 TRs
                
                if onset_tr - pre_trs < 0 or onset_tr + post_trs >= n_trs:
                    continue
                
                # Extract the trial window
                window = global_ts[onset_tr - pre_trs : onset_tr + post_trs]
                baseline = np.mean(global_ts[onset_tr - pre_trs : onset_tr])
                
                # === SIGNATURE 1: INITIAL DIP ===
                dip_segment = global_ts[onset_tr : onset_tr + dip_end_trs]
                dip_magnitude = np.min(dip_segment) - baseline
                
                # === SIGNATURE 2: SPECTRAL SLOPE (Pink Noise) ===
                spectral_window = global_ts[onset_tr : onset_tr + post_trs]
                beta = spectral_slope(spectral_window, tr_sec)
                
                # === SIGNATURE 3: HIGH FREQUENCY POWER ===
                hf = hf_power_fraction(spectral_window, tr_sec)
                
                # === SIGNATURE 4: PEAK AMPLITUDE ===
                peak_segment = global_ts[onset_tr + peak_start_trs : onset_tr + peak_end_trs]
                peak_amplitude = np.max(peak_segment) - baseline
                
                # === CONTEXT: What was the nearest behavioral response? ===
                nearest_rt = np.nan
                time_to_nearest_response = np.nan
                if len(response_events) > 0:
                    diffs = response_events['onset'].values - onset_sec
                    # Find closest response AFTER the image
                    future_mask = diffs > 0
                    if np.any(future_mask):
                        closest_idx = np.argmin(diffs[future_mask])
                        time_to_nearest_response = diffs[future_mask][closest_idx]
                        rt_vals = response_events['reaction_time'].values
                        nearest_rt = rt_vals[np.where(future_mask)[0][closest_idx]]
                
                # Trial position in the run (early=novel, late=habituated)
                trial_position = len([t for t in all_trials if t['Subject'] == sub and t['Run'] == run])
                
                trial = {
                    'Subject': sub,
                    'Run': run,
                    'Onset_sec': onset_sec,
                    'Category': cat,
                    'Trial_Position': trial_position,
                    'Dip_Magnitude': dip_magnitude,
                    'Peak_Amplitude': peak_amplitude,
                    'Spectral_Beta': beta,
                    'HF_Power': hf,
                    'Time_To_Next_Response': time_to_nearest_response,
                    'Nearest_RT': nearest_rt if not pd.isna(nearest_rt) else np.nan,
                }
                all_trials.append(trial)
            
            # Cleanup
            if os.path.exists(nii_path):
                os.remove(nii_path)
    
    df = pd.DataFrame(all_trials)
    print(f"\n\nTotal trials scanned: {len(df)}")
    
    # === IDENTIFY QUANTUM CANDIDATES ===
    # Criteria:
    # 1. Initial Dip present (dip < -0.05 z-score)
    # 2. Spectral slope in pink noise range (-1.3 to -0.7)
    # 3. HF power in top 10%
    
    hf_threshold = df['HF_Power'].quantile(0.90)
    
    df['Has_Dip'] = df['Dip_Magnitude'] < -0.05
    df['Is_Pink'] = (df['Spectral_Beta'] >= -1.3) & (df['Spectral_Beta'] <= -0.7)
    df['Is_HF_Spike'] = df['HF_Power'] >= hf_threshold
    
    df['Quantum_Score'] = df['Has_Dip'].astype(int) + df['Is_Pink'].astype(int) + df['Is_HF_Spike'].astype(int)
    
    candidates = df[df['Quantum_Score'] >= 2].copy()
    
    print(f"\nTrials with Initial Dip: {df['Has_Dip'].sum()} / {len(df)} ({100*df['Has_Dip'].mean():.1f}%)")
    print(f"Trials with Pink Noise:  {df['Is_Pink'].sum()} / {len(df)} ({100*df['Is_Pink'].mean():.1f}%)")
    print(f"Trials with HF Spike:    {df['Is_HF_Spike'].sum()} / {len(df)} ({100*df['Is_HF_Spike'].mean():.1f}%)")
    print(f"\nQUANTUM CANDIDATES (2+ signatures co-occurring): {len(candidates)} / {len(df)}")
    
    if len(candidates) == 0:
        print("\n  NO quantum candidates found. The experiment produced no detectable quantum-like events.")
        return
    
    print("\n" + "=" * 80)
    print(" QUANTUM CANDIDATE DETAILS")
    print("=" * 80)
    
    for _, c in candidates.iterrows():
        sigs = []
        if c['Has_Dip']: sigs.append(f"Initial Dip ({c['Dip_Magnitude']:.3f}z)")
        if c['Is_Pink']: sigs.append(f"Pink Noise (beta={c['Spectral_Beta']:.2f})")
        if c['Is_HF_Spike']: sigs.append(f"HF Spike ({c['HF_Power']:.4f})")
        
        print(f"\n  [{c['Subject']}] Run {c['Run']}, Trial #{c['Trial_Position']}, t={c['Onset_sec']:.1f}s")
        print(f"    Category: {c['Category']}")
        print(f"    Signatures: {' + '.join(sigs)}")
        print(f"    Peak BOLD Amplitude: {c['Peak_Amplitude']:.3f}z")
        if not np.isnan(c['Time_To_Next_Response']):
            print(f"    Time to next button press: {c['Time_To_Next_Response']:.2f}s")
        if not np.isnan(c['Nearest_RT']):
            print(f"    Nearest Reaction Time: {c['Nearest_RT']*1000:.0f}ms")
    
    # === CONTEXTUAL ANALYSIS ===
    print("\n" + "=" * 80)
    print(" CONTEXTUAL ANALYSIS OF QUANTUM CANDIDATES")
    print("=" * 80)
    
    # 1. Category distribution
    print("\n  [A] Category Distribution:")
    cat_counts = candidates['Category'].value_counts()
    total_per_cat = df['Category'].value_counts()
    for cat in cat_counts.index:
        rate = cat_counts[cat] / total_per_cat[cat] * 100
        print(f"    {cat:>15}: {cat_counts[cat]} candidates / {total_per_cat[cat]} total ({rate:.1f}%)")
    
    # 2. Trial position (early = novel, late = habituated)
    print("\n  [B] Trial Position (Early=Novel, Late=Habituated):")
    avg_pos_candidates = candidates['Trial_Position'].mean()
    avg_pos_all = df['Trial_Position'].mean()
    print(f"    Avg trial position for candidates: {avg_pos_candidates:.1f}")
    print(f"    Avg trial position for all trials: {avg_pos_all:.1f}")
    if avg_pos_candidates < avg_pos_all * 0.7:
        print("    -> Candidates cluster EARLY in runs (novel stimuli = more processing demand)")
    elif avg_pos_candidates > avg_pos_all * 1.3:
        print("    -> Candidates cluster LATE in runs (fatigue? or deep encoding?)")
    else:
        print("    -> Candidates spread evenly across the run")
    
    # 3. Reaction time context
    valid_rt = candidates[~candidates['Nearest_RT'].isna()]
    if len(valid_rt) > 0:
        print(f"\n  [C] Behavioral Context (Nearest Button Press):")
        print(f"    Avg RT for quantum candidates: {valid_rt['Nearest_RT'].mean()*1000:.0f}ms")
        all_valid_rt = df[~df['Nearest_RT'].isna()]
        print(f"    Avg RT for all trials:         {all_valid_rt['Nearest_RT'].mean()*1000:.0f}ms")
        if valid_rt['Nearest_RT'].mean() > all_valid_rt['Nearest_RT'].mean() * 1.1:
            print("    -> Subjects were SLOWER to respond near quantum events (distracted by internal processing?)")
        elif valid_rt['Nearest_RT'].mean() < all_valid_rt['Nearest_RT'].mean() * 0.9:
            print("    -> Subjects were FASTER near quantum events (heightened arousal?)")
    
    # 4. Dip-to-Peak ratio (the "wave collapse" shape)
    print(f"\n  [D] Initial Dip to Peak Ratio (Wave Collapse Shape):")
    candidates_with_dip = candidates[candidates['Has_Dip']]
    if len(candidates_with_dip) > 0:
        ratios = candidates_with_dip['Peak_Amplitude'] / (np.abs(candidates_with_dip['Dip_Magnitude']) + 1e-10)
        print(f"    Avg Dip-to-Peak Ratio: {ratios.mean():.2f}")
        print(f"    (A ratio > 2.0 suggests the dip 'triggered' a massive hemodynamic overcompensation)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    hunt_quantum_signatures()
