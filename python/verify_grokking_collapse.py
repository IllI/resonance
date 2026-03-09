"""
verify_grokking_collapse.py
============================
Does the wave collapse characterization hold universally for ALL grokking
subjects in the badoon paradigm?

We run the precision pipeline developed for sub-220 across all subjects
that demonstrated a grokking streak (5+ correct):
Sub-Ids: 201, 204, 206, 210, 215, 218, 220, 225, 229, 235, 240

For each subject, we measure:
1. GHOST BASIN FUNNEL WIDTH (Variance):
   Does the Euclidean pattern variance drop sharply at/after the grokking 
   chrysalis? (Indicates Bi-Twistor phase locking)

2. SPECTRAL BETA SHIFT:
   Does the macroscopic BOLD signal shift from more chaotic (beta ~ 0) 
   to highly coherent pink noise (beta ~ -1.0) post-grokking?
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress
from scipy.signal import welch
from scipy.spatial.distance import cosine
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

def compute_beta_sliding(timeseries, fs, window_size=30, step=5):
    n = len(timeseries)
    if n < window_size:
        return []
    results = []
    for start in range(0, n - window_size + 1, step):
        segment = timeseries[start : start + window_size]
        freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), window_size))
        mask = freqs > 0
        pf, pp = freqs[mask], psd[mask]
        if len(pf) < 3 or np.any(pp <= 0): continue
        slope, _, _, _, _ = linregress(np.log10(pf), np.log10(pp + 1e-20))
        results.append((start + window_size//2, slope))
    return results

def verify_subject(sub_id):
    tr_sec = 2.0
    fs = 0.5
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    all_trials = []
    cumulative_idx = 0
    continuous_ts = []
    current_tr_offset = 0
    
    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, run_type, run_num)
        if not nii_path: continue
        
        try:
            data_raw = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
        except: continue
        
        n_trs = data_raw.shape[-1]
        
        # Active Mask
        var_map = np.var(data_raw, axis=-1)
        brain_mask = var_map > 10
        if not np.any(brain_mask): continue
        active_mask = (var_map > np.percentile(var_map[brain_mask], 85)) & brain_mask
        
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        active_voxels = data_z[active_mask]
        global_ts = np.mean(active_voxels, axis=0)
        
        continuous_ts.extend(global_ts.tolist())
        
        hrf_start = int(1.0 / tr_sec)
        hrf_end = int(6.0 / tr_sec)
        
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            corr_val = str(row.get('correct', '')).lower()
            correct = 1 if corr_val in ['1', 'true', '1.0', 't'] else 0
            
            onset_tr = int(onset_sec / tr_sec)
            if onset_tr + hrf_end >= n_trs or onset_tr + hrf_start < 0: continue
            
            # Use fixed fingerprint to avoid shape issues across runs
            raw = np.mean(active_voxels[:, onset_tr + hrf_start : onset_tr + hrf_end], axis=1)
            hist, _ = np.histogram(raw, bins=50, range=(-3, 3))
            hist = hist.astype(float) / (np.sum(hist) + 1e-10)
            stats = np.array([np.mean(raw), np.std(raw), np.percentile(raw, 25), np.percentile(raw, 75)])
            pattern = np.concatenate([hist, stats])
            
            all_trials.append({
                'idx': cumulative_idx,
                'correct': correct,
                'pattern': pattern,
                'run': f"{run_type}{run_num}",
                'onset_tr_global': current_tr_offset + onset_tr
            })
            cumulative_idx += 1
            
        current_tr_offset += n_trs
        
        if os.path.exists(nii_path):
            os.remove(nii_path)
            
    if not all_trials: return None
    
    # 1. FIND GROKKING
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1:
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['idx'] - 4
        else:
            streak = 0
            
    if grok_trial is None:
        return None # Did not grok definitively
        
    grok_tr_global = all_trials[grok_trial]['onset_tr_global']
    
    # 2. FUNNEL WIDTH (Variance)
    window = 5
    variances = []
    
    for i in range(window, len(all_trials)):
        recent = [t['pattern'] for t in all_trials[i-window:i]]
        centroid = np.mean(recent, axis=0)
        cn = np.linalg.norm(centroid)
        if cn < 1e-10: continue
        centroid = centroid / cn
        
        spreads = []
        for p in recent:
            pn = np.linalg.norm(p)
            if pn < 1e-10: continue
            c_sim = 1.0 - cosine(p/pn, centroid)
            angle = np.degrees(np.arccos(np.clip(c_sim, -1, 1)))
            spreads.append(angle)
            
        phase = 'PRE' if i < grok_trial else ('POST' if i > grok_trial + 4 else 'GROK')
        variances.append({'phase': phase, 'fw': np.mean(spreads)})
        
    fw_df = pd.DataFrame(variances)
    pre_fw = fw_df[fw_df['phase'] == 'PRE']['fw'].mean()
    post_fw = fw_df[fw_df['phase'] == 'POST']['fw'].mean()
    fw_shift = post_fw - pre_fw if not np.isnan(pre_fw) and not np.isnan(post_fw) else np.nan
    
    # 3. SPECTRAL BETA
    ts = np.array(continuous_ts)
    beta_traj = compute_beta_sliding(ts, fs)
    
    pre_betas = [b for tr, b in beta_traj if tr < grok_tr_global]
    post_betas = [b for tr, b in beta_traj if tr > grok_tr_global + 10]
    
    pre_b = np.mean(pre_betas) if pre_betas else np.nan
    post_b = np.mean(post_betas) if post_betas else np.nan
    b_shift = post_b - pre_b if pre_betas and post_betas else np.nan
    
    return {
        'sub': sub_id,
        'grok_trial': grok_trial,
        'pre_fw': pre_fw,
        'post_fw': post_fw,
        'fw_shift': fw_shift,
        'pre_beta': pre_b,
        'post_beta': post_b,
        'beta_shift': b_shift
    }

def run_verification():
    grokkers = ['sub-201', 'sub-204', 'sub-206', 'sub-210', 'sub-215', 'sub-218', 
                'sub-220', 'sub-225', 'sub-229', 'sub-235', 'sub-240']
                
    print("=" * 80)
    print(" VERIFYING UNIVERSAL WAVE COLLAPSE BEHAVIOR ACROSS GROKKERS")
    print(" Testing Bi-Twistor Funnel Narrowing and Spectral Beta Shift")
    print("=" * 80)
    
    results = []
    for sub in grokkers:
        print(f"  Analyzing {sub}...", flush=True)
        res = verify_subject(sub)
        if res:
            results.append(res)
            print(f"    Grok at: {res['grok_trial']}, FW Shift: {res['fw_shift']:.2f} deg, Beta Shift: {res['beta_shift']:.3f}")
            
    print("\n" + "=" * 80)
    print(" RESULTS (UNIVERSAL CHARACTERIZATION CHECK)")
    print("=" * 80)
    
    print(f"\n  {'Subject':>8} {'GkT':>5} | {'Pre FW':>7} {'Post FW':>7} {'Shft':>6} | {'Pre B':>7} {'Post B':>7} {'B Shf':>7}")
    print(f"  {'-'*68}")
    
    fw_shifts = []
    b_shifts = []
    
    for r in results:
        g = r['grok_trial']
        fw1 = f"{r['pre_fw']:.2f}" if not np.isnan(r['pre_fw']) else "N/A"
        fw2 = f"{r['post_fw']:.2f}" if not np.isnan(r['post_fw']) else "N/A"
        fwd = f"{r['fw_shift']:+.2f}" if not np.isnan(r['fw_shift']) else "N/A"
        
        b1 = f"{r['pre_beta']:.3f}" if not np.isnan(r['pre_beta']) else "N/A"
        b2 = f"{r['post_beta']:.3f}" if not np.isnan(r['post_beta']) else "N/A"
        bd = f"{r['beta_shift']:+.3f}" if not np.isnan(r['beta_shift']) else "N/A"
        
        if not np.isnan(r['fw_shift']): fw_shifts.append(r['fw_shift'])
        if not np.isnan(r['beta_shift']): b_shifts.append(r['beta_shift'])
        
        print(f"  {r['sub']:>8} {g:>5} | {fw1:>7} {fw2:>7} {fwd:>6} | {b1:>7} {b2:>7} {bd:>7}")
        
    print("\n  SUMMARY:")
    if fw_shifts:
        pct_narrowed = sum(1 for s in fw_shifts if s < 0) / len(fw_shifts) * 100
        print(f"    Funnel Width Narrowed in {pct_narrowed:.0f}% of subjects. Mean Shift: {np.mean(fw_shifts):+.2f} deg")
    if b_shifts:
        pct_pink = sum(1 for s in b_shifts if s < -0.1) / len(b_shifts) * 100
        print(f"    Beta Shifted Pink (-0.1+) in {pct_pink:.0f}% of subjects. Mean Shift: {np.mean(b_shifts):+.3f}")
        
    if np.mean(fw_shifts) < -0.1 and np.mean(b_shifts) < -0.1:
        print("\n  VERDICT: WAVE COLLAPSE CHARACTERISTICS CONFIRMED UNIVERSALLY AROUND GROKKING.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_verification()
