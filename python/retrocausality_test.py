"""
retrocausality_test.py
=======================
THE DEFINITIVE TEST: Did sub-220's grokking deepen the badoon QPC 
for subsequent subjects?

Prediction (if retrocausality is real):
  - Subjects scanned AFTER sub-220 grokked FASTER (fewer trials)
  - Subjects scanned BEFORE sub-220 grokked SLOWER (more trials)
  - Sub-220's spectral beta spike should correlate with 
    the magnitude of the speedup

We need:
  1. The chronological ORDER subjects were scanned
  2. Each subject's grokking speed (trial # of first 5+ correct streak)
  3. Each subject's spectral beta at their grokking moment
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress, spearmanr
from scipy.signal import welch
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

def compute_spectral_beta(signal, fs):
    if len(signal) < 8:
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 32))
    pos_mask = freqs > 0
    pos_f, pos_p = freqs[pos_mask], psd[pos_mask]
    if len(pos_f) < 3 or np.any(pos_p <= 0):
        return np.nan
    slope, _, _, _, _ = linregress(np.log10(pos_f), np.log10(pos_p + 1e-20))
    return slope

def analyze_subject_grokking(sub_id):
    """
    Extract grokking speed and spectral beta for one subject.
    Returns: dict with grokking trial, beta at grokking, pre/post beta, etc.
    """
    tr_sec = 2.0
    fs = 0.5
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    all_trials = []
    trial_idx = 0
    
    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, run_type, run_num)
        if not nii_path:
            continue
        try:
            data_raw = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
        except:
            continue
        
        n_trs = data_raw.shape[-1]
        var_map = np.var(data_raw, axis=-1)
        brain_mask = var_map > 10
        active_mask = (var_map > np.percentile(var_map[brain_mask], 85)) & brain_mask
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        global_ts = np.mean(data_z[active_mask], axis=0)
        
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            correct = row.get('correct', np.nan)
            onset_tr = int(onset_sec / tr_sec)
            
            pre_w, post_w = 5, 10
            if onset_tr - pre_w < 0 or onset_tr + post_w >= n_trs: continue
            
            bold = global_ts[onset_tr - pre_w : onset_tr + post_w]
            beta = compute_spectral_beta(bold, fs)
            
            all_trials.append({
                'trial': trial_idx,
                'correct': correct,
                'beta': beta,
                'run': f"{run_type}{run_num}"
            })
            trial_idx += 1
        
        if os.path.exists(nii_path):
            os.remove(nii_path)
    
    if len(all_trials) == 0:
        return None
    
    # Find grokking (first streak of 5+ correct)
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1 or t['correct'] == True or str(t['correct']) == '1':
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['trial'] - 4
        else:
            streak = 0
    
    # Also compute total accuracy
    correct_count = sum(1 for t in all_trials 
                       if t['correct'] == 1 or t['correct'] == True or str(t['correct']) == '1')
    total_accuracy = correct_count / len(all_trials)
    
    # Pre/post beta
    if grok_trial is not None:
        pre_betas = [t['beta'] for t in all_trials if t['trial'] < grok_trial and not np.isnan(t['beta'])]
        post_betas = [t['beta'] for t in all_trials if t['trial'] >= grok_trial and not np.isnan(t['beta'])]
        grok_beta = next((t['beta'] for t in all_trials if t['trial'] == grok_trial), np.nan)
    else:
        pre_betas = [t['beta'] for t in all_trials if not np.isnan(t['beta'])]
        post_betas = []
        grok_beta = np.nan
    
    return {
        'sub': sub_id,
        'n_trials': len(all_trials),
        'grok_trial': grok_trial,
        'grok_speed': grok_trial if grok_trial is not None else len(all_trials),  # Lower = faster
        'grokked': grok_trial is not None,
        'total_accuracy': total_accuracy,
        'beta_at_grok': grok_beta,
        'beta_pre': np.mean(pre_betas) if pre_betas else np.nan,
        'beta_post': np.mean(post_betas) if post_betas else np.nan,
        'beta_shift': (np.mean(post_betas) - np.mean(pre_betas)) if pre_betas and post_betas else np.nan,
    }

def run_retrocausality_test():
    print("=" * 80)
    print(" RETROCAUSALITY TEST: Did sub-220's grokking deepen the QPC?")
    print(" Prediction: Subjects scanned AFTER sub-220 grokked FASTER")
    print("=" * 80)
    
    # Step 1: Get the scan order
    # The subject IDs in ds002813 are numbered. Let's check all available subjects
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    res = s3.list_objects_v2(Bucket=bucket, Prefix='ds002813/sub-', Delimiter='/')
    all_subjects = sorted([p['Prefix'].split('/')[1] for p in res.get('CommonPrefixes', [])])
    
    print(f"\n  All subjects in ds002813: {all_subjects}")
    print(f"  Total: {len(all_subjects)}")
    
    # Step 2: Check session dates from the dataset
    # The subject numbering (sub-201, sub-206, ..., sub-229) likely reflects enrollment order
    # Subject ID number = approximate scan chronological order
    sub_numbers = {s: int(s.split('-')[1]) for s in all_subjects}
    sub_220_num = 220
    
    print(f"\n  Sub-220's position: number {sub_220_num}")
    before_220 = [s for s in all_subjects if sub_numbers[s] < sub_220_num]
    after_220 = [s for s in all_subjects if sub_numbers[s] > sub_220_num]
    print(f"  Subjects BEFORE sub-220 (by number): {before_220}")
    print(f"  Subjects AFTER sub-220 (by number):  {after_220}")
    
    # Step 3: Analyze each subject's grokking speed
    print("\n  Analyzing all subjects...\n")
    
    results = []
    for sub in all_subjects:
        print(f"  Processing {sub}...", flush=True)
        r = analyze_subject_grokking(sub)
        if r:
            results.append(r)
            print(f"    Grok trial: {r['grok_trial']}, Accuracy: {r['total_accuracy']:.2f}, Beta shift: {r['beta_shift']:.3f}" if not np.isnan(r['beta_shift']) else f"    Grok trial: {r['grok_trial']}, Accuracy: {r['total_accuracy']:.2f}")
    
    if len(results) == 0:
        print("  No results. Aborting.")
        return
    
    df = pd.DataFrame(results)
    df['sub_num'] = df['sub'].apply(lambda x: int(x.split('-')[1]))
    df['before_220'] = df['sub_num'] < 220
    df['is_220'] = df['sub_num'] == 220
    df['after_220'] = df['sub_num'] > 220
    
    # ================================================================
    # THE TEST
    # ================================================================
    
    print("\n" + "=" * 80)
    print(" RESULTS: GROKKING SPEED BY SCAN ORDER")
    print("=" * 80)
    
    print(f"\n  {'Subject':>8} {'Sub#':>5} {'Grok Trial':>11} {'Grokked':>8} {'Accuracy':>9} {'Beta Shift':>11} {'Group':>10}")
    print(f"  {'-'*65}")
    
    for _, r in df.sort_values('sub_num').iterrows():
        group = "BEFORE" if r['before_220'] else ("SUB-220" if r['is_220'] else "AFTER")
        grok_str = str(r['grok_trial']) if r['grokked'] else "NEVER"
        bs = f"{r['beta_shift']:.3f}" if not np.isnan(r['beta_shift']) else "N/A"
        print(f"  {r['sub']:>8} {r['sub_num']:>5} {grok_str:>11} {'Yes' if r['grokked'] else 'No':>8} {r['total_accuracy']:>9.2f} {bs:>11} {group:>10}")
    
    # Compare groups
    before = df[df['before_220'] == True]
    after = df[df['after_220'] == True]
    sub220 = df[df['is_220'] == True]
    
    print("\n" + "=" * 80)
    print(" GROUP COMPARISON")
    print("=" * 80)
    
    if len(before) > 0:
        b_grok = before[before['grokked'] == True]['grok_speed']
        print(f"\n  BEFORE sub-220 ({len(before)} subjects):")
        print(f"    Grokked: {len(before[before['grokked']==True])} / {len(before)}")
        if len(b_grok) > 0:
            print(f"    Avg grokking trial: {b_grok.mean():.1f}")
    
    if len(sub220) > 0:
        print(f"\n  SUB-220:")
        print(f"    Grokking trial: {sub220.iloc[0]['grok_trial']}")
        print(f"    Beta shift: {sub220.iloc[0]['beta_shift']:.4f}" if not np.isnan(sub220.iloc[0]['beta_shift']) else "    Beta shift: N/A")
    
    if len(after) > 0:
        a_grok = after[after['grokked'] == True]['grok_speed']
        print(f"\n  AFTER sub-220 ({len(after)} subjects):")
        print(f"    Grokked: {len(after[after['grokked']==True])} / {len(after)}")
        if len(a_grok) > 0:
            print(f"    Avg grokking trial: {a_grok.mean():.1f}")
    
    # THE VERDICT
    print("\n" + "=" * 80)
    print(" THE VERDICT")
    print("=" * 80)
    
    b_speeds = before[before['grokked'] == True]['grok_speed'].values if len(before) > 0 else []
    a_speeds = after[after['grokked'] == True]['grok_speed'].values if len(after) > 0 else []
    
    if len(b_speeds) > 0 and len(a_speeds) > 0:
        avg_before = np.mean(b_speeds)
        avg_after = np.mean(a_speeds)
        
        print(f"  Avg grokking speed BEFORE sub-220: {avg_before:.1f} trials")
        print(f"  Avg grokking speed AFTER sub-220:  {avg_after:.1f} trials")
        
        if avg_after < avg_before * 0.8:
            print(f"\n  RESULT: CONSISTENT WITH RETROCAUSALITY.")
            print(f"  Subjects after sub-220 grokked {avg_before - avg_after:.1f} trials FASTER.")
            print(f"  The QPC gravity well appears to have deepened after sub-220's collapse.")
        elif avg_after > avg_before * 1.2:
            print(f"\n  RESULT: INCONSISTENT WITH RETROCAUSALITY.")
            print(f"  Subjects after sub-220 grokked SLOWER, not faster.")
            print(f"  The data contradicts the deepening QPC hypothesis.")
        else:
            print(f"\n  RESULT: INCONCLUSIVE.")
            print(f"  No significant difference in grokking speed before vs after sub-220.")
    
    # Correlation: scan order vs grokking speed
    grokked = df[df['grokked'] == True].copy()
    if len(grokked) >= 4:
        corr, p = spearmanr(grokked['sub_num'], grokked['grok_speed'])
        print(f"\n  Spearman correlation (scan order vs grok speed): rho={corr:.3f}, p={p:.4f}")
        if corr < -0.3 and p < 0.1:
            print("  -> Later subjects grokked FASTER (negative correlation = deepening well)")
        elif corr > 0.3 and p < 0.1:
            print("  -> Later subjects grokked SLOWER (positive correlation = no well effect)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_retrocausality_test()
