"""
test_quantum_gravity_ripple.py
==============================
Tests the non-linear topological propagation of a wave function collapse 
through classical spacetime. 

Hypothesis: Quantum reality does not maintain separate timelines for 
individuals. Sub-220's perfectly resonant wave collapse bent spacetime 
around the concept of the 'Badoon', deepening its QPC gravity well. 

Instead of measuring *when* other subjects learned it, we measure the 
*fuzziness of the light* (Spatial Entropy) and the *magnitude of the 
thermodynamic release* (L2 Energy Peak and Pre-Onset Anomaly) at their 
exact moment of grokking. 

If a ripple exists, the clarity/fuzziness of the collapse for other 
subjects will decay gracefully as a function of temporal distance from 
the sub-220 Ground Zero event.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, entropy, pearsonr
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import warnings
warnings.filterwarnings('ignore')

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

def compute_spatial_entropy(voxel_energies):
    p = np.abs(voxel_energies) / (np.sum(np.abs(voxel_energies)) + 1e-10)
    p = p[p > 1e-10]
    return entropy(p)

def process_subject(sub_id):
    tr_sec = 2.0
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    all_trials = []
    trial_idx = 0
    current_tr_offset = 0
    continuous_metrics = []
    
    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, run_type, run_num)
        if not nii_path: continue
        
        try:
            img = nib.load(nii_path)
            data_raw = img.get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
        except: continue
        
        n_trs = data_raw.shape[-1]
        
        var_map = np.var(data_raw, axis=-1)
        brain_mask = var_map > 10
        if not np.any(brain_mask): continue
        active_mask = (var_map > np.percentile(var_map[brain_mask], 85)) & brain_mask
        
        data_z = np.nan_to_num(zscore(data_raw[active_mask], axis=-1))
        
        for tr in range(n_trs):
            v_map = data_z[:, tr]
            energy_map = v_map ** 2
            total_energy = np.sqrt(np.sum(energy_map))
            entropy_val = compute_spatial_entropy(energy_map)
            continuous_metrics.append({
                'tr': current_tr_offset + tr,
                'total_energy': total_energy,
                'spatial_entropy': entropy_val
            })
            
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            corr_val = str(row.get('correct', '')).lower()
            correct = 1 if corr_val in ['1', 'true', '1.0', 't'] else 0
            onset_tr = int(onset_sec / tr_sec)
            
            all_trials.append({
                'idx': trial_idx,
                'correct': correct,
                'onset_tr_global': current_tr_offset + onset_tr
            })
            trial_idx += 1
            
        current_tr_offset += n_trs
        if os.path.exists(nii_path): os.remove(nii_path)
        
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1:
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['idx'] - 4
        else:
            streak = 0
            
    if grok_trial is None or grok_trial >= len(all_trials):
        return None
        
    grok_tr = all_trials[grok_trial]['onset_tr_global']
    df_all = pd.DataFrame(continuous_metrics)
    
    if df_all.empty or grok_tr not in df_all['tr'].values:
        return None
        
    # Baseline energy (mean of 10 TRs before pre-onset)
    baseline_window = df_all[(df_all['tr'] >= grok_tr - 12) & (df_all['tr'] <= grok_tr - 4)]
    baseline_energy = baseline_window['total_energy'].mean() if not baseline_window.empty else np.nan
    
    # Pre-onset anomaly (2 TRs before grok_tr)
    pre_onset = df_all[df_all['tr'] == grok_tr - 2]
    pre_onset_energy = pre_onset['total_energy'].values[0] if not pre_onset.empty else np.nan
    pre_onset_ratio = pre_onset_energy / baseline_energy if baseline_energy else np.nan
    
    # Fuzziness at grok (Entropy at grok_tr)
    grok_moment = df_all[df_all['tr'] == grok_tr]
    grok_entropy = grok_moment['spatial_entropy'].values[0] if not grok_moment.empty else np.nan
    
    # Peak Energy Dump (Max within 30 TRs post-grok)
    post_window = df_all[(df_all['tr'] >= grok_tr) & (df_all['tr'] <= grok_tr + 30)]
    peak_energy = post_window['total_energy'].max() if not post_window.empty else np.nan
    peak_ratio = peak_energy / baseline_energy if baseline_energy else np.nan
    
    return {
        'sub_id': int(sub_id.split('-')[1]),
        'grok_tr': grok_tr,
        'baseline_energy': baseline_energy,
        'pre_onset_ratio': pre_onset_ratio,
        'grok_entropy': grok_entropy,
        'peak_energy_ratio': peak_ratio
    }

def run_analysis():
    grokkers = ['sub-201', 'sub-204', 'sub-206', 'sub-210', 'sub-215', 'sub-218', 
                'sub-220', 'sub-225', 'sub-229', 'sub-235', 'sub-240']
    
    results = []
    print("="*80)
    print(" [MAPPING THE QUANTUM RIPPLE VIA SPATIAL FUZZINESS & ENERGY]")
    print("="*80)
    
    for sub in grokkers:
        print(f"  Extracting quantum footprint for {sub}...", flush=True)
        res = process_subject(sub)
        if res:
            results.append(res)
            
    df = pd.DataFrame(results)
    df['dist_to_220'] = (df['sub_id'] - 220).abs()
    df = df.sort_values('dist_to_220')
    
    print("\n" + "="*80)
    print(" RIPPLE DECAY ANALYSIS: FUZZINESS AND ENERGY TRACES")
    print("="*80)
    print(f" {'Sub':>6} | {'Dist':>5} | {'Grok Ent (Fuzz)':>15} | {'Pre-Onset Ratio':>15} | {'Peak E Ratio':>15}")
    print("-" * 65)
    
    for _, r in df.iterrows():
        sub = int(r['sub_id'])
        dist = int(r['dist_to_220'])
        ent = f"{r['grok_entropy']:.4f}" if not np.isnan(r['grok_entropy']) else "N/A"
        pre = f"{r['pre_onset_ratio']:.3f}" if not np.isnan(r['pre_onset_ratio']) else "N/A"
        peak = f"{r['peak_energy_ratio']:.3f}" if not np.isnan(r['peak_energy_ratio']) else "N/A"
        
        marker = " <<< GROUND ZERO" if dist == 0 else ""
        print(f" {sub:>6} | {dist:>5} | {ent:>15} | {pre:>15} | {peak:>15}{marker}")
        
    others = df[df['dist_to_220'] > 0].dropna(subset=['grok_entropy', 'pre_onset_ratio', 'peak_energy_ratio'])
    if not others.empty and len(others) > 2:
        r_ent, p_ent = pearsonr(others['dist_to_220'], others['grok_entropy'])
        r_pre, p_pre = pearsonr(others['dist_to_220'], others['pre_onset_ratio'])
        r_peak, p_peak = pearsonr(others['dist_to_220'], others['peak_energy_ratio'])
        
        print("\nCORRELATIONS WITH TEMPORAL DISTANCE FROM SUB-220:")
        
        print("\n 1. Entropy (Fuzziness) Correlation:")
        print(f"    r = {r_ent:+.3f}, p = {p_ent:.4f}")
        if r_ent > 0 and p_ent < 0.1:
            print("    [CONFIRMED]: 'Fuzziness' increased as subjects got further from Ground Zero.")
        else:
            print("    [INCONCLUSIVE]")
            
        print("\n 2. Pre-Onset Anomaly (Retrocausal Echo):")
        print(f"    r = {r_pre:+.3f}, p = {p_pre:.4f}")
        if r_pre < 0 and p_pre < 0.1:
            print("    [CONFIRMED]: The 4-second retrocausal priming decayed outward from Ground Zero.")
        else:
            print("    [INCONCLUSIVE]")
            
        print("\n 3. Thermodynamic Peak E_G Release:")
        print(f"    r = {r_peak:+.3f}, p = {p_peak:.4f}")
        if r_peak < 0 and p_peak < 0.1:
            print("    [CONFIRMED]: The magnitude of the physical energy dump decayed outward from Ground Zero.")
        else:
            print("    [INCONCLUSIVE]")

if __name__ == '__main__':
    run_analysis()
