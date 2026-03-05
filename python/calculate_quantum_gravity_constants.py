"""
calculate_quantum_gravity_constants.py
======================================
Tests the Penrose Time Equation (tau ~ h_bar / E_G) by correlating the 
metabolic energy dump (E_G) with the retrocausal temporal displacement (tau) 
at the moment of grokking.

Framework:
- E_G (Gravitational Self-Energy): The metabolic shockwave (L2 Peak Ratio).
- tau (Coherence Time / Retrocausality): The temporal displacement 
  of the pre-onset anomaly (relative TR where signal first spikes above baseline).
- Fuzziness (Entropy): The 'density' of the QPC well.

If the theory holds: E_G * tau should tend towards a constant (scaled h_bar) 
across all grokking moments in the bi-twistor funnel.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, entropy
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

def compute_spatial_entropy(voxel_energies):
    p = np.abs(voxel_energies) / (np.sum(np.abs(voxel_energies)) + 1e-10)
    p = p[p > 1e-10]
    return entropy(p)

def measure_quantum_constants(sub_id):
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
            continuous_metrics.append({
                'tr': current_tr_offset + tr,
                'total_energy': np.sqrt(np.sum(energy_map)),
                'spatial_entropy': compute_spatial_entropy(energy_map)
            })
            
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            corr_val = str(row.get('correct', '')).lower()
            if corr_val in ['1', 'true', '1.0', 't']:
                all_trials.append({'idx': trial_idx, 'correct': 1, 'onset_tr_global': current_tr_offset + int(onset_sec / tr_sec)})
            else:
                all_trials.append({'idx': trial_idx, 'correct': 0, 'onset_tr_global': current_tr_offset + int(onset_sec / tr_sec)})
            trial_idx += 1
            
        current_tr_offset += n_trs
        if os.path.exists(nii_path): os.remove(nii_path)
        
    # Find Grokking
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1:
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['idx'] - 4
        else:
            streak = 0
    if grok_trial is None: return None

    grok_tr = all_trials[grok_trial]['onset_tr_global']
    df = pd.DataFrame(continuous_metrics)
    
    # 1. Measure Baseline
    baseline = df[(df['tr'] >= grok_tr - 20) & (df['tr'] <= grok_tr - 6)]['total_energy'].mean()
    if np.isnan(baseline): return None

    # 2. Find Tau (Retrocausal displacement)
    # Search backwards from grok_tr for the first significant spike above baseline (>1.05x baseline)
    tau_tr = 0
    for t_back in range(1, 10):
        current_tr = grok_tr - t_back
        val = df[df['tr'] == current_tr]['total_energy'].values
        if len(val) > 0 and val[0] > baseline * 1.05:
            tau_tr = t_back
            break
    tau_seconds = tau_tr * tr_sec

    # 3. Measure E_G (Peak metabolic dump ratio)
    post_window = df[(df['tr'] >= grok_tr) & (df['tr'] <= grok_tr + 25)]
    peak_energy = post_window['total_energy'].max()
    e_g_ratio = peak_energy / baseline

    # 4. Measure Fuzziness (Entropy at grokking)
    grok_entropy = df[df['tr'] == grok_tr]['spatial_entropy'].values[0]

    return {
        'sub_id': sub_id,
        'tau_sec': tau_seconds,
        'e_g_ratio': e_g_ratio,
        'entropy': grok_entropy,
        # Penrose constant check: (E_G spike - 1) * tau
        'penrose_const': (e_g_ratio - 1.0) * tau_seconds
    }

def run_penrose_audit():
    grokkers = ['sub-218', 'sub-220', 'sub-225', 'sub-229', 'sub-235', 'sub-240']
    
    print("=" * 80)
    print(" PENROSE TIME EQUATION AUDIT: Tau vs E_G")
    print(" Testing E_G * tau = Constant (Planck Scale)")
    print("=" * 80)
    
    results = []
    for sub in grokkers:
        print(f"  Analyzing {sub}...", flush=True)
        res = measure_quantum_constants(sub)
        if res:
            results.append(res)
            print(f"    Tau: {res['tau_sec']:.1f}s | E_G Ratio: {res['e_g_ratio']:.3f} | Entropy: {res['entropy']:.4f}")

    print("\n" + "=" * 80)
    print(" RESULTS: THE QUANTUM GRAVITY OF CATEGORICAL TIME")
    print("=" * 80)
    print(f" {'Sub':>8} | {'Tau (s)':>8} | {'E_G Ratio':>10} | {'Entropy':>10} | {'EG * Tau'}")
    print("-" * 60)
    
    for r in results:
        print(f" {r['sub_id']:>8} | {r['tau_sec']:>8.1f} | {r['e_g_ratio']:>10.3f} | {r['entropy']:>10.4f} | {r['penrose_const']:>8.3f}")

if __name__ == "__main__":
    run_penrose_audit()
