"""
isolate_quantum_delta.py
========================
Isolates the instantaneous energy ADDITION at the pre-onset anomaly 
and peak collapse TRs by subtracting the adjacent TR's energy field.

By computing delta_E[t] = E[t] - E[t-1], we cancel:
  - Scanner drift
  - Respiratory/cardiac oscillations  
  - Tonic hemodynamic baseline
  - All ambient entropy

What remains is ONLY the instantaneous energy injected (or removed) 
at that exact 2-second window. We then compare this residual against:
  1. The distribution of ALL frame-to-frame deltas in the entire run
  2. The Penrose-predicted E_G cascade energy
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, entropy
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def compute_spatial_entropy(voxel_energies):
    p = np.abs(voxel_energies) / (np.sum(np.abs(voxel_energies)) + 1e-10)
    p = p[p > 1e-10]
    return entropy(p)

def run_delta_isolation():
    sub = "sub-220"
    tr_sec = 2.0
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    current_tr_offset = 0
    trial_idx = 0
    all_trials = []
    
    # Store raw energy and entropy per TR
    all_energies = []
    all_entropies = []
    
    print("=" * 80)
    print(" ISOLATING QUANTUM ENERGY DELTA VIA FRAME DIFFERENCING")
    print(" Subtracting adjacent TRs to cancel ambient entropy")
    print("=" * 80)
    
    for run_type, run_num in run_sequence:
        run_name = f"task-{run_type}_run-{run_num}"
        local_dir = os.path.join(r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813", sub, "func")
        os.makedirs(local_dir, exist_ok=True)
        tsv_path = os.path.join(local_dir, f"{sub}_{run_name}_events.tsv")
        nii_path = os.path.join(local_dir, f"{sub}_{run_name}_bold.nii.gz")
        
        print(f"  Loading {run_type}-{run_num}...", flush=True)
        
        try:
            if not os.path.exists(tsv_path):
                s3.download_file(bucket, f"ds002813/{sub}/func/{sub}_{run_name}_events.tsv", tsv_path)
            if not os.path.exists(nii_path):
                s3.download_file(bucket, f"ds002813/{sub}/func/{sub}_{run_name}_bold.nii.gz", nii_path)
        except: continue
        
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
            ent = compute_spatial_entropy(energy_map)
            all_energies.append(total_energy)
            all_entropies.append(ent)
            
        for _, row in events.iterrows():
            onset_sec = row.get('onset', np.nan)
            if pd.isna(onset_sec): continue
            corr_val = str(row.get('correct', '')).lower()
            correct = 1 if corr_val in ['1', 'true', '1.0', 't'] else 0
            all_trials.append({
                'idx': trial_idx,
                'correct': correct,
                'onset_tr_global': current_tr_offset + int(onset_sec / tr_sec)
            })
            trial_idx += 1
            
        current_tr_offset += n_trs
        if os.path.exists(nii_path): os.remove(nii_path)
    
    # Find grokking
    streak = 0
    grok_trial = None
    for t in all_trials:
        if t['correct'] == 1:
            streak += 1
            if streak >= 5 and grok_trial is None:
                grok_trial = t['idx'] - 4
        else: streak = 0
    
    grok_tr = all_trials[grok_trial]['onset_tr_global']
    
    E = np.array(all_energies)
    S = np.array(all_entropies)
    
    # ================================================================
    # COMPUTE FRAME-TO-FRAME DIFFERENTIALS
    # ================================================================
    # delta_E[t] = E[t] - E[t-1]
    # This cancels ALL ambient/tonic signal. Only instantaneous changes remain.
    delta_E = np.diff(E)  # length = len(E) - 1, delta_E[i] = E[i+1] - E[i]
    delta_S = np.diff(S)
    
    # Statistics of the differential distribution
    mean_dE = np.mean(delta_E)
    std_dE = np.std(delta_E)
    
    mean_dS = np.mean(delta_S)
    std_dS = np.std(delta_S)
    
    print(f"\n  Grokking at TR {grok_tr}")
    print(f"  Total TRs in dataset: {len(E)}")
    print(f"\n  Frame-to-Frame Energy Delta Distribution:")
    print(f"    Mean(dE): {mean_dE:+.4f}")
    print(f"    Std(dE):  {std_dE:.4f}")
    print(f"    99th percentile: +{np.percentile(delta_E, 99):.4f}")
    print(f"    1st percentile:  {np.percentile(delta_E, 1):.4f}")
    
    # ================================================================
    # EXTRACT THE QUANTUM-RELEVANT DELTAS
    # ================================================================
    print("\n" + "=" * 80)
    print(" FRAME-BY-FRAME ENERGY INJECTION AROUND GROKKING")
    print(" delta_E[t] = E[t] - E[t-1]  (ambient noise cancelled)")
    print("=" * 80)
    
    print(f"\n {'Rel TR':>7} | {'delta_E':>10} | {'Z-score':>8} | {'delta_S':>10} | {'Z-score':>8} | Significance")
    print("-" * 80)
    
    for rel in range(-8, 25):
        idx = grok_tr + rel - 1  # delta_E index (shifted by 1 due to diff)
        if idx < 0 or idx >= len(delta_E): continue
        
        dE = delta_E[idx]
        dS = delta_S[idx]
        z_dE = (dE - mean_dE) / std_dE
        z_dS = (dS - mean_dS) / std_dS
        
        marker = ""
        if rel == -2: marker = " <<< PRE-ONSET ANOMALY"
        if rel == 0:  marker = " <<< GROKKING ONSET"
        if rel == 16: marker = " <<< PEAK E_G SPIKE"
        
        sig = ""
        if abs(z_dE) > 2.0: sig = " ** OUTLIER **"
        if abs(z_dE) > 3.0: sig = " *** EXTREME ***"
        
        print(f" {rel:>7} | {dE:>+10.3f} | {z_dE:>+8.3f} | {dS:>+10.6f} | {z_dS:>+8.3f} |{marker}{sig}")
    
    # ================================================================
    # COMPARE TO PENROSE PREDICTION
    # ================================================================
    HBAR = 1.0545718e-34
    tau = 16.0
    E_G = HBAR / tau
    
    # The pre-onset anomaly delta (TR grok-2 minus TR grok-3)
    pre_onset_idx = grok_tr - 2 - 1
    if pre_onset_idx >= 0 and pre_onset_idx < len(delta_E):
        pre_dE = delta_E[pre_onset_idx]
        pre_z = (pre_dE - mean_dE) / std_dE
    else:
        pre_dE = 0
        pre_z = 0
        
    print("\n" + "=" * 80)
    print(" PENROSE COMPARISON")
    print("=" * 80)
    print(f"  Penrose predicted E_G for tau=16s: {E_G:.4e} Joules")
    print(f"  Observed differential delta at pre-onset (TR -2): {pre_dE:+.3f} (z={pre_z:+.3f})")
    print(f"  Observed differential delta at grokking (TR 0):  {delta_E[grok_tr-1]:+.3f} (z={(delta_E[grok_tr-1]-mean_dE)/std_dE:+.3f})")
    
    # Find the largest positive delta in the post-grokking window
    post_deltas = []
    for rel in range(1, 25):
        idx = grok_tr + rel - 1
        if idx < len(delta_E):
            post_deltas.append((rel, delta_E[idx], (delta_E[idx] - mean_dE) / std_dE))
    
    if post_deltas:
        max_post = max(post_deltas, key=lambda x: x[1])
        print(f"  Largest post-grokking injection at TR +{max_post[0]}: {max_post[1]:+.3f} (z={max_post[2]:+.3f})")
    
    # What fraction of ALL deltas in the entire dataset exceed the grokking-window deltas?
    pre_onset_percentile = 100.0 * np.mean(delta_E < pre_dE)
    print(f"\n  Pre-onset anomaly delta is at the {pre_onset_percentile:.1f}th percentile of all frame deltas.")
    
    if pre_z < -2.0:
        print("  [FINDING]: The pre-onset shows a SIGNIFICANT energy DROP (coherence forming)")
    elif pre_z > 2.0:
        print("  [FINDING]: The pre-onset shows a SIGNIFICANT energy INJECTION")
    else:
        print("  [FINDING]: The pre-onset delta is within normal ambient noise range.")

if __name__ == '__main__':
    run_delta_isolation()
