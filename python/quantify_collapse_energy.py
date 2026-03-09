"""
quantify_collapse_energy.py
===========================
Maps the spatial dispersion and decay rate of the energetic spike 
released during the theoretical Penrose Orch OR wave function collapse 
at the exact moment of grokking (sub-220).

Penrose Equation:
E_G = h_bar / tau
When the superposition collapses due to gravitational self-energy threshold, 
the structural energy is released into the surrounding biological manifold.
Metabolically, this demands an aggressive astroglial clearing response, 
visible as a BOLD contrast anomaly.

We quantify:
1. Total System Energy Magnitude (L2 norm of BOLD anomalies) at each TR.
2. Spatial Dispersion (Entropy of the Energy map). A low entropy means 
   a highly concentrated geometric spike. High entropy means wide dispersion.
3. The Decay Rate (how the energy dissipates through the grey matter over time).
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
    # Normalize to a probability distribution over the brain
    p = np.abs(voxel_energies) / (np.sum(np.abs(voxel_energies)) + 1e-10)
    # Remove zeros to prevent log(0)
    p = p[p > 1e-10]
    return entropy(p)

def run_energy_dispersion_test():
    sub = "sub-220"
    tr_sec = 2.0
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    print("=" * 80)
    print(" QUANTIFYING COLLAPSE ENERGY AND SPATIAL DISPERSION")
    print(" Tracking the E_G wave function collapse in sub-220")
    print("=" * 80)
    
    continuous_ts = []
    run_boundaries = []
    current_tr_offset = 0
    all_trials = []
    trial_idx = 0
    
    # Track continuous metrics across runs
    continuous_metrics = []
    
    for run_type, run_num in run_sequence:
        print(f"  Loading {run_type}-{run_num}...", flush=True)
        nii_path, tsv_path = stream_data(sub, run_type, run_num)
        if not nii_path: continue
        
        try:
            img = nib.load(nii_path)
            data_raw = img.get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
        except: continue
        
        n_trs = data_raw.shape[-1]
        
        # Masking per run
        var_map = np.var(data_raw, axis=-1)
        brain_mask = var_map > 10
        if not np.any(brain_mask): continue
        active_mask = (var_map > np.percentile(var_map[brain_mask], 85)) & brain_mask
        
        data_z = np.nan_to_num(zscore(data_raw[active_mask], axis=-1))
        
        # Compute metrics immediately for this run
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
            
        # Extract events to find grokking
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
        
        if os.path.exists(nii_path):
            os.remove(nii_path)
            
    # Identify grokking TR precisely
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
        print("  Error: No grokking found.")
        return
        
    grok_tr = all_trials[grok_trial]['onset_tr_global']
    print(f"\n  Grokking localized at TR {grok_tr} (Trial {grok_trial})")
    
    # -------------------------------------------------------------------------
    # ANALYZE THE COLLAPSE EVENT
    # -------------------------------------------------------------------------
    
    df_all = pd.DataFrame(continuous_metrics)
    
    # Window of analysis: 10 TRs before to 40 TRs after the grokking onset
    start_tr = max(0, grok_tr - 10)
    end_tr = min(df_all['tr'].max() + 1, grok_tr + 40)
    
    df = df_all[(df_all['tr'] >= start_tr) & (df_all['tr'] < end_tr)].copy()
    df['rel_tr'] = df['tr'] - grok_tr
    
    # Find the peak energy
    peak_tr = df.loc[df['total_energy'].idxmax()]['tr']
    peak_rel = peak_tr - grok_tr
    
    print("\n" + "=" * 80)
    print(" METABOLIC E_G COLLAPSE SIGNATURE (Energy & Dispersion)")
    print("=" * 80)
    
    print(f"\n  {'TR Relative':>12} | {'Total Energy (L2)':>20} | {'Spatial Entropy (Dispersion)':>32}")
    print(f"  {'-'*70}")
    
    for _, r in df.iterrows():
        rt = int(r['rel_tr'])
        marker = ""
        if rt == 0:
            marker = "  <<< GROKKING STIMULUS ONSET"
        elif rt == peak_rel:
            marker = "  <<< PEAK ENERGY SPIKE"
            
        bar_len = min(50, int((r['total_energy'] / df['total_energy'].max()) * 50))
        bar = "#" * bar_len
        
        print(f"  {rt:>12} | {r['total_energy']:>10.1f} {bar:<25} | {r['spatial_entropy']:>10.4f}{marker}")
        
    # Rate of decay calculation
    peak_idx = df['total_energy'].idxmax()
    if peak_idx + 1 < len(df):
        post_peak = df.iloc[peak_idx:]
        
        # Fit an exponential decay curve: E(t) = E0 * e^(-k*t) -> ln(E) = ln(E0) - k*t
        t_vals = np.arange(len(post_peak))
        log_e = np.log(post_peak['total_energy'].values)
        slope, _, _, _, _ = linregress(t_vals, log_e)
        k_decay = -slope
        half_life_trs = np.log(2) / k_decay if k_decay > 0 else np.nan
        
        print("\n" + "=" * 80)
        print(" ENERGY DECAY RATE (POST-COLLAPSE)")
        print("=" * 80)
        
        print(f"  Peak Energy Magnitude:    {post_peak.iloc[0]['total_energy']:.1f}")
        print(f"  Exponential Decay Rate k: {k_decay:.4f}  (Equation: E(t) = E0 * e^{-k_decay:.4f}*t)")
        if not np.isnan(half_life_trs):
            print(f"  Metabolic Half-Life:      {half_life_trs:.1f} TRs ({(half_life_trs * tr_sec):.1f} seconds)")
            print(f"  This is the speed at which astrocytes cleared the $E_G$ wave collapse energy.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_energy_dispersion_test()
