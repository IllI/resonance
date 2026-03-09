"""
sub2_deep_dive.py
=================
A hyper-focused analysis on sub-002 (The Expert Observer).

Objective:
1. Identify WHAT stimuli trigger quantum candidates.
2. Determine WHEN they happen (timeline/habituation correlation).
3. Map WHERE in the brain the quantum structuralization (Initial Dip) physically occurs.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, linregress, pearsonr
from scipy.fft import fft, fftfreq
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json

def stream_data(sub_id="sub-002", run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
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

def run_deep_dive():
    tr_sec = 0.125
    sub = "sub-002"
    runs = ["01", "02", "03", "04"]
    
    print("=====================================================================")
    print(f" EXPERT OBSERVER DEEP DIVE: {sub}")
    print(" Mapping the WHAT, WHEN, and WHERE of Quantum Wave Collapses")
    print("=====================================================================\n")
    
    all_trials = []
    
    for run in runs:
        print(f"Loading Run {run}...")
        nii_path, tsv_path = stream_data(sub, run)
        if not nii_path:
            continue
            
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        
        n_trs = data_raw.shape[-1]
        
        # We need the physical coordinates (x,y,z) of the active voxels
        var_map = np.var(data_raw, axis=-1)
        threshold = np.percentile(var_map, 85) # Top 15% active voxels
        brain_mask = var_map > threshold
        voxel_coords = np.argwhere(brain_mask) # Shape: (N_voxels, 3)
        active_voxels = data_raw[brain_mask]
        
        # Z-score the physical voxels
        data_z = np.nan_to_num(zscore(active_voxels, axis=1))
        
        # Global signal for timing/signatures
        global_ts = np.mean(data_z, axis=0)
        
        image_events = events[events['trial_type'] == 'image'].copy()
        
        for trial_idx, row in image_events.iterrows():
            onset_sec = row['onset']
            cat = row['image_category']
            onset_tr = int(onset_sec / tr_sec)
            
            dip_end_trs = int(1.5 / tr_sec)
            peak_start_trs = int(2.0 / tr_sec)
            peak_end_trs = int(4.0 / tr_sec)
            
            if onset_tr - 4 < 0 or onset_tr + 32 >= n_trs:
                continue
            
            # 1. Global Signatures
            baseline = np.mean(global_ts[onset_tr - 4 : onset_tr])
            dip_segment = global_ts[onset_tr : onset_tr + dip_end_trs]
            global_dip = np.min(dip_segment) - baseline
            
            is_quantum = global_dip < -0.05
            
            # 2. WHERE did it happen? (Physical Voxel Mapping)
            # Find which specific voxels crashed in oxygen (Initial Dip < -0.5 Z)
            # during the first 1.5 seconds.
            voxel_baselines = np.mean(data_z[:, onset_tr - 4 : onset_tr], axis=1)
            voxel_dips = np.min(data_z[:, onset_tr : onset_tr + dip_end_trs], axis=1) - voxel_baselines
            
            # Voxels that physically spiked/burned oxygen instantly
            quantum_voxels_mask = voxel_dips < -0.5 
            n_quantum_voxels = np.sum(quantum_voxels_mask)
            
            center_of_mass = np.nan
            if n_quantum_voxels > 50:
                coords = voxel_coords[quantum_voxels_mask]
                center_of_mass = np.mean(coords, axis=0) # [X, Y, Z]
            
            all_trials.append({
                'Run': int(run),
                'Absolute_Time_Sec': onset_sec + (int(run) - 1) * 380, # approximate continuous time
                'Category': cat,
                'Global_Dip': global_dip,
                'Is_Quantum': is_quantum,
                'N_Quantum_Voxels': n_quantum_voxels,
                'CoM_X': center_of_mass[0] if type(center_of_mass) != float else np.nan,
                'CoM_Y': center_of_mass[1] if type(center_of_mass) != float else np.nan,
                'CoM_Z': center_of_mass[2] if type(center_of_mass) != float else np.nan
            })
            
        # Cleanup
        if os.path.exists(nii_path):
            os.remove(nii_path)

    df = pd.DataFrame(all_trials)
    
    # ---------------------------------------------------------
    # ANALYSIS: THE WHAT
    # ---------------------------------------------------------
    print("\n[1] THE WHAT: Which stimuli evoke the Quantum Wave Collapse?")
    quantum_events = df[df['Is_Quantum'] == True]
    total_events = len(df)
    
    print(f"  Total Trials: {total_events}")
    print(f"  Quantum Initiations: {len(quantum_events)}")
    
    cat_rates = []
    for cat in df['Category'].unique():
        if cat == 'n/a': continue
        cat_trials = df[df['Category'] == cat]
        q_hits = len(cat_trials[cat_trials['Is_Quantum'] == True])
        rate = q_hits / len(cat_trials) * 100
        cat_rates.append((cat, rate, q_hits))
        
    cat_rates.sort(key=lambda x: x[1], reverse=True)
    for c in cat_rates:
        print(f"    {c[0]:>15}: {c[1]:.1f}% trigger rate ({c[2]} events)")
        
    # ---------------------------------------------------------
    # ANALYSIS: THE WHEN
    # ---------------------------------------------------------
    print("\n[2] THE WHEN: Timeseries Correlation (Habituation vs Fatigue)")
    corr, p_val = pearsonr(df['Absolute_Time_Sec'], df['Is_Quantum'].astype(int))
    print(f"  Correlation between Time in Tube and Quantum Events: {corr:.3f} (p={p_val:.3f})")
    
    run_rates = df.groupby('Run')['Is_Quantum'].mean() * 100
    for r, rate in run_rates.items():
        print(f"    Run {r}: {rate:.1f}% quantum hit rate")
        
    if corr < -0.15:
        print("  -> MASSIVE HABITUATION: The brain stopped grokking as it memorized the pipeline.")
    elif corr > 0.15:
        print("  -> FATIGUE/AROUSAL: The brain exerted more quantum structuralization as time went on.")
    else:
        print("  -> CONSTANT: The Ghost Basin operated consistently across the entire 24-minute session.")
        
    # ---------------------------------------------------------
    # ANALYSIS: THE WHERE
    # ---------------------------------------------------------
    print("\n[3] THE WHERE: Spatial Location of the Initial Dip (Quantum Voxels)")
    
    # The brain box is 64x64x9
    # X (0-64): Left to Right
    # Y (0-64): Posterior (Visual) to Anterior (Frontal)
    # Z (0-9):  Inferior to Superior (Limited slice coverage in this fast TR)
    
    valid_coords = quantum_events.dropna(subset=['CoM_X'])
    avg_x = valid_coords['CoM_X'].mean()
    avg_y = valid_coords['CoM_Y'].mean()
    avg_z = valid_coords['CoM_Z'].mean()
    
    print(f"  Average Center of Mass for Quantum Collapses: X={avg_x:.1f}, Y={avg_y:.1f}, Z={avg_z:.1f} (in 64x64x9 grid)")
    
    # Infer brain region based on Y-axis (Posterior=0, Anterior=64)
    if avg_y < 25:
        region = "Deep Posterior (Visual Cortex / Occipital Lobe)"
        desc = "The quantum structuralization is happening purely in the sensory pipeline. Immediate bottom-up grokking."
    elif avg_y > 40:
        region = "Anterior (Prefrontal / Cognitive Control)"
        desc = "The quantum event is high-level cognitive decision making (top-down)."
    else:
        region = "Mid-Brain / Temporal"
        desc = "The event is occurring in the associative memory/semantic networks."
        
    print(f"  Physical Region: {region}")
    print(f"  Biological Meaning: {desc}")
    print(f"  Average number of voxels simultaneously burning oxygen per event: {valid_coords['N_Quantum_Voxels'].mean():.0f} / 36,864 total slice volume")
    print("  -> This equates to a Highly Localized, localized, high-frequency energy burn.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_deep_dive()
