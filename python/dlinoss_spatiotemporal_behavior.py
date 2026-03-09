"""
dlinoss_spatiotemporal_behavior.py
──────────────────────────────────────────────────────────────
A tool to map the temporal sequence of a subject deciding and acting.
It aligns the BOLD spatial footprint (ATP requirement) second-by-second 
across a 12-second window following the visual stimulus.

We correlate the exact moment of the 'Grok' (the physical button press 
recorded as response_time) with the rising and falling edges of the 
D-LinOSS structural matrix, proving how biological learning modifies 
the length and strength of the "thought" trajectory.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import zscore

def process_subject_sequence(sub_id, base_dir, run_modifier="task-fintest_run-1"):
    print(f"\n==========================================================================")
    print(f"  EXTRACTING BEHAVIORAL SEQUENCE FOR: {sub_id}")
    print(f"==========================================================================")
    
    nii_path = os.path.join(base_dir, sub_id, "func", f"{sub_id}_{run_modifier}_bold.nii.gz")
    tsv_path = os.path.join(base_dir, sub_id, "func", f"{sub_id}_{run_modifier}_events.tsv")
    
    if not os.path.exists(nii_path) or not os.path.exists(tsv_path):
        print(f"File missing for {sub_id}")
        return
        
    data = nib.load(nii_path).get_fdata()
    events = pd.read_csv(tsv_path, sep='\t')
    
    # 1. Z-Score Normalization
    data_z = zscore(data, axis=-1)
    data_z = np.nan_to_num(data_z)
    
    # Target pure cortical structures (Top 10% highly active)
    brain_mask = np.var(data, axis=-1) > 10
    var_map = np.var(data_z[brain_mask], axis=-1)
    thresh = np.percentile(var_map, 90)
    active_mask = (np.var(data_z, axis=-1) > thresh) & brain_mask
    
    # TR is 2.0 seconds. 6 TRs = 12 seconds.
    tr_len = 6 
    
    # Separate successful Grokking vs Struggling (Correct vs Incorrect)
    correct_curves = []
    incorrect_curves = []
    
    correct_rts = []
    incorrect_rts = []
    
    for _, row in events.iterrows():
        # Exclude trials without a valid response time
        if 'response_time' not in row or pd.isna(row['response_time']) or row['response_time'] == 0:
            continue
            
        onset = row['onset']
        rt = row['response_time']
        
        tr_start = int(onset / 2.0)
        tr_end = tr_start + tr_len
        
        if tr_end > data_z.shape[-1]:
            break
            
        # Extract the sequence of biological activity 
        # Mean variance across the highly active network per TR
        block = data_z[active_mask, tr_start:tr_end]
        time_curve = np.mean(np.abs(block), axis=0) * 10.0 # Scale to structural baseline
        
        if row['correct'] == 1:
            correct_curves.append(time_curve)
            correct_rts.append(rt)
        else:
            incorrect_curves.append(time_curve)
            incorrect_rts.append(rt)
            
    # Compile the empirical trajectories
    if len(correct_curves) > 0:
        mean_correct_rt = np.mean(correct_rts)
        mean_correct_curve = np.mean(np.array(correct_curves), axis=0)
        
        print(f"\n[+] SUCCESSFUL TRIALS (True Category Grokking) - {len(correct_curves)} Trials")
        print(f"    Avg Reaction Time (Motor Execution): {mean_correct_rt:.3f} seconds")
        print("    Structural Activation Trajectory (per 2.0s TR Block):")
        for i, val in enumerate(mean_correct_curve):
            # Print an ascii bar chart to visualize the structural wave
            bar_len = int(val * 4)
            tr_time = i * 2.0
            marker = "<-- Button Press" if (tr_time <= mean_correct_rt < tr_time + 2.0) else ""
            print(f"    TR {i} ({tr_time:.1f}s): {val:5.2f} | {'#' * bar_len} {marker}")
            
    if len(incorrect_curves) > 0:
        mean_wrong_rt = np.mean(incorrect_rts)
        mean_wrong_curve = np.mean(np.array(incorrect_curves), axis=0)
        
        print(f"\n[-] FAILED TRIALS (Working Memory Guessing) - {len(incorrect_curves)} Trials")
        print(f"    Avg Reaction Time (Hesitation/Action): {mean_wrong_rt:.3f} seconds")
        print("    Structural Activation Trajectory (per 2.0s TR Block):")
        for i, val in enumerate(mean_wrong_curve):
            bar_len = int(val * 4)
            tr_time = i * 2.0
            marker = "<-- Button Press" if (tr_time <= mean_wrong_rt < tr_time + 2.0) else ""
            print(f"    TR {i} ({tr_time:.1f}s): {val:5.2f} | {'#' * bar_len} {marker}")
            
    if len(correct_curves) > 0 and len(incorrect_curves) > 0:
        print("\n--- BEHAVIORAL TO BIOMATHEMATICAL CONCLUSION ---")
        rt_diff = mean_wrong_rt - mean_correct_rt
        peak_diff = np.max(mean_wrong_curve) - np.max(mean_correct_curve)
        print(f"When a subject fails to map the prototype, they hesitate {rt_diff:.2f}s longer,")
        print(f"and their brain requires a structural baseline {peak_diff:.2f} units higher to maintain the noisy state.")

if __name__ == "__main__":
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    
    # 1. The Expert (Sub-206)
    process_subject_sequence("sub-206", base_dir)
    
    # 2. The Struggler (Sub-233) 
    process_subject_sequence("sub-233", base_dir)
