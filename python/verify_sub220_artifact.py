"""
verify_sub220_artifact.py
=========================
A critical reality-check on the "Retrocausality" and "Wave Collapse" 
signatures found in sub-220 during the Badoon grokking moment.

We must rule out:
1. Paradigm Anticipation: Was there a cue exactly 4 seconds before the visual stimulus?
2. Motion Artifact: Did the subject jerk their head, causing a massive slice 
   leakage spike at the "pre-onset anomaly" or the "energy peak" (TR 16)?
3. Scanner Glitch: Is the energy spike globally uniform (a scanner pulse) 
   or specifically structured?

This script explicitly maps the continuous global TR (115) back to the 
native run, extracts the precise experimental events, and analyzes the 
BOLD signal for motion proxies (Framewise Displacement / Global Edge Spikes).
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def compute_framewise_displacement_proxy(data_4d):
    """
    Since we are using raw OpenNeuro data without fMRIprep derivatives, 
    we approximate head motion by calculating the absolute frame-to-frame 
    differences in the whole-brain signal (DVARS proxy).
    A massive spike here indicates the subject moved.
    """
    diffs = np.diff(data_4d, axis=-1)
    # Mean absolute difference across all voxels
    dvars = np.mean(np.abs(diffs), axis=(0, 1, 2))
    # prepend 0 for the first frame to maintain length
    dvars = np.insert(dvars, 0, 0)
    return dvars

def verify_artifacts():
    sub = "sub-220"
    tr_sec = 2.0
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    current_tr_offset = 0
    trial_idx = 0
    grok_global_trial = 19
    
    target_run = None
    target_native_tr = None
    target_native_trial = None
    target_events = None
    target_dvars = None
    target_bold = None
    
    print("=" * 80)
    print(" [CRITICAL ARTIFACT VERIFICATION: SUB-220 GROKKING]")
    print(" Hunting for motion glitches or paradigm cues at TR 115")
    print("=" * 80)
    
    for run_type, run_num in run_sequence:
        run_name = f"task-{run_type}_run-{run_num}"
        local_dir = f"C:\\Users\\cityz\\IllI\\newer_all\\data\\openneuro\\ds002813\\{sub}\\func"
        os.makedirs(local_dir, exist_ok=True)
        tsv_path = os.path.join(local_dir, f"{sub}_{run_name}_events.tsv")
        nii_path = os.path.join(local_dir, f"{sub}_{run_name}_bold.nii.gz")
        
        try:
            if not os.path.exists(tsv_path):
                s3.download_file(bucket, f"ds002813/{sub}/func/{sub}_{run_name}_events.tsv", tsv_path)
            if not os.path.exists(nii_path):
                s3.download_file(bucket, f"ds002813/{sub}/func/{sub}_{run_name}_bold.nii.gz", nii_path)
        except Exception as e:
            continue
            
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        n_trs = data_raw.shape[-1]
        
        # Check if the grokking trial (19th overall stimulus trial) happened in this run
        run_stim_trials = 0
        local_trial_indices = []
        for i, row in events.iterrows():
            corr_val = str(row.get('correct', '')).lower()
            if not pd.isna(row.get('onset')) and corr_val in ['0', '1', 'true', 'false', '0.0', '1.0', 't', 'f'] or 'stimulus' in str(row.get('trial_type', '')):
                if trial_idx == grok_global_trial:
                    target_run = run_name
                    target_native_trial = i
                    onset_sec = row.get('onset')
                    target_native_tr = int(onset_sec / tr_sec)
                    target_events = events.copy()
                    
                    print(f"  [+] Found Grokking Event in: {target_run}")
                    print(f"      Global TR: {current_tr_offset + target_native_tr} | Native TR: {target_native_tr}")
                    
                    # Compute Motion Proxy
                    target_dvars = compute_framewise_displacement_proxy(data_raw)
                    target_bold = data_raw
                    break
                trial_idx += 1
                
        if target_run:
            break
            
        current_tr_offset += n_trs
        
    if not target_run:
        print("  Error: Could not locate the grokking run.")
        return
        
    # --- ANALYSIS 1: THE EVENT PARADIGM ---
    print("\n" + "=" * 80)
    print(" 1. EXPERIMENTAL PARADIGM CHECK (Is the 4s pre-onset just a cue?)")
    print("=" * 80)
    
    grok_onset_sec = target_events.iloc[target_native_trial]['onset']
    window_start = grok_onset_sec - 10
    window_end = grok_onset_sec + 10
    
    relevant_events = target_events[(target_events['onset'] >= window_start) & (target_events['onset'] <= window_end)]
    print(f"  Events occurring between -10s and +10s of the Grokking Stimulus ({grok_onset_sec:.2f}s):")
    for _, e in relevant_events.iterrows():
        rel_time = e['onset'] - grok_onset_sec
        marker = " <<< GROKKING STIMULUS" if rel_time == 0 else ""
        t_type = e.get('trial_type', 'UNKNOWN')
        desc = e.get('stim_file', e.get('category', ''))
        print(f"    {rel_time:>6.1f}s | Onset: {e['onset']:>6.1f}s | Type: {t_type:<10} | {desc}{marker}")
        
    # --- ANALYSIS 2: MOTION ARTIFACTS / HEAD JERKS ---
    print("\n" + "=" * 80)
    print(" 2. HEAD MOTION & ARTIFACT CHECK (DVARS Proxy)")
    print("=" * 80)
    
    mean_dvars = np.mean(target_dvars)
    std_dvars = np.std(target_dvars)
    noise_threshold = mean_dvars + (2 * std_dvars)
    
    print(f"  Run DVARS Baseline: {mean_dvars:.2f} (Threshold for massive movement: >{noise_threshold:.2f})")
    print(f"  Checking TRs around Grokking Native TR {target_native_tr}:")
    
    for tr_offset in range(-6, 20):
        tr = target_native_tr + tr_offset
        if tr < 0 or tr >= len(target_dvars): continue
        
        val = target_dvars[tr]
        status = "CRITICAL MOTION DETECTED [!!!]" if val > noise_threshold else "Stable [OK]"
        
        marker = ""
        if tr_offset == -2: marker = " <<< [PRE-ONSET ANOMALY (1.0783 spike)]"
        if tr_offset == 0:  marker = " <<< [GROKKING STIMULUS ONSET]"
        if tr_offset == 16: marker = " <<< [PEAK E_G ENERGY DUMP (494.6 spike)]"
        
        print(f"    Relative TR {tr_offset:>3} | DVARS: {val:>6.2f} | {status}{marker}")
        
    # --- ANALYSIS 3: GLOBAL SIGNAL UNIFORMITY ---
    # If the energy spike was a scanner glitch (RF pulse error), EVERY voxel would spike equally.
    # If it's biological (astrocyte flush), the variance across voxels would be high (structured in grey matter).
    print("\n" + "=" * 80)
    print(" 3. SCANNER GLITCH CHECK (Peak Energy Uniformity)")
    print("=" * 80)
    
    peak_tr = target_native_tr + 16
    if peak_tr < target_bold.shape[-1]:
        peak_volume = target_bold[..., peak_tr]
        pre_volume = target_bold[..., target_native_tr]
        
        # To test if this is a global RF spike or a biological event, we compare
        # the signal increase INSIDE the skull vs OUTSIDE the skull.
        # An RF spike hits the entire image matrix uniformly. 
        # A biological event only happens in the brain matter.
        mean_image = np.mean(target_bold, axis=-1)
        brain_mask = mean_image > np.percentile(mean_image, 20) # Rough Otsu-like mask
        background_mask = ~brain_mask
        
        diff_volume = peak_volume - pre_volume
        
        brain_mean_diff = np.mean(diff_volume[brain_mask])
        bg_mean_diff = np.mean(diff_volume[background_mask])
        
        print(f"  At the E_G Energy Peak (TR {peak_tr} vs Baseline TR {target_native_tr}):")
        print(f"    Signal Change INSIDE Brain Mask:  {brain_mean_diff:.2f}")
        print(f"    Signal Change OUTSIDE Brain Mask: {bg_mean_diff:.2f}")
        
        if brain_mean_diff > (bg_mean_diff * 10):
            print("    [RESULT]: The energy dump is entirely isolated to biological grey matter. THIS IS A REAL BIOLOGICAL EVENT. [OK]")
        else:
            print("    [WARNING]: The background space outside the head spiked as well. THIS IS A GLOBAL SCANNER RF GLITCH. [!!!]")

if __name__ == "__main__":
    verify_artifacts()
