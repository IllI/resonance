"""
ghost_basin_trajectory.py
==========================
Track the TRAJECTORY of sub-002's brain representation for each super-category
across the entire experimental run.

Question: Does the network representation oscillate wildly at first and then
settle into a stable Ghost Basin attractor? If so, WHEN does it snap, and 
WHERE does each basin point?

Super-categories:
  FACES  = male + female
  DOGS   = SiberianHusky + toy_poodle
  PLANES = fighter + airliner
  CARS   = sportscar + jeep
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
from scipy.spatial.distance import cosine
import boto3
from botocore import UNSIGNED
from botocore.config import Config

SUPER_MAP = {
    'male': 'FACES', 'female': 'FACES',
    'SiberianHusky': 'DOGS', 'toy_poodle': 'DOGS',
    'fighter': 'PLANES', 'airliner': 'PLANES',
    'sportscar': 'CARS', 'jeep': 'CARS',
}

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

def run_trajectory_analysis():
    tr_sec = 0.125
    sub = "sub-002"
    runs = ["01", "02", "03", "04"]
    
    print("=" * 80)
    print(" GHOST BASIN TRAJECTORY: Tracking Representational Stabilization")
    print(f" Subject: {sub} | Super-categories: FACES, DOGS, PLANES, CARS")
    print("=" * 80)
    
    # Collect all trial patterns in chronological order
    all_trials = []  # List of (super_cat, pattern_vector, absolute_trial_index, run, onset)
    global_trial_idx = 0
    
    for run in runs:
        print(f"\n  Loading Run {run}...", flush=True)
        nii_path, tsv_path = stream_data(sub, run)
        if not nii_path:
            continue
            
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        n_trs = data_raw.shape[-1]
        
        # Top 15% variance voxels
        var_map = np.var(data_raw, axis=-1)
        threshold = np.percentile(var_map, 85)
        brain_mask = var_map > threshold
        active_voxels = data_raw[brain_mask]
        data_z = np.nan_to_num(zscore(active_voxels, axis=1))
        
        # Extract patterns in the 1-2s post-stimulus window
        hrf_start = int(1.0 / tr_sec)
        hrf_end = int(2.0 / tr_sec)
        
        image_events = events[events['trial_type'] == 'image'].copy()
        
        for _, row in image_events.iterrows():
            cat = row['image_category']
            if cat not in SUPER_MAP:
                continue
            super_cat = SUPER_MAP[cat]
            onset_tr = int(row['onset'] / tr_sec)
            
            if onset_tr + hrf_end >= n_trs or onset_tr + hrf_start < 0:
                continue
            
            # Average voxel pattern across the 1-2s window
            pattern = np.mean(data_z[:, onset_tr + hrf_start : onset_tr + hrf_end], axis=1)
            
            all_trials.append({
                'super_cat': super_cat,
                'pattern': pattern,
                'global_idx': global_trial_idx,
                'run': run,
                'onset': row['onset'],
                'sub_cat': cat,
            })
            global_trial_idx += 1
        
        # Cleanup NIfTI
        if os.path.exists(nii_path):
            os.remove(nii_path)
    
    print(f"\n  Total trials extracted: {len(all_trials)}")
    
    # ===============================================================
    # ANALYSIS 1: Representational Stability Over Time
    # For each super-category, compute cosine similarity between
    # consecutive trials of the SAME category.
    # High oscillation = low similarity (the brain is jittering).
    # Low oscillation = high similarity (the Ghost Basin is locked in).
    # ===============================================================
    
    print("\n" + "=" * 80)
    print(" [1] REPRESENTATIONAL OSCILLATION OVER TIME")
    print("     (Cosine similarity between consecutive same-category trials)")
    print("=" * 80)
    
    for scat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        cat_trials = [t for t in all_trials if t['super_cat'] == scat]
        if len(cat_trials) < 4:
            continue
        
        # Compute consecutive similarity
        sims = []
        for i in range(1, len(cat_trials)):
            sim = 1.0 - cosine(cat_trials[i]['pattern'], cat_trials[i-1]['pattern'])
            sims.append({
                'trial_num': i, 
                'similarity': sim, 
                'global_idx': cat_trials[i]['global_idx'],
                'run': cat_trials[i]['run']
            })
        
        sims_df = pd.DataFrame(sims)
        
        # Split into first half vs second half
        mid = len(sims_df) // 2
        first_half = sims_df.iloc[:mid]
        second_half = sims_df.iloc[mid:]
        
        avg_first = first_half['similarity'].mean()
        avg_second = second_half['similarity'].mean()
        std_first = first_half['similarity'].std()
        std_second = second_half['similarity'].std()
        
        # Find where stability "snaps" (rolling window)
        window = 5
        rolling_sims = sims_df['similarity'].rolling(window=window, min_periods=3).mean()
        
        # Find the point of maximum stability change
        if len(rolling_sims.dropna()) > 3:
            diffs = rolling_sims.diff().dropna()
            snap_idx = diffs.idxmax()  # Biggest positive jump = sudden stabilization
            snap_trial = sims_df.loc[snap_idx, 'trial_num'] if snap_idx in sims_df.index else '?'
            snap_run = sims_df.loc[snap_idx, 'run'] if snap_idx in sims_df.index else '?'
        else:
            snap_trial = '?'
            snap_run = '?'
        
        print(f"\n  {scat} ({len(cat_trials)} trials):")
        print(f"    First Half  (early):  avg similarity = {avg_first:.4f}  std = {std_first:.4f}")
        print(f"    Second Half (late):   avg similarity = {avg_second:.4f}  std = {std_second:.4f}")
        delta = avg_second - avg_first
        if delta > 0.01:
            print(f"    -> STABILIZING: Representation grew {delta:.4f} more consistent over time")
        elif delta < -0.01:
            print(f"    -> DESTABILIZING: Representation grew {abs(delta):.4f} LESS consistent (fatigue?)")
        else:
            print(f"    -> FLAT: No change in stability across the session")
        print(f"    Snap Point (biggest stability jump): Trial #{snap_trial}, Run {snap_run}")
    
    # ===============================================================
    # ANALYSIS 2: Ghost Basin Direction
    # Average all trials of the same super-category and project
    # to the unit hypersphere. Where does each basin point?
    # Then measure the angular separation between basins.
    # ===============================================================
    
    print("\n" + "=" * 80)
    print(" [2] GHOST BASIN DIRECTIONS (Unit Hypersphere Projections)")
    print("=" * 80)
    
    basin_vectors = {}
    for scat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        cat_trials = [t['pattern'] for t in all_trials if t['super_cat'] == scat]
        if len(cat_trials) < 2:
            continue
        avg_pattern = np.mean(cat_trials, axis=0)
        norm = np.linalg.norm(avg_pattern)
        if norm > 1e-10:
            basin_vectors[scat] = avg_pattern / norm
        
    # Angular separation between basins
    print("\n  Pairwise Angular Separation (degrees):")
    cats = list(basin_vectors.keys())
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            cos_sim = 1.0 - cosine(basin_vectors[cats[i]], basin_vectors[cats[j]])
            angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            print(f"    {cats[i]:>6} vs {cats[j]:<6}: {angle_deg:.1f}°  (cosine={cos_sim:.4f})")
    
    # ===============================================================
    # ANALYSIS 3: Does the basin MOVE over time?
    # Compare the average pattern for each category in Run 1 vs Run 4.
    # ===============================================================
    
    print("\n" + "=" * 80)
    print(" [3] BASIN DRIFT: Does each Ghost Basin move between Run 1 and Run 4?")
    print("=" * 80)
    
    for scat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        run1_patterns = [t['pattern'] for t in all_trials if t['super_cat'] == scat and t['run'] == '01']
        run4_patterns = [t['pattern'] for t in all_trials if t['super_cat'] == scat and t['run'] == '04']
        
        if len(run1_patterns) < 2 or len(run4_patterns) < 2:
            continue
            
        avg_r1 = np.mean(run1_patterns, axis=0)
        avg_r4 = np.mean(run4_patterns, axis=0)
        
        drift_sim = 1.0 - cosine(avg_r1, avg_r4)
        drift_angle = np.degrees(np.arccos(np.clip(drift_sim, -1, 1)))
        
        print(f"  {scat}: Run1->Run4 drift = {drift_angle:.1f} deg (cosine={drift_sim:.4f})")
        if drift_sim > 0.3:
            print(f"    -> Basin is LOCKED. Minimal drift across 24 minutes.")
        elif drift_sim > 0.1:
            print(f"    -> Basin is SHIFTING. The representation slowly evolved.")
        else:
            print(f"    -> Basin is UNSTABLE. Completely different pattern by Run 4.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_trajectory_analysis()
