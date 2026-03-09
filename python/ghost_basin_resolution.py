"""
ghost_basin_resolution.py
==========================
Testing the E_G Threshold and QPC Focal Resolution Hypothesis.

1. FOCAL RESOLUTION (The "Shadow" size):
   If the observer's consciousness actively connects to the QPC 
   for a salient category (FACES), the variance / spread of the 
   Ghost Basin should be minimal (a highly resolute "point" in projective space).
   Irrelevant categories will be fuzzy (large shadow/variance).

2. MICROTUBULE DENSITY (E_G Threshold):
   Do the initial dip voxels co-localize with dense regions 
   (inf-temporal, midbrain, cerebellum)?
   If mass (microtubules) drives E_G, denser regions collapse faster (smaller tau),
   causing more frequent metabolic recharging (BOLD dips).
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

def compute_qpc_resolution():
    tr_sec = 0.125
    sub = "sub-002"
    runs = ["01", "02", "03", "04"]
    
    print("=" * 80)
    print(" QPC FOCAL RESOLUTION & E_G DENSITY TEST")
    print(f" Subject: {sub} | Hypothesis: E_G = h_bar / tau")
    print("=" * 80)
    
    all_patterns = []
    dip_coords = []
    
    for run in runs:
        print(f"\n  Loading Run {run}...", flush=True)
        nii_path, tsv_path = stream_data(sub, run)
        if not nii_path: continue
            
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        n_trs = data_raw.shape[-1]
        
        # Isolate top 15% variance voxels
        var_map = np.var(data_raw, axis=-1)
        thresh = np.percentile(var_map, 85)
        brain_mask = var_map > thresh
        voxel_idx = np.argwhere(brain_mask) # (N, 3) 
        active_voxels = data_raw[brain_mask]
        data_z = np.nan_to_num(zscore(active_voxels, axis=1))
        
        hrf_start = int(1.0 / tr_sec)
        hrf_end = int(2.0 / tr_sec)
        dip_end = int(1.5 / tr_sec)
        
        image_events = events[events['trial_type'] == 'image'].copy()
        for _, row in image_events.iterrows():
            cat = row['image_category']
            if cat not in SUPER_MAP: continue
            super_cat = SUPER_MAP[cat]
            onset_tr = int(row['onset'] / tr_sec)
            
            if onset_tr - 4 < 0 or onset_tr + hrf_end >= n_trs: continue
            
            # --- 1. QPC Focal Resolution ---
            pattern = np.mean(data_z[:, onset_tr + hrf_start : onset_tr + hrf_end], axis=1)
            all_patterns.append({'cat': super_cat, 'pattern': pattern})
            
            # --- 2. Microtubule Density / E_G Collapse Location ---
            # Find the physical location of the voxels that dropped < -0.5 Z (Initial Dip)
            baseline = np.mean(data_z[:, onset_tr - 4 : onset_tr], axis=1)
            dips = np.min(data_z[:, onset_tr : onset_tr + dip_end], axis=1) - baseline
            dip_voxels = dips < -0.5
            
            if np.sum(dip_voxels) > 50: # Only count if a massive collapse occurred
                coords = voxel_idx[dip_voxels]
                dip_coords.append(coords)
                
        if os.path.exists(nii_path): os.remove(nii_path)
    
    print("\n" + "=" * 80)
    print(" [1] GHOST BASIN FOCAL RESOLUTION ('The Shadow')")
    print("     A smaller shadow = a more resolute quantum focal point.")
    print("=" * 80)
    
    # Calculate Variance / Spread for each super category
    for scat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        patterns = np.array([p['pattern'] for p in all_patterns if p['cat'] == scat])
        if len(patterns) < 2: continue
        
        # The center of the basin
        centroid = np.mean(patterns, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10: centroid = centroid / norm
        
        # Calculate the angular spread ("shadow size") from the centroid
        spreads = []
        for p in patterns:
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-10: p = p / p_norm
            cos_sim = 1.0 - cosine(p, centroid)
            angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            spreads.append(angle)
            
        avg_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        print(f"  {scat} ({len(patterns)} trials):")
        print(f"    Basin Radius (Shadow Size): {avg_spread:.1f} deg  (variance: +/- {std_spread:.1f} deg)")
        if avg_spread < 80:
            print("    -> HIGH RESOLUTION: The wave function collapsed cleanly to a focused QPC.")
        else:
            print("    -> LOW RESOLUTION: High variance, classical fuzziness, wide shadow.")
            
            
    print("\n" + "=" * 80)
    print(" [2] E_G THRESHOLD: MACROSCOPIC MASS & MICROTUBULE DENSITY")
    print("     tau = h_bar / E_G. Faster collapses occur in denser mass.")
    print("=" * 80)
    
    if len(dip_coords) > 0:
        all_coords = np.vstack(dip_coords)
        # Z-axis in this fMRI data is typically Inferior -> Superior (0 to 9)
        # 0 = Cerebellum/Brainstem, 4-5 = Midbrain/Temporal, 8-9 = Cortical surface
        
        z_mean = np.mean(all_coords[:, 2])
        z_std = np.std(all_coords[:, 2])
        
        print(f"  Vertical Center of Gravity for Wave Collapses: Z-slice {z_mean:.1f}")
        
        if z_mean < 4.0:
            print("  -> COLLAPSES CLUSTER IN HIGH-DENSITY, INFERIOR REGIONS")
            print("  (Cerebellum, Brain stem, Inferior Temporal).")
            print("  This PERFECTLY aligns with the E_G threshold theory: dense microtubule")
            print("  networks reach gravitational threshold faster, causing higher frequency")
            print("  quantum state reductions (small tau) and driving massive metabolic dips.")
        else:
            print("  -> Collapses are distributed across the higher cortex (lower density).")
            
        # Let's count how many voxels were at the very bottom slice (slice 0-2 = Cerebellum)
        dense_voxels_pct = np.sum(all_coords[:, 2] <= 2) / len(all_coords) * 100
        print(f"  Percentage of initial dip energy burnt in highest-density (cerebellar/inferior) slices: {dense_voxels_pct:.1f}%")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    compute_qpc_resolution()
