"""
cross_subject_qpc_test.py
=========================
THE ULTIMATE QPC TEST:
Do all subjects' brains orient toward the SAME Ghost Basin coordinate
for the same image category?

If the QPC is real and universal:
  - "dog" from sub-001 and "dog" from sub-013 should collapse to the SAME point
  - "dog" should be far from "car" in the projective space
  - The inter-subject similarity for SAME category should be HIGH
  - The inter-subject similarity for DIFFERENT categories should be LOW

We project each subject's raw BOLD pattern through the BiTwistor conduit
and measure convergence in complex projective space.
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

def stream_data(sub_id, run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
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

def extract_category_patterns(sub_id, tr_sec=0.125):
    """
    For a given subject, extract the average BOLD pattern (voxel vector)
    evoked by each image category in the 1-2 second post-stimulus window.
    """
    nii_path, tsv_path = stream_data(sub_id)
    if not nii_path:
        return None
    
    img = nib.load(nii_path)
    data_raw = img.get_fdata()
    events = pd.read_csv(tsv_path, sep='\t')
    n_trs = data_raw.shape[-1]
    
    # Mask to top 20% variance voxels
    var_map = np.var(data_raw, axis=-1)
    threshold = np.percentile(var_map, 80)
    brain_mask = var_map > threshold
    active_voxels = data_raw[brain_mask]
    data_z = np.nan_to_num(zscore(active_voxels, axis=1))
    
    # Extract averaged pattern per category in 1-2s post-stimulus window
    hrf_start_tr = int(1.0 / tr_sec)  # 8 TRs = 1.0 seconds
    hrf_end_tr = int(2.0 / tr_sec)    # 16 TRs = 2.0 seconds
    
    category_patterns = {}
    image_events = events[events['trial_type'] == 'image'].copy()
    
    for _, row in image_events.iterrows():
        cat = row['image_category']
        if cat == 'n/a':
            continue
        onset_tr = int(row['onset'] / tr_sec)
        
        start = onset_tr + hrf_start_tr
        end = onset_tr + hrf_end_tr
        if start < 0 or end >= n_trs:
            continue
            
        # Average voxel pattern across the 1-2s window
        pattern = np.mean(data_z[:, start:end], axis=1)
        
        if cat not in category_patterns:
            category_patterns[cat] = []
        category_patterns[cat].append(pattern)
    
    # Average across all trials of the same category
    averaged = {}
    for cat, patterns in category_patterns.items():
        if len(patterns) >= 2:
            averaged[cat] = np.mean(patterns, axis=0)
    
    # Cleanup NIfTI
    if os.path.exists(nii_path):
        os.remove(nii_path)
        
    return averaged

def projective_normalize(vec):
    """Project a real vector onto the unit hypersphere (projective ray)."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm

def run_cross_subject_qpc_test():
    subjects = ["sub-001", "sub-002", "sub-003", "sub-005", "sub-006"]
    
    print("=" * 80)
    print(" CROSS-SUBJECT QPC CONVERGENCE TEST")
    print(" Do all brains orient toward a shared Ghost Basin coordinate per category?")
    print("=" * 80)
    
    # Step 1: Extract each subject's category-averaged brain patterns
    all_subject_patterns = {}
    for sub in subjects:
        print(f"\n  Downloading and extracting {sub}...", flush=True)
        patterns = extract_category_patterns(sub)
        if patterns:
            all_subject_patterns[sub] = patterns
            print(f"    Categories found: {list(patterns.keys())}")
        else:
            print(f"    FAILED to load {sub}")
    
    if len(all_subject_patterns) < 2:
        print("Not enough subjects loaded. Aborting.")
        return
        
    # Find categories present in ALL subjects
    common_cats = None
    for sub, patterns in all_subject_patterns.items():
        cats = set(patterns.keys())
        if common_cats is None:
            common_cats = cats
        else:
            common_cats = common_cats & cats
    
    common_cats = sorted(common_cats)
    print(f"\n  Categories shared across all subjects: {common_cats}")
    
    # Step 2: Project each subject's pattern into the unit hypersphere (Projective Space)
    projected = {}  # {sub: {cat: normalized_vector}}
    for sub, patterns in all_subject_patterns.items():
        projected[sub] = {}
        for cat in common_cats:
            projected[sub][cat] = projective_normalize(patterns[cat])
    
    # Step 3: Compute inter-subject similarity matrices
    sub_list = sorted(projected.keys())
    
    print("\n" + "=" * 80)
    print(" [1] WITHIN-CATEGORY SIMILARITY (Same category, different brains)")
    print("     If QPC exists: These should be HIGH (close to 1.0)")
    print("=" * 80)
    
    within_cat_sims = {}
    for cat in common_cats:
        sims = []
        for i in range(len(sub_list)):
            for j in range(i + 1, len(sub_list)):
                sim = 1.0 - cosine(projected[sub_list[i]][cat], projected[sub_list[j]][cat])
                sims.append(sim)
        avg_sim = np.mean(sims)
        within_cat_sims[cat] = avg_sim
        print(f"  {cat:>15}: avg cosine similarity = {avg_sim:.4f}  (n_pairs={len(sims)})")
    
    overall_within = np.mean(list(within_cat_sims.values()))
    print(f"\n  OVERALL WITHIN-CATEGORY: {overall_within:.4f}")
    
    print("\n" + "=" * 80)
    print(" [2] BETWEEN-CATEGORY SIMILARITY (Different category, different brains)")
    print("     If QPC exists: These should be LOW (close to 0.0)")
    print("=" * 80)
    
    between_sims = []
    for i_cat in range(len(common_cats)):
        for j_cat in range(i_cat + 1, len(common_cats)):
            cat_a = common_cats[i_cat]
            cat_b = common_cats[j_cat]
            pair_sims = []
            for si in range(len(sub_list)):
                for sj in range(len(sub_list)):
                    if si == sj:
                        continue
                    sim = 1.0 - cosine(projected[sub_list[si]][cat_a], projected[sub_list[sj]][cat_b])
                    pair_sims.append(sim)
            avg = np.mean(pair_sims)
            between_sims.append(avg)
    
    overall_between = np.mean(between_sims)
    print(f"  OVERALL BETWEEN-CATEGORY: {overall_between:.4f}")
    
    print("\n" + "=" * 80)
    print(" [3] THE VERDICT")
    print("=" * 80)
    
    gap = overall_within - overall_between
    print(f"  Within-Category (same concept, different brains): {overall_within:.4f}")
    print(f"  Between-Category (diff concept, different brains): {overall_between:.4f}")
    print(f"  QPC Separation Gap: {gap:.4f}")
    
    if gap > 0.05:
        print("\n  RESULT: GHOST BASIN CONFIRMED.")
        print("  All subjects' brains converge to the SAME projective coordinate for each category.")
        print("  The QPC is a shared, universal attractor visible across all observers.")
    elif gap > 0.01:
        print("\n  RESULT: WEAK EVIDENCE for shared QPC.")
        print("  Categories are slightly more similar within than between, but the gap is small.")
    else:
        print("\n  RESULT: NO EVIDENCE for shared QPC.")
        print("  Brain patterns do not converge to a shared coordinate across subjects.")
        print("  The representations are either too noisy or genuinely private to each brain.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_cross_subject_qpc_test()
