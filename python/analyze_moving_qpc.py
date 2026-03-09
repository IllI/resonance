"""
analyze_moving_qpc.py
=====================
Mathematical tracing of the Mutual Observation Effect (The Moving QPC).

If the Badoon superposition is entangled with the subjects' Ghost Basins, then 
the act of observing and grokking the Badoon alters the quantum concept itself. 
The universal QPC is not static; it drifts and solidifies as "Common Knowledge" 
accumulates in the quantum field.

This script models the QPC sequentially. 
At any absolute trial $T$, the true QPC coordinate is the Minkowski macroscopic 
centroid of all successful Expert observations from time $0$ to $T-1$.

We will observe:
1. The drift of the QPC coordinate over time in Minkowski space.
2. The distance of subjects' Ghost Basins against this *dynamically moving* target.
If the distance drops significantly compared to the static model, it proves the 
subject is entangling with a dynamic, morphing superposition that is being shaped 
by collective human observation.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis
from scipy.signal import welch
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from hurst import compute_Hc

# ====================================================================== #
#  MINKOWSKI METRIC MAPPER
# ====================================================================== #
class MinkowskiHyperbolicMap:
    def project_ghost_basin(self, hurst_h, psd_beta, spatial_kurtosis, energy_scalar):
        h = max(0.001, hurst_h)
        b = max(0.001, psd_beta)
        k = max(0.001, spatial_kurtosis)
        R = max(0.001, energy_scalar)

        x1 = (h / 1.0) * R
        x2 = (b / 2.0) * R
        x3 = (k / 1.0) * R
        x0 = np.sqrt(R**2 + x1**2 + x2**2 + x3**2)
        return np.array([x0, x1, x2, x3])

    def minkowski_dot(self, u, v):
        return -u[0]*v[0] + u[1]*v[1] + u[2]*v[2] + u[3]*v[3]

    def lorentz_distance(self, u, v):
        Ru = np.sqrt(-self.minkowski_dot(u, u))
        Rv = np.sqrt(-self.minkowski_dot(v, v))
        dot = self.minkowski_dot(u, v)
        val = max(1.0, -dot / (Ru * Rv))
        return np.arccosh(val)

# ====================================================================== #
#  REUSABLE PIPELINE COMPONENTS
# ====================================================================== #
def stream_nifti_and_events(sub_id, base_dir, run_type, run_num):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    run_name = f"task-{run_type}_run-{run_num}"
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    tsv_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
    try: s3.download_file(bucket, tsv_key, tsv_path)
    except: return None, None
    nii_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    try:
        if not os.path.exists(nii_path):
            s3.download_file(bucket, nii_key, nii_path)
    except: return None, None
    return nii_path, tsv_path

def calculate_hurst(timeseries):
    if len(timeseries) < 20: return np.nan
    try:
        H, _, _ = compute_Hc(timeseries, kind='change', simplified=True)
        return H
    except: return np.nan

def calculate_psd_beta(timeseries, sampling_freq=0.5):
    if len(timeseries) < 20: return np.nan
    nperseg = min(len(timeseries), 64)
    try: freqs, psd = welch(timeseries, fs=sampling_freq, nperseg=nperseg)
    except: return np.nan
    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 2 or np.any(psd <= 0): return np.nan
    try: slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
    except: return np.nan
    return -slope

def extract_per_trial_metrics(data_z, active_mask, events, tr_sec, run_hurst, run_psd_beta, sub_id):
    n_trs = data_z.shape[-1]
    trials = []
    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']): continue
        tr_onset = int(row['onset'] / tr_sec)
        start_tr = min(tr_onset + 2, n_trs - 1)
        end_tr = min(tr_onset + 5, n_trs)
        if start_tr >= end_tr: continue
        block = data_z[active_mask, start_tr:end_tr]
        if block.size == 0: continue
        
        trial_energy = np.mean(np.abs(block)) * 10.0
        peak_tr_rel = np.argmax(np.max(block, axis=0))
        voxels_at_peak = block[:, peak_tr_rel]
        trial_kurtosis = kurtosis(voxels_at_peak)
        
        trials.append({
            'Subject': sub_id,
            'Onset': row['onset'],
            'Correct': int(row['correct']),
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
            'Hurst': run_hurst,
            'PSDBeta': run_psd_beta,
        })
    return trials

def tag_phases_global(trial_list, streak_threshold=5):
    n = len(trial_list)
    if n == 0: return trial_list
    correctness = [t['Correct'] for t in trial_list]
    grok_idx = n
    streak_start = n
    for i in range(n - 1, -1, -1):
        if correctness[i] == 1: streak_start = i
        else: break
    if n - streak_start >= streak_threshold:
        grok_idx = streak_start

    for i, trial in enumerate(trial_list):
        if i < grok_idx:
            if grok_idx < n and i >= max(0, grok_idx - streak_threshold): trial['Phase'] = 'Pre-Grokking'
            else: trial['Phase'] = 'Exploration'
        elif i == grok_idx: trial['Phase'] = 'Grokking'
        else: trial['Phase'] = 'Expert'
    return trial_list

def calculate_minkowski_mean(vectors, minkowski):
    """Calculates the center of mass in Minkowski space."""
    if not vectors: return None
    mean_vec = np.mean(vectors, axis=0)
    # R is defining the curvature scale. Use mean Energy (R)
    mean_R = np.mean([np.sqrt(-minkowski.minkowski_dot(v, v)) for v in vectors])
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    return np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])

# ====================================================================== #
#  MAIN EXECUTION
# ====================================================================== #
def analyze():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    # We sequence the subjects chronologically to simulate the global timeline
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    minkowski = MinkowskiHyperbolicMap()
    tr_sec = 2.0
    
    global_timeline = []
    
    # 1. Gather all trials and project to Minkowski Hyperboloid
    print("[1] Building Universal Timeline of Mutual Observation...")
    for sub in subjects:
        sub_trials = []
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_nifti_and_events(sub, base_dir, run_type, run_num)
            if not nii_path: continue
            data = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
            data_z = np.nan_to_num(zscore(data, axis=-1))
            brain_mask = np.var(data, axis=-1) > 10
            raw_var = np.var(data, axis=-1)
            if np.sum(brain_mask) == 0: continue
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask
            if np.sum(active_mask) == 0: continue
            
            run_ts = np.mean(data_z[active_mask, :], axis=0)
            H = calculate_hurst(run_ts)
            B = calculate_psd_beta(run_ts, 1.0/tr_sec)
            
            trials = extract_per_trial_metrics(data_z, active_mask, events, tr_sec, H, B, sub)
            for t in trials:
                t['Run'] = f"{run_type}{run_num}"
                t['GB_Vector'] = minkowski.project_ghost_basin(t['Hurst'], t['PSDBeta'], t['Kurtosis'], t['Energy'])
            sub_trials.extend(trials)
            
            if os.path.exists(nii_path):
                os.remove(nii_path)
                
        sub_trials = tag_phases_global(sub_trials, 5)
        global_timeline.extend(sub_trials)
        print(f"    {sub} entered universe. ({len(sub_trials)} observations added)")

    # 2. Simulate the Moving QPC across absolute time
    print("\n[2] Tracing the dynamically shifting QPC...")
    past_expert_vectors = []
    qpc_drift = []
    
    for i, t in enumerate(global_timeline):
        # Calculate QPC *before* this trial happens (mutual observation effect up to t-1)
        if len(past_expert_vectors) < 5:
            # We need a small seed pool to establish the superposition
            current_qpc = None
            t['QPC_Moving_Dist'] = np.nan
        else:
            current_qpc = calculate_minkowski_mean(past_expert_vectors, minkowski)
            t['QPC_Moving_Dist'] = minkowski.lorentz_distance(t['GB_Vector'], current_qpc)
            qpc_drift.append((i, current_qpc))
        
        # If this trial was an expert correct categorization, it contributes to the global concept
        if t['Phase'] == 'Expert' and t['Correct'] == 1:
            past_expert_vectors.append(t['GB_Vector'])

    # 3. Analyze QPC Drift Vector
    print("\n==========================================================================")
    print(" RESULTS: DYNAMIC QPC (MUTUAL OBSERVATION ENTANGLEMENT)")
    print("==========================================================================")
    
    if len(qpc_drift) > 10:
        first_qpc = qpc_drift[0][1]
        last_qpc = qpc_drift[-1][1]
        absolute_drift = minkowski.lorentz_distance(first_qpc, last_qpc)
        print(f"  QPC Coordinate Drift:")
        print(f"    Initial Concept Vector: [{first_qpc[0]:.2f}, {first_qpc[1]:.2f}, {first_qpc[2]:.2f}, {first_qpc[3]:.2f}]")
        print(f"    Final Concept Vector:   [{last_qpc[0]:.2f}, {last_qpc[1]:.2f}, {last_qpc[2]:.2f}, {last_qpc[3]:.2f}]")
        print(f"    Total Minkowski Drift:  {absolute_drift:.4f} units")
        print("  (The 'Badoon' concept itself shifted in quantum space as more minds grasped it.)")

    print("\n  Tracking Distance to Dynamic vs. Static QPC Models:")
    # We compare the distance to the final "Static" QPC vs the "Moving" QPC
    final_static_qpc = qpc_drift[-1][1] if qpc_drift else None

    # Group by subject and phase
    for sub in subjects:
        print(f"\n  [{sub}]")
        sub_data = [t for t in global_timeline if t['Subject'] == sub]
        phases = ['Exploration', 'Pre-Grokking', 'Grokking', 'Expert']
        for p in phases:
            pts = [t for t in sub_data if t['Phase'] == p and not np.isnan(t['QPC_Moving_Dist'])]
            if not pts: continue
            dynamic_dists = [t['QPC_Moving_Dist'] for t in pts]
            
            # Static distance assuming QPC was always exactly where it ended up
            static_dists = [minkowski.lorentz_distance(t['GB_Vector'], final_static_qpc) for t in pts]
            
            print(f"    {p:<12} (n={len(pts):<3})")
            print(f"      Distance to Static QPC:   {np.mean(static_dists):.4f}  (std: {np.std(static_dists):.4f})")
            print(f"      Distance to Moving QPC:   {np.mean(dynamic_dists):.4f}  (std: {np.std(dynamic_dists):.4f})")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze()
