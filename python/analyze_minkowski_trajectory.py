"""
analyze_minkowski_trajectory.py
===============================
Redefines the Lorentz Hyperbolic mapping using the true Minkowski metric 
of quantum-to-electromagnetic expression.

Mathematical correction:
The previous mapping applied arccosh() to variables independently. This did not 
respect the geometric constraints of a Lorentz manifold (spacetime). 

To map the Ghost Basin onto the D=3 Hyperboloid:
Let spatial components be the topological state: x1=Hurst, x2=Beta, x3=Kurtosis.
Let Euclidean Energy be the scaling radius R (distance from Wuji singularity).
The time-like dimension x0 (quantum stabilization energy) must satisfy:
  -x0^2 + x1^2 + x2^2 + x3^2 = -R^2
  x0 = sqrt(R^2 + x1^2 + x2^2 + x3^2)

The Wuji singularity is exactly R=0 (x_i = 0). The brain always exists at R > 0, 
stabilized by its energy, entangled to the singularity but safely outside the event horizon.

The hyperbolic distance is calculated strictly via the Minkowski inner product.

We define the true QPC by taking the Minkowski mean of all subjects' EXPERT trials,
representing the refined, stable categorical coordinate in quantum space, 
and trace the learning trajectory against this true universal point.
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
    def __init__(self):
        pass

    def project_ghost_basin(self, hurst_h, psd_beta, spatial_kurtosis, energy_scalar):
        """
        Projects into a 4D Lorentz vector [x0, x1, x2, x3] satisfying the 
        Minkowski constraint -x0^2 + x1^2 + x2^2 + x3^2 = -R^2.
        """
        # Baseline clipping to prevent negative/zero anomalies
        h = max(0.001, hurst_h)
        b = max(0.001, psd_beta)
        k = max(0.001, spatial_kurtosis)
        R = max(0.001, energy_scalar)

        # Space-like dimensions (scaled to normalize dimensions based on physical variance)
        x1 = (h / 1.0) * R
        x2 = (b / 2.0) * R
        x3 = (k / 1.0) * R

        # Time-like dimension (Quantum stabilization energy)
        x0 = np.sqrt(R**2 + x1**2 + x2**2 + x3**2)
        
        return np.array([x0, x1, x2, x3])

    def minkowski_dot(self, u, v):
        """Minkowski inner product: <u, v>_M = -u0*v0 + u1*v1 + u2*v2 + u3*v3"""
        return -u[0]*v[0] + u[1]*v[1] + u[2]*v[2] + u[3]*v[3]

    def lorentz_distance(self, u, v):
        """
        Hyperbolic distance: d(u,v) = arccosh( -<u,v>_M / (R_u * R_v) )
        """
        Ru = np.sqrt(-self.minkowski_dot(u, u))
        Rv = np.sqrt(-self.minkowski_dot(v, v))
        dot = self.minkowski_dot(u, v)
        
        # Clip to 1.0 to handle floating point precision errors where ratio might be 0.9999999
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
    try:
        s3.download_file(bucket, tsv_key, tsv_path)
    except:
        return None, None
    nii_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    try:
        if not os.path.exists(nii_path):
            s3.download_file(bucket, nii_key, nii_path)
    except:
        return None, None
    return nii_path, tsv_path

def calculate_hurst(timeseries):
    if len(timeseries) < 20: return np.nan
    try:
        H, _, _ = compute_Hc(timeseries, kind='change', simplified=True)
        return H
    except:
        return np.nan

def calculate_psd_beta(timeseries, sampling_freq=0.5):
    if len(timeseries) < 20: return np.nan
    nperseg = min(len(timeseries), 64)
    try:
        freqs, psd = welch(timeseries, fs=sampling_freq, nperseg=nperseg)
    except: return np.nan
    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 2 or np.any(psd <= 0): return np.nan
    try: slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
    except: return np.nan
    return -slope

def extract_per_trial_metrics(data_z, active_mask, events, tr_sec, run_hurst, run_psd_beta):
    n_trs = data_z.shape[-1]
    trials = []
    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']): continue
        correct = int(row['correct'])
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
            'Onset': row['onset'],
            'Correct': correct,
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
            'Hurst': run_hurst,
            'PSDBeta': run_psd_beta,
        })
    return trials

def tag_phases(trial_list, streak_threshold=5):
    n = len(trial_list)
    if n == 0: return trial_list
    correctness = [t['Correct'] for t in trial_list]
    grok_idx = n
    streak_start = n
    for i in range(n - 1, -1, -1):
        if correctness[i] == 1:
            streak_start = i
        else:
            break
    if n - streak_start >= streak_threshold:
        grok_idx = streak_start

    for i, trial in enumerate(trial_list):
        if i < grok_idx:
            if grok_idx < n and i >= max(0, grok_idx - streak_threshold):
                trial['Phase'] = 'Pre-Grokking'
            else:
                trial['Phase'] = 'Exploration'
        elif i == grok_idx:
            trial['Phase'] = 'Grokking'
        else:
            trial['Phase'] = 'Expert'
    return trial_list

# ====================================================================== #
#  MAIN EXECUTION
# ====================================================================== #
def analyze():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-220", "sub-229", "sub-217"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    minkowski = MinkowskiHyperbolicMap()
    tr_sec = 2.0
    sampling_freq = 1.0 / tr_sec
    
    all_subject_trials = {}
    
    # 1. Gather all trials and project to Minkowski Hyperboloid
    print("[1] Extracting raw Ghost Basin topological states...")
    for sub in subjects:
        sub_trials = []
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_nifti_and_events(sub, base_dir, run_type, run_num)
            if not nii_path: continue
            data = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\\t')
            data_z = np.nan_to_num(zscore(data, axis=-1))
            brain_mask = np.var(data, axis=-1) > 10
            raw_var = np.var(data, axis=-1)
            if np.sum(brain_mask) == 0: continue
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask
            if np.sum(active_mask) == 0: continue
            
            run_ts = np.mean(data_z[active_mask, :], axis=0)
            H = calculate_hurst(run_ts)
            B = calculate_psd_beta(run_ts, sampling_freq)
            
            trials = extract_per_trial_metrics(data_z, active_mask, events, tr_sec, H, B)
            for t in trials:
                t['Run'] = f"{run_type}{run_num}"
                t['GB_Vector'] = minkowski.project_ghost_basin(t['Hurst'], t['PSDBeta'], t['Kurtosis'], t['Energy'])
            sub_trials.extend(trials)
            
            if os.path.exists(nii_path):
                os.remove(nii_path)
                
        sub_trials = tag_phases(sub_trials, 5)
        all_subject_trials[sub] = sub_trials
        print(f"    {sub} mapped. ({len(sub_trials)} trials)")

    # 2. Identify the true QPC by averaging the Expert phase vectors
    print("\\n[2] Deriving True QPC from refined Expert Network topologies...")
    expert_vectors = []
    for sub, trials in all_subject_trials.items():
        for t in trials:
            if t['Phase'] == 'Expert':
                expert_vectors.append(t['GB_Vector'])
                
    if not expert_vectors:
        print("ERROR: No expert trials found to derive QPC.")
        return
        
    # To find the hyperbolic centroid, we take the arithmetic mean of the Lorentz vectors
    # and re-normalize it to the hyperboloid by fixing x0.
    mean_vec = np.mean(expert_vectors, axis=0)
    
    # R is defining the curvature scale. We'll use the mean Energy (R)
    mean_R = np.mean([np.sqrt(-minkowski.minkowski_dot(v, v)) for v in expert_vectors])
    
    # x0^2 = R^2 + x1^2 + x2^2 + x3^2
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    
    TRUE_QPC = np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])
    print(f"    MINKOWSKI QPC COORDINATE: {TRUE_QPC}")

    # 3. Calculate Trajectories
    print("\\n[3] Calculating Minkowski Trajectories...")
    for sub, trials in all_subject_trials.items():
        for i, t in enumerate(trials):
            d = minkowski.lorentz_distance(t['GB_Vector'], TRUE_QPC)
            t['QPC_Dist'] = d
            
            # Pointer Vector: movement toward QPC
            if i > 0:
                prev_gb = trials[i-1]['GB_Vector']
                delta = t['GB_Vector'] - prev_gb
                ideal = TRUE_QPC - prev_gb
                delta_norm = np.linalg.norm(delta)
                ideal_norm = np.linalg.norm(ideal)
                if delta_norm < 1e-10 or ideal_norm < 1e-10:
                    t['Pointer'] = 0.0
                else:
                    t['Pointer'] = np.clip(np.dot(delta, ideal) / (delta_norm * ideal_norm), -1.0, 1.0)
            else:
                t['Pointer'] = np.nan

    # 4. Report
    print("\\n==========================================================================")
    print(" RESULTS: MINKOWSKI METRIC - QUANTUM CONVERGENCE")
    print("==========================================================================")
    for sub, trials in all_subject_trials.items():
        print(f"\\n[{sub}]")
        phases = ['Exploration', 'Pre-Grokking', 'Grokking', 'Expert']
        for p in phases:
            pts = [t for t in trials if t['Phase'] == p]
            if not pts: continue
            dists = [t['QPC_Dist'] for t in pts]
            acc = np.mean([t['Correct'] for t in pts]) * 100
            print(f"  {p:<12} (n={len(pts):<3} | Acc: {acc:3.0f}%): Mean Dist = {np.mean(dists):.4f}  (std: {np.std(dists):.4f})")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze()
