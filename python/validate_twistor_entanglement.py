"""
validate_twistor_entanglement.py
================================
Validates the Twistor Coherence separation between true Entanglement (Expert Phase)
and Unentangled states (Exploration Phase), ignoring behavioral 'lucky guesses'.

The script applies the exact Twistor constraints without modification to see what the
data naturally dictates. We determine if a mathematically objective phase boundary 
exists between the Expert and Exploration domains.

1. Calculates Twistor Coherence Score (TCS) strictly via predefined formulas.
2. Performs a statistical t-test between Expert trials and Exploration trials.
3. Finds the natural decision boundary.
4. If the data does not support the hypothesis, we abort and report the raw findings.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, ttest_ind
import boto3
from botocore import UNSIGNED
from botocore.config import Config

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
        Ru = np.sqrt(max(1e-10, -self.minkowski_dot(u, u)))
        Rv = np.sqrt(max(1e-10, -self.minkowski_dot(v, v)))
        dot = self.minkowski_dot(u, v)
        val = max(1.0, -dot / (Ru * Rv))
        return np.arccosh(val)

def stream_data(sub_id, base_dir, run_type, run_num):
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

def calculate_minkowski_mean(vectors, minkowski):
    if not vectors: return None
    mean_vec = np.mean(vectors, axis=0)
    mean_R = np.mean([np.sqrt(max(1e-10, -minkowski.minkowski_dot(v, v))) for v in vectors])
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    return np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])

def extract_trials(data_z, active_mask, events, tr_sec, sub_id):
    n_trs = data_z.shape[-1]
    n_active = int(np.sum(active_mask))
    trials = []

    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']): continue
        
        correct = int(row['correct'])
        rt = row.get('response_time', np.nan)
        if pd.isna(rt) or rt <= 0: rt = np.nan

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
        
        mean_amp = np.mean(np.abs(block))
        peak_amp = np.mean(np.abs(voxels_at_peak))
        s_trial = peak_amp / (mean_amp + 1e-5)

        trials.append({
            'Subject': sub_id,
            'Correct': correct,
            'RT': rt,
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
            'S_trial': s_trial,
            'N_Active': n_active,
            'GB_Vector': MinkowskiHyperbolicMap().project_ghost_basin(0.5, 1.0, trial_kurtosis, trial_energy)
        })
    return trials

def tag_phases(trial_list, streak_threshold=5):
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

def calc_twistor_coherence(theta_minkowski, rt, rt_median, kurtosis, n_active, s_trial):
    r_riemann = 1.0 + abs(kurtosis)
    delta_twistor = theta_minkowski * r_riemann
    peephole_alignment = 1.0 / np.cosh(delta_twistor)
    
    if np.isnan(rt) or rt_median <= 0:
        theta_dyn = 0
    else:
        theta_dyn = (np.pi / 2.0) * min(1.0, abs(1.0 - (rt / rt_median)))
    phase_factor = max(0.0, np.cos(theta_dyn))
    
    n_norm = n_active / 10000.0
    power = n_norm * s_trial
    
    tcs = power * peephole_alignment * phase_factor
    return tcs, peephole_alignment, phase_factor

def validate_model():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    minkowski = MinkowskiHyperbolicMap()
    tr_sec = 2.0

    print("==========================================================================")
    print(" VALIDATING TWISTOR ENTANGLEMENT (EXPERT VS EXPLORATION)")
    print("==========================================================================")

    print("\n[1] Extracting trials and computing Twistor metrics...")
    global_timeline = []
    for sub in subjects:
        sub_trials = []
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_data(sub, base_dir, run_type, run_num)
            if not nii_path: continue
            data = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
            data_z = np.nan_to_num(zscore(data, axis=-1))
            brain_mask = np.var(data, axis=-1) > 10
            raw_var = np.var(data, axis=-1)
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask
            if np.sum(active_mask) == 0: continue

            trials = extract_trials(data_z, active_mask, events, tr_sec, sub)
            for t in trials: t['Run'] = f"{run_type}{run_num}"
            sub_trials.extend(trials)
            if os.path.exists(nii_path): os.remove(nii_path)

        sub_trials = tag_phases(sub_trials, 5)
        global_timeline.extend(sub_trials)

    past_expert_vectors = []
    for t in global_timeline:
        if len(past_expert_vectors) >= 5:
            t['QPC_Now'] = calculate_minkowski_mean(past_expert_vectors, minkowski)
        else:
            t['QPC_Now'] = None
        if t.get('Phase') == 'Expert' and t['Correct'] == 1:
            past_expert_vectors.append(t['GB_Vector'])

    final_qpc = calculate_minkowski_mean(past_expert_vectors, minkowski)
    for t in global_timeline:
        if t['QPC_Now'] is None: t['QPC_Now'] = final_qpc

    all_rts = [t['RT'] for t in global_timeline if not np.isnan(t['RT'])]
    rt_median = np.median(all_rts) if all_rts else 2.5

    for t in global_timeline:
        theta_minkowski = minkowski.lorentz_distance(t['GB_Vector'], t['QPC_Now'])
        tcs, peephole, phase = calc_twistor_coherence(
            theta_minkowski, t['RT'], rt_median, t['Kurtosis'], t['N_Active'], t['S_trial']
        )
        t['Theta_Minkowski'] = theta_minkowski
        t['TCS'] = tcs

    # Data Separation: Expert vs Exploration
    experts = [t['TCS'] for t in global_timeline if t['Phase'] == 'Expert']
    explorers = [t['TCS'] for t in global_timeline if t['Phase'] == 'Exploration']

    print("\n[2] Statistical Significance Test (Independent T-Test)")
    t_stat, p_val = ttest_ind(experts, explorers, equal_var=False)

    print(f"  Expert Trials (n={len(experts)}):      Mean TCS = {np.mean(experts):.4f}, Std = {np.std(experts):.4f}")
    print(f"  Exploration Trials (n={len(explorers)}): Mean TCS = {np.mean(explorers):.4f}, Std = {np.std(explorers):.4f}")
    print(f"  T-Statistic: {t_stat:.4f}  |  P-Value: {p_val:.4e}")

    if p_val < 0.05 and np.mean(experts) > np.mean(explorers):
        print("  -> SIGNIFICANT: The Twistor geometry mathematically distinguishes Entangled vs Unentangled states.")
    else:
        print("  -> ABORT: The data does NOT support a significant difference. Consciousness does not collapse the wave function here.")
        return

    print("\n[3] Training Entanglement Decision Boundary (Expert vs Exploration)")
    best_threshold, best_accuracy = 0.0, 0.0
    best_tp = best_tn = best_fp = best_fn = 0

    all_tcs = sorted(set(experts + explorers))
    total_valid_trials = len(experts) + len(explorers)

    for candidate in np.linspace(min(all_tcs), max(all_tcs), 1000):
        # We define TP as Expert >= threshold, TN as Exploration < threshold
        tp = sum(1 for c in experts if c >= candidate)
        tn = sum(1 for c in explorers if c < candidate)
        fp = sum(1 for c in explorers if c >= candidate)
        fn = sum(1 for c in experts if c < candidate)
        
        acc = (tp + tn) / total_valid_trials
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = candidate
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

    print(f"  Threshold TCS:              {best_threshold:.4f}")
    print(f"  Classification Accuracy:    {best_accuracy:.1%}")
    print(f"  True Positives  (Expert >= th): {best_tp}")
    print(f"  True Negatives  (Exploration<th): {best_tn}")
    print(f"  False Positives (Exploration>=th): {best_fp} (Lucky but mathematically anomalous)")
    print(f"  False Negatives (Expert < th): {best_fn} (Expert but weakly entangled)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    validate_model()
