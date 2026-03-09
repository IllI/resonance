"""
test_orch_or_superradiance.py
=============================
Tests the Hybrid Orch-OR / Superradiance Entanglement Model.

Hypothesis:
  For a subject to correctly categorize a Badoon, two quantum events must succeed:
  1. QED Superradiant Burst: The microtubule Tryptophan network must fire a 
     geometrically coherent electromagnetic ping (measurable via macroscopic proxies).
  2. Orch-OR Gravitational Alignment: That ping must be aimed at the correct 
     gravitational anchor (the QPC coordinate) closely enough to trigger 
     Objective Reduction before decoherence.

Prediction:
  A mathematical phase boundary in E_pot will separate C=1 from C=0 trials.
  Incorrect trials will split into two failure modes:
    - Misfire: High superradiant power, wrong direction (large theta)
    - Fizzle: Right direction (small theta), insufficient superradiant power

Response Time Integration:
  RT correlates with EM network latency. A fast, confident RT indicates the 
  Ghost Basin was already near-coherent with the QPC before the stimulus arrived.
  A slow RT indicates the biological antenna needed time to search/align.
  We normalize RT into a Coherence Latency factor that modulates E_pot.

Reuses:
  - MinkowskiHyperbolicMap (from analyze_moving_qpc.py)
  - S3 streaming, Hurst, PSD Beta calculations
  - Per-trial extraction with active mask
  - Dynamic phase tagging
  - Moving QPC calculation
"""

import os
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
#  MINKOWSKI METRIC MAPPER (from analyze_moving_qpc.py)
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
        Ru = np.sqrt(max(1e-10, -self.minkowski_dot(u, u)))
        Rv = np.sqrt(max(1e-10, -self.minkowski_dot(v, v)))
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

def calculate_minkowski_mean(vectors, minkowski):
    if not vectors: return None
    mean_vec = np.mean(vectors, axis=0)
    mean_R = np.mean([np.sqrt(max(1e-10, -minkowski.minkowski_dot(v, v))) for v in vectors])
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    return np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])


# ====================================================================== #
#  EXTENDED TRIAL EXTRACTION (adds RT, N_active, per-trial kurtosis)
# ====================================================================== #
def extract_trials_full(data_z, active_mask, events, tr_sec, run_hurst, run_psd_beta, sub_id):
    """
    Extracts ALL trials (correct and incorrect) with extended quantum proxies.
    Returns list of dicts including RT, N_active, and all metrics needed for E_pot.
    """
    n_trs = data_z.shape[-1]
    n_active = int(np.sum(active_mask))
    trials = []

    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']):
            continue

        correct = int(row['correct'])
        rt = row.get('response_time', np.nan)
        if pd.isna(rt):
            rt = np.nan

        tr_onset = int(row['onset'] / tr_sec)
        start_tr = min(tr_onset + 2, n_trs - 1)
        end_tr = min(tr_onset + 5, n_trs)
        if start_tr >= end_tr:
            continue

        block = data_z[active_mask, start_tr:end_tr]
        if block.size == 0:
            continue

        trial_energy = np.mean(np.abs(block)) * 10.0
        peak_tr_rel = np.argmax(np.max(block, axis=0))
        voxels_at_peak = block[:, peak_tr_rel]
        trial_kurtosis = kurtosis(voxels_at_peak)

        trials.append({
            'Subject': sub_id,
            'Onset': row['onset'],
            'Correct': correct,
            'RT': rt,
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
            'Hurst': run_hurst,
            'PSDBeta': run_psd_beta,
            'N_Active': n_active,
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


# ====================================================================== #
#  SUPERRADIANCE PROXY CALCULATIONS
# ====================================================================== #
def calculate_superradiant_power(n_active, trial_kurtosis, hurst_h):
    """
    Macroscopic proxy for the collective Superradiant decay rate.
    From Kurian et al.: Gamma_super ∝ N * gamma
    
    N_active:  Dipole mass proxy (number of coherently active voxels)
    Phi:       Burst coherence = 1 / (1 + |kurtosis|). Flat spatial distribution = 
               high synchronization = superradiant. Peaked = incoherent single-site.
    S:         Temporal silence = max(0, 0.5 - H). H < 0.5 = anti-persistent = Wuji silence.
               H > 0.5 = fractal noise = searching = no superradiant burst occurred.
    """
    phi = 1.0 / (1.0 + abs(trial_kurtosis))
    s = max(0.0, 0.5 - hurst_h) if not np.isnan(hurst_h) else 0.0
    
    # Normalize N_active to a 0-1 scale (relative to dataset max ~tens of thousands)
    n_norm = n_active / 10000.0
    
    return n_norm * phi * s


def calculate_coherence_latency(rt, rt_median):
    """
    Converts Response Time into a Coherence Latency factor.
    
    Fast RT (below median): The Ghost Basin was already near-coherent with the QPC.
      The superradiant burst connected immediately. Factor > 1.0 (amplifies E_pot).
    Slow RT (above median): The biological antenna needed extra search time.
      Factor < 1.0 (attenuates E_pot).
    
    Uses a sigmoid-like normalization centered on the median RT.
    """
    if np.isnan(rt) or rt <= 0 or rt_median <= 0:
        return 1.0
    
    # Ratio: RT / median. Values < 1 = fast, > 1 = slow.
    ratio = rt / rt_median
    # Invert and clip: fast responses amplify, slow responses attenuate.
    latency_factor = 1.0 / max(0.1, ratio)
    return min(latency_factor, 3.0)  # cap at 3x amplification


def calculate_entanglement_potential(superradiant_power, theta, coherence_latency):
    """
    E_pot = (Superradiant Power) * (1 / cosh(theta)) * Coherence Latency
    
    theta: Minkowski angular displacement from Ghost Basin to QPC.
           Small theta = aimed correctly. Large theta = misfire.
    1/cosh(theta): Decays exponentially as theta grows. Perfect alignment = 1.0.
    """
    geometric_alignment = 1.0 / np.cosh(theta)
    return superradiant_power * geometric_alignment * coherence_latency


# ====================================================================== #
#  MAIN PIPELINE
# ====================================================================== #
def test_hybrid_model():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    minkowski = MinkowskiHyperbolicMap()
    tr_sec = 2.0

    print("==========================================================================")
    print(" HYBRID ORCH-OR / SUPERRADIANCE ENTANGLEMENT TEST")
    print("==========================================================================")

    # ---- Phase 1: Extract all trials with full metrics ----
    print("\n[1] Extracting all trials with extended quantum proxies...")
    global_timeline = []
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
            B = calculate_psd_beta(run_ts, 1.0 / tr_sec)

            trials = extract_trials_full(data_z, active_mask, events, tr_sec, H, B, sub)
            for t in trials:
                t['Run'] = f"{run_type}{run_num}"
                t['GB_Vector'] = minkowski.project_ghost_basin(
                    t['Hurst'], t['PSDBeta'], t['Kurtosis'], t['Energy']
                )
            sub_trials.extend(trials)
            if os.path.exists(nii_path):
                os.remove(nii_path)

        sub_trials = tag_phases(sub_trials, 5)
        global_timeline.extend(sub_trials)
        print(f"    {sub}: {len(sub_trials)} trials extracted")

    # ---- Phase 2: Build Moving QPC from Expert trials ----
    print("\n[2] Building Dynamic QPC from Expert observations...")
    past_expert_vectors = []
    for t in global_timeline:
        if len(past_expert_vectors) >= 5:
            t['QPC_Now'] = calculate_minkowski_mean(past_expert_vectors, minkowski)
        else:
            t['QPC_Now'] = None
        if t.get('Phase') == 'Expert' and t['Correct'] == 1:
            past_expert_vectors.append(t['GB_Vector'])

    # Final QPC fallback for early trials that had no moving QPC yet
    final_qpc = calculate_minkowski_mean(past_expert_vectors, minkowski)
    for t in global_timeline:
        if t['QPC_Now'] is None:
            t['QPC_Now'] = final_qpc

    # ---- Phase 3: Calculate Entanglement Potential for every trial ----
    print("\n[3] Calculating Entanglement Potential (E_pot) per trial...")

    # Global median RT for coherence latency normalization
    all_rts = [t['RT'] for t in global_timeline if not np.isnan(t['RT']) and t['RT'] > 0]
    rt_median = np.median(all_rts) if all_rts else 1.0

    for t in global_timeline:
        # Minkowski angle to QPC
        theta = minkowski.lorentz_distance(t['GB_Vector'], t['QPC_Now'])
        t['Theta'] = theta

        # Superradiant power proxy
        sp = calculate_superradiant_power(t['N_Active'], t['Kurtosis'], t['Hurst'])
        t['SuperradiantPower'] = sp

        # Coherence latency from RT
        cl = calculate_coherence_latency(t['RT'], rt_median)
        t['CoherenceLatency'] = cl

        # Entanglement Potential
        e_pot = calculate_entanglement_potential(sp, theta, cl)
        t['E_pot'] = e_pot

    # ---- Phase 4: Train threshold model ----
    print("\n[4] Training convergence threshold...")
    correct_epots = [t['E_pot'] for t in global_timeline if t['Correct'] == 1]
    incorrect_epots = [t['E_pot'] for t in global_timeline if t['Correct'] == 0]

    # Optimal threshold: maximize separation between correct and incorrect distributions
    all_epots = sorted(set([t['E_pot'] for t in global_timeline]))
    best_threshold = 0.0
    best_accuracy = 0.0
    best_tp = best_tn = best_fp = best_fn = 0

    for candidate in np.linspace(min(all_epots), max(all_epots), 1000):
        tp = sum(1 for e in correct_epots if e >= candidate)
        tn = sum(1 for e in incorrect_epots if e < candidate)
        fp = sum(1 for e in incorrect_epots if e >= candidate)
        fn = sum(1 for e in correct_epots if e < candidate)
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = candidate
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

    # ---- Phase 5: Report ----
    print("\n==========================================================================")
    print(" RESULTS: ENTANGLEMENT POTENTIAL PHASE BOUNDARY")
    print("==========================================================================")
    print(f"  Global Median RT:           {rt_median:.3f}s")
    print(f"  Total Trials:               {len(global_timeline)}")
    print(f"  Total Correct:              {len(correct_epots)}")
    print(f"  Total Incorrect:            {len(incorrect_epots)}")

    print(f"\n  --- E_pot Distributions ---")
    print(f"  Correct Trials:   mean={np.mean(correct_epots):.6f}  std={np.std(correct_epots):.6f}")
    print(f"  Incorrect Trials: mean={np.mean(incorrect_epots):.6f}  std={np.std(incorrect_epots):.6f}")

    print(f"\n  --- Optimal Threshold Model ---")
    print(f"  Threshold E_pot:            {best_threshold:.6f}")
    print(f"  Classification Accuracy:    {best_accuracy:.1%}")
    print(f"  True Positives  (C=1, E>=T): {best_tp}")
    print(f"  True Negatives  (C=0, E< T): {best_tn}")
    print(f"  False Positives (C=0, E>=T): {best_fp}")
    print(f"  False Negatives (C=1, E< T): {best_fn}")

    # ---- Failure Mode Analysis ----
    print(f"\n  --- Failure Mode Analysis (Incorrect Trials) ---")
    incorrect_trials = [t for t in global_timeline if t['Correct'] == 0]
    if incorrect_trials:
        median_theta = np.median([t['Theta'] for t in incorrect_trials])
        median_sp = np.median([t['SuperradiantPower'] for t in incorrect_trials])

        misfires = [t for t in incorrect_trials if t['Theta'] > median_theta and t['SuperradiantPower'] > median_sp]
        fizzles = [t for t in incorrect_trials if t['Theta'] <= median_theta and t['SuperradiantPower'] <= median_sp]
        mixed = [t for t in incorrect_trials if t not in misfires and t not in fizzles]

        print(f"  Misfires (High Power, Wrong Aim):   {len(misfires)}")
        print(f"  Fizzles  (Right Aim, Weak Power):   {len(fizzles)}")
        print(f"  Mixed    (Partial Failure):          {len(mixed)}")

        if misfires:
            print(f"    Misfire avg Theta:     {np.mean([t['Theta'] for t in misfires]):.4f}")
            print(f"    Misfire avg SP:        {np.mean([t['SuperradiantPower'] for t in misfires]):.6f}")
        if fizzles:
            print(f"    Fizzle avg Theta:      {np.mean([t['Theta'] for t in fizzles]):.4f}")
            print(f"    Fizzle avg SP:         {np.mean([t['SuperradiantPower'] for t in fizzles]):.6f}")

    # ---- Per-Phase Breakdown ----
    print(f"\n  --- E_pot by Learning Phase ---")
    for sub in subjects:
        sub_data = [t for t in global_timeline if t['Subject'] == sub]
        print(f"\n  [{sub}]")
        for phase in ['Exploration', 'Pre-Grokking', 'Grokking', 'Expert']:
            pts = [t for t in sub_data if t['Phase'] == phase]
            if not pts: continue
            epots = [t['E_pot'] for t in pts]
            thetas = [t['Theta'] for t in pts]
            sps = [t['SuperradiantPower'] for t in pts]
            acc = np.mean([t['Correct'] for t in pts]) * 100
            print(f"    {phase:<14} (n={len(pts):<3} Acc:{acc:5.1f}%) "
                  f"E_pot={np.mean(epots):.6f}  Theta={np.mean(thetas):.4f}  SP={np.mean(sps):.6f}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    test_hybrid_model()
