"""
analyze_phase_transition_quantum.py
=====================================
TEST: Is the grokking phase transition a CONSTANT ENERGY QUANTUM?

If grokking is a phase transition (1/f criticality crossing), 
the energy released should be approximately constant across ALL 
subjects, regardless of:
- When they grokked
- How long it took them
- Their individual metabolic baseline

This script:
1. Measures the EXCESS energy at grokking (above each subject's baseline)
2. Tests if that excess is constant across subjects (low CV)
3. Characterizes the spectral beta SHIFT at grokking per ROI
4. Identifies non-grokking HF spikes and compares their energy distribution
   to the grokking spike — are they the same or different populations?
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, pearsonr, spearmanr, ttest_1samp, mannwhitneyu
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
import boto3
from botocore import UNSIGNED
from botocore.config import Config


def stream_data(sub_id, base_dir, run_type, run_num):
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


def spectral_slope(signal, tr_sec=2.0):
    n = len(signal)
    if n < 4:
        return np.nan
    fft_vals = fft(signal)
    freqs = fftfreq(n, d=tr_sec)
    power = np.abs(fft_vals) ** 2
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    if len(pos_freqs) < 2 or np.any(pos_power <= 0):
        return np.nan
    log_f = np.log10(pos_freqs)
    log_p = np.log10(pos_power + 1e-20)
    slope, _, _, _, _ = linregress(log_f, log_p)
    return -slope


def define_key_rois(shape):
    nx, ny, nz = shape[:3]
    rois = {}
    rois['PFC'] = np.zeros(shape[:3], dtype=bool)
    rois['PFC'][:, int(ny*0.75):, int(nz*0.6):] = True
    rois['Parahippocampus'] = np.zeros(shape[:3], dtype=bool)
    rois['Parahippocampus'][int(nx*0.25):int(nx*0.75),
                            int(ny*0.35):int(ny*0.65),
                            :int(nz*0.3)] = True
    rois['Thalamus'] = np.zeros(shape[:3], dtype=bool)
    rois['Thalamus'][int(nx*0.35):int(nx*0.65),
                     int(ny*0.4):int(ny*0.6),
                     int(nz*0.35):int(nz*0.55)] = True
    return rois


def process_subject(sub_id, base_dir, run_sequence, tr_sec=2.0):
    all_trials = []
    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, base_dir, run_type, run_num)
        if not nii_path:
            continue
        data_raw = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        n_trs = data_raw.shape[-1]
        shape = data_raw.shape
        brain_mask = np.var(data_raw, axis=-1) > 10
        rois = define_key_rois(shape)

        for _, row in events.iterrows():
            if 'correct' not in row or pd.isna(row['correct']):
                continue
            correct = int(row['correct'])
            tr_onset = int(row['onset'] / tr_sec)
            hrf_start = min(tr_onset + 2, n_trs - 1)
            hrf_end = min(tr_onset + 5, n_trs)
            ext_start = max(0, tr_onset - 2)
            ext_end = min(n_trs, tr_onset + 8)
            n_ext = ext_end - ext_start
            if hrf_start >= hrf_end or n_ext < 4:
                continue

            trial = {'Correct': correct}

            # Whole brain HF energy
            active_mask = brain_mask & (np.var(data_raw, axis=-1) > np.percentile(
                np.var(data_raw, axis=-1)[brain_mask], 90))
            ts_z = np.mean(data_z[active_mask, ext_start:ext_end], axis=0)
            fft_vals = fft(ts_z)
            freqs = fftfreq(n_ext, d=tr_sec)
            power = np.abs(fft_vals) ** 2
            pos_mask_f = freqs > 0
            pos_freqs = freqs[pos_mask_f]
            pos_power = power[pos_mask_f]
            total_p = np.sum(pos_power)
            hf_mask = pos_freqs > 0.15
            hf_power = np.sum(pos_power[hf_mask]) / (total_p + 1e-10) if np.any(hf_mask) else 0
            trial['HF_Power'] = hf_power
            trial['Total_Power'] = total_p

            # Metabolic energy (raw BOLD amplitude)
            hrf_block = data_z[active_mask, hrf_start:hrf_end]
            trial['E_Metabolic'] = np.mean(np.abs(hrf_block))

            # Per-ROI spectral beta
            for roi_name, roi_mask in rois.items():
                active = roi_mask & brain_mask
                if np.sum(active) < 10:
                    trial[f'Beta_{roi_name}'] = np.nan
                    continue
                roi_ts = np.mean(data_z[active, ext_start:ext_end], axis=0)
                trial[f'Beta_{roi_name}'] = spectral_slope(roi_ts, tr_sec)

            all_trials.append(trial)

        if os.path.exists(nii_path):
            os.remove(nii_path)

    # Tag phases
    n = len(all_trials)
    correctness = [t['Correct'] for t in all_trials]
    grok_idx = n
    streak_start = n
    for i in range(n - 1, -1, -1):
        if correctness[i] == 1:
            streak_start = i
        else:
            break
    if n - streak_start >= 5:
        grok_idx = streak_start

    for i, t in enumerate(all_trials):
        t['Trial_Idx'] = i
        t['Grok_Idx'] = grok_idx
        t['Trials_To_Grok'] = i - grok_idx
        if i < grok_idx:
            t['Phase'] = 'Exploration'
        elif i == grok_idx:
            t['Phase'] = 'Grokking'
        else:
            t['Phase'] = 'Expert'

    return all_trials, grok_idx


def run_analysis():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = [
        "sub-201", "sub-206", "sub-212", "sub-216", "sub-217",
        "sub-220", "sub-226", "sub-229", "sub-237", "sub-240"
    ]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0
    roi_names = ['PFC', 'Parahippocampus', 'Thalamus']

    print("=" * 80, flush=True)
    print(" PHASE TRANSITION ENERGY QUANTUM TEST", flush=True)
    print(" Is the grokking energy a constant quantum across all subjects?", flush=True)
    print("=" * 80, flush=True)

    # Collect per-subject grokking data
    grok_hf_excess = []     # HF power at grokking minus baseline
    grok_e_metabolic = []   # Metabolic energy at grokking
    grok_e_total = []       # Total energy at grokking
    grok_beta_shifts = {r: [] for r in roi_names}
    non_grok_spikes = []    # All HF values from exploration for comparison
    grok_hf_raw = []        # Raw HF power at grokking
    subject_labels = []

    for sub in subjects:
        print(f"\n  Processing {sub}...", flush=True)
        trials, grok_idx = process_subject(sub, base_dir, run_sequence, tr_sec)
        if not trials or grok_idx >= len(trials):
            print(f"    No grokking. Skipping.", flush=True)
            continue

        explorers = [t for t in trials if t['Phase'] == 'Exploration']
        grok_trial = trials[grok_idx]
        grok_window = [t for t in trials if abs(t['Trials_To_Grok']) <= 1]

        if len(explorers) < 5:
            continue

        # Baseline HF power during exploration
        explore_hf_mean = np.mean([t['HF_Power'] for t in explorers])
        explore_hf_std = np.std([t['HF_Power'] for t in explorers])
        explore_met_mean = np.mean([t['E_Metabolic'] for t in explorers])

        # Grokking window HF power (mean of trials -1, 0, +1)
        grok_hf = np.mean([t['HF_Power'] for t in grok_window])
        grok_met = np.mean([t['E_Metabolic'] for t in grok_window])

        # EXCESS = grokking HF - exploration baseline (normalized by std)
        hf_excess_z = (grok_hf - explore_hf_mean) / (explore_hf_std + 1e-10)

        # Raw excess (not z-scored) - the actual energy quantum
        hf_excess_raw = grok_hf - explore_hf_mean

        # Total energy = metabolic + HF
        grok_total = grok_met + grok_hf

        grok_hf_excess.append(hf_excess_z)
        grok_hf_raw.append(hf_excess_raw)
        grok_e_metabolic.append(grok_met)
        grok_e_total.append(grok_total)
        subject_labels.append(sub)

        # Collect spectral beta shifts
        for roi in roi_names:
            explore_beta = np.nanmean([t[f'Beta_{roi}'] for t in explorers])
            grok_beta = np.nanmean([t[f'Beta_{roi}'] for t in grok_window])
            grok_beta_shifts[roi].append(grok_beta - explore_beta)

        # Collect ALL exploration trial HF values for anomaly comparison
        for t in explorers:
            non_grok_spikes.append(t['HF_Power'])

    # ================================================================ #
    #  RESULTS
    # ================================================================ #
    print("\n" + "=" * 80, flush=True)
    print(" [1] IS THE GROKKING ENERGY A CONSTANT QUANTUM?", flush=True)
    print("=" * 80, flush=True)

    arr_excess_z = np.array(grok_hf_excess)
    arr_excess_raw = np.array(grok_hf_raw)
    arr_met = np.array(grok_e_metabolic)
    arr_total = np.array(grok_e_total)

    print(f"\n  Per-Subject Grokking Phase Transition Energy:")
    print(f"  {'Subject':<10} {'HF Excess (z)':>14} {'HF Excess (raw)':>16} {'E_Metabolic':>12} {'E_Total':>10}")
    print(f"  {'-'*65}")
    for i, sub in enumerate(subject_labels):
        print(f"  {sub:<10} {arr_excess_z[i]:>14.3f} {arr_excess_raw[i]:>16.6f} {arr_met[i]:>12.4f} {arr_total[i]:>10.4f}")

    print(f"\n  HF Excess (z-scored) across grokkers:")
    print(f"    Mean:   {np.mean(arr_excess_z):.4f}")
    print(f"    Std:    {np.std(arr_excess_z):.4f}")
    print(f"    CV:     {np.std(arr_excess_z) / (abs(np.mean(arr_excess_z)) + 1e-10):.4f}")

    print(f"\n  HF Excess (raw) across grokkers:")
    print(f"    Mean:   {np.mean(arr_excess_raw):.6f}")
    print(f"    Std:    {np.std(arr_excess_raw):.6f}")
    print(f"    CV:     {np.std(arr_excess_raw) / (abs(np.mean(arr_excess_raw)) + 1e-10):.4f}")

    print(f"\n  E_Total at grokking:")
    print(f"    Mean:   {np.mean(arr_total):.4f}")
    print(f"    Std:    {np.std(arr_total):.4f}")
    print(f"    CV:     {np.std(arr_total) / np.mean(arr_total):.4f}")

    print(f"\n  E_Metabolic at grokking:")
    print(f"    Mean:   {np.mean(arr_met):.4f}")
    print(f"    Std:    {np.std(arr_met):.4f}")
    print(f"    CV:     {np.std(arr_met) / np.mean(arr_met):.4f}")

    # Is E_total more constant than E_metabolic?
    cv_total = np.std(arr_total) / np.mean(arr_total)
    cv_met = np.std(arr_met) / np.mean(arr_met)
    if cv_total < cv_met:
        print(f"\n    >>> E_TOTAL (CV={cv_total:.4f}) IS MORE CONSTANT THAN E_METABOLIC (CV={cv_met:.4f})")
        print(f"    The phase transition consumes a near-constant total energy quantum.")
    else:
        print(f"\n    E_total is NOT more constant than E_metabolic alone.")

    # ================================================================ #
    print("\n" + "=" * 80, flush=True)
    print(" [2] SPECTRAL BETA SHIFT AT GROKKING: Which ROI hits 1/f criticality?", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  {'ROI':<20} {'Mean Shift':>11} {'Std':>8} {'Pink Count':>11} {'Brown Count':>12}")
    print(f"  {'-'*65}")
    for roi in roi_names:
        shifts = np.array(grok_beta_shifts[roi])
        pink_count = np.sum(shifts < -0.3)
        brown_count = np.sum(shifts > 0.3)
        print(f"  {roi:<20} {np.mean(shifts):>11.3f} {np.std(shifts):>8.3f} {pink_count:>11} {brown_count:>12}")

    # Which ROI most consistently shifts pink?
    thalamus_shifts = np.array(grok_beta_shifts['Thalamus'])
    t_stat, t_pval = ttest_1samp(thalamus_shifts, 0)
    print(f"\n  Thalamus beta shift: t={t_stat:.3f}, p={t_pval:.4f}")
    if np.mean(thalamus_shifts) < 0:
        print(f"    Direction: TOWARD PINK (1/f criticality)")
    else:
        print(f"    Direction: TOWARD BROWN (1/f^2 metabolic)")

    # ================================================================ #
    print("\n" + "=" * 80, flush=True)
    print(" [3] GROKKING SPIKES vs NON-GROKKING SPIKES: Same population?", flush=True)
    print("     If grokking is a specific phase transition, its HF spike should be", flush=True)
    print("     distinguishable from random exploration-phase HF fluctuations.", flush=True)
    print("=" * 80, flush=True)

    grok_hf_values = [arr_excess_raw[i] + np.mean(non_grok_spikes) for i in range(len(arr_excess_raw))]

    # Mann-Whitney U test: are grokking HF values from a different distribution?
    if len(non_grok_spikes) > 10 and len(grok_hf_values) > 3:
        u_stat, u_pval = mannwhitneyu(grok_hf_values, non_grok_spikes, alternative='greater')
        print(f"\n  Grokking HF spikes (n={len(grok_hf_values)}) vs Exploration HF (n={len(non_grok_spikes)}):")
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {u_pval:.6f}")
        if u_pval < 0.05:
            print(f"    >>> GROKKING SPIKES ARE FROM A DIFFERENT (HIGHER) DISTRIBUTION")
            print(f"    The phase transition energy is NOT random noise.")
        else:
            print(f"    Grokking spikes are NOT significantly different from exploration noise.")

    # Percentile of grokking spikes within exploration distribution
    explore_hf = np.array(non_grok_spikes)
    print(f"\n  Exploration HF distribution:")
    print(f"    Mean:     {np.mean(explore_hf):.6f}")
    print(f"    Std:      {np.std(explore_hf):.6f}")
    print(f"    95th pct: {np.percentile(explore_hf, 95):.6f}")
    print(f"    99th pct: {np.percentile(explore_hf, 99):.6f}")

    print(f"\n  Grokking HF values and their percentile in exploration distribution:")
    for i, sub in enumerate(subject_labels):
        pctile = np.mean(explore_hf < grok_hf_values[i]) * 100
        flag = " *** OUTLIER" if pctile > 95 else ""
        print(f"    {sub}: {grok_hf_values[i]:.6f} (percentile: {pctile:.1f}%){flag}")

    # ================================================================ #
    # Count anomalous non-grokking spikes that MATCH grokking energy
    print(f"\n  Anomalous non-grokking spikes that match grokking energy range:")
    if len(grok_hf_values) > 0:
        grok_min = min(grok_hf_values)
        grok_max = max(grok_hf_values)
        anomalous = [x for x in non_grok_spikes if grok_min <= x <= grok_max]
        print(f"    Grokking HF range: [{grok_min:.6f}, {grok_max:.6f}]")
        print(f"    Non-grokking trials in that range: {len(anomalous)} / {len(non_grok_spikes)}")
        print(f"    Percentage: {100*len(anomalous)/len(non_grok_spikes):.1f}%")
        if len(anomalous) / len(non_grok_spikes) < 0.05:
            print(f"    >>> GROKKING ENERGY IS RARE (<5% of exploration trials match)")
            print(f"    This supports a DISTINCT phase transition event.")
        else:
            print(f"    Many exploration trials have similar energy.")
            print(f"    Could be: failed alignment attempts, or HF alone isn't distinctive.")

    print(flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_analysis()
