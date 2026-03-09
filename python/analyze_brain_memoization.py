"""
analyze_brain_memoization.py
============================
Empirical analysis of the brain's energy transition from Grokking to Working Memory.

This script does NOT touch D-LinOSS. It measures:
1. Per-trial metabolic energy (BOLD amplitude) across all learning phases
2. Energy efficiency ratio: Exploration vs Expert
3. Trial-by-trial energy trajectory around the grokking moment
4. The "memoization fingerprint" - what the stable expert configuration looks like
5. Per-subject analysis to avoid Simpson's paradox (aggregating hides individual patterns)

Goal: Quantify how the brain memoizes a novel concept, so we can later
compare this to model compute cycles for the same task.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, sem
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


def extract_trial_metrics(data_z, active_mask, events, tr_sec):
    """Extract raw metabolic metrics per trial. No theoretical transforms."""
    n_trs = data_z.shape[-1]
    n_active = int(np.sum(active_mask))
    trials = []

    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']):
            continue
        correct = int(row['correct'])
        rt = row.get('response_time', np.nan)
        if pd.isna(rt) or rt <= 0:
            rt = np.nan

        tr_onset = int(row['onset'] / tr_sec)
        start_tr = min(tr_onset + 2, n_trs - 1)
        end_tr = min(tr_onset + 5, n_trs)
        if start_tr >= end_tr:
            continue

        block = data_z[active_mask, start_tr:end_tr]
        if block.size == 0:
            continue

        # Raw metabolic energy proxy (mean absolute BOLD amplitude in active voxels)
        energy = np.mean(np.abs(block))

        # Spatial spread: how many voxels are significantly activated (above 1 SD)
        n_significant = int(np.sum(np.any(np.abs(block) > 1.0, axis=1)))
        spatial_efficiency = n_significant / max(1, n_active)

        # Temporal sharpness: how concentrated is the activation in time?
        tr_energies = np.mean(np.abs(block), axis=0)  # energy per TR
        peak_tr_energy = np.max(tr_energies)
        mean_tr_energy = np.mean(tr_energies)
        temporal_sharpness = peak_tr_energy / (mean_tr_energy + 1e-8)

        # Spatial coherence: correlation between voxels at peak TR
        peak_tr_idx = np.argmax(tr_energies)
        vox_at_peak = block[:, peak_tr_idx]
        spatial_kurtosis = kurtosis(vox_at_peak)

        # Variance across the block (how noisy is the activation?)
        block_variance = np.var(block)

        trials.append({
            'Correct': correct,
            'RT': rt,
            'Energy': energy,
            'Spatial_Efficiency': spatial_efficiency,
            'Temporal_Sharpness': temporal_sharpness,
            'Spatial_Kurtosis': spatial_kurtosis,
            'Block_Variance': block_variance,
            'N_Active': n_active,
            'N_Significant': n_significant,
        })
    return trials


def tag_phases(trial_list, streak_threshold=5):
    """Tag learning phases based on behavioral correctness streaks."""
    n = len(trial_list)
    if n == 0:
        return trial_list
    correctness = [t['Correct'] for t in trial_list]

    # Find the grokking point: start of the final sustained correct streak
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
        trial['Trial_Idx'] = i
        trial['Trials_From_Grok'] = i - grok_idx  # negative = before, 0 = grokking, positive = after

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


def analyze_memoization():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0

    print("=" * 70)
    print(" BRAIN MEMOIZATION ANALYSIS: ENERGY COST OF LEARNING VS REMEMBERING")
    print("=" * 70)

    all_subject_data = {}

    for sub in subjects:
        sub_trials = []
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_data(sub, base_dir, run_type, run_num)
            if not nii_path:
                continue
            data = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
            data_z = np.nan_to_num(zscore(data, axis=-1))
            brain_mask = np.var(data, axis=-1) > 10
            raw_var = np.var(data, axis=-1)
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask
            if np.sum(active_mask) == 0:
                continue

            trials = extract_trial_metrics(data_z, active_mask, events, tr_sec)
            for t in trials:
                t['Run'] = f"{run_type}{run_num}"
            sub_trials.extend(trials)
            if os.path.exists(nii_path):
                os.remove(nii_path)

        sub_trials = tag_phases(sub_trials, 5)
        all_subject_data[sub] = sub_trials

    # ================================================================== #
    #  PER-SUBJECT ENERGY ANALYSIS
    # ================================================================== #
    print("\n" + "=" * 70)
    print(" [1] PER-SUBJECT METABOLIC COST: EXPLORATION VS EXPERT")
    print("=" * 70)

    for sub, trials in all_subject_data.items():
        explore = [t for t in trials if t['Phase'] == 'Exploration']
        expert = [t for t in trials if t['Phase'] == 'Expert']
        grok = [t for t in trials if t['Phase'] == 'Grokking']

        if not explore or not expert:
            print(f"\n  {sub}: Insufficient data (explore={len(explore)}, expert={len(expert)})")
            continue

        e_explore = np.mean([t['Energy'] for t in explore])
        e_expert = np.mean([t['Energy'] for t in expert])
        e_ratio = e_explore / (e_expert + 1e-8)

        rt_explore = np.nanmean([t['RT'] for t in explore if not np.isnan(t['RT'])])
        rt_expert = np.nanmean([t['RT'] for t in expert if not np.isnan(t['RT'])])

        var_explore = np.mean([t['Block_Variance'] for t in explore])
        var_expert = np.mean([t['Block_Variance'] for t in expert])

        sharp_explore = np.mean([t['Temporal_Sharpness'] for t in explore])
        sharp_expert = np.mean([t['Temporal_Sharpness'] for t in expert])

        spat_eff_explore = np.mean([t['Spatial_Efficiency'] for t in explore])
        spat_eff_expert = np.mean([t['Spatial_Efficiency'] for t in expert])

        print(f"\n  {sub} (Explore: {len(explore)} trials, Expert: {len(expert)} trials)")
        print(f"  {'Metric':<25} {'Exploration':>12} {'Expert':>12} {'Ratio (E/X)':>12}")
        print(f"  {'-'*61}")
        print(f"  {'BOLD Energy':<25} {e_explore:>12.4f} {e_expert:>12.4f} {e_ratio:>12.2f}x")
        print(f"  {'Response Time (s)':<25} {rt_explore:>12.3f} {rt_expert:>12.3f} {rt_explore/rt_expert:>12.2f}x")
        print(f"  {'Block Variance':<25} {var_explore:>12.4f} {var_expert:>12.4f} {var_explore/var_expert:>12.2f}x")
        print(f"  {'Temporal Sharpness':<25} {sharp_explore:>12.4f} {sharp_expert:>12.4f} {sharp_expert/sharp_explore:>12.2f}x")
        print(f"  {'Spatial Efficiency':<25} {spat_eff_explore:>12.4f} {spat_eff_expert:>12.4f} {spat_eff_expert/spat_eff_explore:>12.2f}x")

    # ================================================================== #
    #  TRIAL-BY-TRIAL ENERGY TRAJECTORY AROUND GROKKING
    # ================================================================== #
    print("\n" + "=" * 70)
    print(" [2] ENERGY TRAJECTORY AROUND GROKKING MOMENT (ALL SUBJECTS)")
    print("=" * 70)
    print(f"  {'Trials_From_Grok':>18} {'N':>5} {'Mean_Energy':>12} {'Mean_RT':>10} {'Mean_Variance':>14} {'Mean_Sharpness':>15}")
    print(f"  {'-'*74}")

    # Collect all trials aligned to grokking moment
    all_aligned = []
    for sub, trials in all_subject_data.items():
        for t in trials:
            if 'Trials_From_Grok' in t:
                all_aligned.append(t)

    # Show window of -10 to +15 trials around grokking
    for offset in range(-10, 16):
        bucket = [t for t in all_aligned if t['Trials_From_Grok'] == offset]
        if not bucket:
            continue
        n = len(bucket)
        me = np.mean([t['Energy'] for t in bucket])
        mrt = np.nanmean([t['RT'] for t in bucket if not np.isnan(t['RT'])])
        mv = np.mean([t['Block_Variance'] for t in bucket])
        ms = np.mean([t['Temporal_Sharpness'] for t in bucket])
        marker = " <-- GROK" if offset == 0 else ""
        print(f"  {offset:>18} {n:>5} {me:>12.4f} {mrt:>10.3f} {mv:>14.4f} {ms:>15.4f}{marker}")

    # ================================================================== #
    #  MEMOIZATION FINGERPRINT: WHAT DOES THE STABLE STATE LOOK LIKE?
    # ================================================================== #
    print("\n" + "=" * 70)
    print(" [3] MEMOIZATION FINGERPRINT (EXPERT PHASE STABILITY)")
    print("=" * 70)

    for sub, trials in all_subject_data.items():
        expert = [t for t in trials if t['Phase'] == 'Expert']
        if len(expert) < 5:
            continue

        energies = [t['Energy'] for t in expert]
        rts = [t['RT'] for t in expert if not np.isnan(t['RT'])]
        variances = [t['Block_Variance'] for t in expert]
        sharpness = [t['Temporal_Sharpness'] for t in expert]
        kurtoses = [t['Spatial_Kurtosis'] for t in expert]

        print(f"\n  {sub} Expert Phase ({len(expert)} trials):")
        print(f"    Energy:    mean={np.mean(energies):.4f}  std={np.std(energies):.4f}  CV={np.std(energies)/np.mean(energies):.3f}")
        print(f"    RT:        mean={np.nanmean(rts):.3f}s   std={np.nanstd(rts):.3f}s  CV={np.nanstd(rts)/np.nanmean(rts):.3f}")
        print(f"    Variance:  mean={np.mean(variances):.4f}  std={np.std(variances):.4f}  CV={np.std(variances)/np.mean(variances):.3f}")
        print(f"    Sharpness: mean={np.mean(sharpness):.4f}  std={np.std(sharpness):.4f}  CV={np.std(sharpness)/np.mean(sharpness):.3f}")
        print(f"    Kurtosis:  mean={np.mean(kurtoses):.4f}  std={np.std(kurtoses):.4f}")

        # Coefficient of Variation (CV) tells us how "locked in" the configuration is.
        # Low CV = stable memoized state. High CV = still fluctuating.
        avg_cv = np.mean([
            np.std(energies) / np.mean(energies),
            np.nanstd(rts) / np.nanmean(rts),
            np.std(variances) / np.mean(variances),
            np.std(sharpness) / np.mean(sharpness),
        ])
        print(f"    >> Average CV across metrics: {avg_cv:.3f} (lower = more memoized)")

    # ================================================================== #
    #  SUMMARY: BRAIN EFFICIENCY NUMBERS
    # ================================================================== #
    print("\n" + "=" * 70)
    print(" [4] BRAIN EFFICIENCY SUMMARY")
    print("=" * 70)

    all_explore = []
    all_expert = []
    for sub, trials in all_subject_data.items():
        all_explore.extend([t for t in trials if t['Phase'] == 'Exploration'])
        all_expert.extend([t for t in trials if t['Phase'] == 'Expert'])

    if all_explore and all_expert:
        e_learn = np.mean([t['Energy'] for t in all_explore])
        e_mem = np.mean([t['Energy'] for t in all_expert])
        rt_learn = np.nanmean([t['RT'] for t in all_explore if not np.isnan(t['RT'])])
        rt_mem = np.nanmean([t['RT'] for t in all_expert if not np.isnan(t['RT'])])

        # Correct trials during exploration vs expert
        correct_explore = [t for t in all_explore if t['Correct'] == 1]
        correct_expert = [t for t in all_expert if t['Correct'] == 1]
        acc_explore = len(correct_explore) / max(1, len(all_explore))
        acc_expert = len(correct_expert) / max(1, len(all_expert))

        print(f"  Exploration Phase ({len(all_explore)} trials):")
        print(f"    Accuracy:      {acc_explore:.1%}")
        print(f"    Mean Energy:   {e_learn:.4f}")
        print(f"    Mean RT:       {rt_learn:.3f}s")
        print(f"    Energy * RT:   {e_learn * rt_learn:.4f} (metabolic-time cost per trial)")

        print(f"\n  Expert Phase ({len(all_expert)} trials):")
        print(f"    Accuracy:      {acc_expert:.1%}")
        print(f"    Mean Energy:   {e_mem:.4f}")
        print(f"    Mean RT:       {rt_mem:.3f}s")
        print(f"    Energy * RT:   {e_mem * rt_mem:.4f} (metabolic-time cost per trial)")

        efficiency_gain = (e_learn * rt_learn) / (e_mem * rt_mem + 1e-8)
        print(f"\n  >> Memoization Efficiency Gain: {efficiency_gain:.2f}x")
        print(f"     (The brain spends {efficiency_gain:.1f}x more metabolic-time resources")
        print(f"      during learning than during memoized execution)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze_memoization()
