"""
analyze_grokking_birth_moment.py
================================
Tracks the crystallization of the working memory template at/after grokking.

Strategy:
1. Build a "stable expert template" from the LAST N expert trials (the most
   locked-in working memory state)
2. Measure similarity of EVERY trial to this template (spatial correlation)
3. Walk backward from expert → grokking to find the FIRST trial where the
   template appears above threshold
4. At that "birth moment" trial, perform spectral dissection:
   - FFT of the voxel timeseries within the trial window
   - Power spectral density breakdown across frequency bands
   - Temporal autocorrelation structure (persistence/anti-persistence)
   - Pre-onset BOLD activity (retroactively imposed signals)
   - Comparison to surrounding trials (what changed at this exact moment?)

This analysis looks for quantum signatures hiding in the BOLD signal at
the exact moment the working memory configuration "snaps into place."
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, pearsonr
from scipy.signal import welch
from scipy.fft import fft, fftfreq
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


N_FINGERPRINT_BINS = 50  # Fixed-size histogram bins for spatial fingerprint


def make_fingerprint(block_z):
    """Create a fixed-size statistical fingerprint from a variable-size voxel block.

    This produces the same dimensionality regardless of how many voxels are
    in the active mask, making patterns comparable across runs.

    Returns a vector of length N_FINGERPRINT_BINS + 8 (histogram + stats).
    """
    mean_pattern = np.mean(block_z, axis=1)  # per-voxel mean across time

    # Fixed-size histogram of activation values
    hist, _ = np.histogram(mean_pattern, bins=N_FINGERPRINT_BINS, range=(-3, 3), density=True)

    # Statistical summary (invariant to n_voxels)
    stats = np.array([
        np.mean(mean_pattern),
        np.std(mean_pattern),
        kurtosis(mean_pattern),
        np.percentile(mean_pattern, 25),
        np.percentile(mean_pattern, 50),
        np.percentile(mean_pattern, 75),
        np.percentile(mean_pattern, 95),
        np.mean(np.abs(mean_pattern)),
    ])

    return np.concatenate([hist, stats])


def extract_extended_trial_data(data_z, active_mask, events, tr_sec, data_raw):
    """Extract per-trial fixed-size fingerprint AND raw timeseries for spectral analysis."""
    n_trs = data_z.shape[-1]
    trials = []

    for _, row in events.iterrows():
        if 'correct' not in row or pd.isna(row['correct']):
            continue
        correct = int(row['correct'])
        rt = row.get('response_time', np.nan)
        if pd.isna(rt) or rt <= 0:
            rt = np.nan

        tr_onset = int(row['onset'] / tr_sec)

        # HRF window: TRs 2-5 after onset (4-10 seconds post-stimulus)
        hrf_start = min(tr_onset + 2, n_trs - 1)
        hrf_end = min(tr_onset + 5, n_trs)
        if hrf_start >= hrf_end:
            continue

        # Extended window for spectral analysis: -2 to +8 TRs around onset
        ext_start = max(0, tr_onset - 2)
        ext_end = min(n_trs, tr_onset + 8)

        block_z = data_z[active_mask, hrf_start:hrf_end]
        if block_z.size == 0:
            continue

        # Fixed-size fingerprint (comparable across runs with different masks)
        fingerprint = make_fingerprint(block_z)

        # Extended timeseries for spectral analysis
        ts_z = data_z[active_mask, ext_start:ext_end]
        ts_raw = data_raw[active_mask, ext_start:ext_end]

        # Pre-onset activity (2 TRs before stimulus)
        pre_start = max(0, tr_onset - 2)
        pre_end = tr_onset
        if pre_end > pre_start:
            pre_onset = data_z[active_mask, pre_start:pre_end]
        else:
            pre_onset = np.zeros((np.sum(active_mask), 1))

        trials.append({
            'Correct': correct,
            'RT': rt,
            'Pattern': fingerprint,
            'Timeseries_Z': ts_z,
            'Timeseries_Raw': ts_raw,
            'Pre_Onset': pre_onset,
            'TR_Onset': tr_onset,
            'Ext_Start': ext_start,
            'Ext_End': ext_end,
        })
    return trials


def tag_phases(trial_list, streak_threshold=5):
    n = len(trial_list)
    if n == 0:
        return trial_list
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
        trial['Trial_Idx'] = i
        trial['Grok_Idx'] = grok_idx
        if i < grok_idx:
            trial['Phase'] = 'Exploration'
        elif i == grok_idx:
            trial['Phase'] = 'Grokking'
        else:
            trial['Phase'] = 'Expert'
    return trial_list


def spectral_dissection(timeseries_z, tr_sec=2.0):
    """Perform spectral analysis on a trial's extended voxel timeseries.

    Args:
        timeseries_z: shape (n_voxels, n_timepoints) z-scored BOLD
        tr_sec: TR in seconds

    Returns:
        dict with spectral features
    """
    n_voxels, n_tps = timeseries_z.shape
    if n_tps < 4:
        return None

    # Mean signal across active voxels (the "bulk brain signal")
    mean_signal = np.mean(timeseries_z, axis=0)
    fs = 1.0 / tr_sec  # Sampling frequency (0.5 Hz for TR=2s)

    # FFT
    fft_vals = fft(mean_signal)
    freqs = fftfreq(n_tps, d=tr_sec)
    power = np.abs(fft_vals) ** 2
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]

    # Power spectral density via Welch (more robust with short series)
    if n_tps >= 4:
        nperseg = min(n_tps, max(4, n_tps))
        f_welch, psd_welch = welch(mean_signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    else:
        f_welch, psd_welch = np.array([0]), np.array([0])

    # Temporal autocorrelation structure
    autocorr = np.correlate(mean_signal, mean_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    lag1_autocorr = autocorr[1] if len(autocorr) > 1 else 0

    # Per-voxel variance across time (how synchronized are the voxels?)
    voxel_temporal_var = np.var(timeseries_z, axis=1)
    spatial_sync = np.std(voxel_temporal_var) / (np.mean(voxel_temporal_var) + 1e-8)

    # Inter-voxel correlation at each TR (how coherent is the spatial pattern?)
    # Compare first half vs second half of the window
    mid = n_tps // 2
    if mid > 0 and n_tps - mid > 0:
        early_pattern = np.mean(timeseries_z[:, :mid], axis=1)
        late_pattern = np.mean(timeseries_z[:, mid:], axis=1)
        if np.std(early_pattern) > 0 and np.std(late_pattern) > 0:
            temporal_pattern_stability = pearsonr(early_pattern, late_pattern)[0]
        else:
            temporal_pattern_stability = 0.0
    else:
        temporal_pattern_stability = 0.0

    # Peak frequency (where is most of the spectral power?)
    if len(pos_freqs) > 0:
        peak_freq = pos_freqs[np.argmax(pos_power)]
        total_power = np.sum(pos_power)
        dominant_power_ratio = np.max(pos_power) / (total_power + 1e-10)
    else:
        peak_freq = 0
        dominant_power_ratio = 0

    return {
        'PeakFreq_Hz': peak_freq,
        'DominantPowerRatio': dominant_power_ratio,
        'Lag1_Autocorr': lag1_autocorr,
        'SpatialSync': spatial_sync,
        'TemporalPatternStability': temporal_pattern_stability,
        'MeanSignal': mean_signal,
        'FFT_Freqs': pos_freqs,
        'FFT_Power': pos_power,
        'Welch_Freqs': f_welch,
        'Welch_PSD': psd_welch,
        'TotalPower': np.sum(pos_power) if len(pos_power) > 0 else 0,
    }


def analyze_pre_onset(pre_onset_data):
    """Analyze the BOLD signal BEFORE stimulus onset for retroactive signatures.

    If microtubules or quantum alignment impose structure retroactively,
    we might see unusual pre-stimulus patterns at the grokking moment.
    """
    if pre_onset_data.size == 0:
        return {'PreOnsetEnergy': 0, 'PreOnsetKurtosis': 0, 'PreOnsetSpread': 0}

    pre_energy = np.mean(np.abs(pre_onset_data))
    pre_kurtosis = kurtosis(np.mean(pre_onset_data, axis=1))
    pre_spread = np.std(np.mean(pre_onset_data, axis=1))

    return {
        'PreOnsetEnergy': pre_energy,
        'PreOnsetKurtosis': pre_kurtosis,
        'PreOnsetSpread': pre_spread,
    }


def run_analysis():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0

    print("=" * 75)
    print(" GROKKING BIRTH MOMENT: TRACKING THE WORKING MEMORY TEMPLATE")
    print(" Searching for quantum signatures at the moment of crystallization")
    print("=" * 75)

    for sub in subjects:
        print(f"\n{'='*75}")
        print(f" SUBJECT: {sub}")
        print(f"{'='*75}")

        all_trials = []
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_data(sub, base_dir, run_type, run_num)
            if not nii_path:
                continue
            data_raw = nib.load(nii_path).get_fdata()
            events = pd.read_csv(tsv_path, sep='\t')
            data_z = np.nan_to_num(zscore(data_raw, axis=-1))
            brain_mask = np.var(data_raw, axis=-1) > 10
            raw_var = np.var(data_raw, axis=-1)
            active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask

            trials = extract_extended_trial_data(data_z, active_mask, events, tr_sec, data_raw)
            for t in trials:
                t['Run'] = f"{run_type}{run_num}"
            all_trials.extend(trials)
            if os.path.exists(nii_path):
                os.remove(nii_path)

        all_trials = tag_phases(all_trials, 5)
        experts = [t for t in all_trials if t['Phase'] == 'Expert']
        explorers = [t for t in all_trials if t['Phase'] == 'Exploration']

        if len(experts) < 5:
            print(f"  Too few expert trials ({len(experts)}). Skipping.")
            continue

        # ============================================================ #
        #  STEP 1: Build stable expert template from last N trials
        # ============================================================ #
        n_template = min(10, len(experts))
        template_trials = experts[-n_template:]
        template_pattern = np.mean([t['Pattern'] for t in template_trials], axis=0)

        print(f"\n  [1] EXPERT TEMPLATE built from last {n_template} trials")
        print(f"      Template vector dim: {len(template_pattern)}")

        # ============================================================ #
        #  STEP 2: Measure similarity of EVERY trial to the template
        # ============================================================ #
        for t in all_trials:
            p = t['Pattern']
            if np.std(p) > 0 and np.std(template_pattern) > 0:
                t['TemplateSim'] = pearsonr(p, template_pattern)[0]
            else:
                t['TemplateSim'] = 0.0

        print(f"\n  [2] TEMPLATE SIMILARITY across all {len(all_trials)} trials")
        print(f"      {'Phase':<15} {'Mean Sim':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"      {'-'*51}")

        for phase in ['Exploration', 'Grokking', 'Expert']:
            phase_trials = [t for t in all_trials if t['Phase'] == phase]
            if phase_trials:
                sims = [t['TemplateSim'] for t in phase_trials]
                print(f"      {phase:<15} {np.mean(sims):>10.4f} {np.std(sims):>8.4f} {np.min(sims):>8.4f} {np.max(sims):>8.4f}")

        # ============================================================ #
        #  STEP 3: Find the BIRTH MOMENT (first trial where template appears)
        # ============================================================ #
        grok_idx = all_trials[0]['Grok_Idx'] if all_trials else len(all_trials)

        # Find first trial at or after grokking where similarity exceeds
        # a meaningful threshold (mean_expert - 1*std_expert)
        expert_sims = [t['TemplateSim'] for t in experts]
        sim_threshold = np.mean(expert_sims) - 1.0 * np.std(expert_sims)

        birth_idx = None
        for i, t in enumerate(all_trials):
            if i >= max(0, grok_idx - 3) and t['TemplateSim'] >= sim_threshold:
                birth_idx = i
                break

        if birth_idx is None:
            print(f"\n  [3] Could not find birth moment. Threshold too high.")
            continue

        birth_trial = all_trials[birth_idx]
        print(f"\n  [3] BIRTH MOMENT FOUND at trial {birth_idx} (grok_idx={grok_idx})")
        print(f"      Phase: {birth_trial['Phase']}")
        print(f"      Similarity to stable template: {birth_trial['TemplateSim']:.4f}")
        print(f"      Threshold used: {sim_threshold:.4f}")
        print(f"      Correct: {birth_trial['Correct']}, RT: {birth_trial['RT']:.3f}s")

        # ============================================================ #
        #  STEP 4: SPECTRAL DISSECTION of the birth moment
        # ============================================================ #
        spectral_birth = spectral_dissection(birth_trial['Timeseries_Z'], tr_sec)
        pre_onset_birth = analyze_pre_onset(birth_trial['Pre_Onset'])

        print(f"\n  [4] SPECTRAL DISSECTION OF BIRTH MOMENT (Trial {birth_idx})")
        if spectral_birth:
            print(f"      Peak Frequency:           {spectral_birth['PeakFreq_Hz']:.4f} Hz")
            print(f"      Dominant Power Ratio:      {spectral_birth['DominantPowerRatio']:.4f}")
            print(f"      Lag-1 Autocorrelation:     {spectral_birth['Lag1_Autocorr']:.4f}")
            print(f"      Spatial Synchrony:         {spectral_birth['SpatialSync']:.4f}")
            print(f"      Temporal Pattern Stability: {spectral_birth['TemporalPatternStability']:.4f}")
            print(f"      Total Spectral Power:      {spectral_birth['TotalPower']:.4f}")

        print(f"\n      PRE-ONSET (Before stimulus appeared):")
        print(f"      Pre-Onset Energy:    {pre_onset_birth['PreOnsetEnergy']:.4f}")
        print(f"      Pre-Onset Kurtosis:  {pre_onset_birth['PreOnsetKurtosis']:.4f}")
        print(f"      Pre-Onset Spread:    {pre_onset_birth['PreOnsetSpread']:.4f}")

        # ============================================================ #
        #  STEP 5: COMPARE BIRTH MOMENT to surrounding trials
        # ============================================================ #
        print(f"\n  [5] COMPARISON: Birth Moment vs Surrounding Trials")
        print(f"      {'Trial':>7} {'Phase':<12} {'Sim':>8} {'PeakHz':>8} {'DomPwr':>8} {'Lag1AC':>8} {'SpatSync':>9} {'PreE':>8} {'PreK':>8}")
        print(f"      {'-'*80}")

        window_start = max(0, birth_idx - 5)
        window_end = min(len(all_trials), birth_idx + 8)

        for i in range(window_start, window_end):
            t = all_trials[i]
            sp = spectral_dissection(t['Timeseries_Z'], tr_sec)
            pre = analyze_pre_onset(t['Pre_Onset'])
            marker = " <<< BIRTH" if i == birth_idx else ""
            if sp:
                print(f"      {i:>7} {t['Phase']:<12} {t['TemplateSim']:>8.4f} {sp['PeakFreq_Hz']:>8.4f} {sp['DominantPowerRatio']:>8.4f} {sp['Lag1_Autocorr']:>8.4f} {sp['SpatialSync']:>9.4f} {pre['PreOnsetEnergy']:>8.4f} {pre['PreOnsetKurtosis']:>8.4f}{marker}")

        # ============================================================ #
        #  STEP 6: FATIGUE/DRIFT analysis in Expert phase
        # ============================================================ #
        print(f"\n  [6] EXPERT PHASE DRIFT (Template similarity over time)")
        print(f"      {'Trial':>7} {'Idx_From_Grok':>14} {'TemplateSim':>12} {'RT':>8}")
        print(f"      {'-'*41}")
        for i, t in enumerate(experts):
            idx_from_grok = t['Trial_Idx'] - grok_idx
            rt_str = f"{t['RT']:.3f}" if not np.isnan(t['RT']) else "NaN"
            print(f"      {t['Trial_Idx']:>7} {idx_from_grok:>14} {t['TemplateSim']:>12.4f} {rt_str:>8}")

        # ============================================================ #
        #  STEP 7: STATISTICAL COMPARISON - Birth moment spectral features
        #  vs early exploration AND vs stable expert
        # ============================================================ #
        print(f"\n  [7] QUANTUM SIGNATURE COMPARISON")

        # Collect spectral features for exploration, birth, and expert
        explore_spectrals = []
        for t in explorers[-10:]:  # last 10 exploration trials
            sp = spectral_dissection(t['Timeseries_Z'], tr_sec)
            if sp:
                explore_spectrals.append(sp)

        expert_spectrals = []
        for t in experts[3:]:  # stable expert (skip first 3)
            sp = spectral_dissection(t['Timeseries_Z'], tr_sec)
            if sp:
                expert_spectrals.append(sp)

        birth_sp = spectral_dissection(birth_trial['Timeseries_Z'], tr_sec)

        if explore_spectrals and expert_spectrals and birth_sp:
            metrics = ['PeakFreq_Hz', 'DominantPowerRatio', 'Lag1_Autocorr', 'SpatialSync', 'TemporalPatternStability', 'TotalPower']
            print(f"      {'Metric':<25} {'Late Explore':>13} {'BIRTH MOMENT':>13} {'Stable Expert':>14}")
            print(f"      {'-'*65}")
            for m in metrics:
                explore_val = np.mean([s[m] for s in explore_spectrals])
                birth_val = birth_sp[m]
                expert_val = np.mean([s[m] for s in expert_spectrals])
                # Flag if birth moment is an outlier
                flag = ""
                if m in ['Lag1_Autocorr', 'SpatialSync', 'TemporalPatternStability']:
                    # For these, check if birth is significantly different from both phases
                    explore_std = np.std([s[m] for s in explore_spectrals])
                    expert_std = np.std([s[m] for s in expert_spectrals])
                    if abs(birth_val - explore_val) > 2 * explore_std and abs(birth_val - expert_val) > 2 * expert_std:
                        flag = " *** ANOMALY"
                    elif abs(birth_val - explore_val) > 1.5 * explore_std or abs(birth_val - expert_val) > 1.5 * expert_std:
                        flag = " *"
                print(f"      {m:<25} {explore_val:>13.4f} {birth_val:>13.4f} {expert_val:>14.4f}{flag}")

        # Pre-onset comparison
        print(f"\n      PRE-ONSET COMPARISON:")
        explore_pre = [analyze_pre_onset(t['Pre_Onset']) for t in explorers[-10:]]
        expert_pre = [analyze_pre_onset(t['Pre_Onset']) for t in experts[3:]]
        birth_pre = analyze_pre_onset(birth_trial['Pre_Onset'])

        for pm in ['PreOnsetEnergy', 'PreOnsetKurtosis', 'PreOnsetSpread']:
            ep = np.mean([p[pm] for p in explore_pre])
            bp = birth_pre[pm]
            xp = np.mean([p[pm] for p in expert_pre])
            ep_std = np.std([p[pm] for p in explore_pre])
            xp_std = np.std([p[pm] for p in expert_pre])
            flag = ""
            if abs(bp - ep) > 2 * ep_std and abs(bp - xp) > 2 * xp_std:
                flag = " *** ANOMALY"
            elif abs(bp - ep) > 1.5 * ep_std or abs(bp - xp) > 1.5 * xp_std:
                flag = " *"
            print(f"      {pm:<25} {ep:>13.4f} {bp:>13.4f} {xp:>14.4f}{flag}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_analysis()
