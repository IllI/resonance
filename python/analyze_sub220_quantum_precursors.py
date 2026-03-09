"""
analyze_sub220_quantum_precursors.py
====================================
Deep dive into sub-220 and comparative pre-grokking analysis.

Questions:
1. How long did each subject take to grok? (sub-220 comparative context)
2. What did sub-220's electromagnetic state look like trial-by-trial 
   in the 20 cycles preceding grokking?
3. Are there anomalous spectral signatures before grokking that could
   indicate quantum precursor activity (retrocausal "wake")?
4. Do any subjects show pre-grokking template similarity spikes that
   predict the grokking event before it happens?
5. Is the pre-onset energy anomaly unique to sub-220 or a pattern?

We look for:
- Electromagnetic spikes in the high-frequency end of what fMRI can detect
- Unusual temporal persistence (autocorrelation) that might indicate
  standing wave formation in microtubule networks
- Pre-stimulus BOLD activity that shouldn't be there classically
- Template similarity "precursor pulses" before behavioral grokking
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


N_BINS = 50

def make_fingerprint(block_z):
    mean_pattern = np.mean(block_z, axis=1)
    hist, _ = np.histogram(mean_pattern, bins=N_BINS, range=(-3, 3), density=True)
    stats = np.array([
        np.mean(mean_pattern), np.std(mean_pattern), kurtosis(mean_pattern),
        np.percentile(mean_pattern, 25), np.percentile(mean_pattern, 50),
        np.percentile(mean_pattern, 75), np.percentile(mean_pattern, 95),
        np.mean(np.abs(mean_pattern)),
    ])
    return np.concatenate([hist, stats])


def extract_deep_trial(data_z, active_mask, events, tr_sec, data_raw):
    """Extract comprehensive trial data with full spectral decomposition."""
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
        hrf_start = min(tr_onset + 2, n_trs - 1)
        hrf_end = min(tr_onset + 5, n_trs)
        if hrf_start >= hrf_end:
            continue

        ext_start = max(0, tr_onset - 2)
        ext_end = min(n_trs, tr_onset + 8)

        block_z = data_z[active_mask, hrf_start:hrf_end]
        if block_z.size == 0:
            continue

        fingerprint = make_fingerprint(block_z)

        # Extended timeseries
        ts_z = data_z[active_mask, ext_start:ext_end]

        # Pre-onset (2 TRs before stimulus)
        pre_start = max(0, tr_onset - 2)
        pre_end = tr_onset
        if pre_end > pre_start:
            pre_onset = data_z[active_mask, pre_start:pre_end]
        else:
            pre_onset = np.zeros((n_active, 1))

        # === SPECTRAL FEATURES ===
        mean_signal = np.mean(ts_z, axis=0)
        n_tps = len(mean_signal)
        fs = 1.0 / tr_sec

        # FFT
        if n_tps >= 4:
            fft_vals = fft(mean_signal)
            freqs = fftfreq(n_tps, d=tr_sec)
            power = np.abs(fft_vals) ** 2
            pos_mask = freqs > 0
            pos_freqs = freqs[pos_mask]
            pos_power = power[pos_mask]

            peak_freq = pos_freqs[np.argmax(pos_power)] if len(pos_freqs) > 0 else 0
            total_power = np.sum(pos_power) if len(pos_power) > 0 else 0
            dom_ratio = np.max(pos_power) / (total_power + 1e-10) if len(pos_power) > 0 else 0

            # High-frequency power ratio (above Nyquist/2)
            # For TR=2s, Nyquist = 0.25Hz. "High freq" = > 0.15 Hz
            hf_mask = pos_freqs > 0.15
            hf_power = np.sum(pos_power[hf_mask]) / (total_power + 1e-10) if np.any(hf_mask) else 0
        else:
            peak_freq = 0; total_power = 0; dom_ratio = 0; hf_power = 0

        # Autocorrelation
        if n_tps >= 3:
            autocorr = np.correlate(mean_signal, mean_signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            lag1 = autocorr[1] if len(autocorr) > 1 else 0
            lag2 = autocorr[2] if len(autocorr) > 2 else 0
        else:
            lag1 = 0; lag2 = 0

        # Spatial sync
        voxel_var = np.var(ts_z, axis=1)
        spatial_sync = np.std(voxel_var) / (np.mean(voxel_var) + 1e-8)

        # Pre-onset metrics
        pre_energy = np.mean(np.abs(pre_onset))
        pre_kurt = kurtosis(np.mean(pre_onset, axis=1)) if pre_onset.shape[1] > 0 else 0
        pre_spread = np.std(np.mean(pre_onset, axis=1)) if pre_onset.shape[1] > 0 else 0

        # Energy
        energy = np.mean(np.abs(block_z)) * 10.0

        # Temporal sharpness
        tr_energies = np.mean(np.abs(block_z), axis=0)
        peak_amp = np.max(tr_energies)
        mean_amp = np.mean(tr_energies)
        sharpness = peak_amp / (mean_amp + 1e-8)

        # Pattern stability (early vs late half correlation)
        mid = block_z.shape[1] // 2
        if mid > 0 and block_z.shape[1] - mid > 0:
            ep = np.mean(block_z[:, :mid], axis=1)
            lp = np.mean(block_z[:, mid:], axis=1)
            pat_stab = pearsonr(ep, lp)[0] if np.std(ep) > 0 and np.std(lp) > 0 else 0
        else:
            pat_stab = 0

        trials.append({
            'Correct': correct, 'RT': rt, 'Energy': energy,
            'Kurtosis': kurtosis(np.mean(block_z, axis=1)),
            'Fingerprint': fingerprint,
            'PeakFreq': peak_freq, 'TotalPower': total_power,
            'DomRatio': dom_ratio, 'HF_Power': hf_power,
            'Lag1': lag1, 'Lag2': lag2,
            'SpatialSync': spatial_sync, 'Sharpness': sharpness,
            'PatStab': pat_stab,
            'PreEnergy': pre_energy, 'PreKurt': pre_kurt,
            'PreSpread': pre_spread,
            'N_Active': n_active,
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
        trial['Trials_To_Grok'] = i - grok_idx  # negative = before grokking
        if i < grok_idx:
            trial['Phase'] = 'Exploration'
        elif i == grok_idx:
            trial['Phase'] = 'Grokking'
        else:
            trial['Phase'] = 'Expert'
    return trial_list


def run_analysis():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0

    print("=" * 78)
    print(" SUB-220 DEEP DIVE: QUANTUM PRECURSOR ANALYSIS")
    print(" Searching for retrocausal wake effects before grokking")
    print("=" * 78)

    all_subjects = {}

    for sub in subjects:
        sub_trials = []
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

            trials = extract_deep_trial(data_z, active_mask, events, tr_sec, data_raw)
            sub_trials.extend(trials)
            if os.path.exists(nii_path):
                os.remove(nii_path)

        sub_trials = tag_phases(sub_trials, 5)
        all_subjects[sub] = sub_trials

    # ================================================================ #
    #  SECTION 1: Grokking Timeline Comparison
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [1] GROKKING TIMELINE: HOW LONG DID EACH SUBJECT TAKE?")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        if not trials:
            continue
        grok_idx = trials[0]['Grok_Idx']
        total = len(trials)
        n_expert = len([t for t in trials if t['Phase'] == 'Expert'])
        n_explore = len([t for t in trials if t['Phase'] == 'Exploration'])

        # Accuracy during exploration
        explore_correct = [t for t in trials if t['Phase'] == 'Exploration' and t['Correct'] == 1]
        acc_explore = len(explore_correct) / max(1, n_explore)

        # Running accuracy leading up to grokking (last 10 explore trials)
        last10 = [t for t in trials if t['Phase'] == 'Exploration'][-10:]
        acc_last10 = np.mean([t['Correct'] for t in last10]) if last10 else 0

        print(f"\n  {sub}:")
        print(f"    Grokking at trial:     {grok_idx} / {total}")
        print(f"    Exploration trials:    {n_explore}")
        print(f"    Expert trials:         {n_expert}")
        print(f"    Exploration accuracy:  {acc_explore:.1%}")
        print(f"    Last 10 pre-grok acc:  {acc_last10:.1%}")

    # ================================================================ #
    #  SECTION 2: Sub-220 Pre-Grokking Electromagnetic Trajectory
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [2] SUB-220 PRE-GROKKING TRAJECTORY (Trial-by-Trial, last 25 cycles)")
    print("=" * 78)

    sub220 = all_subjects.get('sub-220', [])
    if sub220:
        grok_idx = sub220[0]['Grok_Idx']
        window_start = max(0, grok_idx - 25)
        window_end = min(len(sub220), grok_idx + 10)

        print(f"  Grokking at trial {grok_idx}")
        print(f"\n  {'Trial':>5} {'dgrok':>6} {'Cor':>3} {'RT':>6} {'Energy':>7} {'PkHz':>6} {'DomPwr':>7} "
              f"{'HF_Pwr':>7} {'Lag1':>6} {'Lag2':>6} {'SSync':>6} {'Sharp':>6} "
              f"{'PreE':>6} {'PreK':>6} {'PreSp':>6}")
        print(f"  {'-'*110}")

        for i in range(window_start, window_end):
            t = sub220[i]
            delta = t['Trials_To_Grok']
            marker = " <<<" if delta == 0 else ""
            rt_s = f"{t['RT']:.2f}" if not np.isnan(t['RT']) else "  NaN"
            print(f"  {i:>5} {delta:>6} {t['Correct']:>3} {rt_s:>6} {t['Energy']:>7.3f} "
                  f"{t['PeakFreq']:>6.3f} {t['DomRatio']:>7.3f} {t['HF_Power']:>7.3f} "
                  f"{t['Lag1']:>6.3f} {t['Lag2']:>6.3f} {t['SpatialSync']:>6.3f} "
                  f"{t['Sharpness']:>6.3f} {t['PreEnergy']:>6.3f} {t['PreKurt']:>6.3f} "
                  f"{t['PreSpread']:>6.3f}{marker}")

    # ================================================================ #
    #  SECTION 3: Pre-Onset Anomaly Scan (All Subjects)
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [3] PRE-ONSET ANOMALY SCAN: Looking for precursor spikes")
    print("     (Trials where PreOnsetEnergy > 1.5 * subject mean)")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        pre_energies = [t['PreEnergy'] for t in trials]
        mean_pre = np.mean(pre_energies)
        std_pre = np.std(pre_energies)
        threshold = mean_pre + 1.5 * std_pre

        anomalies = [(i, t) for i, t in enumerate(trials) if t['PreEnergy'] > threshold]
        
        print(f"\n  {sub} (mean PreE: {mean_pre:.4f}, threshold: {threshold:.4f}):")
        if not anomalies:
            print(f"    No pre-onset anomalies detected")
        else:
            print(f"    {len(anomalies)} anomalous pre-onset spikes:")
            for idx, t in anomalies:
                delta = t['Trials_To_Grok']
                print(f"    Trial {idx:>3} (dgrok={delta:>4}) PreE={t['PreEnergy']:.4f} "
                      f"Phase={t['Phase']:<12} Correct={t['Correct']} "
                      f"Lag1={t['Lag1']:.3f} HF={t['HF_Power']:.3f}")

    # ================================================================ #
    #  SECTION 4: Lag-1 Autocorrelation Anomaly Scan
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [4] TEMPORAL PERSISTENCE ANOMALIES (Lag1 > 1.5s above subject mean)")
    print("     Looking for standing wave formation preceding grokking")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        lag1_vals = [t['Lag1'] for t in trials]
        mean_lag1 = np.mean(lag1_vals)
        std_lag1 = np.std(lag1_vals)
        threshold = mean_lag1 + 1.5 * std_lag1

        anomalies = [(i, t) for i, t in enumerate(trials) if t['Lag1'] > threshold]

        print(f"\n  {sub} (mean Lag1: {mean_lag1:.4f}, threshold: {threshold:.4f}):")
        if not anomalies:
            print(f"    No temporal persistence anomalies")
        else:
            # Count how many are in each phase window
            pre_grok_anoms = [(i, t) for i, t in anomalies if t['Trials_To_Grok'] < 0]
            post_grok_anoms = [(i, t) for i, t in anomalies if t['Trials_To_Grok'] >= 0]
            print(f"    {len(pre_grok_anoms)} pre-grok anomalies | {len(post_grok_anoms)} post-grok anomalies")

            # Show the pre-grokking ones (potential precursors)
            if pre_grok_anoms:
                print(f"    PRE-GROKKING temporal persistence spikes:")
                for idx, t in pre_grok_anoms[-8:]:  # last 8
                    print(f"      Trial {idx:>3} (dgrok={t['Trials_To_Grok']:>4}) Lag1={t['Lag1']:.4f} "
                          f"PreE={t['PreEnergy']:.4f} Correct={t['Correct']} HF={t['HF_Power']:.3f}")

    # ================================================================ #
    #  SECTION 5: High-Frequency Power Anomalies
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [5] HIGH-FREQUENCY POWER ANOMALIES (>0.15 Hz)")
    print("     Searching for quantum-range spectral signatures")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        hf_vals = [t['HF_Power'] for t in trials]
        mean_hf = np.mean(hf_vals)
        std_hf = np.std(hf_vals)
        threshold = mean_hf + 1.5 * std_hf

        anomalies = [(i, t) for i, t in enumerate(trials) if t['HF_Power'] > threshold]
        pre_grok_anoms = [(i, t) for i, t in anomalies if t['Trials_To_Grok'] < 0]

        print(f"\n  {sub} (mean HF: {mean_hf:.4f}, threshold: {threshold:.4f}):")
        print(f"    Total HF anomalies: {len(anomalies)} | Pre-grok: {len(pre_grok_anoms)}")

        if pre_grok_anoms:
            # Check clustering: are they concentrated near grokking?
            deltas = [t['Trials_To_Grok'] for _, t in pre_grok_anoms]
            near_grok = sum(1 for d in deltas if d > -10)
            far_from_grok = sum(1 for d in deltas if d <= -10)
            print(f"    Pre-grok HF spikes within 10 trials of grokking: {near_grok}")
            print(f"    Pre-grok HF spikes > 10 trials before grokking: {far_from_grok}")

            for idx, t in pre_grok_anoms[-6:]:
                print(f"      Trial {idx:>3} (dgrok={t['Trials_To_Grok']:>4}) HF={t['HF_Power']:.4f} "
                      f"PreE={t['PreEnergy']:.4f} Lag1={t['Lag1']:.4f} Correct={t['Correct']}")

    # ================================================================ #
    #  SECTION 6: Template Similarity Precursor Pulses
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [6] TEMPLATE PRECURSOR PULSES")
    print("     Does the expert template briefly appear before grokking?")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        experts = [t for t in trials if t['Phase'] == 'Expert']
        if len(experts) < 5:
            continue

        # Build template from last 10 expert trials
        template = np.mean([t['Fingerprint'] for t in experts[-10:]], axis=0)

        # Calculate similarity for all trials
        for t in trials:
            fp = t['Fingerprint']
            if np.std(fp) > 0 and np.std(template) > 0:
                t['TemplateSim'] = pearsonr(fp, template)[0]
            else:
                t['TemplateSim'] = 0.0

        grok_idx = trials[0]['Grok_Idx']

        # Find exploration trials with high template similarity (precursor pulses)
        explore = [t for t in trials if t['Phase'] == 'Exploration']
        if not explore:
            continue

        sims = [t['TemplateSim'] for t in explore]
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        pulse_threshold = mean_sim + 1.5 * std_sim

        pulses = [t for t in explore if t['TemplateSim'] > pulse_threshold]

        print(f"\n  {sub} (explore sim: mean={mean_sim:.4f}, std={std_sim:.4f}, threshold={pulse_threshold:.4f}):")
        print(f"    Precursor pulses above threshold: {len(pulses)}")

        if pulses:
            # Check if pulses cluster near grokking
            pulse_deltas = [t['Trials_To_Grok'] for t in pulses]
            near = sum(1 for d in pulse_deltas if d > -10)
            far = sum(1 for d in pulse_deltas if d <= -10)
            print(f"    Within 10 trials of grokking: {near} | Earlier: {far}")
            print(f"    Last 5 precursor pulses before grokking:")
            for t in pulses[-5:]:
                print(f"      Trial {t['Trial_Idx']:>3} (dgrok={t['Trials_To_Grok']:>4}) "
                      f"Sim={t['TemplateSim']:.4f} Correct={t['Correct']} "
                      f"PreE={t['PreEnergy']:.4f} Lag1={t['Lag1']:.4f} HF={t['HF_Power']:.3f}")

    # ================================================================ #
    #  SECTION 7: Combined Precursor Score
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [7] COMBINED PRECURSOR INDEX")
    print("     Composite score: TemplateSim_z + PreEnergy_z + Lag1_z + HF_z")
    print("     Higher = more quantum precursor markers simultaneously present")
    print("=" * 78)

    for sub, trials in all_subjects.items():
        if not trials or 'TemplateSim' not in trials[0]:
            continue

        # Z-score each metric within subject
        sims = np.array([t.get('TemplateSim', 0) for t in trials])
        pres = np.array([t['PreEnergy'] for t in trials])
        lags = np.array([t['Lag1'] for t in trials])
        hfs = np.array([t['HF_Power'] for t in trials])

        sim_z = (sims - np.mean(sims)) / (np.std(sims) + 1e-8)
        pre_z = (pres - np.mean(pres)) / (np.std(pres) + 1e-8)
        lag_z = (lags - np.mean(lags)) / (np.std(lags) + 1e-8)
        hf_z = (hfs - np.mean(hfs)) / (np.std(hfs) + 1e-8)

        # Combined Precursor Index (CPI)
        cpi = sim_z + pre_z + lag_z + hf_z

        grok_idx = trials[0]['Grok_Idx']

        # Show the trajectory around grokking
        window_start = max(0, grok_idx - 15)
        window_end = min(len(trials), grok_idx + 10)

        print(f"\n  {sub} Combined Precursor Index (CPI) around grokking:")
        print(f"  {'Trial':>5} {'dgrok':>6} {'CPI':>8} {'SimZ':>7} {'PreEZ':>7} {'LagZ':>7} {'HFZ':>7} {'Cor':>3} {'Phase':<12}")
        print(f"  {'-'*72}")

        for i in range(window_start, window_end):
            t = trials[i]
            delta = t['Trials_To_Grok']
            marker = " <<<"  if delta == 0 else ""
            print(f"  {i:>5} {delta:>6} {cpi[i]:>8.3f} {sim_z[i]:>7.3f} {pre_z[i]:>7.3f} "
                  f"{lag_z[i]:>7.3f} {hf_z[i]:>7.3f} {t['Correct']:>3} {t['Phase']:<12}{marker}")

        # Check for CPI spikes in pre-grokking window
        pre_grok_cpi = cpi[max(0, grok_idx-20):grok_idx]
        if len(pre_grok_cpi) > 0:
            max_cpi = np.max(pre_grok_cpi)
            max_trial = np.argmax(pre_grok_cpi) + max(0, grok_idx - 20)
            mean_cpi = np.mean(cpi)
            std_cpi = np.std(cpi)
            print(f"\n  Pre-grokking CPI peak: {max_cpi:.3f} at trial {max_trial} "
                  f"(dgrok={max_trial - grok_idx})")
            print(f"  Overall CPI: mean={mean_cpi:.3f}, std={std_cpi:.3f}")
            if max_cpi > mean_cpi + 2 * std_cpi:
                print(f"  >>> SIGNIFICANT PRECURSOR SPIKE (>2std above mean)")
            elif max_cpi > mean_cpi + 1.5 * std_cpi:
                print(f"  >>> MODERATE PRECURSOR SPIKE (>1.5std above mean)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_analysis()
