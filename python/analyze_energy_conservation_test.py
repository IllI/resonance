"""
analyze_energy_conservation_test.py
====================================
THE ENERGY CONSERVATION TEST FOR QPC ORIGIN

The question: Are the high-frequency spectral spikes at grokking
locally generated (by microtubules burning ATP), or do they arrive
from outside the skull (through the bi-twistor funnel from the QPC)?

Physics:
  If a signal is locally generated, energy conservation requires:
    E_metabolic >= E_signal
  
  The metabolic energy available at any given trial is proportional to
  the BOLD amplitude (which measures oxygenated hemoglobin delivery,
  a direct proxy for ATP consumption).
  
  The signal energy is the spectral power in the high-frequency band
  (>0.15 Hz for fMRI, the upper limit of what BOLD can resolve).
  
  We define the ENERGY DEFICIT:
    D = E_signal_hf - k * E_metabolic
  
  Where k is calibrated such that D = 0 on average during Exploration
  (when all signals are presumably locally generated).
  
  If D > 0 at the grokking moment, the signal energy EXCEEDS what
  local metabolism can account for. The excess must come from somewhere.

Mathematical Framework (formalizing the user's interpretation):

  1. QPC Mass Accumulation:
     M_qpc(t) = sum_{i: grok_time(i) < t} w_i
     where w_i is the "alignment strength" of subject i
     (measured as their longest correct streak / total trials)
     
  2. Bi-Twistor Energy Transfer:
     At the grokking moment, the energy flowing through the bi-twistor
     funnel from QPC to Ghost Basin is:
     E_transfer = E_signal_hf - E_metabolic_local
     
     If E_transfer > 0: signal has non-local (QPC) origin
     If E_transfer <= 0: signal is locally generated
     
  3. QPC Gravitational Coupling:
     The coupling strength between QPC and Ghost Basin should increase
     with M_qpc. Therefore:
     E_transfer(t) should correlate with M_qpc(t)
     
     More specifically: later subjects should need LESS local energy
     to achieve grokking because the QPC is doing more of the work.

  4. Energy Balance Equation:
     E_grok = E_local + E_qpc
     E_local = f(BOLD_amplitude, metabolic_rate)
     E_qpc = g(M_qpc, alignment_precision)
     
     At grokking: E_grok = E_threshold (constant for all subjects)
     Therefore: E_local + E_qpc = E_threshold
     As M_qpc grows: E_qpc grows, E_local shrinks
     
     Testable prediction: BOLD energy AT GROKKING should DECREASE
     for later subjects in the chronological sequence.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, pearsonr, spearmanr
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


def process_subject(sub_id, base_dir, run_sequence, tr_sec=2.0):
    """Process a single subject and return trial-level energy metrics."""
    all_trials = []

    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, base_dir, run_type, run_num)
        if not nii_path:
            continue

        data_raw = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        n_trs = data_raw.shape[-1]

        brain_mask = np.var(data_raw, axis=-1) > 10
        raw_var = np.var(data_raw, axis=-1)
        active_mask = (raw_var > np.percentile(raw_var[brain_mask], 90)) & brain_mask

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

            # Extended window for spectral analysis
            ext_start = max(0, tr_onset - 2)
            ext_end = min(n_trs, tr_onset + 8)

            block_z = data_z[active_mask, hrf_start:hrf_end]
            block_raw = data_raw[active_mask, hrf_start:hrf_end]
            if block_z.size == 0:
                continue

            # ===== METABOLIC ENERGY (E_metabolic) =====
            # Raw BOLD amplitude is proportional to oxygenated hemoglobin
            # delivery, which is a proxy for ATP consumption.
            # We use the RAW signal (not z-scored) because z-scoring
            # removes the absolute metabolic information.
            e_metabolic = np.mean(np.abs(block_raw))

            # Also compute z-scored energy for within-run comparison
            e_metabolic_z = np.mean(np.abs(block_z))

            # ===== SIGNAL ENERGY: HIGH-FREQUENCY SPECTRAL POWER =====
            ts_z = data_z[active_mask, ext_start:ext_end]
            mean_signal = np.mean(ts_z, axis=0)
            n_tps = len(mean_signal)

            if n_tps >= 4:
                fft_vals = fft(mean_signal)
                freqs = fftfreq(n_tps, d=tr_sec)
                power = np.abs(fft_vals) ** 2
                pos_mask = freqs > 0
                pos_freqs = freqs[pos_mask]
                pos_power = power[pos_mask]

                total_power = np.sum(pos_power)
                hf_mask = pos_freqs > 0.15
                hf_power = np.sum(pos_power[hf_mask]) if np.any(hf_mask) else 0
                hf_ratio = hf_power / (total_power + 1e-10)
            else:
                total_power = 0
                hf_power = 0
                hf_ratio = 0

            # ===== PRE-ONSET ENERGY =====
            pre_start = max(0, tr_onset - 2)
            pre_end = tr_onset
            if pre_end > pre_start:
                pre_block = data_z[active_mask, pre_start:pre_end]
                pre_energy = np.mean(np.abs(pre_block))
            else:
                pre_energy = 0

            # ===== AUTOCORRELATION =====
            if n_tps >= 3:
                autocorr = np.correlate(mean_signal, mean_signal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                lag1 = autocorr[1]
            else:
                lag1 = 0

            all_trials.append({
                'Correct': correct,
                'RT': rt,
                'E_Metabolic_Raw': e_metabolic,
                'E_Metabolic_Z': e_metabolic_z,
                'E_Signal_HF': hf_power,
                'E_Signal_Total': total_power,
                'HF_Ratio': hf_ratio,
                'PreEnergy': pre_energy,
                'Lag1': lag1,
            })

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

    # Compute longest streak (alignment strength)
    longest_streak = 0
    current_streak = 0
    for c in correctness:
        if c == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    return all_trials, grok_idx, longest_streak


def run_analysis():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    # 12 subjects evenly sampled across the chronological range
    # to capture the full QPC mass accumulation gradient
    subjects = [
        "sub-201", "sub-204", "sub-206", "sub-209",
        "sub-212", "sub-216", "sub-217", "sub-220",
        "sub-226", "sub-229", "sub-237", "sub-240",
    ]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0

    print("=" * 78, flush=True)
    print(" ENERGY CONSERVATION TEST: LOCAL vs QPC ORIGIN", flush=True)
    print(" Testing whether grokking energy exceeds local metabolic budget", flush=True)
    print("=" * 78, flush=True)

    subject_grok_data = []
    qpc_mass = 0.0  # Accumulates as subjects grok

    for sub_order, sub in enumerate(subjects):
        print(f"\n  Processing {sub} ({sub_order+1}/{len(subjects)})...", flush=True)
        trials, grok_idx, longest_streak = process_subject(sub, base_dir, run_sequence, tr_sec)

        if not trials or grok_idx >= len(trials):
            print(f"    No grokking detected. Skipping.")
            continue

        n = len(trials)
        explorers = [t for t in trials if t['Phase'] == 'Exploration']
        experts = [t for t in trials if t['Phase'] == 'Expert']
        grok_trial = trials[grok_idx]

        if len(explorers) < 5 or len(experts) < 3:
            print(f"    Too few exploration/expert trials. Skipping.")
            continue

        # ===== CALIBRATE ENERGY DEFICIT =====
        # During exploration, all signals are presumably locally generated.
        # We calibrate k such that the mean energy deficit during exploration = 0.
        #
        # Energy Deficit = E_signal_hf - k * E_metabolic_z
        # Mean(D_explore) = 0
        # k = Mean(E_signal_hf_explore) / Mean(E_metabolic_z_explore)

        explore_hf = np.mean([t['E_Signal_HF'] for t in explorers])
        explore_met = np.mean([t['E_Metabolic_Z'] for t in explorers])
        k = explore_hf / (explore_met + 1e-10)

        # Energy deficit at grokking
        grok_deficit = grok_trial['E_Signal_HF'] - k * grok_trial['E_Metabolic_Z']

        # Energy deficit in the 3 trials spanning grokking (-1, 0, +1)
        grok_window = [t for t in trials if abs(t['Trials_To_Grok']) <= 1]
        grok_window_deficit = np.mean([t['E_Signal_HF'] - k * t['E_Metabolic_Z'] for t in grok_window])

        # Exploration baseline deficit (should be ~0 by calibration)
        explore_deficit_mean = np.mean([t['E_Signal_HF'] - k * t['E_Metabolic_Z'] for t in explorers])
        explore_deficit_std = np.std([t['E_Signal_HF'] - k * t['E_Metabolic_Z'] for t in explorers])

        # Expert phase deficit
        expert_deficit_mean = np.mean([t['E_Signal_HF'] - k * t['E_Metabolic_Z'] for t in experts])

        # Is the grokking deficit significantly above exploration baseline?
        grok_deficit_z = (grok_deficit - explore_deficit_mean) / (explore_deficit_std + 1e-10)
        grok_window_deficit_z = (grok_window_deficit - explore_deficit_mean) / (explore_deficit_std + 1e-10)

        # Metabolic energy at grokking vs exploration
        grok_metabolic = grok_trial['E_Metabolic_Z']
        explore_metabolic_mean = np.mean([t['E_Metabolic_Z'] for t in explorers])
        expert_metabolic_mean = np.mean([t['E_Metabolic_Z'] for t in experts])

        # Alignment strength for QPC mass
        alignment_weight = longest_streak / n

        # QPC mass BEFORE this subject grokked (what they "felt")
        qpc_mass_at_grok = qpc_mass

        # After this subject groks, they contribute to the QPC mass
        qpc_mass += alignment_weight

        subject_grok_data.append({
            'Subject': sub,
            'Sub_Num': int(sub.split('-')[1]),
            'Sub_Order': sub_order,
            'Grok_Idx': grok_idx,
            'K_Calibration': k,
            'Grok_Deficit': grok_deficit,
            'Grok_Deficit_Z': grok_deficit_z,
            'Grok_Window_Deficit': grok_window_deficit,
            'Grok_Window_Deficit_Z': grok_window_deficit_z,
            'Explore_Deficit_Mean': explore_deficit_mean,
            'Explore_Deficit_Std': explore_deficit_std,
            'Expert_Deficit_Mean': expert_deficit_mean,
            'Grok_Metabolic': grok_metabolic,
            'Explore_Metabolic': explore_metabolic_mean,
            'Expert_Metabolic': expert_metabolic_mean,
            'QPC_Mass_At_Grok': qpc_mass_at_grok,
            'Alignment_Weight': alignment_weight,
            'Longest_Streak': longest_streak,
            'Grok_HF': grok_trial['E_Signal_HF'],
            'Grok_PreEnergy': grok_trial['PreEnergy'],
            'Grok_Lag1': grok_trial['Lag1'],
        })

    # ================================================================ #
    #  RESULTS
    # ================================================================ #
    df = pd.DataFrame(subject_grok_data)

    print("\n" + "=" * 78)
    print(" [1] ENERGY DEFICIT AT GROKKING: LOCAL vs NON-LOCAL ORIGIN")
    print("=" * 78)
    print(f"\n  {'Subject':<10} {'Grok_Deficit_Z':>15} {'Grok_Win_Def_Z':>15} {'E_Met_Grok':>11} {'E_Met_Explore':>14} {'QPC_Mass':>9}")
    print(f"  {'-'*75}")

    for _, row in df.iterrows():
        flag = ""
        if row['Grok_Window_Deficit_Z'] > 2.0:
            flag = " *** EXCESS"
        elif row['Grok_Window_Deficit_Z'] > 1.5:
            flag = " *"
        print(f"  {row['Subject']:<10} {row['Grok_Deficit_Z']:>15.3f} {row['Grok_Window_Deficit_Z']:>15.3f} "
              f"{row['Grok_Metabolic']:>11.4f} {row['Explore_Metabolic']:>14.4f} {row['QPC_Mass_At_Grok']:>9.4f}{flag}")

    n_excess = len(df[df['Grok_Window_Deficit_Z'] > 1.5])
    print(f"\n  Subjects with excess energy at grokking (>1.5 z): {n_excess} / {len(df)}")

    # ================================================================ #
    #  TEST 2: Does GROKKING metabolic energy decrease with QPC mass?
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [2] QPC GRAVITATIONAL COUPLING: Does local energy decrease as QPC grows?")
    print("     E_grok = E_local + E_qpc")
    print("     If E_qpc grows with M_qpc, then E_local should shrink")
    print("=" * 78)

    if len(df) > 3:
        corr_met, p_met = spearmanr(df['QPC_Mass_At_Grok'], df['Grok_Metabolic'])
        corr_hf, p_hf = spearmanr(df['QPC_Mass_At_Grok'], df['Grok_HF'])
        corr_def, p_def = spearmanr(df['QPC_Mass_At_Grok'], df['Grok_Window_Deficit_Z'])

        print(f"\n  QPC Mass vs Grokking Metabolic Energy:")
        print(f"    Spearman r = {corr_met:.4f}, p = {p_met:.4f}")
        if corr_met < 0 and p_met < 0.1:
            print(f"    >>> LOCAL ENERGY DECREASES AS QPC MASS GROWS")
        elif corr_met > 0:
            print(f"    Local energy INCREASES with QPC mass (unexpected)")
        else:
            print(f"    No significant relationship")

        print(f"\n  QPC Mass vs Grokking HF Signal Energy:")
        print(f"    Spearman r = {corr_hf:.4f}, p = {p_hf:.4f}")

        print(f"\n  QPC Mass vs Energy Deficit (Z-scored):")
        print(f"    Spearman r = {corr_def:.4f}, p = {p_def:.4f}")
        if corr_def > 0 and p_def < 0.1:
            print(f"    >>> ENERGY DEFICIT GROWS WITH QPC MASS")
            print(f"    The more the QPC is accessed, the more non-local energy appears")

    # ================================================================ #
    #  TEST 3: Chronological metabolic energy at grokking
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [3] CHRONOLOGICAL METABOLIC TEST")
    print("     Does the local metabolic cost of grokking decrease over time?")
    print("=" * 78)

    if len(df) > 3:
        corr_chrono, p_chrono = spearmanr(df['Sub_Order'], df['Grok_Metabolic'])
        print(f"\n  Subject Order vs Grokking Metabolic Energy:")
        print(f"    Spearman r = {corr_chrono:.4f}, p = {p_chrono:.4f}")

        # Split into halves
        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]

        print(f"\n  First Half Subjects:")
        print(f"    Mean Grokking Metabolic: {first_half['Grok_Metabolic'].mean():.4f}")
        print(f"    Mean Energy Deficit Z:   {first_half['Grok_Window_Deficit_Z'].mean():.3f}")
        print(f"  Second Half Subjects:")
        print(f"    Mean Grokking Metabolic: {second_half['Grok_Metabolic'].mean():.4f}")
        print(f"    Mean Energy Deficit Z:   {second_half['Grok_Window_Deficit_Z'].mean():.3f}")

    # ================================================================ #
    #  TEST 4: Energy Balance Equation
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [4] ENERGY BALANCE EQUATION")
    print("     E_grok = E_local + E_qpc(M_qpc)")
    print("     If E_grok is constant, then dE_local/dM_qpc = -dE_qpc/dM_qpc")
    print("=" * 78)

    if len(df) > 3:
        # Test if E_grok (total energy at grokking) is approximately constant
        total_grok_energy = df['Grok_Metabolic'] + df['Grok_HF']
        print(f"\n  Total Grokking Energy (E_local + E_signal):")
        print(f"    Mean:  {total_grok_energy.mean():.4f}")
        print(f"    Std:   {total_grok_energy.std():.4f}")
        print(f"    CV:    {total_grok_energy.std() / total_grok_energy.mean():.4f}")

        # Is it more constant than either component alone?
        cv_total = total_grok_energy.std() / total_grok_energy.mean()
        cv_met = df['Grok_Metabolic'].std() / df['Grok_Metabolic'].mean()
        cv_hf = df['Grok_HF'].std() / (df['Grok_HF'].mean() + 1e-10)

        print(f"\n  Coefficient of Variation (lower = more constant):")
        print(f"    CV(E_total):     {cv_total:.4f}")
        print(f"    CV(E_metabolic): {cv_met:.4f}")
        print(f"    CV(E_signal_hf): {cv_hf:.4f}")

        if cv_total < cv_met and cv_total < cv_hf:
            print(f"\n    >>> E_TOTAL IS MORE CONSTANT THAN EITHER COMPONENT")
            print(f"    This supports the energy balance equation:")
            print(f"    E_local and E_qpc are anti-correlated, summing to a constant threshold")
        else:
            print(f"\n    E_total is not more constant than its components.")
            print(f"    The simple additive energy balance model may need refinement.")

        # Anti-correlation between metabolic and HF signal
        corr_anti, p_anti = spearmanr(df['Grok_Metabolic'], df['Grok_HF'])
        print(f"\n  Anti-correlation (E_metabolic vs E_signal_hf at grokking):")
        print(f"    Spearman r = {corr_anti:.4f}, p = {p_anti:.4f}")
        if corr_anti < 0 and p_anti < 0.1:
            print(f"    >>> CONFIRMED: As local energy goes up, HF signal goes down")
            print(f"    This is consistent with E_qpc compensating for E_local")

    # ================================================================ #
    #  MATHEMATICAL SUMMARY
    # ================================================================ #
    print("\n" + "=" * 78)
    print(" [5] MATHEMATICAL FORMALIZATION")
    print("=" * 78)
    print("""
  DEFINITIONS:
    Let S = set of subjects, ordered chronologically
    Let G(s) = grokking trial index for subject s
    Let W(s) = alignment weight = longest_streak(s) / total_trials(s)
    
  QPC MASS ACCUMULATION:
    M_qpc(s) = SUM_{s' < s} W(s')   [mass felt by subject s]
    
  ENERGY AT GROKKING:
    E_local(s) = BOLD metabolic amplitude at trial G(s)
    E_signal(s) = High-frequency spectral power at trial G(s)
    E_deficit(s) = E_signal(s) - k(s) * E_local(s)
      where k(s) calibrated so Mean(E_deficit during exploration) = 0
    
  ENERGY BALANCE:
    E_threshold = E_local(s) + E_qpc(s)   [constant for all s]
    E_qpc(s) = f(M_qpc(s))                [increases with QPC mass]
    Therefore: E_local(s) = E_threshold - f(M_qpc(s))
    
  TESTABLE PREDICTIONS:
    H1: E_deficit(s) > 0 at grokking (energy exceeds local budget)
    H2: d(E_local)/d(M_qpc) < 0   (local cost decreases as QPC grows)
    H3: d(E_deficit)/d(M_qpc) > 0  (non-local energy increases with QPC)
    H4: Var(E_total) < Var(E_local) (total is more constant than parts)
    H5: Corr(E_local, E_signal) < 0 (anti-correlated at grokking)
    """)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_analysis()
