"""
analyze_bold_quantum_signatures.py
===================================
NEUROANATOMICALLY-GROUNDED QUANTUM SIGNATURE ANALYSIS

Three physical telltales that discriminate quantum (microtubule) events 
from classical (astrocyte) events in BOLD fMRI:

1. SPECTRAL SCALING (1/f vs 1/f^2):
   - Microtubule criticality => pink noise (1/f, beta ~ 1)
   - Astrocyte metabolism => Brownian noise (1/f^2, beta ~ 2)
   - At grokking, does the spectral slope shift toward 1/f?

2. INITIAL DIP (metabolic oxygen depletion):
   - Microtubule burst => instant O2 consumption => brief BOLD dip 
     BEFORE the main hemodynamic peak
   - With TR=2s we can still detect a relative dip: compare TR_onset
     vs TR_onset+1 vs TR_onset+2
   - If BOLD(onset) < BOLD(onset-1) AND BOLD(onset+2) > BOLD(onset),
     that's an initial dip signature

3. SPATIAL SPECIFICITY (internal vs external origin):
   - If signal is INTERNAL (microtubules): activation should be 
     concentrated in microtubule-dense regions (PFC, parahippocampus)
   - If signal is EXTERNAL (ambient field): activation would be
     uniform across the brain (like an antenna)
   - We measure the coefficient of spatial variation: high = localized 
     = internal; low = uniform = external

Additionally:
4. METABOLIC COST TEST:
   - If internally generated, the HF spike must have a metabolic 
     signature: energy consumed = energy produced
   - If the BOLD amplitude in microtubule-dense ROIs is correlated 
     with the HF power, the signal is locally fueled
   - If NOT correlated, the energy came from elsewhere
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, pearsonr, spearmanr, linregress
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


def define_rois(shape):
    """Define approximate neuroanatomical ROIs from voxel coordinates.
    
    In native space fMRI (no MNI warp), we use the brain geometry:
    - dim 0 (x): Right-to-Left (sagittal)
    - dim 1 (y): Posterior-to-Anterior (coronal)  
    - dim 2 (z): Inferior-to-Superior (axial)
    
    Microtubule-dense regions:
    - PFC: anterior, superior (high y, high z)
    - Parahippocampus: medial, inferior-temporal (mid x, low z, mid y)
    - Thalamus: central, deep (mid x, mid y, mid z)
    
    Control regions:
    - Primary Visual (V1): posterior, medial (mid x, low y, mid z)
    - Primary Motor (M1): superior, mid-anterior (mid x, high y, high z)
    """
    nx, ny, nz = shape[:3]
    
    rois = {}
    
    # Prefrontal Cortex: anterior 25%, superior 40%
    rois['PFC'] = np.zeros(shape[:3], dtype=bool)
    rois['PFC'][:, int(ny*0.75):, int(nz*0.6):] = True
    
    # Parahippocampus: medial 50%, inferior 30%, middle coronal
    rois['Parahippocampus'] = np.zeros(shape[:3], dtype=bool)
    rois['Parahippocampus'][int(nx*0.25):int(nx*0.75), 
                            int(ny*0.35):int(ny*0.65), 
                            :int(nz*0.3)] = True
    
    # Thalamus: deep central
    rois['Thalamus'] = np.zeros(shape[:3], dtype=bool)
    rois['Thalamus'][int(nx*0.35):int(nx*0.65),
                     int(ny*0.4):int(ny*0.6),
                     int(nz*0.35):int(nz*0.55)] = True
    
    # V1 (control): posterior
    rois['V1_Control'] = np.zeros(shape[:3], dtype=bool)
    rois['V1_Control'][:, :int(ny*0.2), int(nz*0.3):int(nz*0.7)] = True
    
    # Motor cortex (control): superior, mid-anterior
    rois['Motor_Control'] = np.zeros(shape[:3], dtype=bool)
    rois['Motor_Control'][:, int(ny*0.5):int(ny*0.7), int(nz*0.75):] = True
    
    # Whole brain (for uniformity comparison)
    rois['Whole_Brain'] = np.ones(shape[:3], dtype=bool)
    
    return rois


def spectral_slope(signal, tr_sec=2.0):
    """Compute the spectral slope (beta) of a signal.
    
    1/f^beta:
    - beta ~ 0: white noise (random)
    - beta ~ 1: pink noise (microtubule criticality / self-organized)
    - beta ~ 2: Brownian noise (astrocyte metabolic / diffusion)
    
    Returns beta (the negative slope of log(power) vs log(freq))
    """
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
    
    # Linear regression on log-log scale
    log_f = np.log10(pos_freqs)
    log_p = np.log10(pos_power + 1e-20)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_f, log_p)
    
    # beta is the negative slope (so beta > 0 means power decreases with freq)
    beta = -slope
    
    return beta


def detect_initial_dip(bold_trace, onset_idx):
    """Detect the 'initial dip' in BOLD signal at stimulus onset.
    
    The initial dip is a brief BOLD decrease at onset (microtubule O2 consumption)
    before the main hemodynamic peak (astrocyte vasodilation at +4-6s).
    
    With TR=2s, we check:
    - TR at onset vs TR before onset (is there a dip?)
    - TR at onset vs TR at onset+2 (does it recover and overshoot?)
    
    Returns:
    - dip_magnitude: BOLD(onset-1) - BOLD(onset)  [positive = dip present]
    - recovery_ratio: BOLD(onset+2) / BOLD(onset)  [>1 = overshoot present]
    """
    n = len(bold_trace)
    
    if onset_idx < 1 or onset_idx + 3 >= n:
        return 0.0, 1.0
    
    pre = bold_trace[onset_idx - 1]
    onset = bold_trace[onset_idx]
    post2 = bold_trace[onset_idx + 2]
    post3 = bold_trace[min(onset_idx + 3, n-1)]
    
    dip_magnitude = pre - onset  # positive = dip exists
    peak = max(post2, post3)
    recovery_ratio = peak / (onset + 1e-10) if onset != 0 else 1.0
    
    return dip_magnitude, recovery_ratio


def process_subject_bold(sub_id, base_dir, run_sequence, tr_sec=2.0):
    """Process a subject with ROI-level BOLD analysis."""
    all_trial_data = []
    
    for run_type, run_num in run_sequence:
        nii_path, tsv_path = stream_data(sub_id, base_dir, run_type, run_num)
        if not nii_path:
            continue
        
        data_raw = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        data_z = np.nan_to_num(zscore(data_raw, axis=-1))
        n_trs = data_raw.shape[-1]
        shape = data_raw.shape
        
        # Brain mask
        brain_mask = np.var(data_raw, axis=-1) > 10
        
        # Define ROIs
        rois = define_rois(shape)
        
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
            n_ext = ext_end - ext_start
            if n_ext < 4:
                continue
            
            trial_data = {'Correct': correct, 'RT': rt}
            
            # === PER-ROI ANALYSIS ===
            roi_energies = {}
            roi_betas = {}
            roi_hf_powers = {}
            roi_dips = {}
            roi_recoveries = {}
            
            for roi_name, roi_mask in rois.items():
                # Intersect ROI with brain mask
                active = roi_mask & brain_mask
                n_voxels = int(np.sum(active))
                
                if n_voxels < 10:
                    roi_energies[roi_name] = np.nan
                    roi_betas[roi_name] = np.nan
                    roi_hf_powers[roi_name] = np.nan
                    roi_dips[roi_name] = np.nan
                    roi_recoveries[roi_name] = np.nan
                    continue
                
                # Mean timecourse in this ROI
                roi_ts_z = np.mean(data_z[active, ext_start:ext_end], axis=0)
                roi_ts_raw = np.mean(data_raw[active, ext_start:ext_end], axis=0)
                
                # Energy (z-scored mean abs)
                hrf_ts = data_z[active, hrf_start:hrf_end]
                roi_energies[roi_name] = np.mean(np.abs(hrf_ts))
                
                # Spectral slope (1/f^beta)
                roi_betas[roi_name] = spectral_slope(roi_ts_z, tr_sec)
                
                # HF power
                fft_vals = fft(roi_ts_z)
                freqs = fftfreq(n_ext, d=tr_sec)
                power = np.abs(fft_vals) ** 2
                pos_mask = freqs > 0
                pos_freqs = freqs[pos_mask]
                pos_power = power[pos_mask]
                total_power = np.sum(pos_power)
                hf_mask = pos_freqs > 0.15
                hf_power = np.sum(pos_power[hf_mask]) / (total_power + 1e-10) if np.any(hf_mask) else 0
                roi_hf_powers[roi_name] = hf_power
                
                # Initial dip detection
                # onset_in_ext = the index within the extended window corresponding to actual onset
                onset_in_ext = tr_onset - ext_start
                dip, recovery = detect_initial_dip(roi_ts_z, onset_in_ext)
                roi_dips[roi_name] = dip
                roi_recoveries[roi_name] = recovery
            
            # === SPATIAL UNIFORMITY TEST ===
            # If signal is external (ambient field), all ROIs would show similar energy
            # If internal (microtubules), PFC and Parahippocampus would dominate
            active_roi_energies = [v for k, v in roi_energies.items()
                                   if k != 'Whole_Brain' and not np.isnan(v)]
            if active_roi_energies:
                spatial_cv = np.std(active_roi_energies) / (np.mean(active_roi_energies) + 1e-8)
            else:
                spatial_cv = 0
            
            trial_data['Spatial_CV'] = spatial_cv
            
            for roi_name in rois:
                trial_data[f'Energy_{roi_name}'] = roi_energies.get(roi_name, np.nan)
                trial_data[f'Beta_{roi_name}'] = roi_betas.get(roi_name, np.nan)
                trial_data[f'HF_{roi_name}'] = roi_hf_powers.get(roi_name, np.nan)
                trial_data[f'Dip_{roi_name}'] = roi_dips.get(roi_name, np.nan)
                trial_data[f'Recovery_{roi_name}'] = roi_recoveries.get(roi_name, np.nan)
            
            all_trial_data.append(trial_data)
        
        if os.path.exists(nii_path):
            os.remove(nii_path)
    
    # Tag phases
    n = len(all_trial_data)
    correctness = [t['Correct'] for t in all_trial_data]
    grok_idx = n
    streak_start = n
    for i in range(n - 1, -1, -1):
        if correctness[i] == 1:
            streak_start = i
        else:
            break
    if n - streak_start >= 5:
        grok_idx = streak_start
    
    for i, t in enumerate(all_trial_data):
        t['Trial_Idx'] = i
        t['Grok_Idx'] = grok_idx
        t['Trials_To_Grok'] = i - grok_idx
        if i < grok_idx:
            t['Phase'] = 'Exploration'
        elif i == grok_idx:
            t['Phase'] = 'Grokking'
        else:
            t['Phase'] = 'Expert'
    
    return all_trial_data, grok_idx


def run_analysis():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    # The 10 validated grokking subjects from the chronological subset
    subjects = [
        "sub-201", "sub-206", "sub-212", "sub-216", "sub-217", 
        "sub-220", "sub-226", "sub-229", "sub-237", "sub-240"
    ]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0
    
    roi_names = ['PFC', 'Parahippocampus', 'Thalamus', 'V1_Control', 'Motor_Control']
    microtubule_rois = ['PFC', 'Parahippocampus', 'Thalamus']
    control_rois = ['V1_Control', 'Motor_Control']
    
    print("=" * 80, flush=True)
    print(" NEUROANATOMICAL BOLD QUANTUM SIGNATURE ANALYSIS", flush=True)
    print(" Testing: 1/f scaling, initial dip, spatial specificity, metabolic cost", flush=True)
    print("=" * 80, flush=True)
    
    for sub in subjects:
        print(f"\n{'='*80}", flush=True)
        print(f" SUBJECT: {sub}", flush=True)
        print(f"{'='*80}", flush=True)
        
        trials, grok_idx = process_subject_bold(sub, base_dir, run_sequence, tr_sec)
        if not trials or grok_idx >= len(trials):
            print(f"  No grokking. Skipping.", flush=True)
            continue
        
        explorers = [t for t in trials if t['Phase'] == 'Exploration']
        experts = [t for t in trials if t['Phase'] == 'Expert']
        grok_window = [t for t in trials if abs(t['Trials_To_Grok']) <= 2]
        
        if len(explorers) < 5 or len(experts) < 3:
            print(f"  Too few trials. Skipping.", flush=True)
            continue
        
        # ============================================================
        #  TEST 1: SPECTRAL SCALING (1/f^beta)
        # ============================================================
        print(f"\n  [1] SPECTRAL SCALING: 1/f^beta (beta~1 = microtubule, beta~2 = astrocyte)")
        print(f"  {'ROI':<20} {'Explore beta':>13} {'Grokking beta':>14} {'Expert beta':>12} {'Shift':>8}")
        print(f"  {'-'*68}")
        
        for roi in roi_names:
            col = f'Beta_{roi}'
            explore_beta = np.nanmean([t[col] for t in explorers])
            grok_beta = np.nanmean([t[col] for t in grok_window])
            expert_beta = np.nanmean([t[col] for t in experts])
            shift = grok_beta - explore_beta
            flag = ""
            if abs(shift) > 0.3 and grok_beta < explore_beta:
                flag = " << PINK"  # shifted toward 1/f
            elif abs(shift) > 0.3 and grok_beta > explore_beta:
                flag = " >> BROWN"  # shifted toward 1/f^2
            print(f"  {roi:<20} {explore_beta:>13.3f} {grok_beta:>14.3f} {expert_beta:>12.3f} {shift:>8.3f}{flag}")
        
        # ============================================================
        #  TEST 2: INITIAL DIP DETECTION
        # ============================================================
        print(f"\n  [2] INITIAL DIP: Microtubule O2 consumption before astrocyte vasodilation")
        print(f"  {'ROI':<20} {'Explore Dip':>12} {'Grok Dip':>10} {'Expert Dip':>11} {'Grok Recovery':>14}")
        print(f"  {'-'*70}")
        
        for roi in roi_names:
            col_dip = f'Dip_{roi}'
            col_rec = f'Recovery_{roi}'
            explore_dip = np.nanmean([t[col_dip] for t in explorers])
            grok_dip = np.nanmean([t[col_dip] for t in grok_window])
            expert_dip = np.nanmean([t[col_dip] for t in experts])
            grok_recovery = np.nanmean([t[col_rec] for t in grok_window])
            flag = ""
            # Positive dip + high recovery = classic microtubule signature
            if grok_dip > explore_dip and grok_recovery > 1.1:
                flag = " * DIP+RECOVERY"
            print(f"  {roi:<20} {explore_dip:>12.4f} {grok_dip:>10.4f} {expert_dip:>11.4f} {grok_recovery:>14.3f}{flag}")
        
        # ============================================================
        #  TEST 3: SPATIAL SPECIFICITY (Internal vs External Origin)
        # ============================================================
        print(f"\n  [3] SPATIAL SPECIFICITY: Internal (localized) vs External (uniform)")
        print(f"      High Spatial CV = localized (microtubule) | Low CV = uniform (ambient)")
        
        explore_cv = np.nanmean([t['Spatial_CV'] for t in explorers])
        grok_cv = np.nanmean([t['Spatial_CV'] for t in grok_window])
        expert_cv = np.nanmean([t['Spatial_CV'] for t in experts])
        
        print(f"      Exploration Spatial CV: {explore_cv:.4f}")
        print(f"      Grokking    Spatial CV: {grok_cv:.4f}")
        print(f"      Expert      Spatial CV: {expert_cv:.4f}")
        
        if grok_cv > explore_cv:
            print(f"      >>> GROKKING IS MORE SPATIALLY LOCALIZED THAN EXPLORATION")
            print(f"      This supports INTERNAL origin (microtubule-dense regions dominating)")
        else:
            print(f"      Grokking is not more localized than exploration")
        
        # Which ROIs dominate at grokking?
        print(f"\n      ROI Energy at Grokking (which regions are most active?):")
        roi_grok_energies = {}
        for roi in roi_names:
            col = f'Energy_{roi}'
            e = np.nanmean([t[col] for t in grok_window])
            roi_grok_energies[roi] = e
            marker = " [MT-dense]" if roi in microtubule_rois else " [Control]"
            print(f"        {roi:<20} {e:.4f}{marker}")
        
        # Are microtubule-dense ROIs more active than controls at grokking?
        mt_energy = np.mean([roi_grok_energies[r] for r in microtubule_rois if not np.isnan(roi_grok_energies[r])])
        ctrl_energy = np.mean([roi_grok_energies[r] for r in control_rois if not np.isnan(roi_grok_energies[r])])
        ratio = mt_energy / (ctrl_energy + 1e-10)
        print(f"\n      Microtubule-dense ROI mean energy: {mt_energy:.4f}")
        print(f"      Control ROI mean energy:           {ctrl_energy:.4f}")
        print(f"      Ratio (MT/Control):                {ratio:.3f}")
        if ratio > 1.05:
            print(f"      >>> MICROTUBULE-DENSE REGIONS MORE ACTIVE AT GROKKING")
        
        # ============================================================
        #  TEST 4: METABOLIC COST (Does energy in = energy out?)
        # ============================================================
        print(f"\n  [4] METABOLIC COST: Does HF power correlate with metabolic energy?")
        print(f"      If correlated => internally fueled (microtubules burning ATP)")
        print(f"      If NOT correlated => energy from elsewhere (QPC via bi-twistor)")
        
        for roi in microtubule_rois:
            col_e = f'Energy_{roi}'
            col_hf = f'HF_{roi}'
            
            # Gather all trials with valid data
            es = [t[col_e] for t in trials if not np.isnan(t[col_e]) and not np.isnan(t[col_hf])]
            hfs = [t[col_hf] for t in trials if not np.isnan(t[col_e]) and not np.isnan(t[col_hf])]
            
            if len(es) > 10:
                corr, p = pearsonr(es, hfs)
                flag = ""
                if corr > 0.3 and p < 0.05:
                    flag = " => INTERNALLY FUELED"
                elif abs(corr) < 0.1 or p > 0.1:
                    flag = " => NO METABOLIC LINK (external?)"
                print(f"      {roi:<20} r={corr:.3f}, p={p:.4f}{flag}")
        
        # ============================================================
        #  TEST 5: GROKKING-SPECIFIC REGIONAL ACTIVATION SHIFT
        # ============================================================
        print(f"\n  [5] REGIONAL ACTIVATION SHIFT: Explore -> Grokking -> Expert")
        print(f"  {'ROI':<20} {'Explore E':>10} {'Grok E':>10} {'Expert E':>10} {'Grok/Explore':>13} {'Expert/Explore':>15}")
        print(f"  {'-'*80}")
        
        for roi in roi_names:
            col = f'Energy_{roi}'
            exp_e = np.nanmean([t[col] for t in explorers])
            grk_e = np.nanmean([t[col] for t in grok_window])
            xpt_e = np.nanmean([t[col] for t in experts])
            ge_ratio = grk_e / (exp_e + 1e-10)
            xe_ratio = xpt_e / (exp_e + 1e-10)
            marker = ""
            if ge_ratio > 1.05:
                marker = " UP"
            elif ge_ratio < 0.95:
                marker = " DOWN"
            print(f"  {roi:<20} {exp_e:>10.4f} {grk_e:>10.4f} {xpt_e:>10.4f} {ge_ratio:>13.3f} {xe_ratio:>15.3f}{marker}")
        
        print(flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_analysis()
