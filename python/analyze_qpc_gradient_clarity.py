"""
analyze_qpc_gradient_clarity.py
===============================
Tests whether subjects truly grokked the CONCEPT (Badoon) or just
memorized specific STIMULI (dot patterns they'd seen before).

Key idea: If a subject truly embedded the Badoon concept in working memory
(quantum alignment), their Ghost Basin should point at the QPC regardless
of how far the specific dot pattern is from the category prototype.

If they only learned to recognize specific dot patterns, their QPC alignment
will degrade when presented with DISTANT exemplars (high prot1distance).

We measure "QPC Gradient Clarity" as the stability of the Orbit Incidence
Score across different stimulus distances within the Expert phase.

Uses existing tools:
- bitwistor_orbit.py (Minkowski projection, OIS calculation)
- ElectromagneticSpectralMap (PUC corrections)
- dlinoss_shared_hallucination_mapper.py (data streaming pattern)
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, pearsonr, ttest_ind
import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.append(os.path.dirname(__file__))
from bitwistor_orbit import (
    MinkowskiHyperbolicMap,
    calculate_qpc_centroid,
    calculate_orbit_radius,
    orbit_incidence_score,
    calc_twistor_coherence,
)


def stream_data(sub_id, base_dir, run_type, run_num):
    """Stream a single run's events + BOLD from S3."""
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


def extract_trial_with_stimulus(data_z, active_mask, events, tr_sec, minkowski):
    """Extract per-trial Ghost Basin vectors, behavioral, AND stimulus metadata."""
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

        # Stimulus metadata
        prot1dist = row.get('prot1distance', np.nan)
        trial_type = row.get('trial_type', -1)
        if pd.isna(prot1dist):
            prot1dist = np.nan
        if pd.isna(trial_type):
            trial_type = -1
        else:
            trial_type = int(trial_type)

        tr_onset = int(row['onset'] / tr_sec)
        start_tr = min(tr_onset + 2, n_trs - 1)
        end_tr = min(tr_onset + 5, n_trs)
        if start_tr >= end_tr:
            continue

        block = data_z[active_mask, start_tr:end_tr]
        if block.size == 0:
            continue

        # Energy and spatial metrics
        energy = np.mean(np.abs(block)) * 10.0
        peak_tr = np.argmax(np.mean(np.abs(block), axis=0))
        voxels_at_peak = block[:, peak_tr]
        spatial_kurtosis_val = kurtosis(voxels_at_peak)

        # Hurst proxy (autocorrelation of mean signal)
        mean_ts = np.mean(block, axis=0)
        if len(mean_ts) > 1:
            hurst_proxy = np.abs(np.corrcoef(mean_ts[:-1], mean_ts[1:])[0, 1])
        else:
            hurst_proxy = 0.5
        hurst_proxy = max(0.01, min(0.99, hurst_proxy))

        # PSD beta proxy (variance ratio)
        psd_beta_proxy = max(0.01, np.std(mean_ts) / (np.mean(np.abs(mean_ts)) + 1e-8))

        # Ghost Basin projection
        vec = minkowski.project_ghost_basin(
            hurst_proxy, psd_beta_proxy, abs(spatial_kurtosis_val), energy
        )

        # Burst sharpness
        mean_amp = np.mean(np.abs(block))
        peak_amp = np.mean(np.abs(voxels_at_peak))
        s_trial = peak_amp / (mean_amp + 1e-5)

        trials.append({
            'Correct': correct,
            'RT': rt,
            'TrialType': trial_type,
            'Prot1Distance': prot1dist,
            'Energy': energy,
            'Kurtosis': spatial_kurtosis_val,
            'Hurst': hurst_proxy,
            'PSD_Beta': psd_beta_proxy,
            'S_Trial': s_trial,
            'N_Active': n_active,
            'GhostBasinVec': vec,
        })
    return trials


def tag_phases(trial_list, streak_threshold=5):
    """Tag learning phases based on correctness streaks."""
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
        if i < grok_idx:
            trial['Phase'] = 'Exploration'
        elif i == grok_idx:
            trial['Phase'] = 'Grokking'
        else:
            trial['Phase'] = 'Expert'
    return trial_list


def analyze_gradient_clarity():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0
    minkowski = MinkowskiHyperbolicMap()

    print("=" * 72)
    print(" QPC GRADIENT CLARITY: CONCEPT GROKKING vs STIMULUS MEMORIZATION")
    print("=" * 72)

    all_data = {}

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

            trials = extract_trial_with_stimulus(data_z, active_mask, events, tr_sec, minkowski)
            sub_trials.extend(trials)

            if os.path.exists(nii_path):
                os.remove(nii_path)

        sub_trials = tag_phases(sub_trials, 5)
        all_data[sub] = sub_trials

    # ================================================================== #
    #  STEP 1: Build QPC centroid from Expert trials (all subjects)
    # ================================================================== #
    expert_vecs = []
    for sub, trials in all_data.items():
        for t in trials:
            if t['Phase'] == 'Expert':
                expert_vecs.append(t['GhostBasinVec'])

    qpc_centroid = calculate_qpc_centroid(expert_vecs, minkowski)
    orbit_radius = calculate_orbit_radius(expert_vecs, qpc_centroid, minkowski)

    print(f"\n[1] QPC Centroid (from {len(expert_vecs)} Expert trials):")
    print(f"    Vector: {qpc_centroid}")
    print(f"    Orbit Radius: {orbit_radius:.4f}")

    # ================================================================== #
    #  STEP 2: Calculate OIS for every trial, attach stimulus distance
    # ================================================================== #
    for sub, trials in all_data.items():
        for t in trials:
            d = minkowski.lorentz_distance(t['GhostBasinVec'], qpc_centroid)
            t['Lorentz_Distance'] = d
            t['OIS'] = orbit_incidence_score(d, orbit_radius)

            # TCS (forward cone, for comparison)
            rt_expert = np.nanmedian([
                tt['RT'] for tt in trials if tt['Phase'] == 'Expert' and not np.isnan(tt['RT'])
            ])
            tcs, peep, phase = calc_twistor_coherence(
                d, t['RT'] if not np.isnan(t['RT']) else rt_expert, rt_expert,
                t['Kurtosis'], t['N_Active'], t['S_Trial']
            )
            t['TCS'] = tcs
            t['Peephole'] = peep
            t['PhaseFactor'] = phase

    # ================================================================== #
    #  STEP 3: PER-SUBJECT GRADIENT CLARITY
    #  Does OIS depend on stimulus distance (prot1distance)?
    #  If YES: subject memorized specific dots
    #  If NO: subject grokked the abstract concept
    # ================================================================== #
    print("\n" + "=" * 72)
    print(" [2] PER-SUBJECT QPC GRADIENT CLARITY")
    print("     (Correlation of OIS with Stimulus Distance in Expert Phase)")
    print("=" * 72)

    for sub, trials in all_data.items():
        expert = [t for t in trials if t['Phase'] == 'Expert' and not np.isnan(t['Prot1Distance'])]
        explore = [t for t in trials if t['Phase'] == 'Exploration' and not np.isnan(t['Prot1Distance'])]

        if len(expert) < 5:
            print(f"\n  {sub}: Too few Expert trials with stimulus distance ({len(expert)})")
            continue

        # Expert phase: OIS vs stimulus distance
        ois_vals = [t['OIS'] for t in expert]
        dist_vals = [t['Prot1Distance'] for t in expert]

        if np.std(dist_vals) > 0 and np.std(ois_vals) > 0:
            r_expert, p_expert = pearsonr(ois_vals, dist_vals)
        else:
            r_expert, p_expert = 0.0, 1.0

        # Exploration phase: OIS vs stimulus distance
        if len(explore) >= 5:
            ois_explore = [t['OIS'] for t in explore]
            dist_explore = [t['Prot1Distance'] for t in explore]
            if np.std(dist_explore) > 0 and np.std(ois_explore) > 0:
                r_explore, p_explore = pearsonr(ois_explore, dist_explore)
            else:
                r_explore, p_explore = 0.0, 1.0
        else:
            r_explore, p_explore = np.nan, np.nan

        print(f"\n  {sub} (Expert: {len(expert)} trials, Explore: {len(explore)} trials)")
        print(f"    Expert  Phase:  r(OIS, Prot1Dist) = {r_expert:+.4f}  p = {p_expert:.4f}")
        print(f"    Explore Phase:  r(OIS, Prot1Dist) = {r_explore:+.4f}  p = {p_explore:.4f}")

        # Interpretation
        if abs(r_expert) < 0.15 and p_expert > 0.05:
            print(f"    >> CONCEPT GROKKER: OIS is independent of stimulus distance")
            print(f"       The Ghost Basin points at QPC regardless of which dots were shown")
        elif r_expert < -0.15:
            print(f"    >> CONCEPT GROKKER (inverse): Distant stimuli trigger STRONGER alignment")
            print(f"       Suggests the abstraction layer is more engaged for novel exemplars")
        else:
            print(f"    >> STIMULUS MEMORIZER: OIS degrades with stimulus distance")
            print(f"       Alignment to QPC depends on seeing familiar dot patterns")

    # ================================================================== #
    #  STEP 4: SPLIT EXPERT TRIALS BY STIMULUS NOVELTY
    #  High prot1distance = novel exemplar (far from memorized prototype)
    #  Low prot1distance = familiar exemplar (close to what was seen before)
    # ================================================================== #
    print("\n" + "=" * 72)
    print(" [3] NOVEL vs FAMILIAR STIMULI IN EXPERT PHASE")
    print("     (Does QPC alignment hold for stimuli the brain hasn't memorized?)")
    print("=" * 72)

    all_expert_with_dist = []
    for sub, trials in all_data.items():
        for t in trials:
            if t['Phase'] == 'Expert' and not np.isnan(t['Prot1Distance']):
                t['Subject'] = sub
                all_expert_with_dist.append(t)

    if all_expert_with_dist:
        dists = [t['Prot1Distance'] for t in all_expert_with_dist]
        median_dist = np.median(dists)

        near = [t for t in all_expert_with_dist if t['Prot1Distance'] <= median_dist]
        far = [t for t in all_expert_with_dist if t['Prot1Distance'] > median_dist]

        ois_near = [t['OIS'] for t in near]
        ois_far = [t['OIS'] for t in far]

        print(f"\n  Median stimulus distance: {median_dist:.2f}")
        print(f"  Near trials (prot1dist <= {median_dist:.2f}): N={len(near)}")
        print(f"  Far  trials (prot1dist >  {median_dist:.2f}): N={len(far)}")

        print(f"\n  {'Metric':<30} {'Near (familiar)':>15} {'Far (novel)':>15}")
        print(f"  {'-'*60}")
        print(f"  {'Mean OIS':<30} {np.mean(ois_near):>15.4f} {np.mean(ois_far):>15.4f}")
        print(f"  {'Std OIS':<30} {np.std(ois_near):>15.4f} {np.std(ois_far):>15.4f}")
        print(f"  {'Mean Lorentz Dist':<30} {np.mean([t['Lorentz_Distance'] for t in near]):>15.4f} {np.mean([t['Lorentz_Distance'] for t in far]):>15.4f}")
        print(f"  {'Mean TCS':<30} {np.mean([t['TCS'] for t in near]):>15.4f} {np.mean([t['TCS'] for t in far]):>15.4f}")
        print(f"  {'Mean RT':<30} {np.nanmean([t['RT'] for t in near if not np.isnan(t['RT'])]):>15.3f} {np.nanmean([t['RT'] for t in far if not np.isnan(t['RT'])]):>15.3f}")
        print(f"  {'Mean Energy':<30} {np.mean([t['Energy'] for t in near]):>15.4f} {np.mean([t['Energy'] for t in far]):>15.4f}")
        print(f"  {'Accuracy':<30} {np.mean([t['Correct'] for t in near]):>15.1%} {np.mean([t['Correct'] for t in far]):>15.1%}")

        if len(ois_near) >= 3 and len(ois_far) >= 3:
            t_stat, p_val = ttest_ind(ois_near, ois_far)
            print(f"\n  t-test (OIS: Near vs Far): t={t_stat:.3f}, p={p_val:.4f}")
            if p_val > 0.05:
                print(f"  >> NO SIGNIFICANT DIFFERENCE: QPC alignment is stimulus-invariant")
                print(f"     This supports CONCEPT GROKKING (not stimulus memorization)")
            else:
                print(f"  >> SIGNIFICANT DIFFERENCE: QPC alignment depends on stimulus")
                print(f"     This suggests some degree of STIMULUS MEMORIZATION")

    # ================================================================== #
    #  STEP 5: PER-SUBJECT BREAKDOWN
    # ================================================================== #
    print("\n" + "=" * 72)
    print(" [4] PER-SUBJECT: CONCEPT vs STIMULUS GROKKING VERDICT")
    print("=" * 72)

    for sub, trials in all_data.items():
        expert = [t for t in trials if t['Phase'] == 'Expert' and not np.isnan(t['Prot1Distance'])]
        if len(expert) < 5:
            print(f"\n  {sub}: INSUFFICIENT DATA")
            continue

        dists = [t['Prot1Distance'] for t in expert]
        med = np.median(dists)
        near = [t for t in expert if t['Prot1Distance'] <= med]
        far = [t for t in expert if t['Prot1Distance'] > med]

        ois_near = [t['OIS'] for t in near] if near else [0]
        ois_far = [t['OIS'] for t in far] if far else [0]

        near_ois_mean = np.mean(ois_near)
        far_ois_mean = np.mean(ois_far)
        ois_drop = near_ois_mean - far_ois_mean

        # Gradient clarity = how flat OIS is across stimulus distances
        # Perfect concept grokker: OIS doesn't depend on stimulus → gradient clarity ≈ 0
        # Stimulus memorizer: OIS decays with distance → gradient clarity >> 0
        ois_vals = [t['OIS'] for t in expert]
        dist_vals = [t['Prot1Distance'] for t in expert]
        if np.std(dist_vals) > 0 and np.std(ois_vals) > 0:
            gradient_clarity = abs(pearsonr(ois_vals, dist_vals)[0])
        else:
            gradient_clarity = 0.0

        print(f"\n  {sub}:")
        print(f"    Expert trials: {len(expert)}")
        print(f"    OIS (Near): {near_ois_mean:.4f}  |  OIS (Far): {far_ois_mean:.4f}  |  Drop: {ois_drop:+.4f}")
        print(f"    Gradient Clarity (|r|): {gradient_clarity:.4f}")

        # Classify
        if gradient_clarity < 0.15:
            print(f"    VERDICT: CONCEPT GROKKER")
            print(f"    The brain's quantum compass is stable regardless of the specific stimulus")
        elif gradient_clarity < 0.30:
            print(f"    VERDICT: MIXED (partial concept, partial stimulus)")
        else:
            print(f"    VERDICT: STIMULUS MEMORIZER")
            print(f"    The brain's alignment to QPC depends on recognizing specific dot patterns")

    # ================================================================== #
    #  STEP 6: THE CRITICAL QUESTION - Does the experiment even test concepts?
    # ================================================================== #
    print("\n" + "=" * 72)
    print(" [5] EXPERIMENTAL DESIGN LIMITATION CHECK")
    print("=" * 72)

    # Check stimulus diversity in Expert phase
    for sub, trials in all_data.items():
        expert = [t for t in trials if t['Phase'] == 'Expert' and not np.isnan(t['Prot1Distance'])]
        if expert:
            dists = [t['Prot1Distance'] for t in expert]
            print(f"  {sub} Expert stimulus diversity:")
            print(f"    prot1dist range: [{min(dists):.2f}, {max(dists):.2f}]")
            print(f"    prot1dist std:   {np.std(dists):.2f}")
            print(f"    unique values:   {len(set([round(d, 2) for d in dists]))}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    analyze_gradient_clarity()
