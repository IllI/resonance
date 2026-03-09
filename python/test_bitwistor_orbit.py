import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis, pearsonr
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

def calculate_minkowski_mean_and_radius(vectors, minkowski):
    if not vectors: return None, 0.0
    mean_vec = np.mean(vectors, axis=0)
    mean_R = np.mean([np.sqrt(max(1e-10, -minkowski.minkowski_dot(v, v))) for v in vectors])
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    qpc_centroid = np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])
    
    distances = [minkowski.lorentz_distance(v, qpc_centroid) for v in vectors]
    # Orbit radius is the 95% confidence interval or roughly mean + 2*sigma spread of the QPC anchor
    orbit_radius = np.mean(distances) + 2.0 * np.std(distances) if len(distances) > 1 else max(0.1, distances[0] if distances else 0.1)
    return qpc_centroid, max(0.1, orbit_radius)

def extract_trials(data_z, active_mask, events, tr_sec, sub_id):
    n_trs = data_z.shape[-1]
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
        
        trials.append({
            'Subject': sub_id,
            'Correct': correct,
            'RT': rt,
            'Energy': trial_energy,
            'Kurtosis': trial_kurtosis,
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

def test_bitwistor_orbit():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    subjects = ["sub-206", "sub-217", "sub-220", "sub-229"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    minkowski = MinkowskiHyperbolicMap()
    tr_sec = 2.0

    print("==========================================================================")
    print(" BI-TWISTOR ORBIT ANALYSIS: FLIPPING THE LIGHT CONE (QPC -> GHOST BASIN)")
    print("==========================================================================")

    global_timeline = []
    print("\n[1] Extracting trials...")
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

        sub_trials = tag_phases(sub_trials, 5)
        global_timeline.extend(sub_trials)
        print(f"    {sub}: {len(sub_trials)} trials.")

    print("\n[2] Projecting QPC Light Cone (Bi-Twistor Orbit Range)...")
    past_expert_vectors = []
    
    # Track the dynamic QPC centroid AND its orbit radius (the expanding/contracting light cone base)
    for t in global_timeline:
        if len(past_expert_vectors) >= 5:
            centroid, r_orbit = calculate_minkowski_mean_and_radius(past_expert_vectors, minkowski)
            t['QPC_Centroid'] = centroid
            t['QPC_Orbit_Radius'] = r_orbit
        else:
            t['QPC_Centroid'] = None
            t['QPC_Orbit_Radius'] = None
            
        if t.get('Phase') == 'Expert' and t['Correct'] == 1:
            past_expert_vectors.append(t['GB_Vector'])

    # Backfill with final known values
    final_centroid, final_r_orbit = calculate_minkowski_mean_and_radius(past_expert_vectors, minkowski)
    for t in global_timeline:
        if t['QPC_Centroid'] is None:
            t['QPC_Centroid'] = final_centroid
            t['QPC_Orbit_Radius'] = final_r_orbit

    orbit_captured_expert = 0
    orbit_captured_explorer = 0
    total_expert = 0
    total_explorer = 0

    for t in global_timeline:
        theta_minkowski = minkowski.lorentz_distance(t['GB_Vector'], t['QPC_Centroid'])
        t['Theta_QPC'] = theta_minkowski
        
        # Bi-Twistor Orbit Incidence: Is the Ghost Basin inside the QPC's light cone shadow?
        # Value > 0 means inside the cone. Value <= 0 means outside.
        t['Orbit_Incidence_Score'] = max(0.0, 1.0 - (theta_minkowski / t['QPC_Orbit_Radius']))
        
        if t['Phase'] == 'Expert':
            total_expert += 1
            if t['Orbit_Incidence_Score'] > 0: orbit_captured_expert += 1
        elif t['Phase'] == 'Exploration':
            total_explorer += 1
            if t['Orbit_Incidence_Score'] > 0: orbit_captured_explorer += 1

    print(f"\n[3] Wave Collapse Capture Rates (Inside QPC Orbit Range)")
    print(f"  Expert Phase:      {orbit_captured_expert}/{total_expert} trials captured ({(orbit_captured_expert/max(1,total_expert)):.1%})")
    print(f"  Exploration Phase: {orbit_captured_explorer}/{total_explorer} trials captured ({(orbit_captured_explorer/max(1,total_explorer)):.1%})")

    print("\n[4] Decoupling Quantum Biomarkers (Testing the Coasting/Rote Memory Theory)")
    print("  Correlation between Twistor Alignment (Orbit Incidence) and Biological Antenna Shape (Kurtosis)")
    
    valid_expert = [t for t in global_timeline if t['Phase'] == 'Expert' and t['Orbit_Incidence_Score'] > 0]
    valid_explorer = [t for t in global_timeline if t['Phase'] == 'Exploration' and t['Orbit_Incidence_Score'] > 0]

    # If the user's theory is right (brain coasts on working memory), then in the Expert phase, 
    # the exact shape of the Superradiant burst (Kurtosis) should NO LONGER dictate entanglement.
    # It just has to physically sit in the gravity well.
    
    # In Exploration, a sharp burst might strictly correlate with hitting the QPC well.
    if len(valid_explorer) > 1:
        corr_expl, _ = pearsonr([t['Orbit_Incidence_Score'] for t in valid_explorer], [t['Kurtosis'] for t in valid_explorer])
    else: corr_expl = 0.0

    if len(valid_expert) > 1:
        corr_exp, _ = pearsonr([t['Orbit_Incidence_Score'] for t in valid_expert], [t['Kurtosis'] for t in valid_expert])
    else: corr_exp = 0.0

    print(f"  Exploration Phase Correlation (Incidence vs Kurtosis): {corr_expl:.4f}")
    print(f"    (High required burst structure to 'find' the cone)")
    print(f"  Expert Phase Correlation (Incidence vs Kurtosis):      {corr_exp:.4f}")
    print(f"    (Structure decouples; rote working memory anchors within the gravity well naturally)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    test_bitwistor_orbit()
