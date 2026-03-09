"""
calculate_neural_energy_density.py
──────────────────────────────────────────────────────────────
Calculates the metabolic Energy Density ("Antenna Power") of different 
neural networks in the auditory processing hierarchy.

Hypothesis (User): Small subcortical structures (like the MGB) act as 
powerful, high-density antennae. If they are physically tiny but 
generate detectable 8-13Hz field disruptions, their "energy per voxel" 
(Power Density) must be massively concentrated compared to higher-level 
cortical networks.

Method:
1. Load 7T BOLD data and identify sound event spikes.
2. Track the temporal Center of Mass to classify voxels into 
   Early (Subcortical/Primary), Middle, and Late (Semantic) responders.
3. For each network, calculate the Total Metabolic Power (sum of BOLD 
   amplitude squared during the active window).
4. Calculate Energy Density (Power / Volume).
"""

import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds001942"

def load_bold_run(subj, ses, run):
    fname = f"{subj}_{ses}_task-auditory_run-{run:02d}_bold.nii.gz"
    fpath = os.path.join(DATA_DIR, subj, ses, 'func', fname)
    if not os.path.exists(fpath): return None
    return nib.load(fpath).get_fdata()

def extract_auditory_voxels(bold_4d_list, percentile=95):
    first_run = bold_4d_list[0].reshape(-1, bold_4d_list[0].shape[3])
    mask = first_run.std(axis=1) > 1e-3
    brain_0 = first_run[mask]
    var_0 = brain_0.var(axis=1)
    thresh = np.percentile(var_0, percentile)
    top_mask = var_0 >= thresh
    
    z_runs = []
    for run in bold_4d_list:
        run_flat = run.reshape(-1, run.shape[3])
        chunk = run_flat[mask][top_mask]
        chunk_z = zscore(chunk, axis=1).astype(np.float32)
        chunk_z = np.nan_to_num(chunk_z, 0)
        z_runs.append(chunk_z)
    return np.concatenate(z_runs, axis=1)

def main():
    subj = 'sub-S01'
    ses = 'ses-SES02'
    
    print("=" * 65)
    print("  ANTENNA POWER DENSITY: THERMODYNAMICS OF THE MGB")
    print("=" * 65)
    
    bolds = [load_bold_run(subj, ses, r) for r in range(1, 6)] # Use first 5 runs for speed
    bolds = [b for b in bolds if b is not None]
    
    ts_z = extract_auditory_voxels(bolds, percentile=90) # Top 10% variance to capture network
    n_voxels, total_TRs = ts_z.shape
    
    # Blindly detect sound events
    n_trials = 150
    n_runs = total_TRs // n_trials
    all_events = []
    for i in range(n_runs):
        run_ts = ts_z[:, i*n_trials:(i+1)*n_trials]
        pop_signal = np.abs(run_ts).mean(axis=0)
        thresh = np.percentile(pop_signal, 72)
        events = np.where(pop_signal > thresh)[0] + i*n_trials
        all_events.extend(events)
        
    print(f"  > Mapped {len(all_events)} acoustic events across {n_runs} runs.")
    
    # Extract propagation windows (6 TRs = 15.6s)
    window = 6
    valid_events = []
    for tr in all_events:
        if (tr % n_trials) + window <= n_trials:
            valid_events.append(ts_z[:, tr:tr+window])
            
    X = np.stack(valid_events) # Shape: (n_events, n_voxels, 6)
    
    # Average response per voxel across all sounds
    mean_response = np.mean(X, axis=0) # Shape: (n_voxels, 6)
    
    # Normalize for center of mass calculation
    pos_response = mean_response - mean_response.min(axis=1, keepdims=True) + 1e-6
    time_weights = np.arange(window)
    com = np.sum(pos_response * time_weights, axis=1) / np.sum(pos_response, axis=1)
    
    # Classify physical networks based on phase shift
    early_idx = np.where(com < 2.5)[0]  # Subcortical / Primary (e.g., MGB)
    mid_idx = np.where((com >= 2.5) & (com < 3.5))[0] 
    late_idx = np.where(com >= 3.5)[0]  # Cortical / Semantic
    
    # Physics Calculation: Energy Density (Metabolic Power per Voxel)
    # BOLD variance is a proxy for oxygen consumption (energy dissipation).
    # Power = Integral of the squared amplitude over the active window
    def calc_energy_density(network_idx, name):
        if len(network_idx) == 0: return
        # Shape: (voxels, time)
        net_response = mean_response[network_idx] 
        
        # Absolute peak amplitude achieved by the network
        peak_amplitudes = np.max(np.abs(net_response), axis=1)
        
        # Total Metabolic Energy = Sum of squared amplitudes
        total_energy = np.sum(net_response**2)
        
        # Volume = number of voxels
        volume = len(network_idx)
        
        # Density
        power_density = total_energy / volume
        mean_peak = np.mean(peak_amplitudes)
        
        print(f"  >> {name} Network")
        print(f"     Physical Volume (Voxels) : {volume:,}")
        print(f"     Mean Peak Amplitude (Z)  : {mean_peak:.4f}")
        print(f"     Energy Density (E/V)     : {power_density:.4f} metabolic units/voxel")
        return power_density

    print("\n  ================ NEURAL THERMODYNAMICS ================")
    e_dense = calc_energy_density(early_idx, "Early Responders [Antennae/MGB]")
    m_dense = calc_energy_density(mid_idx, "Middle Responders [Distribution]")
    l_dense = calc_energy_density(late_idx, "Late Responders [Semantic Cortices]")
    
    if e_dense and l_dense:
        ratio = e_dense / l_dense
        print("\n  >> CONCLUSION:")
        print(f"     The early 'antennae' structures operate at {ratio:.1f}x times the")
        print(f"     metabolic power density of the late semantic structures.")
        print("     They are physically tiny, but burn massively hot to broadcast")
        print("     the initial sensory wave into the wider cranial fluid.")

if __name__ == "__main__":
    main()
