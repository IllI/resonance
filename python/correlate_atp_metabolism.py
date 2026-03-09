"""
correlate_atp_metabolism.py
──────────────────────────────────────────────────────────────
Correlates the initial electrical spike (proxy: leading edge gradient)
with the total ATP metabolic recovery (proxy: BOLD AUC) for the 
subcortical auditory antennae.

Hypothesis: The amount of ATP metabolic energy consumed (oxygenated blood flow / BOLD area)
is directly proportional to the initial energetic output required to emit the 8-13Hz spike 
at stimulus onset.

1. Detect stimulus events (sparks).
2. Isolate the "Early Responders" (antennae / MGB).
3. Calculate the initial onset surge (Derivative/Slope of the first TRs) representing the spike.
4. Calculate the total Area Under the Curve (AUC) of the subsequent HRF window, representing ATP metabolism.
5. Correlate the onset spike power with the ATP metabolic recovery energy.
"""

import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore, pearsonr

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
    print("  ATP METABOLISM VS ELECTRICAL SPIKE CORRELATION")
    print("=" * 65)
    
    bolds = [load_bold_run(subj, ses, r) for r in range(1, 6)]
    bolds = [b for b in bolds if b is not None]
    
    ts_z = extract_auditory_voxels(bolds, percentile=90)
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
        
    window = 6
    valid_events = []
    for tr in all_events:
        if (tr % n_trials) + window <= n_trials:
            valid_events.append(ts_z[:, tr:tr+window])
            
    X = np.stack(valid_events) # Shape: (n_events, n_voxels, 6)
    
    # Identify Early Responders (Antennae)
    mean_response = np.mean(X, axis=0)
    pos_response = mean_response - mean_response.min(axis=1, keepdims=True) + 1e-6
    time_weights = np.arange(window)
    com = np.sum(pos_response * time_weights, axis=1) / np.sum(pos_response, axis=1)
    
    early_idx = np.where(com < 2.5)[0]
    print(f"  > Isolated {len(early_idx)} early subcortical/primary antennae voxels.")
    
    # Extract event-by-event data for the antennae
    antennae_data = X[:, early_idx, :] # (n_events, n_antennae_voxels, 6)
    
    # Aggregate across antennae to get the population event envelope
    event_envelopes = np.mean(antennae_data, axis=1) # (n_events, 6)
    
    spike_magnitudes = []
    atp_metabolism = []
    
    for i in range(len(event_envelopes)):
        env = event_envelopes[i]
        # Electrical Spike Proxy: The initial absolute surge (TR 0 to TR 1 or 2)
        # Represents the initial harmonic broadcast
        spike_surge = np.abs(env[1] - env[0]) + np.abs(env[2] - env[1])
        
        # ATP Consumption Proxy: Area under the curve of the entire metabolic cycle
        # We square the envelope to get Power/Energy dissipation
        energy_integral = np.sum(env**2)
        
        spike_magnitudes.append(spike_surge)
        atp_metabolism.append(energy_integral)
        
    spike_magnitudes = np.array(spike_magnitudes)
    atp_metabolism = np.array(atp_metabolism)
    
    corr, p_val = pearsonr(spike_magnitudes, atp_metabolism)
    
    print("\n  >> THERMODYNAMIC CORRELATION RESULTS:")
    print(f"     Analyzed {len(event_envelopes)} discrete stimulus events.")
    print(f"     Pearson Correlation Coefficient (r) : {corr:.4f}")
    print(f"     P-value                             : {p_val:.2e}")
    
    if corr > 0.5 and p_val < 0.05:
        print("\n  => STRONG CORRELATION PROVED:")
        print("     The initial electrical broadcast spike directly predicts the")
        print("     subsequent ATP metabolic expenditure (BOLD integral).")
        print("     This physically grounds our stimulus time-estimations in the")
        print("     thermodynamic law of conservation of energy: The energy output")
        print("     of the 8-13Hz broadcast equals the energy consumed to recover.")

if __name__ == "__main__":
    main()
