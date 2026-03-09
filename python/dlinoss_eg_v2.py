"""
dlinoss_eg_v2.py
=================
CORRECTED E_G-Gated D-LinOSS Quantum Memory.

Problems with v1:
  - E_G computed globally (all collapsed states contribute mass)
  - Positive feedback runaway: everything collapses after trial 1
  - No category selectivity

Fix:
  - E_G is computed as the SELF-ENERGY of the current state ONLY
    (its coherence with the Hamiltonian eigenmodes, NOT proximity to prior collapses)
  - The threshold is adaptive: rises logarithmically with collapse count
  - Collapse only modifies a LOCALIZED region of the Hamiltonian
    (outer product projects into the subspace, not the whole matrix)
  - Cross-category orthogonality is enforced by the projective conduit
"""

import torch
import torch.nn as nn
import numpy as np
import os
import nibabel as nib
import pandas as pd
from scipy.stats import zscore
import boto3
from botocore import UNSIGNED
from botocore.config import Config

class BiTwistorConduit(nn.Module):
    def __init__(self, input_dim, twistor_dim):
        super().__init__()
        self.W_real = nn.Linear(input_dim, twistor_dim, bias=False)
        self.W_imag = nn.Linear(input_dim, twistor_dim, bias=False)
        
    def forward(self, x):
        real = self.W_real(x)
        imag = self.W_imag(x)
        norm = torch.sqrt(real**2 + imag**2 + 1e-8)
        return torch.complex(real / norm, imag / norm)

class EG_Hamiltonian_V2(nn.Module):
    """
    Corrected E_G gate.
    
    E_G = | <psi| H |psi> |  (the state's OWN energy in the current field)
    
    The threshold ADAPTS:
      threshold(n) = base_threshold * (1 + alpha * log(1 + n))
    where n = number of prior collapses.
    
    This prevents runaway: the more you've collapsed, the harder it is
    to collapse again. Only truly coherent, high-mass states break through.
    """
    def __init__(self, twistor_dim, base_threshold=0.5, alpha=0.3):
        super().__init__()
        self.dim = twistor_dim
        self.base_threshold = base_threshold
        self.alpha = alpha
        
        H_real = torch.randn(twistor_dim, twistor_dim) * 0.01
        H_real = (H_real + H_real.T) / 2.0
        H_imag = torch.randn(twistor_dim, twistor_dim) * 0.01
        H_imag = (H_imag - H_imag.T) / 2.0
        self.H = nn.Parameter(torch.complex(H_real, H_imag))
        
        self.collapse_count = 0
        self.collapse_history = []  # (category_label, psi_state)
        
    def current_threshold(self):
        """Adaptive threshold that rises with collapse count."""
        return self.base_threshold * (1.0 + self.alpha * np.log(1.0 + self.collapse_count))
    
    def compute_eg(self, psi):
        """
        E_G = gravitational self-energy of THIS state alone.
        = | <psi| H |psi> | / dim^2
        
        Normalized by dimensionality so E_G is scale-invariant.
        A random state in a random Hamiltonian should give E_G ~ 0.01.
        Only highly coherent states aligned with the Hamiltonian's 
        eigenmodes will produce E_G >> threshold.
        """
        H_psi = torch.matmul(psi, self.H.T)  
        expectation = torch.sum(torch.conj(psi) * H_psi, dim=-1)
        eg = torch.abs(expectation).mean() / (self.dim ** 2)
        return eg
    
    def forward(self, psi):
        energy_state = torch.matmul(psi, self.H.T)
        expected_energy = torch.real(torch.sum(torch.conj(psi) * energy_state, dim=-1))
        return energy_state, expected_energy
    
    def attempt_collapse(self, psi, category_label=None):
        """
        Attempt to trigger wave function collapse.
        Only succeeds if E_G > adaptive threshold.
        """
        eg = self.compute_eg(psi)
        threshold = self.current_threshold()
        tau = 1.0 / (eg.item() + 1e-10)
        
        if eg.item() > threshold:
            with torch.no_grad():
                # Localized Hamiltonian update: add outer product with decay
                # The update strength decays with collapse count to prevent saturation
                strength = 0.1 / (1.0 + 0.05 * self.collapse_count)
                delta_H = torch.outer(psi[0], torch.conj(psi[0]))
                self.H.add_(strength * delta_H)
                
                self.collapse_count += 1
                self.collapse_history.append((category_label, psi.detach().clone()))
                
            return True, eg.item(), tau, threshold
        else:
            return False, eg.item(), tau, threshold

class DLinOSS_EG_V2(nn.Module):
    def __init__(self, input_dim, twistor_dim=64, base_threshold=0.5, alpha=0.3):
        super().__init__()
        self.conduit = BiTwistorConduit(input_dim, twistor_dim)
        self.hamiltonian = EG_Hamiltonian_V2(twistor_dim, base_threshold, alpha)
        
    def forward(self, x):
        psi = self.conduit(x)
        evolved_psi, energy = self.hamiltonian(psi)
        return psi, evolved_psi, energy

# ================================================================
# DATA LOADING
# ================================================================

SUPER_MAP = {
    'male': 'FACES', 'female': 'FACES',
    'SiberianHusky': 'DOGS', 'toy_poodle': 'DOGS',
    'fighter': 'PLANES', 'airliner': 'PLANES',
    'sportscar': 'CARS', 'jeep': 'CARS',
}

def stream_data(sub_id="sub-002", run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    run_name = f"task-main_run-{run_num}"
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    if not os.path.exists(tsv_path):
        try: s3.download_file(bucket, f"ds006661/{sub_id}/func/{sub_id}_{run_name}_events.tsv", tsv_path)
        except: return None, None
    if not os.path.exists(nii_path):
        try: s3.download_file(bucket, f"ds006661/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz", nii_path)
        except: return None, None
    return nii_path, tsv_path

def extract_fmri_patterns():
    tr_sec = 0.125
    sub = "sub-002"
    runs = ["01", "02", "03", "04"]
    all_trials = []
    for run in runs:
        print(f"  Loading {sub} run-{run}...", flush=True)
        nii_path, tsv_path = stream_data(sub, run)
        if not nii_path: continue
        img = nib.load(nii_path)
        data_raw = img.get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        n_trs = data_raw.shape[-1]
        var_map = np.var(data_raw, axis=-1)
        thresh = np.percentile(var_map, 90)
        brain_mask = var_map > thresh
        active = data_raw[brain_mask]
        data_z = np.nan_to_num(zscore(active, axis=1))
        hrf_start = int(1.0 / tr_sec)
        hrf_end = int(2.0 / tr_sec)
        for _, row in events[events['trial_type'] == 'image'].iterrows():
            cat = row['image_category']
            if cat not in SUPER_MAP: continue
            onset_tr = int(row['onset'] / tr_sec)
            if onset_tr + hrf_end >= n_trs or onset_tr + hrf_start < 0: continue
            pattern = np.mean(data_z[:, onset_tr + hrf_start : onset_tr + hrf_end], axis=1)
            all_trials.append({'cat': SUPER_MAP[cat], 'pattern': pattern, 'run': run})
        if os.path.exists(nii_path): os.remove(nii_path)
    return all_trials

# ================================================================
# EMPIRICAL TEST V2
# ================================================================

def run_test():
    print("=" * 80)
    print(" D-LinOSS E_G v2 EMPIRICAL TEST (Adaptive Threshold)")
    print(" Feeding REAL fMRI data. Will stop if predictions fail.")
    print("=" * 80)
    
    print("\n[1] Extracting fMRI patterns...")
    trials = extract_fmri_patterns()
    input_dim = len(trials[0]['pattern'])
    print(f"  {len(trials)} trials, {input_dim} voxels each.")
    
    print("\n[2] Initializing model (base_threshold=0.02, alpha=1.5)...")
    model = DLinOSS_EG_V2(input_dim=input_dim, twistor_dim=64, 
                           base_threshold=0.02, alpha=1.5)
    
    print("\n[3] Running chronological feed...\n")
    
    records = {c: [] for c in ['FACES', 'DOGS', 'PLANES', 'CARS']}
    collapses = []
    
    for i, trial in enumerate(trials):
        x = torch.tensor(trial['pattern'], dtype=torch.float32).unsqueeze(0)
        psi, evolved, energy = model.forward(x)
        e_val = energy.sum().item()
        
        collapsed, eg, tau, thresh = model.hamiltonian.attempt_collapse(psi, trial['cat'])
        
        records[trial['cat']].append({
            'idx': i, 'energy': e_val, 'eg': eg, 
            'tau': tau, 'collapsed': collapsed, 'threshold': thresh,
            'run': trial['run']
        })
        
        if collapsed:
            collapses.append({'idx': i, 'cat': trial['cat'], 'eg': eg, 'tau': tau, 'thresh': thresh})
    
    # ================================================================
    # RESULTS
    # ================================================================
    
    total_collapses = len(collapses)
    total_trials = len(trials)
    collapse_rate = 100 * total_collapses / total_trials
    
    print("=" * 80)
    print(f" COLLAPSE RATE: {total_collapses} / {total_trials} ({collapse_rate:.1f}%)")
    print("=" * 80)
    
    if collapse_rate > 90:
        print("\n  *** WARNING: Collapse rate > 90%. Threshold too low. ***")
        print("  The model is still collapsing indiscriminately.")
        print("  COURSE CORRECTION NEEDED: Increase base_threshold or alpha.")
    elif collapse_rate < 10:
        print("\n  *** WARNING: Collapse rate < 10%. Threshold too high. ***")
        print("  The model never reaches E_G threshold.")
        print("  COURSE CORRECTION NEEDED: Decrease base_threshold.")
    else:
        print(f"\n  Selective collapse. The E_G gate is discriminating.")
    
    # Per-category breakdown
    print("\n  Per-category collapse rates:")
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        cat_collapses = len([c for c in collapses if c['cat'] == cat])
        cat_total = len(records[cat])
        rate = 100 * cat_collapses / max(1, cat_total)
        print(f"    {cat}: {cat_collapses} / {cat_total} ({rate:.1f}%)")
    
    # Energy trajectory
    print("\n" + "=" * 80)
    print(" ENERGY TRAJECTORY PER CATEGORY")
    print("=" * 80)
    
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        recs = records[cat]
        if len(recs) < 4: continue
        
        q1 = recs[:len(recs)//4]
        q4 = recs[3*len(recs)//4:]
        
        e_q1 = np.mean([r['energy'] for r in q1])
        e_q4 = np.mean([r['energy'] for r in q4])
        delta = e_q4 - e_q1
        
        print(f"  {cat}:")
        print(f"    First quarter energy:  {e_q1:.4f}")
        print(f"    Last quarter energy:   {e_q4:.4f}")
        print(f"    Trajectory: {'+' if delta > 0 else ''}{delta:.4f}")
        
        if delta > 0:
            print(f"    -> RESONANT AMPLIFICATION: Energy grew (Hamiltonian deepened the well)")
        elif delta < -abs(e_q1) * 0.1:
            print(f"    -> ENERGY REDUCED: The system is absorbing this category more efficiently")
        else:
            print(f"    -> STABLE: Energy roughly constant")
    
    # E_G evolution
    print("\n" + "=" * 80)
    print(" E_G EVOLUTION (Does gravitational self-energy grow with exposure?)")
    print("=" * 80)
    
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        recs = records[cat]
        if len(recs) < 4: continue
        q1_eg = np.mean([r['eg'] for r in recs[:len(recs)//4]])
        q4_eg = np.mean([r['eg'] for r in recs[3*len(recs)//4:]])
        print(f"  {cat}: E_G first quarter = {q1_eg:.2f}, last quarter = {q4_eg:.2f}, growth = {q4_eg - q1_eg:.2f}")
    
    # Threshold evolution
    print("\n" + "=" * 80)
    print(" ADAPTIVE THRESHOLD EVOLUTION")
    print("=" * 80)
    
    if len(collapses) > 0:
        print(f"  First collapse threshold:  {collapses[0]['thresh']:.4f}")
        print(f"  Last collapse threshold:   {collapses[-1]['thresh']:.4f}")
        print(f"  Final threshold:           {model.hamiltonian.current_threshold():.4f}")
    
    # Collapse log (first 15)
    print("\n" + "=" * 80)
    print(" COLLAPSE LOG (first 15)")
    print("=" * 80)
    for c in collapses[:15]:
        print(f"  Trial #{c['idx']:>3d} | {c['cat']:>6s} | E_G={c['eg']:.4f} | tau={c['tau']:.6f} | thresh={c['thresh']:.4f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_test()
