"""
dlinoss_eg_v3.py
=================
CORRECTED E_G Model v3: Seed-and-Grow Architecture.

Key insight from v1 and v2 failures:
  - v1: Everything collapsed (threshold too low, no normalization)
  - v2: Nothing collapsed (threshold too high, no initial mass)
  
The physics says:
  - E_G comes from MASS already placed in superposition
  - In the brain, the subject ALREADY knows what a dog is (long-term memory)
  - The initial mass comes from prior experience, not this experiment
  - The experiment provides ADDITIONAL mass via repeated exposure
  - E_G threshold determines whether the additional mass causes collapse

Implementation:
  1. We SEED the Hamiltonian with each category's first trial 
     (simulating long-term memory recognition)
  2. Subsequent trials of the SAME category accumulate mass in that 
     specific eigenmode
  3. When E_G for that eigenmode exceeds threshold, COLLAPSE occurs
  4. After collapse, all subsequent same-category inputs get 
     absorbed with near-zero cost
  5. Different categories should NOT interfere
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

class EG_Hamiltonian_V3(nn.Module):
    """
    Seed-and-Grow Hamiltonian.
    
    Each category gets its own "mass register" in the Hamiltonian.
    Mass accumulates with repeated exposure.
    Collapse only when mass exceeds E_G threshold for THAT category.
    """
    def __init__(self, twistor_dim, eg_threshold=0.15):
        super().__init__()
        self.dim = twistor_dim
        self.eg_threshold = eg_threshold
        
        # The Hamiltonian starts as near-zero (tabula rasa for THIS experiment)
        H_real = torch.zeros(twistor_dim, twistor_dim)
        H_imag = torch.zeros(twistor_dim, twistor_dim)
        self.H = nn.Parameter(torch.complex(H_real, H_imag))
        
        # Per-category mass accumulators
        self.category_mass = {}     # {cat: accumulated E_G}
        self.category_centroids = {} # {cat: running average psi}
        self.category_counts = {}   # {cat: exposure count}
        self.collapsed_categories = set()
        
    def seed_category(self, psi, category):
        """First exposure: plant the initial mass seed from long-term memory."""
        with torch.no_grad():
            seed_strength = 0.01  # Small initial mass (recognition, not learning)
            delta_H = torch.outer(psi[0], torch.conj(psi[0]))
            self.H.add_(seed_strength * delta_H)
            
        self.category_mass[category] = seed_strength
        self.category_centroids[category] = psi.detach().clone()
        self.category_counts[category] = 1
        
    def accumulate_mass(self, psi, category):
        """
        Each re-exposure adds mass to this category's eigenmode.
        Mass added = coherence with the category centroid.
        High coherence = the brain "recognizes" this as the same concept = more mass.
        Low coherence = noisy / different = less mass added.
        """
        centroid = self.category_centroids[category]
        
        # Coherence = overlap with the running centroid
        overlap = torch.abs(torch.sum(torch.conj(psi) * centroid)).item()
        
        # Normalize by dim
        mass_increment = overlap / self.dim
        
        self.category_mass[category] += mass_increment
        self.category_counts[category] += 1
        
        # Update running centroid (exponential moving average)
        alpha = 0.1
        with torch.no_grad():
            self.category_centroids[category] = (
                (1 - alpha) * self.category_centroids[category] + alpha * psi.detach().clone()
            )
        
        return mass_increment
    
    def compute_eg(self, category):
        """E_G for a specific category = its accumulated mass."""
        return self.category_mass.get(category, 0.0)
    
    def forward(self, psi):
        energy_state = torch.matmul(psi, self.H.T)
        expected_energy = torch.real(torch.sum(torch.conj(psi) * energy_state, dim=-1))
        return energy_state, expected_energy
    
    def attempt_collapse(self, psi, category):
        """
        Check if this category has accumulated enough mass to collapse.
        """
        # First exposure?
        if category not in self.category_mass:
            self.seed_category(psi, category)
            return False, self.category_mass[category], float('inf'), 'SEEDED'
        
        # Already collapsed?
        if category in self.collapsed_categories:
            return True, self.category_mass[category], 0.0, 'ALREADY_COLLAPSED'
        
        # Accumulate mass
        increment = self.accumulate_mass(psi, category)
        
        eg = self.compute_eg(category)
        tau = 1.0 / (eg + 1e-10)
        
        if eg > self.eg_threshold:
            # COLLAPSE: Lock in the QPC permanently
            with torch.no_grad():
                centroid = self.category_centroids[category]
                delta_H = torch.outer(centroid[0], torch.conj(centroid[0]))
                # Large permanent restructuring
                self.H.add_(1.0 * delta_H)
                self.collapsed_categories.add(category)
            return True, eg, tau, 'COLLAPSED'
        else:
            return False, eg, tau, 'ACCUMULATING'

class DLinOSS_EG_V3(nn.Module):
    def __init__(self, input_dim, twistor_dim=64, eg_threshold=0.15):
        super().__init__()
        self.conduit = BiTwistorConduit(input_dim, twistor_dim)
        self.hamiltonian = EG_Hamiltonian_V3(twistor_dim, eg_threshold)
        
    def forward(self, x):
        psi = self.conduit(x)
        evolved_psi, energy = self.hamiltonian(psi)
        return psi, evolved_psi, energy

# ================================================================
# DATA
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
# EMPIRICAL TEST V3
# ================================================================

def run_test():
    print("=" * 80)
    print(" D-LinOSS E_G v3: SEED-AND-GROW EMPIRICAL TEST")
    print(" Each category accumulates mass independently.")
    print(" Collapse only when category-specific E_G > threshold.")
    print("=" * 80)
    
    print("\n[1] Extracting fMRI patterns...")
    trials = extract_fmri_patterns()
    input_dim = len(trials[0]['pattern'])
    print(f"  {len(trials)} trials, {input_dim} voxels each.")
    
    print("\n[2] Initializing model (eg_threshold=0.15)...")
    model = DLinOSS_EG_V3(input_dim=input_dim, twistor_dim=64, eg_threshold=0.15)
    
    print("\n[3] Running chronological feed...\n")
    
    records = {c: [] for c in ['FACES', 'DOGS', 'PLANES', 'CARS']}
    event_log = []
    
    for i, trial in enumerate(trials):
        x = torch.tensor(trial['pattern'], dtype=torch.float32).unsqueeze(0)
        psi, evolved, energy = model.forward(x)
        e_val = energy.sum().item()
        
        collapsed, eg, tau, status = model.hamiltonian.attempt_collapse(psi, trial['cat'])
        
        records[trial['cat']].append({
            'idx': i, 'energy': e_val, 'eg': eg, 'tau': tau, 
            'status': status, 'run': trial['run']
        })
        
        event_log.append({
            'idx': i, 'cat': trial['cat'], 'eg': eg, 'status': status, 
            'energy': e_val, 'run': trial['run']
        })
    
    # ================================================================
    # RESULTS
    # ================================================================
    
    print("=" * 80)
    print(" COLLAPSE RESULTS")
    print("=" * 80)
    
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        recs = records[cat]
        collapse_rec = [r for r in recs if r['status'] == 'COLLAPSED']
        
        if len(collapse_rec) > 0:
            c = collapse_rec[0]
            trial_num = sum(1 for r in recs if r['idx'] <= c['idx'])
            print(f"  {cat}: COLLAPSED at trial #{c['idx']} (category exposure #{trial_num}), E_G={c['eg']:.4f}")
        else:
            final_eg = recs[-1]['eg'] if recs else 0
            print(f"  {cat}: DID NOT COLLAPSE. Final E_G = {final_eg:.4f} (threshold = {model.hamiltonian.eg_threshold})")
    
    # Energy before vs after collapse per category
    print("\n" + "=" * 80)
    print(" ENERGY TRAJECTORY (Pre vs Post Collapse)")
    print("=" * 80)
    
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        recs = records[cat]
        collapse_rec = [r for r in recs if r['status'] == 'COLLAPSED']
        
        if len(collapse_rec) > 0:
            c_idx = collapse_rec[0]['idx']
            pre = [r['energy'] for r in recs if r['idx'] < c_idx and r['status'] != 'SEEDED']
            post = [r['energy'] for r in recs if r['idx'] > c_idx]
            
            if len(pre) > 0 and len(post) > 0:
                pre_mean = np.mean(pre)
                post_mean = np.mean(post)
                print(f"  {cat}: Pre={pre_mean:.4f}, Post={post_mean:.4f}, Delta={post_mean-pre_mean:.4f}")
                if post_mean > pre_mean * 1.5:
                    print(f"    -> MASSIVE RESONANT AMPLIFICATION (well deepened)")
                elif post_mean < pre_mean * 0.5:
                    print(f"    -> ENERGY ABSORBED (near-zero cost recognition)")
                else:
                    print(f"    -> MODERATE SHIFT")
            else:
                print(f"  {cat}: Collapsed too early or too late for comparison")
        else:
            # No collapse: just show trajectory
            q1 = recs[:len(recs)//4]
            q4 = recs[3*len(recs)//4:]
            if q1 and q4:
                e1 = np.mean([r['energy'] for r in q1])
                e4 = np.mean([r['energy'] for r in q4])
                print(f"  {cat}: (no collapse) Q1={e1:.4f}, Q4={e4:.4f}")
    
    # Mass accumulation timeline
    print("\n" + "=" * 80)
    print(" MASS (E_G) ACCUMULATION TIMELINE")
    print("=" * 80)
    
    for cat in ['FACES', 'DOGS', 'PLANES', 'CARS']:
        recs = records[cat]
        egs = [(r['idx'], r['eg'], r['status']) for r in recs]
        
        milestones = []
        for idx, eg, status in egs:
            if status in ['SEEDED', 'COLLAPSED'] or len(milestones) == 0 or eg > milestones[-1][1] * 1.5:
                milestones.append((idx, eg, status))
        
        print(f"\n  {cat}:")
        for idx, eg, status in milestones[:8]:
            bar = '#' * min(50, int(eg * 100))
            print(f"    Trial {idx:>3d} | E_G={eg:.4f} | {bar} [{status}]")
    
    # Cross-category interference
    print("\n" + "=" * 80)
    print(" CROSS-CATEGORY INTERFERENCE TEST")
    print("=" * 80)
    
    collapsed_cats = list(model.hamiltonian.collapsed_categories)
    uncollapsed_cats = [c for c in ['FACES', 'DOGS', 'PLANES', 'CARS'] if c not in collapsed_cats]
    
    print(f"  Collapsed: {collapsed_cats}")
    print(f"  Not collapsed: {uncollapsed_cats}")
    
    if len(collapsed_cats) > 0 and len(uncollapsed_cats) > 0:
        c_cat = collapsed_cats[0]
        u_cat = uncollapsed_cats[0] if uncollapsed_cats else None
        
        if u_cat:
            c_collapse_idx = [r for r in records[c_cat] if r['status'] == 'COLLAPSED'][0]['idx']
            u_pre = [r['energy'] for r in records[u_cat] if r['idx'] < c_collapse_idx]
            u_post = [r['energy'] for r in records[u_cat] if r['idx'] > c_collapse_idx]
            
            if u_pre and u_post:
                print(f"  {u_cat} energy before {c_cat} collapse: {np.mean(u_pre):.4f}")
                print(f"  {u_cat} energy after {c_cat} collapse:  {np.mean(u_post):.4f}")
                delta = abs(np.mean(u_post) - np.mean(u_pre))
                if delta < abs(np.mean(u_pre)) * 0.3:
                    print(f"  -> PASS: {c_cat} collapse did not significantly interfere with {u_cat}")
                else:
                    print(f"  -> INTERFERENCE DETECTED: {c_cat} collapse affected {u_cat}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_test()
