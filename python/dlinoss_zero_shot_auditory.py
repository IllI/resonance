"""
dlinoss_zero_shot_auditory.py
──────────────────────────────────────────────────────────────
Tests the transferability of D-LinOSS latent "ghost basins" across
different human brains in the 7T auditory dataset (ds001942).

Theory under test:
"If the physical structure of the brain determines how fields resonate
and decode acoustic packets, then the 'ghost basins' learned on Subject 1
should transfer to Subject 2, without fine-tuning."

Method:
1. Extract top variance voxels from S01 and S02 (auditory active).
2. Compress both brains into 128 Functional Eigenmodes (SVD) to standardize
   the input dimension WITHOUT forcing spatial anatomical alignment.
3. Blindly detect the 42 complex sound events per run for both subjects.
4. Train D-LinOSS autoencoder SOLELY on S01's temporal eigenmode flow.
5. Freeze the model. Pass S02's eigenmodes into S01's latent space.
6. Evaluate zero-shot cross-subject transfer.
"""

import os
import sys
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds001942"

# Inline D-LinOSS Autoencoder adapted for eigenmodes
class DLinOSSAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden_features
        # Encoder
        self.W_in = nn.Linear(in_features, hidden_features)
        self.A = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.G = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        nn.init.orthogonal_(self.A)
        nn.init.eye_(self.G)
        self.G.data *= -0.1

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * 2),
            nn.GELU(),
            nn.Linear(hidden_features * 2, in_features)
        )

    def encoder_step(self, x, h, u):
        proj = self.W_in(x)
        v = u + torch.tanh(proj)
        A_sym = 0.5 * (self.A + self.A.t())
        G_neg = -F.softplus(self.G)
        forcing = F.linear(v, A_sym) + F.linear(h, G_neg)
        return h + v, v + forcing

    def forward(self, seqs):
        b, seq_len, _ = seqs.size()
        h = torch.zeros(b, self.hidden, device=seqs.device)
        u = torch.zeros(b, self.hidden, device=seqs.device)
        for t in range(seq_len):
            h, u = self.encoder_step(seqs[:, t, :], h, u)
        # Reconstruct the 3-TR response as a mean spatial eigenmode pattern
        mean_spatial = seqs.mean(dim=1)
        recon = self.decoder(h)
        return h, recon, mean_spatial

def load_bold_run(subj, ses, run):
    fname = f"{subj}_{ses}_task-auditory_run-{run:02d}_bold.nii.gz"
    fpath = os.path.join(DATA_DIR, subj, ses, 'func', fname)
    if not os.path.exists(fpath):
        return None
    try:
        data = nib.load(fpath).get_fdata()
        return data
    except Exception:
        return None

def extract_auditory_voxels_and_svd(bold_4d_list, percentile=98, n_components=128):
    """
    Extract top variance voxels from the FIRST run, apply as mask to all runs,
    then compute SVD to return functional eigenmodes.
    """
    if not bold_4d_list:
        return None

    # Find mask from the first run's variance to save memory
    first_run = bold_4d_list[0].reshape(-1, bold_4d_list[0].shape[3])
    mask = first_run.std(axis=1) > 1e-3
    brain_0 = first_run[mask]
    var_0 = brain_0.var(axis=1)
    thresh = np.percentile(var_0, percentile)
    top_mask = var_0 >= thresh
    # This boolean array determines exactly which flattened voxels to keep
    
    # Extract and Z-score those specific voxels from all runs
    z_runs = []
    for run in bold_4d_list:
        run_flat = run.reshape(-1, run.shape[3])
        chunk = run_flat[mask][top_mask]
        # Memory explicit downcast to float32
        chunk_z = zscore(chunk, axis=1).astype(np.float32)
        chunk_z = np.nan_to_num(chunk_z, 0)
        z_runs.append(chunk_z)
        
    concat_z = np.concatenate(z_runs, axis=1)

    # Fit TruncatedSVD to create 'Functional Eigenmodes'
    print(f"      Selected {concat_z.shape[0]} voxels. Computing SVD ({n_components} components)...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    eigen_timeseries = svd.fit_transform(concat_z.T).T  # shape (n_comp, total_time)
    
    # Repartition back into runs
    run_eigenmodes = []
    t_start = 0
    for run in z_runs:
        n_t = run.shape[1]
        run_eigenmodes.append(eigen_timeseries[:, t_start:t_start+n_t])
        t_start += n_t
        
    return run_eigenmodes

def detect_sound_events_from_eigenmodes(eigen_ts):
    """
    Detects sound events using the dominant functional eigenmodes.
    Component 0 (the highest variance) usually tracks the massive 
    stimulus-vs-silence shift that the whole auditory network exhibits.
    """
    pop_signal = np.abs(eigen_ts).mean(axis=0)
    # Threshold at 72nd percentile (expect ~42 sounds out of 150 TRs)
    threshold = np.percentile(pop_signal, 72)
    is_sound = pop_signal > threshold
    return np.where(is_sound)[0]

def process_subject(subj, ses='ses-SES02', runs=11, n_components=128):
    print(f"  > Processing {subj}...")
    bolds = []
    for r in range(1, runs+1):
        bold = load_bold_run(subj, ses, r)
        if bold is not None:
            bolds.append(bold)
    
    if not bolds:
        print(f"    No data found for {subj}")
        return None
        
    print(f"    Loaded {len(bolds)} runs.")
    all_runs_eigen = extract_auditory_voxels_and_svd(bolds, percentile=98, n_components=n_components)
    
    # Extract temporal windows for detected sound events
    all_events = []
    window = 3
    event_counts = []
    
    for run_eigen in all_runs_eigen:
        sound_trs = detect_sound_events_from_eigenmodes(run_eigen)
        event_counts.append(len(sound_trs))
        
        for tr in sound_trs:
            if tr + window <= run_eigen.shape[1]:
                # Extracted window shape: (window, n_components)
                all_events.append(run_eigen[:, tr:tr+window].T)
                
    print(f"    Mean detected events per run: {np.mean(event_counts):.1f}")
    X = np.stack(all_events)
    print(f"    Extracted {len(X)} overall multi-TR response windows.")
    return torch.tensor(X, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("  ZERO-SHOT CROSS-SUBJECT AUDITORY ALIGNMENT (Testing NFT)")
    print("=" * 65)

    n_components = 128
    hidden = 64
    
    # 1. Process Subject 1 (Source)
    x_s1 = process_subject('sub-S01', 'ses-SES02', n_components=n_components)
    if x_s1 is None:
        print("Missing sub-S01.")
        return
        
    # 2. Train D-LinOSS Autoencoder entirely on Subject 1
    model = DLinOSSAutoencoder(n_components, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    loader_s1 = DataLoader(TensorDataset(x_s1), batch_size=32, shuffle=True)
    print("\n  >> Training D-LinOSS on Source Brain (sub-S01)...")
    for epoch in range(40):
        total_loss = 0
        for batch in loader_s1:
            seq = batch[0].to(device)
            optimizer.zero_grad()
            latent, recon, target = model(seq)
            loss = F.mse_loss(recon, target) + 0.01 * latent.norm(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"     Epoch {epoch+1}/40 | Loss: {total_loss/len(loader_s1):.4f}")

    # Extract S1 Latent Clusters
    model.eval()
    with torch.no_grad():
        h_s1, _, _ = model(x_s1.to(device))
    h_s1 = h_s1.cpu().numpy()
    h_s1_norm = h_s1 / (np.linalg.norm(h_s1, axis=1, keepdims=True) + 1e-8)
    
    # K=10 based on our prior analysis of multi-peak resonances mixed with noise
    K = 10 
    kmeans_s1 = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_s1 = kmeans_s1.fit_predict(h_s1_norm)
    score_s1 = silhouette_score(h_s1_norm, labels_s1, metric='cosine')
    print(f"\n  >> Subject 1 Internal Basin Structure (K={K}):")
    print(f"     Silhouette Score: {score_s1:.4f}")
    
    # 3. Process Subject 2 (Target)
    print("\n  >> Waiting for Subject 2 data...")
    x_s2 = process_subject('sub-S02', 'ses-SES02', n_components=n_components)
    if x_s2 is None:
        print("     sub-S02 not fully downloaded yet. Checking ses-SES01 instead...")
        x_s2 = process_subject('sub-S02', 'ses-SES01', n_components=n_components)
    
    if x_s2 is None:
        print("Missing sub-S02. Exiting Zero-Shot Test.")
        return

    # 4. Zero-Shot Testing on Subject 2
    print("\n  >> ZERO-SHOT PASS (Feeding sub-S02 into sub-S01's field equations)...")
    with torch.no_grad():
        h_s2, recon_s2, target_s2 = model(x_s2.to(device))
        
    # Evaluate reconstruction accuracy degradation
    mse_s1 = F.mse_loss(model(x_s1.to(device))[1], model(x_s1.to(device))[2]).item()
    mse_s2 = F.mse_loss(recon_s2, target_s2).item()
    print(f"     Reconstruction Error S1 (Train): {mse_s1:.4f}")
    print(f"     Reconstruction Error S2 (Zero) : {mse_s2:.4f} (+{(mse_s2-mse_s1)/mse_s1*100:.1f}%)")
    
    h_s2 = h_s2.cpu().numpy()
    h_s2_norm = h_s2 / (np.linalg.norm(h_s2, axis=1, keepdims=True) + 1e-8)
    
    # Using S1's predefined cluster centroids to assign S2's packets
    labels_s2_mapped = kmeans_s1.predict(h_s2_norm)
    
    try:
        score_s2 = silhouette_score(h_s2_norm, labels_s2_mapped, metric='cosine')
    except ValueError:
        score_s2 = 0.0 # Assigns everything to one cluster
        
    print("\n  >> ZERO-SHOT CLUSTERING RESULTS:")
    print(f"     Subject 2 Silhouette in S1's Ghost Basin: {score_s2:.4f}")
    
    print("\n  INTERPRETATION:")
    if score_s2 > 0.15: # Highly structured
        print("  => SUCCESS: The functional resonant structures transferred natively!")
        print("     SVD functionally aligns the 'antennae', and D-LinOSS correctly")
        print("     reads the target's temporal frequency states using S1's tuning.")
    elif score_s2 > 0.05:
        print("  => PARTIAL TRANSFER: Subject 2's field dynamics partially localized")
        print("     inside S1's tuned latent space.")
    else:
        print("  => COLLAPSE: Subject 2's brain signals crashed when forced into Subject 1's")
        print("     field tuning parameters (A, G). This proves that the field geometry")
        print("     is intimately bound to the individual's strict biological substrate.")
        print("     (Functional eigenmodes aligned horizontally by variance are not enough")
        print("     to align temporal phase transitions. The 'hum' is unique per head.)")

if __name__ == "__main__":
    main()
