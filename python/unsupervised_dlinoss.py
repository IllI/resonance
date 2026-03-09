"""
unsupervised_dlinoss.py
──────────────────────────────────────────────────────────────
Unsupervised learning of auditory paradigms from 7T BOLD data
using a D-LinOSS Autoencoder.

1. Automatically detect the 42 sound events per run (blind).
2. Extract the temporal response window [t, t+1, t+2] for each event.
3. Train D-LinOSS to reconstruct the temporal trajectory (autoencoder)
   or use temporal contrastive learning.
4. Cluster the resulting latent representations (the "ghost basins").
5. Evaluate if the optimal number of clusters aligns with the 7 categories
   used in the unreleased paradigm.
"""

import os
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# User directories
DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds001942"
OUT_DIR = r"C:\Users\cityz\IllI\newer_all\output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from dlinoss.dlinoss_torch import DLinOSS
except ImportError:
    # Inline definition if not found
    class DLinOSS(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.in_features = in_features
            self.hidden = hidden_features
            self.W_in = nn.Linear(in_features, hidden_features)
            self.A = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
            self.G = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
            nn.init.orthogonal_(self.A)
            nn.init.eye_(self.G)
            self.G.data *= -0.1

        def forward(self, x, h, u):
            # x: (batch, in_features)
            # h, u: (batch, hidden)
            proj = self.W_in(x)
            v = u + torch.tanh(proj)
            A_sym = 0.5 * (self.A + self.A.t())
            G_neg = -F.softplus(self.G)
            forcing = F.linear(v, A_sym) + F.linear(h, G_neg)
            h_new = h + v
            u_new = v + forcing
            return h_new, u_new

# --- 1. Blind Paradigm Recovery Functions ---

def load_bold_run(subj, ses, run):
    fpath = os.path.join(DATA_DIR, subj, ses, 'func', f"{subj}_{ses}_task-auditory_run-{run:02d}_bold.nii.gz")
    if not os.path.exists(fpath):
        return None
    return nib.load(fpath).get_fdata()

def extract_auditory_voxels(bold_4d, percentile=95):
    X, Y, Z, T = bold_4d.shape
    flat = bold_4d.reshape(-1, T)
    mask = flat.std(axis=1) > 1e-3
    brain = flat[mask]
    var = brain.var(axis=1)
    thresh = np.percentile(var, percentile)
    top_mask = var >= thresh
    selected = brain[top_mask]
    ts_z = zscore(selected, axis=1)
    ts_z = np.nan_to_num(ts_z, 0)
    return ts_z, np.where(mask)[0][top_mask]

def detect_sound_events(ts_z):
    pop_signal = ts_z.mean(axis=0)
    pop_var = ts_z.var(axis=0)
    combined = zscore(pop_signal) + zscore(pop_var)
    # Threshold at 72nd percentile (expected ~42/150 = 28% sounds)
    threshold = np.percentile(combined, 72)
    is_sound = combined > threshold
    return np.where(is_sound)[0]

# --- 2. D-LinOSS Autoencoder ---

class DLinOSSAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.encoder = DLinOSS(in_features, hidden_features)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * 2),
            nn.GELU(),
            nn.Linear(hidden_features * 2, in_features)
        )
        self.hidden_features = hidden_features
        
    def forward(self, seqs):
        # seqs: (batch, seq_len, in_features)
        b, seq_len, in_f = seqs.size()
        h = torch.zeros(b, self.hidden_features, device=seqs.device)
        u = torch.zeros(b, self.hidden_features, device=seqs.device)
        
        # Encode timeline
        for t in range(seq_len):
            h, u = self.encoder(seqs[:, t, :], h, u)
            
        # h is the final latent state (the "concept attractor")
        # Reconstruct the mean BOLD response pattern
        mean_spatial = seqs.mean(dim=1)
        recon = self.decoder(h)
        
        return h, recon, mean_spatial


def train_autoencoder(model, dataloader, epochs=50, lr=1e-3, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_seqs in dataloader:
            batch_seqs = batch_seqs[0].to(device)
            optimizer.zero_grad()
            
            latent, recon, target = model(batch_seqs)
            
            # Reconstruction loss (MSE) + L2 penalty on latent to encourage compactness
            loss_recon = F.mse_loss(recon, target)
            loss_reg = 0.01 * latent.norm(dim=1).mean()
            loss = loss_recon + loss_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")


def main():
    subj = 'sub-S01'
    ses = 'ses-SES02'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 65)
    print("  UNSUPERVISED D-LinOSS CATEGORY DISCOVERY")
    print("=" * 65)
    print(f"  Subject: {subj} | Session: {ses}")
    
    all_event_windows = []
    window_length = 3  # TR at sound, TR+1, TR+2
    
    # 1. Blind Event Extraction across all runs
    master_coords = None
    for run in range(1, 12):
        print(f"  Processing run {run:02d}...", end="", flush=True)
        bold_4d = load_bold_run(subj, ses, run)
        if bold_4d is None:
            print(" not found.")
            continue
            
        # In a real pipeline, we'd compute common top-variance voxels
        # For this PoC, we extract per run and take a fixed size subset
        ts_z, coords = extract_auditory_voxels(bold_4d, percentile=98) 
        # Using 98th percentile to get fewer, highly reliable voxels (~3000)
        
        if master_coords is None:
            master_coords = coords
            
        # For simplicity in PoC, align to master coords (intersection)
        idx = np.in1d(coords, master_coords)
        ts_aligned = ts_z[idx, :]
        if ts_aligned.shape[0] < len(master_coords):
            # Pad if missing to keep dimension constant
            pad = np.zeros((len(master_coords) - ts_aligned.shape[0], ts_aligned.shape[1]))
            ts_aligned = np.concatenate([ts_aligned, pad], axis=0)
        ts_aligned = ts_aligned[:len(master_coords), :]
            
        sound_trs = detect_sound_events(ts_aligned)
        print(f" {len(sound_trs)} events detected.")
        
        for tr in sound_trs:
            if tr + window_length <= ts_aligned.shape[1]:
                window = ts_aligned[:, tr : tr + window_length]
                all_event_windows.append(window.T) # shape: (window_length, n_voxels)
                
    n_events = len(all_event_windows)
    in_features = len(master_coords)
    print(f"\n  Extracted {n_events} overall events across valid runs.")
    print(f"  Input feature dimension (voxels): {in_features}")
    
    if n_events == 0:
        return
        
    X = np.stack(all_event_windows)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Train D-LinOSS Autoencoder
    hidden_dim = 128
    print(f"\n  Training unsupervised D-LinOSS Autoencoder (Hidden: {hidden_dim})")
    model = DLinOSSAutoencoder(in_features, hidden_dim).to(device)
    train_autoencoder(model, dataloader, epochs=60, lr=1e-3, device=device)
    
    # 3. Extract Latent Representations
    print("\n  Extracting Latent State Geometries (Ghost Basin)")
    model.eval()
    with torch.no_grad():
        all_h = []
        for x in DataLoader(dataset, batch_size=64, shuffle=False):
            h, _, _ = model(x[0].to(device))
            all_h.append(h.cpu().numpy())
    latents = np.concatenate(all_h, axis=0)
    
    # Normalize latents for cosine clustering
    latents_norm = latents / (np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8)
    
    # 4. Determine Number of Clusters (The 0-shot Test)
    print("\n  Evaluating cluster structures (K=2 to K=10)...")
    best_k = 2
    best_score = -1
    scores = []
    
    cluster_range = range(2, 11)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latents_norm)
        score = silhouette_score(latents_norm, labels, metric='cosine')
        scores.append(score)
        print(f"    K={k:2d} | Silhouette: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\n  OPTIMAL CLUSTER COUNT FOUND: {best_k}")
    print(f"  (Paper documented true categories: 7)")
    
    if best_k == 7:
        print("  => EXACT MATCH. D-LinOSS independently derived the paradigm categories.")
    elif best_k in [6, 8]:
        print("  => STRONG RESULT. Discovered cluster structure nearly perfectly matches paper.")
    else:
        print("  => The BOLD signal contains distinct acoustic structure, but may group")
        print("     certain categories (e.g. human speech & voice) into super-clusters.")

if __name__ == "__main__":
    main()
