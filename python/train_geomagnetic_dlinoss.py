"""
train_geomagnetic_dlinoss.py

Trains a pure PyTorch D-LinOSS learner exclusively on the 72-hour SWMF/BOU 
geomagnetic topology (1-minute resolution). 

Objective: Treat the magnetic field perturbation as a sparse signal (like fMRI)
and observe the internal mathematical state of the D-LinOSS memory lattice.
We calculate the Spectral Entropy over time to detect the precise "grokking" 
moment (wave collapse) when the network aligns with the Earth's gravity well 
perturbation geometry.
"""

import sys
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))

from dlinoss.dlinoss_torch import DLinOSSModel

BOU_NPZ = ROOT / "usgs_geomag_data" / "bou_1min_storm_window.npz"
OUT_DIR = ROOT / "storm_results" / "geomagnetic_grokking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We bypass torch_directml entirely to avoid the complex64 deadlock
# and the AdamW PCIe scalar loop bottleneck.
device = torch.device("cpu")
print(f"Bypassing GPU constraints: executing on native backend {device}")

def load_and_prepare_data():
    if not BOU_NPZ.exists():
        raise FileNotFoundError(f"Missing {BOU_NPZ}")
    
    d = np.load(BOU_NPZ)
    H = d["H"].astype(np.float32)
    dH = d["dH"].astype(np.float32)
    
    # Clean NaNs
    H = np.nan_to_num(H, nan=np.nanmean(H))
    dH = np.nan_to_num(dH, nan=0.0)
    
    # Normalize features (Z-score)
    H_norm = (H - H.mean()) / (H.std() + 1e-8)
    dH_norm = (dH - dH.mean()) / (dH.std() + 1e-8)
    
    # Combine into sequence [L, 2]
    # We slice aggressively to isolate the precise storm maximum intensity (indices 2000:3000)
    # This bypasses the Python autograd bloat and trains only on the exact causal timeframe
    X_full = np.stack([H_norm, dH_norm], axis=-1)[2000:3000]
    
    # Auto-regressive task: Predict t+1 from t
    # To improve learning, we feed a rolling window.
    # We reduce sequence length to 10 to drastically shrink the depth of 
    # the unrolled Python execution graph, speeding up the CPU math 10x.
    SEQ_LEN = 10 # 10 minutes
    
    X_chunks, Y_chunks = [], []
    for i in range(0, len(X_full) - SEQ_LEN):
        X_chunks.append(X_full[i : i + SEQ_LEN])
        Y_chunks.append(X_full[i + 1 : i + SEQ_LEN + 1])
        
    X_t = torch.tensor(np.array(X_chunks), dtype=torch.float32, device=device)
    Y_t = torch.tensor(np.array(Y_chunks), dtype=torch.float32, device=device)
    
    return X_t, Y_t

def calc_spectral_entropy(model):
    """
    Calculates the Shannon Entropy of the D-LinOSS Eigenvalue distribution.
    When the model 'groks' the underlying geometry, a few eigenmodes should 
    dominate the state space, causing a sharp collapse in entropy (wave collapse).
    """
    all_eigs = model.get_all_eigenvalues()
    # Flatten complex eigenvalues
    flat_eigs = torch.cat([e.flatten() for e in all_eigs])
    
    # Magnitudes |lambda|
    mags = flat_eigs.abs()
    
    # Normalize to form a probability distribution
    p = mags / (mags.sum() + 1e-12)
    
    # Shannon Entropy S = -sum(p * log(p))
    # Filter zeros to avoid log(0)
    p_safe = p[p > 1e-8]
    entropy = -torch.sum(p_safe * torch.log(p_safe)).item()
    
    return entropy

def train():
    print("Loading BOU magnetic topology...")
    X, Y = load_and_prepare_data()
    print(f"Dataset shape: {X.shape} -> {Y.shape}")
    
    model = DLinOSSModel(
        input_dim=2,
        state_dim=32,      # 32 oscillators per block
        hidden_dim=64,     # Latent dimension
        output_dim=2,
        num_blocks=2,      # 2 D-LinOSS layers
        classification=False
    ).to(device)
    
    # Restoring AdamW to maintain the second-moment Ghost Basin matrices
    # foreach=False bypasses the `aten::lerp` bug in DirectML without needing fallback
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4, foreach=False)
    criterion = nn.MSELoss()
    
    epochs = 150
    batch_size = 16
    num_batches = X.shape[0] // batch_size
    
    history = []
    
    print("\n--- Commencing D-LinOSS Magnetic Grokking Search ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle batches
        perm = torch.randperm(X.shape[0])
        X_shuf = X[perm]
        Y_shuf = Y[perm]
        
        for b in range(num_batches):
            x_b = X_shuf[b*batch_size : (b+1)*batch_size]
            y_b = Y_shuf[b*batch_size : (b+1)*batch_size]
            
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= num_batches
        
        # Eval Spectral Properties
        model.eval()
        with torch.no_grad():
            entropy = calc_spectral_entropy(model)
            spec_max = model.spectral_summary()["max_spectral_radius"]
        
        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "entropy": entropy,
            "spectral_radius": spec_max
        })
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.5f} | Spectral Entropy: {entropy:.4f} | Max Radius: {spec_max:.4f}")
            sys.stdout.flush()

    print("\n--- Training Complete ---")
    
    # Save history to locate the collapse
    out_file = OUT_DIR / "grokking_history.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"Saved grokking dynamics to {out_file}")

if __name__ == "__main__":
    train()
