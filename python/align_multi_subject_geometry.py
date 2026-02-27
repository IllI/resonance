#!/usr/bin/env python3
"""
Optimization Trajectory Alignment (Zero-Shot)
Based on the "Bathtub/Appliance dynamic frequency dissipation" hypothesis.
"""

import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from dlinoss.dlinoss_torch import DLinOSSModel
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

P = print
def print(*args, **kwargs):
    kwargs['flush'] = True
    P(*args, **kwargs)

def preprocess_run(run_ts, TR):
    N, T = run_ts.shape
    cleaned = run_ts.copy().astype(np.float32)
    t_axis = np.arange(T, dtype=np.float32)
    for i in range(N):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    cleaned = (cleaned - mu) / sigma
    return np.clip(cleaned, -5.0, 5.0).T  # (T, N)

def process_subject(subject_dir, max_runs=12):
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
    # Biological Realism: Hemodynamic delay shift (~5 seconds)
    shift_trs = max(1, int(round(5.0 / TR)))
    
    all_labels = []
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        shifted = ['rest'] * shift_trs + run_labels[:-shift_trs] if shift_trs > 0 else run_labels
        all_labels.extend(shifted)
        
    return ts_2d, all_labels, coords_array, TR

def extract_events(X, labels, categories):
    events = []
    current_cat = None
    curr_start = 0
    for i, lbl in enumerate(labels):
        if lbl != current_cat:
            if current_cat in categories:
                events.append((current_cat, X[curr_start:i]))
            current_cat = lbl
            curr_start = i
    if current_cat in categories:
        events.append((current_cat, X[curr_start:]))
    return events

class SuperpositionDLinOSS(torch.nn.Module):
    def __init__(self, in_dim, h_dim, s_dim, n_blocks, n_classes):
        super().__init__()
        from dlinoss.dlinoss_torch import DLinOSSBlock
        
        # Shared Bathtub alignment (assumption: local variance sorted manifolds share topological significance)
        self.W = torch.nn.Linear(in_dim, h_dim)
        self.W_T = torch.nn.Linear(h_dim, in_dim)
        
        # Noguchi Superposition State (LSTM)
        self.lstm = torch.nn.LSTM(h_dim, h_dim, batch_first=True)
        self.integration_fc = torch.nn.Linear(h_dim * 2, h_dim)
        
        # Autoencoder Vision Decoder
        self.vision_decoder = torch.nn.Sequential(
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.GELU(),
            torch.nn.Linear(h_dim, h_dim)
        )
        
        self.B = torch.nn.ModuleList([
            DLinOSSBlock(state_dim=s_dim, hidden_dim=h_dim, drop_rate=0.1)
            for _ in range(n_blocks)
        ])
        
        # Latent to Categorical Intuition
        self.classifier = torch.nn.Linear(h_dim, n_classes)
        
    def forward(self, x_seq):
        # x_seq: (T, in_dim) representing the full event trajectory
        x_seq_batch = x_seq.unsqueeze(0) # (1, T, in_dim)
        
        # Project visual field
        v = F.relu(self.W(x_seq_batch)) # (1, T, h_dim)
        
        # Superposition Integration
        lstm_out, _ = self.lstm(v)
        
        # Noguchi Dropout Deflation Fix
        ss = F.dropout(v, p=0.5, training=self.training)
        os_state = F.dropout(lstm_out, p=0.5, training=self.training)
        so = F.relu(self.integration_fc(torch.cat([ss, os_state], dim=-1)))
        
        # Decode features from superposed integrated state
        h = self.vision_decoder(so)
        h_flat = h.squeeze(0) # (T, h_dim)
        
        # Apply shared physics dynamics
        for block in self.B:
            h_flat = block(h_flat)
            
        recon = self.W_T(h_flat)
        logits = self.classifier(h_flat)
        
        return recon, h_flat, logits

# Removed measure_splash_trajectory abstraction in favor of end-to-end Autoencoder mappings

def perform_cross_validation(data_dir, subjects, test_subject, max_voxels=600):
    print(f"\n{'='*75}")
    print(f"  MULTI-SUBJECT 'BATHTUB APPLIANCE' ALIGNMENT")
    print(f"  Train: {[s for s in subjects if s != test_subject]}")
    print(f"  Test:  [{test_subject}]")
    print(f"{'='*75}")

    train_subjects = [s for s in subjects if s != test_subject]
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    print("  [Phase 1 & 2] Individual Cortical Bathtub (Local Variance)...")
    
    def extract_local_manifold(ts_2d, TR, num_voxels):
        variances = np.var(ts_2d, axis=1)
        top_idx = np.argsort(variances)[-num_voxels:]
        manifold_ts = ts_2d[top_idx, :]
        return preprocess_run(manifold_ts, TR)
    
    x_train_dict, y_train_dict = {}, {}
    for sub in train_subjects:
        print(f"    Loading {sub}...")
        path = os.path.join(data_dir, sub)
        ts_2d, labels, _, TR = process_subject(path)
        u = extract_local_manifold(ts_2d, TR, max_voxels)
        x_train_dict[sub] = torch.tensor(u, dtype=torch.float32)
        y_train_dict[sub] = labels
        
    # Test Subject
    print(f"    Loading {test_subject}...")
    path = os.path.join(data_dir, test_subject)
    te_ts_2d, te_labels, _, te_TR = process_subject(path)
    X_test = torch.tensor(extract_local_manifold(te_ts_2d, te_TR, max_voxels), dtype=torch.float32)

    HIDDEN_DIM = 128
    CHUNK = 256
    
    model = SuperpositionDLinOSS(
        in_dim=max_voxels, h_dim=HIDDEN_DIM, s_dim=64, n_blocks=2, n_classes=len(categories)
    )

    print("  [Phase 4] Joint LSTM Autoencoder Training (Learning the Sequential Resonance)...")
    opt_train = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Generic physics optimizer
    
    # Extract events across subjects
    train_events = []
    for sub in train_subjects:
        events = extract_events(x_train_dict[sub], y_train_dict[sub], categories)
        for cat, x_ev in events:
            train_events.append((cat, x_ev))
            
    sigma_0_sq = 0.1
    alpha = 0.05
    gamma_con = 0.05
    
    import random
    
    for ep in range(80):
        model.train()
        loss_r_sum = 0
        loss_c_sum = 0
        random.shuffle(train_events)
        
        for cat, x_ev in train_events:
            opt_train.zero_grad()
            
            T_steps = x_ev.shape[0]
            # Mindsight Cortical Gaussian Diffusion along time
            diff_vars = sigma_0_sq + alpha * torch.arange(T_steps, device=x_ev.device).float().unsqueeze(1)
            diffusion_noise = torch.randn_like(x_ev) * (diff_vars ** 0.5) * 0.01
            x_ev_diffused = x_ev + diffusion_noise
            
            recon, h, logits = model(x_ev_diffused)
            
            l_recon = F.mse_loss(recon, x_ev) # Map to pure biological output
            activation_loss = gamma_con * torch.mean(torch.exp(-torch.abs(recon) / 0.1))
            
            targ_idx = torch.tensor([cat_to_idx[cat]] * T_steps, dtype=torch.long, device=logits.device)
            l_class = F.cross_entropy(logits, targ_idx)
            
            loss = l_recon + activation_loss + l_class
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_train.step()
            
            loss_r_sum += l_recon.item()
            loss_c_sum += l_class.item()
            
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1:3d}: Recon MSE = {loss_r_sum/len(train_events):.4f} | Class CE = {loss_c_sum/len(train_events):.4f}")

    print("\n  [Phase 6] 'Bathtub & Appliance' Dynamic Zero-Shot Extraction...")
    test_events = extract_events(X_test, te_labels, categories)
    
    correct = 0
    total = 0
    per_cat_correct = {c: 0 for c in categories}
    per_cat_total = {c: 0 for c in categories}
    
    model.eval()
    with torch.no_grad():
        for true_cat, x_ev in test_events:
            _, _, logits = model(x_ev)
            
            # Predict overarching intuition category spanning event
            mean_logits = logits.mean(dim=0)
            pred_idx = mean_logits.argmax().item()
            pred_cat = categories[pred_idx]
            
            if pred_cat == true_cat:
                correct += 1
                per_cat_correct[true_cat] += 1
            total += 1
            per_cat_total[true_cat] += 1
        # Obsolete best_cat matching logic removed
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n  {'='*65}")
    print(f"  RESULT (Test: {test_subject})")
    print(f"  Zero-Shot Accuracy: {accuracy*100:.1f}% ({accuracy/(1/8):.1f}x chance)")
    print(f"  Per-category Events:")
    for c in categories:
        t = per_cat_total[c]
        cr = per_cat_correct[c]
        pct = (cr/t*100) if t > 0 else 0
        print(f"    {c:15s}: {cr:3d}/{t:3d} = {pct:5.1f}%")
    print(f"  {'='*65}\n")
    
    return accuracy

def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')
    subjects = ['sub-1', 'sub-2', 'sub-3', 'sub-4', 'sub-5', 'sub-6']
    test_subjects = ['sub-4', 'sub-5', 'sub-6']
    
    accs = []
    for test_sub in test_subjects:
        acc = perform_cross_validation(base, subjects, test_sub, max_voxels=600)
        accs.append(acc)
        
    print(f"\n{'='*65}")
    print(f"  FINAL: Mean Cross-Subject Event Accuracy: {np.mean(accs)*100:.1f}%")
    print(f"  Per-fold: {[f'{a*100:.1f}%' for a in accs]}")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
