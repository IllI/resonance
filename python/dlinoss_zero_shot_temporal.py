import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import CLIPProcessor, CLIPModel

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

def extract_clip_anchors(categories):
    print("Establishing absolute Superposition Anchors (CLIP ViT-B/32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    text_inputs = [f"a photo of a {cat}" if cat != 'scrambledpix' else "a scrambled nonsense static image" for cat in categories]
    
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        
    features = F.normalize(features, p=2, dim=-1)
    return {cat: features[i] for i, cat in enumerate(categories)}

def process_subject(subject_dir, max_runs=12):
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
    # We do NOT inherently shift the labels globally here because we need 
    # granular control over isolating the T0 (Visual) and T1-T3 (Ghost/HRF) windows.
    all_labels_unshifted = []
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        all_labels_unshifted.extend(run_labels)
        
    return ts_2d, all_labels_unshifted, coords_array, TR

def extract_dataset_events(ts_2d, labels, categories, subject_id, coords_array):
    max_voxels = 1000
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    active_coords = coords_array[top_idx]
    
    N, T = ts_2d.shape
    cleaned = ts_2d[top_idx, :].copy().astype(np.float32)
    t_axis = np.arange(T, dtype=np.float32)
    for i in range(max_voxels):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    manifold_ts = np.clip((cleaned - mu) / sigma, -5.0, 5.0).T  # (T, max_voxels)
    
    # We construct unshifted event blocks to slice later
    events = []
    current_cat = None
    curr_start = 0
    for i, lbl in enumerate(labels):
        if lbl != current_cat:
            if current_cat in categories:
                events.append((subject_id, current_cat, manifold_ts[curr_start:i], active_coords))
            current_cat = lbl
            curr_start = i
            
    if current_cat in categories:
        events.append((subject_id, current_cat, manifold_ts[curr_start:], active_coords))
        
    return events

class SparseAutoencoder(nn.Module):
    def __init__(self, in_dim, expansion_factor=4):
        super().__init__()
        self.features_dim = in_dim * expansion_factor
        self.encoder_weight = nn.Parameter(torch.empty(self.features_dim, in_dim))
        nn.init.kaiming_uniform_(self.encoder_weight, a=np.sqrt(5))
        self.encoder_bias = nn.Parameter(torch.zeros(self.features_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(in_dim))
        
    def forward(self, x):
        z = F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))
        x_recon = F.linear(z, self.encoder_weight.t(), self.decoder_bias)
        return z, x_recon

class DLinOSSSuperpositionMapper(nn.Module):
    def __init__(self, in_dim=1000, target_dim=512, dlinoss_state_dim=32, dlinoss_hidden=64):
        super().__init__()
        self.dlinoss = DLinOSSModel(
            input_dim=in_dim,
            state_dim=dlinoss_state_dim,
            hidden_dim=dlinoss_hidden,
            output_dim=in_dim,
            num_blocks=2,
            drop_rate=0.1
        )
        self.sae = SparseAutoencoder(in_dim=in_dim, expansion_factor=4)
        
        self.semantic_map = nn.Sequential(
            nn.Linear(self.sae.features_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, target_dim),
            nn.LayerNorm(target_dim)
        )
        self.lstm = nn.LSTM(target_dim, target_dim, batch_first=True)
        
    def forward(self, x_seq):
        x_oscillating = self.dlinoss(x_seq)
        sparse_features, dlinoss_recon = self.sae(x_oscillating)
        x_mapped = self.semantic_map(sparse_features)
        
        # We process the sequence through LSTM to capture temporal dynamics up to the extracted window point
        lstm_out, (hn, cn) = self.lstm(x_mapped) 
        event_vector = hn[-1] # Shape: (B, target_dim)
        
        return F.normalize(event_vector.squeeze(0), p=2, dim=0), sparse_features, dlinoss_recon

def train_temporal_superposition():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    
    clip_anchors = extract_clip_anchors(categories)
    all_events = []
    
    for subj in ['sub-4', 'sub-5', 'sub-6']:
        print(f"Loading data for {subj}...")
        path = os.path.join(data_dir, subj)
        if not os.path.exists(path):
            continue
        ts_2d, labels, coords, TR = process_subject(path)
        evs = extract_dataset_events(ts_2d, labels, categories, subj, coords)
        all_events.extend(evs)
    
    # We enforce a strict temporal slice! 
    # A block is roughly 24 seconds. TR is 2.5 seconds.
    # We want to isolate exactly the timeframe immediately after the stimulus is processed,
    # mapping to the "Ghost". We want to cut OFF the motor cortex decision and sustained block exhaustion.
    # Let's say: TR 1 to 4 (approx 2.5s to 10s after stimulus onset, accounting for HRF delay 
    # capturing the exact cognitive phase-shift of the Superposition).
    ghost_window_start = 1
    ghost_window_end = 5
    
    sliced_events = []
    for s, c, x, coords in all_events:
        if x.shape[0] >= ghost_window_end:
            # We strictly extract ONLY the empirical "Ghost" processing window
            x_sliced = x[ghost_window_start:ghost_window_end]
            sliced_events.append((s, c, x_sliced, coords))

    train_events = [ev for ev in sliced_events if ev[0] in ['sub-4', 'sub-6']]
    test_events = [ev for ev in sliced_events if ev[0] == 'sub-5']
    
    model = DLinOSSSuperpositionMapper(in_dim=1000, target_dim=512)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    epochs = 40
    
    print("\n--- Training Superposition Mapper EXCLUSIVELY on Isolated 'Ghost' Temporal Window ---")
    print(f"Isolating HRF-adjusted Phase shift strictly between {ghost_window_start*2.5}s to {ghost_window_end*2.5}s...")
    
    model.train()
    for ep in range(epochs):
        loss_anchor_sum = 0
        random.shuffle(train_events)
        
        for subj, cat, x_ev, _ in train_events:
            opt.zero_grad()
            
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0)
            pred_vec, z_features, x_recon = model(x_ev_tensor)
            
            with torch.no_grad():
                dlinoss_target = model.dlinoss(x_ev_tensor)
                
            true_anchor = clip_anchors[cat]
            anchor_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            recon_loss = F.mse_loss(x_recon, dlinoss_target)
            sparsity_loss = torch.mean(torch.abs(z_features))
            
            total_loss = anchor_loss + recon_loss + (0.001 * sparsity_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            loss_anchor_sum += anchor_loss.item()
            
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:2d} | 'Ghost Window' Basin Mapping Loss: {loss_anchor_sum/len(train_events):.4f}")
            
    print("\n--- Discovering Zero-Shot Oscillating Mountains in Unseen Subject 5 ---")
    print("Using only the isolated temporally-sliced cognitive recognition wave (Scrubbing motor noise).")
    
    model.eval()
    test_vecs = []
    test_labels = []
    
    with torch.no_grad():
        for subj, true_cat, x_ev, _ in test_events:
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0)
            pred_vec, _, _ = model(x_ev_tensor)
            test_vecs.append(pred_vec.cpu())
            test_labels.append(true_cat)
            
    mountain_camps = {cat: [] for cat in categories}
    for label, vec in zip(test_labels, test_vecs):
        mountain_camps[label].append(vec)

    correct_mountains = 0
    
    for cat in categories:
        events = mountain_camps[cat]
        if not events: continue
        
        stacked_events = torch.stack(events)
        centroid = F.normalize(stacked_events.mean(dim=0), p=2, dim=0)
        
        centroid_sim_to_true = F.cosine_similarity(centroid.unsqueeze(0), clip_anchors[cat].unsqueeze(0)).item()
        best_centroid_cat = max(categories, key=lambda c: F.cosine_similarity(centroid.unsqueeze(0), clip_anchors[c].unsqueeze(0)).item())
        
        if best_centroid_cat == cat:
            correct_mountains += 1
            
        print(f"\nStimulus Mountain: '{cat}'")
        print(f"  Ghost Centroid Sim to True Superposition: {centroid_sim_to_true:.3f}")
        if best_centroid_cat == cat:
            print(f"  [✓] Resonance Convergence: PERFECT ALIGNMENT to '{best_centroid_cat}'")
        else:
            print(f"  [x] Resonance Convergence: Misaligned to '{best_centroid_cat}'")

    print(f"\nOverall 'Ghost Window' Zero-Shot Topology Accuracy: {(correct_mountains/len(categories))*100:.1f}%")

if __name__ == "__main__":
    train_temporal_superposition()
