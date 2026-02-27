import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import CLIPProcessor, CLIPModel

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

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
    
    shift_trs = max(1, int(round(5.0 / TR)))
    
    all_labels = []
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        shifted = ['rest'] * shift_trs + run_labels[:-shift_trs] if shift_trs > 0 else run_labels
        all_labels.extend(shifted)
        
    return ts_2d, all_labels, coords_array, TR

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
    def __init__(self, in_dim=1000, expansion_factor=4):
        super().__init__()
        # Math: Y = ZA (Superposition compression)
        # We want to map Y (in_dim) back to Z (features_dim) which is much larger
        self.features_dim = in_dim * expansion_factor
        
        # W acts as the un-embedding matrix (recovering features from brain space)
        self.encoder_weight = nn.Parameter(torch.empty(self.features_dim, in_dim))
        nn.init.kaiming_uniform_(self.encoder_weight, a=np.sqrt(5))
        self.encoder_bias = nn.Parameter(torch.zeros(self.features_dim))
        
        # Decoder attempts to reconstruct the brain space using W^T to enforce tied weights
        # (This forces the Gram matrix W^T W formulation described in the paper)
        self.decoder_bias = nn.Parameter(torch.zeros(in_dim))
        
    def forward(self, x):
        # 1. Recover True Features: z = ReLU(Wx + b)
        # ReLU enforces sparsity, dropping out the "Interference Terms" that are below 0
        z = F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))
        
        # 2. Reconstruct Brain Signal: x' = W^T z + c
        reconstructed_x = F.linear(z, self.encoder_weight.t(), self.decoder_bias)
        return z, reconstructed_x

class SAEDisentangledMapper(nn.Module):
    def __init__(self, in_dim=1000, target_dim=512):
        super().__init__()
        # 1. The SAE Layer to filter cross-talk
        self.sae = SparseAutoencoder(in_dim=in_dim, expansion_factor=4)
        
        # 2. The Semantic Journey Mapper
        # Reads the purified *sparse features* rather than raw noisy voxels
        self.signal_encode = nn.Sequential(
            nn.Linear(self.sae.features_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, target_dim),
            nn.LayerNorm(target_dim)
        )
        self.lstm = nn.LSTM(target_dim, target_dim, batch_first=True)
        
    def forward(self, x_seq):
        # x_seq: (T, in_dim)
        sparse_features, reconstructed_seq = self.sae(x_seq)
        
        x_mapped = self.signal_encode(sparse_features)
        lstm_out, (hn, cn) = self.lstm(x_mapped.unsqueeze(0)) 
        event_vector = hn[-1].squeeze(0)
        
        return F.normalize(event_vector, p=2, dim=0), sparse_features, reconstructed_seq

def train_sae_superposition():
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
        
    print(f"\nExtracted {len(all_events)} Total Brain Expressions.")
    
    t_dim_max = max(x.shape[0] for _, _, x, _ in all_events)
    def pad_events(events, target_t):
        padded = []
        for s, c, x, coords in events:
            x_t = torch.tensor(x, dtype=torch.float32)
            if x_t.shape[0] < target_t:
                pad_tensor = torch.zeros((target_t - x_t.shape[0], x_t.shape[1]), dtype=x_t.dtype)
                x_t = torch.cat([x_t, pad_tensor], dim=0)
            elif x_t.shape[0] > target_t:
                x_t = x_t[:target_t]
            padded.append((s, c, x_t, coords))
        return padded
        
    all_events = pad_events(all_events, t_dim_max)
    
    # LEAVE ONE SUBJECT OUT
    train_events = [ev for ev in all_events if ev[0] in ['sub-4', 'sub-6']]
    test_events = [ev for ev in all_events if ev[0] == 'sub-5']
    
    print(f"Training Basin: {len(train_events)} events")
    print(f"Testing Basin (Sub-5): {len(test_events)} events")
    
    model = SAEDisentangledMapper(in_dim=1000, target_dim=512)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Slightly lower weight decay since SAE aids regularization
    
    epochs = 60
    l1_lambda = 0.001 # Sparsity penalty (Forces features to be rarely active)
    recon_lambda = 1.0 # Importance of reconstructing the original biological signal accurately
    
    print("\n--- Training Model with Mathematical Interference Untangling ---")
    
    model.train()
    for ep in range(epochs):
        loss_anchor_sum = 0
        loss_recon_sum = 0
        loss_sparsity_sum = 0
        
        random.shuffle(train_events)
        
        for subj, cat, x_ev, _ in train_events:
            opt.zero_grad()
            
            # Extract Vector, True Features (Z), and Signal Reconstruction
            pred_vec, z_features, x_recon = model(x_ev)
            true_anchor = clip_anchors[cat]
            
            # 1. Semantic Mapping Loss (to Absolute Ghost Basin)
            anchor_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            
            # 2. Reconstruction Loss (Ensuring the SAE accurately models the signal)
            recon_loss = F.mse_loss(x_recon, x_ev)
            
            # 3. Sparsity Loss (L1 norm on Z to eliminate Crosstalk/Interference)
            sparsity_loss = torch.mean(torch.abs(z_features))
            
            total_loss = anchor_loss + (recon_lambda * recon_loss) + (l1_lambda * sparsity_loss)
            
            total_loss.backward()
            opt.step()
            
            loss_anchor_sum += anchor_loss.item()
            loss_recon_sum += recon_loss.item()
            loss_sparsity_sum += sparsity_loss.item()
            
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:2d} | Anchor: {loss_anchor_sum/len(train_events):.4f} | Recon: {loss_recon_sum/len(train_events):.4f} | Sparsity: {loss_sparsity_sum/len(train_events):.4f}")
            
    print("\n--- Validation (Untangling Unseen Subject 5) ---")
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for subj, true_cat, x_ev, _ in test_events:
            # We don't care about anatomical coordinates anymore because we mathematically untangled the signal!
            pred_vec, _, _ = model(x_ev)
            
            best_sim = -1.0
            best_cat = None
            for probe_cat in categories:
                anchor = clip_anchors[probe_cat]
                sim = F.cosine_similarity(pred_vec.unsqueeze(0), anchor.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cat = probe_cat
                    
            if best_cat == true_cat:
                correct += 1
                
    acc = correct / len(test_events) if len(test_events) > 0 else 0
    print(f"\nZero-Shot Unseen Subject Categorization (SAE Filtered): {acc*100:.1f}% ({correct}/{len(test_events)}) - Chance is 12.5%")

if __name__ == "__main__":
    train_sae_superposition()
