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
        # x_seq: (B, T, in_dim). Usually B=1 in our loop.
        x_oscillating = self.dlinoss(x_seq)
        
        sparse_features, dlinoss_recon = self.sae(x_oscillating)
        
        # Map through semantic linear layers: (B, T, target_dim)
        x_mapped = self.semantic_map(sparse_features)
        
        # Use LSTM to process the temporal oscillatory sequence correctly
        lstm_out, (hn, cn) = self.lstm(x_mapped) 
        event_vector = hn[-1] # (B, target_dim)
        
        return F.normalize(event_vector.squeeze(0), p=2, dim=0), sparse_features, dlinoss_recon

def train_dlinoss_superposition():
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
    
    train_events = [ev for ev in all_events if ev[0] in ['sub-4', 'sub-6']]
    test_events = [ev for ev in all_events if ev[0] == 'sub-5']
    
    model = DLinOSSSuperpositionMapper(in_dim=1000, target_dim=512)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    epochs = 40
    l1_lambda = 0.001
    recon_lambda = 1.0
    
    print("\n--- Training D-LinOSS + SAE + Superposition Mapper ---")
    
    model.train()
    for ep in range(epochs):
        loss_anchor_sum = 0
        loss_recon_sum = 0
        loss_sparsity_sum = 0
        
        random.shuffle(train_events)
        
        for subj, cat, x_ev, _ in train_events:
            opt.zero_grad()
            
            pred_vec, z_features, x_recon = model(x_ev.unsqueeze(0))
            
            # The SAE is trying to reconstruct the output of D-LinOSS, 
            # while D-LinOSS is mapping the raw BOLD into an oscillatory shape.
            # Using D-LinOSS output as a target ensures we care about the field resonance.
            with torch.no_grad():
                dlinoss_target = model.dlinoss(x_ev.unsqueeze(0))
                
            true_anchor = clip_anchors[cat]
            
            # Semantic Alignment to Ghost Basin
            anchor_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            
            # SAE matches the field generated by D-LinOSS
            recon_loss = F.mse_loss(x_recon, dlinoss_target)
            
            # Deflate cross-talk interference
            sparsity_loss = torch.mean(torch.abs(z_features))
            
            total_loss = anchor_loss + (recon_lambda * recon_loss) + (l1_lambda * sparsity_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            loss_anchor_sum += anchor_loss.item()
            loss_recon_sum += recon_loss.item()
            loss_sparsity_sum += sparsity_loss.item()
            
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:2d} | Basin Mapping: {loss_anchor_sum/len(train_events):.4f} | Field Recon: {loss_recon_sum/len(train_events):.4f} | Sparse: {loss_sparsity_sum/len(train_events):.4f}")
            
    print("\n--- Discovering Zero-Shot Oscillating Mountains in Subject 5 ---")
    model.eval()
    test_vecs = []
    test_labels = []
    
    with torch.no_grad():
        for subj, true_cat, x_ev, _ in test_events:
            pred_vec, _, _ = model(x_ev.unsqueeze(0))
            test_vecs.append(pred_vec.cpu())
            test_labels.append(true_cat)
            
    # Group thoughts by the True Stimulus Category
    mountain_camps = {cat: [] for cat in categories}
    for label, vec in zip(test_labels, test_vecs):
        mountain_camps[label].append(vec)

    correct_mountains = 0
    print("\n[ Topology Analysis of True 'Mountain' Centroids via D-LinOSS Phase Resonance ]")
    
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
        print(f"  Field Centroid Sim to True Superposition: {centroid_sim_to_true:.3f}")
        if best_centroid_cat == cat:
            print(f"  [✓] Resonance Convergence: PERFECT ALIGNMENT to '{best_centroid_cat}'")
        else:
            print(f"  [x] Resonance Convergence: Misaligned to '{best_centroid_cat}'")

    print(f"\nOverall Resonance Shape Convergence Accuracy: {(correct_mountains/len(categories))*100:.1f}%")

if __name__ == "__main__":
    train_dlinoss_superposition()
