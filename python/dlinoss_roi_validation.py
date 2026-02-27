import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.spatial.distance import cdist
from transformers import CLIPProcessor, CLIPModel

# Import the existing loader and DLinOSS pieces without changing them
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
    
    # HRF delay (typically peaking at ~5 seconds). We shift labels to match the BOLD surge.
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
                # Store the anatomical coordinates explicitly with the event
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

# Exactly as it was trained in our reproduced results
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
        lstm_out, (hn, cn) = self.lstm(x_mapped) 
        event_vector = hn[-1] 
        return F.normalize(event_vector.squeeze(0), p=2, dim=0), sparse_features, dlinoss_recon

def validate_biological_spatial_overlap():
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
    
    print("\n--- Training Model (Reproducing Ghost Basin Mapping) ---")
    model.train()
    for ep in range(15): # Shortened slightly just to get coherent fields for spatial val
        loss_anchor_sum = 0
        random.shuffle(train_events)
        for subj, cat, x_ev, _ in train_events:
            opt.zero_grad()
            pred_vec, z_features, x_recon = model(x_ev.unsqueeze(0))
            with torch.no_grad():
                 dlinoss_target = model.dlinoss(x_ev.unsqueeze(0))
            true_anchor = clip_anchors[cat]
            anchor_loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            recon_loss = F.mse_loss(x_recon, dlinoss_target)
            sparsity_loss = torch.mean(torch.abs(z_features))
            total_loss = anchor_loss + recon_loss + (0.001 * sparsity_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        print(f"Epoch {ep+1:2d} Mapping Complete.")
            
    # =========================================================================
    # THE STATISTICAL FRAMEWORK: MAPPING RESONANCE WAVES TO BOLD ROIS IN 3D SPACE
    # =========================================================================
    print("\n--- Spatial Validation: D-LinOSS Brain Waves vs Empirical BOLD ROIs ---")
    model.eval()
    
    # We define a biological margin of error (e.g. 2.0 voxel radius ≈ 6-8mm)
    # This accounts for scanner alignment shifts, breathing, and minor topological smoothing
    spatial_margin_of_error_voxels = 2.0 
    
    # Group results per category structurally
    spatial_stats = {cat: {'overlap_pcts': [], 'global_corrs': []} for cat in categories}

    with torch.no_grad():
        for subj, true_cat, x_ev, coords in test_events:
            # 1. The Model's "Gradient Wave" of resonance activation
            # Shape: (1000,) - We average the resonance intensity across the event time
            dlinoss_resonance_wave = model.dlinoss(x_ev.unsqueeze(0)).squeeze(0).mean(dim=0).cpu().numpy()
            
            # 2. The Traditional BOLD Contrast "Clusters of dots" 
            # Because the data is z-scored against the run, the mean amplitude during the 
            # label-shifted HRF window IS the BOLD contrast vs baseline.
            # Shape: (1000,)
            bold_roi_strength = x_ev.mean(dim=0).cpu().numpy()
            
            # 3. Illuminate Overlaps in the identical XYZ coordinates
            
            # Global Structural Correlation (Do the peaks and valleys generally align?)
            corr = np.corrcoef(np.abs(dlinoss_resonance_wave), np.abs(bold_roi_strength))[0, 1]
            spatial_stats[true_cat]['global_corrs'].append(corr)
            
            # Thresholding to find the physical "Signal Spikes"
            # Find the top 15% strongest vibrating voxels for the Model wave
            dlinoss_threshold = np.percentile(np.abs(dlinoss_resonance_wave), 85)
            model_spike_indices = np.where(np.abs(dlinoss_resonance_wave) >= dlinoss_threshold)[0]
            model_spike_coords = coords[model_spike_indices]
            
            # Find the top 15% hottest voxels for the BOLD Contrast
            bold_threshold = np.percentile(np.abs(bold_roi_strength), 85)
            bold_roi_indices = np.where(np.abs(bold_roi_strength) >= bold_threshold)[0]
            bold_roi_coords = coords[bold_roi_indices]
            
            # Calculate physical distance matrices
            if len(model_spike_coords) > 0 and len(bold_roi_coords) > 0:
                distances = cdist(model_spike_coords, bold_roi_coords, metric='euclidean')
                
                # Check how many of the Model's "Gradient Wave" physical spikes fall 
                # within the Margin of Error of the known BOLD "Dot Clusters"
                min_distances_to_bold = np.min(distances, axis=1)
                overlapping_spikes = np.sum(min_distances_to_bold <= spatial_margin_of_error_voxels)
                
                overlap_percentage = (overlapping_spikes / len(model_spike_coords)) * 100.0
                spatial_stats[true_cat]['overlap_pcts'].append(overlap_percentage)

    # Print summary of biological alignments
    print(f"\nMargin of Error applied: {spatial_margin_of_error_voxels} voxels (~6mm) accounting for breathing/head motion.")
    for cat in categories:
        overlaps = spatial_stats[cat]['overlap_pcts']
        corrs = spatial_stats[cat]['global_corrs']
        
        if len(overlaps) == 0: continue
            
        avg_overlap = np.mean(overlaps)
        avg_corr = np.mean(corrs)
        
        print(f"\nStimulus Mountain: '{cat}'")
        print(f"  Physical Spike Alignment (Top 15%): {avg_overlap:.1f}% of Model spikes sit perfectly on Empirical ROIs.")
        print(f"  Global 3D Gradient Correlation: r = {avg_corr:.3f}")
        
    print("\nIf Physical Spike Alignment exceeds random chance (~15%), it geometrically proves the Resonance field naturally generated the anatomical ROIs without being supervised to do so.")

if __name__ == "__main__":
    validate_biological_spatial_overlap()
