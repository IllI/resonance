import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.optimize import quadratic_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

def extract_clip_anchors(categories):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = [f"a photo of a {cat}" for cat in categories]
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = F.normalize(features, p=2, dim=-1)
    return {cat: features[i] for i, cat in enumerate(categories)}

def process_subject(subject_dir, max_runs=12):
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
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
    manifold_ts = np.clip((cleaned - mu) / sigma, -5.0, 5.0).T 
    
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

class SpatiotemporalEnhancement(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.temporal_attn_net = nn.Sequential(
            nn.Linear(target_dim, target_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(target_dim // 2, 1)
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x_seq):
        gate = self.feature_gate(x_seq)
        x_gated = x_seq * gate
        attn_scores = self.temporal_attn_net(x_gated)
        attn_weights = F.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * x_gated, dim=1)
        return context_vector, attn_weights.squeeze(-1)

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

class GhostBasinDynamicMapper(nn.Module):
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
        self.spatiotemporal = SpatiotemporalEnhancement(target_dim)
        
    def forward(self, x_seq):
        x_oscillating = self.dlinoss(x_seq)
        sparse_features, dlinoss_recon = self.sae(x_oscillating)
        x_mapped = self.semantic_map(sparse_features)
        event_vector, attn_weights = self.spatiotemporal(x_mapped)
        # Return un-normalized vector for geometric operations
        return event_vector.squeeze(0), sparse_features, dlinoss_recon, attn_weights

def run_baseline_anchored_hyperalignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    target_categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'shoe']
    all_categories = target_categories + ['scrambledpix']
    
    clip_anchors = extract_clip_anchors(target_categories)
    anchor_matrix = torch.stack([clip_anchors[cat].cpu() for cat in target_categories]).numpy()
    
    all_events = []
    subjects_to_load = [f'sub-{i}' for i in range(1, 7)]
    for subj in subjects_to_load:
        print(f"Loading data for {subj}...")
        path = os.path.join(data_dir, subj)
        if not os.path.exists(path): continue
        ts_2d, labels, coords, TR = process_subject(path)
        evs = extract_dataset_events(ts_2d, labels, all_categories, subj, coords)
        all_events.extend(evs)
    
    test_subject = 'sub-5'
    
    train_events = [ev for ev in all_events if ev[0] != test_subject]
    test_events = [ev for ev in all_events if ev[0] == test_subject]
    
    print(f"\n--- Fortifying Model with Spatiotemporal Enhancement on {len(train_events)} uncrippled events ---")
    
    model = GhostBasinDynamicMapper(in_dim=1000, target_dim=512).to(device)
    for cat in clip_anchors:
        clip_anchors[cat] = clip_anchors[cat].to(device)
        
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)    
    
    for ep in range(35):
        model.eval()
        # 1. Establish the "frenzied" static network baseline for each subject
        subject_baselines = {}
        with torch.no_grad():
            for subj in set(ev[0] for ev in train_events):
                subj_scrambled = [ev for ev in train_events if ev[0] == subj and ev[1] == 'scrambledpix']
                vecs = []
                for _, _, x_ev, _ in subj_scrambled:
                    if x_ev.shape[0] < 2: continue
                    x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_vec, _, _, _ = model(x_ev_tensor)
                    vecs.append(pred_vec)
                if vecs:
                    subject_baselines[subj] = torch.stack(vecs).mean(dim=0)
                    
        model.train()
        loss_anchor_sum = 0
        train_targets = [ev for ev in train_events if ev[1] != 'scrambledpix']
        random.shuffle(train_targets)
        
        for subj, cat, x_ev, _ in train_targets:
            if subj not in subject_baselines: continue # Need a baseline to train
            opt.zero_grad()
            if x_ev.shape[0] < 2: continue
            
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            pred_vec, z_features, x_recon, _ = model(x_ev_tensor)
            
            with torch.no_grad():
                dlinoss_target = model.dlinoss(x_ev_tensor)
                
            # Core theory: Subtract the baseline 'static' origin to get the true ORIENTATION
            oriented_vec = pred_vec - subject_baselines[subj]
            oriented_vec = F.normalize(oriented_vec, p=2, dim=0) # Normalize to pure orientation
            
            true_anchor = clip_anchors[cat]
            anchor_loss = 1.0 - F.cosine_similarity(oriented_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            recon_loss = F.mse_loss(x_recon, dlinoss_target)
            sparsity_loss = torch.mean(torch.abs(z_features))
            total_loss = anchor_loss + recon_loss + (0.001 * sparsity_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_anchor_sum += anchor_loss.item()
        
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1:2d} | Mean Anchor Loss: {loss_anchor_sum/len(train_targets):.4f}")
            
    print("\n--- Zero-Shot Setup: Anchoring Unanalyzed Subject to the Static Baseline ---")
    model.eval()
    
    # Identify Test Subject's Baseline
    test_scrambled = [ev for ev in test_events if ev[1] == 'scrambledpix']
    test_baseline_vecs = []
    with torch.no_grad():
        for _, _, x_ev, _ in test_scrambled:
            if x_ev.shape[0] < 2: continue
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            pred_vec, _, _, attn_weights = model(x_ev_tensor)
            test_baseline_vecs.append(pred_vec)
            if 'print_once' not in locals():
                w = attn_weights[0].cpu().numpy()
                print("Baseline Frenzied Outlier (scrambledpix) Dilation:")
                print("TR Intervals:", [f"T{i}" for i in range(len(w))])
                print("Weights Priority:", [f"{val:.3f}" for val in w])
                print_once = True
                
    test_baseline = torch.stack(test_baseline_vecs).mean(dim=0)
    
    test_oriented_vecs = []
    test_labels = []
    
    test_targets = [ev for ev in test_events if ev[1] != 'scrambledpix']
    with torch.no_grad():
        for subj, true_cat, x_ev, _ in test_targets:
            if x_ev.shape[0] < 2: continue
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            pred_vec, _, _, _ = model(x_ev_tensor)
            
            # Extract oriented structure!
            oriented_vec = pred_vec - test_baseline
            oriented_vec = F.normalize(oriented_vec, p=2, dim=0)
            
            test_oriented_vecs.append(oriented_vec.cpu().numpy())
            test_labels.append(true_cat)
            
    test_oriented_vecs = np.array(test_oriented_vecs)
    
    print("\nClustering Base-Anchored Orientations into 7 Unknown Geometric Camps...")
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=15)
    kmeans.fit(test_oriented_vecs)
    sub5_centroids = kmeans.cluster_centers_
    sub5_centroids = sub5_centroids / np.linalg.norm(sub5_centroids, axis=1, keepdims=True)
    
    sub5_sim = 1 - squareform(pdist(sub5_centroids, metric='cosine'))
    anchor_sim = 1 - squareform(pdist(anchor_matrix, metric='cosine'))
    
    res = quadratic_assignment(sub5_sim, anchor_sim, options={'maximize': True})
    col_ind = res.col_ind
    
    matched_anchors = anchor_matrix[col_ind]
    matched_sub5_centroids = sub5_centroids 
    
    R, scale = orthogonal_procrustes(matched_sub5_centroids, matched_anchors)
    print("Zero-Shot Orthogonal Shape Alignment Complete.")
    
    rotated_test_vecs = np.dot(test_oriented_vecs, R)
    rotated_test_vecs = torch.tensor(rotated_test_vecs, dtype=torch.float32)
    rotated_test_vecs = F.normalize(rotated_test_vecs, p=2, dim=1)
    
    mountain_camps = {cat: [] for cat in target_categories}
    for label, vec in zip(test_labels, rotated_test_vecs):
        mountain_camps[label].append(vec)

    correct_mountains = 0
    print(f"\n--- Evaluating {test_subject}'s Oriented Zero-Shot Categorization ---")
    
    for cat in target_categories:
        events = mountain_camps[cat]
        if not events: continue
        
        stacked_events = torch.stack(events)
        centroid = F.normalize(stacked_events.mean(dim=0), p=2, dim=0)
        
        centroid_sim_to_true = F.cosine_similarity(centroid.unsqueeze(0), clip_anchors[cat].cpu().unsqueeze(0)).item()
        
        best_sim = -float('inf')
        best_centroid_cat = None
        for probe_cat in target_categories:
             sim = F.cosine_similarity(centroid.unsqueeze(0), clip_anchors[probe_cat].cpu().unsqueeze(0)).item()
             if sim > best_sim:
                 best_sim = sim
                 best_centroid_cat = probe_cat
        
        if best_centroid_cat == cat:
            correct_mountains += 1
            
        print(f"\nStimulus Mountain: '{cat}'")
        print(f"  Rotated Ghost Sim to True Superposition: {centroid_sim_to_true:.3f}")
        if best_centroid_cat == cat:
            print(f"  [✓] Resonance Convergence: PERFECT ALIGNMENT to '{best_centroid_cat}'")
        else:
            print(f"  [x] Resonance Convergence: Misaligned to '{best_centroid_cat}'")

    print(f"\nOverall Final Zero-Shot Categorical Accuracy: {(correct_mountains/len(target_categories))*100:.1f}%")

if __name__ == "__main__":
    run_baseline_anchored_hyperalignment()
