import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
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
        lstm_out, (hn, cn) = self.lstm(x_mapped) 
        event_vector = hn[-1] 
        return F.normalize(event_vector.squeeze(0), p=2, dim=0), sparse_features, dlinoss_recon

def unsupervised_manifold_rotation():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    
    clip_anchors = extract_clip_anchors(categories)
    # Target anchor matrix (K, D)
    anchor_matrix = torch.stack([clip_anchors[cat].cpu() for cat in categories]).numpy()
    
    all_events = []
    
    for subj in ['sub-4', 'sub-5', 'sub-6']:
        print(f"Loading data for {subj}...")
        path = os.path.join(data_dir, subj)
        if not os.path.exists(path): continue
        ts_2d, labels, coords, TR = process_subject(path)
        evs = extract_dataset_events(ts_2d, labels, categories, subj, coords)
        all_events.extend(evs)
    
    # ISOLATING THE STIMULUS PROCESSING DECISION WINDOW
    # We ignore the initial novelty response and we cut out the mind-wandering 
    # and motor cortex reactions.
    ghost_window_start = 1
    ghost_window_end = 4
    
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
    print("\n--- Training Superposition Mapper on Decision Window ---")
    model.train()
    for ep in range(40):
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
            
    print("\n--- Unsupervised Hyperalignment via Manifold Rotation ---")
    print("Subject 5 is unanalyzed. We must mathematically discover their cognitive rotation.")
    model.eval()
    test_vecs = []
    test_labels = []
    
    with torch.no_grad():
        for subj, true_cat, x_ev, _ in test_events:
            x_ev_tensor = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0)
            pred_vec, _, _ = model(x_ev_tensor)
            test_vecs.append(pred_vec.cpu().numpy())
            test_labels.append(true_cat)
            
    test_vecs = np.array(test_vecs)
    
    # 1. Unsupervised Clustering to find Subject 5's internal "Mountains" (Ghost points)
    # We do not use the labels. We just find 8 dense clusters in their mind.
    print("Clustering Unanalyzed Subject into 8 Unknown Geometric Camps...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(test_vecs)
    sub5_centroids = kmeans.cluster_centers_
    
    # Normalizecentroids exactly like Superpositions
    sub5_centroids = sub5_centroids / np.linalg.norm(sub5_centroids, axis=1, keepdims=True)
    
    # 2. Bipartite Matching: We calculate the Euclidean distance between all 8 unknown internal 
    # Mountains and all 8 Universal Superpositions, then find the optimal matching that limits total stretch.
    print("Calculating Optimal Bipartite Shape Matching (aligning the constellation)...")
    cost_matrix = cdist(sub5_centroids, anchor_matrix, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Order Subject 5's centroids so they match the expected shape of the target matrix
    matched_sub5_centroids = sub5_centroids[row_ind]
    matched_anchors = anchor_matrix[col_ind]
    
    # 3. Orthogonal Procrustes Rotation
    # We calculate the exact mathematical Rotation Matrix (R) that optimally turns Subject 5's
    # distorted neural manifold into the Universal Superposition orientation.
    print("Computing Unsupervised Orthogonal Procrustes Rotation Matrix (R)...")
    R, scale = orthogonal_procrustes(matched_sub5_centroids, matched_anchors)
    
    # 4. Apply the discovered rotation to EVERY thought Subject 5 had
    rotated_test_vecs = np.dot(test_vecs, R)
    rotated_test_vecs = torch.tensor(rotated_test_vecs)
    
    # Normalize post-rotation
    rotated_test_vecs = F.normalize(rotated_test_vecs, p=2, dim=1)
    
    mountain_camps = {cat: [] for cat in categories}
    for label, vec in zip(test_labels, rotated_test_vecs):
        mountain_camps[label].append(vec)

    correct_mountains = 0
    print("\n--- Evaluating Subject 5's Zero-Shot Rotated Concept Recognition ---")
    
    for cat in categories:
        events = mountain_camps[cat]
        if not events: continue
        
        stacked_events = torch.stack(events)
        centroid = F.normalize(stacked_events.mean(dim=0), p=2, dim=0)
        
        centroid_sim_to_true = F.cosine_similarity(centroid.unsqueeze(0), clip_anchors[cat].cpu().unsqueeze(0)).item()
        
        best_sim = -float('inf')
        best_centroid_cat = None
        for probe_cat in categories:
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

    print(f"\nOverall Orthogonally Rotated Zero-Shot Accuracy: {(correct_mountains/len(categories))*100:.1f}%")

if __name__ == "__main__":
    unsupervised_manifold_rotation()
