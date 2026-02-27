import os
import torch
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
    # We return the coords_array so we can correlate abstract patterns to anatomical landmarks
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
    
    # Physical brain locations of the activation patterns
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

class BackwardSRMMapper(torch.nn.Module):
    def __init__(self, in_dim=1000, target_dim=512):
        super().__init__()
        # Instead of generic space, we map BACKWARD from the known CLIP latent anchor
        # The stream learns to resolve its noisy biology into the clean semantic target
        self.signal_encode = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, target_dim),
            torch.nn.LayerNorm(target_dim)
        )
        self.lstm = torch.nn.LSTM(target_dim, target_dim, batch_first=True)
        
    def forward(self, x_seq):
        x_mapped = self.signal_encode(x_seq)
        lstm_out, (hn, cn) = self.lstm(x_mapped.unsqueeze(0)) 
        event_vector = hn[-1].squeeze(0)
        return F.normalize(event_vector, p=2, dim=0)

def train_backward_srm():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    
    clip_anchors = extract_clip_anchors(categories)
    all_events = []
    
    # We load Subject 4, 5, 6
    for subj in ['sub-4', 'sub-5', 'sub-6']:
        print(f"\nLoading data for {subj}...")
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
    
    # LEAVE ONE SUBJECT OUT to test finding the Ghost Basin natively
    train_events = [ev for ev in all_events if ev[0] in ['sub-4', 'sub-6']]
    test_events = [ev for ev in all_events if ev[0] == 'sub-5']
    
    print(f"Training Basin (Mapping towards Superposition Anchors): {len(train_events)} events")
    print(f"Testing Basin (Unseen Gen): {len(test_events)} events from Sub-5")
    
    model = BackwardSRMMapper(in_dim=1000, target_dim=512)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 40
    print("\n--- Training Model to Anchor Streams to Known Superpositions ---")
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        random.shuffle(train_events)
        
        for subj, cat, x_ev, _ in train_events:
            opt.zero_grad()
            pred_vec = model(x_ev)
            true_anchor = clip_anchors[cat]
            
            # Distance mapping to the true absolute anchor
            loss = 1.0 - F.cosine_similarity(pred_vec.unsqueeze(0), true_anchor.unsqueeze(0)).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        print(f"Epoch {ep+1:2d} | Anchor Mapping Loss: {loss_sum/len(train_events):.4f}")
            
    print("\n--- Validation and BOLD ROI Correlation ---")
    
    model.eval()
    correct = 0
    cat_anatomies = {c: [] for c in categories}
    
    with torch.no_grad():
        for subj, true_cat, x_ev, coords in test_events:
            pred_vec = model(x_ev)
            
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
                # The prediction correctly matched the ghost basin!
                # We reward/track the anatomical ROI coordinates that produced this successful expression!
                centroid = coords.mean(axis=0) # [X, Y, Z] center of the 1000 voxels
                cat_anatomies[true_cat].append(centroid)
                
    acc = correct / len(test_events) if len(test_events) > 0 else 0
    print(f"\nZero-Shot Unseen Subject Categorization: {acc*100:.1f}% ({correct}/{len(test_events)}) - Chance is 12.5%")
    
    print("\n[Anatomical Landmark Correlation for Correct Ghost Basin Matches]")
    print("These reflect the physical BOLD center-of-mass that successfully generated the conceptual superposition:")
    for cat in categories:
        coords_list = cat_anatomies[cat]
        if coords_list:
            mean_coord = np.mean(coords_list, axis=0)
            print(f"  Category '{cat}' successfully anchored from mean anatomy XYZ: [{mean_coord[0]:.1f}, {mean_coord[1]:.1f}, {mean_coord[2]:.1f}]")
        else:
            print(f"  Category '{cat}' had no successful zero-shot anchors in Sub-5.")

if __name__ == "__main__":
    train_backward_srm()
