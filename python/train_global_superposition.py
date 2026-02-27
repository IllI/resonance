import os
import torch
import torch.nn.functional as F
import numpy as np
import random

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

def process_subject(subject_dir, max_runs=12):
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
    shift_trs = max(1, int(round(5.0 / TR)))
    
    all_labels = []
    run_indices = []
    for run_idx, (events, bold) in enumerate(zip(events_runs, bold_runs)):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        shifted = ['rest'] * shift_trs + run_labels[:-shift_trs] if shift_trs > 0 else run_labels
        all_labels.extend(shifted)
        run_indices.extend([run_idx] * len(shifted))
        
    return ts_2d, all_labels, run_indices, TR

def extract_dataset_events(ts_2d, labels, run_indices, categories, subject_id):
    max_voxels = 1000
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    
    N, T = ts_2d.shape
    cleaned = ts_2d[top_idx, :].copy().astype(np.float32)
    t_axis = np.arange(T, dtype=np.float32)
    for i in range(max_voxels):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    cleaned = (cleaned - mu) / sigma
    manifold_ts = np.clip(cleaned, -5.0, 5.0).T  # (T, max_voxels)
    
    events = []
    current_cat = None
    curr_start = 0
    for i, lbl in enumerate(labels):
        if lbl != current_cat:
            if current_cat in categories:
                run_idx = run_indices[curr_start]
                if run_idx < 4: phase = "Early"
                elif run_idx < 8: phase = "Mid"
                else: phase = "Late"
                events.append((subject_id, phase, current_cat, manifold_ts[curr_start:i]))
            current_cat = lbl
            curr_start = i
            
    if current_cat in categories:
        run_idx = run_indices[curr_start]
        if run_idx < 4: phase = "Early"
        elif run_idx < 8: phase = "Mid"
        else: phase = "Late"
        events.append((subject_id, phase, current_cat, manifold_ts[curr_start:]))
        
    return events

class UnifiedSuperpositionModel(torch.nn.Module):
    def __init__(self, in_dim, n_classes, latent_dim=128):
        super().__init__()
        # 1. The True Global Superpositions
        # These represent the pure categorical form of the stimulus in the semantic space
        # They are entirely subject and state independent. They are the ideal.
        self.global_superpositions = torch.nn.Parameter(torch.randn(n_classes, latent_dim))
        
        # 2. The Universal Brain Pattern Mapper
        # This takes arbitrary high-variance electrical/BOLD channels and learns to 
        # project them towards the global superpositions over a journey (LSTM)
        self.signal_encode = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, latent_dim),
            torch.nn.LayerNorm(latent_dim)
        )
        self.lstm = torch.nn.LSTM(latent_dim, latent_dim, batch_first=True)
        
    def forward(self, x_seq):
        # x_seq: (T, in_dim)
        x_mapped = self.signal_encode(x_seq)
        lstm_out, (hn, cn) = self.lstm(x_mapped.unsqueeze(0)) 
        event_vector = hn[-1].squeeze(0) # (latent_dim,)
        
        # The return is the isolated brain pattern represented as a unit directional vector
        return F.normalize(event_vector, p=2, dim=0)

    def get_superposition(self, cat_idx):
        # Returns the generic mathematical coordinate representing the pure Category
        return F.normalize(self.global_superpositions[cat_idx], p=2, dim=0)

def train_global_superpositions():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    all_events = []
    
    # We load TWO separate subjects to prove the model can learn a SINGLE global superposition
    # despite completely different anatomical markers and exhaustion profiles.
    for subj in ['sub-4', 'sub-5', 'sub-6']:
        print(f"Loading data for {subj}...")
        path = os.path.join(data_dir, subj)
        if not os.path.exists(path):
            continue
        ts_2d, labels, run_indices, TR = process_subject(path)
        evs = extract_dataset_events(ts_2d, labels, run_indices, categories, subj)
        all_events.extend(evs)
        
    print(f"Extracted {len(all_events)} Total Events across all subjects and temporal phases.")
    
    t_dim_max = max(x.shape[0] for _, _, _, x in all_events)
    def pad_events(events, target_t):
        padded = []
        for s, p, c, x in events:
            x_t = torch.tensor(x, dtype=torch.float32)
            if x_t.shape[0] < target_t:
                pad_tensor = torch.zeros((target_t - x_t.shape[0], x_t.shape[1]), dtype=x_t.dtype)
                x_t = torch.cat([x_t, pad_tensor], dim=0)
            elif x_t.shape[0] > target_t:
                x_t = x_t[:target_t]
            padded.append((s, p, c, x_t))
        return padded
        
    all_events = pad_events(all_events, t_dim_max)
    
    # LEAVE ONE SUBJECT OUT (LOSO)
    # We strictly enforce that test_events are from an entirely UNSEEN subject (sub-5).
    # This proves the emergence of the generic "Ghost Basin".
    train_events = [ev for ev in all_events if ev[0] in ['sub-4', 'sub-6']]
    test_events = [ev for ev in all_events if ev[0] == 'sub-5']
    
    print(f"Training Basin (Memorization): {len(train_events)} events from Sub-4 and Sub-6")
    print(f"Testing Basin (Generalization): {len(test_events)} events from UNSEEN Sub-5")
    
    model = UnifiedSuperpositionModel(in_dim=1000, n_classes=len(categories), latent_dim=128)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 60
    print("\n--- Training Model to map any Subject/Phase to a Unitary Category Superposition ---")
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        random.shuffle(train_events)
        
        for subj, phase, cat, x_ev in train_events:
            opt.zero_grad()
            
            # The directional vector derived from this specific isolated brain pattern
            brain_vector = model(x_ev) 
            
            # The global theoretical superposition
            target_superposition = model.get_superposition(cat_to_idx[cat])
            
            # Force the brain vector to point towards the global superposition!
            cos_sim = F.cosine_similarity(brain_vector.unsqueeze(0), target_superposition.unsqueeze(0))
            loss = 1.0 - cos_sim.mean()
            
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        if (ep + 1) % 10 == 0:
            # We track the emergence of the Ghost Basin over the Memorization Basin here!
            model.eval()
            correct = 0
            with torch.no_grad():
                for _, _, t_cat, x_test in test_events:
                    test_vec = model(x_test)
                    best_sim = -1.0
                    best_c = None
                    for pc in categories:
                        s_sim = F.cosine_similarity(test_vec.unsqueeze(0), model.get_superposition(cat_to_idx[pc]).unsqueeze(0)).item()
                        if s_sim > best_sim:
                            best_sim = s_sim
                            best_c = pc
                    if best_c == t_cat:
                        correct += 1
            test_acc = correct / len(test_events) if len(test_events) > 0 else 0
            print(f"Epoch {ep+1:2d} | Train Loss (Memorization): {loss_sum/len(train_events):.4f} | Zero-Shot Unseen Subj Acc (Generalization): {test_acc*100:.1f}%")
            model.train()
            
    print("\n--- Training Complete ---")
    
    # Let's inspect how the different conditions (Subject/Phase) align with the superpositions!
    print("\nTesting Alignment of Unseen Data to Global Superpositions...")
    model.eval()
    
    correct = 0
    phase_metrics = {"Early": [], "Mid": [], "Late": []}
    subj_metrics = {"sub-4": [], "sub-6": []}
    
    with torch.no_grad():
        for subj, phase, true_cat, x_ev in test_events:
            brain_vector = model(x_ev)
            
            best_sim = -1.0
            best_cat = None
            
            for probe_cat in categories:
                superposition = model.get_superposition(cat_to_idx[probe_cat])
                sim = F.cosine_similarity(brain_vector.unsqueeze(0), superposition.unsqueeze(0)).item()
                
                if probe_cat == true_cat:
                    # Record the distance/alignment of this specific condition to the CORRECT true superposition
                    phase_metrics[phase].append(sim)
                    subj_metrics[subj].append(sim)
                    
                if sim > best_sim:
                    best_sim = sim
                    best_cat = probe_cat
                    
            if best_cat == true_cat:
                correct += 1
                
    acc = correct / len(test_events) if len(test_events) > 0 else 0
    print(f"\nZero-Shot Categorical Inference Accuracy: {acc*100:.1f}% ({correct}/{len(test_events)}) - Chance is 12.5%")
    
    print("\n[Alignment Diagnostics: Cosine Similarity pointing towards True Superposition]")
    for ph, sims in phase_metrics.items():
        if len(sims) > 0:
            print(f"  Phase [{ph}] Mean Alignment: {np.mean(sims):.3f} (Variance: {np.var(sims):.4f}) | N={len(sims)}")
            
    for sub, sims in subj_metrics.items():
        if len(sims) > 0:
            print(f"  Subject [{sub}] Mean Alignment: {np.mean(sims):.3f} (Variance: {np.var(sims):.4f}) | N={len(sims)}")

if __name__ == "__main__":
    train_global_superpositions()
