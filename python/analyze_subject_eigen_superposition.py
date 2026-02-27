import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import svd
import random

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

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

class UnanchoredShapeDNN(torch.nn.Module):
    def __init__(self, t_dim, n_classes):
        super().__init__()
        # The 'Shape' is defined by the Temporal Topological Gram Matrix (T x T)
        # and the Frequency Spectrum Magnitude (T/2 + 1)
        # No spatially anchored W weights are used!
        
        freq_dim = (t_dim // 2) + 1
        shape_feature_dim = (t_dim * t_dim) + freq_dim
        
        h_dim = 128
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(shape_feature_dim, h_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LayerNorm(h_dim)
        )
        self.classifier = torch.nn.Linear(h_dim, n_classes)

    def forward(self, x_seq):
        # x_seq: (T, N_voxels)
        # 1. Unanchored Abstract Geometry (Temporal Gram Matrix)
        # How the floating blob transitions state-to-state over time, marginalized over space.
        x_norm = F.normalize(x_seq, p=2, dim=1)
        gram_matrix = torch.matmul(x_norm, x_norm.transpose(0, 1)) # (T, T)
        
        # 2. Unanchored Spiking Frequency (Magnitude Spectrum)
        # The pure electromagnetic frequency amplitudes, invariant to phase/temporal shifts
        freq_amps = torch.abs(torch.fft.rfft(x_seq, dim=0)).mean(dim=1) # (T/2 + 1)
        
        # Combine the pure shapes
        unanchored_shape = torch.cat([gram_matrix.flatten(), freq_amps.flatten()], dim=0)
        
        # Process through DNN
        event_representation = self.dnn(unanchored_shape) # (h_dim,)
        logits = self.classifier(event_representation)
        
        return logits, event_representation

def analyze_eigen_superposition():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    subject = 'sub-4'
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    print(f"Loading data for single subject: {subject}")
    path = os.path.join(data_dir, subject)
    ts_2d, labels, conditions, TR = process_subject(path)
    
    max_voxels = 1000
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    manifold_ts = preprocess_run(ts_2d[top_idx, :], TR)
    
    # Split into Train / Test temporally (first 70% train, last 30% test)
    split_idx = int(manifold_ts.shape[0] * 0.7)
    train_ts = manifold_ts[:split_idx]
    train_labels = labels[:split_idx]
    test_ts = manifold_ts[split_idx:]
    test_labels = labels[split_idx:]
    
    train_events = extract_events(torch.tensor(train_ts, dtype=torch.float32), train_labels, categories)
    test_events = extract_events(torch.tensor(test_ts, dtype=torch.float32), test_labels, categories)
    
    print(f"Extracted {len(train_events)} Training Events, {len(test_events)} Testing Events.")
    
    # Find the generic T_dim block length (since blocks are padded/extracted to roughly similar lengths)
    t_dim_max = max(x.shape[0] for _, x in train_events)
    
    # Pad events to a uniform t_dim for the static Gram Matrix calculation
    def pad_events(events, target_t):
        padded = []
        for c, x in events:
            if x.shape[0] < target_t:
                pad_tensor = torch.zeros((target_t - x.shape[0], x.shape[1]), dtype=x.dtype)
                x = torch.cat([x, pad_tensor], dim=0)
            elif x.shape[0] > target_t:
                x = x[:target_t]
            padded.append((c, x))
        return padded
        
    train_events = pad_events(train_events, t_dim_max)
    test_events = pad_events(test_events, t_dim_max)
    
    model = UnanchoredShapeDNN(t_dim=t_dim_max, n_classes=len(categories))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    epochs = 40
    print("\n--- Training Model & Tracking Loss of 'Ghost in the Box' Unanchored Representation ---")
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        random.shuffle(train_events)
        
        for cat, x_ev in train_events:
            opt.zero_grad()
            logits, _ = model(x_ev)
            target = torch.tensor(cat_to_idx[cat], dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        print(f"Epoch {ep+1:2d} | Unanchored Shape Loss: {loss_sum/len(train_events):.4f}")
        
    print("\nTraining Complete. Unanchored geometric representation learned.")
    print("\n--- Testing Phase: Evaluating Superposition Robustness ---")
    
    def evaluate_events(events, label_modifier="Standard"):
        model.eval()
        correct = 0
        with torch.no_grad():
            for true_cat, x_ev in events:
                logits, _ = model(x_ev)
                pred_idx = logits.argmax().item()
                if categories[pred_idx] == true_cat:
                    correct += 1
        acc = correct / len(events) if len(events) > 0 else 0
        print(f"[{label_modifier} Playback] Accuracy: {acc*100:.1f}% ({correct}/{len(events)})")

    # 1. Standard Testing Playback
    evaluate_events(test_events, "Standard Ordered")
    
    # 2. Mixed up / Shuffled Events Playback
    shuffled_test_events = test_events.copy()
    random.shuffle(shuffled_test_events)
    evaluate_events(shuffled_test_events, "Shuffled Sequence")
    
    # 3. Reversed Playback (Playing the stimulus fMRI time series backwards)
    reversed_test_events = []
    for cat, x_ev in test_events:
        reversed_ev = torch.flip(x_ev, dims=[0])
        reversed_test_events.append((cat, reversed_ev))
    evaluate_events(reversed_test_events, "Time-Reversed")

if __name__ == "__main__":
    analyze_eigen_superposition()
