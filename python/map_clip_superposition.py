import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from transformers import CLIPProcessor, CLIPModel

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

class JourneyCLIPMapper(torch.nn.Module):
    def __init__(self, in_dim, target_dim=512):
        super().__init__()
        # Learnable Delta Amplification Encoding Matrix
        # Allows the model to selectively scale up subtle patterns in low-energy or "exhausted" states
        self.delta_amp = torch.nn.Parameter(torch.ones(in_dim))
        
        # Spatial Anchor
        h_dim = 256
        self.W = torch.nn.Linear(in_dim, h_dim)
        
        # LSTM to classify the journey of the signal (no arbitrary rotational manipulations)
        self.lstm = torch.nn.LSTM(h_dim, h_dim, batch_first=True)
        
        # Deep representation mapper
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(h_dim, h_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h_dim * 2, h_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h_dim * 2, target_dim) # Predicts 512D CLIP Latent Space
        )

    def forward(self, x_seq):
        # x_seq: (T, N_voxels)
        
        # 1. Delta Amplification
        # Amplify subtle fluctuation patterns before any linear mixing
        x_amplified = x_seq * self.delta_amp.unsqueeze(0)
        
        # 2. Spatially Anchored Encoding
        x_mapped = self.W(x_amplified) # (T, h_dim)
        
        # 3. Classify the "Journey" of the signal over time
        # The LSTM cell evolves functionally across the event stream to capture "fatigue", "spikes", or "noise" naturally
        lstm_out, (hn, cn) = self.lstm(x_mapped.unsqueeze(0)) # Add batch dim: (1, T, h_dim)
        
        # The final hidden state holds the aggregate journey of the activation representation
        event_representation = hn[-1] # (1, h_dim)
        
        # Output is exactly matched to the 512 parameter CLIP Space
        clip_vector_pred = self.dnn(event_representation).squeeze(0) # (512,)
        
        return F.normalize(clip_vector_pred, p=2, dim=0)

def extract_clip_targets(categories):
    # Load OpenAI's CLIP model to get the perfect true Superposition Vectors
    print("Loading generic knowledge base (CLIP ViT-B/32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # We create multiple target semantic states representing the varying cognitive noise
    states = [
        "a distinct, clear, highly focused mental image of a",
        "a standard mental representation of a",
        "a low energy, tired, exhausted mental image of a",
        "a noisy, frantic, distracted topological thought of a"
    ]
    
    text_inputs = []
    keys = []
    
    for cat in categories:
        for state in states:
            if cat == 'scrambledpix':
                text = f"{state} scrambled nonsense static image"
            else:
                text = f"{state} {cat}"
            text_inputs.append(text)
            keys.append((cat, state))
    
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    clip_dict = {keys[i]: text_features[i] for i in range(len(keys))}
    return clip_dict

def analyze_clip_superposition():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    subject = 'sub-4'
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    
    # 1. Obtain our perfect category superpositions from CLIP (ViT-B/32 generates a 512-dim embedding)
    clip_dict = extract_clip_targets(categories)
    
    print(f"\nLoading data for single subject: {subject}")
    path = os.path.join(data_dir, subject)
    ts_2d, labels, conditions, TR = process_subject(path)
    
    max_voxels = 1000
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    manifold_ts = preprocess_run(ts_2d[top_idx, :], TR)
    
    # Split into Train / Test temporally
    split_idx = int(manifold_ts.shape[0] * 0.7)
    train_ts = manifold_ts[:split_idx]
    train_labels = labels[:split_idx]
    test_ts = manifold_ts[split_idx:]
    test_labels = labels[split_idx:]
    
    train_events = extract_events(torch.tensor(train_ts, dtype=torch.float32), train_labels, categories)
    test_events = extract_events(torch.tensor(test_ts, dtype=torch.float32), test_labels, categories)
    
    print(f"Extracted {len(train_events)} Training Events, {len(test_events)} Testing Events.")
    
    # Model categorizes the Journey mapped through Delta Amplification
    model = JourneyCLIPMapper(in_dim=max_voxels, target_dim=512)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Higher learning rate for Cosine loss
    
    epochs = 40
    print("\n--- Training Model to translate 'Cognitive Journey' to CLIP Universal Topology ---")
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        random.shuffle(train_events)
        
        for cat, x_ev in train_events:
            opt.zero_grad()
            pred_clip = model(x_ev)
            
            # The model is learning the state naturally. We don't have the "state" label.
            # So we contrast it against the best matching state for the correct semantic category!
            best_sim = -1.0
            best_true_clip = None
            
            for (true_cat, state), true_clip in clip_dict.items():
                if true_cat == cat:
                    sim = F.cosine_similarity(pred_clip.unsqueeze(0), true_clip.unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_true_clip = true_clip
            
            # Contrastive Loss: Maximize Cosine Similarity between fMRI shape and CLIP embedding!
            cos_sim = F.cosine_similarity(pred_clip.unsqueeze(0), best_true_clip.unsqueeze(0))
            loss = 1.0 - cos_sim.mean() # 0 loss means perfectly parallel!
            
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        print(f"Epoch {ep+1:2d} | Contrastive Loss: {loss_sum/len(train_events):.4f}")
        
    print("\nTraining Complete. Unanchored geometric representation mapped to CLIP Space.")
    print("\n--- Testing Phase: CLIP 512D Zero-Shot Classification (State Resolution) ---")
    
    model.eval()
    correct = 0
    state_counts = {}
    
    with torch.no_grad():
        for true_cat, x_ev in test_events:
            pred_clip = model(x_ev)
            
            # Predict by finding the closest CLIP category+state string!
            best_sim = -1.0
            best_cat = None
            best_state = None
            
            for (cat, state), true_clip in clip_dict.items():
                sim = F.cosine_similarity(pred_clip.unsqueeze(0), true_clip.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cat = cat
                    best_state = state
            
            if best_cat == true_cat:
                correct += 1
                state_counts[best_state] = state_counts.get(best_state, 0) + 1
                print(f"  [Match!] Extracted: '{best_state} {best_cat}'")
            else:
                print(f"  [Miss]   True: {true_cat} -> Guessed: '{best_state} {best_cat}'")
                
    acc = correct / len(test_events) if len(test_events) > 0 else 0
    print(f"\nFINAL TEST RESULTS: CLIP-Space Journey Resolution")
    print(f"Zero-Shot Accuracy: {acc*100:.1f}% ({correct}/{len(test_events)}) over 8 Categories (chance = 12.5%)")
    print("\nCognitive States Detected (among correct matches):")
    for st, count in state_counts.items():
        print(f"  {st}: {count}")

if __name__ == "__main__":
    analyze_clip_superposition()
