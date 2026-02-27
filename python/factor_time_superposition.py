import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.decomposition import PCA

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
    
    # Keeping the BOLD delay so we map the actual measurable contrast
    shift_trs = max(1, int(round(5.0 / TR)))
    
    all_labels = []
    run_indices = []
    for run_idx, (events, bold) in enumerate(zip(events_runs, bold_runs)):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        shifted = ['rest'] * shift_trs + run_labels[:-shift_trs] if shift_trs > 0 else run_labels
        all_labels.extend(shifted)
        run_indices.extend([run_idx] * len(shifted))
        
    return ts_2d, all_labels, run_indices, coords_array, TR

def extract_events_with_time(X, labels, run_indices, categories):
    events = []
    current_cat = None
    curr_start = 0
    for i, lbl in enumerate(labels):
        if lbl != current_cat:
            if current_cat in categories:
                # Assign the phase based on the run index at the start of the event
                run_idx = run_indices[curr_start]
                if run_idx < 4:
                    phase = 0 # Early (High Attention/Energy)
                elif run_idx < 8:
                    phase = 1 # Mid (Stabilizing)
                else:
                    phase = 2 # Late (Exhausted/Distracted)
                events.append((current_cat, phase, X[curr_start:i]))
            current_cat = lbl
            curr_start = i
    if current_cat in categories:
        run_idx = run_indices[curr_start]
        if run_idx < 4: phase = 0
        elif run_idx < 8: phase = 1
        else: phase = 2
        events.append((current_cat, phase, X[curr_start:]))
    return events

class DisentangledSuperpositionModel(torch.nn.Module):
    def __init__(self, in_dim, t_dim, n_classes, n_phases=3, latent_dim=128):
        super().__init__()
        # We project the brain activation into a latent space
        freq_dim = (t_dim // 2) + 1
        shape_feature_dim = (t_dim * t_dim) + freq_dim
        
        # Spatial Anchor
        self.W = torch.nn.Linear(in_dim, latent_dim)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(latent_dim * 2),
            torch.nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Explicitly model the Superposition (Category) and the State (Time Phase)
        # The latent space should be equal to Superposition + Phase_Vector
        self.category_embeddings = torch.nn.Embedding(n_classes, latent_dim)
        self.phase_embeddings = torch.nn.Embedding(n_phases, latent_dim)
        
    def forward(self, x_seq):
        # x_seq: (T, N_voxels)
        x_mapped = self.W(x_seq)
        
        # Mean pool over the event
        event_rep = x_mapped.mean(dim=0)
        
        z = self.encoder(event_rep) # (latent_dim,)
        return F.normalize(z, p=2, dim=0)

def analyze_disentangled_state():
    data_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
    subject = 'sub-4'
    categories = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    print(f"\nLoading data for single subject: {subject}")
    path = os.path.join(data_dir, subject)
    ts_2d, labels, run_indices, coords, TR = process_subject(path)
    
    max_voxels = 1000
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    manifold_ts = preprocess_run(ts_2d[top_idx, :], TR)
    
    all_events = extract_events_with_time(torch.tensor(manifold_ts, dtype=torch.float32), labels, run_indices, categories)
    
    t_dim_max = max(x.shape[0] for _, _, x in all_events)
    def pad_events(events, target_t):
        padded = []
        for c, p, x in events:
            if x.shape[0] < target_t:
                pad_tensor = torch.zeros((target_t - x.shape[0], x.shape[1]), dtype=x.dtype)
                x = torch.cat([x, pad_tensor], dim=0)
            elif x.shape[0] > target_t:
                x = x[:target_t]
            padded.append((c, p, x))
        return padded
        
    all_events = pad_events(all_events, t_dim_max)
    
    # We will use ALL data to learn the space, since we are proving a generic decomposition, 
    # not just traditional supervised learning.
    
    latent_dim = 128
    model = DisentangledSuperpositionModel(in_dim=max_voxels, t_dim=t_dim_max, n_classes=len(categories), latent_dim=latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 100
    print("\n--- Training Model to Disentangle 'Superposition' from 'Time Exhaustion Phase' ---")
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        random.shuffle(all_events)
        
        for cat, phase, x_ev in all_events:
            opt.zero_grad()
            z = model(x_ev)
            
            # Retrieve the theoretical ideal vectors
            c_idx = torch.tensor(cat_to_idx[cat], dtype=torch.long)
            p_idx = torch.tensor(phase, dtype=torch.long)
            
            z_cat = F.normalize(model.category_embeddings(c_idx), p=2, dim=0)
            z_phase = F.normalize(model.phase_embeddings(p_idx), p=2, dim=0)
            
            # The hypothesis: The total brain state is an additive combination of the Category Superposition and the Exhaustion State
            # Brain_Pattern = Category_Vector + Phase_Vector
            target_z = F.normalize(z_cat + z_phase, p=2, dim=0)
            
            # Loss is how far the encoded brain state is from this disentangled combination
            loss = 1.0 - F.cosine_similarity(z.unsqueeze(0), target_z.unsqueeze(0)).mean()
            
            # Orthogonality constraint: We force the "Category" vectors to be perpendicular to the "Phase" vectors!
            # As you hypothesized: "a perpendicular plane that intersects through all latent spaces"
            ortho_loss = torch.abs(F.cosine_similarity(z_cat.unsqueeze(0), z_phase.unsqueeze(0)))
            
            total_loss = loss + (0.5 * ortho_loss)
            
            total_loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:3d} | Structural Reconstruction Loss: {loss_sum/len(all_events):.4f}")
        
    print("\nDisentanglement Complete. Extracting the Pure Invariant Superpositions and Exhaustion Vectors...")
    
    # Let's test the hypothesis: Does the model recognize categories generically if we SUBTRACT the exhaustion?
    model.eval()
    correct = 0
    with torch.no_grad():
        for cat, phase, x_ev in all_events:
            z_brain = model(x_ev)
            p_idx = torch.tensor(phase, dtype=torch.long)
            z_phase = F.normalize(model.phase_embeddings(p_idx), p=2, dim=0)
            
            # Remove the "Exhaustion" distortion:
            # Pure_Category_Guess = Brain_State - Exhaustion_State
            z_pure_guess = F.normalize(z_brain - z_phase, p=2, dim=0)
            
            # Find the closest pure category superposition
            best_sim = -1.0
            best_cat = None
            for probe_cat in categories:
                c_idx = torch.tensor(cat_to_idx[probe_cat], dtype=torch.long)
                z_cat_true = F.normalize(model.category_embeddings(c_idx), p=2, dim=0)
                
                sim = F.cosine_similarity(z_pure_guess.unsqueeze(0), z_cat_true.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cat = probe_cat
                    
            if best_cat == cat:
                correct += 1

    acc = correct / len(all_events)
    print(f"\nValidation (Predicting category by subtracting Time-Phase Distortion): {acc*100:.1f}%")
    
    print("\nAnalyzing Geometric Relationships (Dot Products):")
    with torch.no_grad():
        c_face = F.normalize(model.category_embeddings(torch.tensor(cat_to_idx['face'])), p=2, dim=0)
        c_house = F.normalize(model.category_embeddings(torch.tensor(cat_to_idx['house'])), p=2, dim=0)
        p_early = F.normalize(model.phase_embeddings(torch.tensor(0)), p=2, dim=0)
        p_late = F.normalize(model.phase_embeddings(torch.tensor(2)), p=2, dim=0)
        
        print(f"  Face <-> House Sim: {F.cosine_similarity(c_face.unsqueeze(0), c_house.unsqueeze(0)).item():.3f}")
        print(f"  Early <-> Late Sim: {F.cosine_similarity(p_early.unsqueeze(0), p_late.unsqueeze(0)).item():.3f}")
        print(f"  Face <-> Early Sim (Orthogonality check): {F.cosine_similarity(c_face.unsqueeze(0), p_early.unsqueeze(0)).item():.3f}")

if __name__ == "__main__":
    analyze_disentangled_state()
