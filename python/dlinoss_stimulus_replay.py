"""
dlinoss_stimulus_replay.py
──────────────────────────────────────────────────────────────────────────────
Stimulus-by-Stimulus Replay of the Haxby Experiment

Philosophy
──────────
Instead of training the model on CATEGORY LABELS, we train it to learn the 
electromagnetic dynamics of each INDIVIDUAL STIMULUS EVENT. The category 
structure is something we discover AFTERWARD — exactly as Haxby et al. did.

The Haxby 2001 Paper's Breakthrough
────────────────────────────────────
Haxby et al. showed that category information is DISTRIBUTED across ventral
temporal cortex — not locked to specialized regions like FFA or PPA. The key
finding was:

  1. Even-odd run split: train on even runs, test on odd runs
  2. Within-category correlation > between-category correlation
  3. This held EVEN when maximally-responsive voxels (FFA/PPA) were EXCLUDED
  4. 96% accuracy in pairwise category discrimination

This means the "face network" and "house network" are NOT just FFA and PPA.
They are distributed, overlapping patterns across the ENTIRE ventral temporal
cortex. Every voxel carries SOME information about EVERY category — this is
exactly the definition of SUPERPOSITION.

Our Approach: Replay the Experiment
────────────────────────────────────
1. For each subject, replay the experiment stimulus-by-stimulus:
   - 12 stimuli per block × 8 blocks per run × 12 runs = 1152 stimulus events
   - Each stimulus is 500ms ON, 1500ms ISI (2s per stimulus, within 2.5s TR)
   
2. Train D-LinOSS to predict the NEXT neural state from the CURRENT one
   (self-supervised, no labels needed). The model learns what neural dynamics
   look like when processing visual stimuli.

3. Extract the model's internal oscillator states during each stimulus.
   These states encode the spectral EM patterns of the brain's response.

4. THEN — and only then — ask: do the learned states cluster by category?
   If yes, the model has discovered the distributed category representation
   that Haxby found.

5. Use the Haxby findings to VALIDATE: the learned "face" subspace should
   overlap with known FFA coordinates, and "house" should overlap with PPA.
   But the model should ALSO capture the distributed component that exists
   OUTSIDE these ROIs.

The Damping Model
─────────────────
The D-LinOSS damping matrix G is trained to suppress non-stimulus-related 
dynamics (scanner noise, physiological drift, motor preparation). What's
LEFT after damping is the pure stimulus-driven oscillation — the recognition
wave. We can then characterize what the damped-out signal looked like to 
understand what the model learned NOT to look for.

Pipeline
────────
[BOLD per-run] → stimulus-locked event extraction (per individual image)
             → D-LinOSS self-supervised training (predict next state)
             → Extract oscillator states per stimulus event
             → Haxby-style even/odd split correlation analysis  
             → Category emerges from correlation structure
             → Validate against FFA/PPA ROI coordinates
             → Characterize damped-out signal (what NOT to look for)

Usage
────────
  python dlinoss_stimulus_replay.py
"""

import os, math, random, sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Constants & Haxby experimental parameters
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR        = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS        = [f'sub-{i}' for i in range(1, 7)]
TARGET_CATS     = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'shoe']
ALL_CATS        = TARGET_CATS + ['scrambledpix']

# Haxby experimental timing
STIM_DURATION_S = 0.5           # Image shown for 500ms
ISI_S           = 1.5           # Inter-stimulus interval
TRIAL_S         = 2.0           # Total trial = stim + ISI
BLOCK_DURATION_S = 24.0         # 12 stimuli × 2s
REST_DURATION_S  = 12.0         # Rest between blocks
N_STIM_PER_BLOCK = 12
HRF_PEAK_TRS     = 2            # HRF peaks ~5s after stimulus (2 TRs at TR=2.5s)
HRF_WINDOW_TRS   = 2            # Average this many TRs for each event

# Model parameters  
IN_DIM          = 1000          # top-variance voxels
DLINOSS_STATE   = 32
DLINOSS_HIDDEN  = 64
N_EPOCHS        = 30
LR              = 1e-3

# Known ROI coordinates from Haxby et al. (in MNI-ish voxel space)
# These are approximate — the paper reports Talairach coordinates
# FFA: ~(39, -55, -20) in Talairach → roughly mid-fusiform gyrus
# PPA: ~(27, -41, -10) in Talairach → parahippocampal gyrus
# We'll validate by checking if our model's face/house subspaces
# have peak activation near these regions
KNOWN_ROIS = {
    'face': {
        'name': 'Fusiform Face Area (FFA)',
        'talairach': np.array([39, -55, -20]),
        'description': 'Right fusiform gyrus, maximally responsive to faces',
    },
    'house': {
        'name': 'Parahippocampal Place Area (PPA)', 
        'talairach': np.array([27, -41, -10]),
        'description': 'Parahippocampal gyrus, maximally responsive to houses/places',
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass 
class StimulusEvent:
    """A single stimulus presentation within the experiment."""
    category: str               # e.g. 'face', 'cat', 'scrambledpix'
    onset_s: float              # Onset time in seconds (within run)
    onset_tr: int               # Which TR this stimulus falls in
    run_id: int                 # Which run (0-11)
    block_idx: int              # Which block within the run (0-7)
    stim_idx_in_block: int      # Position within block (0-11)
    
    # Is this a 1-back repeat? (same IMAGE as previous stimulus)
    is_repeat: bool = False
    
    # The BOLD data at this TR [V] (extracted after preprocessing)
    bold_vector: Optional[np.ndarray] = None
    
    # Coordinates of active voxels
    coords: Optional[np.ndarray] = None


@dataclass
class SubjectReplay:
    """Complete stimulus-by-stimulus replay of one subject's experiment."""
    subject_id: str
    TR: float
    events: List[StimulusEvent] = field(default_factory=list)
    
    # Preprocessed BOLD timeseries [T_total, V]
    manifold_ts: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    
    # Model-derived oscillator states per event [n_events, state_dim]
    oscillator_states: Optional[np.ndarray] = None
    
    # Category-responsive voxels (Haxby-style F-test selection)
    cat_responsive_idx: Optional[np.ndarray] = None
    cat_responsive_bold: Optional[np.ndarray] = None  # Per-event [n_events, V_responsive]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Experiment replay: extract individual stimulus events
# ──────────────────────────────────────────────────────────────────────────────

def replay_experiment(subject_dir: str, subject_id: str, 
                       max_runs: int = 12) -> SubjectReplay:
    """
    Replay the entire Haxby experiment for one subject, extracting
    the BOLD response at EACH individual stimulus presentation.
    
    The Haxby design:
      - 12 runs per subject
      - 8 category blocks per run (randomized order)
      - Each block: 12 stimuli × 2s = 24s
      - Stimulus: 500ms image, 1500ms ISI
      - Rest: 12s between blocks
      - 1-back task: different views of same face/object
    """
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    
    # Global voxel selection
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
    max_voxels = IN_DIM
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]
    coords = coords_array[top_idx]
    
    replay = SubjectReplay(subject_id=subject_id, TR=TR, coords=coords)
    
    # Process each run
    running_tr_offset = 0
    all_manifold = []
    
    for run_idx, (bold_data, events) in enumerate(zip(bold_runs, events_runs)):
        run_T = bold_data.shape[-1]
        
        # Preprocess this run
        ts_run = ts_2d[top_idx, running_tr_offset:running_tr_offset + run_T]
        manifold_run = _preprocess_run(ts_run)  # [T_run, V]
        all_manifold.append(manifold_run)
        
        if not events:
            running_tr_offset += run_T
            continue
        
        # Parse individual stimulus events
        prev_cat = None
        block_idx = -1
        stim_in_block = 0
        
        for ev in events:
            onset_s = ev['onset']
            cat = ev['trial_type']
            
            # Detect block boundaries
            if cat != prev_cat:
                block_idx += 1
                stim_in_block = 0
                prev_cat = cat
            
            # Map stimulus onset to TR, then shift by HRF delay
            onset_tr = int(round(onset_s / TR))
            hrf_tr = onset_tr + HRF_PEAK_TRS  # Where the BOLD response actually peaks
            
            if hrf_tr < 0 or hrf_tr >= run_T:
                stim_in_block += 1
                continue
            
            # Extract the BOLD vector averaged over the HRF window
            # This captures the hemodynamic response to this stimulus
            hrf_end = min(hrf_tr + HRF_WINDOW_TRS, run_T)
            bold_vec = manifold_run[hrf_tr:hrf_end].mean(axis=0)
            
            stimulus = StimulusEvent(
                category=cat,
                onset_s=onset_s,
                onset_tr=running_tr_offset + onset_tr,  # global TR
                run_id=run_idx,
                block_idx=block_idx,
                stim_idx_in_block=stim_in_block,
                bold_vector=bold_vec,
                coords=coords,
            )
            
            replay.events.append(stimulus)
            stim_in_block += 1
        
        running_tr_offset += run_T
    
    # Concatenate all manifold data
    replay.manifold_ts = np.concatenate(all_manifold, axis=0)
    
    print(f"  {subject_id}: {len(replay.events)} stimulus events across "
          f"{max(e.run_id for e in replay.events) + 1} runs, "
          f"TR={TR:.2f}s")
    
    # Count per category
    cat_counts = defaultdict(int)
    for ev in replay.events:
        cat_counts[ev.category] += 1
    for cat in sorted(cat_counts.keys()):
        print(f"    {cat}: {cat_counts[cat]} events")
    
    # ── Haxby-style category-responsive voxel selection ──
    # Instead of top-variance (which includes motor cortex, scanner noise),
    # select voxels whose BOLD response differs across categories (F-test)
    _select_category_responsive_voxels(replay)
    
    return replay


def _preprocess_run(ts_2d: np.ndarray) -> np.ndarray:
    """Detrend + z-score [V, T] → [T, V]."""
    V, T = ts_2d.shape
    cleaned = ts_2d.copy().astype(np.float32)
    t_axis = np.arange(T, dtype=np.float32)
    for i in range(V):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return np.clip((cleaned - mu) / sigma, -5.0, 5.0).T


def _select_category_responsive_voxels(replay: SubjectReplay,
                                          max_voxels: int = 500,
                                          p_threshold: float = 1e-4):
    """
    Haxby's key methodological choice: select voxels whose response 
    DIFFERS SIGNIFICANTLY across categories. This restricts analysis to
    "object-selective" cortex — approximately the ventral temporal cortex.
    
    Uses a one-way ANOVA F-test across categories.
    This is what made the original paper work — by excluding voxels that 
    respond the same to everything, you're left with only the voxels
    that carry category information.
    """
    from scipy.stats import f_oneway
    
    # Group BOLD vectors by category
    cat_bold = defaultdict(list)
    for ev in replay.events:
        if ev.bold_vector is not None and ev.category in ALL_CATS:
            cat_bold[ev.category].append(ev.bold_vector)
    
    if len(cat_bold) < 2:
        print("    [warn] Not enough categories for F-test")
        return
    
    # Stack all BOLD vectors per category
    cat_arrays = {cat: np.stack(vecs) for cat, vecs in cat_bold.items()}
    V = list(cat_arrays.values())[0].shape[1]
    
    # Per-voxel F-test across categories
    f_stats = np.zeros(V)
    p_values = np.ones(V)
    
    groups = [cat_arrays[cat] for cat in sorted(cat_arrays.keys())]
    
    for v in range(V):
        # Get this voxel's values for each category  
        voxel_groups = [g[:, v] for g in groups]
        try:
            f_stat, p_val = f_oneway(*voxel_groups)
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
                p_values[v] = p_val
        except Exception:
            pass
    
    # Select voxels below p-threshold, up to max
    significant = np.where(p_values < p_threshold)[0]
    
    if len(significant) == 0:
        # Fallback: take top-max_voxels by F-statistic
        significant = np.argsort(f_stats)[-max_voxels:]
        print(f"    F-test: no voxels at p<{p_threshold}, "
              f"using top {max_voxels} by F-stat")
    elif len(significant) > max_voxels:
        # Sort by F-stat and take top
        top_within = np.argsort(f_stats[significant])[-max_voxels:]
        significant = significant[top_within]
        print(f"    F-test: {len(significant)} voxels at p<{p_threshold}, "
              f"kept top {max_voxels}")
    else:
        print(f"    F-test: {len(significant)} category-responsive voxels "
              f"(p<{p_threshold})")
    
    replay.cat_responsive_idx = significant
    
    # Extract category-responsive BOLD for each event
    responsive_bold = []
    for ev in replay.events:
        if ev.bold_vector is not None:
            responsive_bold.append(ev.bold_vector[significant])
        else:
            responsive_bold.append(np.zeros(len(significant)))
    replay.cat_responsive_bold = np.stack(responsive_bold)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  D-LinOSS self-supervised training (no labels!)
# ──────────────────────────────────────────────────────────────────────────────

class StimulusDLinOSS(nn.Module):
    """
    Self-supervised D-LinOSS model for stimulus-driven dynamics.
    
    Trained to predict the NEXT neural state from the current one.
    The damping matrix G learns to suppress scanner noise, physiological
    drift, and non-stimulus-related activity. What remains after damping
    is the pure stimulus-driven oscillation.
    """
    def __init__(self, in_dim=IN_DIM, state_dim=DLINOSS_STATE, 
                 hidden_dim=DLINOSS_HIDDEN, n_blocks=3):
        super().__init__()
        self.dlinoss = DLinOSSModel(
            input_dim=in_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=in_dim,
            num_blocks=n_blocks,
            drop_rate=0.1,
        )
        # Reconstruction head: predict next TR from current oscillator state
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
    
    def forward(self, x_seq):
        """
        x_seq: [B, T, V] — sequence of BOLD vectors
        Returns:
            osc_states: [B, T, V] — oscillator output states
            predictions: [B, T, V] — predicted next states
        """
        osc_states = self.dlinoss(x_seq)  # [B, T, V]
        predictions = self.predictor(osc_states)
        return osc_states, predictions
    
    def get_damping_profile(self):
        """Extract what the model has learned to suppress."""
        damping_weights = {}
        for name, param in self.dlinoss.named_parameters():
            if 'gamma' in name.lower() or 'damp' in name.lower():
                damping_weights[name] = param.detach().cpu().numpy()
        return damping_weights


def train_self_supervised(replay: SubjectReplay, device: str = 'cpu',
                           n_epochs: int = N_EPOCHS) -> StimulusDLinOSS:
    """
    Train D-LinOSS in a completely self-supervised manner.
    
    The model learns to predict the next TR from the current one.
    No labels are used. The damping matrix learns to filter out
    non-stimulus-related dynamics on its own.
    """
    model = StimulusDLinOSS(in_dim=IN_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    ts = replay.manifold_ts  # [T_total, V]
    T_total = ts.shape[0]
    
    # Create training sequences: sliding windows of 10 TRs
    window_size = 10
    sequences = []
    for start in range(0, T_total - window_size, window_size // 2):
        seq = ts[start:start + window_size]
        sequences.append(torch.tensor(seq, dtype=torch.float32))
    
    print(f"\n  Training self-supervised D-LinOSS ({len(sequences)} sequences)...")
    
    for epoch in range(n_epochs):
        random.shuffle(sequences)
        total_loss = 0.0
        
        for seq in sequences:
            x = seq.unsqueeze(0).to(device)  # [1, T, V]
            
            osc_states, predictions = model(x)
            
            # Self-supervised loss: predict next TR from current
            # predictions[:, t, :] should predict x[:, t+1, :]
            target = x[:, 1:, :]                    # [1, T-1, V]
            pred   = predictions[:, :-1, :]          # [1, T-1, V]
            
            loss = F.mse_loss(pred, target)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(sequences)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: loss = {avg_loss:.6f}")
    
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Extract oscillator states per stimulus event
# ──────────────────────────────────────────────────────────────────────────────

def extract_oscillator_responses(replay: SubjectReplay, model: StimulusDLinOSS,
                                   device: str = 'cpu') -> np.ndarray:
    """
    For each stimulus event, extract the D-LinOSS oscillator state.
    
    This is the spectral EM pattern that the model has learned to associate
    with processing that particular stimulus. The damping matrix has already
    filtered out scanner noise, drift, and motor artifacts.
    """
    model.eval()
    states = []
    
    # Process in context windows around each stimulus
    context_trs = 5  # TRs of context before/after the stimulus
    
    with torch.no_grad():
        for ev in replay.events:
            tr = ev.onset_tr
            start = max(0, tr - context_trs)
            end = min(replay.manifold_ts.shape[0], tr + context_trs + 1)
            
            if end - start < 3:
                states.append(np.zeros(IN_DIM))
                continue
            
            window = replay.manifold_ts[start:end]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            
            osc_states, _ = model(x)
            
            # The oscillator state at the stimulus TR
            stim_idx = tr - start
            if stim_idx < osc_states.shape[1]:
                state = osc_states[0, stim_idx].cpu().numpy()
            else:
                state = osc_states[0, -1].cpu().numpy()
            
            states.append(state)
    
    replay.oscillator_states = np.stack(states)
    return replay.oscillator_states


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Haxby-style even/odd split correlation analysis
# ──────────────────────────────────────────────────────────────────────────────

def haxby_correlation_analysis(replay: SubjectReplay, 
                                 use_oscillator: bool = True,
                                 block_level: bool = False) -> Dict:
    """
    Replicate the exact analysis from Haxby et al. 2001:
    
    1. Split data into even runs and odd runs
    2. Compute average response pattern per category in each half
    3. Within-category correlation: e.g., corr(face_even, face_odd)
    4. Between-category correlation: e.g., corr(face_even, house_odd)
    5. If within > between for all categories → distributed representation exists
    
    When block_level=True, we first AVERAGE all stimuli within each block
    (12 stimuli per block), matching Haxby's original approach. This is
    critical because it dramatically improves SNR.
    
    This is the gold standard that made the paper groundbreaking.
    """
    categories = ALL_CATS
    
    # Split events by even/odd runs
    even_events = [ev for ev in replay.events if ev.run_id % 2 == 0]
    odd_events  = [ev for ev in replay.events if ev.run_id % 2 == 1]
    
    def get_category_patterns(events, use_osc, use_responsive=False):
        """Compute average pattern per category."""
        patterns = {}
        for cat in categories:
            cat_events = [ev for ev in events if ev.category == cat]
            if not cat_events:
                continue
            
            if use_responsive and replay.cat_responsive_bold is not None:
                # Use category-responsive voxels (Haxby's actual method)
                indices = [replay.events.index(ev) for ev in cat_events]
                vecs = replay.cat_responsive_bold[indices]
            elif use_osc and replay.oscillator_states is not None:
                indices = [replay.events.index(ev) for ev in cat_events]
                vecs = replay.oscillator_states[indices]
            else:
                vecs = np.stack([ev.bold_vector for ev in cat_events 
                                if ev.bold_vector is not None])
            
            if len(vecs) > 0:
                if block_level:
                    block_groups = defaultdict(list)
                    for ev_i, ev in enumerate(cat_events):
                        block_key = (ev.run_id, ev.block_idx)
                        block_groups[block_key].append(vecs[ev_i])
                    block_avgs = [np.stack(v).mean(axis=0) for v in block_groups.values()]
                    patterns[cat] = np.stack(block_avgs).mean(axis=0)
                else:
                    patterns[cat] = vecs.mean(axis=0)
        
        return patterns
    
    # When using category-responsive voxels, force use_responsive mode
    use_responsive = (not use_oscillator) and (replay.cat_responsive_bold is not None)
    even_patterns = get_category_patterns(even_events, use_oscillator, use_responsive)
    odd_patterns  = get_category_patterns(odd_events, use_oscillator, use_responsive)
    
    # Compute correlation matrix
    cats_available = sorted(set(even_patterns.keys()) & set(odd_patterns.keys()))
    n_cats = len(cats_available)
    
    corr_matrix = np.zeros((n_cats, n_cats))
    for i, cat_even in enumerate(cats_available):
        for j, cat_odd in enumerate(cats_available):
            r, _ = pearsonr(even_patterns[cat_even], odd_patterns[cat_odd])
            corr_matrix[i, j] = r
    
    # Within-category vs between-category
    within_corrs = {}
    between_corrs = {}
    
    for i, cat in enumerate(cats_available):
        within_corrs[cat] = corr_matrix[i, i]
        between_vals = [corr_matrix[i, j] for j in range(n_cats) if j != i]
        between_corrs[cat] = np.mean(between_vals) if between_vals else 0.0
    
    # Accuracy: for each even-run category, does it match best with same odd-run category?
    correct = 0
    for i, cat in enumerate(cats_available):
        best_match = cats_available[np.argmax(corr_matrix[i])]
        if best_match == cat:
            correct += 1
    
    accuracy = correct / n_cats if n_cats > 0 else 0
    
    level_str = " [BLOCK-AVG]" if block_level else " [STIM-LEVEL]"
    mode = ("Oscillator" if use_oscillator else "Raw BOLD") + level_str
    
    results = {
        'mode': mode,
        'correlation_matrix': corr_matrix,
        'categories': cats_available,
        'within_correlations': within_corrs,
        'between_correlations': between_corrs,
        'pairwise_accuracy': accuracy,
    }
    
    return results


def print_haxby_results(results: Dict, subject_id: str):
    """Pretty-print the Haxby correlation analysis results."""
    mode = results['mode']
    cats = results['categories']
    within = results['within_correlations']
    between = results['between_correlations']
    accuracy = results['pairwise_accuracy']
    
    print(f"\n  ── Haxby Even/Odd Correlation Analysis ({mode}) ──")
    print(f"  Subject: {subject_id}")
    print(f"\n  {'Category':<14}  {'Within-r':<10}  {'Between-r':<10}  {'Δ':<8}  {'Win?'}")
    print(f"  {'─'*52}")
    
    n_wins = 0
    for cat in cats:
        w = within[cat]
        b = between[cat]
        delta = w - b
        win = "✓" if delta > 0 else "✗"
        if delta > 0:
            n_wins += 1
        print(f"  {cat:<14}  {w:>8.4f}    {b:>8.4f}    {delta:>6.4f}  {win}")
    
    print(f"\n  Categories where within > between: {n_wins}/{len(cats)}")
    print(f"  Pairwise identification accuracy:  {accuracy*100:.1f}%")
    print(f"  (Haxby et al. 2001 reported:       96% for 7 categories)")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Spatial ROI validation (face → FFA, house → PPA)
# ──────────────────────────────────────────────────────────────────────────────

def validate_spatial_rois(replay: SubjectReplay, 
                           use_oscillator: bool = True) -> Dict:
    """
    Validate that the model's category-specific patterns map back to
    known neuroanatomical regions.
    
    For each category, find voxels with highest activation in the 
    category-specific pattern and compare their coordinates to known ROIs.
    """
    results = {}
    
    for cat in TARGET_CATS:
        cat_events = [ev for ev in replay.events if ev.category == cat]
        if not cat_events:
            continue
        
        if use_oscillator and replay.oscillator_states is not None:
            indices = [replay.events.index(ev) for ev in cat_events]
            vecs = replay.oscillator_states[indices]
        else:
            vecs = np.stack([ev.bold_vector for ev in cat_events 
                           if ev.bold_vector is not None])
        
        if len(vecs) == 0:
            continue
        
        # Average pattern for this category
        cat_pattern = vecs.mean(axis=0)
        
        # Find voxels with highest activation
        top_k = 50  # top 50 most activated voxels
        top_voxel_idx = np.argsort(np.abs(cat_pattern))[-top_k:]
        top_coords = replay.coords[top_voxel_idx]
        
        # Centroid of top-activated voxels
        activation_centroid = top_coords.mean(axis=0)
        
        results[cat] = {
            'centroid': activation_centroid,
            'top_coords': top_coords,
            'top_activations': cat_pattern[top_voxel_idx],
            'activation_spread': np.std(top_coords, axis=0),
        }
        
        # Check against known ROIs
        if cat in KNOWN_ROIS:
            roi = KNOWN_ROIS[cat]
            # Note: we can't directly compare voxel coords to Talairach
            # without MNI registration, but we CAN check relative positions
            print(f"    '{cat}' activation centroid: "
                  f"({activation_centroid[0]:.0f}, "
                  f"{activation_centroid[1]:.0f}, "
                  f"{activation_centroid[2]:.0f}) voxel space")
            print(f"    Known ROI ({roi['name']}): "
                  f"Talairach ({roi['talairach'][0]}, "
                  f"{roi['talairach'][1]}, "
                  f"{roi['talairach'][2]})")
            print(f"    Activation spread (std): "
                  f"({results[cat]['activation_spread'][0]:.1f}, "
                  f"{results[cat]['activation_spread'][1]:.1f}, "
                  f"{results[cat]['activation_spread'][2]:.1f})")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Damping profile analysis (what the model learned NOT to look for)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_damping_profile(model: StimulusDLinOSS, replay: SubjectReplay):
    """
    Analyze what the D-LinOSS damping matrix learned to suppress.
    
    The damping matrix G controls how quickly different oscillatory modes
    decay. High damping = the model treats this as noise to suppress.
    Low damping = the model treats this as signal to preserve.
    
    By examining what's damped out, we learn what neural dynamics the model
    considers irrelevant to stimulus processing — scanner artifacts, 
    physiological noise, motor preparation, etc.
    """
    model.eval()
    
    print(f"\n  ── Damping Profile Analysis ──")
    
    # Extract damping-related parameters
    damping = model.get_damping_profile()
    
    if damping:
        for name, weights in damping.items():
            print(f"    {name}: shape={weights.shape}, "
                  f"mean={weights.mean():.4f}, "
                  f"std={weights.std():.4f}, "  
                  f"min={weights.min():.4f}, "
                  f"max={weights.max():.4f}")
    
    # Compare oscillator states during stimulus vs rest
    stim_events = [ev for ev in replay.events if ev.category != 'rest']
    
    if replay.oscillator_states is not None:
        stim_indices = list(range(len(stim_events)))
        stim_states = replay.oscillator_states[stim_indices]
        
        # Signal properties during stimuli
        stim_norms = np.linalg.norm(stim_states, axis=1)
        print(f"\n    Stimulus oscillator states:")
        print(f"      Mean norm:  {stim_norms.mean():.4f}")
        print(f"      Std norm:   {stim_norms.std():.4f}")
        
        # Compare category-level signal strength
        for cat in ALL_CATS:
            cat_idx = [i for i, ev in enumerate(replay.events) 
                       if ev.category == cat and i < len(replay.oscillator_states)]
            if cat_idx:
                cat_norms = np.linalg.norm(replay.oscillator_states[cat_idx], axis=1)
                print(f"      {cat:<15}: mean_norm={cat_norms.mean():.4f}, "
                      f"std={cat_norms.std():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Cross-subject category emergence analysis
# ──────────────────────────────────────────────────────────────────────────────

def cross_subject_emergence(replays: List[SubjectReplay]) -> Dict:
    """
    After independent per-subject training, check if categories emerge
    consistently across subjects.
    
    Key question: if we train each subject's model independently (no labels),
    do the learned oscillator states cluster into the SAME categories?
    
    This is the Ghost Basin test — if the categorical superposition exists,
    it should emerge independently in each subject's trained model.
    """
    print(f"\n  ── Cross-Subject Category Emergence ──")
    
    # For each subject, cluster their oscillator states
    subject_clusters = {}
    
    for replay in replays:
        if replay.oscillator_states is None:
            continue
        
        # Exclude scrambledpix for clustering
        target_idx = [i for i, ev in enumerate(replay.events) 
                      if ev.category in TARGET_CATS]
        target_states = replay.oscillator_states[target_idx]
        target_labels = [replay.events[i].category for i in target_idx]
        
        if len(target_states) < 7:
            continue
        
        # KMeans with 7 clusters (one per category)
        km = KMeans(n_clusters=7, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(target_states)
        
        # Check cluster-to-category mapping
        # For each cluster, what's the dominant category?
        cluster_cats = {}
        for cid in range(7):
            members = [target_labels[i] for i, c in enumerate(cluster_ids) if c == cid]
            if members:
                from collections import Counter
                most_common = Counter(members).most_common(1)[0]
                cluster_cats[cid] = {
                    'dominant': most_common[0],
                    'purity': most_common[1] / len(members),
                    'n': len(members),
                }
        
        subject_clusters[replay.subject_id] = cluster_cats
        
        # Print cluster purity
        avg_purity = np.mean([cc['purity'] for cc in cluster_cats.values()])
        print(f"\n    {replay.subject_id}: avg cluster purity = {avg_purity:.1%}")
        for cid, info in sorted(cluster_cats.items()):
            print(f"      Cluster {cid}: {info['dominant']:<12} "
                  f"purity={info['purity']:.0%} "
                  f"(n={info['n']})")
    
    return subject_clusters


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Main pipeline 
# ──────────────────────────────────────────────────────────────────────────────

def run_stimulus_replay():
    """Full stimulus-by-stimulus replay pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("  STIMULUS-BY-STIMULUS EXPERIMENT REPLAY")
    print("  Training on individual events, discovering categories afterward")
    print("=" * 70)
    
    # ── Phase 1: Replay experiments ──
    print("\n" + "─" * 70)
    print("Phase 1: Replaying experiment for all subjects")
    print("─" * 70)
    
    replays = []
    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            print(f"  [skip] {subj} not found")
            continue
        replay = replay_experiment(path, subj)
        replays.append(replay)
    
    # ── Phase 2: Self-supervised training per subject ──
    print("\n" + "─" * 70)
    print("Phase 2: Self-supervised D-LinOSS training (NO LABELS)")
    print("─" * 70)
    
    models = {}
    for replay in replays:
        print(f"\n  Training model for {replay.subject_id}...")
        model = train_self_supervised(replay, device=device, n_epochs=N_EPOCHS)
        models[replay.subject_id] = model
        
        # Extract oscillator states
        print(f"  Extracting oscillator states for {replay.subject_id}...")
        extract_oscillator_responses(replay, model, device=device)
    
    # ── Phase 3: Haxby-style correlation analysis ──
    print("\n" + "─" * 70)
    print("Phase 3: Haxby even/odd correlation analysis")
    print("─" * 70)
    
    for replay in replays:
        # Run 4 conditions: Raw/Oscillator × Stimulus/Block-level
        for use_osc in [False, True]:
            for block_lvl in [False, True]:
                results = haxby_correlation_analysis(
                    replay, use_oscillator=use_osc, block_level=block_lvl
                )
                print_haxby_results(results, replay.subject_id)
    
    # ── Phase 4: Spatial ROI validation ──
    print("\n" + "─" * 70)
    print("Phase 4: Spatial ROI validation (face→FFA, house→PPA)")
    print("─" * 70)
    
    for replay in replays[:2]:  # Just first 2 for speed
        print(f"\n  {replay.subject_id}:")
        validate_spatial_rois(replay, use_oscillator=True)
    
    # ── Phase 5: Damping profile ──
    print("\n" + "─" * 70)
    print("Phase 5: Damping profile (what the model learned to suppress)")
    print("─" * 70)
    
    for replay in replays[:2]:
        if replay.subject_id in models:
            analyze_damping_profile(models[replay.subject_id], replay)
    
    # ── Phase 6: Cross-subject category emergence ──
    print("\n" + "─" * 70)
    print("Phase 6: Cross-subject category emergence test")
    print("─" * 70)
    
    cross_subject_emergence(replays)
    
    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_stimulus_replay()
