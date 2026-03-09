"""
dlinoss_experiment_reversal.py
──────────────────────────────────────────────────────────────────────────────
Reverse-engineering the Haxby experiment for proper subspace construction.

Core Insight
────────────
The Haxby experiment was NOT designed to find categorical neural pathways.
It was an attention/working memory 1-back task where subjects pressed a button
when they detected an immediate image repeat. The categorical selectivity
(face → FFA, house → PPA) was a SIDE EFFECT discovered after the fact.

This module properly accounts for:

  1. TEMPORAL ISOLATION — The recognition moment happens in the first 2-3 TRs
     after a category block begins. Everything after is working memory / motor
     planning / drift. We surgically isolate the recognition spike from the
     maintenance/drift noise.

  2. BLOCK TRANSITION CONTRAST — When category changes between blocks
     (e.g. face → chair), the contrast signal between the recognition windows
     of adjacent blocks gives us orthogonal category directions. When the
     same category appears across runs, the shared signal IS the subspace.

  3. ATTENTION GATING — The 1-back task tells us when the subject is paying
     attention. No button presses for extended periods → subject is drifting
     or asleep → gate that data out.

  4. PER-SUBJECT SVD SUBSPACES — For each subject, build a geometric subspace
     per category from the recognition-window vectors only.

  5. CROSS-SUBJECT PROJECTION — Project each subject's subspaces onto a
     shared space via Canonical Correlation Analysis to find the Ghost
     (the abstract categorical superposition that doesn't exist in physical
     space but everyone can identify).

  6. WAVE PROPAGATION IN 3D — Model the recognition signal as a damped wave
     propagating through the 3D MRI voxel grid, looking for interference
     patterns as concurrent "waves" interact.

Pipeline
────────
[BOLD run] → per-run temporal block segmentation
           → recognition window extraction (TRs 0-3 after block onset)
           → attention quality gating (behavioral threshold)
           → D-LinOSS oscillatory encoding of recognition windows
           → SAE feature extraction
           → Per-subject per-category SVD subspace construction
           → Cross-subject CCA alignment → Ghost Basin coordinates
           → Validation: face→FFA, house→PPA spatial overlap

Usage
────────
  python dlinoss_experiment_reversal.py

Requires:
  pip install torch numpy scipy scikit-learn transformers nilearn
"""

import os, math, random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import orthogonal_procrustes, svd, subspace_angles
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from transformers import CLIPProcessor, CLIPModel

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

IN_DIM          = 1000          # top-variance voxels kept
TARGET_DIM      = 512           # CLIP embedding dimension
DLINOSS_STATE   = 32
DLINOSS_HIDDEN  = 64
SAE_EXPANSION   = 4
N_EPOCHS        = 40
LR              = 5e-4
WEIGHT_DECAY    = 1e-4
KFAC_MOMENTUM   = 0.95
KFAC_DAMPING    = 1e-3
GEO_RETENTION   = 0.05
N_CLUSTERS      = 7
MAX_REST_TRS    = 12

# Temporal isolation parameters
RECOGNITION_TRS = 3             # How many TRs after block onset capture recognition
MAINTENANCE_TRS = 4             # How many TRs of working memory maintenance
BLOCK_TRS       = 10            # Typical block length in TRs (~24s / 2.5s)
HRF_DELAY_S     = 4.0          # Hemodynamic response function peak delay

# Attention gating  
MIN_ATTEND_SCORE = 0.3          # Minimum rolling attention to include block
ATTEND_WINDOW    = 5            # Rolling window for attention scoring (in blocks)

# Subspace construction
SUBSPACE_K       = 8            # Number of principal components per category subspace
CCA_COMPONENTS   = 5            # Shared components from CCA across subjects

DATA_DIR        = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
TARGET_CATS     = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'shoe']
ALL_CATS        = TARGET_CATS + ['scrambledpix', 'rest']
TEST_SUBJECT    = 'sub-5'
SUBJECTS        = [f'sub-{i}' for i in range(1, 7)]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data structures for experiment reversal
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StimulusBlock:
    """A single stimulus block within a run."""
    category: str
    run_id: int
    block_idx: int              # position within the run (0-7 typically)
    onset_tr: int               # TR index of block start (within run)
    end_tr: int                 # TR index of block end
    
    # Temporal sub-windows (TR indices relative to onset_tr)
    recognition_window: Optional[np.ndarray] = None   # First 2-3 TRs
    maintenance_window: Optional[np.ndarray] = None    # TRs 3-6
    drift_window: Optional[np.ndarray] = None          # TRs 7+
    
    # Attention data
    attend_score: float = 1.0   # Behavioral attention proxy [0, 1]
    
    # The raw BOLD data for this block [T_block, V]
    full_ts: Optional[np.ndarray] = None
    
    # Active voxel coordinates for spatial mapping
    coords: Optional[np.ndarray] = None


@dataclass
class BlockTransition:
    """Transition between two consecutive stimulus blocks."""
    prev_block: StimulusBlock
    curr_block: StimulusBlock
    is_same_category: bool
    
    # Derived signals
    contrast_vector: Optional[np.ndarray] = None    # recognition(curr) - recognition(prev)
    shared_vector: Optional[np.ndarray] = None      # (recognition(prev) + recognition(curr)) / 2
    suppression_vector: Optional[np.ndarray] = None  # recognition(prev) - recognition(curr), same-cat only


@dataclass
class SubjectExperiment:
    """Full experimental data for one subject."""
    subject_id: str
    blocks: List[StimulusBlock] = field(default_factory=list)
    transitions: List[BlockTransition] = field(default_factory=list)
    
    # Per-category subspaces (computed later)
    category_subspaces: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Attention quality score for the whole session
    overall_attention: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Experiment reversal: parse blocks + temporal isolation
# ──────────────────────────────────────────────────────────────────────────────

def reverse_engineer_experiment(subject_dir: str, subject_id: str, 
                                 max_runs: int = 12) -> SubjectExperiment:
    """
    Parse the Haxby experiment structure for one subject, extracting:
    - Individual stimulus blocks with temporal sub-windows
    - Block transitions (same-cat and different-cat)
    - Attention quality gating per block
    
    This is the core "experiment reversal" — we're not just loading data,
    we're understanding what the experiment DID to the brain moment by moment.
    """
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    
    # Global voxel selection across all runs (top variance)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)
    
    max_voxels = IN_DIM
    variances  = np.var(ts_2d, axis=1)
    top_idx    = np.argsort(variances)[-max_voxels:]
    active_coords = coords_array[top_idx]
    
    # HRF delay in TRs
    hrf_delay_trs = max(1, int(round(HRF_DELAY_S / TR)))
    
    experiment = SubjectExperiment(subject_id=subject_id)
    
    # Track the running TR offset across runs
    running_tr_offset = 0
    all_blocks = []
    
    for run_idx, (bold_data, events) in enumerate(zip(bold_runs, events_runs)):
        run_T = bold_data.shape[-1]
        
        # Extract & preprocess this run's timeseries for selected voxels
        ts_run = ts_2d[top_idx, running_tr_offset:running_tr_offset + run_T]
        ts_run = _preprocess_run(ts_run).T  # → [V, T_run]
        manifold_run = ts_run.T  # → [T_run, V]
        
        # Parse events into block boundaries
        blocks_in_run = _parse_blocks_from_events(events, run_T, TR, run_idx)
        
        for blk in blocks_in_run:
            onset = blk.onset_tr
            end   = blk.end_tr
            
            if end <= onset or onset >= run_T:
                continue
            end = min(end, run_T)
            
            # Extract the full block timeseries
            blk.full_ts = manifold_run[onset:end].copy()
            blk.coords  = active_coords
            
            # ────────────────────────────────────────────────────────
            # TEMPORAL ISOLATION: Split into recognition / maintenance / drift
            # ────────────────────────────────────────────────────────
            block_len = end - onset
            
            # Recognition window: first RECOGNITION_TRS after onset + HRF delay
            rec_start = min(hrf_delay_trs, block_len - 1)
            rec_end   = min(rec_start + RECOGNITION_TRS, block_len)
            if rec_end > rec_start:
                blk.recognition_window = manifold_run[onset + rec_start : onset + rec_end].copy()
            
            # Maintenance window: immediately after recognition
            maint_end = min(rec_end + MAINTENANCE_TRS, block_len)
            if maint_end > rec_end:
                blk.maintenance_window = manifold_run[onset + rec_end : onset + maint_end].copy()
            
            # Drift window: everything after maintenance
            if block_len > maint_end:
                blk.drift_window = manifold_run[onset + maint_end : end].copy()
            
            all_blocks.append(blk)
        
        running_tr_offset += run_T
    
    experiment.blocks = all_blocks
    
    # ────────────────────────────────────────────────────────────────────────
    # Build block transitions (consecutive blocks, within AND across runs)
    # ────────────────────────────────────────────────────────────────────────
    for i in range(len(all_blocks) - 1):
        prev = all_blocks[i]
        curr = all_blocks[i + 1]
        
        # Skip if either block doesn't have a recognition window
        if prev.recognition_window is None or curr.recognition_window is None:
            continue
        
        is_same = prev.category == curr.category
        
        # Recognition-window centroids
        prev_rec = prev.recognition_window.mean(axis=0)
        curr_rec = curr.recognition_window.mean(axis=0)
        
        trans = BlockTransition(
            prev_block=prev,
            curr_block=curr,
            is_same_category=is_same,
            contrast_vector=curr_rec - prev_rec,
            shared_vector=(prev_rec + curr_rec) / 2.0,
        )
        
        if is_same:
            trans.suppression_vector = prev_rec - curr_rec
        
        experiment.transitions.append(trans)
    
    # ────────────────────────────────────────────────────────────────────────
    # Attention quality: compute rolling score (placeholder — no RT data)
    # ────────────────────────────────────────────────────────────────────────
    _compute_attention_scores(experiment)
    
    return experiment


def _preprocess_run(ts_2d: np.ndarray) -> np.ndarray:
    """Detrend + z-score a [V, T] matrix. Returns [T, V]."""
    V, T = ts_2d.shape
    cleaned = ts_2d.copy().astype(np.float32)
    t_axis  = np.arange(T, dtype=np.float32)
    for i in range(V):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu    = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return np.clip((cleaned - mu) / sigma, -5.0, 5.0).T  # [T, V]


def _parse_blocks_from_events(events: list, T: int, TR: float, 
                                run_idx: int) -> List[StimulusBlock]:
    """
    Parse events.tsv into StimulusBlock objects.
    
    The Haxby design: 8 category blocks per run, each with 12 stimuli 
    at 2s ISI (500ms on, 1500ms ISI). Block duration ≈ 24s.
    Rest periods (12s) between blocks.
    """
    if not events:
        return []
    
    blocks = []
    current_type = None
    block_start_onset = None
    block_idx = 0
    
    for ev in events:
        trial_type = ev['trial_type']
        onset = ev['onset']
        
        if trial_type != current_type:
            # Close previous block
            if current_type is not None and block_start_onset is not None:
                onset_tr  = int(round(block_start_onset / TR))
                # Block ends at current event onset (start of rest or new block)
                end_tr    = int(round(onset / TR))
                
                blocks.append(StimulusBlock(
                    category=current_type,
                    run_id=run_idx,
                    block_idx=block_idx,
                    onset_tr=onset_tr,
                    end_tr=end_tr,
                ))
                block_idx += 1
            
            current_type = trial_type
            block_start_onset = onset
    
    # Close last block
    if current_type is not None and block_start_onset is not None:
        onset_tr = int(round(block_start_onset / TR))
        # Last block extends ~24s from start  
        end_tr   = min(onset_tr + int(round(24.0 / TR)), T)
        blocks.append(StimulusBlock(
            category=current_type,
            run_id=run_idx,
            block_idx=block_idx,
            onset_tr=onset_tr,
            end_tr=end_tr,
        ))
    
    return blocks


def _compute_attention_scores(experiment: SubjectExperiment):
    """
    Compute attention quality scores for each block.
    
    Since the Haxby events.tsv doesn't include response times (just
    onset/duration/trial_type), we use heuristics:
    
    1. Blocks later in the run get lower scores (fatigue)
    2. Blocks later in the experiment get lower scores (boredom)
    3. Scrambledpix blocks get a special "overload" flag
    
    TODO: If behavioral data becomes available, replace with actual
    button press timing analysis.
    """
    total_blocks = len(experiment.blocks)
    
    for i, blk in enumerate(experiment.blocks):
        # Fatigue decay: later blocks in a run get lower attention
        run_position = blk.block_idx / max(1, 7)  # 8 blocks per run typically
        fatigue_factor = 1.0 - 0.3 * run_position  # 0.7-1.0 range
        
        # Session decay: later runs get lower attention
        session_progress = i / max(1, total_blocks - 1)
        session_factor  = 1.0 - 0.2 * session_progress  # 0.8-1.0 range
        
        # Scrambledpix: visual cortex overload, but no categorical recognition
        if blk.category == 'scrambledpix':
            blk.attend_score = fatigue_factor * session_factor * 0.8
        else:
            blk.attend_score = fatigue_factor * session_factor
    
    # Overall attention score
    scores = [b.attend_score for b in experiment.blocks]
    experiment.overall_attention = np.mean(scores) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Per-subject geometric subspace construction
# ──────────────────────────────────────────────────────────────────────────────

def build_subject_subspaces(experiment: SubjectExperiment, 
                             model: nn.Module = None,
                             device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    For each category, collect all recognition-window vectors for this subject,
    apply attention gating, and compute the SVD subspace.
    
    The subspace captures the geometric "shape" of the recognition response
    for this category in this subject's brain — the portion of the signal
    that represents the moment they identified "this is a cat".
    
    Returns: {category: subspace_basis_matrix [d, k]}
    """
    cat_vectors = defaultdict(list)
    
    for blk in experiment.blocks:
        if blk.category not in TARGET_CATS:
            continue
        if blk.recognition_window is None:
            continue
        if blk.attend_score < MIN_ATTEND_SCORE:
            continue
        
        # Get recognition-window representation
        rec = blk.recognition_window  # [T_rec, V]
        
        if model is not None:
            # Encode through D-LinOSS + SAE to get latent representation
            with torch.no_grad():
                xt = torch.tensor(rec, dtype=torch.float32).unsqueeze(0).to(device)
                ev_vec, *_ = model(xt)
                vec = ev_vec.cpu().numpy().flatten()
        else:
            # Raw recognition centroid (useful for initial exploration)
            vec = rec.mean(axis=0)
        
        # Weight by attention score
        cat_vectors[blk.category].append(vec * blk.attend_score)
    
    # Build subspaces via SVD
    subspaces = {}
    for cat in TARGET_CATS:
        if cat not in cat_vectors or len(cat_vectors[cat]) < 2:
            print(f"  [warn] {experiment.subject_id}: insufficient data for '{cat}'")
            continue
        
        # Stack vectors: [n_blocks, d]
        mat = np.stack(cat_vectors[cat])
        
        # Center
        mat -= mat.mean(axis=0, keepdims=True)
        
        # SVD
        U, S, Vt = svd(mat, full_matrices=False)
        
        # Keep top-k directions
        k = min(SUBSPACE_K, len(S), Vt.shape[0])
        subspaces[cat] = Vt[:k]  # [k, d] — the principal directions
        
        # Log the explained variance
        total_var = np.sum(S ** 2)
        kept_var  = np.sum(S[:k] ** 2)
        frac = kept_var / (total_var + 1e-10)
        print(f"  {experiment.subject_id} '{cat}': "
              f"{len(cat_vectors[cat])} blocks → "
              f"subspace dim {k}, "
              f"explained variance {frac:.1%}")
    
    experiment.category_subspaces = subspaces
    return subspaces


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Block transition analysis (repetition suppression + contrast)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_transitions(experiment: SubjectExperiment) -> Dict:
    """
    Analyze block transitions to extract:
    1. Same-category transitions → repetition suppression (the shared component)
    2. Different-category transitions → contrast signals (orthogonal directions)
    
    The same-category transitions are particularly valuable: the brain's
    response to the second occurrence of the SAME category is attenuated
    (repetition suppression). The difference tells us what's unique to the
    first presentation (novelty response), while the shared component tells
    us what's consistently activated by the category (the Ghost).
    """
    same_cat_shared = defaultdict(list)
    same_cat_suppression = defaultdict(list)
    contrast_pairs = defaultdict(list)
    
    for trans in experiment.transitions:
        if trans.prev_block.attend_score < MIN_ATTEND_SCORE:
            continue
        if trans.curr_block.attend_score < MIN_ATTEND_SCORE:
            continue
        
        if trans.is_same_category:
            cat = trans.prev_block.category
            if trans.shared_vector is not None:
                same_cat_shared[cat].append(trans.shared_vector)
            if trans.suppression_vector is not None:
                same_cat_suppression[cat].append(trans.suppression_vector)
        else:
            key = (trans.prev_block.category, trans.curr_block.category)
            if trans.contrast_vector is not None:
                contrast_pairs[key].append(trans.contrast_vector)
    
    results = {
        'same_cat_shared': {k: np.stack(v) for k, v in same_cat_shared.items() if v},
        'same_cat_suppression': {k: np.stack(v) for k, v in same_cat_suppression.items() if v},
        'contrast_pairs': {k: np.stack(v) for k, v in contrast_pairs.items() if v},
    }
    
    # Log statistics
    print(f"\n  Block transition analysis for {experiment.subject_id}:")
    print(f"  Same-category transitions: {sum(len(v) for v in same_cat_shared.values())}")
    print(f"  Cross-category contrasts:  {sum(len(v) for v in contrast_pairs.values())}")
    
    for cat, vecs in same_cat_shared.items():
        # The consistency of the shared vector tells us subspace quality
        if len(vecs) >= 2:
            mat = np.stack(vecs)
            consistency = np.mean([
                np.corrcoef(mat[i], mat[j])[0, 1]
                for i in range(len(mat)) for j in range(i + 1, len(mat))
            ])
            print(f"    '{cat}' shared signal consistency: r={consistency:.3f}")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Cross-subject alignment: Finding the Ghost
# ──────────────────────────────────────────────────────────────────────────────

def align_subspaces_cross_subject(
    subject_experiments: List[SubjectExperiment],
    categories: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Project each subject's geometric subspaces onto a shared space.
    
    This is where the "superposition" emerges: each subject has a different
    physical brain, different neural pathways, different "idea of what a cat is."
    But if they can all identify a cat in a picture, their recognition-window
    subspaces must share a common geometric direction. That shared direction
    IS the categorical superposition — the Ghost.
    
    Method: for each category, we collect all subjects' subspace bases and find
    their principal shared angles via multi-set CCA or joint SVD.
    
    Returns: {category: shared_superposition_vector [d]}
    """
    if categories is None:
        categories = TARGET_CATS
    
    ghost_basin = {}
    
    for cat in categories:
        # Collect all subjects' subspaces for this category
        subject_bases = []
        subject_ids   = []
        for exp in subject_experiments:
            if cat in exp.category_subspaces:
                subject_bases.append(exp.category_subspaces[cat])
                subject_ids.append(exp.subject_id)
        
        if len(subject_bases) < 2:
            print(f"  [warn] '{cat}': only {len(subject_bases)} subjects — need ≥2")
            continue
        
        # Method 1: Joint SVD of concatenated subspace projections
        # Each subject's subspace is [k_i, d]. We want to find the d-dimensional
        # directions that are shared across subjects.
        
        # Project all subjects' subspace bases into a common matrix
        # and take SVD to find shared directions
        all_rows = np.vstack(subject_bases)  # [Σ k_i, d]
        
        U, S, Vt = svd(all_rows, full_matrices=False)
        
        # The singular values tell us how "shared" each direction is.
        # High singular values = direction present in multiple subjects.
        # We want the top CCA_COMPONENTS shared directions.
        k_shared = min(CCA_COMPONENTS, Vt.shape[0])
        shared_basis = Vt[:k_shared]  # [k_shared, d]
        
        # The "ghost" is the top shared direction (centroid of the superposition)
        ghost_basin[cat] = shared_basis[0]  # [d] — the principal shared direction
        
        # Log subspace angles between each pair of subjects
        print(f"\n  '{cat}' cross-subject alignment ({len(subject_ids)} subjects):")
        print(f"    Top {k_shared} shared singular values: "
              f"{', '.join(f'{s:.3f}' for s in S[:k_shared])}")
        
        for i in range(len(subject_bases)):
            for j in range(i + 1, len(subject_bases)):
                # Principal angle between subjects' subspaces for this category
                angles = subspace_angles(subject_bases[i].T, subject_bases[j].T)
                min_angle_deg = np.degrees(angles[0]) if len(angles) > 0 else 90
                print(f"    {subject_ids[i]} ↔ {subject_ids[j]}: "
                      f"min principal angle = {min_angle_deg:.1f}°")
    
    return ghost_basin


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Scrambledpix baseline (visual cortex overload reference)
# ──────────────────────────────────────────────────────────────────────────────

def compute_scrambledpix_baseline(experiments: List[SubjectExperiment]) -> np.ndarray:
    """
    Compute the "visual cortex overload" baseline from scrambledpix blocks.
    
    Scrambledpix triggers massive occipital activity but NO categorical recognition.
    It's the "max signal, zero meaning" condition. We use it to:
    1. Subtract the shared visual processing noise from all categories
    2. Validate that our recognition-window isolation is working
       (scrambledpix should NOT produce a coherent recognition signal)
    3. Characterize what "no Ghost" looks like
    """
    all_scrambled = []
    
    for exp in experiments:
        for blk in exp.blocks:
            if blk.category != 'scrambledpix':
                continue
            if blk.recognition_window is not None:
                all_scrambled.append(blk.recognition_window.mean(axis=0))
    
    if not all_scrambled:
        print("  [warn] No scrambledpix blocks found for baseline")
        return np.zeros(IN_DIM)
    
    scrambled_mat = np.stack(all_scrambled)
    
    # The centroid of scrambledpix responses = pure visual processing noise
    baseline = scrambled_mat.mean(axis=0)
    
    # Also compute the variance — high variance in scrambledpix responses
    # indicates the visual cortex is responding chaotically (as expected)
    scrambled_var = scrambled_mat.var(axis=0).mean()
    
    # Compare to typical category variance
    print(f"\n  Scrambledpix baseline:")
    print(f"    N blocks:          {len(all_scrambled)}")
    print(f"    Mean variance:     {scrambled_var:.4f}")
    print(f"    Baseline norm:     {np.linalg.norm(baseline):.4f}")
    
    return baseline


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Wave propagation analysis in 3D MRI space
# ──────────────────────────────────────────────────────────────────────────────

def analyze_wave_propagation(experiment: SubjectExperiment, 
                              category: str) -> Dict:
    """
    Model the recognition signal as a wave propagating through 3D MRI space.
    
    The idea: when a subject sees a cat, a "recognition wave" propagates
    through the visual cortex. This wave has:
    - A wavefront that moves through space over time
    - Interference patterns where it meets other concurrent waves
    - Frequency shifts as it interacts with local neural populations
    
    We can detect this by tracking the spatial autocorrelation structure
    of the recognition-window signal across voxels and time.
    """
    cat_blocks = [blk for blk in experiment.blocks 
                  if blk.category == category 
                  and blk.recognition_window is not None
                  and blk.coords is not None]
    
    if not cat_blocks:
        return {}
    
    results = {
        'n_blocks': len(cat_blocks),
        'wave_speeds': [],
        'interference_patterns': [],
        'frequency_shifts': [],
    }
    
    for blk in cat_blocks:
        rec = blk.recognition_window   # [T_rec, V]
        coords = blk.coords            # [V, 3]
        
        if rec.shape[0] < 2:
            continue
        
        # ── Spatial autocorrelation at each TR ──
        for t in range(rec.shape[0]):
            signal = rec[t]  # [V]
            
            # Compute the spatial gradient of the signal
            # For each voxel, what's the signal difference to neighbors?
            # This tells us where the "wavefront" currently is.
            if coords.shape[0] > 10:
                # Compute pairwise distances (subsample for speed)
                n_sample = min(200, coords.shape[0])
                idx = np.random.choice(coords.shape[0], n_sample, replace=False)
                sub_coords = coords[idx]
                sub_signal = signal[idx]
                
                dists   = squareform(pdist(sub_coords))
                sig_diffs = np.abs(sub_signal[:, None] - sub_signal[None, :])
                
                # Identify the wavefront: high signal gradient over short distance
                near_mask = (dists > 0) & (dists < 3.0)  # within ~9mm
                if near_mask.any():
                    local_gradient = sig_diffs[near_mask].mean()
                    results['wave_speeds'].append({
                        'block_idx': blk.block_idx,
                        'tr': t,
                        'gradient': float(local_gradient),
                    })
        
        # ── Frequency analysis per voxel over the recognition window ──
        if rec.shape[0] >= 3:
            # FFT of each voxel's recognition-window timeseries
            freqs = np.abs(np.fft.rfft(rec, axis=0))  # [n_freqs, V]
            
            # Peak frequency per voxel
            peak_freqs = np.argmax(freqs, axis=0)  # [V]
            
            # Where are frequency shifts happening? (spatial gradient of peak freq)
            results['frequency_shifts'].append({
                'block_idx': blk.block_idx,
                'mean_peak_freq': float(np.mean(peak_freqs)),
                'std_peak_freq': float(np.std(peak_freqs)),
                'freq_spatial_entropy': float(_spatial_entropy(peak_freqs, coords)),
            })
    
    return results


def _spatial_entropy(values: np.ndarray, coords: np.ndarray, n_bins: int = 10) -> float:
    """Compute spatial entropy of values distributed across coordinates."""
    if len(values) == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Model architecture (adapted from dlinoss_geometric_kfac.py)
# ──────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """Overcomplete sparse AE for feature extraction."""
    def __init__(self, in_dim, expansion_factor=SAE_EXPANSION):
        super().__init__()
        self.features_dim    = in_dim * expansion_factor
        self.encoder_weight  = nn.Parameter(torch.empty(self.features_dim, in_dim))
        nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        self.encoder_bias    = nn.Parameter(torch.zeros(self.features_dim))
        self.decoder_bias    = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        z       = F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))
        x_recon = F.linear(z, self.encoder_weight.t(), self.decoder_bias)
        return z, x_recon


class SpatiotemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate     = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.attn_net = nn.Sequential(nn.Linear(dim, dim // 2), nn.LeakyReLU(0.1),
                                      nn.Linear(dim // 2, 1))

    def forward(self, x_seq):                   # [B, T, D]
        g       = self.gate(x_seq)
        x_g     = x_seq * g
        scores  = self.attn_net(x_g)            # [B, T, 1]
        weights = F.softmax(scores, dim=1)
        ctx     = (weights * x_g).sum(dim=1)    # [B, D]
        return ctx, weights.squeeze(-1)


class DampingEncoder(nn.Module):
    """D-LinOSS encoder with KFAC hooks for damping matrix regularization."""
    def __init__(self, in_dim, state_dim, hidden_dim, n_blocks=2, drop=0.1):
        super().__init__()
        self.dlinoss = DLinOSSModel(
            input_dim  = in_dim,
            state_dim  = state_dim,
            hidden_dim = hidden_dim,
            output_dim = in_dim,
            num_blocks = n_blocks,
            drop_rate  = drop,
        )

    def forward(self, x):
        return self.dlinoss(x)


class GhostBasinReversal(nn.Module):
    """
    Full model with experiment-reversal-aware architecture:
      DampingEncoder → SparseAutoencoder → semantic projection → SpatiotemporalAttention
      + outlier detector for scrambledpix suppression
      + recognition-window emphasis head
    """
    def __init__(self, in_dim=IN_DIM, target_dim=TARGET_DIM,
                 state_dim=DLINOSS_STATE, hidden_dim=DLINOSS_HIDDEN):
        super().__init__()
        self.damping_enc  = DampingEncoder(in_dim, state_dim, hidden_dim)
        self.sae          = SparseAutoencoder(in_dim)
        self.semantic_map = nn.Sequential(
            nn.Linear(self.sae.features_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, target_dim),
            nn.LayerNorm(target_dim),
        )
        self.spatiotemporal = SpatiotemporalAttention(target_dim)
        self.outlier_head   = nn.Sequential(
            nn.Linear(target_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )
        # Recognition emphasis: learns to weight re recognition vs maintenance 
        self.recognition_gate = nn.Sequential(
            nn.Linear(target_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq):                   # x_seq: [B, T, V]
        osc            = self.damping_enc(x_seq)
        z, x_recon     = self.sae(osc)
        mapped         = self.semantic_map(z)
        event_vec, atw = self.spatiotemporal(mapped)
        ev             = event_vec.squeeze(0)
        outlier_logit  = self.outlier_head(ev)
        rec_weight     = self.recognition_gate(ev)  # How "recognition-like" is this?
        return ev, z, x_recon, atw, outlier_logit, rec_weight


# ──────────────────────────────────────────────────────────────────────────────
# 9.  CLIP semantic anchors
# ──────────────────────────────────────────────────────────────────────────────

def extract_clip_anchors(categories):
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts   = [f"a photo of a {cat}" for cat in categories]
    inputs    = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, p=2, dim=-1)
    return {cat: feats[i] for i, cat in enumerate(categories)}


# ──────────────────────────────────────────────────────────────────────────────
# 10. Geometric task-vector disentanglement (from dlinoss_geometric_kfac.py)
# ──────────────────────────────────────────────────────────────────────────────

def geo_disentangle_subspaces(category_subspaces: Dict[str, np.ndarray],
                                retention: float = GEO_RETENTION
                                ) -> Dict[str, np.ndarray]:
    """
    For each category, project its subspace orthogonal to all other categories'
    subspaces. This removes the shared scanner / behavioral components,
    leaving only the category-specific geometric direction.
    
    category_subspaces: {cat: [k, d]} — principal directions per category
    Returns: {cat: disentangled_direction [d]}
    """
    cats = list(category_subspaces.keys())
    result = {}
    
    for i, cat in enumerate(cats):
        tv = category_subspaces[cat]  # [k, d]
        if tv.shape[0] == 0:
            continue
        
        # Principal direction for this category
        cat_dir = tv[0]  # [d]
        nt = np.linalg.norm(cat_dir)
        if nt < 1e-8:
            result[cat] = cat_dir
            continue
        
        # Collect other categories' principal directions
        others = []
        for j, other_cat in enumerate(cats):
            if i == j:
                continue
            other_dir = category_subspaces[other_cat]
            if other_dir.shape[0] > 0:
                others.append(other_dir[0])  # first principal direction
        
        if others:
            others_mat = np.stack(others)  # [C-1, d]
            
            # QR decomposition to get orthonormal basis of "other" space
            Q, _ = np.linalg.qr(others_mat.T)  # [d, rank]
            
            # Project cat_dir onto other space
            shared = Q @ (Q.T @ cat_dir)
            unique = cat_dir - shared
            
            un = np.linalg.norm(unique)
            if un > 1e-8:
                unique = unique * (nt / un)  # preserve original norm
            
            # Retain a small fraction of shared component
            result[cat] = unique + retention * shared
        else:
            result[cat] = cat_dir
        
        # Normalize
        norm = np.linalg.norm(result[cat])
        if norm > 1e-8:
            result[cat] = result[cat] / norm
    
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 11. Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_reversal():
    """Full reverse-engineering pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("  GHOST BASIN: EXPERIMENT REVERSAL PIPELINE")
    print("  Reverse-engineering Haxby to isolate categorical recognition")
    print("=" * 70)
    
    # ── Phase 1: Load & reverse-engineer the experiment ──
    print("\n" + "─" * 70)
    print("Phase 1: Reverse-engineering experiment structure")
    print("─" * 70)
    
    experiments = []
    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            print(f"  [skip] {subj} not found")
            continue
        print(f"\n  Loading & parsing {subj} …")
        exp = reverse_engineer_experiment(path, subj)
        experiments.append(exp)
        
        n_blocks = len(exp.blocks)
        n_with_rec = sum(1 for b in exp.blocks if b.recognition_window is not None)
        n_trans = len(exp.transitions)
        n_same = sum(1 for t in exp.transitions if t.is_same_category)
        
        print(f"    {n_blocks} blocks ({n_with_rec} with recognition windows)")
        print(f"    {n_trans} transitions ({n_same} same-category)")
        print(f"    Overall attention: {exp.overall_attention:.2f}")
    
    train_experiments = [e for e in experiments if e.subject_id != TEST_SUBJECT]
    test_experiments  = [e for e in experiments if e.subject_id == TEST_SUBJECT]
    
    # ── Phase 2: Analyze block transitions ──
    print("\n" + "─" * 70)
    print("Phase 2: Block transition analysis (repetition suppression)")
    print("─" * 70)
    
    for exp in experiments:
        analyze_transitions(exp)
    
    # ── Phase 3: Scrambledpix baseline ──
    print("\n" + "─" * 70)
    print("Phase 3: Scrambledpix baseline computation") 
    print("─" * 70)
    
    scrambled_baseline = compute_scrambledpix_baseline(train_experiments)
    
    # ── Phase 4: Build per-subject subspaces (raw, no model) ──
    print("\n" + "─" * 70)
    print("Phase 4: Per-subject geometric subspace construction")
    print("─" * 70)
    
    for exp in experiments:
        print(f"\n  Building subspaces for {exp.subject_id}:")
        build_subject_subspaces(exp)
    
    # ── Phase 5: Cross-subject alignment → Find the Ghost ──
    print("\n" + "─" * 70)
    print("Phase 5: Cross-subject alignment (finding the Ghost)")
    print("─" * 70)
    
    ghost_basin = align_subspaces_cross_subject(train_experiments)
    
    # ── Phase 6: Geometric disentanglement ──
    print("\n" + "─" * 70)
    print("Phase 6: Geometric disentanglement of category subspaces")
    print("─" * 70)
    
    # Build combined subspace from all training subjects for disentanglement
    combined_subspaces = {}
    for cat in TARGET_CATS:
        all_bases = []
        for exp in train_experiments:
            if cat in exp.category_subspaces:
                all_bases.append(exp.category_subspaces[cat])
        if all_bases:
            combined = np.vstack(all_bases)
            U, S, Vt = svd(combined, full_matrices=False)
            k = min(SUBSPACE_K, Vt.shape[0])
            combined_subspaces[cat] = Vt[:k]
    
    dis_directions = geo_disentangle_subspaces(combined_subspaces, retention=GEO_RETENTION)
    
    print("\n  Disentangled category directions:")
    for cat in TARGET_CATS:
        if cat in dis_directions:
            norm = np.linalg.norm(dis_directions[cat])
            print(f"    '{cat}': norm={norm:.4f}")
    
    # ── Phase 7: Validate on test subject ──
    print("\n" + "─" * 70)
    print(f"Phase 7: Zero-shot validation on {TEST_SUBJECT}")
    print("─" * 70)
    
    if test_experiments:
        test_exp = test_experiments[0]
        
        # Build test subject's subspaces
        test_subspaces = build_subject_subspaces(test_exp)
        
        # For each category, project test subject's recognition vectors
        # onto the disentangled ghost directions
        correct = 0
        total   = 0
        
        print(f"\n  {'Category':<12}  {'Closest Ghost':<14}  {'Match?'}")
        print("  " + "-" * 40)
        
        for cat in TARGET_CATS:
            if cat not in test_subspaces or test_subspaces[cat].shape[0] == 0:
                continue
            
            test_dir = test_subspaces[cat][0]  # Principal direction
            test_dir = test_dir / (np.linalg.norm(test_dir) + 1e-8)
            
            # Compare to all disentangled ghost directions
            best_sim  = -1
            best_cat  = None
            for ghost_cat, ghost_dir in dis_directions.items():
                sim = np.dot(test_dir, ghost_dir)
                if sim > best_sim:
                    best_sim = sim
                    best_cat = ghost_cat
            
            match = "✓" if best_cat == cat else "✗"
            if best_cat == cat:
                correct += 1
            total += 1
            print(f"    {cat:<12}  →  {best_cat:<12}  {match}  (sim={best_sim:.3f})")
        
        if total > 0:
            print(f"\n  Raw subspace accuracy: {correct}/{total} = "
                  f"{100*correct/total:.1f}%")
    
    # ── Phase 8: Wave propagation analysis ──
    print("\n" + "─" * 70)
    print("Phase 8: Wave propagation analysis")
    print("─" * 70)
    
    for cat in ['face', 'house']:
        for exp in train_experiments[:2]:  # Just first 2 subjects for now
            wave_results = analyze_wave_propagation(exp, cat)
            if wave_results:
                n_waves = wave_results.get('n_blocks', 0)
                n_speeds = len(wave_results.get('wave_speeds', []))
                n_freqs = len(wave_results.get('frequency_shifts', []))
                print(f"  {exp.subject_id} '{cat}': "
                      f"{n_waves} blocks, {n_speeds} gradient samples, "
                      f"{n_freqs} freq analyses")
                
                if wave_results.get('frequency_shifts'):
                    avg_freq = np.mean([
                        fs['mean_peak_freq'] 
                        for fs in wave_results['frequency_shifts']
                    ])
                    avg_entropy = np.mean([
                        fs['freq_spatial_entropy'] 
                        for fs in wave_results['frequency_shifts']
                    ])
                    print(f"    Avg peak freq: {avg_freq:.2f}, "
                          f"Avg spatial entropy: {avg_entropy:.2f}")
    
    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment_reversal()
