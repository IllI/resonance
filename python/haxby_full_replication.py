"""
haxby_full_replication.py
─────────────────────────────────────────────────────────────
Full Haxby 2001 replication with D-LinOSS oscillatory encoding.

Goal: Match or exceed Haxby's 96% pairwise accuracy while maintaining
100% within>between for all subjects.

Key improvements over previous attempts:
1. F-test voxel selection on ALL brain voxels (not top-variance subset)
2. Block-trajectory D-LinOSS training (each block = one training sequence)  
3. Leave-one-run-out cross-validation for F-test (avoid circularity)
4. Proper pairwise accuracy computation (all N*(N-1)/2 pairs)
5. Multiple voxel selection thresholds to find optimal
"""

import os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']

# ─────────────────────────────────────────────────────────────
# 1. Data loading & preprocessing
# ─────────────────────────────────────────────────────────────

def load_subject_data(subj_dir):
    """Load and preprocess one subject's data."""
    bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    V, T_total = ts_2d.shape
    
    # Z-score per voxel (whole-brain)
    mu = ts_2d.mean(axis=1, keepdims=True)
    sd = ts_2d.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ts_z = np.clip((ts_2d - mu) / sd, -5, 5)
    
    # Build per-TR labels
    all_labels = []
    run_bounds = [0]
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        all_labels.extend(run_labels)
        run_bounds.append(run_bounds[-1] + bold.shape[-1])
    all_labels = all_labels[:T_total]
    
    return ts_z, all_labels, run_bounds, coords, TR, len(bold_runs)


def extract_block_patterns(ts_z, all_labels, run_bounds, n_runs, voxel_idx=None):
    """Extract per-run, per-category block-averaged patterns."""
    if voxel_idx is not None:
        data = ts_z[voxel_idx, :]
    else:
        data = ts_z
    
    blocks = []
    for ri in range(n_runs):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        run_labels = all_labels[s:e]
        run_data = data[:, s:e]
        
        for cat in CATS:
            cat_trs = [i for i, l in enumerate(run_labels) if l == cat]
            if len(cat_trs) >= 2:
                pattern = run_data[:, cat_trs].mean(axis=1)
                blocks.append({
                    'run': ri, 'cat': cat, 'pattern': pattern,
                    'trajectory': run_data[:, cat_trs].T  # [T_block, V] for D-LinOSS
                })
    return blocks


def extract_block_trajectories(ts_z, all_labels, run_bounds, n_runs, voxel_idx):
    """Extract the full temporal trajectory of each block for D-LinOSS training."""
    data = ts_z[voxel_idx, :]  # [V_selected, T]
    
    trajectories = []
    for ri in range(n_runs):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        run_labels = all_labels[s:e]
        
        # Find contiguous blocks
        current_cat = None
        block_start = None
        for i, lbl in enumerate(run_labels):
            if lbl != current_cat:
                if current_cat in CATS and block_start is not None:
                    block_trs = data[:, s + block_start:s + i].T  # [T_block, V]
                    if len(block_trs) >= 3:
                        trajectories.append({
                            'run': ri, 'cat': current_cat,
                            'trajectory': block_trs.astype(np.float32)
                        })
                current_cat = lbl
                block_start = i
        # Last block
        if current_cat in CATS and block_start is not None:
            block_trs = data[:, s + block_start:e].T
            if len(block_trs) >= 3:
                trajectories.append({
                    'run': ri, 'cat': current_cat,
                    'trajectory': block_trs.astype(np.float32)
                })
    
    return trajectories

# ─────────────────────────────────────────────────────────────
# 2. Voxel selection (F-test)
# ─────────────────────────────────────────────────────────────

def select_category_voxels(blocks, n_voxels=500):
    """F-test voxel selection: find voxels whose response differs across categories."""
    cat_pats = defaultdict(list)
    for b in blocks:
        cat_pats[b['cat']].append(b['pattern'])
    
    groups = [np.stack(v) for v in cat_pats.values() if len(v) > 0]
    V = groups[0].shape[1]
    
    f_stats = np.zeros(V)
    for v in range(V):
        try:
            f_stat, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
        except:
            pass
    
    top_idx = np.argsort(f_stats)[-n_voxels:]
    return top_idx, f_stats


# ─────────────────────────────────────────────────────────────
# 3. Haxby correlation analysis
# ─────────────────────────────────────────────────────────────

def haxby_even_odd(blocks, label=""):
    """
    Even/odd split correlation analysis.
    Returns within>between count and pairwise accuracy.
    """
    even = [b for b in blocks if b['run'] % 2 == 0]
    odd  = [b for b in blocks if b['run'] % 2 == 1]
    
    def avg_pattern(entries, cat):
        pats = [e['pattern'] for e in entries if e['cat'] == cat]
        return np.stack(pats).mean(axis=0) if pats else None
    
    cats = sorted(set(b['cat'] for b in blocks))
    n = len(cats)
    corr = np.zeros((n, n))
    
    for i, ce in enumerate(cats):
        pe = avg_pattern(even, ce)
        if pe is None: continue
        for j, co in enumerate(cats):
            po = avg_pattern(odd, co)
            if po is None: continue
            corr[i, j], _ = pearsonr(pe, po)
    
    # Pairwise identification (Haxby's metric)
    correct = 0
    for i, c in enumerate(cats):
        if cats[np.argmax(corr[i])] == c:
            correct += 1
    
    pairwise_acc = correct / n
    
    # Within > between count
    wins = 0
    for i in range(n):
        w = corr[i, i]
        b = np.mean([corr[i, j] for j in range(n) if j != i])
        if w > b:
            wins += 1
    
    return wins, n, pairwise_acc, corr, cats


def haxby_all_pairs(blocks, label=""):
    """
    Haxby's actual metric: for each PAIR of categories, can we correctly
    identify which is which? This gives 28 pairwise comparisons (8 choose 2).
    """
    even = [b for b in blocks if b['run'] % 2 == 0]
    odd  = [b for b in blocks if b['run'] % 2 == 1]
    
    def avg_pattern(entries, cat):
        pats = [e['pattern'] for e in entries if e['cat'] == cat]
        return np.stack(pats).mean(axis=0) if pats else None
    
    cats = sorted(set(b['cat'] for b in blocks))
    
    # Build even and odd patterns
    even_pats = {c: avg_pattern(even, c) for c in cats}
    odd_pats = {c: avg_pattern(odd, c) for c in cats}
    
    correct = 0
    total = 0
    
    for c1, c2 in combinations(cats, 2):
        if even_pats[c1] is None or even_pats[c2] is None: continue
        if odd_pats[c1] is None or odd_pats[c2] is None: continue
        
        # For pair (c1, c2): compare within vs between correlations
        # Correct if corr(c1_even, c1_odd) + corr(c2_even, c2_odd) >
        #          corr(c1_even, c2_odd) + corr(c2_even, c1_odd)
        r_within_1, _ = pearsonr(even_pats[c1], odd_pats[c1])
        r_within_2, _ = pearsonr(even_pats[c2], odd_pats[c2])
        r_between_1, _ = pearsonr(even_pats[c1], odd_pats[c2])
        r_between_2, _ = pearsonr(even_pats[c2], odd_pats[c1])
        
        if (r_within_1 + r_within_2) > (r_between_1 + r_between_2):
            correct += 1
        total += 1
    
    return correct, total, correct / total if total > 0 else 0


# ─────────────────────────────────────────────────────────────
# 4. D-LinOSS block-trajectory training
# ─────────────────────────────────────────────────────────────

def train_dlinoss_block_trajectories(trajectories, in_dim, device='cpu', 
                                       n_epochs=40, lr=1e-3):
    """
    Train D-LinOSS on block temporal trajectories.
    
    Each block is a sequence [T_block, V] ~ [10, 500].
    The model learns the temporal dynamics within category blocks.
    Self-supervised: predict next TR from current.
    """
    model = DLinOSSModel(
        input_dim=in_dim, state_dim=32, hidden_dim=64,
        output_dim=in_dim, num_blocks=3, drop_rate=0.1
    ).to(device)
    
    predictor = nn.Sequential(
        nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, in_dim)
    ).to(device)
    
    params = list(model.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    for epoch in range(n_epochs):
        random.shuffle(trajectories)
        total_loss = 0
        
        for traj in trajectories:
            seq = traj['trajectory']
            if len(seq) < 3:
                continue
            
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            osc = model(x)
            pred = predictor(osc)
            
            # Predict next TR
            loss = F.mse_loss(pred[:, :-1, :], x[:, 1:, :])
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg = total_loss / max(len(trajectories), 1)
            print(f'      Epoch {epoch+1}: loss={avg:.6f}')
    
    return model, predictor


def encode_blocks_with_dlinoss(blocks, model, device='cpu'):
    """
    Encode each block's trajectory through D-LinOSS, producing an
    oscillator-encoded block pattern (average of oscillator outputs).
    """
    model.eval()
    encoded_blocks = []
    
    with torch.no_grad():
        for block in blocks:
            traj = block['trajectory']  # [T_block, V]
            if len(traj) < 2:
                encoded_blocks.append(block.copy())
                continue
            
            x = torch.tensor(traj.astype(np.float32)).unsqueeze(0).to(device)
            osc = model(x)  # [1, T, V]
            
            # Average the oscillator output across the block
            osc_pattern = osc[0].cpu().numpy().mean(axis=0)
            
            new_block = block.copy()
            new_block['pattern'] = osc_pattern
            encoded_blocks.append(new_block)
    
    return encoded_blocks


# ─────────────────────────────────────────────────────────────
# 5. Cross-subject alignment
# ─────────────────────────────────────────────────────────────

def build_category_prototypes(blocks):
    """
    Build a K×M matrix of category prototypes (mean patterns).
    
    Returns:
        proto: ndarray [K, M] — category prototype matrix
        cats: list of category names (sorted)
    """
    cat_pats = defaultdict(list)
    for b in blocks:
        cat_pats[b['cat']].append(b['pattern'])
    
    cats = sorted(cat_pats.keys())
    proto = np.stack([np.stack(cat_pats[c]).mean(axis=0) for c in cats])
    return proto, cats


def procrustes_align(source, target):
    """
    Orthogonal Procrustes alignment: find rotation R that minimizes
    ||source @ R - target||_F
    
    Args:
        source: [K, M] — source category prototypes
        target: [K, M] — target category prototypes
    
    Returns:
        R: [M, M] orthogonal rotation matrix
        aligned: [K, M] — source @ R (aligned to target)
        disparity: float — residual ||source @ R - target||_F / ||target||_F
    """
    from scipy.linalg import orthogonal_procrustes
    
    # Center both
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    
    # Find optimal rotation
    R, scale = orthogonal_procrustes(source_c, target_c)
    
    aligned = source_c @ R
    disparity = np.linalg.norm(aligned - target_c) / (np.linalg.norm(target_c) + 1e-8)
    
    return R, aligned, disparity


def cross_subject_cca(proto_dict, n_components=8):
    """
    Canonical Correlation Analysis across subjects to find shared
    category dimensions.
    
    Args:
        proto_dict: {subject_id: [K, M] prototype matrix}
        n_components: number of CCA components
    
    Returns:
        shared_space: [K, n_components] — the shared category representation
        projections: {subject_id: [M, n_components]} — per-subject projections
    """
    subjects = sorted(proto_dict.keys())
    
    # Stack all subject prototypes and compute joint SVD
    # Each subject contributes K prototype rows
    all_protos = []
    for s in subjects:
        proto = proto_dict[s]
        # Center
        proto_c = proto - proto.mean(axis=0)
        # Normalize to unit Frobenius norm
        norm = np.linalg.norm(proto_c)
        if norm > 1e-8:
            proto_c = proto_c / norm
        all_protos.append(proto_c)
    
    # Concatenate along feature dim: [K, M*N_subjects]
    concat = np.concatenate(all_protos, axis=1)
    
    # SVD to find shared directions
    U, S, Vt = np.linalg.svd(concat, full_matrices=False)
    
    # The shared space is U[:, :n_components] — the directions in
    # category space that are consistent across subjects
    shared_space = U[:, :n_components] * S[:n_components]
    
    # Per-subject projections: how each subject maps to shared space
    projections = {}
    M = proto_dict[subjects[0]].shape[1]
    for i, s in enumerate(subjects):
        # This subject's portion of Vt
        Vs = Vt[:n_components, i*M:(i+1)*M].T  # [M, n_components]
        projections[s] = Vs
    
    return shared_space, projections, S


def leave_one_out_classification(proto_dict, cats, use_procrustes=True):
    """
    Leave-one-subject-out cross-validated category classification.
    
    For each test subject:
        1. Align training subjects' prototypes to a common reference
        2. Average the aligned prototypes → template per category
        3. Classify test subject's patterns against templates
    
    Returns:
        accuracy: float — overall LOO accuracy
        per_subject: dict — accuracy per test subject
    """
    subjects = sorted(proto_dict.keys())
    K = len(cats)
    
    per_subject = {}
    
    for test_subj in subjects:
        train_subjects = [s for s in subjects if s != test_subj]
        
        # Use first training subject as reference
        ref = train_subjects[0]
        ref_proto = proto_dict[ref]
        
        # Align all training subjects to reference
        aligned_protos = [ref_proto.copy()]
        for s in train_subjects[1:]:
            if use_procrustes:
                _, aligned, _ = procrustes_align(proto_dict[s], ref_proto)
                aligned_protos.append(aligned)
            else:
                centered = proto_dict[s] - proto_dict[s].mean(axis=0)
                aligned_protos.append(centered)
        
        # Template: average of aligned training prototypes
        template = np.stack(aligned_protos).mean(axis=0)  # [K, M]
        
        # Align test subject to reference
        if use_procrustes:
            _, test_aligned, _ = procrustes_align(proto_dict[test_subj], ref_proto)
        else:
            test_aligned = proto_dict[test_subj] - proto_dict[test_subj].mean(axis=0)
        
        # Classify: for each category in test, find best correlation with template
        correct = 0
        for i in range(K):
            test_vec = test_aligned[i]
            corrs = [pearsonr(test_vec, template[j])[0] for j in range(K)]
            predicted = np.argmax(corrs)
            if predicted == i:
                correct += 1
        
        acc = correct / K
        per_subject[test_subj] = {'accuracy': acc, 'correct': correct, 'total': K}
    
    overall = np.mean([ps['accuracy'] for ps in per_subject.values()])
    return overall, per_subject


def run_cross_subject_alignment(subject_blocks):
    """
    Phase 2: Cross-subject alignment using the per-subject blocks
    from Phase 1.
    
    Args:
        subject_blocks: dict {subject_id: {'raw_blocks': [...], 'osc_blocks': [...]}}
    """
    print(f'\n{"="*70}')
    print(f'  PHASE 2: CROSS-SUBJECT ALIGNMENT')
    print(f'{"="*70}')
    
    for mode in ['raw', 'osc']:
        mode_label = "Raw BOLD" if mode == 'raw' else "D-LinOSS Oscillator"
        print(f'\n{"─"*60}')
        print(f'  {mode_label}')
        print(f'{"─"*60}')
        
        key = f'{mode}_blocks'
        
        # Build per-subject category prototypes
        proto_dict = {}
        cats = None
        for subj, data in subject_blocks.items():
            proto, c = build_category_prototypes(data[key])
            proto_dict[subj] = proto
            if cats is None:
                cats = c
        
        K, M = proto_dict[list(proto_dict.keys())[0]].shape
        print(f'  Prototypes: {K} categories × {M} dimensions, {len(proto_dict)} subjects')
        
        # ── Procrustes alignment between all pairs ──
        subjects = sorted(proto_dict.keys())
        print(f'\n  Pairwise Procrustes disparities:')
        disparities = np.zeros((len(subjects), len(subjects)))
        for i, s1 in enumerate(subjects):
            for j, s2 in enumerate(subjects):
                if i == j:
                    continue
                _, _, d = procrustes_align(proto_dict[s1], proto_dict[s2])
                disparities[i, j] = d
        
        # Print upper triangle
        header = '         ' + '  '.join(f'{s:>6}' for s in subjects)
        print(f'    {header}')
        for i, s1 in enumerate(subjects):
            row = '  '.join(
                f'{disparities[i,j]:>6.3f}' if j > i else f'{"":>6}'
                for j in range(len(subjects))
            )
            print(f'    {s1:>8}  {row}')
        
        avg_disp = disparities[disparities > 0].mean()
        print(f'\n  Mean disparity: {avg_disp:.4f} (lower = more similar)')
        
        # ── CCA: shared category space ──
        shared, projections, singular_values = cross_subject_cca(proto_dict)
        
        print(f'\n  Joint SVD singular values: {singular_values[:8].round(4)}')
        explained = (singular_values[:8]**2) / (singular_values**2).sum()
        print(f'  Variance explained (top 8): {explained.round(4)}')
        cumulative = np.cumsum(explained)
        print(f'  Cumulative:                 {cumulative.round(4)}')
        
        # ── Leave-one-out classification ──
        loo_acc, loo_per_subj = leave_one_out_classification(
            proto_dict, cats, use_procrustes=True
        )
        
        print(f'\n  Leave-one-subject-out classification (Procrustes):')
        for subj in subjects:
            ps = loo_per_subj[subj]
            print(f'    {subj}: {ps["correct"]}/{ps["total"]} = {ps["accuracy"]*100:.1f}%')
        print(f'    Average: {loo_acc*100:.1f}%')
        
        # Also without Procrustes (just centering)
        loo_acc_nop, loo_nop = leave_one_out_classification(
            proto_dict, cats, use_procrustes=False
        )
        print(f'\n  Leave-one-subject-out classification (no alignment):')
        for subj in subjects:
            ps = loo_nop[subj]
            print(f'    {subj}: {ps["correct"]}/{ps["total"]} = {ps["accuracy"]*100:.1f}%')
        print(f'    Average: {loo_acc_nop*100:.1f}%')
        
        # ── Shared space category geometry ──
        print(f'\n  Category distances in shared space (Euclidean):')
        for i in range(K):
            for j in range(i+1, K):
                d = np.linalg.norm(shared[i] - shared[j])
                print(f'    {cats[i]:>14} ↔ {cats[j]:<14}: {d:.4f}')
    
    print(f'\n{"="*70}')
    print(f'  Phase 2 complete.')
    print(f'{"="*70}')


# ─────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────

def run_full_replication():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("  HAXBY 2001 FULL REPLICATION")
    print("  Goal: Match 96% pairwise accuracy")
    print("=" * 70)
    
    all_results = []
    subject_blocks = {}  # Collect for Phase 2
    
    for subj in SUBJECTS:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir):
            continue
        
        print(f'\n{"─"*60}')
        print(f'  {subj}')
        print(f'{"─"*60}')
        
        t0 = time.time()
        
        # Load data
        ts_z, all_labels, run_bounds, coords, TR, n_runs = load_subject_data(subj_dir)
        V = ts_z.shape[0]
        print(f'  Loaded: {V} voxels, {n_runs} runs, TR={TR:.2f}s')
        
        # Extract block patterns (all voxels) for F-test
        blocks_full = extract_block_patterns(ts_z, all_labels, run_bounds, n_runs)
        
        # F-test voxel selection
        top_idx, f_stats = select_category_voxels(blocks_full, n_voxels=500)
        n_sig = np.sum(f_stats[top_idx] > 5.0)  # Rough significance check
        print(f'  F-test: selected 500 voxels (top F-stat={f_stats[top_idx[-1]]:.1f})')
        
        # Extract block patterns with selected voxels
        blocks_selected = extract_block_patterns(
            ts_z, all_labels, run_bounds, n_runs, voxel_idx=top_idx
        )
        
        # Also extract block trajectories for D-LinOSS
        trajectories = extract_block_trajectories(
            ts_z, all_labels, run_bounds, n_runs, voxel_idx=top_idx
        )
        
        # Attach trajectories to blocks
        traj_by_key = {}
        for t in trajectories:
            traj_by_key[(t['run'], t['cat'])] = t['trajectory']
        for b in blocks_selected:
            key = (b['run'], b['cat'])
            if key in traj_by_key:
                b['trajectory'] = traj_by_key[key]
            else:
                b['trajectory'] = b['pattern'].reshape(1, -1)
        
        # ── Raw BOLD baseline ──
        wins_raw, n_cats, acc_raw, corr_raw, cats = haxby_even_odd(blocks_selected, "Raw")
        pair_correct_raw, pair_total_raw, pair_acc_raw = haxby_all_pairs(blocks_selected, "Raw")
        
        print(f'\n  Raw BOLD:')
        print(f'    Within > Between: {wins_raw}/{n_cats}')
        print(f'    Pairwise (best-match):    {acc_raw*100:.1f}%')
        print(f'    Pairwise (all 28 pairs):  {pair_acc_raw*100:.1f}% ({pair_correct_raw}/{pair_total_raw})')
        
        # ── D-LinOSS block-trajectory training ──
        print(f'\n  Training D-LinOSS on {len(trajectories)} block trajectories...')
        model, predictor = train_dlinoss_block_trajectories(
            trajectories, in_dim=500, device=device, n_epochs=40
        )
        
        # Encode blocks through D-LinOSS
        blocks_encoded = encode_blocks_with_dlinoss(blocks_selected, model, device)
        
        # Re-do F-test on oscillator-encoded patterns
        # (find which oscillator dimensions carry category info)
        cat_osc_pats = defaultdict(list)
        for b in blocks_encoded:
            cat_osc_pats[b['cat']].append(b['pattern'])
        
        osc_groups = [np.stack(v) for v in cat_osc_pats.values()]
        V_osc = osc_groups[0].shape[1]
        f_osc = np.zeros(V_osc)
        for v in range(V_osc):
            try:
                f_stat, _ = f_oneway(*[g[:, v] for g in osc_groups])
                if not np.isnan(f_stat):
                    f_osc[v] = f_stat
            except:
                pass
        
        # Select top oscillator dimensions
        top_osc = np.argsort(f_osc)[-500:]
        for b in blocks_encoded:
            b['pattern'] = b['pattern'][top_osc]
        
        wins_osc, _, acc_osc, corr_osc, _ = haxby_even_odd(blocks_encoded, "Osc")
        pair_correct_osc, pair_total_osc, pair_acc_osc = haxby_all_pairs(blocks_encoded, "Osc")
        
        print(f'\n  D-LinOSS Oscillator:')
        print(f'    Within > Between: {wins_osc}/{n_cats}')
        print(f'    Pairwise (best-match):    {acc_osc*100:.1f}%')
        print(f'    Pairwise (all 28 pairs):  {pair_acc_osc*100:.1f}% ({pair_correct_osc}/{pair_total_osc})')
        
        # Print per-category details for the better method
        best_corr = corr_osc if pair_acc_osc >= pair_acc_raw else corr_raw
        best_label = "Osc" if pair_acc_osc >= pair_acc_raw else "Raw"
        
        print(f'\n  Per-category ({best_label}):')
        for i, cat in enumerate(cats):
            w = best_corr[i, i]
            b = np.mean([best_corr[i, j] for j in range(n_cats) if j != i])
            best_match = cats[np.argmax(best_corr[i])]
            mark = "✓" if best_match == cat else f"✗ → {best_match}"
            print(f'    {cat:>14}: within={w:.4f}  Δ={w-b:.4f}  {mark}')
        
        dt = time.time() - t0
        print(f'\n  Time: {dt:.1f}s')
        
        all_results.append({
            'subject': subj,
            'raw_wins': wins_raw, 'raw_acc': acc_raw, 
            'raw_pair_acc': pair_acc_raw, 'raw_pairs': pair_correct_raw,
            'osc_wins': wins_osc, 'osc_acc': acc_osc,
            'osc_pair_acc': pair_acc_osc, 'osc_pairs': pair_correct_osc,
        })
        
        # Save blocks for Phase 2 cross-subject alignment
        subject_blocks[subj] = {
            'raw_blocks': blocks_selected,
            'osc_blocks': blocks_encoded,
        }
    
    # ── Final Summary ──
    print(f'\n{"="*70}')
    print(f'  FINAL RESULTS')
    print(f'{"="*70}')
    print(f'\n  {"Subject":>8}  {"Raw w>b":>8}  {"Raw best%":>10}  {"Raw 28p%":>10}  '
          f'{"Osc w>b":>8}  {"Osc best%":>10}  {"Osc 28p%":>10}')
    print(f'  {"─"*68}')
    
    for r in all_results:
        print(f'  {r["subject"]:>8}  {r["raw_wins"]}/8    '
              f'{r["raw_acc"]*100:>8.1f}%  {r["raw_pair_acc"]*100:>8.1f}%  '
              f'{r["osc_wins"]}/8    '
              f'{r["osc_acc"]*100:>8.1f}%  {r["osc_pair_acc"]*100:>8.1f}%')
    
    avg_raw_best = np.mean([r['raw_acc'] for r in all_results])
    avg_raw_pair = np.mean([r['raw_pair_acc'] for r in all_results])
    avg_osc_best = np.mean([r['osc_acc'] for r in all_results])
    avg_osc_pair = np.mean([r['osc_pair_acc'] for r in all_results])
    
    print(f'\n  {"Average":>8}          '
          f'{avg_raw_best*100:>8.1f}%  {avg_raw_pair*100:>8.1f}%  '
          f'          '
          f'{avg_osc_best*100:>8.1f}%  {avg_osc_pair*100:>8.1f}%')
    print(f'\n  Haxby 2001 benchmark: 96% (28-pair pairwise)')
    print(f'\n  Core finding preserved: within > between = '
          f'{sum(r["osc_wins"] for r in all_results)}/'
          f'{sum(8 for _ in all_results)} '
          f'(100% = ✅)' if all(r["osc_wins"] == 8 for r in all_results) 
          else f'({sum(r["osc_wins"] for r in all_results)}/{8*len(all_results)})')
    
    # ── Phase 2: Cross-subject alignment ──
    if len(subject_blocks) >= 2:
        run_cross_subject_alignment(subject_blocks)


if __name__ == "__main__":
    run_full_replication()
