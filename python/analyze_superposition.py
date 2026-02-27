"""
Superposition Analysis: Per-Category Loss Decomposition
=========================================================

If the brain encodes visual categories in superposition (Elhage et al. 2022),
then M category directions are packed into a P-dimensional space where M > P.

Signature of superposition:
  - Some category pairs interfere (positive similarity) -> higher loss
  - Some category pairs are well-separated (negative similarity) -> lower loss
  - The loss structure should be CONSISTENT across subjects if it reflects
    the geometry of the encoding, not subject-specific noise.

We decompose the global reconstruction loss into per-category contributions:
  L_total = sum_cat [ L_cat * (n_cat / n_total) ]
  L_cat   = mean over TRs where label == cat of ||y - x||^2

If categories are in superposition, we expect:
  1. L_cat varies across categories (not uniform)
  2. The ranking of L_cat is similar across subjects
  3. Categories with high mutual similarity have higher L_cat
     (interference -> harder to reconstruct)
"""

import os
import sys
import json
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

import nibabel as nib
import torch
from dlinoss.dlinoss_torch import DLinOSSModel
from dlinoss.mri_artifact_correction import MRIHardwareConfig, PhysiologicalNoiseRegressor
from run_visual_dlinoss import (
    load_haxby_subject, build_block_labels,
    extract_brain_voxels, preprocess_bold,
)


def compute_per_category_loss(model, x_tensor, labels):
    """
    Decompose reconstruction loss by stimulus category.

    Returns dict mapping category -> {loss, n_trs, contribution}
    """
    model.eval()
    with torch.no_grad():
        y = model(x_tensor)

    x_np = x_tensor.numpy()
    y_np = y.numpy()

    # Per-timepoint squared error
    per_tr_loss = np.mean((y_np - x_np) ** 2, axis=1)  # (T,)

    categories = sorted(set(l for l in labels if l != 'rest'))
    total_trs = len(labels)

    results = {}
    for cat in categories + ['rest']:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if len(idx) == 0:
            continue
        cat_loss = per_tr_loss[idx].mean()
        cat_std = per_tr_loss[idx].std()
        results[cat] = {
            'mean_loss': float(cat_loss),
            'std_loss': float(cat_std),
            'n_trs': len(idx),
            'fraction': len(idx) / total_trs,
            'weighted_contribution': float(cat_loss * len(idx) / total_trs),
        }

    return results


def compute_category_geometry(model, x_tensor, labels):
    """
    Analyze the geometry of category representations in oscillator space.

    For each category, extract the mean model output (the learned representation).
    Then compute:
    1. Pairwise cosine similarities -> interference structure
    2. Singular values of the category representation matrix
       -> if rank < n_categories, categories are in superposition
    3. Per-category "exclusivity" = how much of the pattern is unique
    """
    model.eval()
    with torch.no_grad():
        y = model(x_tensor)

    y_np = y.numpy()
    x_np = x_tensor.numpy()

    categories = sorted(set(l for l in labels if l != 'rest'))

    # Mean representation per category
    cat_reps = {}
    cat_residuals = {}
    for cat in categories:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if len(idx) > 0:
            cat_reps[cat] = y_np[idx].mean(axis=0)         # mean output
            cat_residuals[cat] = (y_np[idx] - x_np[idx]).mean(axis=0)  # mean error

    # Build representation matrix: (n_cats, n_voxels)
    cat_names = sorted(cat_reps.keys())
    rep_matrix = np.array([cat_reps[c] for c in cat_names])

    # SVD of representation matrix
    U, S, Vt = np.linalg.svd(rep_matrix, full_matrices=False)
    n_cats = len(cat_names)

    # Effective rank: how many dimensions needed to explain 95% variance
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    eff_rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    eff_rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)

    # Cosine similarity matrix
    norms = np.linalg.norm(rep_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = rep_matrix / norms
    cosine_sim = normed @ normed.T

    # Per-category exclusivity: what fraction of this category's pattern
    # is orthogonal to all other categories?
    exclusivity = {}
    for i, cat in enumerate(cat_names):
        # Project this category onto the subspace spanned by others
        others = np.delete(rep_matrix, i, axis=0)
        if others.shape[0] > 0:
            # Projection using pseudoinverse
            proj = others.T @ np.linalg.pinv(others.T)
            projected = proj @ rep_matrix[i]
            residual = rep_matrix[i] - projected
            excl = np.linalg.norm(residual) / (np.linalg.norm(rep_matrix[i]) + 1e-10)
            exclusivity[cat] = float(excl)
        else:
            exclusivity[cat] = 1.0

    return {
        'singular_values': S.tolist(),
        'cumulative_variance': cumvar.tolist(),
        'effective_rank_95': eff_rank_95,
        'effective_rank_90': eff_rank_90,
        'n_categories': n_cats,
        'cosine_similarity': cosine_sim.tolist(),
        'category_names': cat_names,
        'exclusivity': exclusivity,
    }


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')
    subjects_with_events = ['sub-4', 'sub-5', 'sub-6']

    print("=" * 70)
    print("  SUPERPOSITION ANALYSIS: Per-Category Loss Decomposition")
    print("  Dataset: Haxby ds000105 (Visual Object Recognition)")
    print("=" * 70)

    all_cat_losses = {}  # subject -> category -> loss
    all_geometries = {}

    for subj in subjects_with_events:
        subj_dir = os.path.join(base, subj)
        print(f"\n  Processing {subj}...")

        # Load and preprocess (same as run_visual_dlinoss)
        bold_runs, events_runs, TR = load_haxby_subject(subj_dir, max_runs=4)
        bold_concat = np.concatenate(bold_runs, axis=-1)

        all_labels = []
        for events, bold in zip(events_runs, bold_runs):
            T_run = bold.shape[-1]
            run_labels = build_block_labels(events, T_run, TR)
            all_labels.extend(run_labels)

        ts_2d, brain_mask, coords = extract_brain_voxels(bold_concat, percentile=25)
        variances = np.var(ts_2d, axis=1)
        top_idx = np.argsort(variances)[-400:]
        selected_ts = ts_2d[top_idx]

        hw = MRIHardwareConfig(B0=3.0, TR=TR)
        cleaned = preprocess_bold(selected_ts, TR, hw)
        clean_input = cleaned.T  # (T, N)

        x = torch.tensor(clean_input, dtype=torch.float32)

        # Build and train model
        model = DLinOSSModel(
            input_dim=400, state_dim=32, hidden_dim=64,
            output_dim=400, num_blocks=3, drop_rate=0.0,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(80):
            optimizer.zero_grad()
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, x)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"    Final global loss: {loss.item():.6f}")

        # Per-category loss
        cat_losses = compute_per_category_loss(model, x, all_labels)
        all_cat_losses[subj] = cat_losses

        # Geometry
        geometry = compute_category_geometry(model, x, all_labels)
        all_geometries[subj] = geometry

    # ====================================================================
    #  RESULTS
    # ====================================================================

    categories = sorted(set(
        cat for subj_losses in all_cat_losses.values()
        for cat in subj_losses.keys() if cat != 'rest'
    ))

    # 1. Per-category loss table
    print(f"\n{'='*70}")
    print(f"  1. PER-CATEGORY RECONSTRUCTION LOSS")
    print(f"{'='*70}")
    print(f"\n  If categories are in superposition, L_cat should vary")
    print(f"  AND the ranking should be consistent across subjects.\n")

    header = f"  {'Category':15s}" + "".join(f"  {s:>10s}" for s in subjects_with_events) + "    Mean   Rank"
    print(header)
    print(f"  {'-'*75}")

    cat_means = {}
    for cat in categories:
        vals = []
        row = f"  {cat:15s}"
        for subj in subjects_with_events:
            if cat in all_cat_losses[subj]:
                v = all_cat_losses[subj][cat]['mean_loss']
                vals.append(v)
                row += f"  {v:10.4f}"
            else:
                row += f"  {'n/a':>10s}"
        mean_val = np.mean(vals) if vals else 0
        cat_means[cat] = mean_val
        row += f"  {mean_val:8.4f}"
        print(row)

    # Add rest
    rest_vals = []
    row = f"  {'rest':15s}"
    for subj in subjects_with_events:
        if 'rest' in all_cat_losses[subj]:
            v = all_cat_losses[subj]['rest']['mean_loss']
            rest_vals.append(v)
            row += f"  {v:10.4f}"
    row += f"  {np.mean(rest_vals):8.4f}"
    print(row)

    # Rankings
    print(f"\n  Loss rankings (1 = hardest to reconstruct):")
    for subj in subjects_with_events:
        subj_cats = {c: all_cat_losses[subj][c]['mean_loss']
                     for c in categories if c in all_cat_losses[subj]}
        ranked = sorted(subj_cats.items(), key=lambda x: -x[1])
        rank_str = " > ".join(f"{c}({v:.3f})" for c, v in ranked)
        print(f"    {subj}: {rank_str}")

    # Cross-subject rank correlation
    print(f"\n  Cross-subject rank consistency:")
    from scipy.stats import spearmanr
    for i, s1 in enumerate(subjects_with_events):
        for j, s2 in enumerate(subjects_with_events):
            if j <= i:
                continue
            v1 = [all_cat_losses[s1].get(c, {}).get('mean_loss', 0) for c in categories]
            v2 = [all_cat_losses[s2].get(c, {}).get('mean_loss', 0) for c in categories]
            rho, pval = spearmanr(v1, v2)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            print(f"    {s1} vs {s2}: Spearman rho = {rho:.3f} (p = {pval:.4f}) {sig}")

    # 2. Superposition geometry
    print(f"\n{'='*70}")
    print(f"  2. REPRESENTATION GEOMETRY (Superposition Test)")
    print(f"{'='*70}")
    print(f"\n  If 8 categories are encoded in superposition,")
    print(f"  the effective rank should be < 8.")
    print(f"  Rank = 8 means fully separated (no superposition).")
    print(f"  Rank < 8 means categories share neural dimensions.\n")

    for subj in subjects_with_events:
        g = all_geometries[subj]
        print(f"  {subj}:")
        print(f"    Singular values: {[f'{s:.3f}' for s in g['singular_values']]}")
        print(f"    Cumulative variance: {[f'{v:.2%}' for v in g['cumulative_variance']]}")
        print(f"    Effective rank (90% var): {g['effective_rank_90']} / {g['n_categories']}")
        print(f"    Effective rank (95% var): {g['effective_rank_95']} / {g['n_categories']}")

        # Superposition ratio: effective_rank / n_categories
        sp_ratio = g['effective_rank_95'] / g['n_categories']
        if sp_ratio < 0.5:
            verdict = "STRONG superposition"
        elif sp_ratio < 0.75:
            verdict = "MODERATE superposition"
        elif sp_ratio < 1.0:
            verdict = "MILD superposition"
        else:
            verdict = "NO superposition (fully separated)"
        print(f"    Superposition ratio: {sp_ratio:.2f} -> {verdict}")

    # 3. Category exclusivity
    print(f"\n{'='*70}")
    print(f"  3. CATEGORY EXCLUSIVITY (per-category uniqueness)")
    print(f"{'='*70}")
    print(f"\n  Exclusivity = fraction of pattern orthogonal to all other categories.")
    print(f"  Low exclusivity = high overlap = strong superposition for that category.\n")

    header = f"  {'Category':15s}" + "".join(f"  {s:>10s}" for s in subjects_with_events) + "    Mean"
    print(header)
    print(f"  {'-'*60}")
    for cat in categories:
        row = f"  {cat:15s}"
        vals = []
        for subj in subjects_with_events:
            excl = all_geometries[subj]['exclusivity'].get(cat, 0)
            row += f"  {excl:10.3f}"
            vals.append(excl)
        row += f"  {np.mean(vals):8.3f}"
        print(row)

    # 4. Loss-similarity correlation
    print(f"\n{'='*70}")
    print(f"  4. INTERFERENCE TEST: Does similarity predict loss?")
    print(f"{'='*70}")
    print(f"\n  If superposition causes interference, categories with")
    print(f"  HIGH similarity to others should have HIGHER loss.\n")

    for subj in subjects_with_events:
        g = all_geometries[subj]
        sim = np.array(g['cosine_similarity'])
        cats = g['category_names']

        # Mean off-diagonal similarity per category
        mean_sim = []
        cat_loss = []
        for i, cat in enumerate(cats):
            off_diag = [sim[i, j] for j in range(len(cats)) if j != i]
            mean_sim.append(np.mean(off_diag))
            cat_loss.append(all_cat_losses[subj].get(cat, {}).get('mean_loss', 0))

        rho, pval = spearmanr(mean_sim, cat_loss)
        print(f"  {subj}: similarity-loss correlation rho = {rho:.3f} (p = {pval:.4f})")
        if rho > 0:
            print(f"    -> POSITIVE: higher similarity = higher loss (superposition interference)")
        else:
            print(f"    -> NEGATIVE: higher similarity = lower loss (no interference evidence)")

    # 5. Grand summary
    print(f"\n{'='*70}")
    print(f"  SUPERPOSITION VERDICT")
    print(f"{'='*70}")

    mean_ranks_95 = np.mean([all_geometries[s]['effective_rank_95'] for s in subjects_with_events])
    mean_ranks_90 = np.mean([all_geometries[s]['effective_rank_90'] for s in subjects_with_events])

    print(f"\n  8 visual categories, mean effective rank:")
    print(f"    At 90% variance: {mean_ranks_90:.1f} / 8")
    print(f"    At 95% variance: {mean_ranks_95:.1f} / 8")
    print(f"    Superposition ratio: {mean_ranks_95/8:.2f}")

    if mean_ranks_95 < 4:
        print(f"\n  STRONG EVIDENCE for superposition.")
        print(f"  {8 - mean_ranks_95:.0f} categories share neural dimensions.")
    elif mean_ranks_95 < 6:
        print(f"\n  MODERATE EVIDENCE for superposition.")
        print(f"  Categories partially overlap in representational space.")
    elif mean_ranks_95 < 8:
        print(f"\n  MILD EVIDENCE for superposition.")
        print(f"  Most categories are separated but some share dimensions.")
    else:
        print(f"\n  NO EVIDENCE for superposition.")
        print(f"  Categories are fully separated in the model's representation.")


if __name__ == '__main__':
    main()
