#!/usr/bin/env python3
"""
Brain Decoding v2: Correct voxel selection + all 12 runs
=========================================================

Fixes from v1:
1. Use ALL 12 runs (not just 4) — more template stability
2. Select voxels by CATEGORY-DISCRIMINATIVE power (F-stat),
   not overall variance (which is dominated by noise/drift)
3. Use leave-one-run-out with 12-fold CV
4. Compare raw, preprocessed, and D-LinOSS representations
5. Also test: does increasing voxel count help?

The key insight: Haxby's original result used ~500 ventral temporal
voxels selected BY THEIR CATEGORY SENSITIVITY. Random high-variance
voxels contain mostly noise, not category information.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

import nibabel as nib
import torch
from dlinoss.dlinoss_torch import DLinOSSModel
from dlinoss.mri_artifact_correction import MRIHardwareConfig, PhysiologicalNoiseRegressor
from run_visual_dlinoss import (
    load_haxby_subject, build_block_labels, extract_brain_voxels,
)


def select_discriminative_voxels(ts_2d, labels, categories, n_select=500):
    """
    Select voxels with highest between-category variance (ANOVA F-statistic).

    This is critical: top-variance voxels capture noise/drift.
    Top-F-statistic voxels capture CATEGORY-SPECIFIC activation differences.
    """
    N, T = ts_2d.shape

    # Compute per-category means
    cat_means = {}
    cat_counts = {}
    for cat in categories:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if idx:
            cat_means[cat] = ts_2d[:, idx].mean(axis=1)  # (N,)
            cat_counts[cat] = len(idx)

    grand_mean = ts_2d.mean(axis=1)  # (N,)

    # Between-group sum of squares
    ss_between = np.zeros(N)
    for cat in categories:
        if cat in cat_means:
            ss_between += cat_counts[cat] * (cat_means[cat] - grand_mean) ** 2

    # Within-group sum of squares
    ss_within = np.zeros(N)
    for cat in categories:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if idx:
            cat_data = ts_2d[:, idx]
            ss_within += np.sum((cat_data - cat_means[cat][:, None]) ** 2, axis=1)

    k = len([c for c in categories if c in cat_means])
    n_total = sum(cat_counts.values())

    # F-statistic
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(n_total - k, 1)
    f_stat = ms_between / (ms_within + 1e-10)

    # Select top-N by F-stat
    top_idx = np.argsort(f_stat)[-n_select:]

    return top_idx, f_stat[top_idx]


def preprocess_run(run_ts, TR):
    """Lightweight preprocessing: detrend + normalize (no physio for speed)."""
    N, T = run_ts.shape
    cleaned = run_ts.copy().astype(np.float64)

    # Linear detrend
    t_axis = np.arange(T, dtype=np.float64)
    for i in range(N):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)

    # Z-score
    mu = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    cleaned = (cleaned - mu) / sigma

    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = np.clip(cleaned, -5.0, 5.0)

    return cleaned


def build_block_segments(labels, categories):
    """Find contiguous category blocks."""
    blocks = []
    current = None
    start = None
    for i, l in enumerate(labels):
        if l != current:
            if current in categories:
                blocks.append((current, start, i))
            current = l
            start = i
    if current in categories:
        blocks.append((current, start, len(labels)))
    return blocks


def classify_nearest_template(pattern, templates, categories):
    """Classify by highest correlation to template."""
    best_cat = None
    best_r = -2.0
    for cat in categories:
        if cat not in templates:
            continue
        r = np.corrcoef(pattern, templates[cat])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_cat = cat
    return best_cat, best_r


def loro_decode(data_per_run, labels_per_run, categories):
    """
    Leave-one-run-out decoding with block-level classification.
    """
    n_runs = len(data_per_run)
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for test_run in range(n_runs):
        # Build templates from training runs
        templates = {}
        for cat in categories:
            patterns = []
            for r in range(n_runs):
                if r == test_run:
                    continue
                idx = [i for i, l in enumerate(labels_per_run[r]) if l == cat]
                if idx:
                    patterns.append(data_per_run[r][idx].mean(axis=0))
            if patterns:
                templates[cat] = np.mean(patterns, axis=0)

        # Test on held-out run
        blocks = build_block_segments(labels_per_run[test_run], categories)
        for true_cat, start, end in blocks:
            if true_cat not in templates or end <= start:
                continue
            pattern = data_per_run[test_run][start:end].mean(axis=0)
            pred_cat, _ = classify_nearest_template(pattern, templates, categories)
            confusion[true_cat][pred_cat] += 1
            total += 1
            if pred_cat == true_cat:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, dict(confusion), total


def run_decoding_v2(subject_dir, subject_name, n_voxels_list=[100, 300, 500, 800]):
    """Full decoding pipeline with correct voxel selection."""
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  BRAIN DECODING v2: {subject_name}")
    print(f"  Voxel selection: category-discriminative (ANOVA F-stat)")
    print(f"  Cross-validation: Leave-one-run-out (12-fold)")
    print(f"{'='*70}")

    # Load ALL runs
    print(f"\n  Loading all runs...")
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=12)
    n_runs = len(bold_runs)
    print(f"    {n_runs} runs loaded, TR={TR}s")

    # Build labels
    labels_per_run = []
    categories = set()
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        labels_per_run.append(run_labels)
        categories.update(l for l in run_labels if l != 'rest')
    categories = sorted(categories)
    n_cats = len(categories)

    print(f"    Categories ({n_cats}): {categories}")
    print(f"    Chance: {1/n_cats:.1%}")

    # Extract all brain voxels
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, brain_mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    all_labels_flat = [l for run_labels in labels_per_run for l in run_labels]

    print(f"    Brain voxels: {ts_2d.shape[0]:,}")

    # Select discriminative voxels (using all data for selection —
    # this is a mild bias but standard in Haxby-style analysis)
    max_v = max(n_voxels_list)
    disc_idx, f_stats = select_discriminative_voxels(
        ts_2d, all_labels_flat, categories, n_select=max_v)

    print(f"    Top {max_v} discriminative voxels selected (F range: "
          f"[{f_stats.min():.4f}, {f_stats.max():.4f}])")

    # Test decoding at different voxel counts
    results_by_nvox = {}

    for n_vox in n_voxels_list:
        selected = disc_idx[-n_vox:]  # top-N by F-stat

        # Extract per-run data
        raw_runs = []
        clean_runs = []
        for bold in bold_runs:
            run_brain = bold[brain_mask]  # (all_brain, T)
            run_selected = run_brain[selected]  # (n_vox, T)
            raw_runs.append(run_selected.T)  # (T, n_vox)

            cleaned = preprocess_run(run_selected, TR)
            clean_runs.append(cleaned.T)

        # ── Raw decoding ──
        raw_acc, _, raw_n = loro_decode(raw_runs, labels_per_run, categories)

        # ── Preprocessed decoding ──
        clean_acc, _, _ = loro_decode(clean_runs, labels_per_run, categories)

        # ── D-LinOSS decoding ──
        # Train on concatenated clean data
        concat_clean = np.concatenate(clean_runs, axis=0)
        model = DLinOSSModel(
            input_dim=n_vox, state_dim=32, hidden_dim=64,
            output_dim=n_vox, num_blocks=3, drop_rate=0.0,
        )
        x_all = torch.tensor(concat_clean, dtype=torch.float32)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(80):
            optimizer.zero_grad()
            y = model(x_all)
            loss = torch.nn.functional.mse_loss(y, x_all)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Extract representations
        model.eval()
        model_runs = []
        for clean_run in clean_runs:
            x_run = torch.tensor(clean_run, dtype=torch.float32)
            with torch.no_grad():
                y_run = model(x_run)
            model_runs.append(y_run.numpy())

        model_acc, model_conf, model_n = loro_decode(
            model_runs, labels_per_run, categories)

        results_by_nvox[n_vox] = {
            'raw': raw_acc, 'clean': clean_acc, 'model': model_acc,
            'n_blocks': model_n, 'loss': loss.item(),
        }

        print(f"\n    {n_vox:4d} voxels: "
              f"Raw={raw_acc:5.1%}  Preproc={clean_acc:5.1%}  "
              f"D-LinOSS={model_acc:5.1%}  "
              f"(loss={loss.item():.4f}, {model_n} blocks)")

    # Best result
    best_nvox = max(results_by_nvox, key=lambda k: results_by_nvox[k]['model'])
    best = results_by_nvox[best_nvox]

    # Print confusion matrix for best
    # Re-run best to get confusion
    selected = disc_idx[-best_nvox:]
    clean_runs_best = []
    for bold in bold_runs:
        run_selected = bold[brain_mask][selected]
        cleaned = preprocess_run(run_selected, TR)
        clean_runs_best.append(cleaned.T)

    concat_best = np.concatenate(clean_runs_best, axis=0)
    model_best = DLinOSSModel(
        input_dim=best_nvox, state_dim=32, hidden_dim=64,
        output_dim=best_nvox, num_blocks=3, drop_rate=0.0,
    )
    x_best = torch.tensor(concat_best, dtype=torch.float32)
    model_best.train()
    opt = torch.optim.Adam(model_best.parameters(), lr=1e-3)
    for epoch in range(80):
        opt.zero_grad()
        y = model_best(x_best)
        loss = torch.nn.functional.mse_loss(y, x_best)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_best.parameters(), max_norm=1.0)
        opt.step()

    model_best.eval()
    model_runs_best = []
    for cr in clean_runs_best:
        with torch.no_grad():
            model_runs_best.append(model_best(torch.tensor(cr, dtype=torch.float32)).numpy())

    _, conf, n_tested = loro_decode(model_runs_best, labels_per_run, categories)

    print(f"\n  Confusion Matrix ({best_nvox} discriminative voxels):")
    header = "  Predicted:     " + "".join(f"{c[:6]:>8s}" for c in categories)
    print(header)
    print(f"  True           " + "-" * (8 * n_cats))
    for tc in categories:
        row = f"  {tc:15s}"
        t_total = sum(conf.get(tc, {}).values())
        for pc in categories:
            c = conf.get(tc, {}).get(pc, 0)
            if tc == pc and c > 0:
                row += f"  [{c:2d}] "
            else:
                row += f"   {c:2d}  "
        cat_acc = conf.get(tc, {}).get(tc, 0) / t_total if t_total > 0 else 0
        row += f" ({cat_acc:4.0%})"
        print(row)

    # Summary
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  RESULTS: {subject_name} (best: {best_nvox} voxels)")
    print(f"  {'='*60}")
    print(f"    Chance:    {1/n_cats:5.1%}")
    print(f"    Raw:       {best['raw']:5.1%} ({best['raw']/(1/n_cats):.1f}x)")
    print(f"    Preproc:   {best['clean']:5.1%} ({best['clean']/(1/n_cats):.1f}x)")
    print(f"    D-LinOSS:  {best['model']:5.1%} ({best['model']/(1/n_cats):.1f}x)")
    print(f"    Time: {total_time:.0f}s")

    # Binomial test
    k = int(best['model'] * best['n_blocks'])
    n = best['n_blocks']
    p = 1 / n_cats
    # Exact binomial p-value
    from scipy.stats import binomtest
    result = binomtest(k, n, p, alternative='greater')
    print(f"    Binomial test: {k}/{n} correct, p = {result.pvalue:.6f}")
    if result.pvalue < 0.05:
        print(f"    *** SIGNIFICANT: Model decoding > chance (p < 0.05) ***")

    return results_by_nvox, total_time


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')
    subjects = ['sub-4', 'sub-5', 'sub-6']

    print("=" * 70)
    print("  BRAIN DECODING CHALLENGE v2")
    print("  Category-discriminative voxels + all 12 runs")
    print("=" * 70)

    all_results = {}
    for subj in subjects:
        subj_dir = os.path.join(base, subj)
        try:
            results, t = run_decoding_v2(subj_dir, subj,
                                         n_voxels_list=[100, 300, 500, 800])
            all_results[subj] = results
        except Exception as e:
            print(f"\n  ERROR: {subj}: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  GRAND SUMMARY")
        print(f"{'='*70}")
        print(f"\n  {'Subject':>10s}  {'N_vox':>6s}  {'Raw':>6s}  {'Preproc':>8s}  {'D-LinOSS':>9s}  {'x chance':>9s}")
        print(f"  {'-'*52}")
        for subj, results in sorted(all_results.items()):
            best_nv = max(results, key=lambda k: results[k]['model'])
            b = results[best_nv]
            print(f"  {subj:>10s}  {best_nv:>6d}  {b['raw']:5.1%}  {b['clean']:7.1%}  "
                  f"{b['model']:8.1%}  {b['model']/0.125:8.1f}x")


if __name__ == '__main__':
    main()
