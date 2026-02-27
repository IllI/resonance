#!/usr/bin/env python3
"""
Brain Decoding Test: Can D-LinOSS identify what a subject is seeing?
=====================================================================

THE DEFINITIVE TEST OF SUPERPOSITION:

If the D-LinOSS model has genuinely learned the brain's category-specific
electromagnetic activation patterns, then:

  1. Train the model BLIND (self-supervised, no category labels)
  2. Extract the model's internal representation for each timepoint
  3. Build "category templates" from TRAINING runs
  4. On HELD-OUT runs, classify each block by nearest template
  5. If accuracy > 12.5% (chance for 8 categories), the model
     has captured category-discriminative neural dynamics

This is the classic Haxby et al. (2001) brain decoding paradigm,
but using D-LinOSS oscillator representations instead of raw voxels.

Cross-validation: Leave-one-run-out (LORO)
  - Train template on 3 runs, test on 1 run
  - Repeat for each run as test
  - Report mean accuracy

We also compare:
  A. Raw voxel decoding (baseline — how well can raw data decode?)
  B. D-LinOSS representation decoding (does the model add value?)
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict, Counter

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


# ============================================================================
#  Decoding Functions
# ============================================================================

def build_block_segments(labels, categories):
    """
    Find contiguous blocks of each category in the label sequence.

    Returns list of (category, start_tr, end_tr) tuples.
    """
    blocks = []
    current_cat = None
    block_start = None

    for i, label in enumerate(labels):
        if label != current_cat:
            if current_cat is not None and current_cat in categories:
                blocks.append((current_cat, block_start, i))
            current_cat = label
            block_start = i

    # Close last block
    if current_cat in categories:
        blocks.append((current_cat, block_start, len(labels)))

    return blocks


def classify_nearest_template(test_pattern, templates, cat_names):
    """
    Classify a test pattern by correlation to category templates.

    Returns (predicted_category, confidence_dict)
    """
    correlations = {}
    for cat in cat_names:
        r = np.corrcoef(test_pattern, templates[cat])[0, 1]
        correlations[cat] = r if np.isfinite(r) else -1.0

    predicted = max(correlations, key=correlations.get)
    return predicted, correlations


def decode_with_representations(data_per_run, labels_per_run, categories):
    """
    Leave-one-run-out cross-validation decoding.

    Args:
        data_per_run: list of (T_run, N) arrays (one per run)
        labels_per_run: list of label lists (one per run)
        categories: list of category names

    Returns:
        accuracy, confusion_matrix, detailed_results
    """
    n_runs = len(data_per_run)
    all_predictions = []
    all_true = []
    confusion = defaultdict(lambda: defaultdict(int))

    for test_run in range(n_runs):
        # Training: all runs except test
        train_data = [data_per_run[r] for r in range(n_runs) if r != test_run]
        train_labels = [labels_per_run[r] for r in range(n_runs) if r != test_run]

        # Build category templates from training runs
        templates = {}
        for cat in categories:
            cat_patterns = []
            for data, labels in zip(train_data, train_labels):
                idx = [i for i, l in enumerate(labels) if l == cat]
                if idx:
                    cat_patterns.append(data[idx].mean(axis=0))
            if cat_patterns:
                templates[cat] = np.mean(cat_patterns, axis=0)

        # Test: classify blocks in held-out run
        test_data = data_per_run[test_run]
        test_labels = labels_per_run[test_run]
        blocks = build_block_segments(test_labels, categories)

        for true_cat, start, end in blocks:
            if true_cat not in templates:
                continue
            # Average pattern over the block
            block_pattern = test_data[start:end].mean(axis=0)
            predicted, _ = classify_nearest_template(block_pattern, templates, list(templates.keys()))

            all_predictions.append(predicted)
            all_true.append(true_cat)
            confusion[true_cat][predicted] += 1

    # Compute accuracy
    correct = sum(1 for p, t in zip(all_predictions, all_true) if p == t)
    total = len(all_predictions)
    accuracy = correct / total if total > 0 else 0

    return accuracy, dict(confusion), total


# ============================================================================
#  Full Pipeline
# ============================================================================

def run_decoding_test(subject_dir, subject_name, max_runs=4, max_voxels=400):
    """
    Run the complete brain decoding test for one subject.

    Steps:
    1. Load BOLD data + events for each run separately
    2. Preprocess each run
    3. Train D-LinOSS on concatenated data (self-supervised, no labels)
    4. Extract representations for each run
    5. Decode with leave-one-run-out cross-validation
    6. Compare raw voxel decoding vs model-based decoding
    """
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  BRAIN DECODING TEST: {subject_name}")
    print(f"  Can D-LinOSS identify what the subject is seeing?")
    print(f"{'='*70}")

    # ── Load data ──
    print(f"\n  Stage 1: Loading data...")
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    n_runs = len(bold_runs)

    print(f"    {n_runs} runs, TR={TR}s, shape={bold_runs[0].shape}")

    # ── Build per-run labels ──
    labels_per_run = []
    categories = set()
    for i, (events, bold) in enumerate(zip(events_runs, bold_runs)):
        T_run = bold.shape[-1]
        run_labels = build_block_labels(events, T_run, TR)
        labels_per_run.append(run_labels)
        cats = set(l for l in run_labels if l != 'rest')
        categories.update(cats)
        print(f"    Run {i+1}: {T_run} TRs, categories: {cats}")

    categories = sorted(categories)
    print(f"    Categories: {categories}")
    print(f"    Chance level: {1/len(categories):.1%} ({len(categories)} categories)")

    # ── Extract consistent voxel set across runs ──
    print(f"\n  Stage 2: Voxel selection + preprocessing...")

    # Use mean across all runs for voxel selection
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, brain_mask, coords = extract_brain_voxels(bold_concat, percentile=25)
    variances = np.var(ts_2d, axis=1)
    top_idx = np.argsort(variances)[-max_voxels:]

    print(f"    Brain voxels: {ts_2d.shape[0]:,}")
    print(f"    Selected: {len(top_idx)} high-variance voxels")

    # Extract and preprocess per-run data using same voxel mask
    hw = MRIHardwareConfig(B0=3.0, TR=TR)
    raw_per_run = []     # raw voxel timeseries per run
    clean_per_run = []   # preprocessed per run

    for i, bold in enumerate(bold_runs):
        run_ts = bold[brain_mask][top_idx]  # (N_voxels, T_run)
        raw_per_run.append(run_ts.T)  # (T_run, N_voxels)

        cleaned = preprocess_bold(run_ts, TR, hw)  # (N_voxels, T_run)
        clean_per_run.append(cleaned.T)  # (T_run, N_voxels)

    # ── Train D-LinOSS (self-supervised, NO labels) ──
    print(f"\n  Stage 3: Training D-LinOSS (self-supervised, blind to categories)...")
    concat_clean = np.concatenate(clean_per_run, axis=0)  # (T_total, N)
    print(f"    Training data: {concat_clean.shape}")

    model = DLinOSSModel(
        input_dim=max_voxels,
        state_dim=32,
        hidden_dim=64,
        output_dim=max_voxels,
        num_blocks=3,
        drop_rate=0.0,
    )

    x_all = torch.tensor(concat_clean, dtype=torch.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        y = model(x_all)
        loss = torch.nn.functional.mse_loss(y, x_all)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}: loss = {loss.item():.6f}")

    # ── Extract model representations per run ──
    print(f"\n  Stage 4: Extracting model representations...")
    model.eval()
    model_per_run = []

    for i, clean_run in enumerate(clean_per_run):
        x_run = torch.tensor(clean_run, dtype=torch.float32)
        with torch.no_grad():
            y_run = model(x_run)
        model_per_run.append(y_run.numpy())  # (T_run, N)

    # ── DECODING TEST A: Raw voxel patterns ──
    print(f"\n  Stage 5A: Decoding from RAW voxel patterns...")
    raw_acc, raw_conf, raw_n = decode_with_representations(
        raw_per_run, labels_per_run, categories)

    print(f"    Raw voxel accuracy: {raw_acc:.1%} ({int(raw_acc*raw_n)}/{raw_n})")
    print(f"    Chance: {1/len(categories):.1%}")

    # ── DECODING TEST B: Preprocessed patterns ──
    print(f"\n  Stage 5B: Decoding from PREPROCESSED patterns...")
    clean_acc, clean_conf, clean_n = decode_with_representations(
        clean_per_run, labels_per_run, categories)

    print(f"    Preprocessed accuracy: {clean_acc:.1%} ({int(clean_acc*clean_n)}/{clean_n})")

    # ── DECODING TEST C: D-LinOSS representations ──
    print(f"\n  Stage 5C: Decoding from D-LinOSS representations...")
    model_acc, model_conf, model_n = decode_with_representations(
        model_per_run, labels_per_run, categories)

    print(f"    D-LinOSS accuracy: {model_acc:.1%} ({int(model_acc*model_n)}/{model_n})")

    # ── Confusion matrix for model decoding ──
    print(f"\n  D-LinOSS Confusion Matrix:")
    header = "               " + "".join(f"{c[:6]:>8s}" for c in categories)
    print(f"    Predicted: {header}")
    print(f"    True       {'':15s}" + "-" * (8 * len(categories)))
    for true_cat in categories:
        row = f"    {true_cat:15s}"
        total_for_cat = sum(model_conf.get(true_cat, {}).values())
        for pred_cat in categories:
            count = model_conf.get(true_cat, {}).get(pred_cat, 0)
            if count > 0 and true_cat == pred_cat:
                row += f"   [{count:2d}] "  # highlight diagonal
            else:
                row += f"    {count:2d}  "
        if total_for_cat > 0:
            cat_acc = model_conf.get(true_cat, {}).get(true_cat, 0) / total_for_cat
            row += f"  ({cat_acc:.0%})"
        print(row)

    # ── Per-category accuracy ──
    print(f"\n  Per-category accuracy (D-LinOSS model):")
    cat_accs = {}
    for cat in categories:
        total = sum(model_conf.get(cat, {}).values())
        correct = model_conf.get(cat, {}).get(cat, 0)
        acc = correct / total if total > 0 else 0
        cat_accs[cat] = acc
        bar = '#' * int(acc * 20)
        print(f"    {cat:15s}: {acc:5.1%} {bar}")

    # ── Summary ──
    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  DECODING RESULTS: {subject_name}")
    print(f"  {'='*60}")
    print(f"    Chance level:          {1/len(categories):5.1%}")
    print(f"    Raw voxel decoding:    {raw_acc:5.1%}  ({raw_acc/(1/len(categories)):.1f}x chance)")
    print(f"    Preprocessed decoding: {clean_acc:5.1%}  ({clean_acc/(1/len(categories)):.1f}x chance)")
    print(f"    D-LinOSS decoding:     {model_acc:5.1%}  ({model_acc/(1/len(categories)):.1f}x chance)")
    print(f"    Time: {total_time:.1f}s")

    improvement = (model_acc - raw_acc) / raw_acc * 100 if raw_acc > 0 else 0
    if model_acc > 1.5 / len(categories):
        print(f"\n    VERDICT: YES — D-LinOSS captures category-specific brain patterns")
        print(f"    The model can identify what the subject is seeing from brain activity alone.")
    else:
        print(f"\n    VERDICT: Performance near chance — needs more data or voxels.")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..',
                              'dlinoss_results', f'Decoding_{subject_name}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'subject': subject_name,
        'n_runs': n_runs,
        'n_categories': len(categories),
        'categories': categories,
        'chance_level': 1 / len(categories),
        'raw_accuracy': raw_acc,
        'preprocessed_accuracy': clean_acc,
        'model_accuracy': model_acc,
        'n_blocks_tested': model_n,
        'per_category_accuracy': cat_accs,
        'confusion_matrix': {k: dict(v) for k, v in model_conf.items()},
        'training_losses': losses,
        'final_loss': losses[-1] if losses else None,
        'time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'decoding_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')
    subjects_with_events = ['sub-4', 'sub-5', 'sub-6']

    print("=" * 70)
    print("  BRAIN DECODING CHALLENGE")
    print("  Can D-LinOSS read the mind from brain activation patterns?")
    print("=" * 70)
    print(f"\n  Task: 8-way visual category classification")
    print(f"  Method: Self-supervised D-LinOSS + nearest-template decoding")
    print(f"  Validation: Leave-one-run-out cross-validation")
    print(f"  Chance = 12.5%")

    all_results = {}
    for subj in subjects_with_events:
        subj_dir = os.path.join(base, subj)
        try:
            result = run_decoding_test(subj_dir, subj, max_runs=4, max_voxels=400)
            all_results[subj] = result
        except Exception as e:
            print(f"\n  ERROR: {subj}: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  GRAND SUMMARY: Cross-Subject Decoding")
        print(f"{'='*70}")
        print(f"\n  {'Subject':>10s} {'Chance':>8s} {'Raw':>8s} {'Preproc':>8s} {'D-LinOSS':>10s} {'vs Chance':>10s}")
        print(f"  {'-'*56}")

        raw_accs = []
        model_accs = []
        for subj, r in sorted(all_results.items()):
            ch = r['chance_level']
            raw_accs.append(r['raw_accuracy'])
            model_accs.append(r['model_accuracy'])
            print(f"  {subj:>10s} {ch:7.1%} {r['raw_accuracy']:7.1%} "
                  f"{r['preprocessed_accuracy']:7.1%} {r['model_accuracy']:9.1%} "
                  f"{r['model_accuracy']/ch:9.1f}x")

        mean_raw = np.mean(raw_accs)
        mean_model = np.mean(model_accs)
        print(f"  {'MEAN':>10s} {'12.5%':>8s} {mean_raw:7.1%} "
              f"{'':>8s} {mean_model:9.1%} {mean_model/0.125:9.1f}x")

        print(f"\n  Statistical significance:")
        from scipy.stats import binom_test
        for subj, r in sorted(all_results.items()):
            n = r['n_blocks_tested']
            k = int(r['model_accuracy'] * n)
            p_val = 1 - sum(
                np.math.comb(n, i) * (0.125 ** i) * (0.875 ** (n - i))
                for i in range(k)
            )
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"    {subj}: {k}/{n} correct, p = {p_val:.6f} {sig}")

    return all_results


if __name__ == '__main__':
    main()
