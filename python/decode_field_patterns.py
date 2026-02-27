#!/usr/bin/env python3
"""
Brain Field Decoding via D-LinOSS Oscillator States
=====================================================

THE CORRECT APPROACH:

The D-LinOSS model has learned the BOLD temporal dynamics as oscillator states.
The signal flow is:

    u(t) [voxels] --encoder--> h(t) --B--> x(t) [oscillator states] --C--> y(t) --decoder--> [voxels]

Where:
    B: (P, H) complex — maps encoded input INTO oscillator space
    x(t): (P,) complex — the oscillator state at time t
    C: (H, P) complex — maps oscillator states BACK to feature space

So each oscillator "listens to" specific voxels (via B) and "projects to"
specific voxels (via C). The oscillator states x(t) capture the
electromagnetic field dynamics at time t.

To identify what the subject is seeing:
    1. Train D-LinOSS self-supervised (no labels)
    2. Extract oscillator states x(t) for each timepoint
    3. From x(t), reconstruct which voxels are activated via C
    4. The spatial activation pattern identifies the category
    5. Compare: do same-category blocks produce similar oscillator patterns?
    6. Can we predict unseen blocks from learned oscillator signatures?

This uses the model's LEARNED internal representation, not just its output.
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
import torch.nn.functional as F
from dlinoss.dlinoss_torch import DLinOSSModel, DLinOSSLayer, parallel_scan
from dlinoss.mri_artifact_correction import MRIHardwareConfig, PhysiologicalNoiseRegressor
from run_visual_dlinoss import (
    load_haxby_subject, build_block_labels,
    extract_brain_voxels, preprocess_bold,
)


# ============================================================================
#  Extract Internal Oscillator States
# ============================================================================

def extract_oscillator_states(model, x_input):
    """
    Run the model and extract the oscillator states x(t) at each layer.

    Returns:
        layer_states: list of (L, P) complex arrays — one per D-LinOSS block
        encoded: (L, H) — encoder output
        decoded: (L, N) — final model output
    """
    model.eval()
    layer_states = []

    with torch.no_grad():
        # Encoder
        h = model.encoder(x_input)  # (L, H)
        encoded = h.numpy().copy()

        # Each block
        for block in model.blocks:
            # Block structure: norm → SSM → GLU → residual
            residual = h
            h_normed = block.norm(h)

            # --- SSM forward with state extraction ---
            ssm = block.ssm
            L, H = h_normed.shape
            P = ssm.state_dim

            B_complex = torch.view_as_complex(ssm.B.contiguous())
            C_complex = torch.view_as_complex(ssm.C.contiguous())
            A, G, dt = ssm._soft_project(ssm.A_diag, ssm.G_diag, ssm.dt_raw)

            Bu = h_normed.to(B_complex.dtype) @ B_complex.T  # (L, P)

            S = 1.0 + dt * G
            M11 = 1.0 / S
            M12 = -dt * A / S
            M21 = dt / S
            M22 = 1.0 - dt.pow(2) * A / S

            M = torch.cat([M11, M12, M21, M22])
            M_elements = M.unsqueeze(0).repeat(L, 1)

            F1 = (dt / S).unsqueeze(0) * Bu
            F2 = (dt.pow(2) / S).unsqueeze(0) * Bu
            F_cat = torch.cat([F1, F2], dim=-1)

            _, xs_re = parallel_scan(M_elements, F_cat.real)
            _, xs_im = parallel_scan(M_elements, F_cat.imag)

            # Full state: [velocity, position] for each oscillator
            # Position is what C projects from
            ys_re = xs_re[:, P:]
            ys_im = xs_im[:, P:]
            osc_states = torch.complex(ys_re, ys_im)  # (L, P) complex

            layer_states.append(osc_states.numpy())

            # Complete SSM forward for next block
            ssm_out = (osc_states @ C_complex.T).real + ssm.D.unsqueeze(0) * h_normed

            # GLU + residual
            h = block.glu(ssm_out) + residual

        # Decoder
        decoded = model.decoder(h).numpy()

    return layer_states, encoded, decoded


def get_projection_matrices(model):
    """
    Extract B and C matrices from each D-LinOSS block.

    B: how voxels map INTO oscillator space (what each oscillator listens to)
    C: how oscillator states map BACK to voxel space (what each oscillator drives)
    """
    projections = []
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            ssm = block.ssm
            B = torch.view_as_complex(ssm.B.contiguous()).numpy()  # (P, H)
            C = torch.view_as_complex(ssm.C.contiguous()).numpy()  # (H, P)
            D = ssm.D.numpy()  # (H,)
            projections.append({'B': B, 'C': C, 'D': D})
    return projections


# ============================================================================
#  Category Analysis via Oscillator States
# ============================================================================

def analyze_category_oscillator_patterns(layer_states, labels, categories, TR):
    """
    For each category, compute the mean oscillator activation pattern.

    The oscillator state x(t) = r(t) · exp(i·theta(t)) has:
    - magnitude |x(t)| = activation strength
    - phase arg(x(t)) = oscillation phase

    If categories produce distinct oscillator patterns, then:
    - Different categories should activate different oscillators
    - Same category should activate SAME oscillators consistently
    """
    results = {}

    for layer_idx, states in enumerate(layer_states):
        # states: (L, P) complex
        P = states.shape[1]
        magnitudes = np.abs(states)  # (L, P)

        # Mean magnitude per oscillator per category
        cat_profiles = {}
        for cat in categories:
            idx = [i for i, l in enumerate(labels) if l == cat]
            if idx:
                cat_profiles[cat] = magnitudes[idx].mean(axis=0)  # (P,)

        rest_idx = [i for i, l in enumerate(labels) if l == 'rest']
        if rest_idx:
            cat_profiles['rest'] = magnitudes[rest_idx].mean(axis=0)

        # Which oscillators are most category-selective?
        # Compute selectivity index per oscillator:
        # SI = max(cat_mag) - mean(other_cat_mag) / std
        selectivity = np.zeros(P)
        dominant_cat = [''] * P
        for p in range(P):
            mags = {c: cat_profiles[c][p] for c in categories if c in cat_profiles}
            if len(mags) < 2:
                continue
            vals = list(mags.values())
            best_cat = max(mags, key=mags.get)
            best_val = mags[best_cat]
            others = [v for c, v in mags.items() if c != best_cat]
            selectivity[p] = (best_val - np.mean(others)) / (np.std(vals) + 1e-10)
            dominant_cat[p] = best_cat

        results[f'layer_{layer_idx}'] = {
            'cat_profiles': cat_profiles,
            'selectivity': selectivity,
            'dominant_cat': dominant_cat,
        }

    return results


def decode_from_oscillator_states(layer_states, labels_per_run, categories,
                                  run_boundaries):
    """
    Decode category from oscillator states using leave-one-run-out CV.

    For each held-out run:
    1. Build category templates from training runs (mean oscillator magnitude)
    2. For each test block, compute correlation to each template
    3. Predict = highest correlation template

    This decodes directly from the learned field dynamics.
    """
    n_layers = len(layer_states)
    n_runs = len(labels_per_run)

    # Results per layer
    layer_results = {}

    for li in range(n_layers):
        states = layer_states[li]  # (L_total, P) complex
        magnitudes = np.abs(states)

        correct = 0
        total = 0
        confusion = defaultdict(lambda: defaultdict(int))

        for test_run in range(n_runs):
            # Split data by run boundaries
            test_start = run_boundaries[test_run]
            test_end = run_boundaries[test_run + 1]
            test_mags = magnitudes[test_start:test_end]
            test_labels = labels_per_run[test_run]

            # Training: everything except test run
            train_mags = np.concatenate([
                magnitudes[run_boundaries[r]:run_boundaries[r+1]]
                for r in range(n_runs) if r != test_run
            ], axis=0)
            train_labels = [l for r in range(n_runs) if r != test_run
                           for l in labels_per_run[r]]

            # Build templates
            templates = {}
            for cat in categories:
                idx = [i for i, l in enumerate(train_labels) if l == cat]
                if idx:
                    templates[cat] = train_mags[idx].mean(axis=0)

            # Find blocks in test run
            blocks = []
            current = None
            start = None
            for i, l in enumerate(test_labels):
                if l != current:
                    if current in categories:
                        blocks.append((current, start, i))
                    current = l
                    start = i
            if current in categories:
                blocks.append((current, start, len(test_labels)))

            # Classify each block
            for true_cat, s, e in blocks:
                if true_cat not in templates or e <= s:
                    continue
                block_profile = test_mags[s:e].mean(axis=0)

                # Nearest template by correlation
                best_cat = None
                best_r = -2.0
                for cat, tmpl in templates.items():
                    r = np.corrcoef(block_profile, tmpl)[0, 1]
                    if np.isfinite(r) and r > best_r:
                        best_r = r
                        best_cat = cat

                confusion[true_cat][best_cat] += 1
                total += 1
                if best_cat == true_cat:
                    correct += 1

        accuracy = correct / total if total > 0 else 0
        layer_results[f'layer_{li}'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion': {k: dict(v) for k, v in confusion.items()},
        }

    return layer_results


def map_oscillators_to_voxels(projections, model, osc_results, categories, coords):
    """
    Map the most category-selective oscillators back to voxel space.

    For each category:
    1. Find oscillators with highest activation for that category
    2. Look at the C matrix column for that oscillator
    3. The C column tells us which voxels that oscillator drives
    4. Map back to 3D coordinates

    This answers: WHERE in the brain is the activation coming from?
    """
    results = {}

    for layer_key, layer_data in osc_results.items():
        layer_idx = int(layer_key.split('_')[1])
        C = projections[layer_idx]['C']  # (H, P) complex
        C_mag = np.abs(C)  # magnitude of projection weights

        cat_profiles = layer_data['cat_profiles']
        selectivity = layer_data['selectivity']

        cat_voxel_maps = {}
        for cat in categories:
            if cat not in cat_profiles:
                continue
            profile = cat_profiles[cat]

            # Top 5 oscillators for this category
            top_osc = np.argsort(profile)[-5:]

            # For each top oscillator, find which voxels it projects to strongest
            voxel_weights = np.zeros(C_mag.shape[0])  # (H,)
            for osc_idx in top_osc:
                voxel_weights += C_mag[:, osc_idx] * profile[osc_idx]

            cat_voxel_maps[cat] = {
                'top_oscillators': top_osc.tolist(),
                'oscillator_magnitudes': profile[top_osc].tolist(),
                'voxel_weight_range': [float(voxel_weights.min()),
                                       float(voxel_weights.max())],
                'top_voxel_indices': np.argsort(voxel_weights)[-10:].tolist(),
            }

        results[layer_key] = cat_voxel_maps

    return results


# ============================================================================
#  Main Pipeline
# ============================================================================

def run_field_decoding(subject_dir, subject_name, max_runs=12, max_voxels=500):
    """Complete field-based decoding pipeline."""
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  BRAIN FIELD DECODING: {subject_name}")
    print(f"  Method: D-LinOSS oscillator state analysis")
    print(f"{'='*70}")

    # ── Load ──
    print(f"\n  Loading data...")
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    n_runs = len(bold_runs)
    print(f"    {n_runs} runs, TR={TR}s")

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

    # ── Voxel selection (ANOVA F-stat for category-discriminative voxels) ──
    print(f"\n  Selecting category-discriminative voxels...")
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, brain_mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    all_labels_flat = [l for rl in labels_per_run for l in rl]

    N_brain = ts_2d.shape[0]

    # ANOVA F-stat selection
    cat_means = {}
    cat_counts = {}
    for cat in categories:
        idx = [i for i, l in enumerate(all_labels_flat) if l == cat]
        if idx:
            cat_means[cat] = ts_2d[:, idx].mean(axis=1)
            cat_counts[cat] = len(idx)

    grand_mean = ts_2d.mean(axis=1)
    ss_between = np.zeros(N_brain)
    for cat in categories:
        if cat in cat_means:
            ss_between += cat_counts[cat] * (cat_means[cat] - grand_mean) ** 2

    ss_within = np.zeros(N_brain)
    for cat in categories:
        idx = [i for i, l in enumerate(all_labels_flat) if l == cat]
        if idx:
            ss_within += np.sum((ts_2d[:, idx] - cat_means[cat][:, None]) ** 2, axis=1)

    k = len(cat_means)
    n_total = sum(cat_counts.values())
    f_stat = (ss_between / max(k-1, 1)) / (ss_within / max(n_total-k, 1) + 1e-10)

    top_idx = np.argsort(f_stat)[-max_voxels:]
    selected_coords = coords[top_idx]
    print(f"    Brain voxels: {N_brain:,}")
    print(f"    Selected: {len(top_idx)} (F range: [{f_stat[top_idx].min():.4f}, "
          f"{f_stat[top_idx].max():.4f}])")

    # ── Preprocess per-run ──
    print(f"\n  Preprocessing...")
    hw = MRIHardwareConfig(B0=3.0, TR=TR)
    clean_runs = []
    run_boundaries = [0]

    for bold in bold_runs:
        run_selected = bold[brain_mask][top_idx]  # (N_vox, T_run)
        # Lightweight preprocess: detrend + normalize
        N_v, T_r = run_selected.shape
        cleaned = run_selected.copy().astype(np.float64)
        t_axis = np.arange(T_r, dtype=np.float64)
        for i in range(N_v):
            coeffs = np.polyfit(t_axis, cleaned[i], 1)
            cleaned[i] -= np.polyval(coeffs, t_axis)
        mu = cleaned.mean(axis=1, keepdims=True)
        sigma = cleaned.std(axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        cleaned = (cleaned - mu) / sigma
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned = np.clip(cleaned, -5.0, 5.0)
        clean_runs.append(cleaned.T)  # (T_run, N_vox)
        run_boundaries.append(run_boundaries[-1] + T_r)

    concat_clean = np.concatenate(clean_runs, axis=0)
    T_total = concat_clean.shape[0]
    print(f"    Total: ({T_total}, {max_voxels})")

    # ── Train D-LinOSS (self-supervised, BLIND to categories) ──
    print(f"\n  Training D-LinOSS (self-supervised, no labels)...")
    model = DLinOSSModel(
        input_dim=max_voxels, state_dim=32, hidden_dim=64,
        output_dim=max_voxels, num_blocks=3, drop_rate=0.0,
    )

    x_all = torch.tensor(concat_clean, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        optimizer.zero_grad()
        y = model(x_all)
        loss = torch.nn.functional.mse_loss(y, x_all)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}: loss = {loss.item():.6f}")

    # ── Extract oscillator states ──
    print(f"\n  Extracting oscillator states...")
    layer_states, encoded, decoded = extract_oscillator_states(model, x_all)

    for i, states in enumerate(layer_states):
        print(f"    Layer {i}: states shape {states.shape}, "
              f"mean |x| = {np.abs(states).mean():.4f}")

    # ── Analyze category-specific oscillator patterns ──
    print(f"\n  Analyzing category-specific field patterns...")
    all_labels = [l for rl in labels_per_run for l in rl]
    osc_results = analyze_category_oscillator_patterns(
        layer_states, all_labels, categories, TR)

    for layer_key, data in osc_results.items():
        selectivity = data['selectivity']
        dominant = data['dominant_cat']
        print(f"\n    {layer_key}:")
        print(f"      Mean selectivity: {selectivity.mean():.3f}")
        print(f"      Max selectivity:  {selectivity.max():.3f}")

        # Show top 5 most selective oscillators
        top5 = np.argsort(selectivity)[-5:][::-1]
        for osc in top5:
            cat = dominant[osc]
            profile = data['cat_profiles']
            osc_mags = {c: profile[c][osc] for c in categories if c in profile}
            best = max(osc_mags.values())
            ratio = best / (np.mean(list(osc_mags.values())) + 1e-10)
            print(f"        Osc {osc:2d}: dominant={cat:15s}  "
                  f"selectivity={selectivity[osc]:.2f}  "
                  f"|x|={best:.3f}  ratio={ratio:.2f}x")

    # ── Decode from oscillator states ──
    print(f"\n  Decoding categories from oscillator field patterns...")
    decode_results = decode_from_oscillator_states(
        layer_states, labels_per_run, categories, run_boundaries)

    for layer_key, res in decode_results.items():
        acc = res['accuracy']
        mult = acc / (1/n_cats)
        sig = "***" if mult > 2.0 else "**" if mult > 1.5 else "*" if mult > 1.0 else ""
        print(f"    {layer_key}: accuracy = {acc:.1%} ({mult:.1f}x chance) {sig}")

    best_layer = max(decode_results, key=lambda k: decode_results[k]['accuracy'])
    best_acc = decode_results[best_layer]['accuracy']

    # Show confusion matrix for best layer
    best_conf = decode_results[best_layer]['confusion']
    print(f"\n  Best layer: {best_layer}")
    print(f"  Confusion matrix:")
    header = "  Predicted:     " + "".join(f"{c[:6]:>8s}" for c in categories)
    print(header)
    print(f"  True           " + "-" * (8 * n_cats))
    for tc in categories:
        row = f"  {tc:15s}"
        t_total_cat = sum(best_conf.get(tc, {}).values())
        for pc in categories:
            c = best_conf.get(tc, {}).get(pc, 0)
            if tc == pc and c > 0:
                row += f"  [{c:2d}] "
            else:
                row += f"   {c:2d}  "
        cat_acc = best_conf.get(tc, {}).get(tc, 0) / t_total_cat if t_total_cat > 0 else 0
        row += f" ({cat_acc:4.0%})"
        print(row)

    # ── Map oscillators back to voxel space ──
    print(f"\n  Mapping oscillators back to voxel coordinates...")
    projections = get_projection_matrices(model)
    voxel_maps = map_oscillators_to_voxels(
        projections, model, osc_results, categories, selected_coords)

    for layer_key, cat_maps in voxel_maps.items():
        print(f"\n    {layer_key}:")
        for cat, info in cat_maps.items():
            top_osc = info['top_oscillators']
            top_vox = info['top_voxel_indices']
            # Map voxel indices back through decoder to get spatial location
            print(f"      {cat:15s}: top oscillators {top_osc}, "
                  f"top voxel features {top_vox}")

    # ── Summary ──
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  RESULTS: {subject_name}")
    print(f"  {'='*60}")
    print(f"    Chance:  {1/n_cats:5.1%}")
    print(f"    Best:    {best_acc:5.1%} ({best_acc/(1/n_cats):.1f}x chance) [{best_layer}]")
    print(f"    Time:    {total_time:.0f}s")

    if best_acc > 1.5 / n_cats:
        print(f"\n    The model's learned oscillator states contain")
        print(f"    category-specific activation patterns that can identify")
        print(f"    what the subject was viewing from brain field dynamics alone.")
    else:
        print(f"\n    Oscillator states show weak category separation.")
        print(f"    May need more voxels, blocks, or training epochs.")

    # ── Binomial significance test ──
    n_tested = decode_results[best_layer]['total']
    k_correct = decode_results[best_layer]['correct']
    p_chance = 1 / n_cats
    from scipy.stats import binomtest
    btest = binomtest(k_correct, n_tested, p_chance, alternative='greater')
    print(f"    Binomial test: {k_correct}/{n_tested}, p = {btest.pvalue:.6f}")
    if btest.pvalue < 0.05:
        print(f"    *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..',
                              'dlinoss_results', f'FieldDecode_{subject_name}')
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'subject': subject_name,
        'n_runs': n_runs,
        'n_categories': n_cats,
        'categories': categories,
        'chance_level': 1/n_cats,
        'n_voxels': max_voxels,
        'training_loss': loss.item(),
        'decoding_results': {k: {'accuracy': v['accuracy'], 'total': v['total']}
                            for k, v in decode_results.items()},
        'best_layer': best_layer,
        'best_accuracy': best_acc,
        'p_value': btest.pvalue,
        'time': total_time,
    }

    with open(os.path.join(output_dir, 'field_decoding.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved to: {output_dir}")

    return summary


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro', 'ds000105')
    subjects = ['sub-4', 'sub-5', 'sub-6']

    print("=" * 70)
    print("  BRAIN FIELD DECODING via D-LinOSS Oscillator States")
    print("  Can learned electromagnetic dynamics identify visual stimuli?")
    print("=" * 70)

    all_results = {}
    for subj in subjects:
        subj_dir = os.path.join(base, subj)
        try:
            result = run_field_decoding(subj_dir, subj,
                                        max_runs=12, max_voxels=500)
            all_results[subj] = result
        except Exception as e:
            print(f"\n  ERROR: {subj}: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  GRAND SUMMARY: Field-Based Brain Decoding")
        print(f"{'='*70}")
        print(f"\n  {'Subject':>10s}  {'Chance':>7s}  {'Best Layer':>12s}  "
              f"{'Accuracy':>9s}  {'x Chance':>9s}  {'p-value':>10s}")
        print(f"  {'-'*60}")
        for subj, r in sorted(all_results.items()):
            ch = r['chance_level']
            print(f"  {subj:>10s}  {ch:6.1%}  {r['best_layer']:>12s}  "
                  f"{r['best_accuracy']:8.1%}  {r['best_accuracy']/ch:8.1f}x  "
                  f"{r['p_value']:9.6f}")

        mean_acc = np.mean([r['best_accuracy'] for r in all_results.values()])
        print(f"\n  Mean accuracy: {mean_acc:.1%} ({mean_acc/0.125:.1f}x chance)")


if __name__ == '__main__':
    main()
