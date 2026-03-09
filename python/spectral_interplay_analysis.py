"""
spectral_interplay_analysis.py
───────────────────────────────────────────────────────────────
Analyzes the spatial interplay patterns in category blocks.

While fMRI BOLD is limited to ~0.2Hz Nyquist (TR=2.5s), different
neural frequency bands (1-350Hz) leave distinct SPATIAL signatures
in the BOLD pattern:
  - Gamma (30-100+Hz) → positive BOLD correlation (strongest driver)
  - Alpha/Beta (8-30Hz) → negative BOLD (desynchronization)
  - The "call and response" between brain regions appears as
    spatial covariance patterns between distant voxels

We analyze:
1. Spatial covariance structure per category (which regions co-activate)
2. How this structure EVOLVES within blocks (early vs late)
3. Whether face/house show DIFFERENT interplay patterns than man-made

Does NOT modify haxby_full_replication.py or any existing pipeline.
"""

import os, sys
import numpy as np
from scipy.stats import f_oneway
from scipy.signal import hilbert
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)


def analyze_spatial_interplay(subj_dir, subj_id):
    """
    For each category block, compute the spatial covariance matrix
    of the BOLD signal across the selected voxels.

    The covariance captures which brain regions "talk to each other"
    during each category — the call-and-response pattern.
    """
    bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    V, T_total = ts_2d.shape

    mu = ts_2d.mean(axis=1, keepdims=True)
    sd = ts_2d.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ts_z = np.clip((ts_2d - mu) / sd, -5, 5)

    all_labels = []
    run_bounds = [0]
    for events, bold in zip(events_runs, bold_runs):
        all_labels.extend(build_block_labels(events, bold.shape[-1], TR))
        run_bounds.append(run_bounds[-1] + bold.shape[-1])
    all_labels = all_labels[:T_total]

    # F-test voxel selection (same as replication)
    cat_pats = defaultdict(list)
    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        for cat in CATS:
            cm = [i for i, l in enumerate(rl) if l == cat]
            if len(cm) >= 2:
                cat_pats[cat].append(ts_z[:, s:e][:, cm].mean(axis=1))

    groups = [np.stack(v) for v in cat_pats.values()]
    f_stats = np.zeros(V)
    for v in range(V):
        try:
            f_stat, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
        except:
            pass
    top_idx = np.argsort(f_stats)[-200:]  # 200 for tractable covariance

    # Also grab coordinates for spatial analysis
    sel_coords = coords[top_idx]  # [200, 3]

    results = {}

    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        data = ts_z[top_idx, s:e]  # [200, T_run]

        for cat in CATS:
            cat_trs = [i for i, l in enumerate(rl) if l == cat]
            if len(cat_trs) < 4:
                continue

            block_data = data[:, cat_trs]  # [200, T_block]
            n_vox, T_block = block_data.shape

            # 1. Spatial covariance: which voxels co-vary within this block
            cov = np.corrcoef(block_data)  # [200, 200]

            # 2. Temporal variance (signal amplitude)
            temporal_var = np.var(block_data, axis=1)  # [200]

            # 3. EARLY vs LATE: split and compare covariance
            mid = T_block // 2
            early_data = block_data[:, :mid]
            late_data = block_data[:, mid:]

            early_cov = np.corrcoef(early_data) if mid >= 2 else cov
            late_cov = np.corrcoef(late_data) if (T_block - mid) >= 2 else cov

            if cat not in results:
                results[cat] = {
                    'cov_matrices': [], 'early_covs': [], 'late_covs': [],
                    'temporal_vars': [], 'coords': sel_coords,
                }

            results[cat]['cov_matrices'].append(cov)
            results[cat]['early_covs'].append(early_cov)
            results[cat]['late_covs'].append(late_cov)
            results[cat]['temporal_vars'].append(temporal_var)

    return results


def characterize_interplay(results):
    """
    From the covariance matrices, extract the interplay characteristics:
    - Network density (how many voxels are correlated)
    - Network strength (mean correlation among correlated pairs)
    - Spatial spread (distance between correlated voxels)
    - Early vs late change in network structure
    """
    metrics = {}

    for cat, data in results.items():
        avg_cov = np.stack(data['cov_matrices']).mean(axis=0)
        avg_early = np.stack(data['early_covs']).mean(axis=0)
        avg_late = np.stack(data['late_covs']).mean(axis=0)
        coords = data['coords']
        n = avg_cov.shape[0]

        # Upper triangle (exclude diagonal)
        triu = np.triu_indices(n, k=1)
        corrs = avg_cov[triu]
        early_corrs = avg_early[triu]
        late_corrs = avg_late[triu]

        # Network density: fraction of pairs with |r| > 0.3
        threshold = 0.3
        density = np.mean(np.abs(corrs) > threshold)
        early_density = np.mean(np.abs(early_corrs) > threshold)
        late_density = np.mean(np.abs(late_corrs) > threshold)

        # Network strength: mean |correlation| among significant pairs
        sig_mask = np.abs(corrs) > threshold
        strength = np.mean(np.abs(corrs[sig_mask])) if sig_mask.sum() > 0 else 0
        early_strength = np.mean(np.abs(early_corrs[sig_mask])) if sig_mask.sum() > 0 else 0
        late_strength = np.mean(np.abs(late_corrs[sig_mask])) if sig_mask.sum() > 0 else 0

        # Positive vs negative correlations (excitatory vs inhibitory interplay)
        pos_frac = np.mean(corrs > threshold) / max(density, 1e-8)
        neg_frac = np.mean(corrs < -threshold) / max(density, 1e-8)

        # Spatial spread: mean distance between correlated voxels
        dists = np.zeros_like(corrs)
        idx_i, idx_j = triu
        for k in range(len(idx_i)):
            dists[k] = np.linalg.norm(coords[idx_i[k]] - coords[idx_j[k]])

        if sig_mask.sum() > 0:
            spread = np.mean(dists[sig_mask])
        else:
            spread = 0

        # Temporal variance profile
        avg_var = np.stack(data['temporal_vars']).mean(axis=0)

        metrics[cat] = {
            'density': density,
            'early_density': early_density,
            'late_density': late_density,
            'strength': strength,
            'early_strength': early_strength,
            'late_strength': late_strength,
            'pos_frac': pos_frac,
            'neg_frac': neg_frac,
            'spread': spread,
            'mean_var': np.mean(avg_var),
            'density_growth': late_density - early_density,
            'strength_growth': late_strength - early_strength,
        }

    return metrics


def main():
    print("=" * 70)
    print("  SPATIAL INTERPLAY ANALYSIS")
    print("  How brain regions co-activate during each category")
    print("  (the 'call and response' pattern)")
    print("=" * 70)

    all_metrics = defaultdict(lambda: defaultdict(list))

    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            continue

        print(f"\n  {subj}...", end="", flush=True)
        results = analyze_spatial_interplay(path, subj)
        metrics = characterize_interplay(results)
        print(" done")

        for cat, m in metrics.items():
            for key, val in m.items():
                all_metrics[cat][key].append(val)

    # Cross-subject summary
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT NETWORK CHARACTERISTICS")
    print(f"{'='*70}")

    print(f"\n  {'Category':>14} {'Density':>8} {'Strength':>9} {'Pos%':>6} "
          f"{'Neg%':>6} {'Spread':>8} {'Amplitude':>10}")
    print(f"  {'─'*65}")

    for cat in CATS:
        if cat not in all_metrics:
            continue
        m = all_metrics[cat]
        print(f"  {cat:>14} "
              f"{np.mean(m['density']):>8.3f} "
              f"{np.mean(m['strength']):>9.3f} "
              f"{np.mean(m['pos_frac'])*100:>5.1f}% "
              f"{np.mean(m['neg_frac'])*100:>5.1f}% "
              f"{np.mean(m['spread']):>8.1f}mm "
              f"{np.mean(m['mean_var']):>10.4f}")

    # Network growth: early → late
    print(f"\n{'='*70}")
    print(f"  NETWORK GROWTH (Early → Late within blocks)")
    print(f"{'='*70}")

    print(f"\n  {'Category':>14} {'Early Den':>10} {'Late Den':>10} "
          f"{'ΔDensity':>10} {'ΔStrength':>10}  Interpretation")
    print(f"  {'─'*72}")

    for cat in CATS:
        if cat not in all_metrics:
            continue
        m = all_metrics[cat]
        ed = np.mean(m['early_density'])
        ld = np.mean(m['late_density'])
        dg = np.mean(m['density_growth'])
        sg = np.mean(m['strength_growth'])

        if dg > 0.02:
            interp = "NETWORK EXPANDS (more regions join)"
        elif dg < -0.02:
            interp = "NETWORK CONTRACTS (regions drop out)"
        else:
            interp = "STABLE network"

        print(f"  {cat:>14} {ed:>10.3f} {ld:>10.3f} "
              f"{dg:>+10.3f} {sg:>+10.3f}  {interp}")

    # Group comparison
    print(f"\n{'='*70}")
    print(f"  GROUP COMPARISON")
    print(f"{'='*70}")

    face_house = ['face', 'house']
    manmade = ['bottle', 'scissors', 'shoe']

    for group_name, group_cats in [("Face+House", face_house),
                                    ("Man-made", manmade),
                                    ("Scrambledpix", ['scrambledpix'])]:
        grp_density = []
        grp_strength = []
        grp_pos = []
        grp_spread = []
        grp_dgrowth = []
        for cat in group_cats:
            if cat in all_metrics:
                grp_density.extend(all_metrics[cat]['density'])
                grp_strength.extend(all_metrics[cat]['strength'])
                grp_pos.extend(all_metrics[cat]['pos_frac'])
                grp_spread.extend(all_metrics[cat]['spread'])
                grp_dgrowth.extend(all_metrics[cat]['density_growth'])

        print(f"\n  {group_name}:")
        print(f"    Network density:  {np.mean(grp_density):.3f} ± {np.std(grp_density):.3f}")
        print(f"    Network strength: {np.mean(grp_strength):.3f} ± {np.std(grp_strength):.3f}")
        print(f"    Excitatory:       {np.mean(grp_pos)*100:.1f}%")
        print(f"    Spatial spread:   {np.mean(grp_spread):.1f}mm ± {np.std(grp_spread):.1f}mm")
        print(f"    Density growth:   {np.mean(grp_dgrowth):+.3f} (early→late)")

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION")
    print(f"{'='*70}")
    print("""
  The "call and response" pattern between brain regions during category
  processing is captured by the spatial covariance structure.

  Key signatures to look for:
  1. FACE/HOUSE: Dense, strong, STABLE networks (dedicated circuitry)
     → FFA/PPA activate immediately, recruit consistent downstream regions
  2. MAN-MADE: Sparser initial networks that EXPAND over the block
     → Recognition builds through repetition, recruiting more regions
  3. SCRAMBLEDPIX: Dense visual cortex activation, minimal downstream
     → Low-level visual processing without semantic engagement

  These covariance patterns reflect the SPATIAL footprint of different
  frequency band interactions (1-350Hz) as projected through the BOLD
  hemodynamic response function. While we can't directly observe
  the high-frequency dynamics, the spatial covariance structure tells
  us WHICH regions are coupled — and that coupling reflects the
  underlying oscillatory interplay.
""")


if __name__ == "__main__":
    main()
