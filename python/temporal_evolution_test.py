"""
temporal_evolution_test.py
───────────────────────────────────────────────────────────────
Tests whether category recognition signals GROW within blocks.

Hypothesis: Face/house activate specialized networks (FFA/PPA) immediately,
while bottle/scissors/shoe recognition emerges gradually over repeated
exposure within a block.

Does NOT modify haxby_full_replication.py or any existing pipeline.
"""

import os, sys
import numpy as np
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)


def analyze_temporal_evolution(subj_dir, subj_id):
    """
    Split each block into early (first half of TRs) and late (second half).
    Compare within-category correlation strength for each half.
    """
    bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    V, T_total = ts_2d.shape

    # Z-score
    mu = ts_2d.mean(axis=1, keepdims=True)
    sd = ts_2d.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ts_z = np.clip((ts_2d - mu) / sd, -5, 5)

    # Labels
    all_labels = []
    run_bounds = [0]
    for events, bold in zip(events_runs, bold_runs):
        all_labels.extend(build_block_labels(events, bold.shape[-1], TR))
        run_bounds.append(run_bounds[-1] + bold.shape[-1])
    all_labels = all_labels[:T_total]

    # F-test voxel selection (same as replication)
    cat_pats_full = defaultdict(list)
    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        for cat in CATS:
            cm = [i for i, l in enumerate(rl) if l == cat]
            if len(cm) >= 2:
                cat_pats_full[cat].append(ts_z[:, s:e][:, cm].mean(axis=1))

    groups = [np.stack(v) for v in cat_pats_full.values()]
    f_stats = np.zeros(V)
    for v in range(V):
        try:
            f_stat, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
        except:
            pass
    top500 = np.argsort(f_stats)[-500:]

    # Extract per-block EARLY and LATE patterns
    early_blocks = []  # First half of each block's TRs
    late_blocks = []   # Second half of each block's TRs
    full_blocks = []

    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        data = ts_z[top500, s:e]  # [500, T_run]

        for cat in CATS:
            cat_trs = [i for i, l in enumerate(rl) if l == cat]
            if len(cat_trs) < 4:
                continue

            mid = len(cat_trs) // 2
            early_trs = cat_trs[:mid]
            late_trs = cat_trs[mid:]

            early_pat = data[:, early_trs].mean(axis=1)
            late_pat = data[:, late_trs].mean(axis=1)
            full_pat = data[:, cat_trs].mean(axis=1)

            early_blocks.append({'run': ri, 'cat': cat, 'pattern': early_pat})
            late_blocks.append({'run': ri, 'cat': cat, 'pattern': late_pat})
            full_blocks.append({'run': ri, 'cat': cat, 'pattern': full_pat})

    def haxby_analysis(blocks, label):
        even = [b for b in blocks if b['run'] % 2 == 0]
        odd  = [b for b in blocks if b['run'] % 2 == 1]

        def avg(entries, cat):
            ps = [e['pattern'] for e in entries if e['cat'] == cat]
            return np.stack(ps).mean(axis=0) if ps else None

        cats = sorted(set(b['cat'] for b in blocks))
        results = {}
        for cat in cats:
            pe = avg(even, cat)
            po = avg(odd, cat)
            if pe is None or po is None:
                continue
            within_r, _ = pearsonr(pe, po)
            between_rs = []
            for other in cats:
                if other == cat:
                    continue
                po2 = avg(odd, other)
                if po2 is not None:
                    r, _ = pearsonr(pe, po2)
                    between_rs.append(r)
            between_r = np.mean(between_rs) if between_rs else 0
            results[cat] = {
                'within': within_r,
                'between': between_r,
                'delta': within_r - between_r,
                'win': within_r > between_r,
            }
        return results

    early_results = haxby_analysis(early_blocks, "Early")
    late_results = haxby_analysis(late_blocks, "Late")
    full_results = haxby_analysis(full_blocks, "Full")

    return early_results, late_results, full_results


def main():
    print("=" * 70)
    print("  TEMPORAL EVOLUTION OF CATEGORY RECOGNITION")
    print("  Early (first half of block) vs Late (second half)")
    print("=" * 70)

    # Collect per-category early/late deltas across subjects
    all_early = defaultdict(list)
    all_late = defaultdict(list)
    all_full = defaultdict(list)
    all_growth = defaultdict(list)  # late_delta - early_delta

    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            continue

        print(f"\n{'─'*60}")
        print(f"  {subj}")
        print(f"{'─'*60}")

        early, late, full = analyze_temporal_evolution(path, subj)

        print(f"\n  {'Category':>14}  {'Early Δ':>9}  {'Late Δ':>9}  {'Full Δ':>9}  {'Growth':>9}  Signal Type")
        print(f"  {'─'*75}")

        for cat in CATS:
            if cat not in early or cat not in late:
                continue
            e_d = early[cat]['delta']
            l_d = late[cat]['delta']
            f_d = full[cat]['delta']
            growth = l_d - e_d

            all_early[cat].append(e_d)
            all_late[cat].append(l_d)
            all_full[cat].append(f_d)
            all_growth[cat].append(growth)

            # Classify signal type
            if e_d > 0.2 and l_d > 0.2:
                sig_type = "SUSTAINED (FFA/PPA-like)"
            elif e_d < 0.1 and l_d > 0.2:
                sig_type = "EMERGING (recognition builds)"
            elif growth > 0.1:
                sig_type = "GROWING"
            elif growth < -0.1:
                sig_type = "FADING"
            else:
                sig_type = "FLAT"

            print(f"  {cat:>14}  {e_d:>9.4f}  {l_d:>9.4f}  {f_d:>9.4f}  {growth:>+9.4f}  {sig_type}")

    # Summary across subjects
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT SUMMARY (mean ± std)")
    print(f"{'='*70}")

    print(f"\n  {'Category':>14}  {'Early Δ':>12}  {'Late Δ':>12}  {'Growth':>12}  {'Type':>20}")
    print(f"  {'─'*72}")

    cat_types = {}
    for cat in CATS:
        if cat not in all_early:
            continue
        me = np.mean(all_early[cat])
        ml = np.mean(all_late[cat])
        mg = np.mean(all_growth[cat])
        se = np.std(all_early[cat])
        sl = np.std(all_late[cat])
        sg = np.std(all_growth[cat])

        if me > 0.15 and ml > 0.15:
            ctype = "SUSTAINED"
        elif mg > 0.05:
            ctype = "EMERGING"
        elif mg < -0.05:
            ctype = "FADING"
        else:
            ctype = "STABLE"

        cat_types[cat] = ctype

        print(f"  {cat:>14}  {me:>5.3f}±{se:.3f}  {ml:>5.3f}±{sl:.3f}  "
              f"{mg:>+5.3f}±{sg:.3f}  {ctype:>20}")

    # Test the hypothesis
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS TEST")
    print(f"{'='*70}")

    face_house = ['face', 'house']
    manmade = ['bottle', 'scissors', 'shoe']

    fh_early = np.mean([np.mean(all_early[c]) for c in face_house if c in all_early])
    fh_late = np.mean([np.mean(all_late[c]) for c in face_house if c in all_late])
    fh_growth = fh_late - fh_early

    mm_early = np.mean([np.mean(all_early[c]) for c in manmade if c in all_early])
    mm_late = np.mean([np.mean(all_late[c]) for c in manmade if c in all_late])
    mm_growth = mm_late - mm_early

    print(f"\n  H0: Recognition signal is constant for all categories")
    print(f"  H1: Face/house are SUSTAINED; bottle/scissors/shoe EMERGE over block")
    print(f"\n  Face+House group:")
    print(f"    Early Δ: {fh_early:.4f}")
    print(f"    Late Δ:  {fh_late:.4f}")
    print(f"    Growth:  {fh_growth:+.4f}")
    print(f"\n  Man-made group (bottle, scissors, shoe):")
    print(f"    Early Δ: {mm_early:.4f}")
    print(f"    Late Δ:  {mm_late:.4f}")
    print(f"    Growth:  {mm_growth:+.4f}")
    print(f"\n  Difference in growth:")
    print(f"    Man-made growth - Face/house growth = {mm_growth - fh_growth:+.4f}")

    if mm_growth > fh_growth + 0.02:
        print(f"\n  ✓ SUPPORTS H1: Man-made categories grow more than face/house")
    elif abs(mm_growth - fh_growth) < 0.02:
        print(f"\n  ~ INCONCLUSIVE: Similar growth rates (Δ < 0.02)")
    else:
        print(f"\n  ✗ CONTRADICTS H1: Face/house grow more than man-made")

    print(f"\n  NOTE: The 1-35Hz oscillatory band captured by BOLD contrast")
    print(f"        spans delta(1-4Hz), theta(4-8Hz), alpha(8-12Hz),")
    print(f"        beta(12-30Hz), and low gamma(30-35Hz).")
    print(f"        Face/house may activate broader-band responses (sustained)")
    print(f"        while man-made objects may show narrower-band signatures")
    print(f"        that build up through repetition priming.")


if __name__ == "__main__":
    main()
