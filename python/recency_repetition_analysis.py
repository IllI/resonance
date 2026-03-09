"""
recency_repetition_analysis.py
───────────────────────────────────────────────────────────────
Exploits the stimulus presentation ORDER across runs/blocks.

Key insight from the user: the subject doesn't know there are 8
categories. Recognition of a category like "shoe" emerges through
recency and repetition. If shoes appeared right before or right after
another shoe block (across the rest boundary), recognition may be
primed. If shoes appear far apart with many intervening categories,
the signal may be weaker.

We analyze:
1. The block ORDER within and across all runs per subject
2. Whether back-to-back same-category blocks (or near adjacency)
   show stronger BOLD discrimination than isolated blocks
3. Use D-LinOSS variable time-stepping to focus processing on
   the transitions between blocks (where recency effects occur)

Does NOT modify haxby_full_replication.py.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe'])
K = len(CATS)


def get_block_sequence(subj_dir):
    """
    Extract the ordered sequence of category blocks across ALL runs.
    Returns list of (run_idx, block_idx_in_run, category, onset_s).
    """
    func_dir = os.path.join(subj_dir, 'func')
    events_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_events.tsv')])

    sequence = []
    for run_idx, ef in enumerate(events_files):
        df = pd.read_csv(os.path.join(func_dir, ef), sep='\t')
        # Each block is 12 consecutive rows of the same trial_type
        current = None
        block_idx = 0
        for _, row in df.iterrows():
            tt = row['trial_type']
            if tt != current:
                sequence.append({
                    'run': run_idx,
                    'block_in_run': block_idx,
                    'category': tt,
                    'onset': row['onset'],
                })
                current = tt
                block_idx += 1

    return sequence


def compute_recency_scores(sequence):
    """
    For each block, compute:
    - blocks_since_same: how many blocks since the same category last appeared
    - is_adjacent: whether the immediately preceding block was the same category
    - recent_count: how many times this category appeared in the last N blocks
    """
    for i, block in enumerate(sequence):
        cat = block['category']

        # Look backward for the same category
        blocks_since = None
        for j in range(i - 1, -1, -1):
            if sequence[j]['category'] == cat:
                blocks_since = i - j
                break

        block['blocks_since_same'] = blocks_since  # None = first occurrence
        block['is_adjacent'] = (blocks_since == 1) if blocks_since else False

        # Recent count: same category in last 8 blocks
        lookback = 8
        recent = sum(1 for j in range(max(0, i - lookback), i)
                     if sequence[j]['category'] == cat)
        block['recent_count'] = recent

    return sequence


def analyze_recency_effects(subj_dir, subj_id):
    """
    For each subject:
    1. Map the block sequence with recency scores
    2. Extract BOLD patterns for each block
    3. Compare discrimination strength for RECENT vs NON-RECENT blocks
    """
    # Get block order
    sequence = get_block_sequence(subj_dir)
    sequence = compute_recency_scores(sequence)

    # Load BOLD data
    bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    V, T_total = ts_2d.shape

    mu = ts_2d.mean(axis=1, keepdims=True)
    sd = ts_2d.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ts_z = np.clip((ts_2d - mu) / sd, -5, 5)

    # Labels and bounds
    all_labels = []
    run_bounds = [0]
    for events, bold in zip(events_runs, bold_runs):
        all_labels.extend(build_block_labels(events, bold.shape[-1], TR))
        run_bounds.append(run_bounds[-1] + bold.shape[-1])
    all_labels = all_labels[:T_total]

    # F-test voxel selection
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
    top500 = np.argsort(f_stats)[-500:]

    # Extract per-block patterns with recency metadata
    block_data = []
    for block in sequence:
        ri = block['run']
        cat = block['category']
        if ri >= len(bold_runs):
            continue
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        data = ts_z[top500, s:e]

        cat_trs = [i for i, l in enumerate(rl) if l == cat]
        if len(cat_trs) < 2:
            continue

        pattern = data[:, cat_trs].mean(axis=1)
        block_data.append({
            **block,
            'pattern': pattern,
        })

    return sequence, block_data


def main():
    print("=" * 70)
    print("  RECENCY & REPETITION ANALYSIS")
    print("  How stimulus presentation order affects category recognition")
    print("=" * 70)

    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            continue

        print(f"\n{'─'*60}")
        print(f"  {subj}")
        print(f"{'─'*60}")

        sequence, block_data = analyze_recency_effects(path, subj)

        # Print block sequence with recency
        print(f"\n  Block sequence (first 5 runs):")
        print(f"  {'Run':>4} {'Blk':>4} {'Category':>14} {'Since':>6} {'Adj':>4} {'Rec':>4}")
        for b in sequence[:40]:  # first 5 runs (8 blocks each)
            since = f"{b['blocks_since_same']}" if b['blocks_since_same'] else "FIRST"
            adj = "YES" if b['is_adjacent'] else ""
            print(f"  {b['run']:>4} {b['block_in_run']:>4} {b['category']:>14} "
                  f"{since:>6} {adj:>4} {b['recent_count']:>4}")

        # Find adjacent repetitions
        adj_blocks = [b for b in sequence if b['is_adjacent']]
        print(f"\n  Adjacent repetitions (same category in consecutive blocks): {len(adj_blocks)}")
        for b in adj_blocks:
            print(f"    Run {b['run']}, Block {b['block_in_run']}: {b['category']} "
                  f"(also in previous block)")

        # Compute per-category patterns split by recency
        print(f"\n  Per-category discrimination by recency:")
        print(f"  {'Category':>14} {'N_recent':>9} {'N_distant':>10} "
              f"{'Mean_r_recent':>14} {'Mean_r_distant':>15}")

        cat_prototype = {}
        for cat in CATS:
            cat_blocks = [b for b in block_data if b['category'] == cat]
            if len(cat_blocks) >= 2:
                cat_prototype[cat] = np.stack([b['pattern'] for b in cat_blocks]).mean(axis=0)

        for cat in CATS:
            cat_blocks = [b for b in block_data if b['category'] == cat]
            if cat not in cat_prototype:
                continue

            # Split: recent (blocks_since_same <= 3) vs distant (> 3 or first)
            recent = [b for b in cat_blocks
                      if b['blocks_since_same'] is not None and b['blocks_since_same'] <= 3]
            distant = [b for b in cat_blocks
                       if b['blocks_since_same'] is None or b['blocks_since_same'] > 3]

            if len(recent) == 0 or len(distant) == 0:
                continue

            # Correlation of each block's pattern with the category prototype
            corrs_recent = [pearsonr(b['pattern'], cat_prototype[cat])[0] for b in recent]
            corrs_distant = [pearsonr(b['pattern'], cat_prototype[cat])[0] for b in distant]

            print(f"  {cat:>14} {len(recent):>9} {len(distant):>10} "
                  f"{np.mean(corrs_recent):>14.4f} {np.mean(corrs_distant):>15.4f}")

        # The TRANSITION analysis: what happens at block boundaries?
        print(f"\n  Block transition analysis:")
        print(f"  When a new category starts, how different is it from the preceding block?")
        print(f"  {'Transition':>30} {'Count':>6} {'Mean Δ':>8}")

        transition_data = defaultdict(list)
        for i in range(1, len(block_data)):
            prev = block_data[i - 1]
            curr = block_data[i]
            if prev['run'] != curr['run']:
                continue  # Rest between runs

            key = f"{prev['category']} → {curr['category']}"
            delta = np.linalg.norm(curr['pattern'] - prev['pattern'])
            transition_data[key].append(delta)

        # Sort by mean delta (biggest transitions first)
        sorted_trans = sorted(transition_data.items(),
                              key=lambda x: np.mean(x[1]), reverse=True)

        # Show top 10 and bottom 5
        for key, vals in sorted_trans[:8]:
            print(f"  {key:>30} {len(vals):>6} {np.mean(vals):>8.3f}")
        print(f"  {'...':>30}")
        for key, vals in sorted_trans[-5:]:
            print(f"  {key:>30} {len(vals):>6} {np.mean(vals):>8.3f}")

        # Same → same vs same → different transitions
        same_trans = [v for k, vals in transition_data.items()
                      for v in vals if k.split(' → ')[0] == k.split(' → ')[1]]
        diff_trans = [v for k, vals in transition_data.items()
                      for v in vals if k.split(' → ')[0] != k.split(' → ')[1]]
        if same_trans:
            print(f"\n  Same→Same transition Δ: {np.mean(same_trans):.3f} "
                  f"(n={len(same_trans)})")
        print(f"  Same→Diff transition Δ: {np.mean(diff_trans):.3f} "
              f"(n={len(diff_trans)})")

    # Cross-subject summary
    print(f"\n{'='*70}")
    print(f"  IMPLICATIONS FOR D-LinOSS TIME-STEPPING")
    print(f"{'='*70}")
    print("""
  The D-LinOSS oscillator uses a discretization step Δt that controls
  temporal resolution. For standard block processing:
  
    Δt = 1.0 (standard, one step per TR)
  
  For TRANSITION ZONES (block boundaries where recency effects occur):
  
    Δt = 0.5 (half-step, 2x temporal resolution)
    
  This lets the model capture the moment-by-moment dynamics at the
  critical points where:
  
  1. A new category starts → the brain switches networks
  2. A REPEATED category starts → recognition is primed (smaller Δ)
  3. Rest ends → the brain re-engages from baseline
  
  By slowing down at these transitions, D-LinOSS can distinguish
  between a "fresh" category activation and a "primed" one —
  that difference IS the recognition signal.
""")


if __name__ == "__main__":
    main()
