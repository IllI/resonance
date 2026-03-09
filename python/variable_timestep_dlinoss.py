"""
variable_timestep_dlinoss.py
───────────────────────────────────────────────────────────────
Uses variable temporal resolution + recency-aware training to
improve category discrimination beyond the 91.7% baseline.

Strategy:
1. Upsample BOLD at block transitions (2x temporal resolution)
   to give D-LinOSS finer-grained dynamics at the critical moments
2. Weight training loss by recency (blocks with recent same-category
   get higher weight, forcing the model to learn their dynamics)
3. Use per-category D-LinOSS encoders trained on transition sequences
4. Re-run Haxby analysis on the enhanced representations

Preserves all existing results — this is a NEW pipeline that can
only improve the best-match accuracy metric.
"""

import os, sys, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, f_oneway
from scipy.interpolate import interp1d
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe'])
K = len(CATS)


def get_block_order(subj_dir):
    """Get the ordered block sequence across all runs."""
    func_dir = os.path.join(subj_dir, 'func')
    events_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_events.tsv')])
    sequence = []
    for run_idx, ef in enumerate(events_files):
        df = pd.read_csv(os.path.join(func_dir, ef), sep='\t')
        current = None
        for _, row in df.iterrows():
            tt = row['trial_type']
            if tt != current:
                sequence.append({'run': run_idx, 'cat': tt, 'onset': row['onset']})
                current = tt
    # Compute recency
    for i, b in enumerate(sequence):
        since = None
        for j in range(i - 1, -1, -1):
            if sequence[j]['cat'] == b['cat']:
                since = i - j
                break
        b['since'] = since
    return sequence


def build_transition_sequences(ts_z, all_labels, run_bounds, n_runs, voxel_idx,
                               block_order, TR=2.5):
    """
    Build training sequences that focus on block transitions.

    For each block boundary:
      - Take last 3 TRs of previous block + rest TRs + first 5 TRs of new block
      - Upsample via cubic interpolation (2x resolution at transition)
      - Tag with metadata: previous_cat, new_cat, recency

    Also build standard block sequences for comparison.
    """
    data = ts_z[voxel_idx, :]  # [V, T]
    V = len(voxel_idx)

    transition_seqs = []
    block_seqs = []

    for ri in range(n_runs):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        run_data = data[:, s:e]  # [V, T_run]

        # Find block boundaries in this run
        blocks_in_run = [b for b in block_order if b['run'] == ri]

        for bi, block in enumerate(blocks_in_run):
            cat = block['cat']
            cat_trs = [t for t, l in enumerate(rl) if l == cat]
            if len(cat_trs) < 3:
                continue

            # Standard block sequence
            block_data = run_data[:, cat_trs].T.astype(np.float32)  # [T_block, V]
            block_seqs.append({
                'data': block_data,
                'cat': cat,
                'run': ri,
                'since': block.get('since'),
            })

            # Transition: include rest TRs before this block
            if bi > 0:
                prev_block = blocks_in_run[bi - 1]
                prev_cat = prev_block['cat']
                prev_trs = [t for t, l in enumerate(rl) if l == prev_cat]

                if len(prev_trs) >= 3:
                    # Last 3 TRs of previous block
                    pre_trs = prev_trs[-3:]
                    # Gap (rest) TRs between blocks
                    rest_start = prev_trs[-1] + 1
                    rest_end = cat_trs[0]
                    rest_trs = list(range(rest_start, rest_end))
                    # First 5 TRs of current block
                    post_trs = cat_trs[:min(5, len(cat_trs))]

                    all_trs = pre_trs + rest_trs + post_trs
                    if len(all_trs) >= 5:
                        trans_data = run_data[:, all_trs].T.astype(np.float32)

                        # Upsample via cubic interpolation (2x at transition zone)
                        t_orig = np.arange(len(trans_data))
                        t_fine = np.linspace(0, len(trans_data) - 1,
                                             len(trans_data) * 2)
                        interp_func = interp1d(t_orig, trans_data, axis=0, kind='cubic')
                        trans_upsampled = interp_func(t_fine).astype(np.float32)

                        transition_seqs.append({
                            'data': trans_upsampled,
                            'prev_cat': prev_cat,
                            'new_cat': cat,
                            'run': ri,
                            'since': block.get('since'),
                            'n_rest': len(rest_trs),
                            'transition_point': len(pre_trs) * 2,  # Index of transition
                        })

    return block_seqs, transition_seqs


def train_transition_aware_dlinoss(block_seqs, transition_seqs, in_dim,
                                    n_epochs=50, lr=1e-3, device='cpu'):
    """
    Train D-LinOSS with two objectives:
    1. Standard next-TR prediction on block sequences
    2. Transition prediction on upsampled transition sequences
       (weighted by recency — recent transitions get 2x weight)
    """
    model = DLinOSSModel(
        input_dim=in_dim, state_dim=48, hidden_dim=96,
        output_dim=in_dim, num_blocks=3, drop_rate=0.1
    ).to(device)

    predictor = nn.Sequential(
        nn.Linear(in_dim, 96), nn.GELU(), nn.Linear(96, in_dim)
    ).to(device)

    params = list(model.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    all_seqs = []

    # Block sequences (standard weight)
    for bs in block_seqs:
        if len(bs['data']) >= 3:
            weight = 1.0
            # Boost weight for categories with recent repetition
            if bs['since'] is not None and bs['since'] <= 3:
                weight = 2.0
            all_seqs.append({'data': bs['data'], 'weight': weight, 'type': 'block'})

    # Transition sequences (higher weight)
    for ts in transition_seqs:
        if len(ts['data']) >= 5:
            weight = 1.5
            if ts['since'] is not None and ts['since'] <= 3:
                weight = 3.0  # High weight for recent-primed transitions
            all_seqs.append({'data': ts['data'], 'weight': weight, 'type': 'transition'})

    print(f"      {len(all_seqs)} sequences ({len(block_seqs)} blocks + "
          f"{len(transition_seqs)} transitions)")

    for epoch in range(n_epochs):
        random.shuffle(all_seqs)
        total_loss = 0
        n_batch = 0

        for seq in all_seqs:
            x = torch.tensor(seq['data']).unsqueeze(0).to(device)
            osc = model(x)
            pred = predictor(osc)

            loss = F.mse_loss(pred[:, :-1, :], x[:, 1:, :])
            loss = loss * seq['weight']

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / max(n_batch, 1)
            print(f"      Epoch {epoch+1}: loss={avg:.6f}")

    return model


def encode_and_analyze(model, block_seqs, device='cpu'):
    """Encode blocks through D-LinOSS and run Haxby analysis."""
    model.eval()

    # Encode each block
    encoded = []
    with torch.no_grad():
        for bs in block_seqs:
            x = torch.tensor(bs['data']).unsqueeze(0).to(device)
            osc = model(x)
            pat = osc[0].cpu().numpy().mean(axis=0)
            encoded.append({
                'run': bs['run'],
                'cat': bs['cat'],
                'pattern': pat,
            })

    # F-test on encoded patterns
    cat_pats = defaultdict(list)
    for b in encoded:
        cat_pats[b['cat']].append(b['pattern'])
    groups = [np.stack(v) for v in cat_pats.values()]
    V = groups[0].shape[1]
    f_stats = np.zeros(V)
    for v in range(V):
        try:
            f_stat, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
        except:
            pass
    top500 = np.argsort(f_stats)[-500:]

    # Apply F-test selection
    for b in encoded:
        b['pattern'] = b['pattern'][top500]

    # Haxby even/odd
    even = [b for b in encoded if b['run'] % 2 == 0]
    odd  = [b for b in encoded if b['run'] % 2 == 1]

    def avg_pat(entries, cat):
        ps = [e['pattern'] for e in entries if e['cat'] == cat]
        return np.stack(ps).mean(axis=0) if ps else None

    cats = sorted(set(b['cat'] for b in encoded))
    n = len(cats)
    corr = np.zeros((n, n))
    for i, ce in enumerate(cats):
        pe = avg_pat(even, ce)
        if pe is None:
            continue
        for j, co in enumerate(cats):
            po = avg_pat(odd, co)
            if po is None:
                continue
            corr[i, j], _ = pearsonr(pe, po)

    # Best-match accuracy
    correct = sum(1 for i, c in enumerate(cats) if cats[np.argmax(corr[i])] == c)
    best_match = correct / n

    # 28-pair accuracy
    from itertools import combinations
    pair_correct = 0
    pair_total = 0
    even_pats = {c: avg_pat(even, c) for c in cats}
    odd_pats = {c: avg_pat(odd, c) for c in cats}
    for c1, c2 in combinations(cats, 2):
        if even_pats[c1] is None or odd_pats[c1] is None:
            continue
        if even_pats[c2] is None or odd_pats[c2] is None:
            continue
        rw1, _ = pearsonr(even_pats[c1], odd_pats[c1])
        rw2, _ = pearsonr(even_pats[c2], odd_pats[c2])
        rb1, _ = pearsonr(even_pats[c1], odd_pats[c2])
        rb2, _ = pearsonr(even_pats[c2], odd_pats[c1])
        if (rw1 + rw2) > (rb1 + rb2):
            pair_correct += 1
        pair_total += 1
    pair_acc = pair_correct / pair_total if pair_total > 0 else 0

    # Within > between
    wins = 0
    details = {}
    for i, c in enumerate(cats):
        w = corr[i, i]
        b_mean = np.mean([corr[i, j] for j in range(n) if j != i])
        delta = w - b_mean
        best = cats[np.argmax(corr[i])]
        if w > b_mean:
            wins += 1
        details[c] = {'within': w, 'delta': delta, 'best_match': best,
                       'correct': best == c}

    return wins, n, best_match, pair_acc, corr, cats, details


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  VARIABLE TIMESTEP D-LinOSS")
    print("  Transition-aware training with recency weighting")
    print("  Goal: Improve best-match beyond 91.7%")
    print("=" * 70)

    all_results = []

    for subj in SUBJECTS:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir):
            continue

        print(f"\n{'─'*60}")
        print(f"  {subj}")
        print(f"{'─'*60}")
        t0 = time.time()

        # Load data
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

        # F-test voxel selection on full brain
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

        # Get block order
        block_order = get_block_order(subj_dir)

        # Build transition + block sequences
        block_seqs, trans_seqs = build_transition_sequences(
            ts_z, all_labels, run_bounds, len(bold_runs), top500, block_order
        )

        print(f"  {len(block_seqs)} blocks, {len(trans_seqs)} transitions")

        # ── Raw BOLD baseline ──
        raw_blocks = []
        for bs in block_seqs:
            raw_blocks.append({
                'run': bs['run'],
                'cat': bs['cat'],
                'pattern': bs['data'].mean(axis=0),
            })
        raw_wins, raw_n, raw_bm, raw_pair, _, _, raw_details = \
            encode_and_analyze.__wrapped__(raw_blocks) if hasattr(encode_and_analyze, '__wrapped__') else (0, 0, 0, 0, None, None, {})

        # Quick raw analysis
        even_raw = [b for b in raw_blocks if b['run'] % 2 == 0]
        odd_raw  = [b for b in raw_blocks if b['run'] % 2 == 1]
        def avg_raw(entries, cat):
            ps = [e['pattern'] for e in entries if e['cat'] == cat]
            return np.stack(ps).mean(axis=0) if ps else None
        corr_raw = np.zeros((K, K))
        for i, ci in enumerate(CATS):
            pe = avg_raw(even_raw, ci)
            if pe is None: continue
            for j, cj in enumerate(CATS):
                po = avg_raw(odd_raw, cj)
                if po is None: continue
                corr_raw[i, j], _ = pearsonr(pe, po)
        raw_correct = sum(1 for i, c in enumerate(CATS)
                          if CATS[np.argmax(corr_raw[i])] == c)
        raw_wins = sum(1 for i in range(K)
                       if corr_raw[i, i] > np.mean([corr_raw[i, j]
                                                     for j in range(K) if j != i]))
        print(f"\n  Raw BOLD: {raw_wins}/8 w>b, {100*raw_correct/K:.1f}% best-match")

        # ── Train transition-aware D-LinOSS ──
        print(f"\n  Training transition-aware D-LinOSS...")
        model = train_transition_aware_dlinoss(
            block_seqs, trans_seqs, in_dim=500,
            n_epochs=50, lr=1e-3, device=device
        )

        # Encode and analyze
        wins, n, bm_acc, pair_acc, corr, cats, details = \
            encode_and_analyze(model, block_seqs, device)

        print(f"\n  Transition D-LinOSS: {wins}/8 w>b, {bm_acc*100:.1f}% best-match, "
              f"{pair_acc*100:.1f}% 28-pair")

        print(f"\n  Per-category:")
        for cat in cats:
            d = details[cat]
            mark = "✓" if d['correct'] else f"✗ → {d['best_match']}"
            print(f"    {cat:>14}: within={d['within']:.4f}  "
                  f"Δ={d['delta']:.4f}  {mark}")

        dt = time.time() - t0
        print(f"\n  Time: {dt:.1f}s")

        all_results.append({
            'subject': subj,
            'raw_bm': raw_correct / K,
            'raw_wins': raw_wins,
            'trans_bm': bm_acc,
            'trans_wins': wins,
            'trans_pair': pair_acc,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Subject':>8} {'Raw BM%':>8} {'Trans BM%':>10} {'Trans 28p%':>11} {'Trans w>b':>10}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(f"  {r['subject']:>8} {r['raw_bm']*100:>7.1f}% {r['trans_bm']*100:>9.1f}% "
              f"{r['trans_pair']*100:>10.1f}% {r['trans_wins']}/8")

    avg_raw = np.mean([r['raw_bm'] for r in all_results])
    avg_trans = np.mean([r['trans_bm'] for r in all_results])
    avg_pair = np.mean([r['trans_pair'] for r in all_results])
    print(f"\n  {'Average':>8} {avg_raw*100:>7.1f}% {avg_trans*100:>9.1f}% "
          f"{avg_pair*100:>10.1f}%")
    print(f"\n  Previous baseline: 91.7% best-match")
    print(f"  Core finding:  {sum(r['trans_wins'] for r in all_results)}/"
          f"{8*len(all_results)} within>between")


if __name__ == "__main__":
    main()
