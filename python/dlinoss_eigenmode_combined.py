"""
dlinoss_eigenmode_combined.py
───────────────────────────────────────────────────────────────
Combined D-LinOSS temporal encoding × functional eigenmode
spatial decomposition.

Mathematical basis: dlinoss_eigenmode_math.md

Pipeline:
  X[T,V] → DLinOSS → OSC[T,V] → Φᵀ @ OSC[t] → C[T,M]
         → [mean, var, AUC] → f[3M] → F-test → Haxby

Parseval's theorem guarantees: if M=V, Haxby result is identical
to voxel space. Truncating to M* < V can only improve by removing
noise-dominated modes.
"""

import os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, f_oneway
from scipy.linalg import eigh
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS  = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)


# ── 1. Functional connectivity Laplacian ──────────────────────

def build_functional_laplacian(ts_z, top_idx):
    data = ts_z[top_idx, :]
    corr = np.corrcoef(data)
    np.fill_diagonal(corr, 0)
    W_exc = np.maximum(corr, 0)
    D_exc = np.diag(W_exc.sum(axis=1))
    return D_exc - W_exc                          # L_fc (excitatory)


def compute_eigenmodes(L):
    evals, evecs = eigh(L)                        # ascending order
    return evals, evecs                            # Φ: [V, M]


# ── 2. D-LinOSS training (block trajectories) ────────────────

def train_dlinoss(trajectories, in_dim, n_epochs=40, lr=1e-3, device='cpu'):
    model = DLinOSSModel(
        input_dim=in_dim, state_dim=48, hidden_dim=96,
        output_dim=in_dim, num_blocks=3, drop_rate=0.1
    ).to(device)
    pred = nn.Sequential(
        nn.Linear(in_dim, 96), nn.GELU(), nn.Linear(96, in_dim)
    ).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(pred.parameters()), lr=lr, weight_decay=1e-4
    )
    for epoch in range(n_epochs):
        random.shuffle(trajectories)
        total = 0
        for traj in trajectories:
            if len(traj) < 3: continue
            x = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(device)
            osc = model(x)
            loss = F.mse_loss(pred(osc)[:, :-1, :], x[:, 1:, :])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1}: loss={total/max(len(trajectories),1):.5f}")
    return model


# ── 3. Combined encoding ──────────────────────────────────────

def encode_block_combined(model, block_traj, evecs, device='cpu'):
    """
    block_traj: [T, V] numpy float32
    evecs:      [V, M] eigenvectors (Φ)

    Returns f:[3M] = concat(mean_c, var_c, auc_c)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(block_traj).unsqueeze(0).to(device)   # [1,T,V]
        osc = model(x)[0].cpu().numpy()                         # [T,V]

    # Project each timestep: C[t] = Φᵀ @ osc[t]
    C = osc @ evecs                                             # [T, M]

    mean_c = C.mean(axis=0)                                     # [M]
    var_c  = C.var(axis=0)                                      # [M]
    auc_c  = np.abs(C).mean(axis=0)                             # [M]

    return np.concatenate([mean_c, var_c, auc_c])               # [3M]


# ── 4. Haxby analysis ─────────────────────────────────────────

def haxby_analysis(blocks):
    """Standard even/odd correlation Haxby analysis."""
    even = [b for b in blocks if b['run'] % 2 == 0]
    odd  = [b for b in blocks if b['run'] % 2 == 1]

    def avg(entries, cat):
        ps = [e['pattern'] for e in entries if e['cat'] == cat]
        return np.stack(ps).mean(axis=0) if ps else None

    cats = CATS
    n = len(cats)
    corr = np.zeros((n, n))
    for i, ce in enumerate(cats):
        pe = avg(even, ce)
        if pe is None: continue
        for j, co in enumerate(cats):
            po = avg(odd, co)
            if po is None: continue
            corr[i, j], _ = pearsonr(pe, po)

    correct = sum(1 for i, c in enumerate(cats) if cats[np.argmax(corr[i])] == c)
    wins    = sum(1 for i in range(n)
                  if corr[i, i] > np.mean([corr[i, j] for j in range(n) if j != i]))

    ep = {c: avg(even, c) for c in cats}
    op = {c: avg(odd, c)  for c in cats}
    pc = pt = 0
    for c1, c2 in combinations(cats, 2):
        if any(p is None for p in [ep[c1], ep[c2], op[c1], op[c2]]): continue
        rw1, _ = pearsonr(ep[c1], op[c1]); rw2, _ = pearsonr(ep[c2], op[c2])
        rb1, _ = pearsonr(ep[c1], op[c2]); rb2, _ = pearsonr(ep[c2], op[c1])
        if (rw1 + rw2) > (rb1 + rb2): pc += 1
        pt += 1

    details = {}
    for i, c in enumerate(cats):
        w = corr[i, i]
        b = np.mean([corr[i, j] for j in range(n) if j != i])
        best = cats[np.argmax(corr[i])]
        details[c] = {'within': w, 'delta': w - b, 'correct': best == c, 'best': best}

    return wins, n, correct/n, pc/pt if pt > 0 else 0, corr, details


# ── 5. Main ───────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  D-LinOSS × FUNCTIONAL EIGENMODE (COMBINED)")
    print("  Temporal encoding × spatial network decomposition")
    print("=" * 70)

    all_results = []

    for subj in SUBJECTS:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir): continue

        print(f"\n{'─'*60}\n  {subj}\n{'─'*60}")
        t0 = time.time()

        # Load
        bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
        bold_concat = np.concatenate(bold_runs, axis=-1)
        ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
        V, T_total = ts_2d.shape

        mu = ts_2d.mean(axis=1, keepdims=True)
        sd = ts_2d.std(axis=1, keepdims=True); sd[sd < 1e-8] = 1.0
        ts_z = np.clip((ts_2d - mu) / sd, -5, 5)

        all_labels = []; run_bounds = [0]
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
                if not np.isnan(f_stat): f_stats[v] = f_stat
            except: pass
        top200 = np.argsort(f_stats)[-200:]   # 200 for tractable eigh

        # Functional eigenmode basis
        print(f"  Building functional eigenmode basis...", end="", flush=True)
        L = build_functional_laplacian(ts_z, top200)
        evals, evecs = compute_eigenmodes(L)   # evecs: [200, 200]
        print(f" done  (λ range: {evals[1]:.3f}–{evals[-1]:.1f})")

        # Extract block trajectories for D-LinOSS training
        trajectories = []
        block_meta   = []   # run, cat for each block
        for ri in range(len(bold_runs)):
            s, e = run_bounds[ri], run_bounds[ri + 1]
            rl = all_labels[s:e]
            data = ts_z[top200, s:e]
            current_cat = None; block_start = None
            for i, lbl in enumerate(rl):
                if lbl != current_cat:
                    if current_cat in CATS and block_start is not None:
                        traj = data[:, block_start:i].T.astype(np.float32)
                        if len(traj) >= 3:
                            trajectories.append(traj)
                            block_meta.append({'run': ri, 'cat': current_cat})
                    current_cat = lbl; block_start = i
            if current_cat in CATS and block_start is not None:
                traj = data[:, block_start:].T.astype(np.float32)
                if len(traj) >= 3:
                    trajectories.append(traj)
                    block_meta.append({'run': ri, 'cat': current_cat})

        print(f"  {len(trajectories)} block trajectories")

        # ── Raw BOLD baseline (mean pattern, same 200 voxels) ──
        raw_blocks = []
        for traj, meta in zip(trajectories, block_meta):
            raw_blocks.append({**meta, 'pattern': traj.mean(axis=0)})
        rw, rn, rbm, rpa, _, rd = haxby_analysis(raw_blocks)
        print(f"\n  Raw BOLD (200 voxels): {rw}/8 w>b, {rbm*100:.1f}% bm, {rpa*100:.1f}% 28p")

        # ── Train D-LinOSS ──
        print(f"  Training D-LinOSS ({len(trajectories)} seqs, 40 epochs)...")
        model = train_dlinoss(trajectories, in_dim=200, n_epochs=40, device=device)

        # ── Combined encoding: f = [mean_c, var_c, auc_c] ──
        print(f"  Encoding blocks through D-LinOSS × eigenmodes...")
        M = evecs.shape[1]   # = 200
        combined_blocks = []
        for traj, meta in zip(trajectories, block_meta):
            f = encode_block_combined(model, traj, evecs, device)   # [3M=600]
            combined_blocks.append({**meta, 'pattern': f})

        # F-test on combined features (3M=600 dims)
        cat_combined = defaultdict(list)
        for b in combined_blocks:
            cat_combined[b['cat']].append(b['pattern'])
        grps = [np.stack(v) for v in cat_combined.values()]
        F3M = grps[0].shape[1]
        f3m_stats = np.zeros(F3M)
        for v in range(F3M):
            try:
                fs, _ = f_oneway(*[g[:, v] for g in grps])
                if not np.isnan(fs): f3m_stats[v] = fs
            except: pass
        top_feats = np.argsort(f3m_stats)[-500:]
        for b in combined_blocks:
            b['pattern'] = b['pattern'][top_feats]

        cw, cn, cbm, cpa, _, cd = haxby_analysis(combined_blocks)
        print(f"\n  Combined (DLinOSS × Eigenmodes):")
        print(f"    {cw}/8 w>b,  {cbm*100:.1f}% best-match,  {cpa*100:.1f}% 28-pair")

        # Per-category breakdown
        print(f"\n  Per-category (combined):")
        for cat in CATS:
            if cat in cd:
                d = cd[cat]
                mark = "✓" if d['correct'] else f"✗ → {d['best']}"
                print(f"    {cat:>14}: w={d['within']:.4f}  Δ={d['delta']:.4f}  {mark}")

        print(f"\n  Time: {time.time()-t0:.1f}s")
        all_results.append({
            'subject': subj,
            'raw_wins': rw, 'raw_bm': rbm, 'raw_pair': rpa,
            'comb_wins': cw, 'comb_bm': cbm, 'comb_pair': cpa,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Subject':>8}  {'Raw BM%':>8}  {'Raw 28p%':>9}  "
          f"{'Comb BM%':>9}  {'Comb 28p%':>10}  {'w>b':>5}")
    print(f"  {'─'*58}")
    for r in all_results:
        print(f"  {r['subject']:>8}  {r['raw_bm']*100:>7.1f}%  "
              f"{r['raw_pair']*100:>8.1f}%  "
              f"{r['comb_bm']*100:>8.1f}%  "
              f"{r['comb_pair']*100:>9.1f}%  "
              f"{r['comb_wins']}/8")
    avg_r = np.mean([r['raw_bm'] for r in all_results])
    avg_c = np.mean([r['comb_bm'] for r in all_results])
    avg_p = np.mean([r['comb_pair'] for r in all_results])
    print(f"\n  {'Average':>8}  {avg_r*100:>7.1f}%  "
          f"           {avg_c*100:>8.1f}%  {avg_p*100:>9.1f}%")
    print(f"\n  Previous best (transition D-LinOSS): 89.6% bm")
    print(f"  D-LinOSS baseline (haxby_full_replication): 91.7% bm")


if __name__ == "__main__":
    main()
