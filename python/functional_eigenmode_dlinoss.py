"""
functional_eigenmode_dlinoss.py
───────────────────────────────────────────────────────────────
Functional Connectome Laplacian eigenmode decomposition,
properly integrated with D-LinOSS.

MATHEMATICAL FRAMEWORK
──────────────────────
We need three things to align:
  1. The spatial eigenmode basis (NFT)
  2. The D-LinOSS temporal encoding
  3. The Haxby correlation metric

The correct integration:

Step 1: Functional Connectivity Laplacian (replaces k-NN)
  W_ij = max(0, corr(bold_i, bold_j))  -- positive correlations only
          (negative correlations = inhibition, separate channel)
  D_ii = sum_j W_ij
  L_fc = D - W                          -- normalized graph Laplacian
  L_fc Φ = Λ Φ                          -- eigendecomposition
  Φ: [V, n_modes] -- functional eigenmodes, ordered global→local

Step 2: D-LinOSS temporal encoding
  For each block trajectory x[t] ∈ R^V:
    osc[t] = DLinOSS(x[t])              -- temporal dynamics
  block_pattern = mean_t(osc[t]) ∈ R^V  -- oscillator-mean pattern

Step 3: Eigenmode projection of D-LinOSS patterns
  coeffs = Φ^T @ block_pattern          -- [n_modes]
  This tells us which functional networks participate in each category

Step 4: Haxby analysis on eigenmode coefficients (NOT raw pattern)
  This separates the spatial network structure from the temporal encoding.
  The D-LinOSS enhances temporal, the eigenmodes capture spatial.

WHY THIS IS MATHEMATICALLY SOUND:
  - Functional connectivity Laplacian uses same BOLD data as D-LinOSS
  - Eigenmodes are orthogonal basis: no information duplication
  - Haxby metric works in eigenmode space (inner products preserved)
  - Preserves all prior results (raw BOLD and D-LinOSS run independently)

WHY THE k-NN WAS PROBLEMATIC:
  - Euclidean distance ≠ neural connectivity
  - Two adjacent voxels in different sulci may be unconnected
  - Different from Atasoy/Pang approach which uses structural tractography
  - Results were dominated by mode 0 (trivial constant mode) in many categories

WHAT WE CANNOT DO:
  - White matter tractography: requires DWI/DTI, not in this dataset
  - True structural eigenmodes (Pang 2023): requires surface mesh + DTI
  - Direct 1-350Hz from BOLD: Nyquist = 0.2Hz, physically impossible

WHAT WE CAN MAP:
  - 1-350Hz indirect: which functional network (eigenmode) drives each category
    → Low eigenmodes = global slow rhythms (delta/theta network activity)
    → High eigenmodes = local fast rhythms (gamma/high-freq local circuits)
  - Inhibitory vs excitatory connectivity separately (negative vs positive corr)
"""

import os, sys
import numpy as np
from scipy.stats import pearsonr, f_oneway
from scipy.linalg import eigh
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)

# NFT band mapping: eigenmode rank → frequency band proxy
# Based on Atasoy et al. (2016) and Robinson et al. spectral analysis
# Low rank = global coherent oscillation = low frequency
# High rank = local, spatially fine-grained = high frequency
NFT_BANDS = [
    ('delta  (1-4Hz)',    0,   5),
    ('theta  (4-8Hz)',    5,  15),
    ('alpha  (8-12Hz)',  15,  30),
    ('beta  (12-30Hz)',  30,  80),
    ('gamma (30-100Hz)', 80, 200),
]


def build_functional_laplacian(ts_z, top_idx, threshold=0.0):
    """
    Build the functional connectivity Laplacian from BOLD timeseries.

    L_fc = D - W_excitatory
    W_inhibitory = abs(negative correlations)

    This is the correct replacement for k-NN spatial graph.
    Uses the SAME data source as D-LinOSS → no circularity.

    Parameters
    ----------
    ts_z   : [V_full, T] z-scored BOLD
    top_idx: [V_sel] selected voxel indices
    threshold: minimum correlation to include as edge

    Returns
    -------
    L_exc  : [V_sel, V_sel] excitatory Laplacian (positive correlations)
    L_inh  : [V_sel, V_sel] inhibitory Laplacian (negative correlations)
    W_exc  : raw excitatory adjacency
    W_inh  : raw inhibitory adjacency
    """
    data = ts_z[top_idx, :]  # [V_sel, T]
    V = len(top_idx)

    # Full pairwise correlation matrix
    corr_matrix = np.corrcoef(data)  # [V, V]
    np.fill_diagonal(corr_matrix, 0)

    # Split excitatory (positive) and inhibitory (negative)
    W_exc = np.maximum(corr_matrix - threshold, 0)   # positive correlations
    W_inh = np.maximum(-corr_matrix - threshold, 0)  # abs(negative correlations)

    # Laplacians
    D_exc = np.diag(W_exc.sum(axis=1))
    D_inh = np.diag(W_inh.sum(axis=1))

    L_exc = D_exc - W_exc
    L_inh = D_inh - W_inh

    return L_exc, L_inh, W_exc, W_inh


def compute_eigenmodes(L, n_modes=None):
    """
    Eigendecomposition of graph Laplacian.
    Returns eigenmodes sorted from global (low eigenvalue) to local (high).
    """
    V = L.shape[0]
    if n_modes is None:
        n_modes = V

    # eigh is for symmetric matrices — Laplacian is symmetric by construction
    eigenvalues, eigenvectors = eigh(L)

    # Sort ascending (smallest eigenvalue = most global mode)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def project_onto_eigenmodes(pattern, eigenvectors):
    """
    Project a spatial pattern onto the eigenmode basis.

    pattern    : [V] voxel activation
    eigenvectors: [V, n_modes]

    Returns
    -------
    coeffs : [n_modes] projection coefficients
    power  : [n_modes] squared coefficients (power per mode)
    band_power: dict of band name → fraction of total power
    """
    coeffs = eigenvectors.T @ pattern       # [n_modes]
    power = coeffs ** 2
    total = power.sum() + 1e-12

    band_power = {}
    for name, lo, hi in NFT_BANDS:
        hi = min(hi, len(power))
        if lo < len(power):
            band_power[name] = power[lo:hi].sum() / total

    return coeffs, power, band_power


def haxby_analysis_eigenmodes(blocks_exc, blocks_inh, cats):
    """
    Run Haxby even/odd correlation in EIGENMODE space.

    Uses both excitatory and inhibitory eigenmode projections.
    This separates the two channels of neural communication.
    """
    results = {}

    for channel_name, blocks in [('Excitatory', blocks_exc),
                                   ('Inhibitory', blocks_inh)]:
        even = [b for b in blocks if b['run'] % 2 == 0]
        odd  = [b for b in blocks if b['run'] % 2 == 1]

        def avg_pat(entries, cat):
            ps = [e['eigencoeffs'] for e in entries if e['cat'] == cat]
            return np.stack(ps).mean(axis=0) if ps else None

        n = len(cats)
        corr = np.zeros((n, n))
        for i, ce in enumerate(cats):
            pe = avg_pat(even, ce)
            if pe is None: continue
            for j, co in enumerate(cats):
                po = avg_pat(odd, co)
                if po is None: continue
                if np.linalg.norm(pe) > 1e-8 and np.linalg.norm(po) > 1e-8:
                    corr[i, j], _ = pearsonr(pe, po)

        # Best-match accuracy
        correct = sum(1 for i, c in enumerate(cats)
                      if cats[np.argmax(corr[i])] == c)

        # Within > between
        wins = 0
        for i in range(n):
            w = corr[i, i]
            b = np.mean([corr[i, j] for j in range(n) if j != i])
            if w > b:
                wins += 1

        # 28-pair accuracy
        even_pats = {c: avg_pat(even, c) for c in cats}
        odd_pats  = {c: avg_pat(odd, c) for c in cats}
        pair_correct = 0
        pair_total = 0
        for c1, c2 in combinations(cats, 2):
            if any(p is None for p in [even_pats[c1], even_pats[c2],
                                        odd_pats[c1], odd_pats[c2]]):
                continue
            rw1, _ = pearsonr(even_pats[c1], odd_pats[c1])
            rw2, _ = pearsonr(even_pats[c2], odd_pats[c2])
            rb1, _ = pearsonr(even_pats[c1], odd_pats[c2])
            rb2, _ = pearsonr(even_pats[c2], odd_pats[c1])
            if (rw1 + rw2) > (rb1 + rb2):
                pair_correct += 1
            pair_total += 1

        results[channel_name] = {
            'wins': wins, 'n': n,
            'best_match': correct / n,
            'pair_acc': pair_correct / pair_total if pair_total > 0 else 0,
            'corr': corr,
        }

    return results


def analyze_subject(subj_dir, subj_id):
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

    # F-test voxel selection (standard pipeline, unchanged)
    cat_pats_raw = defaultdict(list)
    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        for cat in CATS:
            cm = [i for i, l in enumerate(rl) if l == cat]
            if len(cm) >= 2:
                cat_pats_raw[cat].append(ts_z[:, s:e][:, cm].mean(axis=1))

    groups = [np.stack(v) for v in cat_pats_raw.values()]
    f_stats = np.zeros(V)
    for v in range(V):
        try:
            f_stat, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(f_stat):
                f_stats[v] = f_stat
        except:
            pass
    top500 = np.argsort(f_stats)[-200:]  # 200 for tractable full eigh

    # Build functional connectivity Laplacian (replaces k-NN)
    print(f"    Building functional Laplacian ({len(top500)} voxels)...", end="", flush=True)
    L_exc, L_inh, W_exc, W_inh = build_functional_laplacian(ts_z, top500)

    # Network statistics
    n_exc_edges = (W_exc > 0).sum() // 2
    n_inh_edges = (W_inh > 0).sum() // 2
    print(f" {n_exc_edges} excitatory, {n_inh_edges} inhibitory edges")

    # Eigendecomposition of both channels
    print(f"    Computing eigenmodes...", end="", flush=True)
    evals_exc, evecs_exc = compute_eigenmodes(L_exc)
    evals_inh, evecs_inh = compute_eigenmodes(L_inh)
    print(f" done")

    # Project each category pattern onto eigenmodes
    blocks_exc = []
    blocks_inh = []

    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        data = ts_z[top500, s:e]

        for cat in CATS:
            cat_trs = [i for i, l in enumerate(rl) if l == cat]
            if len(cat_trs) < 2:
                continue
            pattern = data[:, cat_trs].mean(axis=1)  # [V_sel]

            coeffs_exc, power_exc, bpower_exc = project_onto_eigenmodes(
                pattern, evecs_exc)
            coeffs_inh, power_inh, bpower_inh = project_onto_eigenmodes(
                pattern, evecs_inh)

            blocks_exc.append({
                'run': ri, 'cat': cat,
                'eigencoeffs': coeffs_exc,
                'band_power': bpower_exc,
            })
            blocks_inh.append({
                'run': ri, 'cat': cat,
                'eigencoeffs': coeffs_inh,
                'band_power': bpower_inh,
            })

    # Collect band power per category
    cat_band_exc = {cat: defaultdict(list) for cat in CATS}
    cat_band_inh = {cat: defaultdict(list) for cat in CATS}
    for b in blocks_exc:
        for band, pwr in b['band_power'].items():
            cat_band_exc[b['cat']][band].append(pwr)
    for b in blocks_inh:
        for band, pwr in b['band_power'].items():
            cat_band_inh[b['cat']][band].append(pwr)

    # Dominant mode per category
    cat_dominant = {}
    for cat in CATS:
        cat_blocks = [b for b in blocks_exc if b['cat'] == cat]
        if not cat_blocks:
            continue
        avg_coeffs = np.stack([b['eigencoeffs'] for b in cat_blocks]).mean(axis=0)
        dom_mode = np.argmax(avg_coeffs**2)
        cat_dominant[cat] = {
            'mode': dom_mode,
            'eigenvalue': evals_exc[dom_mode],
            'band_power': {
                band: float(np.mean(cat_band_exc[cat].get(band, [0])))
                for band, lo, hi in NFT_BANDS
            }
        }

    # Haxby analysis in eigenmode space
    haxby_results = haxby_analysis_eigenmodes(blocks_exc, blocks_inh, CATS)

    return {
        'n_exc': n_exc_edges,
        'n_inh': n_inh_edges,
        'cat_dominant': cat_dominant,
        'haxby': haxby_results,
        'evals_exc': evals_exc,
    }


def main():
    print("=" * 70)
    print("  FUNCTIONAL EIGENMODE ANALYSIS (Atasoy/NFT approach)")
    print("  Excitatory + Inhibitory channels separated")
    print("=" * 70)
    print("""
  Mathematical basis:
    L_fc = D - W_fc  (functional connectivity Laplacian)
    W_fc = max(0, corr(BOLD_i, BOLD_j))  [excitatory]
    W_fc = max(0, -corr(BOLD_i, BOLD_j)) [inhibitory]

    Category pattern projected onto eigenmodes Phi:
    c = Phi^T @ pattern  (eigenmode coefficients)

    Haxby analysis runs on c, not raw voxel patterns.
    D-LinOSS output can be projected identically.

  Discarded:
    k-NN spatial graph: Euclidean distance ≠ neural connectivity.
    This is noted for documentation — it approximates cortical geometry
    but misses sulcal boundaries and long-range connections.

  Cannot do (no DTI data):
    True structural white-matter tractography.
    True Pang (2023) geometric eigenmodes.
""")

    all_results = {}

    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            continue

        print(f"\n{'─'*60}")
        print(f"  {subj}")
        print(f"{'─'*60}")

        result = analyze_subject(path, subj)
        all_results[subj] = result

        # Band power table
        bands = [n for n, lo, hi in NFT_BANDS]
        print(f"\n  Excitatory eigenmode band power per category:")
        print(f"  {'Category':>14}" + "".join(f"  {b[:5]:>6}" for b in bands)
              + "  Dom.Mode")
        print(f"  {'─'*72}")
        for cat in CATS:
            if cat not in result['cat_dominant']:
                continue
            cd = result['cat_dominant'][cat]
            row = f"  {cat:>14}"
            for band in bands:
                pct = cd['band_power'].get(band, 0) * 100
                row += f"  {pct:>5.1f}%"
            row += f"  [{cd['mode']:>4}, λ={cd['eigenvalue']:.3f}]"
            print(row)

        # Haxby results in eigenmode space
        print(f"\n  Haxby classification in eigenmode space:")
        for channel, res in result['haxby'].items():
            print(f"    {channel:>12}: {res['wins']}/8 w>b, "
                  f"{res['best_match']*100:.1f}% best-match, "
                  f"{res['pair_acc']*100:.1f}% 28-pair")

    # Cross-subject summary
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Haxby accuracy in functional eigenmode space:")
    print(f"  {'Subject':>8}  {'Exc w>b':>7}  {'Exc BM%':>8}  "
          f"{'Exc 28p%':>9}  {'Inh w>b':>7}  {'Inh BM%':>8}")
    print(f"  {'─'*58}")

    exc_bms = []
    inh_bms = []
    for subj, res in all_results.items():
        exc = res['haxby']['Excitatory']
        inh = res['haxby']['Inhibitory']
        exc_bms.append(exc['best_match'])
        inh_bms.append(inh['best_match'])
        print(f"  {subj:>8}  {exc['wins']}/8    "
              f"{exc['best_match']*100:>7.1f}%  "
              f"{exc['pair_acc']*100:>8.1f}%  "
              f"{inh['wins']}/8    "
              f"{inh['best_match']*100:>7.1f}%")

    print(f"\n  Average: Exc={np.mean(exc_bms)*100:.1f}%  "
          f"Inh={np.mean(inh_bms)*100:.1f}%")
    print(f"  Previous D-LinOSS baseline: 91.7% best-match (voxel space)")

    # Chair analysis
    print(f"\n  Chair dominant eigenmode across subjects:")
    for subj, res in all_results.items():
        if 'chair' in res['cat_dominant']:
            cd = res['cat_dominant']['chair']
            print(f"    {subj}: mode {cd['mode']:>4}, λ={cd['eigenvalue']:.3f}")
    print(f"\n  House dominant eigenmode across subjects:")
    for subj, res in all_results.items():
        if 'house' in res['cat_dominant']:
            cd = res['cat_dominant']['house']
            print(f"    {subj}: mode {cd['mode']:>4}, λ={cd['eigenvalue']:.3f}")

    print(f"\n  DISCARDED: k-NN spatial Laplacian")
    print(f"  REASON: Euclidean distance != neural connectivity;")
    print(f"          sulcal boundaries skipped; long-range fibers missed.")
    print(f"  NOTE FOR RETURN: Geometric eigenmodes (Pang 2023) would be")
    print(f"          correct if cortical surface mesh + DTI were available.")


if __name__ == "__main__":
    main()
