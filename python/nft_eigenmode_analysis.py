"""
nft_eigenmode_analysis.py
───────────────────────────────────────────────────────────────
Neural Field Theory (NFT) eigenmode decomposition of BOLD data.

The key physical constraint:
  BOLD Nyquist = 1/(2*TR) = 1/(2*2.5) = 0.2 Hz
  → Direct 1-350Hz oscillation is INVISIBLE in BOLD.

BUT: Neural Field Theory (Robinson et al. 1997, 2016) tells us that
the SPATIAL patterns of BOLD encode the STANDING WAVE GEOMETRY of
the underlying neural field. Specifically:

  The eigenmodes of the brain's Laplacian operator (spatial graph)
  correspond to the resonant modes of the neural field.
  
  Low eigenmodes  → global, slow oscillations (delta, theta)
  High eigenmodes → local, fast oscillations (beta, gamma)
  
  A category with strong BOLD response in a spatially COHERENT mode
  → driven by a global brain rhythm
  
  A category with response spread across MANY high modes
  → driven by local, fast oscillations (or noise)

We also check sub-2 head motion via temporal SNR (tSNR):
  tSNR = mean(BOLD) / std(BOLD) per voxel
  Low tSNR → head motion artifacts
"""

import os, sys
import numpy as np
from scipy.stats import f_oneway
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)

# NFT frequency band mapping from eigenmode index
# Based on Robinson et al. (2016): brain eigenmodes map to oscillation bands
# Mode rank (low=global/slow, high=local/fast)
NFT_BANDS = {
    'delta (1-4Hz)':   (0,   5),   # modes 0-4: global cortical modes
    'theta (4-8Hz)':   (5,  15),   # modes 5-14
    'alpha (8-12Hz)':  (15,  30),
    'beta (12-30Hz)':  (30,  80),
    'gamma (30-100Hz)':(80, 200),
    'high (100-350Hz)':(200, 500),
}


def compute_tsnr(ts_2d):
    """Temporal SNR: mean/std per voxel. Head motion lowers tSNR."""
    mn = ts_2d.mean(axis=1)
    sd = ts_2d.std(axis=1)
    sd[sd < 1e-10] = 1e-10
    return mn / sd


def build_spatial_graph_laplacian(coords, top_idx, k_neighbors=6):
    """
    Build the graph Laplacian from voxel spatial coordinates.
    
    Nodes = selected voxels (top_idx)
    Edges = k nearest neighbors in MNI space
    
    The Laplacian eigenmodes are the 'resonant modes' of the neural field
    on the cortical manifold approximated by these voxels.
    
    L = D - W
    where W_ij = exp(-||r_i - r_j||² / σ²) for k-NN pairs
    """
    sel_coords = coords[top_idx].astype(float)  # [V, 3]
    V = len(sel_coords)
    
    # Compute pairwise distances (subset to keep tractable)
    from scipy.spatial.distance import cdist
    dists = cdist(sel_coords, sel_coords)
    sigma = np.median(dists[dists > 0])  # adaptive bandwidth
    
    # k-NN adjacency
    W = np.zeros((V, V))
    for i in range(V):
        sorted_j = np.argsort(dists[i])
        neighbors = sorted_j[1:k_neighbors+1]  # exclude self
        for j in neighbors:
            w = np.exp(-dists[i, j]**2 / (2 * sigma**2))
            W[i, j] = w
            W[j, i] = w  # symmetric
    
    # Degree matrix and Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    return L, sigma


def decompose_into_eigenmodes(category_pattern, eigenvectors):
    """
    Project a category pattern onto the brain's eigenmodes.
    
    eigenvectors: [V, n_modes] — columns are spatial eigenmodes
    category_pattern: [V] — voxel activation pattern
    
    Returns:
        coefficients: [n_modes] — how much each mode contributes
        band_power: dict — total power per frequency band
    """
    # Project onto eigenmodes
    coeffs = eigenvectors.T @ category_pattern  # [n_modes]
    power = coeffs ** 2
    
    band_power = {}
    for band_name, (lo, hi) in NFT_BANDS.items():
        hi = min(hi, len(power))
        if lo < len(power):
            band_power[band_name] = power[lo:hi].sum() / (power.sum() + 1e-12)
    
    return coeffs, band_power


def analyze_subject(subj_dir, subj_id, n_modes=500):
    """Full NFT eigenmode analysis for one subject."""
    
    # Load BOLD
    bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
    V, T_total = ts_2d.shape
    
    # ── Step 1: tSNR (motion proxy) ──
    tsnr = compute_tsnr(ts_2d)
    
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
    top_idx = np.argsort(f_stats)[-200:]  # 200 voxels for tractable Laplacian
    
    # ── Step 2: Build spatial graph Laplacian ──
    L, sigma = build_spatial_graph_laplacian(coords, top_idx, k_neighbors=8)
    
    # ── Step 3: Eigendecomposition ──
    # Smallest eigenvalues = smoothest (most global) modes
    n_modes = min(n_modes, 195)  # can't exceed V-1
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Sort by eigenvalue (ascending = global to local)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]  # [V, n_modes]
    
    # ── Step 4: Project category patterns onto eigenmodes ──
    cat_prototypes = {}
    for cat in CATS:
        if cat in cat_pats:
            pats = [p[top_idx] for p in cat_pats[cat]]
            cat_prototypes[cat] = np.stack(pats).mean(axis=0)
    
    cat_eigenmodes = {}
    for cat, proto in cat_prototypes.items():
        coeffs, band_power = decompose_into_eigenmodes(proto, eigenvectors)
        cat_eigenmodes[cat] = {
            'coefficients': coeffs,
            'band_power': band_power,
            'dominant_mode': np.argmax(coeffs**2),
        }
    
    return {
        'tsnr_mean': tsnr.mean(),
        'tsnr_std': tsnr.std(),
        'tsnr_low_fraction': (tsnr < 10).mean(),  # fraction of low-tSNR voxels
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'cat_eigenmodes': cat_eigenmodes,
        'sigma': sigma,
        'top_idx': top_idx,
        'coords': coords,
    }


def main():
    print("=" * 70)
    print("  NEURAL FIELD THEORY EIGENMODE ANALYSIS")
    print("  1-350Hz band structure from BOLD spatial patterns")
    print("=" * 70)
    print("""
  Physical note:
    BOLD Nyquist = 0.2Hz — direct neural oscillations are NOT visible.
    
    BUT: NFT (Robinson et al.) shows that spatial BOLD patterns encode
    the STANDING WAVE GEOMETRY of the underlying neural field.
    
    Brain spatial eigenmodes map to frequency bands:
      Mode 0-4   (smoothest/global)  → delta  (1-4Hz)
      Mode 5-14                      → theta  (4-8Hz)
      Mode 15-29                     → alpha  (8-12Hz)
      Mode 30-79                     → beta   (12-30Hz)
      Mode 80-199                    → gamma  (30-100Hz)
      Mode 200+  (most local)        → high   (100-350Hz)
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
        
        # tSNR report
        print(f"\n  Temporal SNR (motion proxy):")
        print(f"    Mean tSNR: {result['tsnr_mean']:.1f}  "
              f"Std: {result['tsnr_std']:.1f}  "
              f"Low-tSNR voxels (<10): {result['tsnr_low_fraction']*100:.1f}%")
        
        # Eigenmode band power per category
        print(f"\n  NFT Band Power by Category (% of total spatial power):")
        bands = list(NFT_BANDS.keys())
        header = f"  {'Category':>14}" + "".join(f"  {b[:5]:>6}" for b in bands)
        print(header)
        print(f"  {'─'*75}")
        
        for cat in CATS:
            if cat not in result['cat_eigenmodes']:
                continue
            em = result['cat_eigenmodes'][cat]
            bp = em['band_power']
            row = f"  {cat:>14}"
            for band in bands:
                pct = bp.get(band, 0) * 100
                row += f"  {pct:>5.1f}%"
            dom = em['dominant_mode']
            row += f"  [dominant mode {dom}]"
            print(row)
        
        # Dominant modes per category
        print(f"\n  Dominant eigenmode (spatial scale of activation):")
        for cat in CATS:
            if cat not in result['cat_eigenmodes']:
                continue
            em = result['cat_eigenmodes'][cat]
            dom = em['dominant_mode']
            ev = result['eigenvalues'][dom]
            # Classify band
            for band, (lo, hi) in NFT_BANDS.items():
                if lo <= dom < hi:
                    band_name = band
                    break
            else:
                band_name = "ultra-high"
            print(f"    {cat:>14}: mode {dom:>4}, eigenvalue={ev:.4f}, "
                  f"band={band_name}")
    
    # Cross-subject comparison: is chair's eigenmode the odd one out?
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT NFT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  tSNR comparison (higher = less motion artifact):")
    for subj, res in all_results.items():
        tsnr = res['tsnr_mean']
        low_pct = res['tsnr_low_fraction'] * 100
        flag = " *** LOW TSNR (possible motion)" if tsnr < 40 else ""
        print(f"    {subj}: mean tSNR = {tsnr:.1f}  "
              f"low-quality voxels = {low_pct:.1f}%{flag}")
    
    print(f"\n  Dominant NFT band per category (most common across subjects):")
    for cat in CATS:
        band_votes = defaultdict(int)
        mode_vals = []
        for subj, res in all_results.items():
            if cat not in res['cat_eigenmodes']:
                continue
            em = res['cat_eigenmodes'][cat]
            dom = em['dominant_mode']
            mode_vals.append(dom)
            for band, (lo, hi) in NFT_BANDS.items():
                if lo <= dom < hi:
                    band_votes[band] += 1
                    break
        
        top_band = max(band_votes, key=band_votes.get) if band_votes else "unknown"
        mean_mode = np.mean(mode_vals) if mode_vals else 0
        std_mode = np.std(mode_vals) if mode_vals else 0
        print(f"    {cat:>14}: mode {mean_mode:>5.1f}±{std_mode:>4.1f}  "
              f"band={top_band}")
    
    print(f"\n  Key question: Is 'chair' in a distinct eigenmode band?")
    print(f"  If chair's dominant mode is significantly higher than others,")
    print(f"  it suggests chair recognition requires more LOCAL (fast) oscillations")
    print(f"  rather than the global standing waves driving face/house recognition.")


if __name__ == "__main__":
    main()
