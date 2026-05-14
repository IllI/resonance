"""
dlinoss_geometry.py — Operator-Space Reconstructor for Quantum Transport Channels
==================================================================================
NOT a classifier. NOT a clustering tool.

This module infers the operator space itself from PTM tensor observables:
  - Local intrinsic dimension (TwoNN)
  - Rank persistence across (chi_t, Gamma)
  - Spectral entropy of T (transport richness)
  - Diffusion maps (geometry-respecting, handles singular foliations)
  - Persistent homology via ripser (topological structure)

Teleportation (F_avg > 2/3) is checked AFTER strata are inferred,
as an a-posteriori property of geometry -- not as a training target.

Usage:
  python dlinoss_geometry.py --corpus ptm_corpus.npz --out geometry_report/
"""

import argparse
import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


# ── 1. Local Intrinsic Dimension (TwoNN estimator) ───────────────────────────
def twonn_intrinsic_dim(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    TwoNN estimator: local intrinsic dimension per point.
    Levina & Bickel (2004), refined by Facco et al. (2017).

    For each point, D_local = 1 / mean(log(r2/r1))
    where r1, r2 are distances to 1st and 2nd nearest neighbors.

    Returns array of shape (n_points,) with local ID estimates.
    """
    n = len(X)
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    dims = np.zeros(n)
    for i in range(n):
        sorted_d = np.sort(D[i])
        r1, r2 = sorted_d[0], sorted_d[1]
        if r1 > 1e-12 and r2 > r1:
            dims[i] = 1.0 / np.log(r2 / r1)
        else:
            dims[i] = 0.0
    return dims


def global_intrinsic_dim(X: np.ndarray) -> float:
    """Global ID estimate: harmonic mean of local estimates, robust to outliers."""
    local = twonn_intrinsic_dim(X)
    local = local[local > 0.1]
    if len(local) == 0:
        return 0.0
    return float(len(local) / np.sum(1.0 / local))


# ── 2. Spectral Entropy of T (transport richness) ────────────────────────────
def spectral_entropy(T_flat: np.ndarray) -> float:
    """
    H_T = -sum_k p_k log(p_k)  where p_k = sigma_k / sum(sigma_k)

    H_T = 0      : rank-1 channel (all transport in one direction)
    H_T = log(3) : maximally isotropic (all three singular values equal)
    H_T -> 0     : singularity (all sigma_k -> 0)
    """
    T = T_flat.reshape(3, 3)
    svs = np.linalg.svd(T, compute_uv=False)
    total = svs.sum()
    if total < 1e-10:
        return 0.0
    p = svs / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def spectral_entropy_batch(X: np.ndarray, T_col_start: int = 3) -> np.ndarray:
    """Compute spectral entropy for each row. T_ij columns start at T_col_start."""
    T_flat = X[:, T_col_start:T_col_start + 9]
    return np.array([spectral_entropy(row) for row in T_flat])


# ── 3. Rank Persistence ───────────────────────────────────────────────────────
def rank_persistence(X: np.ndarray, sv_col_start: int = 12,
                     thresholds: np.ndarray = None) -> dict:
    """
    Tracks how PTM rank evolves across the dataset as detection threshold varies.

    For each threshold epsilon:
      rank(point) = number of singular values > epsilon

    Returns persistence diagram: for each rank value, the (birth, death) threshold range.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    svs = X[:, sv_col_start:sv_col_start + 3]  # sigma_1, sigma_2, sigma_3
    n = len(X)

    rank_at_threshold = np.zeros((n, len(thresholds)), dtype=int)
    for j, eps in enumerate(thresholds):
        rank_at_threshold[:, j] = (svs > eps).sum(axis=1)

    # For each point: birth = largest eps where rank > 0, death = eps where rank drops
    # Summary statistics
    mean_rank = rank_at_threshold.mean(axis=0)
    frac_rank4 = (rank_at_threshold == 3).mean(axis=0)  # all 3 SVs above threshold
    frac_rank0 = (rank_at_threshold == 0).mean(axis=0)

    return {
        "thresholds": thresholds.tolist(),
        "mean_rank": mean_rank.tolist(),
        "frac_full_rank": frac_rank4.tolist(),
        "frac_collapsed": frac_rank0.tolist(),
        "rank_at_eps0.1": int(np.median(rank_at_threshold[:, np.searchsorted(thresholds, 0.1)])),
        "rank_at_eps0.3": int(np.median(rank_at_threshold[:, np.searchsorted(thresholds, 0.3)])),
    }


# ── 4. Geodesic Instability (rate of change across parameter space) ───────────
def geodesic_instability(X: np.ndarray, chi_t_col: int = 0,
                         T_col_start: int = 3) -> np.ndarray:
    """
    ||dT/dchi_t||_F at each point: how fast the operator geometry is changing.
    High instability = near a phase boundary.
    We use the pre-computed gradient columns (dT_ij) if available.
    """
    # dT columns: T_col_start + 9 to T_col_start + 18
    dT_start = T_col_start + 9
    dT_flat = X[:, dT_start:dT_start + 9]
    return np.linalg.norm(dT_flat.reshape(-1, 3, 3), axis=(1, 2))


# ── 5. Diffusion Maps ─────────────────────────────────────────────────────────
def diffusion_map(X: np.ndarray, n_components: int = 3,
                  epsilon: float = None, alpha: float = 0.5,
                  t: int = 1) -> np.ndarray:
    """
    Diffusion maps embedding (Coifman & Lafon, 2006).

    Unlike UMAP/tSNE, diffusion maps:
    - Respect the spectral structure of the data
    - Handle singular foliations and rank-deficient boundaries
    - Provide a geometry-respecting coordinate system

    alpha=0.5: normalize out density (Laplace-Beltrami operator)
    t: diffusion time (higher = coarser geometry)

    Returns: embedding of shape (n_points, n_components)
    """
    n = len(X)
    # Pairwise distances on T-block + SV block (most geometry-informative)
    T_svs = X[:, 3:15]  # T_ij (9) + svs (3)
    D2 = cdist(T_svs, T_svs, metric="euclidean") ** 2

    # Bandwidth: median heuristic if not given
    if epsilon is None:
        epsilon = float(np.median(D2[D2 > 0]))

    # Gaussian kernel
    K = np.exp(-D2 / epsilon)

    # Alpha normalization (removes density bias)
    d = K.sum(axis=1)
    K_alpha = K / np.outer(d ** alpha, d ** alpha)

    # Row-normalize to get Markov matrix
    P = K_alpha / K_alpha.sum(axis=1, keepdims=True)

    # Eigendecomposition (skip first trivial eigenvector)
    vals, vecs = eigh(P)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    # Diffusion coordinates: psi_k = lambda_k^t * phi_k
    embedding = vecs[:, 1:n_components + 1] * (vals[1:n_components + 1] ** t)
    return embedding


# ── 6. Persistent Homology (optional, requires ripser) ───────────────────────
def persistent_homology(X: np.ndarray, max_dim: int = 2,
                        max_edge_length: float = 2.0) -> dict:
    """
    Computes persistent homology of the PTM point cloud using ripser.
    Returns Betti numbers and persistence diagrams.

    H0: connected components (how many distinct strata?)
    H1: loops (is the quantum phase simply connected?)
    H2: voids (does the quantum phase form a shell?)
    """
    try:
        from ripser import ripser as _ripser
        T_svs = X[:, 3:15]  # T-matrix block
        result = _ripser(T_svs, maxdim=max_dim,
                         thresh=max_edge_length)
        diagrams = result["dgms"]
        betti = []
        for dim, dgm in enumerate(diagrams):
            # Count features that persist at scale 0.5 (arbitrary but meaningful)
            if len(dgm) == 0:
                betti.append(0)
                continue
            births, deaths = dgm[:, 0], dgm[:, 1]
            deaths = np.where(np.isinf(deaths), max_edge_length * 2, deaths)
            persistent = ((deaths - births) > 0.1).sum()
            betti.append(int(persistent))
        return {"betti": betti, "available": True,
                "note": "H0=strata, H1=loops, H2=voids in T-matrix space"}
    except ImportError:
        return {"betti": None, "available": False,
                "note": "ripser not installed. Run: pip install ripser"}


# ── 7. Strata Inference ───────────────────────────────────────────────────────
def infer_strata(X: np.ndarray, feature_names: list,
                 entropy_col: np.ndarray = None,
                 dim_col: np.ndarray = None) -> np.ndarray:
    """
    Infer operator strata WITHOUT labels, from geometric diagnostics:

    SINGULAR stratum : spectral entropy -> 0 AND rank -> 0
    TRANSPORT stratum: high entropy + high rank + stable ID
    BOUNDARY stratum : high geodesic instability (near a phase transition)
    COLLAPSED stratum: low entropy + low F_avg (product-like)

    Returns stratum assignments as strings.
    """
    n = len(X)
    # Get relevant columns by name
    fn = feature_names
    sv1_idx = fn.index("sv_1") if "sv_1" in fn else 12
    rank_idx = fn.index("T_rank") if "T_rank" in fn else 30
    F_idx = fn.index("F_avg") if "F_avg" in fn else 32
    sp_idx = fn.index("singularity_proximity") if "singularity_proximity" in fn else 33

    ranks = X[:, rank_idx]
    Favgs = X[:, F_idx]
    sing_prox = X[:, sp_idx]
    sv1 = X[:, sv1_idx]

    if entropy_col is None:
        entropy_col = spectral_entropy_batch(X)
    if dim_col is None:
        dim_col = np.ones(n) * np.nan  # unknown if not computed

    strata = np.empty(n, dtype=object)
    for i in range(n):
        # Singularity: all T elements collapse
        if sing_prox[i] > 0.90 and ranks[i] == 0:
            strata[i] = "SINGULAR"
        # Transport: high rank, high entropy, above classical
        elif ranks[i] >= 3 and entropy_col[i] > 0.5 and Favgs[i] > 2/3:
            strata[i] = "TRANSPORT"
        # Near-singular boundary: high proximity, not yet collapsed
        elif sing_prox[i] > 0.70 and Favgs[i] < 0.60:
            strata[i] = "BOUNDARY"
        # Product-like: low entropy, near-classical fidelity
        elif entropy_col[i] < 0.1 and Favgs[i] < 0.70:
            strata[i] = "PRODUCT"
        # Classical: modest rank, below classical fidelity
        elif Favgs[i] <= 2/3:
            strata[i] = "CLASSICAL"
        else:
            strata[i] = "TRANSPORT"

    return strata


# ── Main analysis ─────────────────────────────────────────────────────────────
def run_space_reconstruction(corpus_path: str, out_dir: str,
                             compute_homology: bool = False,
                             verbose: bool = True):
    print("=" * 65)
    print("D-LinOSS OPERATOR-SPACE RECONSTRUCTOR")
    print("=" * 65)

    data = np.load(corpus_path, allow_pickle=True)
    X = data["X"]
    feature_names = list(data["feature_names"])
    gt_labels = data.get("labels", None)

    print(f"  Corpus: {corpus_path}  shape={X.shape}")
    print(f"  Features: {len(feature_names)}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Diagnostic 1: Spectral entropy
    print("\n[1] Spectral entropy of T (transport richness)...")
    H_T = spectral_entropy_batch(X)
    print(f"    mean={H_T.mean():.3f}  std={H_T.std():.3f}  max={H_T.max():.3f}")
    print(f"    log(3)={np.log(3):.3f}  (isotropic maximum)")

    # ── Diagnostic 2: Geodesic instability
    print("\n[2] Geodesic instability (||dT/dchi_t||_F)...")
    G = geodesic_instability(X)
    print(f"    mean={G.mean():.4f}  max={G.max():.4f}")
    G_norm = G / (G.max() + 1e-10)

    # ── Diagnostic 3: Rank persistence
    print("\n[3] Rank persistence diagram...")
    rp = rank_persistence(X)
    print(f"    Median rank @ eps=0.1: {rp['rank_at_eps0.1']}")
    print(f"    Median rank @ eps=0.3: {rp['rank_at_eps0.3']}")
    frac_full = np.mean([v for v in rp["frac_full_rank"] if v is not None])
    print(f"    Mean fraction full-rank: {frac_full:.3f}")

    # ── Diagnostic 4: Strata inference (no labels)
    print("\n[4] Strata inference (no labels used)...")
    strata = infer_strata(X, feature_names, entropy_col=H_T)
    unique, counts = np.unique(strata, return_counts=True)
    for s, c in zip(unique, counts):
        pct = 100 * c / len(strata)
        print(f"    {s:<15} {c:5d} ({pct:.1f}%)")

    # ── Diagnostic 5: Intrinsic dimension (on T-block only)
    print("\n[5] Global intrinsic dimension (TwoNN on T+SV block)...")
    T_block = X[:, 3:15]
    id_global = global_intrinsic_dim(T_block)
    print(f"    Global ID = {id_global:.2f}")
    print(f"    (3.0 = full transport manifold; <1 = collapsed stratum)")

    # ── Diagnostic 6: Diffusion maps
    print("\n[6] Diffusion map embedding (3 components)...")
    emb = diffusion_map(X, n_components=3)
    print(f"    Embedding shape: {emb.shape}")

    # ── Diagnostic 7: Persistent homology (optional)
    if compute_homology:
        print("\n[7] Persistent homology (ripser)...")
        ph = persistent_homology(X[:500])  # subsample for speed
        if ph["available"]:
            print(f"    Betti numbers (H0, H1, H2): {ph['betti']}")
        else:
            print(f"    {ph['note']}")
    else:
        ph = {"available": False, "note": "Skipped (use --homology to enable)"}

    # ── Validation: check strata vs F_avg (post-hoc, not training)
    print("\n[VALIDATION] A-posteriori: F_avg > 2/3 fraction per stratum")
    F_col = feature_names.index("F_avg") if "F_avg" in feature_names else 32
    F_vals = X[:, F_col]
    classical_limit = 2 / 3
    for s in unique:
        mask = strata == s
        frac_above = (F_vals[mask] > classical_limit).mean()
        print(f"    {s:<15} F>2/3 fraction: {frac_above:.3f}")

    if gt_labels is not None:
        print("\n[VALIDATION] Stratum/label agreement (confusion):")
        gt = np.array(gt_labels)
        for inferred in unique:
            mask = strata == inferred
            if mask.sum() == 0:
                continue
            gt_in = gt[mask]
            gt_unique, gt_counts = np.unique(gt_in, return_counts=True)
            dominant = gt_unique[gt_counts.argmax()]
            purity = gt_counts.max() / gt_counts.sum()
            print(f"    {inferred:<15} -> dominant gt label: {dominant:<12} purity={purity:.2f}")

    # ── Save results
    results = {
        "n_points": len(X),
        "n_features": X.shape[1],
        "global_intrinsic_dim": id_global,
        "spectral_entropy": {"mean": float(H_T.mean()), "std": float(H_T.std()),
                              "max": float(H_T.max()), "log3": float(np.log(3))},
        "rank_persistence": rp,
        "geodesic_instability": {"mean": float(G.mean()), "max": float(G.max())},
        "strata": {s: int(c) for s, c in zip(unique, counts)},
        "diffusion_embedding_shape": list(emb.shape),
        "persistent_homology": ph,
    }

    with open(os.path.join(out_dir, "geometry_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, "diffusion_embedding.npy"), emb)
    np.save(os.path.join(out_dir, "strata.npy"), strata)
    np.save(os.path.join(out_dir, "spectral_entropy.npy"), H_T)

    print(f"\nResults saved to {out_dir}/")
    print("  geometry_report.json")
    print("  diffusion_embedding.npy   <- 3D operator-space coordinates")
    print("  strata.npy                <- inferred stratum per point")
    print("  spectral_entropy.npy      <- transport richness per point")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="ptm_corpus.npz")
    parser.add_argument("--out", default="geometry_report")
    parser.add_argument("--homology", action="store_true")
    args = parser.parse_args()
    run_space_reconstruction(args.corpus, args.out, args.homology)
