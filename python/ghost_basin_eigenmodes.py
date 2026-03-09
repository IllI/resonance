"""
ghost_basin_eigenmodes.py
───────────────────────────────────────────────────────────────
Tests whether the functional eigenmode basis reveals the Ghost Basin
more clearly than raw voxel space.

The Ghost Basin (from Phase 2, haxby_full_replication.py):
  - After Procrustes alignment, LOO classification = 100%
  - 7 CCA components explain 100% of shared variance
  - Category prototypes for face/house are maximally separated
  - Man-made objects cluster together

Hypothesis:
  The functional eigenmode basis provides a MORE NATURAL coordinate
  system for the Ghost Basin because:
  1. Eigenmodes capture functional NETWORK structure (not anatomy)
  2. Network topology is more conserved across subjects than voxel locations
  3. Procrustes disparity should be LOWER in eigenmode space

Prediction:
  - LOO accuracy >= 100% (preserved)
  - Procrustes disparity REDUCED vs voxel space
  - Category geometry more interpretable (fewer components needed)

Does NOT modify haxby_full_replication.py.
"""

import os, sys
import numpy as np
from scipy.stats import f_oneway
from scipy.linalg import eigh, orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS  = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)


def build_functional_laplacian(ts_z, top_idx):
    data = ts_z[top_idx, :]
    corr = np.corrcoef(data)
    np.fill_diagonal(corr, 0)
    W_exc = np.maximum(corr, 0)
    D_exc = np.diag(W_exc.sum(axis=1))
    return D_exc - W_exc


def get_subject_data(subj_dir, n_voxels=500):
    """Load BOLD, extract voxels, build eigenmodes, return prototypes."""
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
            fs, _ = f_oneway(*[g[:, v] for g in groups])
            if not np.isnan(fs): f_stats[v] = fs
        except: pass
    top_idx = np.argsort(f_stats)[-200:]   # 200 for tractable eigh

    # Functional eigenmode basis (excitatory)
    L = build_functional_laplacian(ts_z, top_idx)
    evals, evecs = eigh(L)   # [200,200] ascending

    # Category prototypes in VOXEL space (top500 from full brain F-test)
    top500 = np.argsort(f_stats)[-n_voxels:]
    proto_voxel = {}
    for cat in CATS:
        if cat in cat_pats:
            pats = [p[top500] for p in cat_pats[cat]]
            proto_voxel[cat] = np.stack(pats).mean(axis=0)  # [500]

    # Category prototypes in EIGENMODE space
    # First get avg pattern in 200-voxel space, then project onto eigenmodes
    proto_eigen_raw = {}
    for cat in CATS:
        if cat in cat_pats:
            pats = [p[top_idx] for p in cat_pats[cat]]
            avg_pat = np.stack(pats).mean(axis=0)     # [200]
            proto_eigen_raw[cat] = evecs.T @ avg_pat  # Φᵀp → [200] eigencoeffs

    # Keep top M modes by discriminative power (F-test on eigencoeffs)
    M_keep = 50   # top 50 eigenmode dimensions
    all_coeffs = {cat: proto_eigen_raw[cat] for cat in CATS
                  if cat in proto_eigen_raw}
    # F-test across categories for each mode
    mode_groups = []
    for cat in CATS:
        if cat in all_coeffs:
            mode_groups.append(all_coeffs[cat])
    # Just use all 200 for now — F-test on cross-subject variance later
    proto_eigen = {}
    for cat, coeffs in all_coeffs.items():
        proto_eigen[cat] = coeffs  # [200]

    return proto_voxel, proto_eigen, evals, evecs, top500, top_idx


def procrustes_align(source, target):
    """Align source → target via orthogonal Procrustes rotation."""
    R, scale = orthogonal_procrustes(source, target)
    aligned = source @ R
    disparity = np.linalg.norm(aligned - target, 'fro') / (
        np.linalg.norm(target, 'fro') + 1e-8)
    return aligned, R, disparity


def build_prototype_matrix(prototypes, cats):
    """Stack category prototypes into [K, D] matrix."""
    rows = []
    for cat in cats:
        if cat in prototypes:
            rows.append(prototypes[cat])
        else:
            rows.append(np.zeros_like(next(iter(prototypes.values()))))
    return np.stack(rows)   # [K, D]


def loo_classification(proto_matrices, space_name):
    """
    Leave-One-Subject-Out classification in a given space.
    proto_matrices: list of [K, D] arrays, one per subject
    Returns accuracy, mean disparity, per-subject details
    """
    n_subjects = len(proto_matrices)
    correct = 0
    total = 0
    disparities = []
    details = []

    # Reference = sub-1 (index 0)
    ref = proto_matrices[0]

    # Procrustes-align all subjects to reference
    aligned = [ref]
    for i in range(1, n_subjects):
        al, R, disp = procrustes_align(proto_matrices[i], ref)
        aligned.append(al)
        disparities.append(disp)

    # LOO: hold out subject i, build template from rest
    for i in range(n_subjects):
        template_others = [aligned[j] for j in range(n_subjects) if j != i]
        template = np.stack(template_others).mean(axis=0)  # [K, D]

        test = aligned[i]   # already aligned

        # Correlate each test category against all template categories
        cat_correct = 0
        for cat_idx in range(len(CATS)):
            test_pat = test[cat_idx]
            sims = [np.corrcoef(test_pat, template[j])[0, 1]
                    for j in range(len(CATS))]
            pred = np.argmax(sims)
            if pred == cat_idx:
                cat_correct += 1
                correct += 1
            total += 1

        details.append({
            'subject': SUBJECTS[i],
            'cat_acc': cat_correct / len(CATS),
        })

    return correct / total, np.mean(disparities) if disparities else 0, details


def analyze_category_geometry(aligned_stacks, space_name):
    """
    After alignment, analyze the geometry of the category space.
    aligned_stacks: list of [K, D] arrays (Procrustes-aligned)
    """
    # Mean prototype per category across subjects
    mean_proto = np.stack(aligned_stacks).mean(axis=0)  # [K, D]

    # Pairwise distances between categories
    dist_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dist_matrix[i, j] = np.linalg.norm(mean_proto[i] - mean_proto[j])

    # Variance explained by category (inter-category spread vs total)
    grand_mean = mean_proto.mean(axis=0)
    between_var = np.mean([np.linalg.norm(mean_proto[i] - grand_mean)**2
                           for i in range(K)])
    within_var = np.mean([
        np.mean([np.linalg.norm(s[i] - mean_proto[i])**2
                 for s in aligned_stacks])
        for i in range(K)
    ])
    discrimination_ratio = between_var / (within_var + 1e-8)

    return dist_matrix, discrimination_ratio, mean_proto


def main():
    print("=" * 70)
    print("  GHOST BASIN IN EIGENMODE SPACE")
    print("  Cross-subject alignment: voxel space vs functional eigenmodes")
    print("=" * 70)

    # Load all subjects
    print("\n  Loading all subjects...")
    all_proto_voxel = []
    all_proto_eigen = []
    loaded_subjects = []

    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            continue
        print(f"    {subj}...", end="", flush=True)
        pv, pe, evals, evecs, top500, top200 = get_subject_data(path)
        all_proto_voxel.append(build_prototype_matrix(pv, CATS))
        all_proto_eigen.append(build_prototype_matrix(pe, CATS))
        loaded_subjects.append(subj)
        print(f" done")

    N = len(loaded_subjects)
    print(f"\n  {N} subjects loaded")

    # ── Voxel space LOO ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  GHOST BASIN: VOXEL SPACE")
    print(f"{'─'*60}")
    vox_acc, vox_disp, vox_details = loo_classification(all_proto_voxel, "Voxel")
    print(f"  LOO accuracy:         {vox_acc*100:.1f}%")
    print(f"  Mean Procrustes disp: {vox_disp:.4f}")
    for d in vox_details:
        print(f"    {d['subject']}: {d['cat_acc']*100:.1f}%")

    # Category geometry in voxel space
    ref_vox = all_proto_voxel[0]
    aligned_vox = [ref_vox]
    for i in range(1, N):
        al, _, _ = procrustes_align(all_proto_voxel[i], ref_vox)
        aligned_vox.append(al)
    dist_vox, disc_vox, mean_vox = analyze_category_geometry(aligned_vox, "Voxel")

    print(f"\n  Category separation (discrimination ratio): {disc_vox:.3f}")
    print(f"\n  Pairwise category distances (voxel space, top 5 largest):")
    dists_flat = [(dist_vox[i, j], CATS[i], CATS[j])
                  for i in range(K) for j in range(i+1, K)]
    dists_flat.sort(reverse=True)
    for d, c1, c2 in dists_flat[:5]:
        print(f"    {c1:>14} <-> {c2:<14}: {d:.4f}")
    print(f"  ... top 5 smallest:")
    for d, c1, c2 in dists_flat[-5:]:
        print(f"    {c1:>14} <-> {c2:<14}: {d:.4f}")

    # ── Eigenmode space LOO ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  GHOST BASIN: FUNCTIONAL EIGENMODE SPACE")
    print(f"{'─'*60}")
    eig_acc, eig_disp, eig_details = loo_classification(all_proto_eigen, "Eigenmode")
    print(f"  LOO accuracy:         {eig_acc*100:.1f}%")
    print(f"  Mean Procrustes disp: {eig_disp:.4f}")
    for d in eig_details:
        print(f"    {d['subject']}: {d['cat_acc']*100:.1f}%")

    # Category geometry in eigenmode space
    ref_eig = all_proto_eigen[0]
    aligned_eig = [ref_eig]
    for i in range(1, N):
        al, _, _ = procrustes_align(all_proto_eigen[i], ref_eig)
        aligned_eig.append(al)
    dist_eig, disc_eig, mean_eig = analyze_category_geometry(aligned_eig, "Eigenmode")

    print(f"\n  Category separation (discrimination ratio): {disc_eig:.3f}")
    print(f"\n  Pairwise category distances (eigenmode space, top 5 largest):")
    dists_flat_e = [(dist_eig[i, j], CATS[i], CATS[j])
                    for i in range(K) for j in range(i+1, K)]
    dists_flat_e.sort(reverse=True)
    for d, c1, c2 in dists_flat_e[:5]:
        print(f"    {c1:>14} <-> {c2:<14}: {d:.4f}")
    print(f"  ... top 5 smallest:")
    for d, c1, c2 in dists_flat_e[-5:]:
        print(f"    {c1:>14} <-> {c2:<14}: {d:.4f}")

    # ── Comparison ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GHOST BASIN: SPACE COMPARISON")
    print(f"{'='*70}")

    print(f"\n  Metric                   Voxel Space    Eigenmode Space    Change")
    print(f"  {'─'*60}")
    print(f"  LOO Accuracy:            {vox_acc*100:>8.1f}%      {eig_acc*100:>8.1f}%"
          f"        {(eig_acc-vox_acc)*100:>+.1f}%")
    print(f"  Procrustes Disparity:    {vox_disp:>10.4f}      {eig_disp:>10.4f}"
          f"    {(eig_disp-vox_disp):>+.4f}")
    print(f"  Discrimination Ratio:    {disc_vox:>10.3f}      {disc_eig:>10.3f}"
          f"    {(disc_eig-disc_vox):>+.3f}")

    # Ghost Basin interpretation
    print(f"\n{'─'*60}")
    print(f"  GHOST BASIN INTERPRETATION")
    print(f"{'─'*60}")

    if eig_disp < vox_disp:
        pct_reduction = (vox_disp - eig_disp) / vox_disp * 100
        print(f"""
  Procrustes disparity REDUCED by {pct_reduction:.1f}% in eigenmode space.

  This confirms the Ghost Basin hypothesis:
  The shared categorical representation across subjects is MORE naturally
  expressed in functional connectivity eigenmode space than in voxel space.

  Why: Functional eigenmodes capture the network topology — the pattern of
  which brain regions co-activate — which is preserved across subjects even
  when the anatomical locations differ. The ghost basin IS the shared
  topology of categorical activation patterns.

  The eigenmode basis acts as a "natural coordinate system" for the ghost
  basin — one that subjects share despite anatomical differences.
""")
    else:
        print(f"""
  Procrustes disparity did NOT decrease in eigenmode space.
  This suggests the voxel-space alignment already captures the shared
  structure, and the eigenmode basis (from individual subjects' FC matrices)
  does not provide a more universal coordinate system.

  Possible reason: Each subject's eigenmodes are computed from their own
  FC matrix, so the eigenmode spaces are themselves idiosyncratic.
  A shared eigenmode basis (from group-average FC) would be needed.
""")

    # Chair in the ghost basin
    print(f"  CHAIR in the Ghost Basin:")
    for space_name, dist_mat in [("Voxel", dist_vox), ("Eigenmode", dist_eig)]:
        chair_idx = CATS.index('chair')
        chair_dists = [(dist_mat[chair_idx, j], CATS[j])
                       for j in range(K) if j != chair_idx]
        chair_dists.sort()
        nearest = chair_dists[0]
        farthest = chair_dists[-1]
        print(f"  {space_name:>10}: nearest={nearest[1]} ({nearest[0]:.3f}), "
              f"farthest={farthest[1]} ({farthest[0]:.3f})")


if __name__ == "__main__":
    main()
