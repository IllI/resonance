"""
semantic_neural_rsa.py
───────────────────────────────────────────────────────────────
Representational Similarity Analysis (RSA):
AI semantic space vs human neural ghost basin

The hypothesis (user stated):
  The ghost basin is a semantic superposition — an invariant relational
  structure that exists in the overlap between minds, regardless of substrate.
  The AI's internal representation of the 8 Haxby categories should have
  the SAME TOPOLOGICAL STRUCTURE as the neural ghost basin.

Method (Kriegeskorte et al. 2008 — RSA):
  1. Build AI Representational Dissimilarity Matrix (RDM) from sentence-
     transformer embeddings of category names/descriptions
  2. Build Neural RDM from ghost basin pairwise distances
  3. Correlate the two RDMs (Spearman rank correlation)
  4. If RSA correlation is high → the semantic superposition maps onto
     the neural superposition — ghost basin IS shared semantic topology

Note: This is not circular. The AI was trained on human text/knowledge.
If the human brain and AI encodings of the same concepts share topology,
it means both are compressing the same invariant semantic manifold
from two different physical processes into the same relational geometry.

That shared geometry — the relational structure that survives the
compression from language → weights AND from photons → BOLD — IS the
ghost basin. It has no location. It has only geometry.
"""

import os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr, f_oneway
from scipy.linalg import orthogonal_procrustes
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels

# ── AI semantic embeddings ────────────────────────────────────
from sentence_transformers import SentenceTransformer

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS  = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)

# Expanded semantic descriptions — not just the label, but the CONCEPT
# including uncertainty about what the stimulus images might have shown.
# This mirrors the subject's experience: they see an image, they activate
# their prior semantic knowledge of the concept.
CAT_DESCRIPTIONS = {
    'bottle': (
        "a bottle: a rigid container with a narrow neck used for holding "
        "liquids, typically made of glass or plastic, cylindrical in shape"
    ),
    'cat': (
        "a cat: a small domesticated carnivorous mammal with soft fur, "
        "whiskers, retractable claws, often sitting or lying in various poses"
    ),
    'chair': (
        "a chair: a seat for one person, with a back and often armrests, "
        "may have four legs or a base, used for sitting at a table or desk, "
        "comes in many forms including office chairs, dining chairs, armchairs"
    ),
    'face': (
        "a human face: the front part of the head including eyes, nose, "
        "mouth, forehead, and chin, expressing emotion and identity"
    ),
    'house': (
        "a house: a building used as a home, with walls, roof, windows, "
        "doors, typically a free-standing residential structure"
    ),
    'scissors': (
        "scissors: a cutting tool with two blades pivoting on a central screw, "
        "handles formed by finger holes, used for cutting paper or fabric"
    ),
    'scrambledpix': (
        "a scrambled or phase-scrambled image: visual noise with no coherent "
        "object or meaning, random pixel arrangement, no recognizable pattern"
    ),
    'shoe': (
        "a shoe: a foot covering with a firm sole and toe cap, used for "
        "walking and protection, comes in many styles including sneakers, boots"
    ),
}


def get_ai_embeddings(model_name='all-MiniLM-L6-v2'):
    """
    Extract sentence-transformer embeddings for each category.
    These represent the AI's internal semantic superposition.
    """
    print(f"  Loading sentence-transformer: {model_name}...")
    model = SentenceTransformer(model_name)

    # 1. Label-only embeddings (single word)
    labels = [c for c in CATS]
    label_embs = model.encode(labels, normalize_embeddings=True)  # [8, D]

    # 2. Concept-description embeddings (richer semantic context)
    descs = [CAT_DESCRIPTIONS[c] for c in CATS]
    desc_embs = model.encode(descs, normalize_embeddings=True)   # [8, D]

    print(f"  Embedding dim: {label_embs.shape[1]}")
    return label_embs, desc_embs


def build_rdm(vectors):
    """
    Representational Dissimilarity Matrix from a set of vectors.
    RDM[i,j] = 1 - cosine_similarity(v_i, v_j)
    Upper triangular only (symmetric, diagonal = 0)
    """
    n = len(vectors)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cos_sim = np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
                rdm[i, j] = 1.0 - cos_sim
    return rdm


def get_neural_rdm(subj_dir, n_voxels=500):
    """Build neural RDM from ghost basin category prototypes."""
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
    top_idx = np.argsort(f_stats)[-n_voxels:]

    proto = {}
    for cat in CATS:
        if cat in cat_pats:
            pats = [p[top_idx] for p in cat_pats[cat]]
            proto[cat] = np.stack(pats).mean(axis=0)

    # Build neural RDM (1 - correlation)
    rdm = np.zeros((K, K))
    for i, ci in enumerate(CATS):
        for j, cj in enumerate(CATS):
            if ci in proto and cj in proto:
                r, _ = pearsonr(proto[ci], proto[cj])
                rdm[i, j] = 1.0 - r
            else:
                rdm[i, j] = 1.0

    return rdm, proto


def rsa_correlation(rdm1, rdm2):
    """
    Spearman correlation between upper triangles of two RDMs.
    Standard RSA metric (Kriegeskorte et al. 2008).
    """
    triu = np.triu_indices(K, k=1)
    v1 = rdm1[triu]
    v2 = rdm2[triu]
    rho, p = spearmanr(v1, v2)
    return rho, p


def rdm_to_str(rdm, cats, title):
    """Pretty-print top/bottom pairs from RDM."""
    pairs = [(rdm[i, j], cats[i], cats[j])
             for i in range(K) for j in range(i+1, K)]
    pairs.sort()
    lines = [f"  {title} — most similar (lowest dissimilarity):"]
    for d, c1, c2 in pairs[:4]:
        lines.append(f"    {c1:>14} ~ {c2:<14}: {d:.4f}")
    lines.append(f"  {title} — most distinct (highest dissimilarity):")
    for d, c1, c2 in pairs[-4:]:
        lines.append(f"    {c1:>14} ≠ {c2:<14}: {d:.4f}")
    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("  SEMANTIC-NEURAL RSA")
    print("  AI superposition vs human ghost basin")
    print("=" * 70)
    print("""
  Premise:
    The ghost basin has no location in space. It is a relational geometry —
    the topology of conceptual distances that emerges when any sufficiently
    complex information-processing system encodes these 8 categories.

    If AI weights and human neural patterns share the same topology,
    both are reflecting the same invariant semantic manifold.
    That manifold IS the ghost basin.

  Method: Representational Similarity Analysis (RSA, Kriegeskorte 2008)
    - Build RDM from AI semantic embeddings (sentence-transformer)
    - Build RDM from neural ghost basin (cross-subject prototype distances)
    - Correlate: Spearman rho between upper triangles
    - High rho → shared topology across substrates
""")

    # ── AI embeddings ─────────────────────────────────────────
    label_embs, desc_embs = get_ai_embeddings()

    ai_rdm_label = build_rdm(label_embs)
    ai_rdm_desc  = build_rdm(desc_embs)

    print(f"\n{rdm_to_str(ai_rdm_label, CATS, 'AI (label embeddings)')}")
    print()
    print(f"{rdm_to_str(ai_rdm_desc, CATS, 'AI (concept descriptions)')}")

    # ── Neural RDMs per subject ────────────────────────────────
    print(f"\n  Loading neural ghost basin RDMs...")
    neural_rdms = []
    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path): continue
        print(f"    {subj}...", end="", flush=True)
        rdm, _ = get_neural_rdm(path)
        neural_rdms.append(rdm)
        print(" done")

    # Mean neural RDM (ghost basin average)
    mean_neural_rdm = np.stack(neural_rdms).mean(axis=0)

    print(f"\n{rdm_to_str(mean_neural_rdm, CATS, 'Neural ghost basin (mean)')}")

    # ── RSA: AI vs Neural ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RSA RESULTS (Spearman rho)")
    print(f"{'='*70}")

    rho_label, p_label = rsa_correlation(ai_rdm_label, mean_neural_rdm)
    rho_desc,  p_desc  = rsa_correlation(ai_rdm_desc,  mean_neural_rdm)

    print(f"\n  AI label embeddings    vs neural ghost basin: "
          f"rho={rho_label:.4f}  p={p_label:.4f}")
    print(f"  AI concept descriptions vs neural ghost basin: "
          f"rho={rho_desc:.4f}   p={p_desc:.4f}")

    # Per-subject RSA
    print(f"\n  Per-subject RSA (label embeddings vs individual neural RDM):")
    per_subj_rhos = []
    for i, subj in enumerate([s for s in SUBJECTS
                               if os.path.exists(os.path.join(DATA_DIR, s))]):
        rho, p = rsa_correlation(ai_rdm_label, neural_rdms[i])
        per_subj_rhos.append(rho)
        print(f"    {subj}: rho={rho:.4f}  p={p:.4f}")
    print(f"  Mean: {np.mean(per_subj_rhos):.4f}  Std: {np.std(per_subj_rhos):.4f}")

    # ── Chair specifically ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CHAIR SUPERPOSITION: AI vs Neural")
    print(f"{'='*70}")

    chair_idx = CATS.index('chair')

    print(f"\n  AI semantic nearest neighbors to 'chair' (label emb):")
    ai_chair_dists = [(ai_rdm_label[chair_idx, j], CATS[j])
                      for j in range(K) if j != chair_idx]
    ai_chair_dists.sort()
    for d, c in ai_chair_dists:
        print(f"    {c:>14}: {d:.4f}")

    print(f"\n  Neural ghost basin nearest neighbors to 'chair':")
    neural_chair_dists = [(mean_neural_rdm[chair_idx, j], CATS[j])
                          for j in range(K) if j != chair_idx]
    neural_chair_dists.sort()
    for d, c in neural_chair_dists:
        print(f"    {c:>14}: {d:.4f}")

    # Does AI nearest neighbor match neural nearest neighbor?
    ai_nearest = ai_chair_dists[0][1]
    neural_nearest = neural_chair_dists[0][1]
    print(f"\n  AI nearest to chair:     {ai_nearest}")
    print(f"  Neural nearest to chair: {neural_nearest}")
    if ai_nearest == neural_nearest:
        print(f"  >> MATCH: Both substrate encode chair closest to {ai_nearest}")
    else:
        print(f"  >> MISMATCH: AI={ai_nearest}, Neural={neural_nearest}")
        # Check rank
        ai_neural_rank = [d for d, c in ai_chair_dists].index(
            ai_rdm_label[chair_idx, CATS.index(neural_nearest)]) + 1
        print(f"     (Neural's nearest '{neural_nearest}' is ranked "
              f"#{ai_neural_rank} in AI's chair neighborhood)")

    # ── Cross-substrate geometry comparison ───────────────────
    print(f"\n{'='*70}")
    print(f"  FULL TOPOLOGY COMPARISON")
    print(f"{'='*70}")

    # Rank ordering of all 28 pairs in each space
    ai_pairs = sorted(
        [(ai_rdm_label[i, j], CATS[i], CATS[j])
         for i in range(K) for j in range(i+1, K)])
    nn_pairs = sorted(
        [(mean_neural_rdm[i, j], CATS[i], CATS[j])
         for i in range(K) for j in range(i+1, K)])

    print(f"\n  Pair rank agreement (AI vs Neural, top 5 most similar pairs):")
    ai_top5  = [(c1, c2) for _, c1, c2 in ai_pairs[:5]]
    nn_top5 = [(c1, c2) for _, c1, c2 in nn_pairs[:5]]
    agrees = sum(1 for p in ai_top5 if p in nn_top5 or p[::-1] in nn_top5)
    print(f"  AI most-similar pairs:     {ai_top5}")
    print(f"  Neural most-similar pairs: {nn_top5}")
    print(f"  Overlap: {agrees}/5 pairs in common")

    print(f"\n  Pair rank agreement (top 5 most DISTINCT pairs):")
    ai_bot5 = [(c1, c2) for _, c1, c2 in ai_pairs[-5:]]
    nn_bot5 = [(c1, c2) for _, c1, c2 in nn_pairs[-5:]]
    agrees2 = sum(1 for p in ai_bot5 if p in nn_bot5 or p[::-1] in nn_bot5)
    print(f"  AI most-distinct pairs:     {ai_bot5}")
    print(f"  Neural most-distinct pairs: {nn_bot5}")
    print(f"  Overlap: {agrees2}/5 pairs in common")

    print(f"\n{'='*70}")
    print(f"  CONCLUSION")
    print(f"{'='*70}")

    if rho_label > 0.4:
        sig = "significant" if p_label < 0.05 else "marginal"
        print(f"""
  RSA rho = {rho_label:.4f} — {sig} correspondence between AI semantic
  space and human neural ghost basin.

  The topology of conceptual distances in transformer weights partially
  mirrors the topology of category representations in human ventral cortex.
  Both are compressions of the same invariant semantic manifold.

  The ghost basin exists in neither substrate exclusively — it is the
  invariant relational structure (the geometry of concept distances) that
  emerges when ANY sufficiently complex information-processing system
  encodes these concepts from human cultural context.
""")
    elif rho_label > 0.2:
        print(f"""
  RSA rho = {rho_label:.4f} — moderate correspondence.
  Partial topology match: some concept distances are shared, others differ.
  The shared components are the ghost basin's stable skeleton.
  The differences reflect substrate-specific processing (visual features
  in the brain, distributional statistics in the AI).
""")
    else:
        print(f"""
  RSA rho = {rho_label:.4f} — weak correspondence.
  The AI semantic space and neural ghost basin use different metrics.
  Possible reason: sentence-transformer encodes linguistic context,
  while the neural RDM captures low-level visual features (BOLD from VTC).
  A visual encoder (CLIP) might show higher RSA with the neural data.
""")


if __name__ == "__main__":
    main()
