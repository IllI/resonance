"""Offline AQ-1c operator grammar diagnostics.

This script does not contact TPU resources. It inspects the current AQ-1c
source/operator payload in feature space before we spend another TPU run.
"""

import json
import math

import jax.numpy as jnp

from program_ap_tunnel_eigen_recovery import (
    build_operator_identifiability_payload,
    build_semantic_motifs,
    build_layer_feature_table,
    graph_cosine_batch_jax,
    motif_feature,
    motif_transition_graph,
    nearest_class_indices_jax,
)


def pairwise_stats(values):
    distances = []
    cosines = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            a = values[i]
            b = values[j]
            distances.append(float(jnp.linalg.norm(a - b)))
            cosines.append(float(graph_cosine_batch_jax(a[None, :], b[None, :])[0]))
    return {
        "min_distance": min(distances),
        "max_cosine": max(cosines),
        "distances": distances,
        "cosines": cosines,
    }


def confusion(true_labels, pred_labels, n_true, n_pred):
    mat = [[0 for _ in range(n_pred)] for _ in range(n_true)]
    for true, pred in zip(true_labels, pred_labels):
        mat[int(true)][int(pred)] += 1
    return mat


def mutual_info(labels_a, labels_b):
    labels_a = [int(x) for x in labels_a]
    labels_b = [int(x) for x in labels_b]
    total = float(len(labels_a))
    a_values = sorted(set(labels_a))
    b_values = sorted(set(labels_b))
    mi = 0.0
    for a in a_values:
        pa = labels_a.count(a) / total
        for b in b_values:
            pab = sum(1 for x, y in zip(labels_a, labels_b) if x == a and y == b) / total
            if pab == 0.0:
                continue
            pb = labels_b.count(b) / total
            mi += pab * math.log2(pab / (pa * pb))
    return mi


def main():
    motifs, _ = build_semantic_motifs(8, aq_payload=True)
    motif_names = list(motifs.keys())
    seqs = jnp.stack([motifs[name] for name in motif_names])
    motif_features = jnp.stack([motif_feature(seq) for seq in seqs])
    graph_features = jnp.stack([motif_transition_graph(seq) for seq in seqs])
    layer_features = build_layer_feature_table(seqs)

    _, _, _, _, info = build_operator_identifiability_payload(
        motif_names,
        seqs,
        motif_features,
        graph_features,
        layer_features,
    )

    n_sources = int(info["n_sources"])
    n_operators = int(info["n_operators"])
    source_labels = [int(x) for x in info["source_indices"].tolist()]
    operator_labels = [int(x) for x in info["operator_indices"].tolist()]
    pair_labels = list(range(len(source_labels)))

    source_catalog = info["motif_feature_catalog"]
    operator_catalog = info["operator_code_catalog"]
    l5_codes = info["transform_code"]
    source_codes = info["source_features"]

    source_pred = nearest_class_indices_jax(source_codes, source_catalog).tolist()
    operator_pred = nearest_class_indices_jax(l5_codes, operator_catalog).tolist()
    l5_cluster = operator_pred

    delta_vectors = jnp.concatenate(
        [info["delta_graph"], info["delta_phase"], info["delta_recurrence"]],
        axis=1,
    )
    operator_delta_signatures = []
    operator_delta_variance = []
    for op_idx in range(n_operators):
        rows = delta_vectors[jnp.array(operator_labels) == op_idx]
        mean = jnp.mean(rows, axis=0)
        operator_delta_signatures.append(mean)
        operator_delta_variance.append(float(jnp.mean(jnp.sum((rows - mean) ** 2, axis=1))))
    operator_delta_signatures = jnp.stack(operator_delta_signatures)

    result = {
        "source_code_pairwise": pairwise_stats(source_catalog),
        "operator_code_pairwise": pairwise_stats(operator_catalog),
        "operator_delta_pairwise": pairwise_stats(operator_delta_signatures),
        "operator_delta_variance_by_operator": operator_delta_variance,
        "operator_invariance_score": float(1.0 / (1.0 + max(operator_delta_variance))),
        "mi_l5_source": mutual_info(l5_cluster, source_labels),
        "mi_l5_operator": mutual_info(l5_cluster, operator_labels),
        "mi_l5_target_pair": mutual_info(l5_cluster, pair_labels),
        "source_confusion": confusion(source_labels, source_pred, n_sources, n_sources),
        "operator_confusion": confusion(operator_labels, operator_pred, n_operators, n_operators),
        "passes_operator_code_distance_gate": bool(
            pairwise_stats(operator_catalog)["min_distance"] >= 0.15
            and pairwise_stats(operator_catalog)["max_cosine"] <= 0.85
        ),
        "passes_operator_delta_distance_gate": bool(
            pairwise_stats(operator_delta_signatures)["min_distance"] >= 0.15
            and pairwise_stats(operator_delta_signatures)["max_cosine"] <= 0.85
        ),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
