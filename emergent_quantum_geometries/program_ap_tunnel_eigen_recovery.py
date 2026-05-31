"""
Program AP - Tunnel-Eigen Guided Folded Recovery
================================================
Tests whether folded semantic motifs can be recovered more reliably when
D-LinOSS unfolds the compressed manifold through stable tunneling eigenmodes
before semantic graph decoding.

This program extends Program AO:
  - keeps tunnel-affinity coupling and low-alpha/high-beta scoring
  - adds tunnel_eigen_unfolded as the main recovery controller
  - adds tunnel_eigen_residual for AP-R stability tests
  - adds AP-S stability/fidelity lanes for residual tuning
  - tests block_permute_size2 and block_permute_size2_within_random controls
  - leads with worst_control_margin, not noisy cosine
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import threading
from functools import partial

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

SHUTDOWN_FLAG = False

AP_S_FIDELITY_APR_REFERENCE = {
    0.30: {11: 0.0803, 29: 0.0762, 47: 0.1007},
    0.45: {11: 0.0258, 29: 0.0183, 47: 0.0287},
}

AP_U_REFERENCE_TOLERANCE = 0.003


def current_git_commit():
    env_commit = os.environ.get("PROGRAM_AP_GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def tpu_backend_summary():
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "has_tpu": any(getattr(device, "platform", "") == "tpu" for device in devices),
    }


def select_backend(require_tpu=False):
    info = tpu_backend_summary()
    if require_tpu and not info["has_tpu"]:
        raise RuntimeError(f"JAX is available but no TPU device was found: {info}")
    return "jax", info


def watchdog_and_heartbeat():
    global SHUTDOWN_FLAG
    while not SHUTDOWN_FLAG:
        try:
            with open("heartbeat.txt", "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.read().decode("utf-8") == "TRUE":
                    print("[WATCHDOG] Preemption imminent; saving partial state.")
                    SHUTDOWN_FLAG = True
        except Exception:
            pass
        time.sleep(5)


watchdog_thread = threading.Thread(target=watchdog_and_heartbeat, daemon=True)
watchdog_thread.start()


def build_semantic_motifs(seq_len, aq_payload=False):
    labels = jnp.arange(4, dtype=jnp.int32)
    t = jnp.arange(seq_len, dtype=jnp.int32)

    if aq_payload:
        def repeat_pattern(values):
            pattern = jnp.array(values, dtype=jnp.int32)
            return pattern[t % pattern.shape[0]]

        return {
            "cyclic": repeat_pattern([0, 1, 2, 3]),
            "fractal_nested": repeat_pattern([0, 1, 0, 2, 0, 1, 0, 3]),
            "chirp": repeat_pattern([0, 1, 2, 3, 2, 1]),
            "braided": repeat_pattern([0, 2, 1, 3, 1, 2, 0, 3]),
            "palindrome": repeat_pattern([0, 1, 2, 3, 3, 2, 1, 0]),
            "sparse": jnp.where(t % 7 == 0, 3, jnp.where(t % 5 == 0, 2, 0)),
            "bifurcating": repeat_pattern([0, 1, 0, 1, 2, 3, 2, 3]),
            "alternating": repeat_pattern([0, 0, 1, 0, 0, 1, 1, 0]),
        }, labels

    cyclic = jnp.where(t % 2 == 0, 0, 1)
    fractal = jnp.where(t % 2 == 0, 0, jnp.where(t % 4 == 1, 1, 2))
    chirp_cycle = jnp.array([0, 1, 2, 3, 2, 1], dtype=jnp.int32)
    chirp = chirp_cycle[t % chirp_cycle.shape[0]]
    braid_cycle = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    braided = braid_cycle[t % braid_cycle.shape[0]]

    return {
        "cyclic": cyclic,
        "fractal": fractal,
        "chirp": chirp,
        "braided": braided,
    }, labels


def markov_shuffle_sequence(seq, key):
    # Preserve the empirical transition multiset while destroying long-range ordering.
    pairs = jnp.stack([seq[:-1], seq[1:]], axis=1)
    perm = jax.random.permutation(key, pairs.shape[0])
    shuffled_pairs = pairs[perm]
    return jnp.concatenate([seq[:1], shuffled_pairs[:, 1]], axis=0)


def block_permute_sequence(seq, key, block_size):
    n_blocks = seq.shape[0] // block_size
    trimmed = seq[: n_blocks * block_size].reshape((n_blocks, block_size))
    perm = jax.random.permutation(key, n_blocks)
    shuffled = trimmed[perm].reshape((-1,))
    return jnp.concatenate([shuffled, seq[n_blocks * block_size :]], axis=0)


def block_permute_within_random_sequence(seq, key, block_size):
    key_blocks, key_inner = jax.random.split(key)
    n_blocks = seq.shape[0] // block_size
    trimmed = seq[: n_blocks * block_size].reshape((n_blocks, block_size))
    block_perm = jax.random.permutation(key_blocks, n_blocks)
    keys = jax.random.split(key_inner, n_blocks)
    inner_perm = jnp.stack([jax.random.permutation(keys[i], block_size) for i in range(n_blocks)])
    shuffled = jnp.take_along_axis(trimmed[block_perm], inner_perm, axis=1).reshape((-1,))
    return jnp.concatenate([shuffled, seq[n_blocks * block_size :]], axis=0)


def apply_sequence_control(seqs, control_name, key, block_size):
    if control_name == "markov_shuffle":
        keys = jax.random.split(key, seqs.shape[0])
        return jnp.stack([markov_shuffle_sequence(seq, keys[i]) for i, seq in enumerate(seqs)])
    if control_name in ("block_permute", "block_permute_size2"):
        keys = jax.random.split(key, seqs.shape[0])
        return jnp.stack([block_permute_sequence(seq, keys[i], block_size) for i, seq in enumerate(seqs)])
    if control_name == "block_permute_size2_within_random":
        keys = jax.random.split(key, seqs.shape[0])
        return jnp.stack([block_permute_within_random_sequence(seq, keys[i], block_size) for i, seq in enumerate(seqs)])
    return seqs


def motif_feature(seq):
    one_hot = jax.nn.one_hot(seq, 4)
    occ = jnp.mean(one_hot, axis=0)
    pair = one_hot[:-1, :, None] * one_hot[1:, None, :]
    trans = jnp.mean(pair, axis=0).reshape(-1)
    centered = one_hot - occ
    fft_mag = jnp.abs(jnp.fft.rfft(centered[:, 0], n=seq.shape[0]))[:6]
    fft_mag = fft_mag / (jnp.linalg.norm(fft_mag) + 1e-8)
    return jnp.concatenate([occ, trans, fft_mag])


def motif_transition_graph(seq):
    one_hot = jax.nn.one_hot(seq, 4)
    pair = one_hot[:-1, :, None] * one_hot[1:, None, :]
    trans = jnp.sum(pair, axis=0)
    trans = trans / (jnp.sum(trans) + 1e-8)
    return trans.reshape(-1)


def motif_layer_features(seq):
    one_hot = jax.nn.one_hot(seq, 4)
    occ = jnp.mean(one_hot, axis=0)

    pair = one_hot[:-1, :, None] * one_hot[1:, None, :]
    trans = jnp.sum(pair, axis=0)
    trans = trans / (jnp.sum(trans) + 1e-8)

    centered = one_hot - occ
    fft_mag = jnp.abs(jnp.fft.rfft(centered, axis=0, n=seq.shape[0]))[:6].reshape(-1)
    fft_mag = fft_mag / (jnp.linalg.norm(fft_mag) + 1e-8)

    recurrence = []
    max_lag = min(8, int(seq.shape[0]) - 1)
    for lag in range(1, max_lag + 1):
        recurrence.append(jnp.mean((seq[:-lag] == seq[lag:]).astype(jnp.float32)))
    while len(recurrence) < 8:
        recurrence.append(jnp.float32(0.0))
    recurrence.append(jnp.mean((seq == seq[::-1]).astype(jnp.float32)))

    return {
        "L1_occupation": occ,
        "L2_transition": trans.reshape(-1),
        "L3_fft_phase": fft_mag,
        "L4_recurrence": jnp.array(recurrence, dtype=jnp.float32),
    }


def build_layer_feature_table(seqs):
    layer_names = ("L1_occupation", "L2_transition", "L3_fft_phase", "L4_recurrence")
    per_motif = [motif_layer_features(seq) for seq in seqs]
    return {name: jnp.stack([features[name] for features in per_motif]) for name in layer_names}


def build_transform_field_payload(motif_names, seqs, motif_features, graph_features, layer_features):
    pair_names = [
        ("cyclic", "chirp"),
        ("palindrome", "braided"),
        ("sparse", "bifurcating"),
        ("fractal_nested", "alternating"),
    ]
    name_to_idx = {name: i for i, name in enumerate(motif_names)}
    src_idx = jnp.array([name_to_idx[src] for src, _ in pair_names], dtype=jnp.int32)
    tgt_idx = jnp.array([name_to_idx[tgt] for _, tgt in pair_names], dtype=jnp.int32)
    halfway = int(seqs.shape[1]) // 2
    left = seqs[src_idx, :halfway]
    right = seqs[tgt_idx, halfway:]
    pair_seqs = jnp.concatenate([left, right], axis=1)

    src_features = motif_features[src_idx]
    tgt_features = motif_features[tgt_idx]
    src_graph = graph_features[src_idx]
    tgt_graph = graph_features[tgt_idx]
    delta_graph = tgt_graph - src_graph
    pair_features = jnp.concatenate([src_features, tgt_features, tgt_features - src_features], axis=1)
    pair_graph_features = jnp.concatenate([src_graph, tgt_graph, delta_graph], axis=1)

    delta_l1 = layer_features["L1_occupation"][tgt_idx] - layer_features["L1_occupation"][src_idx]
    delta_l2 = layer_features["L2_transition"][tgt_idx] - layer_features["L2_transition"][src_idx]
    delta_l3 = layer_features["L3_fft_phase"][tgt_idx] - layer_features["L3_fft_phase"][src_idx]
    delta_l4 = layer_features["L4_recurrence"][tgt_idx] - layer_features["L4_recurrence"][src_idx]
    transform_code = jnp.concatenate([delta_l1, delta_l2, delta_l3, delta_l4], axis=1)
    pair_layer_features = {
        "L1_occupation": layer_features["L1_occupation"][tgt_idx],
        "L2_transition": layer_features["L2_transition"][tgt_idx],
        "L3_fft_phase": layer_features["L3_fft_phase"][tgt_idx],
        "L4_recurrence": layer_features["L4_recurrence"][tgt_idx],
        "L5_transform": transform_code,
    }
    transform_info = {
        "pair_names": [f"{src}_to_{tgt}" for src, tgt in pair_names],
        "source_indices": src_idx,
        "target_indices": tgt_idx,
        "source_features": src_features,
        "target_features": tgt_features,
        "transform_code": transform_code,
        "delta_graph": delta_graph,
        "delta_phase": delta_l3,
        "delta_recurrence": delta_l4,
        "motif_feature_catalog": motif_features,
    }
    return pair_seqs, pair_features, pair_graph_features, pair_layer_features, transform_info


def build_operator_conditioned_payload(motif_names, seqs, motif_features, graph_features, layer_features):
    operator_specs = [
        ("symbol_shift", jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)),
        ("symbol_swap_pairs", jnp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=jnp.int32)),
        ("reverse_time", jnp.array([3, 2, 1, 0, 3, 2, 1, 0], dtype=jnp.int32)),
        ("phase_roll", jnp.array([0, 2, 0, 2, 0, 2, 0, 2], dtype=jnp.int32)),
    ]

    def apply_operator(seq, op_name):
        if op_name == "symbol_shift":
            return (seq + 1) % 4
        if op_name == "symbol_swap_pairs":
            symbol_map = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
            return symbol_map[seq]
        if op_name == "reverse_time":
            return seq[::-1]
        if op_name == "phase_roll":
            return jnp.roll(seq, 1)
        return seq

    def marker_for_length(marker, seq_len):
        repeats = (int(seq_len) + int(marker.shape[0]) - 1) // int(marker.shape[0])
        return jnp.tile(marker, repeats)[:seq_len]

    source_indices = []
    operator_indices = []
    pair_names = []
    payload_seqs = []
    target_seqs = []
    for src_idx, src_name in enumerate(motif_names):
        for op_idx, (op_name, marker) in enumerate(operator_specs):
            marker_full = marker_for_length(marker, seqs.shape[1])
            source_indices.append(src_idx)
            operator_indices.append(op_idx)
            pair_names.append(f"{src_name}_with_{op_name}")
            # Payload contains source-plus-operator structure only. The derived
            # target is kept out of the transported sequence and used for eval.
            payload_seqs.append((seqs[src_idx] + marker_full) % 4)
            target_seqs.append(apply_operator(seqs[src_idx], op_name))

    src_idx = jnp.array(source_indices, dtype=jnp.int32)
    op_idx = jnp.array(operator_indices, dtype=jnp.int32)
    payload_seqs = jnp.stack(payload_seqs)
    target_seqs = jnp.stack(target_seqs)
    target_features = jnp.stack([motif_feature(seq) for seq in target_seqs])
    target_graph_features = jnp.stack([motif_transition_graph(seq) for seq in target_seqs])
    target_layer_features = build_layer_feature_table(target_seqs)

    operator_codes = []
    for op_name, marker in operator_specs:
        marker_full = marker_for_length(marker, seqs.shape[1])
        op_targets = jnp.stack([apply_operator(seq, op_name) for seq in seqs])
        op_target_layers = build_layer_feature_table(op_targets)
        deltas = []
        for layer_name in ("L1_occupation", "L2_transition", "L3_fft_phase", "L4_recurrence"):
            deltas.append(jnp.mean(op_target_layers[layer_name] - layer_features[layer_name], axis=0))
        marker_hist = jnp.mean(jax.nn.one_hot(marker_full, 4), axis=0)
        operator_codes.append(jnp.concatenate([marker_hist] + deltas))
    operator_code_catalog = jnp.stack(operator_codes)
    operator_code = operator_code_catalog[op_idx]

    source_features = motif_features[src_idx]
    source_graph = graph_features[src_idx]
    pair_features = jnp.concatenate([source_features, operator_code], axis=1)
    pair_graph_features = jnp.stack([motif_transition_graph(seq) for seq in payload_seqs])
    pair_layer_features = {
        "L1_occupation": layer_features["L1_occupation"][src_idx],
        "L2_transition": layer_features["L2_transition"][src_idx],
        "L3_fft_phase": layer_features["L3_fft_phase"][src_idx],
        "L4_recurrence": layer_features["L4_recurrence"][src_idx],
        "L5_transform": operator_code,
    }
    transform_info = {
        "operator_conditioned": True,
        "pair_names": pair_names,
        "source_indices": src_idx,
        "operator_indices": op_idx,
        "source_features": source_features,
        "target_features": target_features,
        "transform_code": operator_code,
        "delta_graph": target_graph_features - source_graph,
        "delta_phase": target_layer_features["L3_fft_phase"] - layer_features["L3_fft_phase"][src_idx],
        "delta_recurrence": target_layer_features["L4_recurrence"] - layer_features["L4_recurrence"][src_idx],
        "motif_feature_catalog": motif_features,
        "operator_code_catalog": operator_code_catalog,
        "target_feature_catalog": target_features,
        "source_names": motif_names,
        "operator_names": [name for name, _ in operator_specs],
        "n_sources": len(motif_names),
        "n_operators": len(operator_specs),
    }
    return payload_seqs, pair_features, pair_graph_features, pair_layer_features, transform_info


def build_operator_identifiability_payload(motif_names, seqs, motif_features, graph_features, layer_features):
    source_names = ["cyclic", "fractal_nested", "palindrome", "sparse"]
    source_idx_list = [motif_names.index(name) for name in source_names if name in motif_names]
    source_idx_base = jnp.array(source_idx_list, dtype=jnp.int32)
    seqs = seqs[source_idx_base]
    motif_features = motif_features[source_idx_base]
    graph_features = graph_features[source_idx_base]
    layer_features = {name: values[source_idx_base] for name, values in layer_features.items()}
    motif_names = [motif_names[i] for i in source_idx_list]
    source_one_hot = jnp.eye(len(motif_names), dtype=jnp.float32)
    source_code_catalog = jnp.concatenate(
        [
            4.0 * source_one_hot,
            motif_features,
            layer_features["L1_occupation"],
            layer_features["L2_transition"],
            layer_features["L3_fft_phase"],
            layer_features["L4_recurrence"],
        ],
        axis=1,
    )

    operator_specs = [
        ("MIRROR_TIME", jnp.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.int32)),
        ("ROTATE_PHASE", jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32)),
        ("SYMBOL_SHIFT", jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)),
        ("SWAP_ATTRACTOR", jnp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=jnp.int32)),
    ]

    def apply_operator(seq, op_name):
        if op_name == "MIRROR_TIME":
            return seq[::-1]
        if op_name == "ROTATE_PHASE":
            return jnp.roll(seq, 1)
        if op_name == "SYMBOL_SHIFT":
            return (seq + 1) % 4
        if op_name == "SWAP_ATTRACTOR":
            symbol_map = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
            return symbol_map[seq]
        return seq

    def marker_for_length(marker, seq_len):
        repeats = (int(seq_len) + int(marker.shape[0]) - 1) // int(marker.shape[0])
        return jnp.tile(marker, repeats)[:seq_len]

    source_indices = []
    operator_indices = []
    pair_names = []
    payload_seqs = []
    target_seqs = []
    for src_idx, src_name in enumerate(motif_names):
        for op_idx, (op_name, marker) in enumerate(operator_specs):
            source_indices.append(src_idx)
            operator_indices.append(op_idx)
            pair_names.append(f"{src_name}_with_{op_name}")
            payload_seqs.append(seqs[src_idx])
            target_seqs.append(apply_operator(seqs[src_idx], op_name))

    src_idx = jnp.array(source_indices, dtype=jnp.int32)
    op_idx = jnp.array(operator_indices, dtype=jnp.int32)
    payload_seqs = jnp.stack(payload_seqs)
    target_seqs = jnp.stack(target_seqs)
    target_features = jnp.stack([motif_feature(seq) for seq in target_seqs])
    target_graph_features = jnp.stack([motif_transition_graph(seq) for seq in target_seqs])
    target_layer_features = build_layer_feature_table(target_seqs)

    operator_codes = []
    for op_idx_value, (op_name, marker) in enumerate(operator_specs):
        marker_full = marker_for_length(marker, seqs.shape[1])
        op_targets = jnp.stack([apply_operator(seq, op_name) for seq in seqs])
        op_target_layers = build_layer_feature_table(op_targets)
        deltas = []
        for layer_name in ("L1_occupation", "L2_transition", "L3_fft_phase", "L4_recurrence"):
            deltas.append(jnp.mean(op_target_layers[layer_name] - layer_features[layer_name], axis=0))
        marker_hist = jnp.mean(jax.nn.one_hot(marker_full, 4), axis=0)
        op_one_hot = jax.nn.one_hot(op_idx_value, len(operator_specs), dtype=jnp.float32)
        operator_codes.append(jnp.concatenate([4.0 * op_one_hot, 2.0 * marker_hist] + deltas))
    operator_code_catalog = jnp.stack(operator_codes)
    operator_code = operator_code_catalog[op_idx]

    source_features = source_code_catalog[src_idx]
    source_graph = graph_features[src_idx]
    pair_features = jnp.concatenate([source_features, operator_code], axis=1)
    pair_graph_features = jnp.stack([motif_transition_graph(seq) for seq in payload_seqs])
    pair_layer_features = {
        "L1_occupation": layer_features["L1_occupation"][src_idx],
        "L2_transition": layer_features["L2_transition"][src_idx],
        "L3_fft_phase": layer_features["L3_fft_phase"][src_idx],
        "L4_recurrence": layer_features["L4_recurrence"][src_idx],
        "L5_transform": operator_code,
    }
    transform_info = {
        "operator_conditioned": True,
        "pair_names": pair_names,
        "source_indices": src_idx,
        "operator_indices": op_idx,
        "source_features": source_features,
        "target_features": target_features,
        "transform_code": operator_code,
        "delta_graph": target_graph_features - source_graph,
        "delta_phase": target_layer_features["L3_fft_phase"] - layer_features["L3_fft_phase"][src_idx],
        "delta_recurrence": target_layer_features["L4_recurrence"] - layer_features["L4_recurrence"][src_idx],
        "motif_feature_catalog": source_code_catalog,
        "source_sequence_catalog": seqs,
        "operator_code_catalog": operator_code_catalog,
        "target_feature_catalog": target_features,
        "source_names": motif_names,
        "operator_names": [name for name, _ in operator_specs],
        "n_sources": len(motif_names),
        "n_operators": len(operator_specs),
    }
    return payload_seqs, pair_features, pair_graph_features, pair_layer_features, transform_info


def build_sequential_state_gate_payload(motif_names, seqs, motif_features, graph_features, layer_features):
    source_names = ["cyclic", "fractal_nested", "palindrome", "sparse"]
    source_idx_list = [motif_names.index(name) for name in source_names if name in motif_names]
    source_idx_base = jnp.array(source_idx_list, dtype=jnp.int32)
    seqs = seqs[source_idx_base]
    motif_features = motif_features[source_idx_base]
    graph_features = graph_features[source_idx_base]
    layer_features = {name: values[source_idx_base] for name, values in layer_features.items()}
    motif_names = [motif_names[i] for i in source_idx_list]

    source_one_hot = jnp.eye(len(motif_names), dtype=jnp.float32)
    state_code_catalog = jnp.concatenate(
        [
            4.0 * source_one_hot,
            motif_features,
            layer_features["L1_occupation"],
            layer_features["L2_transition"],
            layer_features["L3_fft_phase"],
            layer_features["L4_recurrence"],
        ],
        axis=1,
    )

    source_indices = []
    target_indices = []
    pair_names = []
    payload_seqs = []
    for src_idx, src_name in enumerate(motif_names):
        tgt_idx = (src_idx + 1) % len(motif_names)
        source_indices.append(src_idx)
        target_indices.append(tgt_idx)
        pair_names.append(f"{src_name}_then_{motif_names[tgt_idx]}")
        payload_seqs.append(seqs[src_idx])

    src_idx = jnp.array(source_indices, dtype=jnp.int32)
    tgt_idx = jnp.array(target_indices, dtype=jnp.int32)
    payload_seqs = jnp.stack(payload_seqs)

    source_features = state_code_catalog[src_idx]
    target_features = state_code_catalog[tgt_idx]
    delta_state = target_features - source_features
    transform_code = jnp.concatenate([target_features, delta_state], axis=1)
    source_graph = graph_features[src_idx]
    target_graph = graph_features[tgt_idx]
    delta_graph = target_graph - source_graph
    source_layers = {name: values[src_idx] for name, values in layer_features.items()}
    target_layers = {name: values[tgt_idx] for name, values in layer_features.items()}
    delta_l1 = target_layers["L1_occupation"] - source_layers["L1_occupation"]
    delta_l2 = target_layers["L2_transition"] - source_layers["L2_transition"]
    delta_l3 = target_layers["L3_fft_phase"] - source_layers["L3_fft_phase"]
    delta_l4 = target_layers["L4_recurrence"] - source_layers["L4_recurrence"]

    # The second gate is represented in the feature geometry, while the
    # transported sequence remains the source semantic state.
    target_gate_scale = jnp.float32(1.0)
    delta_gate_scale = jnp.float32(1.0)
    pair_features = jnp.concatenate(
        [source_features, target_gate_scale * target_features, delta_gate_scale * delta_state, transform_code],
        axis=1,
    )
    pair_graph_features = jnp.concatenate(
        [source_graph, target_gate_scale * target_graph, delta_gate_scale * delta_graph],
        axis=1,
    )
    pair_layer_features = {
        "L1_occupation": source_layers["L1_occupation"],
        "L2_transition": source_layers["L2_transition"],
        "L3_fft_phase": source_layers["L3_fft_phase"],
        "L4_recurrence": source_layers["L4_recurrence"],
        "L5_transform": jnp.concatenate(
            [
                target_layers["L1_occupation"],
                target_layers["L2_transition"],
                target_layers["L3_fft_phase"],
                target_layers["L4_recurrence"],
                delta_l1,
                delta_l2,
                delta_l3,
                delta_l4,
            ],
            axis=1,
        ),
    }
    transform_info = {
        "sequential_gate": True,
        "pair_names": pair_names,
        "source_indices": src_idx,
        "target_indices": tgt_idx,
        "source_features": source_features,
        "target_features": target_features,
        "transform_code": transform_code,
        "delta_graph": delta_graph,
        "delta_phase": delta_l3,
        "delta_recurrence": delta_l4,
        "motif_feature_catalog": state_code_catalog,
        "source_sequence_catalog": seqs,
        "source_names": motif_names,
        "target_names": motif_names,
        "target_gate_scale": float(target_gate_scale),
        "delta_gate_scale": float(delta_gate_scale),
        "transition_gate": "target_state_plus_delta",
    }
    return payload_seqs, pair_features, pair_graph_features, pair_layer_features, transform_info


def build_image_patch_codec_payload(seq_len, residual_vector=False, patch_limit=16):
    del seq_len
    motif_specs = [
        ("flat", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("h_edge", [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3]),
        ("v_edge", [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3]),
        ("diagonal", [0, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 3, 2, 3, 3, 3]),
        ("checkerboard", [0, 3, 0, 3, 3, 0, 3, 0, 0, 3, 0, 3, 3, 0, 3, 0]),
        ("gradient", [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        ("periodic", [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1]),
        ("noise_burst", [2, 0, 3, 0, 2, 3, 0, 2, 0, 2, 0, 3, 3, 0, 2, 0]),
    ]
    motif_names = [name for name, _ in motif_specs]
    motif_seqs = jnp.stack([jnp.array(values, dtype=jnp.int32) for _, values in motif_specs])
    motif_pixels = motif_seqs.astype(jnp.float32) / 3.0
    motif_features = jnp.stack([motif_feature(seq) for seq in motif_seqs])
    motif_graph_features = jnp.stack([motif_transition_graph(seq) for seq in motif_seqs])
    motif_layer_features_table = build_layer_feature_table(motif_seqs)
    motif_layer_code = jnp.concatenate(
        [
            motif_layer_features_table["L1_occupation"],
            motif_layer_features_table["L2_transition"],
            motif_layer_features_table["L3_fft_phase"],
            motif_layer_features_table["L4_recurrence"],
        ],
        axis=1,
    )
    motif_code_catalog = jnp.concatenate([motif_features, motif_layer_code], axis=1)

    patch_limit = int(patch_limit)
    patch_motif_indices_all = jnp.array([0, 5, 1, 2, 3, 4, 6, 7, 5, 2, 3, 1, 7, 6, 4, 0], dtype=jnp.int32)
    patch_motif_indices = patch_motif_indices_all[:patch_limit]
    payload_seqs = motif_seqs[patch_motif_indices]
    residuals_all = jnp.array(
        [-0.04, -0.02, 0.03, 0.01, -0.01, 0.02, 0.04, -0.03, 0.00, 0.03, -0.02, 0.02, 0.01, -0.04, 0.04, 0.00],
        dtype=jnp.float32,
    )
    residuals = residuals_all[:patch_limit]
    rows = jnp.repeat(jnp.arange(4, dtype=jnp.float32), 4)
    cols = jnp.tile(jnp.arange(4, dtype=jnp.float32), 4)
    h_basis = (rows - jnp.mean(rows)) / (jnp.max(jnp.abs(rows - jnp.mean(rows))) + 1e-8)
    v_basis = (cols - jnp.mean(cols)) / (jnp.max(jnp.abs(cols - jnp.mean(cols))) + 1e-8)
    diag_basis = ((rows + cols) - jnp.mean(rows + cols)) / (jnp.max(jnp.abs((rows + cols) - jnp.mean(rows + cols))) + 1e-8)
    residual_vectors = jnp.stack(
        [
            residuals,
            jnp.array([0.05, -0.04, 0.03, -0.02, 0.04, -0.03, 0.02, -0.05, 0.01, 0.04, -0.02, 0.03, -0.04, 0.02, 0.05, -0.01], dtype=jnp.float32)[:patch_limit],
            jnp.array([0.00, 0.06, -0.04, 0.02, -0.03, 0.05, -0.02, 0.03, 0.04, -0.05, 0.02, -0.01, 0.05, -0.03, 0.01, -0.04], dtype=jnp.float32)[:patch_limit],
            jnp.array([0.02, -0.03, 0.05, -0.04, 0.01, -0.02, 0.04, -0.05, -0.04, 0.03, -0.01, 0.05, 0.00, 0.04, -0.03, 0.02], dtype=jnp.float32)[:patch_limit],
            jnp.array([-0.02, 0.04, -0.03, 0.05, 0.02, -0.04, 0.01, -0.03, 0.03, -0.01, 0.04, -0.02, -0.05, 0.02, -0.04, 0.01], dtype=jnp.float32)[:patch_limit],
        ],
        axis=1,
    )
    motif_patch_pixels = motif_pixels[patch_motif_indices]
    centered_motif = motif_patch_pixels - jnp.mean(motif_patch_pixels, axis=1, keepdims=True)
    if residual_vector:
        patch_pixels = jnp.clip(
            motif_patch_pixels
            + residual_vectors[:, 0:1]
            + residual_vectors[:, 1:2] * centered_motif
            + residual_vectors[:, 2:3] * h_basis[None, :]
            + residual_vectors[:, 3:4] * v_basis[None, :]
            + residual_vectors[:, 4:5] * diag_basis[None, :],
            0.0,
            1.0,
        )
        residual_feature = residual_vectors
    else:
        patch_pixels = jnp.clip(motif_patch_pixels + residuals[:, None], 0.0, 1.0)
        residual_feature = residuals[:, None]
    patch_features = motif_code_catalog[patch_motif_indices]
    position_feature = jax.nn.one_hot(jnp.arange(patch_limit, dtype=jnp.int32), patch_limit, dtype=jnp.float32)
    source_features = jnp.concatenate([patch_features, residual_feature, position_feature], axis=1)
    graph_features = jnp.concatenate(
        [motif_graph_features[patch_motif_indices], residual_feature, position_feature],
        axis=1,
    )
    base_layer_features = build_layer_feature_table(payload_seqs)
    layer_features = {
        "L1_occupation": base_layer_features["L1_occupation"],
        "L2_transition": base_layer_features["L2_transition"],
        "L3_fft_phase": base_layer_features["L3_fft_phase"],
        "L4_recurrence": base_layer_features["L4_recurrence"],
        "L5_transform": jnp.concatenate([residual_feature, position_feature], axis=1),
    }
    transform_info = {
        "image_patch_codec": True,
        "pair_names": [f"patch_{i:02d}_{motif_names[int(patch_motif_indices[i])]}" for i in range(patch_limit)],
        "source_features": source_features,
        "target_features": source_features,
        "transform_code": jnp.concatenate([residual_feature, position_feature], axis=1),
        "delta_graph": graph_features,
        "delta_phase": base_layer_features["L3_fft_phase"],
        "delta_recurrence": base_layer_features["L4_recurrence"],
        "patch_pixels": patch_pixels,
        "motif_pixel_catalog": motif_pixels,
        "motif_code_catalog": motif_code_catalog,
        "patch_motif_indices": patch_motif_indices,
        "residuals": residuals,
        "residual_vectors": residual_vectors,
        "residual_basis_h": h_basis,
        "residual_basis_v": v_basis,
        "residual_basis_diag": diag_basis,
        "residual_vector_codec": bool(residual_vector),
        "fast_codec_probe": bool(patch_limit < 16),
        "source_names": motif_names,
        "patch_grid": [2, 4] if patch_limit == 8 else [4, 4],
        "patch_shape": [4, 4],
    }
    return payload_seqs, source_features, graph_features, layer_features, transform_info


def build_tunneling_kernel(features, barrier_base, barrier_alpha, tunnel_temp):
    diffs = features[:, None, :] - features[None, :, :]
    dist = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-8)
    barrier = barrier_base + barrier_alpha * dist
    kernel = jnp.exp(-barrier / tunnel_temp)
    kernel = kernel.at[jnp.diag_indices(kernel.shape[0])].set(0.0)
    kernel = kernel / (jnp.sum(kernel, axis=1, keepdims=True) + 1e-8)
    return kernel, barrier


def build_node_tunnel_affinity(motif_entropy, motif_features, barrier_base, barrier_alpha, tunnel_temp):
    kernel, _ = build_tunneling_kernel(motif_features, barrier_base, barrier_alpha, tunnel_temp)
    motif_response = jax.nn.softmax(motif_entropy.T, axis=1)
    affinity = motif_response @ kernel @ motif_response.T
    affinity = 0.5 * (affinity + affinity.T)
    affinity = affinity / (jnp.max(affinity) + 1e-8)
    return affinity


def tunnel_eigen_guidance(tunnel_affinity, top_k):
    sym_affinity = 0.5 * (tunnel_affinity + tunnel_affinity.T)
    vals, vecs = jnp.linalg.eigh(sym_affinity)
    idx = jnp.argsort(vals)[-top_k:]
    top_vals = jnp.maximum(vals[idx], 0.0)
    basis = vecs[:, idx]
    projection = basis @ basis.T
    projected_vals = top_vals / (jnp.max(top_vals) + 1e-8)
    mode_support = jnp.sum((basis * basis) * projected_vals[None, :], axis=1)
    mode_support = mode_support / (jnp.max(mode_support) + 1e-8)
    return projection, mode_support, top_vals


def adaptive_tunnel_eigen_guidance(tunnel_affinity, energy_threshold=0.90):
    sym_affinity = 0.5 * (tunnel_affinity + tunnel_affinity.T)
    vals, vecs = jnp.linalg.eigh(sym_affinity)
    vals = jnp.maximum(vals, 0.0)
    order = jnp.argsort(vals)[::-1]
    sorted_vals = vals[order]
    sorted_vecs = vecs[:, order]
    energy = sorted_vals / (jnp.sum(sorted_vals) + 1e-8)
    cumulative = jnp.cumsum(energy)
    keep = jnp.logical_or(cumulative <= energy_threshold, jnp.arange(vals.shape[0]) == 0)
    weights = jnp.where(keep, energy / (jnp.max(energy) + 1e-8), 0.0)
    projection = (sorted_vecs * weights[None, :]) @ sorted_vecs.T
    projection = 0.5 * (projection + projection.T)
    mode_support = jnp.sum((sorted_vecs * sorted_vecs) * weights[None, :], axis=1)
    mode_support = mode_support / (jnp.max(mode_support) + 1e-8)
    return projection, mode_support, sorted_vals * keep


def stationary_distribution(kernel, steps=64):
    p = jnp.ones((kernel.shape[0],), dtype=jnp.float32) / kernel.shape[0]
    for _ in range(steps):
        p = p @ kernel
        p = p / (jnp.sum(p) + 1e-8)
    return p


def build_interval_nodes(seq_len, max_span):
    nodes = []
    for u in range(seq_len):
        upper = min(seq_len, u + max_span)
        for v in range(u + 1, upper + 1):
            nodes.append((u, v))
    return jnp.array(nodes, dtype=jnp.int32)


def interval_entropy(seq, node):
    u, v = int(node[0]), int(node[1])
    window = seq[u:v]
    one_hot = jax.nn.one_hot(window, 4)
    probs = jnp.mean(one_hot, axis=0)
    probs_safe = jnp.where(probs > 1e-8, probs, 1.0)
    symbol_entropy = -jnp.sum(jnp.where(probs > 1e-8, probs * jnp.log(probs_safe), 0.0))

    def transition_entropy():
        pair = one_hot[:-1, :, None] * one_hot[1:, None, :]
        p = jnp.mean(pair, axis=0).reshape(-1)
        p_safe = jnp.where(p > 1e-8, p, 1.0)
        return -jnp.sum(jnp.where(p > 1e-8, p * jnp.log(p_safe), 0.0))

    trans_entropy = jax.lax.cond(window.shape[0] > 1, transition_entropy, lambda: 0.0)
    return symbol_entropy + 0.5 * trans_entropy


def interval_entropy_table(seqs, motif_weights, nodes):
    motif_entropy = []
    for seq in seqs:
        motif_entropy.append(jnp.array([interval_entropy(seq, node) for node in nodes]))
    motif_entropy = jnp.stack(motif_entropy)
    return motif_weights @ motif_entropy, motif_entropy


def motif_interval_entropy_table(seqs, nodes):
    motif_entropy = []
    for seq in seqs:
        motif_entropy.append(jnp.array([interval_entropy(seq, node) for node in nodes]))
    return jnp.stack(motif_entropy)


def crofton_weights(nodes, entropy_values, seq_len):
    grid = jnp.zeros((seq_len + 1, seq_len + 1), dtype=jnp.float32)
    for i, node in enumerate(nodes):
        grid = grid.at[int(node[0]), int(node[1])].set(entropy_values[i])

    weights = []
    for node in nodes:
        u, v = int(node[0]), int(node[1])
        if u + 1 < v and v + 1 <= seq_len:
            val = grid[u, v] - grid[u + 1, v] - grid[u, v + 1] + grid[u + 1, v + 1]
        else:
            val = grid[u, v]
        weights.append(jnp.abs(val))
    weights = jnp.array(weights)
    return weights / (jnp.max(weights) + 1e-8)


def build_kinematic_coupling(
    nodes,
    crofton,
    tunneling_strength,
    ibm_lite=False,
    tunnel_affinity=None,
    top_k_eigenmodes=0,
    top_k_mode="fixed",
    projection_alpha=1.0,
):
    u = nodes[:, 0].astype(jnp.int32)
    v = nodes[:, 1].astype(jnp.int32)
    u1 = u[:, None]
    v1 = v[:, None]
    u2 = u[None, :]
    v2 = v[None, :]

    dist_a = jnp.abs(u1 - u2) + jnp.abs(v1 - v2)
    dist_b = jnp.abs(u1 - v2) + jnp.abs(v1 - u2)
    endpoint_dist = jnp.minimum(dist_a, dist_b).astype(jnp.float32)

    overlap = jnp.maximum(0, jnp.minimum(v1, v2) - jnp.maximum(u1, u2)).astype(jnp.float32)
    span_norm = jnp.maximum((v1 - u1).astype(jnp.float32), (v2 - u2).astype(jnp.float32))
    span_norm = jnp.maximum(span_norm, 1.0)

    allowed = jnp.logical_or(endpoint_dist <= 2.0, overlap > 0.0)
    adj = (overlap + 1.0) / (span_norm + endpoint_dist + 1.0)
    adj = jnp.where(allowed, adj, 0.0)
    adj = adj * (1.0 - jnp.eye(nodes.shape[0], dtype=jnp.float32))
    if ibm_lite:
        idx = jnp.arange(nodes.shape[0])
        nearest_neighbor = jnp.abs(idx[:, None] - idx[None, :]) <= 2
        adj = jnp.where(nearest_neighbor, adj, 0.0)

    mode_support = jnp.zeros((nodes.shape[0],), dtype=jnp.float32)
    eigen_projection = jnp.eye(nodes.shape[0], dtype=jnp.float32)
    if tunnel_affinity is None:
        tunnel_affinity = 0.5 * (crofton[:, None] + crofton[None, :])
    elif top_k_mode == "adaptive_90":
        eigen_projection, mode_support, _ = adaptive_tunnel_eigen_guidance(tunnel_affinity, 0.90)
        projected_adj = eigen_projection @ adj @ eigen_projection
        projected_adj = jnp.maximum(projected_adj, 0.0)
        adj = projection_alpha * projected_adj + (1.0 - projection_alpha) * adj
        adj = adj * (1.0 - jnp.eye(nodes.shape[0], dtype=jnp.float32))
    elif top_k_eigenmodes and top_k_eigenmodes > 0:
        eigen_projection, mode_support, _ = tunnel_eigen_guidance(tunnel_affinity, int(top_k_eigenmodes))
        projected_adj = eigen_projection @ adj @ eigen_projection
        projected_adj = jnp.maximum(projected_adj, 0.0)
        adj = projection_alpha * projected_adj + (1.0 - projection_alpha) * adj
        adj = adj * (1.0 - jnp.eye(nodes.shape[0], dtype=jnp.float32))

    coupling = adj * (1.0 + tunneling_strength * tunnel_affinity)
    coupling = coupling / (jnp.sum(coupling, axis=1, keepdims=True) + 1e-8)
    return coupling, mode_support, eigen_projection


@partial(jax.jit, static_argnums=(5,))
def evolve_dlinoss(x0, omega, coupling, damping, feedback, steps, dt):
    def step(x, _):
        phase_lock = feedback * coupling @ x
        dx = (-damping + 1j * omega) * x + phase_lock
        x_next = x + dt * dx
        x_next = x_next / (jnp.linalg.norm(x_next) + 1e-8)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, jnp.arange(steps))
    return xs


def dlinoss_mode_count(signal, threshold=0.03):
    centered = signal - jnp.mean(signal, axis=0, keepdims=True)
    _, s, _ = jnp.linalg.svd(centered, full_matrices=False)
    return int(jnp.sum(s > s[0] * threshold)) if s.shape[0] else 0


def filament_signature(xs, crofton, top_frac):
    phases = jnp.unwrap(jnp.angle(xs), axis=0)
    centered = phases - jnp.mean(phases, axis=0, keepdims=True)
    cov = centered.T @ centered / max(phases.shape[0] - 1, 1)
    node_strength = jnp.mean(jnp.abs(cov), axis=1) * (1.0 + crofton)
    k = max(2, int(crofton.shape[0] * top_frac))
    top_idx = jnp.argsort(node_strength)[-k:]
    signature = jnp.zeros_like(node_strength)
    signature = signature.at[top_idx].set(node_strength[top_idx])
    signature = signature / (jnp.linalg.norm(signature) + 1e-8)
    return signature


def soft_filament_signature(xs, crofton):
    phases = jnp.unwrap(jnp.angle(xs), axis=0)
    centered = phases - jnp.mean(phases, axis=0, keepdims=True)
    cov = centered.T @ centered / max(phases.shape[0] - 1, 1)
    node_strength = jnp.mean(jnp.abs(cov), axis=1) * (1.0 + crofton)
    signature = jax.nn.softmax(node_strength / (jnp.std(node_strength) + 1e-8))
    return signature / (jnp.linalg.norm(signature) + 1e-8)


def graph_cosine_batch(pred, target):
    import numpy as np

    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return np.sum(pred * target, axis=1)


def graph_cosine_batch_jax(pred, target):
    pred = pred / (jnp.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (jnp.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return jnp.sum(pred * target, axis=1)


def nearest_class_accuracy_jax(pred, prototypes, target_idx):
    chosen = nearest_class_indices_jax(pred, prototypes)
    return jnp.mean((chosen == target_idx).astype(jnp.float32))


def nearest_class_indices_jax(pred, prototypes):
    pred_norm = pred / (jnp.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototypes / (jnp.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    return jnp.argmax(pred_norm @ proto_norm.T, axis=1)


def semantic_mi_proxy_jax(pred, target):
    pred_prob = jax.nn.softmax(pred, axis=1)
    target_prob = jax.nn.softmax(target, axis=1)
    joint = jnp.mean(pred_prob[:, :, None] * target_prob[:, None, :], axis=0)
    px = jnp.sum(joint, axis=1, keepdims=True)
    py = jnp.sum(joint, axis=0, keepdims=True)
    mi = jnp.sum(jnp.where(joint > 1e-8, joint * jnp.log(joint / (px * py + 1e-8) + 1e-8), 0.0))
    return mi / (jnp.log(jnp.float32(pred.shape[1])) + 1e-8)


def ridge_decode(train_x, train_y, test_x, ridge):
    import numpy as np

    xtx = train_x.T @ train_x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    weights = np.linalg.solve(xtx + reg, train_x.T @ train_y)
    return test_x @ weights


def ridge_decode_jax(train_x, train_y, test_x, ridge):
    xtx = train_x.T @ train_x
    reg = ridge * jnp.eye(xtx.shape[0], dtype=train_x.dtype)
    weights = jnp.linalg.solve(xtx + reg, train_x.T @ train_y)
    return test_x @ weights


def filament_metrics(
    xs,
    nodes,
    crofton,
    motif_entropy,
    motif_features,
    motif_graphs,
    motif_weights,
    structured_target_feature,
    structured_target_graph,
    top_frac,
):
    phases = jnp.unwrap(jnp.angle(xs), axis=0)
    phase_spread = jnp.var(phases, axis=1)
    centered = phases - jnp.mean(phases, axis=0, keepdims=True)
    cov = centered.T @ centered / max(phases.shape[0] - 1, 1)
    node_strength = jnp.mean(jnp.abs(cov), axis=1) * (1.0 + crofton)

    k = max(2, int(nodes.shape[0] * top_frac))
    top_idx = jnp.argsort(node_strength)[-k:]
    rt_idx = jnp.argsort(crofton)[-k:]
    overlap = jnp.sum(jnp.isin(top_idx, rt_idx)) / float(k)

    weights = node_strength[top_idx]
    weights = weights / (jnp.sum(weights) + 1e-8)
    motif_scores = jnp.sum(motif_entropy[:, top_idx] * weights[None, :], axis=1)
    motif_scores = jax.nn.softmax(motif_scores)

    reconstructed = motif_scores @ motif_features
    target = motif_weights @ motif_features
    recon_cos = jnp.dot(reconstructed, target) / (
        jnp.linalg.norm(reconstructed) * jnp.linalg.norm(target) + 1e-8
    )
    structured_recon_cos = jnp.dot(reconstructed, structured_target_feature) / (
        jnp.linalg.norm(reconstructed) * jnp.linalg.norm(structured_target_feature) + 1e-8
    )
    reconstructed_graph = motif_scores @ motif_graphs
    graph_cos = jnp.dot(reconstructed_graph, structured_target_graph) / (
        jnp.linalg.norm(reconstructed_graph) * jnp.linalg.norm(structured_target_graph) + 1e-8
    )
    graph_l1 = jnp.mean(jnp.abs(reconstructed_graph - structured_target_graph))

    baseline_weights = crofton[rt_idx]
    baseline_weights = baseline_weights / (jnp.sum(baseline_weights) + 1e-8)
    baseline_scores = jnp.sum(motif_entropy[:, rt_idx] * baseline_weights[None, :], axis=1)
    baseline_scores = jax.nn.softmax(baseline_scores)
    baseline_reconstructed = baseline_scores @ motif_features
    baseline_cos = jnp.dot(baseline_reconstructed, target) / (
        jnp.linalg.norm(baseline_reconstructed) * jnp.linalg.norm(target) + 1e-8
    )
    baseline_structured_cos = jnp.dot(baseline_reconstructed, structured_target_feature) / (
        jnp.linalg.norm(baseline_reconstructed) * jnp.linalg.norm(structured_target_feature) + 1e-8
    )
    baseline_graph = baseline_scores @ motif_graphs
    baseline_graph_cos = jnp.dot(baseline_graph, structured_target_graph) / (
        jnp.linalg.norm(baseline_graph) * jnp.linalg.norm(structured_target_graph) + 1e-8
    )

    crofton_mass = jnp.sum(crofton[top_idx])
    bound = 1.0 - jnp.exp(-crofton_mass / (jnp.sum(crofton) + 1e-8))
    compression_ratio = k / float(nodes.shape[0])
    slope = jnp.polyfit(jnp.arange(phase_spread.shape[0], dtype=jnp.float32), phase_spread, deg=1)[0]

    return {
        "phase_spread_final": float(phase_spread[-1]),
        "phase_spread_slope": float(slope),
        "filament_rt_overlap": float(overlap),
        "semantic_reconstruction_cosine": float(recon_cos),
        "structured_target_cosine": float(structured_recon_cos),
        "structured_transition_graph_cosine": float(graph_cos),
        "structured_transition_graph_l1": float(graph_l1),
        "crofton_baseline_cosine": float(baseline_cos),
        "crofton_baseline_structured_cosine": float(baseline_structured_cos),
        "crofton_baseline_graph_cosine": float(baseline_graph_cos),
        "dlinoss_lift_over_crofton": float(recon_cos - baseline_cos),
        "dlinoss_graph_lift_over_crofton": float(graph_cos - baseline_graph_cos),
        "filament_crofton_bound": float(bound),
        "compression_ratio": float(compression_ratio),
        "dlinoss_K_eff": dlinoss_mode_count(phases),
    }


def random_transition_kernel(key, n_states):
    raw = jax.random.uniform(key, (n_states, n_states), minval=0.05, maxval=1.0)
    raw = raw.at[jnp.diag_indices(n_states)].set(0.0)
    return raw / (jnp.sum(raw, axis=1, keepdims=True) + 1e-8)


def semantic_mixtures(key, n_states, n_samples, alpha):
    raw = jax.random.gamma(key, alpha, shape=(n_samples, n_states))
    return raw / (jnp.sum(raw, axis=1, keepdims=True) + 1e-8)


def recovery_state_weights(args, key, n_states, n_samples, alpha):
    if (
        getattr(args, "aq_1a_tf", False)
        or getattr(args, "aq_1a_tf_s29", False)
        or getattr(args, "aq_1b", False)
        or getattr(args, "aq_1c", False)
        or getattr(args, "aq_seq_0", False)
        or getattr(args, "aq_img_0", False)
        or getattr(args, "aq_img_1", False)
        or getattr(args, "aq_img_1_lite", False)
    ):
        order = jax.random.permutation(key, n_states)
        idx = order[jnp.arange(n_samples) % n_states]
        return jax.nn.one_hot(idx, n_states, dtype=jnp.float32)
    return semantic_mixtures(key, n_states, n_samples, alpha)


def resolve_projection_alpha(ctrl_cfg, noise_level=None):
    if ctrl_cfg.get("adaptive_alpha", False):
        noise = 0.0 if noise_level is None else float(noise_level)
        alpha = float(ctrl_cfg.get("base_alpha", 0.50)) + float(ctrl_cfg.get("adaptive_scale", 0.30)) * noise
        return max(0.0, min(1.0, alpha))
    return float(ctrl_cfg.get("projection_alpha", 0.0))


def build_dynamics_bundle(args, ctrl_cfg, nodes, motif_entropy, motif_features, coupling_entropy, noise_level=None):
    tunnel_affinity = build_node_tunnel_affinity(
        coupling_entropy,
        motif_features,
        args.barrier_base,
        args.barrier_alpha,
        args.tunnel_temp,
    )
    ones = jnp.ones((nodes.shape[0],), dtype=jnp.float32)
    projection_alpha = resolve_projection_alpha(ctrl_cfg, noise_level)
    coupling, mode_support, eigen_projection = build_kinematic_coupling(
        nodes,
        ones,
        ctrl_cfg["tunnel"],
        ctrl_cfg.get("ibm_lite", False),
        tunnel_affinity,
        ctrl_cfg.get("top_k_eigenmodes", 0),
        ctrl_cfg.get("top_k_mode", "fixed"),
        projection_alpha if projection_alpha > 0.0 else 1.0,
    )
    damping = jnp.float32(ctrl_cfg["damping"])
    if ctrl_cfg.get("eigen_damping_scale", False):
        spread = jnp.float32(ctrl_cfg.get("damping_spread", 0.35))
        damping = damping * (1.0 + spread * (0.5 - mode_support))
    return {
        "coupling": coupling,
        "damping": damping,
        "feedback": jnp.float32(ctrl_cfg["feedback"]),
        "eigen_projection": eigen_projection,
        "projection_alpha": jnp.float32(projection_alpha),
        "projection_enabled": projection_alpha > 0.0,
        "adaptive_k_chosen": int(jnp.sum(mode_support > 0.0)),
    }


def signatures_for_weights(
    args,
    weights_batch,
    nodes,
    motif_entropy,
    ctrl_cfg,
    control_name,
    key_control,
    dynamics_bundle=None,
):
    def one_signature(weights, sample_idx):
        entropy_values = weights @ motif_entropy
        crofton = crofton_weights(nodes, entropy_values, args.sequence_length)
        if control_name == "random_crofton":
            crofton = crofton[jax.random.permutation(jax.random.fold_in(key_control, sample_idx), crofton.shape[0])]

        phase_weights = weights
        if control_name == "phase_scramble":
            phase_weights = semantic_mixtures(
                jax.random.fold_in(key_control, 20_000 + sample_idx),
                weights.shape[0],
                1,
                args.mixture_alpha,
            )[0]

        omega = args.omega_scale * (crofton + 0.05 * jnp.log1p(nodes[:, 1] - nodes[:, 0]))
        semantic_phase = motif_entropy.T @ phase_weights
        semantic_phase = semantic_phase / (jnp.std(semantic_phase) + 1e-8)
        if control_name == "entangled_phase_noise":
            semantic_phase = semantic_phase + args.recovery_noise * jax.random.normal(
                jax.random.fold_in(key_control, 30_000 + sample_idx),
                semantic_phase.shape,
            )

        x0 = jnp.sqrt(crofton + 1e-3) * jnp.exp(1j * semantic_phase)
        x0 = x0.astype(jnp.complex64)
        x0 = x0 / (jnp.linalg.norm(x0) + 1e-8)
        if dynamics_bundle is not None and dynamics_bundle.get("projection_enabled", False):
            alpha = dynamics_bundle["projection_alpha"]
            projected_x0 = dynamics_bundle["eigen_projection"] @ x0
            x0 = alpha * projected_x0 + (1.0 - alpha) * x0
            x0 = x0 / (jnp.linalg.norm(x0) + 1e-8)
        coupling = dynamics_bundle["coupling"] if dynamics_bundle is not None else build_kinematic_coupling(
            nodes, crofton, ctrl_cfg["tunnel"], ctrl_cfg.get("ibm_lite", False)
        )[0]
        damping = dynamics_bundle["damping"] if dynamics_bundle is not None else jnp.float32(ctrl_cfg["damping"])
        feedback = dynamics_bundle["feedback"] if dynamics_bundle is not None else jnp.float32(ctrl_cfg["feedback"])
        xs = evolve_dlinoss(
            x0,
            omega,
            coupling,
            damping,
            feedback,
            int(args.dlinoss_steps),
            jnp.float32(args.dt),
        )
        if ctrl_cfg.get("soft_decode", False):
            return soft_filament_signature(xs, crofton)
        return filament_signature(xs, crofton, args.top_frac)

    sample_ids = jnp.arange(weights_batch.shape[0], dtype=jnp.int32)
    return jax.vmap(one_signature)(weights_batch, sample_ids)


def evaluate_semantic_recovery(
    args,
    seed_key,
    ctrl_cfg,
    nodes,
    seqs,
    motif_features,
    graph_features,
    layer_features,
    noise_level,
    transform_info=None,
):
    key_train, key_test, key_noise, key_shuffle, key_block, key_block_inner, key_crofton, key_phase = jax.random.split(seed_key, 8)
    train_w = recovery_state_weights(args, key_train, seqs.shape[0], args.recovery_train_states, args.mixture_alpha)
    test_w = recovery_state_weights(args, key_test, seqs.shape[0], args.recovery_test_states, args.mixture_alpha)
    target_graphs = test_w @ graph_features

    clean_entropy = motif_interval_entropy_table(seqs, nodes)
    resolved_alpha = resolve_projection_alpha(ctrl_cfg, noise_level)
    dynamics_bundle = build_dynamics_bundle(args, ctrl_cfg, nodes, clean_entropy, motif_features, clean_entropy, noise_level)
    train_x = signatures_for_weights(
        args, train_w, nodes, clean_entropy, ctrl_cfg, "structured", key_train, dynamics_bundle
    )
    clean_test_x = signatures_for_weights(
        args, test_w, nodes, clean_entropy, ctrl_cfg, "structured", key_test, dynamics_bundle
    )
    old_noise = args.recovery_noise
    args.recovery_noise = noise_level
    noisy_test_x = signatures_for_weights(
        args, test_w, nodes, clean_entropy, ctrl_cfg, "entangled_phase_noise", key_noise, dynamics_bundle
    )
    fast_codec_probe = bool(transform_info is not None and transform_info.get("fast_codec_probe", False))
    if fast_codec_probe:
        phase_scramble_x = noisy_test_x
        topology_shuffle_x = noisy_test_x
        block_permute_x = noisy_test_x
        block_permute_inner_x = noisy_test_x
        crofton_random_x = noisy_test_x
    else:
        phase_scramble_x = signatures_for_weights(
            args, test_w, nodes, clean_entropy, ctrl_cfg, "phase_scramble", key_phase, dynamics_bundle
        )
        shuffled_seqs = apply_sequence_control(seqs, "markov_shuffle", key_shuffle, args.control_block_size)
        shuffled_entropy = motif_interval_entropy_table(shuffled_seqs, nodes)
        topology_shuffle_x = signatures_for_weights(
            args, test_w, nodes, shuffled_entropy, ctrl_cfg, "structured", key_shuffle, dynamics_bundle
        )
        block_seqs = apply_sequence_control(seqs, "block_permute_size2", key_block, args.control_block_size)
        block_entropy = motif_interval_entropy_table(block_seqs, nodes)
        block_permute_x = signatures_for_weights(
            args, test_w, nodes, block_entropy, ctrl_cfg, "structured", key_block, dynamics_bundle
        )
        block_inner_seqs = apply_sequence_control(seqs, "block_permute_size2_within_random", key_block_inner, args.control_block_size)
        block_inner_entropy = motif_interval_entropy_table(block_inner_seqs, nodes)
        block_permute_inner_x = signatures_for_weights(
            args, test_w, nodes, block_inner_entropy, ctrl_cfg, "structured", key_block_inner, dynamics_bundle
        )
        crofton_random_x = signatures_for_weights(
            args, test_w, nodes, clean_entropy, ctrl_cfg, "random_crofton", key_crofton, dynamics_bundle
        )
    args.recovery_noise = old_noise

    train_y = train_w @ graph_features
    clean_pred = ridge_decode_jax(train_x, train_y, clean_test_x, args.decoder_ridge)
    noisy_pred = ridge_decode_jax(train_x, train_y, noisy_test_x, args.decoder_ridge)
    topology_pred = ridge_decode_jax(train_x, train_y, topology_shuffle_x, args.decoder_ridge)
    block_pred = ridge_decode_jax(train_x, train_y, block_permute_x, args.decoder_ridge)
    block_inner_pred = ridge_decode_jax(train_x, train_y, block_permute_inner_x, args.decoder_ridge)
    crofton_pred = ridge_decode_jax(train_x, train_y, crofton_random_x, args.decoder_ridge)
    phase_pred = ridge_decode_jax(train_x, train_y, phase_scramble_x, args.decoder_ridge)

    clean_cos = graph_cosine_batch_jax(clean_pred, target_graphs)
    noisy_cos = graph_cosine_batch_jax(noisy_pred, target_graphs)
    topology_cos = graph_cosine_batch_jax(topology_pred, target_graphs)
    block_cos = graph_cosine_batch_jax(block_pred, target_graphs)
    block_inner_cos = graph_cosine_batch_jax(block_inner_pred, target_graphs)
    crofton_cos = graph_cosine_batch_jax(crofton_pred, target_graphs)
    phase_cos = graph_cosine_batch_jax(phase_pred, target_graphs)

    mean_graph = jnp.mean(train_y, axis=0, keepdims=True)
    mean_cos = graph_cosine_batch_jax(jnp.repeat(mean_graph, target_graphs.shape[0], axis=0), target_graphs)
    frame_consistency = 1.0 - jnp.mean(jnp.abs(clean_cos - noisy_cos))
    control_margin = jnp.mean(noisy_cos) - jnp.mean(block_inner_cos)
    minimum_control_margin = jnp.min(
        jnp.array(
            [
                jnp.mean(noisy_cos) - jnp.mean(topology_cos),
                jnp.mean(noisy_cos) - jnp.mean(block_cos),
                control_margin,
                jnp.mean(noisy_cos) - jnp.mean(crofton_cos),
                jnp.mean(noisy_cos) - jnp.mean(phase_cos),
            ]
        )
    )
    folded_score = (
        args.score_alpha * jnp.mean(noisy_cos)
        + args.score_beta * control_margin
        + args.score_gamma * frame_consistency
        - args.score_delta * ctrl_cfg.get("gate_cost", 0.0)
    )
    semantic_mi = semantic_mi_proxy_jax(noisy_pred, target_graphs)
    topology_match = jnp.mean(noisy_cos) - jnp.mean(topology_cos)

    layer_metrics = {}
    full_layer_features = jnp.concatenate([layer_features[name] for name in layer_features.keys()], axis=1)
    full_train_y = train_w @ full_layer_features
    full_target_y = test_w @ full_layer_features
    full_clean_pred = ridge_decode_jax(train_x, full_train_y, clean_test_x, args.decoder_ridge)
    full_noisy_pred = ridge_decode_jax(train_x, full_train_y, noisy_test_x, args.decoder_ridge)
    full_clean_cos = graph_cosine_batch_jax(full_clean_pred, full_target_y)
    full_noisy_cos = graph_cosine_batch_jax(full_noisy_pred, full_target_y)

    start = 0
    target_energy = []
    noisy_energy = []
    for layer_name, values in layer_features.items():
        width = values.shape[1]
        end = start + width
        clean_layer_pred = full_clean_pred[:, start:end]
        noisy_layer_pred = full_noisy_pred[:, start:end]
        target_layer = full_target_y[:, start:end]
        clean_layer_cos = graph_cosine_batch_jax(clean_layer_pred, target_layer)
        noisy_layer_cos = graph_cosine_batch_jax(noisy_layer_pred, target_layer)
        layer_metrics[layer_name] = {
            "clean_cosine": float(jnp.mean(clean_layer_cos)),
            "noisy_cosine": float(jnp.mean(noisy_layer_cos)),
            "delta_noisy_minus_clean": float(jnp.mean(noisy_layer_cos) - jnp.mean(clean_layer_cos)),
        }
        target_energy.append(jnp.linalg.norm(target_layer, axis=1))
        noisy_energy.append(jnp.linalg.norm(noisy_layer_pred, axis=1))
        start = end

    target_energy = jnp.stack(target_energy, axis=1)
    noisy_energy = jnp.stack(noisy_energy, axis=1)
    target_energy = target_energy / (jnp.sum(target_energy, axis=1, keepdims=True) + 1e-8)
    noisy_energy = noisy_energy / (jnp.sum(noisy_energy, axis=1, keepdims=True) + 1e-8)
    inter_layer_leakage_proxy = 0.5 * jnp.mean(jnp.sum(jnp.abs(noisy_energy - target_energy), axis=1))
    transform_metrics = {}
    if transform_info is not None:
        source_y = train_w @ transform_info["source_features"]
        target_y = train_w @ transform_info["target_features"]
        transform_y = train_w @ transform_info["transform_code"]
        delta_graph_y = train_w @ transform_info["delta_graph"]
        delta_phase_y = train_w @ transform_info["delta_phase"]
        delta_recurrence_y = train_w @ transform_info["delta_recurrence"]

        source_target = test_w @ transform_info["source_features"]
        target_target = test_w @ transform_info["target_features"]
        transform_target = test_w @ transform_info["transform_code"]
        delta_graph_target = test_w @ transform_info["delta_graph"]
        delta_phase_target = test_w @ transform_info["delta_phase"]
        delta_recurrence_target = test_w @ transform_info["delta_recurrence"]

        source_pred = ridge_decode_jax(train_x, source_y, noisy_test_x, args.decoder_ridge)
        target_pred = ridge_decode_jax(train_x, target_y, noisy_test_x, args.decoder_ridge)
        transform_pred = ridge_decode_jax(train_x, transform_y, noisy_test_x, args.decoder_ridge)
        delta_graph_pred = ridge_decode_jax(train_x, delta_graph_y, noisy_test_x, args.decoder_ridge)
        delta_phase_pred = ridge_decode_jax(train_x, delta_phase_y, noisy_test_x, args.decoder_ridge)
        delta_recurrence_pred = ridge_decode_jax(train_x, delta_recurrence_y, noisy_test_x, args.decoder_ridge)

        pair_idx = jnp.argmax(test_w, axis=1)
        if transform_info.get("image_patch_codec", False):
            patch_pixel_y = train_w @ transform_info["patch_pixels"]
            patch_pixel_target = test_w @ transform_info["patch_pixels"]
            pixel_pred = ridge_decode_jax(train_x, patch_pixel_y, noisy_test_x, args.decoder_ridge)
            pixel_pred = jnp.clip(pixel_pred, 0.0, 1.0)
            pixel_mse = jnp.mean((pixel_pred - patch_pixel_target) ** 2)
            pixel_psnr = -10.0 * jnp.log10(pixel_mse + 1e-8)
            motif_chosen = nearest_class_indices_jax(source_pred[:, : transform_info["motif_code_catalog"].shape[1]], transform_info["motif_code_catalog"])
            true_motif = transform_info["patch_motif_indices"][pair_idx]
            motif_exact = jnp.mean((motif_chosen == true_motif).astype(jnp.float32))
            if transform_info.get("residual_vector_codec", False):
                residual_pred = transform_pred[:, :5]
                residual_target = transform_info["residual_vectors"][pair_idx]
                residual_mae = jnp.mean(jnp.abs(residual_pred - residual_target))
                chosen_catalog = transform_info["motif_pixel_catalog"][motif_chosen]
                chosen_centered = chosen_catalog - jnp.mean(chosen_catalog, axis=1, keepdims=True)
                catalog_recon = (
                    chosen_catalog
                    + residual_pred[:, 0:1]
                    + residual_pred[:, 1:2] * chosen_centered
                    + residual_pred[:, 2:3] * transform_info["residual_basis_h"][None, :]
                    + residual_pred[:, 3:4] * transform_info["residual_basis_v"][None, :]
                    + residual_pred[:, 4:5] * transform_info["residual_basis_diag"][None, :]
                )
            else:
                residual_pred = transform_pred[:, 0]
                residual_target = transform_info["residuals"][pair_idx]
                residual_mae = jnp.mean(jnp.abs(residual_pred - residual_target))
                catalog_recon = transform_info["motif_pixel_catalog"][motif_chosen] + residual_pred[:, None]
            catalog_recon = jnp.clip(catalog_recon, 0.0, 1.0)
            catalog_mse = jnp.mean((catalog_recon - patch_pixel_target) ** 2)
            catalog_psnr = -10.0 * jnp.log10(catalog_mse + 1e-8)
            patch_luma_target = jnp.mean(patch_pixel_target, axis=1)
            patch_luma_pred = jnp.mean(pixel_pred, axis=1)
            transform_metrics = {
                "image_patch_motif_recovery": float(motif_exact),
                "image_patch_pixel_mse": float(pixel_mse),
                "image_patch_psnr": float(pixel_psnr),
                "image_patch_catalog_psnr": float(catalog_psnr),
                "image_patch_residual_mae": float(residual_mae),
                "image_patch_luma_mae": float(jnp.mean(jnp.abs(patch_luma_pred - patch_luma_target))),
                "source_motif_recovery": float(motif_exact),
                "target_motif_recovery": float(motif_exact),
                "derived_next_accuracy": 0.0,
                "transform_class_recovery": float(motif_exact),
                "transform_exact_match": float(motif_exact),
                "L5_transform_recovery": float(jnp.mean(graph_cosine_batch_jax(transform_pred, transform_target))),
                "delta_graph_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_graph_pred, delta_graph_target))),
                "delta_phase_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_phase_pred, delta_phase_target))),
                "delta_recurrence_match": float(jnp.mean(graph_cosine_batch_jax(delta_recurrence_pred, delta_recurrence_target))),
                "patch_true_indices": pair_idx.tolist(),
                "patch_true_motif_indices": true_motif.tolist(),
                "patch_pred_motif_indices": motif_chosen.tolist(),
                "patch_luma_target": patch_luma_target.tolist(),
                "patch_luma_pred": patch_luma_pred.tolist(),
                "source_names": transform_info.get("source_names", []),
            }
        elif transform_info.get("operator_conditioned", False):
            source_idx = transform_info["source_indices"][pair_idx]
            operator_idx = transform_info["operator_indices"][pair_idx]
            n_operators = int(transform_info["n_operators"])
            source_chosen = nearest_class_indices_jax(source_pred, transform_info["motif_feature_catalog"])
            operator_chosen = nearest_class_indices_jax(transform_pred, transform_info["operator_code_catalog"])
            direct_target_chosen = nearest_class_indices_jax(target_pred, transform_info["target_feature_catalog"])
            derived_pair_idx = source_chosen * n_operators + operator_chosen
            derived_target = transform_info["target_feature_catalog"][derived_pair_idx]
            true_target = transform_info["target_feature_catalog"][pair_idx]
            swapped_operator_pair = source_chosen * n_operators + ((operator_chosen + 1) % n_operators)
            swapped_source_pair = ((source_chosen + 1) % int(transform_info["n_sources"])) * n_operators + operator_chosen
            source_exact = jnp.mean((source_chosen == source_idx).astype(jnp.float32))
            operator_exact = jnp.mean((operator_chosen == operator_idx).astype(jnp.float32))
            derived_target_accuracy = jnp.mean((derived_pair_idx == pair_idx).astype(jnp.float32))
            direct_target_leakage = jnp.mean((direct_target_chosen == pair_idx).astype(jnp.float32))
            operator_application_error = 1.0 - jnp.mean(graph_cosine_batch_jax(derived_target, true_target))
            source_confusion = (
                jnp.zeros((int(transform_info["n_sources"]), int(transform_info["n_sources"])), dtype=jnp.int32)
                .at[source_idx, source_chosen]
                .add(1)
            )
            operator_confusion = (
                jnp.zeros((n_operators, n_operators), dtype=jnp.int32)
                .at[operator_idx, operator_chosen]
                .add(1)
            )
            composition_success = (derived_pair_idx == pair_idx).astype(jnp.float32)
            per_operator_success = jnp.stack(
                [
                    jnp.sum(jnp.where(operator_idx == i, composition_success, 0.0))
                    / (jnp.sum((operator_idx == i).astype(jnp.float32)) + 1e-8)
                    for i in range(n_operators)
                ]
            )

            transform_metrics = {
                "L5_transform_recovery": float(jnp.mean(graph_cosine_batch_jax(transform_pred, transform_target))),
                "operator_recovery": float(operator_exact),
                "derived_target_accuracy": float(derived_target_accuracy),
                "operator_application_error": float(operator_application_error),
                "source_recovery": float(source_exact),
                "target_direct_leakage_test": float(direct_target_leakage),
                "operator_swap_consistency": float(jnp.mean((swapped_operator_pair != derived_pair_idx).astype(jnp.float32))),
                "source_swap_consistency": float(jnp.mean((swapped_source_pair != derived_pair_idx).astype(jnp.float32))),
                "operator_invariance_score": float(jnp.min(per_operator_success)),
                "source_confusion": source_confusion.tolist(),
                "operator_confusion": operator_confusion.tolist(),
                "source_true_indices": source_idx.tolist(),
                "source_pred_indices": source_chosen.tolist(),
                "operator_true_indices": operator_idx.tolist(),
                "operator_pred_indices": operator_chosen.tolist(),
                "pair_true_indices": pair_idx.tolist(),
                "pair_derived_indices": derived_pair_idx.tolist(),
                "source_names": transform_info.get("source_names", []),
                "operator_names": transform_info.get("operator_names", []),
                "transform_exact_match": float(derived_target_accuracy),
                "source_motif_recovery": float(source_exact),
                "target_motif_recovery": float(direct_target_leakage),
                "transform_class_recovery": float(operator_exact),
                "delta_graph_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_graph_pred, delta_graph_target))),
                "delta_phase_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_phase_pred, delta_phase_target))),
                "delta_recurrence_match": float(jnp.mean(graph_cosine_batch_jax(delta_recurrence_pred, delta_recurrence_target))),
            }
        else:
            source_idx = transform_info["source_indices"][pair_idx]
            target_idx = transform_info["target_indices"][pair_idx]
            source_chosen = nearest_class_indices_jax(source_pred, transform_info["motif_feature_catalog"])
            target_chosen = nearest_class_indices_jax(target_pred, transform_info["motif_feature_catalog"])
            transform_chosen = nearest_class_indices_jax(transform_pred, transform_info["transform_code"])
            derived_target_chosen = (source_chosen + 1) % int(transform_info.get("n_sources", transform_info["motif_feature_catalog"].shape[0]))
            source_exact = jnp.mean((source_chosen == source_idx).astype(jnp.float32))
            target_exact = jnp.mean((target_chosen == target_idx).astype(jnp.float32))
            transform_exact = jnp.mean((transform_chosen == pair_idx).astype(jnp.float32))
            derived_target_exact = jnp.mean((derived_target_chosen == target_idx).astype(jnp.float32))
            transform_exact_match = jnp.mean(
                jnp.logical_and(
                    jnp.logical_and(source_chosen == source_idx, target_chosen == target_idx),
                    transform_chosen == pair_idx,
                ).astype(jnp.float32)
            )

            transform_metrics = {
                "L5_transform_recovery": float(jnp.mean(graph_cosine_batch_jax(transform_pred, transform_target))),
                "transform_exact_match": float(transform_exact_match),
                "source_motif_recovery": float(source_exact),
                "target_motif_recovery": float(target_exact),
                "derived_next_accuracy": float(derived_target_exact),
                "transform_class_recovery": float(transform_exact),
                "sequential_delta_cosine": float(jnp.mean(graph_cosine_batch_jax(transform_pred, transform_target))),
                "delta_graph_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_graph_pred, delta_graph_target))),
                "delta_phase_cosine": float(jnp.mean(graph_cosine_batch_jax(delta_phase_pred, delta_phase_target))),
                "delta_recurrence_match": float(jnp.mean(graph_cosine_batch_jax(delta_recurrence_pred, delta_recurrence_target))),
                "source_true_indices": source_idx.tolist(),
                "source_pred_indices": source_chosen.tolist(),
                "target_true_indices": target_idx.tolist(),
                "target_pred_indices": target_chosen.tolist(),
                "derived_target_indices": derived_target_chosen.tolist(),
                "transition_true_indices": pair_idx.tolist(),
                "transition_pred_indices": transform_chosen.tolist(),
                "source_names": transform_info.get("source_names", []),
                "target_names": transform_info.get("target_names", transform_info.get("source_names", [])),
            }

    return {
        "clean_recovery_graph_cosine": float(jnp.mean(clean_cos)),
        "noisy_recovery_graph_cosine": float(jnp.mean(noisy_cos)),
        "topology_shuffle_graph_cosine": float(jnp.mean(topology_cos)),
        "block_permute_graph_cosine": float(jnp.mean(block_cos)),
        "block_permute_within_random_graph_cosine": float(jnp.mean(block_inner_cos)),
        "crofton_random_graph_cosine": float(jnp.mean(crofton_cos)),
        "phase_scramble_graph_cosine": float(jnp.mean(phase_cos)),
        "mean_baseline_graph_cosine": float(jnp.mean(mean_cos)),
        "noisy_lift_over_mean": float(jnp.mean(noisy_cos) - jnp.mean(mean_cos)),
        "noisy_lift_over_topology_shuffle": float(jnp.mean(noisy_cos) - jnp.mean(topology_cos)),
        "noisy_lift_over_block_permute": float(jnp.mean(noisy_cos) - jnp.mean(block_cos)),
        "noisy_lift_over_block_permute_within_random": float(control_margin),
        "noisy_lift_over_crofton_random": float(jnp.mean(noisy_cos) - jnp.mean(crofton_cos)),
        "noisy_lift_over_phase_scramble": float(jnp.mean(noisy_cos) - jnp.mean(phase_cos)),
        "worst_control_margin": float(control_margin),
        "minimum_hard_control_lift": float(control_margin),
        "minimum_all_control_lift": float(minimum_control_margin),
        "frame_consistency": float(frame_consistency),
        "folded_recovery_score": float(folded_score),
        "semantic_MI": float(semantic_mi),
        "topology_match": float(topology_match),
        "pareto_margin": float(control_margin),
        "pareto_fidelity": float(jnp.mean(noisy_cos)),
        "pareto_gate_cost": float(ctrl_cfg.get("gate_cost", 0.0)),
        "leaderboard_lane": ctrl_cfg.get("lane", "core"),
        "gate_cost": float(ctrl_cfg.get("gate_cost", 0.0)),
        "projection_alpha": float(resolved_alpha),
        "resolved_projection_alpha": float(resolved_alpha),
        "adaptive_alpha": bool(ctrl_cfg.get("adaptive_alpha", False)),
        "adaptive_k_chosen": int(dynamics_bundle.get("adaptive_k_chosen", 0)),
        "recovery_noise": float(noise_level),
        "recovery_train_states": int(args.recovery_train_states),
        "recovery_test_states": int(args.recovery_test_states),
        "untransported_ridge_sanity_cosine": float(jnp.mean(full_clean_cos)),
        "full_layer_noisy_cosine": float(jnp.mean(full_noisy_cos)),
        "inter_layer_leakage_proxy": float(inter_layer_leakage_proxy),
        "layer_recovery": layer_metrics,
        "transform_field": transform_metrics,
    }


def ap_cell_debug_metadata(args, ctrl_name, ctrl_cfg, noise_idx, noise_level, seed, depth, key_ctrl_idx, recovery_foldin):
    if getattr(args, "aq_1a_tf", False) or getattr(args, "aq_1a_tf_s29", False) or getattr(args, "aq_seq_0", False):
        mode_name = "--aq-1a-tf"
        if getattr(args, "aq_seq_0", False):
            mode_name = "--aq-seq-0"
    elif getattr(args, "aq_0b", False):
        mode_name = "--aq-0b"
    elif getattr(args, "aq_0", False):
        mode_name = "--aq-0"
    elif getattr(args, "ap_u", False):
        mode_name = "--ap-u"
    else:
        mode_name = "standard"
    return {
        "controller_name": ctrl_name,
        "noise": float(noise_level),
        "seed": int(seed),
        "recovery_seed": int(seed),
        "recovery_foldin": int(recovery_foldin),
        "recovery_key_slot": int(key_ctrl_idx),
        "adversary_seed": int(seed),
        "adversary_foldins": {
            "markov_shuffle": 3,
            "block_permute_size2": 4,
            "block_permute_size2_within_random": 5,
            "random_crofton": 6,
            "phase_scramble": 7,
        },
        "block_size": int(args.control_block_size),
        "within_block_random": True,
        "score_alpha": float(args.score_alpha),
        "score_beta": float(args.score_beta),
        "score_gamma": float(args.score_gamma),
        "score_delta": float(args.score_delta),
        "hard_controls": [
            "topology_shuffle",
            "block_permute_size2",
            "block_permute_size2_within_random",
            "crofton_random",
            "phase_scramble",
        ],
        "min_lift_aggregation": "worst_control_margin = noisy_cosine - block_permute_size2_within_random_cosine",
        "top_k_mode": ctrl_cfg.get("top_k_mode", "fixed"),
        "damping_spread": float(ctrl_cfg.get("damping_spread", 0.0)),
        "key_lock_source": ctrl_cfg.get("key_lock_source", ""),
        "git_commit": getattr(args, "git_commit", "unknown"),
        "mode": mode_name,
        "stage": ctrl_cfg.get("stage", "core"),
        "noise_index": int(noise_idx),
        "depth": int(depth),
    }


def evaluate_heldout_transport(
    args,
    seed_key,
    control_name,
    ctrl_cfg,
    nodes,
    seqs,
    graph_features,
):
    import numpy as np

    key_train, key_test, key_control = jax.random.split(seed_key, 3)
    train_w = semantic_mixtures(key_train, seqs.shape[0], args.n_train_states, args.mixture_alpha)
    test_w = semantic_mixtures(key_test, seqs.shape[0], args.n_test_states, args.mixture_alpha)

    seqs_trial = seqs
    if control_name in ("markov_shuffle", "block_permute"):
        seqs_trial = apply_sequence_control(seqs, control_name, key_control, args.control_block_size)

    motif_entropy = motif_interval_entropy_table(seqs_trial, nodes)

    def build_signature(weights, idx):
        entropy_values = weights @ motif_entropy
        crofton = crofton_weights(nodes, entropy_values, args.sequence_length)
        if control_name == "random_crofton":
            crofton = crofton[jax.random.permutation(jax.random.fold_in(key_control, idx), crofton.shape[0])]

        phase_weights = weights
        if control_name == "phase_scramble":
            phase_weights = semantic_mixtures(
                jax.random.fold_in(key_control, 10_000 + idx),
                seqs.shape[0],
                1,
                args.mixture_alpha,
            )[0]

        omega = args.omega_scale * (crofton + 0.05 * jnp.log1p(nodes[:, 1] - nodes[:, 0]))
        semantic_phase = motif_entropy.T @ phase_weights
        semantic_phase = semantic_phase / (jnp.std(semantic_phase) + 1e-8)
        x0 = jnp.sqrt(crofton + 1e-3) * jnp.exp(1j * semantic_phase)
        x0 = x0.astype(jnp.complex64)
        x0 = x0 / (jnp.linalg.norm(x0) + 1e-8)
        coupling = build_kinematic_coupling(nodes, crofton, ctrl_cfg["tunnel"], ctrl_cfg.get("ibm_lite", False))[0]
        xs = evolve_dlinoss(
            x0,
            omega,
            coupling,
            jnp.float32(ctrl_cfg["damping"]),
            jnp.float32(ctrl_cfg["feedback"]),
            int(args.dlinoss_steps),
            jnp.float32(args.dt),
        )
        return filament_signature(xs, crofton, args.top_frac)

    train_x = []
    test_x = []
    for i, weights in enumerate(train_w):
        train_x.append(np.array(build_signature(weights, i), dtype=np.float32))
    for i, weights in enumerate(test_w):
        test_x.append(np.array(build_signature(weights, 1000 + i), dtype=np.float32))

    train_x = np.stack(train_x)
    test_x = np.stack(test_x)
    train_y = np.array(train_w @ graph_features, dtype=np.float32)
    test_y = np.array(test_w @ graph_features, dtype=np.float32)

    pred = ridge_decode(train_x, train_y, test_x, args.decoder_ridge)
    cosine = graph_cosine_batch(pred, test_y)

    mean_train = np.mean(train_y, axis=0, keepdims=True)
    mean_pred = np.repeat(mean_train, test_y.shape[0], axis=0)
    mean_cosine = graph_cosine_batch(mean_pred, test_y)

    random_perm = np.array(jax.random.permutation(key_control, test_y.shape[0]))
    shuffled_cosine = graph_cosine_batch(pred, test_y[random_perm])

    return {
        "heldout_graph_cosine": float(np.mean(cosine)),
        "heldout_graph_cosine_std": float(np.std(cosine)),
        "mean_baseline_graph_cosine": float(np.mean(mean_cosine)),
        "shuffled_target_graph_cosine": float(np.mean(shuffled_cosine)),
        "heldout_lift_over_mean": float(np.mean(cosine) - np.mean(mean_cosine)),
        "heldout_lift_over_shuffled": float(np.mean(cosine) - np.mean(shuffled_cosine)),
        "n_train_states": int(args.n_train_states),
        "n_test_states": int(args.n_test_states),
    }


def experiment_name(args):
    if getattr(args, "aq_img_1", False):
        return "semantic_image_patch_codec_aq_img_1"
    if getattr(args, "aq_img_1_lite", False):
        return "semantic_image_patch_codec_aq_img_1_lite"
    if getattr(args, "aq_img_0", False):
        return "semantic_image_patch_codec_aq_img_0"
    if getattr(args, "aq_seq_0", False):
        return "semantic_sequence_operator_gate_aq_seq_0"
    if getattr(args, "aq_1c", False):
        return "semantic_operator_identifiability_aq1c"
    if getattr(args, "aq_1b", False):
        return "semantic_operator_conditioned_aq1b"
    if getattr(args, "aq_1a_tf", False) or getattr(args, "aq_1a_tf_s29", False):
        return "semantic_transform_field_aq1a_tf"
    if getattr(args, "aq_0b", False):
        return "semantic_subspace_transport_aq0b"
    if getattr(args, "aq_0", False):
        return "semantic_subspace_transport_aq0"
    if getattr(args, "ap_u", False):
        return "tunnel_eigen_reference_locked_ap_u"
    if getattr(args, "ap_t", False):
        return "tunnel_eigen_adaptive_alpha_ap_t"
    if getattr(args, "ap_s_fidelity", False):
        return "tunnel_eigen_stability_fidelity_ap_s_alpha_bracket"
    if getattr(args, "ap_s_mech", False):
        return "tunnel_eigen_stability_fidelity_ap_s_mech"
    if getattr(args, "ap_s", False):
        return "tunnel_eigen_stability_fidelity_ap_s"
    if getattr(args, "ap_r", False):
        return "tunnel_eigen_residual_folded_recovery_ap_r"
    if getattr(args, "ap_lite", False):
        return "tunnel_eigen_guided_folded_recovery_ap_lite"
    return "tunnel_eigen_guided_folded_recovery"


def run_program_ap(args):
    global SHUTDOWN_FLAG
    t0 = time.time()
    _, backend_info = select_backend(require_tpu=getattr(args, "require_tpu", False))

    print("=" * 88)
    print("Program AP - Tunnel-Eigen Guided Folded Recovery")
    print(f"Backend: {backend_info['backend']} | TPU visible: {backend_info['has_tpu']}")
    print("=" * 88)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_ap_summary.json")

    use_aq_payload = (
        getattr(args, "aq_0b", False)
        or getattr(args, "aq_1a_tf", False)
        or getattr(args, "aq_1a_tf_s29", False)
        or getattr(args, "aq_1b", False)
        or getattr(args, "aq_1c", False)
        or getattr(args, "aq_seq_0", False)
        or getattr(args, "aq_img_0", False)
        or getattr(args, "aq_img_1", False)
    )
    motifs, _ = build_semantic_motifs(args.sequence_length, aq_payload=use_aq_payload)
    motif_names = list(motifs.keys())
    seqs = jnp.stack([motifs[name] for name in motif_names])
    features = jnp.stack([motif_feature(seq) for seq in seqs])
    graph_features = jnp.stack([motif_transition_graph(seq) for seq in seqs])
    layer_features = build_layer_feature_table(seqs)
    transform_info = None
    if getattr(args, "aq_img_0", False) or getattr(args, "aq_img_1", False) or getattr(args, "aq_img_1_lite", False):
        seqs, features, graph_features, layer_features, transform_info = build_image_patch_codec_payload(
            args.sequence_length,
            residual_vector=getattr(args, "aq_img_1", False) or getattr(args, "aq_img_1_lite", False),
            patch_limit=8 if getattr(args, "aq_img_1_lite", False) else 16,
        )
        source_sample = seqs[0]
        print("payload_seqs sample:", jax.device_get(seqs[0]).tolist())
        print("image motif sample:", jax.device_get(source_sample).tolist())
        print("payload/source match:", bool(jax.device_get(jnp.array_equal(seqs[0], source_sample))))
        motif_names = transform_info["pair_names"]
    elif getattr(args, "aq_seq_0", False):
        seqs, features, graph_features, layer_features, transform_info = build_sequential_state_gate_payload(
            motif_names,
            seqs,
            features,
            graph_features,
            layer_features,
        )
        source_sample = transform_info["source_sequence_catalog"][int(transform_info["source_indices"][0])]
        print("payload_seqs sample:", jax.device_get(seqs[0]).tolist())
        print("source motif sample:", jax.device_get(source_sample).tolist())
        print("payload/source match:", bool(jax.device_get(jnp.array_equal(seqs[0], source_sample))))
        motif_names = transform_info["pair_names"]
    elif getattr(args, "aq_1c", False):
        seqs, features, graph_features, layer_features, transform_info = build_operator_identifiability_payload(
            motif_names,
            seqs,
            features,
            graph_features,
            layer_features,
        )
        source_sample = transform_info["source_sequence_catalog"][int(transform_info["source_indices"][0])]
        print("payload_seqs sample:", jax.device_get(seqs[0]).tolist())
        print("source motif sample:", jax.device_get(source_sample).tolist())
        print("payload/source match:", bool(jax.device_get(jnp.array_equal(seqs[0], source_sample))))
        motif_names = transform_info["pair_names"]
    elif getattr(args, "aq_1b", False):
        seqs, features, graph_features, layer_features, transform_info = build_operator_conditioned_payload(
            motif_names,
            seqs,
            features,
            graph_features,
            layer_features,
        )
        motif_names = transform_info["pair_names"]
    elif getattr(args, "aq_1a_tf", False) or getattr(args, "aq_1a_tf_s29", False):
        seqs, features, graph_features, layer_features, transform_info = build_transform_field_payload(
            motif_names,
            seqs,
            features,
            graph_features,
            layer_features,
        )
        motif_names = transform_info["pair_names"]
    tunneling_kernel_base, barrier_base = build_tunneling_kernel(
        features, args.barrier_base, args.barrier_alpha, args.tunnel_temp
    )
    structured_target_weights = stationary_distribution(tunneling_kernel_base)
    structured_target_feature = structured_target_weights @ features
    structured_target_graph = structured_target_weights @ graph_features
    controls = ["structured"]
    if args.include_controls:
        controls.extend(["markov_shuffle", "block_permute_size2", "block_permute_size2_within_random", "phase_scramble", "random_crofton"])

    records = []
    controllers = {
        "free": {"damping": args.damping, "feedback": 0.00, "tunnel": 0.00, "gate_cost": 0.00},
        "filament_stabilized": {"damping": args.damping * 0.7, "feedback": 0.18, "tunnel": 1.00, "gate_cost": 0.20},
        "filament_stabilized_v1": {
            "damping": args.damping * 0.70,
            "feedback": 0.18,
            "tunnel": 1.00,
            "gate_cost": 0.20,
            "lane": "mechanism",
        },
        "filament_stabilized_v2": {
            "damping": args.damping * 0.70,
            "feedback": 0.21,
            "tunnel": 1.15,
            "gate_cost": 0.20,
        },
        "ibm_lite_folded": {
            "damping": args.damping * 0.78,
            "feedback": 0.13,
            "tunnel": 0.70,
            "gate_cost": 0.12,
            "ibm_lite": True,
        },
        "tunnel_eigen_unfolded": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_eigenmodes": 4,
            "eigen_damping_scale": True,
            "gate_cost": 0.25,
        },
        "tunnel_eigen_residual": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
        },
        "tunnel_eigen_residual_APR": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "mechanism",
        },
        "APR_reference_lock": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "reproducibility",
            "stage": "stage0_reference_lock",
            "key_ctrl_idx": 1,
        },
        "APR_APT_path": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "reproducibility",
            "stage": "stage0_apt_path",
            "key_ctrl_idx": 1,
            "key_lock_source": "APR_reference_lock",
        },
        "tunnel_eigen_residual_alpha045": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.45,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "fidelity",
        },
        "tunnel_eigen_residual_alpha055": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "fidelity",
        },
        "tunnel_eigen_residual_alpha065": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.65,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "fidelity",
        },
        "tunnel_eigen_residual_adaptive": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "adaptive_alpha": True,
            "base_alpha": 0.50,
            "adaptive_scale": 0.30,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "adaptive",
        },
        "tunnel_eigen_residual_adaptive_v2_shifted": {
            "damping": args.damping * 0.65,
            "feedback": 0.20,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "adaptive_alpha": True,
            "base_alpha": 0.56,
            "adaptive_scale": 0.22,
            "eigen_damping_scale": True,
            "damping_spread": 0.60,
            "gate_cost": 0.25,
            "lane": "adaptive",
            "stage": "stage1_variant",
        },
        "tunnel_eigen_softdecode": {
            "damping": args.damping * 0.65,
            "feedback": 0.18,
            "tunnel": 1.00,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.55,
            "eigen_damping_scale": True,
            "damping_spread": 0.50,
            "soft_decode": True,
            "gate_cost": 0.23,
            "lane": "fidelity",
        },
        "ibm_lite_residual": {
            "damping": args.damping * 0.72,
            "feedback": 0.14,
            "tunnel": 0.80,
            "top_k_mode": "adaptive_90",
            "projection_alpha": 0.45,
            "eigen_damping_scale": True,
            "damping_spread": 0.45,
            "ibm_lite": True,
            "gate_cost": 0.14,
            "lane": "hardware",
        },
    }
    if getattr(args, "ap_lite", False):
        controllers = {
            name: controllers[name]
            for name in ("filament_stabilized", "tunnel_eigen_unfolded")
        }
    if getattr(args, "ap_r", False):
        controllers = {
            name: controllers[name]
            for name in ("filament_stabilized_v2", "tunnel_eigen_residual")
        }
    if getattr(args, "ap_s", False):
        controllers = {
            name: controllers[name]
            for name in (
                "filament_stabilized_v1",
                "tunnel_eigen_residual_APR",
                "tunnel_eigen_residual_alpha045",
                "tunnel_eigen_residual_alpha065",
                "tunnel_eigen_softdecode",
                "ibm_lite_residual",
            )
        }
    if getattr(args, "ap_s_mech", False):
        controllers = {
            name: controllers[name]
            for name in ("filament_stabilized_v1", "tunnel_eigen_residual_APR")
        }
    if getattr(args, "ap_s_fidelity", False):
        controllers = {
            name: controllers[name]
            for name in (
                "tunnel_eigen_residual_alpha045",
                "tunnel_eigen_residual_APR",
                "tunnel_eigen_residual_alpha065",
            )
        }
    if getattr(args, "ap_t", False):
        controllers = {
            name: controllers[name]
            for name in (
                "tunnel_eigen_residual_APR",
                "tunnel_eigen_residual_alpha065",
                "tunnel_eigen_residual_adaptive",
            )
        }
    if getattr(args, "ap_u", False):
        controllers = {
            name: controllers[name]
            for name in (
                "APR_reference_lock",
                "APR_APT_path",
                "tunnel_eigen_residual_alpha065",
                "tunnel_eigen_residual_adaptive",
                "tunnel_eigen_residual_adaptive_v2_shifted",
            )
        }
        controllers["tunnel_eigen_residual_alpha065"]["stage"] = "stage1_variant"
        controllers["tunnel_eigen_residual_alpha065"]["key_ctrl_idx"] = 2
        controllers["tunnel_eigen_residual_adaptive"]["stage"] = "stage1_variant"
        controllers["tunnel_eigen_residual_adaptive"]["key_ctrl_idx"] = 3
        controllers["tunnel_eigen_residual_adaptive_v2_shifted"]["key_ctrl_idx"] = 4
    if getattr(args, "aq_0", False):
        keep = ["free", "tunnel_eigen_residual_alpha065"]
        if getattr(args, "aq0_include_adaptive_v2", False):
            keep.append("tunnel_eigen_residual_adaptive_v2_shifted")
        controllers = {name: controllers[name] for name in keep}
    if getattr(args, "aq_0b", False):
        keep = ["free", "tunnel_eigen_residual_alpha055", "tunnel_eigen_residual_alpha065"]
        controllers = {name: controllers[name] for name in keep}
    if (
        getattr(args, "aq_1a_tf", False)
        or getattr(args, "aq_1a_tf_s29", False)
        or getattr(args, "aq_1b", False)
    ):
        keep = ["free", "tunnel_eigen_residual_alpha055", "tunnel_eigen_residual_alpha065"]
        controllers = {name: controllers[name] for name in keep}
    if (
        getattr(args, "aq_1c", False)
        or getattr(args, "aq_seq_0", False)
        or getattr(args, "aq_img_0", False)
        or getattr(args, "aq_img_1", False)
        or getattr(args, "aq_img_1_lite", False)
    ):
        controllers = {"free": controllers["free"]}

    for depth in args.depth_sweep:
        max_span = min(args.sequence_length, max(2, args.base_span * (2 ** int(depth))))
        nodes = build_interval_nodes(args.sequence_length, max_span)
        print(f"\nDepth={depth} nodes={nodes.shape[0]} max_span={max_span}")

        motif_entropy_cache = {
            "structured": motif_interval_entropy_table(seqs, nodes),
        }
        depth_record = {
            "mera_depth": int(depth),
            "max_span": int(max_span),
            "n_kinematic_nodes": int(nodes.shape[0]),
            "mean_barrier": float(jnp.mean(barrier_base)),
            "trials": [],
            "heldout_transport": {},
            "semantic_recovery": {},
        }

        for seed in args.seeds:
            seed_key = jax.random.PRNGKey(int(seed))
            for control_idx, control_name in enumerate(controls):
                key_control = jax.random.fold_in(seed_key, control_idx)
                seqs_trial = seqs
                tunneling_kernel = tunneling_kernel_base
                if control_name in ("markov_shuffle", "block_permute", "block_permute_size2", "block_permute_size2_within_random"):
                    seqs_trial = apply_sequence_control(seqs, control_name, key_control, args.control_block_size)
                    if control_name not in motif_entropy_cache:
                        motif_entropy_cache[control_name] = motif_interval_entropy_table(seqs_trial, nodes)

                motif_weights = stationary_distribution(tunneling_kernel)
                motif_entropy = motif_entropy_cache.get(control_name, motif_entropy_cache["structured"])
                entropy_values = motif_weights @ motif_entropy
                crofton = crofton_weights(nodes, entropy_values, args.sequence_length)
                if control_name == "random_crofton":
                    crofton = crofton[jax.random.permutation(key_control, crofton.shape[0])]

                omega = args.omega_scale * (crofton + 0.05 * jnp.log1p(nodes[:, 1] - nodes[:, 0]))
                semantic_phase = motif_entropy.T @ motif_weights
                semantic_phase = semantic_phase / (jnp.std(semantic_phase) + 1e-8)
                x0 = jnp.sqrt(crofton + 1e-3) * jnp.exp(1j * semantic_phase)
                x0 = x0.astype(jnp.complex64)
                x0 = x0 / (jnp.linalg.norm(x0) + 1e-8)

                trial = {
                    "seed": int(seed),
                    "control": control_name,
                    "motif_weights": {name: float(motif_weights[i]) for i, name in enumerate(motif_names)},
                    "controllers": {},
                }

                for ctrl_name, cfg in controllers.items():
                    tunnel_affinity = build_node_tunnel_affinity(
                        motif_entropy_cache["structured"],
                        features,
                        args.barrier_base,
                        args.barrier_alpha,
                        args.tunnel_temp,
                    )
                    coupling, mode_support, eigen_projection = build_kinematic_coupling(
                        nodes,
                        crofton,
                        cfg["tunnel"],
                        cfg.get("ibm_lite", False),
                        tunnel_affinity,
                        cfg.get("top_k_eigenmodes", 0),
                        cfg.get("top_k_mode", "fixed"),
                        cfg.get("projection_alpha", 1.0),
                    )
                    damping = jnp.float32(cfg["damping"])
                    if cfg.get("eigen_damping_scale", False):
                        spread = jnp.float32(cfg.get("damping_spread", 0.35))
                        damping = damping * (1.0 + spread * (0.5 - mode_support))
                    if cfg.get("projection_alpha", 0.0) > 0.0:
                        alpha = jnp.float32(cfg.get("projection_alpha", 0.0))
                        projected_x0 = eigen_projection @ x0
                        ctrl_x0 = alpha * projected_x0 + (1.0 - alpha) * x0
                        ctrl_x0 = ctrl_x0 / (jnp.linalg.norm(ctrl_x0) + 1e-8)
                    else:
                        ctrl_x0 = x0
                    xs = evolve_dlinoss(
                        ctrl_x0,
                        omega,
                        coupling,
                        damping,
                        jnp.float32(cfg["feedback"]),
                        int(args.dlinoss_steps),
                        jnp.float32(args.dt),
                    )
                    metrics = filament_metrics(
                        xs,
                        nodes,
                        crofton,
                        motif_entropy,
                        features,
                        graph_features,
                        motif_weights,
                        structured_target_feature,
                        structured_target_graph,
                        args.top_frac,
                    )
                    trial["controllers"][ctrl_name] = metrics
                    print(
                        f"  seed={seed:>3} {control_name:>17} {ctrl_name:>21} -> "
                        f"Recon:{metrics['semantic_reconstruction_cosine']:.4f} "
                        f"Graph:{metrics['structured_transition_graph_cosine']:.4f} "
                        f"Lift:{metrics['dlinoss_lift_over_crofton']:+.4f} "
                        f"Overlap:{metrics['filament_rt_overlap']:.3f} "
                        f"K:{metrics['dlinoss_K_eff']}"
                    )

                    if SHUTDOWN_FLAG:
                        print("[WATCHDOG] Exiting cleanly due to preemption.")
                        return

                depth_record["trials"].append(trial)
                partial = {
                    "experiment": "tunnel_eigen_guided_folded_recovery",
                    "motifs": motif_names,
                    "controls": controls,
                    "records": records + [depth_record],
                    "runtime_s": round(time.time() - t0, 1),
                }
                with open(summary_path, "w") as f:
                    json.dump(partial, f, indent=2)

        if args.run_heldout_transport:
            print("\n  Held-out compressed topology transport:")
            heldout_controls = controls if args.heldout_include_controls else ["structured"]
            for control_idx, control_name in enumerate(heldout_controls):
                depth_record["heldout_transport"][control_name] = {}
                for ctrl_idx, (ctrl_name, cfg) in enumerate(controllers.items()):
                    heldout_key = jax.random.fold_in(jax.random.PRNGKey(int(args.heldout_seed)), depth * 100 + control_idx * 10 + ctrl_idx)
                    metrics = evaluate_heldout_transport(
                        args,
                        heldout_key,
                        control_name,
                        cfg,
                        nodes,
                        seqs,
                        graph_features,
                    )
                    depth_record["heldout_transport"][control_name][ctrl_name] = metrics
                    print(
                        f"    {control_name:>17} {ctrl_name:>21} -> "
                        f"HeldoutGraph:{metrics['heldout_graph_cosine']:.4f} "
                        f"MeanLift:{metrics['heldout_lift_over_mean']:+.4f} "
                        f"ShuffleLift:{metrics['heldout_lift_over_shuffled']:+.4f}"
                    )

        if args.run_semantic_recovery:
            print("\n  Semantic folded-field recovery:")
            ap_u_stage0_passed = None
            ap_u_calibration = []
            controller_items = list(controllers.items())
            if getattr(args, "ap_u", False):
                stage0_items = [(name, cfg) for name, cfg in controller_items if cfg.get("stage", "").startswith("stage0")]
                stage1_items = [(name, cfg) for name, cfg in controller_items if cfg.get("stage") == "stage1_variant"]
                print("    [AP-U Stage 0] reference-lock calibration")
                active_stages = [("stage0", stage0_items)]
            else:
                stage1_items = []
                active_stages = [("standard", controller_items)]

            for stage_name, stage_items in active_stages:
                ap_u_stage0_failed_early = False
                for noise_idx, noise_level in enumerate(args.recovery_noise_sweep):
                    depth_record["semantic_recovery"].setdefault(str(noise_level), {})
                    print(f"    noise={noise_level:.2f}")
                    for seed in args.recovery_seeds:
                        seed_key = jax.random.PRNGKey(int(seed))
                        depth_record["semantic_recovery"][str(noise_level)].setdefault(str(seed), {})
                        print(f"      seed={seed}")
                        for ctrl_idx, (ctrl_name, cfg) in enumerate(stage_items):
                            key_ctrl_idx = int(cfg.get("key_ctrl_idx", ctrl_idx))
                            recovery_foldin = depth * 1000 + noise_idx * 100 + key_ctrl_idx
                            recovery_key = jax.random.fold_in(seed_key, recovery_foldin)
                            metrics = evaluate_semantic_recovery(
                                args,
                                recovery_key,
                                cfg,
                                nodes,
                                seqs,
                                features,
                                graph_features,
                                layer_features,
                                noise_level,
                                transform_info,
                            )
                            metrics.update(
                                ap_cell_debug_metadata(
                                    args,
                                    ctrl_name,
                                    cfg,
                                    noise_idx,
                                    noise_level,
                                    seed,
                                    depth,
                                    key_ctrl_idx,
                                    recovery_foldin,
                                )
                            )
                            if getattr(args, "ap_u", False) and stage_name == "stage0":
                                ref_margin = AP_S_FIDELITY_APR_REFERENCE[round(float(noise_level), 2)][int(seed)]
                                delta = float(metrics["worst_control_margin"] - ref_margin)
                                metrics["ap_s_reference_margin"] = float(ref_margin)
                                metrics["ap_s_reference_delta"] = delta
                                metrics["ap_s_reference_within_tolerance"] = abs(delta) <= AP_U_REFERENCE_TOLERANCE
                                if abs(delta) > AP_U_REFERENCE_TOLERANCE:
                                    ap_u_stage0_failed_early = True
                                ap_u_calibration.append(
                                    {
                                        "controller": ctrl_name,
                                        "noise": float(noise_level),
                                        "seed": int(seed),
                                        "margin": float(metrics["worst_control_margin"]),
                                        "reference_margin": float(ref_margin),
                                        "delta": delta,
                                        "within_tolerance": abs(delta) <= AP_U_REFERENCE_TOLERANCE,
                                    }
                                )
                            depth_record["semantic_recovery"][str(noise_level)][str(seed)][ctrl_name] = metrics
                            print(
                                f"        {ctrl_name:>23} -> "
                                f"Margin:{metrics['worst_control_margin']:+.4f} "
                                f"Score:{metrics['folded_recovery_score']:.4f} "
                                f"Noisy:{metrics['noisy_recovery_graph_cosine']:.4f} "
                                f"Block:{metrics['noisy_lift_over_block_permute']:+.4f} "
                                f"Topo:{metrics['noisy_lift_over_topology_shuffle']:+.4f}"
                            )
                        if ap_u_stage0_failed_early:
                            print("      [AP-U Stage 0] calibration mismatch detected; stopping calibration early")
                            break
                    if ap_u_stage0_failed_early:
                        break
                if getattr(args, "ap_u", False) and stage_name == "stage0":
                    ap_u_stage0_passed = all(item["within_tolerance"] for item in ap_u_calibration)
                    depth_record["ap_u_calibration"] = {
                        "reference_source": "AP-S-Fidelity APR margins",
                        "tolerance": AP_U_REFERENCE_TOLERANCE,
                        "passed": bool(ap_u_stage0_passed),
                        "stopped_early": bool(ap_u_stage0_failed_early),
                        "cells": ap_u_calibration,
                    }
                    if ap_u_stage0_passed:
                        print("    [AP-U Stage 0] PASS: running Stage 1 variants")
                        active_stages.append(("stage1", stage1_items))
                    else:
                        print("    [AP-U Stage 0] FAIL: skipping Stage 1 variant interpretation")

                if getattr(args, "ap_u", False) and stage_name == "stage1":
                    promotion = {}
                    for ctrl_name, _ in stage1_items:
                        cells = []
                        for noise_level in args.recovery_noise_sweep:
                            for seed in args.recovery_seeds:
                                ctrl_map = depth_record["semantic_recovery"][str(noise_level)][str(seed)]
                                metrics = ctrl_map[ctrl_name]
                                apr_metrics = ctrl_map["APR_reference_lock"]
                                alpha065_metrics = ctrl_map.get("tunnel_eigen_residual_alpha065")
                                cells.append(
                                    {
                                        "noise": float(noise_level),
                                        "seed": int(seed),
                                        "margin": float(metrics["worst_control_margin"]),
                                        "noisy_cosine": float(metrics["noisy_recovery_graph_cosine"]),
                                        "beats_apr": bool(
                                            metrics["worst_control_margin"] > apr_metrics["worst_control_margin"]
                                        ),
                                        "beats_alpha065": bool(
                                            alpha065_metrics is not None
                                            and metrics["worst_control_margin"] > alpha065_metrics["worst_control_margin"]
                                        ),
                                    }
                                )
                        positive_cells = sum(cell["margin"] > 0.0 for cell in cells)
                        beats_apr_cells = sum(cell["beats_apr"] for cell in cells)
                        beats_alpha065_cells = sum(cell["beats_alpha065"] for cell in cells)
                        min_noisy = min(cell["noisy_cosine"] for cell in cells)
                        promotion[ctrl_name] = {
                            "positive_cells": int(positive_cells),
                            "beats_apr_cells": int(beats_apr_cells),
                            "beats_alpha065_cells": int(beats_alpha065_cells),
                            "min_noisy_cosine": float(min_noisy),
                            "promoted": bool(
                                positive_cells == 6
                                and beats_apr_cells >= 4
                                and beats_alpha065_cells >= 3
                                and min_noisy >= 0.80
                            ),
                            "cells": cells,
                        }
                    depth_record["ap_u_promotion"] = promotion
                    depth_record["ap_u_promotion_rule"] = {
                        "stage0_required": True,
                        "margin_positive_cells_required": 6,
                        "beats_apr_cells_required": 4,
                        "beats_alpha065_cells_required": 3,
                        "minimum_noisy_cosine": 0.80,
                    }

                partial = {
                    "experiment": experiment_name(args),
                    "motifs": motif_names,
                    "parameters": vars(args),
                    "records": records + [depth_record],
                    "runtime_s": round(time.time() - t0, 1),
                    "partial": True,
                }
                with open(summary_path, "w") as f:
                    json.dump(partial, f, indent=2)

        records.append(depth_record)

    out = {
        "experiment": experiment_name(args),
        "scientific_thread": [
            "Recovery basin volume from OAT/Dicke work",
            "Semantic motifs from Program AM",
            "Tunneling-shaped basin transitions",
            "MERA kinematic interval filaments",
            "D-LinOSS damped mode reconstruction",
            "Tunnel-eigen guided unfolding and IBM-lite hardware constraints",
        ],
        "motifs": motif_names,
        "parameters": vars(args),
        "records": records,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 112)
    print(f"{'depth':>5} {'control':>12} {'ctrl':>21} {'recon':>8} {'graph':>8} {'g_lift':>8} {'overlap':>8} {'K_eff':>6} {'comp':>7}")
    print("-" * 112)
    for rec in records:
        for control_name in controls:
            for ctrl in controllers.keys():
                vals = [
                    trial["controllers"][ctrl]
                    for trial in rec["trials"]
                    if trial["control"] == control_name and ctrl in trial["controllers"]
                ]
                if not vals:
                    continue
                recon = float(np.mean([m["semantic_reconstruction_cosine"] for m in vals]))
                graph = float(np.mean([m["structured_transition_graph_cosine"] for m in vals]))
                graph_lift = float(np.mean([m["dlinoss_graph_lift_over_crofton"] for m in vals]))
                overlap = float(np.mean([m["filament_rt_overlap"] for m in vals]))
                k_eff = float(np.mean([m["dlinoss_K_eff"] for m in vals]))
                comp = float(np.mean([m["compression_ratio"] for m in vals]))
                print(
                    f"{rec['mera_depth']:>5} {control_name[:12]:>12} {ctrl:>21} "
                    f"{recon:>8.4f} {graph:>8.4f} {graph_lift:>8.4f} {overlap:>8.3f} {k_eff:>6.2f} {comp:>7.3f}"
                )
    print("=" * 112)
    if args.run_heldout_transport:
        print("\n" + "=" * 112)
        print(f"{'depth':>5} {'control':>12} {'ctrl':>21} {'heldout':>8} {'mean_l':>8} {'shuf_l':>8}")
        print("-" * 112)
        for rec in records:
            for control_name, ctrl_map in rec.get("heldout_transport", {}).items():
                for ctrl, metrics in ctrl_map.items():
                    print(
                        f"{rec['mera_depth']:>5} {control_name[:12]:>12} {ctrl:>21} "
                        f"{metrics['heldout_graph_cosine']:>8.4f} "
                        f"{metrics['heldout_lift_over_mean']:>8.4f} "
                        f"{metrics['heldout_lift_over_shuffled']:>8.4f}"
                    )
        print("=" * 112)

    if args.run_semantic_recovery:
        print("\n" + "=" * 112)
        print(f"{'depth':>5} {'noise':>6} {'seed':>5} {'ctrl':>23} {'margin':>8} {'score':>8} {'noisy':>8} {'block_l':>8} {'topo_l':>8}")
        print("-" * 112)
        for rec in records:
            for noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                for seed, ctrl_map in seed_map.items():
                    for ctrl, metrics in ctrl_map.items():
                        print(
                            f"{rec['mera_depth']:>5} {float(noise_level):>6.2f} {int(seed):>5} {ctrl:>23} "
                            f"{metrics['worst_control_margin']:>8.4f} "
                            f"{metrics['folded_recovery_score']:>8.4f} "
                            f"{metrics['noisy_recovery_graph_cosine']:>8.4f} "
                            f"{metrics['noisy_lift_over_block_permute']:>8.4f} "
                            f"{metrics['noisy_lift_over_topology_shuffle']:>8.4f}"
                        )
        print("=" * 112)
        aq_l5_mode = (
            getattr(args, "aq_1a_tf", False)
            or getattr(args, "aq_1a_tf_s29", False)
            or getattr(args, "aq_1b", False)
            or getattr(args, "aq_1c", False)
            or getattr(args, "aq_seq_0", False)
            or getattr(args, "aq_img_0", False)
            or getattr(args, "aq_img_1", False)
            or getattr(args, "aq_img_1_lite", False)
        )
        if getattr(args, "aq_0b", False) or aq_l5_mode:
            print("\n" + "=" * 112)
            if aq_l5_mode:
                print(
                    f"{'depth':>5} {'noise':>6} {'seed':>5} {'ctrl':>23} {'sanity':>8} "
                    f"{'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5':>8} {'leak':>8}"
                )
            else:
                print(
                    f"{'depth':>5} {'seed':>5} {'ctrl':>23} {'sanity':>8} "
                    f"{'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'leak':>8}"
                )
            print("-" * 112)
            for rec in records:
                for noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for seed, ctrl_map in seed_map.items():
                        for ctrl, metrics in ctrl_map.items():
                            layers = metrics.get("layer_recovery", {})
                            prefix = (
                                f"{rec['mera_depth']:>5} {float(noise_level):>6.2f} {int(seed):>5} {ctrl:>23} "
                                if aq_l5_mode
                                else f"{rec['mera_depth']:>5} {int(seed):>5} {ctrl:>23} "
                            )
                            row = (
                                f"{prefix}"
                                f"{metrics.get('untransported_ridge_sanity_cosine', 0.0):>8.4f} "
                                f"{layers.get('L1_occupation', {}).get('noisy_cosine', 0.0):>8.4f} "
                                f"{layers.get('L2_transition', {}).get('noisy_cosine', 0.0):>8.4f} "
                                f"{layers.get('L3_fft_phase', {}).get('noisy_cosine', 0.0):>8.4f} "
                                f"{layers.get('L4_recurrence', {}).get('noisy_cosine', 0.0):>8.4f} "
                            )
                            if aq_l5_mode:
                                row += f"{layers.get('L5_transform', {}).get('noisy_cosine', 0.0):>8.4f} "
                            row += f"{metrics.get('inter_layer_leakage_proxy', 0.0):>8.4f}"
                            print(row)
            print("=" * 112)
        if (
            getattr(args, "aq_1a_tf", False)
            or getattr(args, "aq_1a_tf_s29", False)
            or getattr(args, "aq_seq_0", False)
            or getattr(args, "aq_img_0", False)
            or getattr(args, "aq_img_1", False)
            or getattr(args, "aq_img_1_lite", False)
        ):
            print("\n" + "=" * 112)
            print(
                f"{'depth':>5} {'noise':>6} {'seed':>5} {'ctrl':>23} "
                f"{'L5':>8} {'exact':>8} {'src':>8} {'tgt':>8} {'derived':>8} {'class':>8} {'dG':>8} {'dP':>8} {'dR':>8}"
            )
            print("-" * 112)
            for rec in records:
                for noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for seed, ctrl_map in seed_map.items():
                        for ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            print(
                                f"{rec['mera_depth']:>5} {float(noise_level):>6.2f} {int(seed):>5} {ctrl:>23} "
                                f"{tf.get('L5_transform_recovery', 0.0):>8.4f} "
                                f"{tf.get('transform_exact_match', 0.0):>8.4f} "
                                f"{tf.get('source_motif_recovery', 0.0):>8.4f} "
                                f"{tf.get('target_motif_recovery', 0.0):>8.4f} "
                                f"{tf.get('derived_next_accuracy', 0.0):>8.4f} "
                                f"{tf.get('transform_class_recovery', 0.0):>8.4f} "
                                f"{tf.get('delta_graph_cosine', 0.0):>8.4f} "
                                f"{tf.get('delta_phase_cosine', 0.0):>8.4f} "
                                f"{tf.get('delta_recurrence_match', 0.0):>8.4f}"
                            )
            print("=" * 112)
        if getattr(args, "aq_img_0", False) or getattr(args, "aq_img_1", False) or getattr(args, "aq_img_1_lite", False):
            print("\n" + "=" * 112)
            if getattr(args, "aq_img_1_lite", False):
                title = "AQ-IMG-1-LITE vector-residual image patch codec readout"
            else:
                title = "AQ-IMG-1 vector-residual image patch codec readout" if getattr(args, "aq_img_1", False) else "AQ-IMG-0 image patch codec readout"
            print(f"{title:^112}")
            for rec in records:
                for _noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for _seed, ctrl_map in seed_map.items():
                        for _ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            print(
                                f"motif={tf.get('image_patch_motif_recovery', 0.0):.4f} "
                                f"pixel_psnr={tf.get('image_patch_psnr', 0.0):.4f} "
                                f"catalog_psnr={tf.get('image_patch_catalog_psnr', 0.0):.4f} "
                                f"residual_mae={tf.get('image_patch_residual_mae', 0.0):.4f} "
                                f"luma_mae={tf.get('image_patch_luma_mae', 0.0):.4f}"
                            )
            print(f"{'row':>4} {'true_motif':>18} {'pred_motif':>18} {'target_luma':>12} {'pred_luma':>12}")
            print("-" * 112)
            for rec in records:
                for _noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for _seed, ctrl_map in seed_map.items():
                        for _ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            names = tf.get("source_names", [])
                            true_motifs = tf.get("patch_true_motif_indices", [])
                            pred_motifs = tf.get("patch_pred_motif_indices", [])
                            target_luma = tf.get("patch_luma_target", [])
                            pred_luma = tf.get("patch_luma_pred", [])
                            for row, (motif_t, motif_p, luma_t, luma_p) in enumerate(
                                zip(true_motifs, pred_motifs, target_luma, pred_luma)
                            ):
                                true_name = names[motif_t] if motif_t < len(names) else str(motif_t)
                                pred_name = names[motif_p] if motif_p < len(names) else str(motif_p)
                                print(f"{row:>4} {true_name:>18} {pred_name:>18} {float(luma_t):>12.4f} {float(luma_p):>12.4f}")
            print("=" * 112)
        if getattr(args, "aq_seq_0", False):
            print("\n" + "=" * 112)
            print(f"{'AQ-SEQ-0 decoded sequential state readout':^112}")
            print(
                f"{'row':>4} {'true_source':>18} {'pred_source':>18} "
                f"{'true_next':>18} {'pred_next':>18} {'derived_next':>18} {'transition_ok':>13}"
            )
            print("-" * 112)
            for rec in records:
                for _noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for _seed, ctrl_map in seed_map.items():
                        for _ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            source_names = tf.get("source_names", [])
                            target_names = tf.get("target_names", source_names)
                            true_sources = tf.get("source_true_indices", [])
                            pred_sources = tf.get("source_pred_indices", [])
                            true_targets = tf.get("target_true_indices", [])
                            pred_targets = tf.get("target_pred_indices", [])
                            derived_targets = tf.get("derived_target_indices", [])
                            true_transitions = tf.get("transition_true_indices", [])
                            pred_transitions = tf.get("transition_pred_indices", [])
                            for row, (src_t, src_p, tgt_t, tgt_p, tgt_d, tr_t, tr_p) in enumerate(
                                zip(true_sources, pred_sources, true_targets, pred_targets, derived_targets, true_transitions, pred_transitions)
                            ):
                                true_source = source_names[src_t] if src_t < len(source_names) else str(src_t)
                                pred_source = source_names[src_p] if src_p < len(source_names) else str(src_p)
                                true_target = target_names[tgt_t] if tgt_t < len(target_names) else str(tgt_t)
                                pred_target = target_names[tgt_p] if tgt_p < len(target_names) else str(tgt_p)
                                derived_target = target_names[tgt_d] if tgt_d < len(target_names) else str(tgt_d)
                                print(
                                    f"{row:>4} {true_source:>18} {pred_source:>18} "
                                    f"{true_target:>18} {pred_target:>18} {derived_target:>18} {str(tr_t == tr_p):>13}"
                                )
            print("=" * 112)
        if getattr(args, "aq_1b", False) or getattr(args, "aq_1c", False):
            print("\n" + "=" * 112)
            print(
                f"{'depth':>5} {'noise':>6} {'seed':>5} {'ctrl':>23} "
                f"{'L5':>8} {'op':>8} {'derived':>8} {'leak':>8} {'src':>8} {'appErr':>8} {'opSwap':>8} {'srcSwap':>8}"
            )
            print("-" * 112)
            for rec in records:
                for noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for seed, ctrl_map in seed_map.items():
                        for ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            print(
                                f"{rec['mera_depth']:>5} {float(noise_level):>6.2f} {int(seed):>5} {ctrl:>23} "
                                f"{tf.get('L5_transform_recovery', 0.0):>8.4f} "
                                f"{tf.get('operator_recovery', 0.0):>8.4f} "
                                f"{tf.get('derived_target_accuracy', 0.0):>8.4f} "
                                f"{tf.get('target_direct_leakage_test', 0.0):>8.4f} "
                                f"{tf.get('source_recovery', 0.0):>8.4f} "
                                f"{tf.get('operator_application_error', 0.0):>8.4f} "
                                f"{tf.get('operator_swap_consistency', 0.0):>8.4f} "
                                f"{tf.get('source_swap_consistency', 0.0):>8.4f}"
                            )
            print("=" * 112)
        if getattr(args, "aq_1c", False):
            print("\n" + "=" * 112)
            print(f"{'AQ-1c decoded semantic state readout':^112}")
            print(
                f"{'row':>4} {'true_source':>18} {'pred_source':>18} "
                f"{'true_op':>18} {'pred_op':>18} {'derived_ok':>10}"
            )
            print("-" * 112)
            for rec in records:
                for _noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for _seed, ctrl_map in seed_map.items():
                        for _ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            source_names = tf.get("source_names", [])
                            operator_names = tf.get("operator_names", [])
                            true_sources = tf.get("source_true_indices", [])
                            pred_sources = tf.get("source_pred_indices", [])
                            true_ops = tf.get("operator_true_indices", [])
                            pred_ops = tf.get("operator_pred_indices", [])
                            true_pairs = tf.get("pair_true_indices", [])
                            derived_pairs = tf.get("pair_derived_indices", [])
                            for row, (src_t, src_p, op_t, op_p, pair_t, pair_p) in enumerate(
                                zip(true_sources, pred_sources, true_ops, pred_ops, true_pairs, derived_pairs)
                            ):
                                true_source = source_names[src_t] if src_t < len(source_names) else str(src_t)
                                pred_source = source_names[src_p] if src_p < len(source_names) else str(src_p)
                                true_op = operator_names[op_t] if op_t < len(operator_names) else str(op_t)
                                pred_op = operator_names[op_p] if op_p < len(operator_names) else str(op_p)
                                print(
                                    f"{row:>4} {true_source:>18} {pred_source:>18} "
                                    f"{true_op:>18} {pred_op:>18} {str(pair_t == pair_p):>10}"
                                )
            print("=" * 112)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--depth-sweep", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--base-span", type=int, default=2)
    parser.add_argument("--dlinoss-steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--omega-scale", type=float, default=1.5)
    parser.add_argument("--barrier-base", type=float, default=0.25)
    parser.add_argument("--barrier-alpha", type=float, default=1.75)
    parser.add_argument("--tunnel-temp", type=float, default=0.85)
    parser.add_argument("--top-frac", type=float, default=0.12)
    parser.add_argument("--out-dir", default="program_ao_results")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--run-heldout-transport", action="store_true")
    parser.add_argument("--heldout-include-controls", action="store_true")
    parser.add_argument("--run-semantic-recovery", action="store_true")
    parser.add_argument("--n-train-states", type=int, default=24)
    parser.add_argument("--n-test-states", type=int, default=12)
    parser.add_argument("--mixture-alpha", type=float, default=0.45)
    parser.add_argument("--decoder-ridge", type=float, default=0.02)
    parser.add_argument("--heldout-seed", type=int, default=314159)
    parser.add_argument("--recovery-train-states", type=int, default=24)
    parser.add_argument("--recovery-test-states", type=int, default=12)
    parser.add_argument("--recovery-noise", type=float, default=0.18)
    parser.add_argument("--recovery-noise-sweep", nargs="+", type=float, default=[0.18, 0.30, 0.45])
    parser.add_argument("--recovery-seed", type=int, default=271828)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--control-block-size", type=int, default=2)
    parser.add_argument("--score-alpha", type=float, default=0.08)
    parser.add_argument("--score-beta", type=float, default=0.60)
    parser.add_argument("--score-gamma", type=float, default=0.25)
    parser.add_argument("--score-delta", type=float, default=0.07)
    parser.add_argument("--git-commit", default=current_git_commit())
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--ap-focused",
        action="store_true",
        help="Run the fast TPU/JAX recovery path: semantic recovery only, no CPU-heavy held-out sweep.",
    )
    parser.add_argument(
        "--ap-lite",
        action="store_true",
        help="Run the minimal high-noise AP-lite matrix: two controllers, one noise, three seeds.",
    )
    parser.add_argument(
        "--ap-r",
        action="store_true",
        help="Run Program AP-R: residual tunnel-eigen recovery against filament_stabilized_v2.",
    )
    parser.add_argument(
        "--ap-s",
        action="store_true",
        help="Run Program AP-S: stability plus fidelity lanes for residual tunnel-eigen recovery.",
    )
    parser.add_argument(
        "--ap-s-mech",
        action="store_true",
        help="Run Program AP-S mechanism lane only: AP-R residual vs filament baseline across 0.30/0.45.",
    )
    parser.add_argument(
        "--ap-s-fidelity",
        action="store_true",
        help="Run Program AP-S alpha bracket: residual alpha 0.45/0.55/0.65 across 0.30/0.45.",
    )
    parser.add_argument(
        "--ap-t",
        action="store_true",
        help="Run Program AP-T: adaptive projection alpha against fixed alpha 0.55/0.65 anchors.",
    )
    parser.add_argument(
        "--ap-u",
        action="store_true",
        help="Run Program AP-U: reference-lock APR calibration before adaptive-alpha promotion.",
    )
    parser.add_argument(
        "--aq-0",
        dest="aq_0",
        action="store_true",
        help="Run Program AQ-0: noise-free semantic transport ceiling with free and alpha065 baselines.",
    )
    parser.add_argument(
        "--aq-0b",
        dest="aq_0b",
        action="store_true",
        help="Run Program AQ-0b: noise-free 8-motif layer diagnostic with free, alpha055, and alpha065.",
    )
    parser.add_argument(
        "--aq-1a-tf",
        dest="aq_1a_tf",
        action="store_true",
        help="Run Program AQ-1a-TF: noisy motif transformation-field recovery with L5 transform metrics.",
    )
    parser.add_argument(
        "--aq-1a-tf-s29",
        dest="aq_1a_tf_s29",
        action="store_true",
        help="Run the AQ-1a-TF seed-29 follow-up gate only.",
    )
    parser.add_argument(
        "--aq-1b",
        dest="aq_1b",
        action="store_true",
        help="Run Program AQ-1b: operator-conditioned folded recovery with target held out.",
    )
    parser.add_argument(
        "--aq-1c",
        dest="aq_1c",
        action="store_true",
        help="Run Program AQ-1c: noise-free operator identifiability gate.",
    )
    parser.add_argument(
        "--aq-seq-0",
        dest="aq_seq_0",
        action="store_true",
        help="Run Program AQ-SEQ-0: noise-free sequential semantic state gate.",
    )
    parser.add_argument(
        "--aq-img-0",
        dest="aq_img_0",
        action="store_true",
        help="Run Program AQ-IMG-0: noise-free image patch semantic codec gate.",
    )
    parser.add_argument(
        "--aq-img-1",
        dest="aq_img_1",
        action="store_true",
        help="Run Program AQ-IMG-1: noise-free image patch codec gate with vector residual encoding.",
    )
    parser.add_argument(
        "--aq-img-1-lite",
        dest="aq_img_1_lite",
        action="store_true",
        help="Run Program AQ-IMG-1-LITE: fast 8-patch vector residual codec gate.",
    )
    parser.add_argument(
        "--aq0-include-adaptive-v2",
        action="store_true",
        help="Optional AQ-0 comparison arm: include tunnel_eigen_residual_adaptive_v2_shifted.",
    )
    args = parser.parse_args()

    if args.ap_focused:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 48)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 12)
        args.recovery_test_states = min(args.recovery_test_states, 6)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.control_block_size = 2
        print("[AP FOCUSED] tunnel-eigen recovery by seed; hard controls evaluated inside JAX decoder")

    if args.ap_lite:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP LITE] high-noise tunnel-eigen test; partial summary written after every seed")

    if args.ap_r:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-R] residual tunnel-eigen test; partial summary written after every seed")

    if args.ap_s:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-S] stability plus fidelity lanes; partial summary written after every seed")

    if args.ap_s_mech:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-S-MECH] mechanism lane only; partial summary written after every seed")

    if args.ap_s_fidelity:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-S-FIDELITY] alpha bracket only; partial summary written after every seed")

    if args.ap_t:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-T] adaptive alpha test; partial summary written after every seed")

    if args.ap_u:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 32)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 8)
        args.recovery_test_states = min(args.recovery_test_states, 4)
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29, 47]
        args.control_block_size = 2
        print("[AP-U] reference-lock calibration plus gated variant promotion")

    if args.aq_0:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 24)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 4)
        args.recovery_test_states = min(args.recovery_test_states, 2)
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11, 29]
        args.control_block_size = 2
        print("[AQ-0] fast noise-free semantic ceiling with free + alpha065 (2 seeds)")

    if args.aq_0b:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 24)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 4)
        args.recovery_test_states = min(args.recovery_test_states, 2)
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11, 29]
        args.control_block_size = 2
        print("[AQ-0b] noise-free 8-motif layer diagnostic with free + alpha055 + alpha065")

    if args.aq_1a_tf or args.aq_1a_tf_s29:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 24)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = 4
        args.recovery_test_states = 4
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [29] if args.aq_1a_tf_s29 else [11]
        args.control_block_size = 2
        seed_label = "seed 29 follow-up" if args.aq_1a_tf_s29 else "seed 11 gate"
        print(f"[AQ-1a-TF] gated noisy 8-motif transformation-field diagnostic with L5 metrics ({seed_label})")

    if args.aq_1b:
        args.sequence_length = 8
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 24)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = 4
        args.recovery_test_states = 4
        args.recovery_noise_sweep = [0.30, 0.45]
        args.recovery_seeds = [11, 29]
        args.control_block_size = 2
        print("[AQ-1b] compact operator-conditioned recovery gate with target held out")

    if args.aq_1c:
        args.sequence_length = 8
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 16)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = 16
        args.recovery_test_states = 16
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11]
        args.control_block_size = 2
        print("[AQ-1c] noise-free 4x4 operator identifiability gate with free controller")

    if args.aq_seq_0:
        args.sequence_length = 8
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 16)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = 4
        args.recovery_test_states = 4
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11]
        args.control_block_size = 2
        print("[AQ-SEQ-0] noise-free sequential semantic state gate with source payload + next-state feature gate")

    if args.aq_img_0 or args.aq_img_1 or args.aq_img_1_lite:
        args.sequence_length = 8 if args.aq_img_1_lite else 16
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 16)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = 16
        args.recovery_test_states = 8 if args.aq_img_1_lite else 16
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11]
        args.control_block_size = 2
        if args.aq_img_1_lite:
            print("[AQ-IMG-1-LITE] fast noise-free 8-patch codec gate with vector residual grammar")
        elif args.aq_img_1:
            print("[AQ-IMG-1] noise-free 4x4 image patch codec gate with vector residual grammar")
        else:
            print("[AQ-IMG-0] noise-free 4x4 image patch semantic codec gate")

    if args.smoke_test:
        aq_tf_mode = args.aq_1a_tf or args.aq_1a_tf_s29
        aq_l5_mode = aq_tf_mode or args.aq_1b or args.aq_1c or args.aq_seq_0 or args.aq_img_0 or args.aq_img_1 or args.aq_img_1_lite
        args.sequence_length = 8 if (args.aq_0b or aq_l5_mode) else 12
        args.depth_sweep = [2]
        args.seeds = [11]
        args.dlinoss_steps = 4 if (args.aq_0b or aq_l5_mode) else 12
        args.top_frac = 0.2
        args.include_controls = True
        args.run_heldout_transport = True
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.n_train_states = 2 if (args.aq_0b or aq_l5_mode) else 3
        args.n_test_states = 1 if (args.aq_0b or aq_l5_mode) else 2
        args.recovery_train_states = 16 if (args.aq_1c or args.aq_img_0 or args.aq_img_1) else (8 if args.aq_img_1_lite else (4 if aq_l5_mode else (2 if args.aq_0b else 3)))
        args.recovery_test_states = 16 if (args.aq_1c or args.aq_img_0 or args.aq_img_1) else (8 if args.aq_img_1_lite else (4 if aq_l5_mode else (1 if args.aq_0b else 2)))
        args.recovery_noise_sweep = [0.30] if aq_l5_mode else ([0.00] if args.aq_0b else [0.18])
        args.recovery_noise_sweep = [0.00] if (args.aq_1c or args.aq_seq_0 or args.aq_img_0 or args.aq_img_1 or args.aq_img_1_lite) else args.recovery_noise_sweep
        args.recovery_seeds = [11] if (args.aq_0b or aq_l5_mode) else args.recovery_seeds
        args.control_block_size = 2 if (args.aq_0b or aq_l5_mode) else 3
        print("[SMOKE TEST]")

    run_program_ap(args)
