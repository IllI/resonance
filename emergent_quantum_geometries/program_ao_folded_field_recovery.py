"""
Program AO - Folded-Field Recovery + Hard Controls
==================================================
Tests whether folded semantic motifs can be encoded as recoverable filament
geometry in discretized MERA kinematic space.

This program extends Program AN:
  - semantic state = motif transition geometry, not raw symbol identity
  - tunneling weights = barrier-shaped basin transitions
  - kinematic nodes = boundary intervals (u, v)
  - D-LinOSS observable = damped complex phase flow over interval nodes
  - folded recovery score = semantic graph + topology + frame consistency - gate cost
  - hard controls preserve shallow statistics while breaking recurrence topology
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import threading
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

SHUTDOWN_FLAG = False


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


def build_semantic_motifs(seq_len):
    labels = jnp.arange(4, dtype=jnp.int32)
    t = jnp.arange(seq_len, dtype=jnp.int32)

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


def apply_sequence_control(seqs, control_name, key, block_size):
    if control_name == "markov_shuffle":
        keys = jax.random.split(key, seqs.shape[0])
        return jnp.stack([markov_shuffle_sequence(seq, keys[i]) for i, seq in enumerate(seqs)])
    if control_name == "block_permute":
        keys = jax.random.split(key, seqs.shape[0])
        return jnp.stack([block_permute_sequence(seq, keys[i], block_size) for i, seq in enumerate(seqs)])
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


def build_tunneling_kernel(features, barrier_base, barrier_alpha, tunnel_temp):
    diffs = features[:, None, :] - features[None, :, :]
    dist = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-8)
    barrier = barrier_base + barrier_alpha * dist
    kernel = jnp.exp(-barrier / tunnel_temp)
    kernel = kernel.at[jnp.diag_indices(kernel.shape[0])].set(0.0)
    kernel = kernel / (jnp.sum(kernel, axis=1, keepdims=True) + 1e-8)
    return kernel, barrier


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


def build_kinematic_coupling(nodes, crofton, tunneling_strength, ibm_lite=False):
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

    filament_bias = 0.5 * (crofton[:, None] + crofton[None, :])
    coupling = adj * (1.0 + tunneling_strength * filament_bias)
    coupling = coupling / (jnp.sum(coupling, axis=1, keepdims=True) + 1e-8)
    return coupling


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


def graph_cosine_batch(pred, target):
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return np.sum(pred * target, axis=1)


def graph_cosine_batch_jax(pred, target):
    pred = pred / (jnp.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (jnp.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return jnp.sum(pred * target, axis=1)


def ridge_decode(train_x, train_y, test_x, ridge):
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


def signatures_for_weights(args, weights_batch, nodes, motif_entropy, ctrl_cfg, control_name, key_control):
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
        coupling = build_kinematic_coupling(nodes, crofton, ctrl_cfg["tunnel"], ctrl_cfg.get("ibm_lite", False))
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

    sample_ids = jnp.arange(weights_batch.shape[0], dtype=jnp.int32)
    return jax.vmap(one_signature)(weights_batch, sample_ids)


def evaluate_semantic_recovery(args, seed_key, ctrl_cfg, nodes, seqs, graph_features, noise_level):
    key_train, key_test, key_noise, key_shuffle, key_block, key_crofton, key_phase = jax.random.split(seed_key, 7)
    train_w = semantic_mixtures(key_train, seqs.shape[0], args.recovery_train_states, args.mixture_alpha)
    test_w = semantic_mixtures(key_test, seqs.shape[0], args.recovery_test_states, args.mixture_alpha)
    target_graphs = test_w @ graph_features

    clean_entropy = motif_interval_entropy_table(seqs, nodes)
    train_x = signatures_for_weights(args, train_w, nodes, clean_entropy, ctrl_cfg, "structured", key_train)
    clean_test_x = signatures_for_weights(args, test_w, nodes, clean_entropy, ctrl_cfg, "structured", key_test)
    old_noise = args.recovery_noise
    args.recovery_noise = noise_level
    noisy_test_x = signatures_for_weights(args, test_w, nodes, clean_entropy, ctrl_cfg, "entangled_phase_noise", key_noise)
    phase_scramble_x = signatures_for_weights(args, test_w, nodes, clean_entropy, ctrl_cfg, "phase_scramble", key_phase)
    args.recovery_noise = old_noise

    shuffled_seqs = apply_sequence_control(seqs, "markov_shuffle", key_shuffle, args.control_block_size)
    shuffled_entropy = motif_interval_entropy_table(shuffled_seqs, nodes)
    topology_shuffle_x = signatures_for_weights(args, test_w, nodes, shuffled_entropy, ctrl_cfg, "structured", key_shuffle)
    block_seqs = apply_sequence_control(seqs, "block_permute", key_block, args.control_block_size)
    block_entropy = motif_interval_entropy_table(block_seqs, nodes)
    block_permute_x = signatures_for_weights(args, test_w, nodes, block_entropy, ctrl_cfg, "structured", key_block)
    crofton_random_x = signatures_for_weights(args, test_w, nodes, clean_entropy, ctrl_cfg, "random_crofton", key_crofton)

    train_y = train_w @ graph_features
    clean_pred = ridge_decode_jax(train_x, train_y, clean_test_x, args.decoder_ridge)
    noisy_pred = ridge_decode_jax(train_x, train_y, noisy_test_x, args.decoder_ridge)
    topology_pred = ridge_decode_jax(train_x, train_y, topology_shuffle_x, args.decoder_ridge)
    block_pred = ridge_decode_jax(train_x, train_y, block_permute_x, args.decoder_ridge)
    crofton_pred = ridge_decode_jax(train_x, train_y, crofton_random_x, args.decoder_ridge)
    phase_pred = ridge_decode_jax(train_x, train_y, phase_scramble_x, args.decoder_ridge)

    clean_cos = graph_cosine_batch_jax(clean_pred, target_graphs)
    noisy_cos = graph_cosine_batch_jax(noisy_pred, target_graphs)
    topology_cos = graph_cosine_batch_jax(topology_pred, target_graphs)
    block_cos = graph_cosine_batch_jax(block_pred, target_graphs)
    crofton_cos = graph_cosine_batch_jax(crofton_pred, target_graphs)
    phase_cos = graph_cosine_batch_jax(phase_pred, target_graphs)

    mean_graph = jnp.mean(train_y, axis=0, keepdims=True)
    mean_cos = graph_cosine_batch_jax(jnp.repeat(mean_graph, target_graphs.shape[0], axis=0), target_graphs)
    frame_consistency = 1.0 - jnp.mean(jnp.abs(clean_cos - noisy_cos))
    control_margin = jnp.min(
        jnp.array(
            [
                jnp.mean(noisy_cos) - jnp.mean(topology_cos),
                jnp.mean(noisy_cos) - jnp.mean(block_cos),
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

    return {
        "clean_recovery_graph_cosine": float(jnp.mean(clean_cos)),
        "noisy_recovery_graph_cosine": float(jnp.mean(noisy_cos)),
        "topology_shuffle_graph_cosine": float(jnp.mean(topology_cos)),
        "block_permute_graph_cosine": float(jnp.mean(block_cos)),
        "crofton_random_graph_cosine": float(jnp.mean(crofton_cos)),
        "phase_scramble_graph_cosine": float(jnp.mean(phase_cos)),
        "mean_baseline_graph_cosine": float(jnp.mean(mean_cos)),
        "noisy_lift_over_mean": float(jnp.mean(noisy_cos) - jnp.mean(mean_cos)),
        "noisy_lift_over_topology_shuffle": float(jnp.mean(noisy_cos) - jnp.mean(topology_cos)),
        "noisy_lift_over_block_permute": float(jnp.mean(noisy_cos) - jnp.mean(block_cos)),
        "noisy_lift_over_crofton_random": float(jnp.mean(noisy_cos) - jnp.mean(crofton_cos)),
        "noisy_lift_over_phase_scramble": float(jnp.mean(noisy_cos) - jnp.mean(phase_cos)),
        "minimum_hard_control_lift": float(control_margin),
        "frame_consistency": float(frame_consistency),
        "folded_recovery_score": float(folded_score),
        "gate_cost": float(ctrl_cfg.get("gate_cost", 0.0)),
        "recovery_noise": float(noise_level),
        "recovery_train_states": int(args.recovery_train_states),
        "recovery_test_states": int(args.recovery_test_states),
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
        coupling = build_kinematic_coupling(nodes, crofton, ctrl_cfg["tunnel"], ctrl_cfg.get("ibm_lite", False))
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


def run_program_ao(args):
    global SHUTDOWN_FLAG
    t0 = time.time()
    _, backend_info = select_backend(require_tpu=getattr(args, "require_tpu", False))

    print("=" * 88)
    print("Program AO - Folded-Field Recovery + Hard Controls")
    print(f"Backend: {backend_info['backend']} | TPU visible: {backend_info['has_tpu']}")
    print("=" * 88)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "program_ao_summary.json")

    motifs, _ = build_semantic_motifs(args.sequence_length)
    motif_names = list(motifs.keys())
    seqs = jnp.stack([motifs[name] for name in motif_names])
    features = jnp.stack([motif_feature(seq) for seq in seqs])
    graph_features = jnp.stack([motif_transition_graph(seq) for seq in seqs])
    tunneling_kernel_base, barrier_base = build_tunneling_kernel(
        features, args.barrier_base, args.barrier_alpha, args.tunnel_temp
    )
    structured_target_weights = stationary_distribution(tunneling_kernel_base)
    structured_target_feature = structured_target_weights @ features
    structured_target_graph = structured_target_weights @ graph_features
    controls = ["structured"]
    if args.include_controls:
        controls.extend(["markov_shuffle", "block_permute", "phase_scramble", "random_crofton"])

    records = []
    controllers = {
        "free": {"damping": args.damping, "feedback": 0.00, "tunnel": 0.00, "gate_cost": 0.00},
        "filament_stabilized": {"damping": args.damping * 0.7, "feedback": 0.18, "tunnel": 1.00, "gate_cost": 0.20},
        "folded_decoder": {"damping": args.damping * 0.62, "feedback": 0.24, "tunnel": 1.25, "gate_cost": 0.34},
        "ibm_lite_folded": {
            "damping": args.damping * 0.78,
            "feedback": 0.13,
            "tunnel": 0.70,
            "gate_cost": 0.12,
            "ibm_lite": True,
        },
    }

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
                if control_name in ("markov_shuffle", "block_permute"):
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
                    coupling = build_kinematic_coupling(nodes, crofton, cfg["tunnel"], cfg.get("ibm_lite", False))
                    xs = evolve_dlinoss(
                        x0,
                        omega,
                        coupling,
                        jnp.float32(cfg["damping"]),
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
                    "experiment": "folded_field_recovery_hard_controls",
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
            for noise_idx, noise_level in enumerate(args.recovery_noise_sweep):
                depth_record["semantic_recovery"][str(noise_level)] = {}
                print(f"    noise={noise_level:.2f}")
                for ctrl_idx, (ctrl_name, cfg) in enumerate(controllers.items()):
                    recovery_key = jax.random.fold_in(
                        jax.random.PRNGKey(int(args.recovery_seed)),
                        depth * 1000 + noise_idx * 100 + ctrl_idx,
                    )
                    metrics = evaluate_semantic_recovery(
                        args,
                        recovery_key,
                        cfg,
                        nodes,
                        seqs,
                        graph_features,
                        noise_level,
                    )
                    depth_record["semantic_recovery"][str(noise_level)][ctrl_name] = metrics
                    print(
                        f"      {ctrl_name:>17} -> "
                        f"Score:{metrics['folded_recovery_score']:.4f} "
                        f"Noisy:{metrics['noisy_recovery_graph_cosine']:.4f} "
                        f"CtrlMin:{metrics['minimum_hard_control_lift']:+.4f} "
                        f"TopoLift:{metrics['noisy_lift_over_topology_shuffle']:+.4f} "
                        f"PhaseLift:{metrics['noisy_lift_over_phase_scramble']:+.4f}"
                    )

        records.append(depth_record)

    out = {
        "experiment": "folded_field_recovery_hard_controls",
        "scientific_thread": [
            "Recovery basin volume from OAT/Dicke work",
            "Semantic motifs from Program AM",
            "Tunneling-shaped basin transitions",
            "MERA kinematic interval filaments",
            "D-LinOSS damped mode reconstruction",
            "Folded-frame decoder and IBM-lite hardware constraints",
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
        print(f"{'depth':>5} {'noise':>6} {'ctrl':>17} {'score':>8} {'noisy':>8} {'ctrl_l':>8} {'topo_l':>8} {'phase_l':>8}")
        print("-" * 112)
        for rec in records:
            for noise_level, ctrl_map in rec.get("semantic_recovery", {}).items():
                for ctrl, metrics in ctrl_map.items():
                    print(
                            f"{rec['mera_depth']:>5} {float(noise_level):>6.2f} {ctrl:>17} "
                            f"{metrics['folded_recovery_score']:>8.4f} "
                            f"{metrics['noisy_recovery_graph_cosine']:>8.4f} "
                            f"{metrics['minimum_hard_control_lift']:>8.4f} "
                            f"{metrics['noisy_lift_over_topology_shuffle']:>8.4f} "
                            f"{metrics['noisy_lift_over_phase_scramble']:>8.4f}"
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
    parser.add_argument("--control-block-size", type=int, default=4)
    parser.add_argument("--score-alpha", type=float, default=0.45)
    parser.add_argument("--score-beta", type=float, default=0.30)
    parser.add_argument("--score-gamma", type=float, default=0.20)
    parser.add_argument("--score-delta", type=float, default=0.05)
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--ao-focused",
        action="store_true",
        help="Run the fast TPU/JAX recovery path: semantic recovery only, no CPU-heavy held-out sweep.",
    )
    args = parser.parse_args()

    if args.ao_focused:
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 48)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = min(args.recovery_train_states, 12)
        args.recovery_test_states = min(args.recovery_test_states, 6)
        print("[AO FOCUSED] semantic recovery only; hard controls evaluated inside JAX decoder")

    if args.smoke_test:
        args.sequence_length = 12
        args.depth_sweep = [2]
        args.seeds = [11]
        args.dlinoss_steps = 12
        args.top_frac = 0.2
        args.include_controls = True
        args.run_heldout_transport = True
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.n_train_states = 3
        args.n_test_states = 2
        args.recovery_train_states = 3
        args.recovery_test_states = 2
        args.recovery_noise_sweep = [0.18]
        args.control_block_size = 3
        print("[SMOKE TEST]")

    run_program_ao(args)
