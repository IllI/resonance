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
import struct
import subprocess
import sys
import time
import urllib.request
import threading
import zlib
from functools import partial

import numpy as np

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


def write_grayscale_png(path, image):
    """Write a small 8-bit grayscale PNG without adding image dependencies."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("write_grayscale_png expects a 2D uint8 array")
    height, width = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))

    def chunk(kind, data):
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def read_png_luma(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")

    pos = 8
    width = height = bit_depth = color_type = None
    palette = None
    transparency = None
    idat = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(">IIBBBBB", chunk)
            if interlace != 0:
                raise ValueError(f"Interlaced PNGs are not supported: {path}")
            if bit_depth != 8:
                raise ValueError(f"Only 8-bit PNGs are supported, got bit depth {bit_depth}: {path}")
        elif kind == b"PLTE":
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape((-1, 3))
        elif kind == b"tRNS":
            transparency = np.frombuffer(chunk, dtype=np.uint8)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break

    if width is None or height is None:
        raise ValueError(f"PNG missing IHDR: {path}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG color type {color_type}: {path}")

    raw = zlib.decompress(b"".join(idat))
    stride = int(width) * channels
    rows = []
    prev = np.zeros((stride,), dtype=np.uint8)
    idx = 0
    for _row in range(int(height)):
        filter_type = raw[idx]
        idx += 1
        cur = np.frombuffer(raw[idx : idx + stride], dtype=np.uint8).copy()
        idx += stride
        bpp = channels
        for i in range(stride):
            left = int(cur[i - bpp]) if i >= bpp else 0
            up = int(prev[i])
            up_left = int(prev[i - bpp]) if i >= bpp else 0
            if filter_type == 1:
                cur[i] = (int(cur[i]) + left) & 0xFF
            elif filter_type == 2:
                cur[i] = (int(cur[i]) + up) & 0xFF
            elif filter_type == 3:
                cur[i] = (int(cur[i]) + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                cur[i] = (int(cur[i]) + predictor) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type}: {path}")
        rows.append(cur.copy())
        prev = cur

    pixels = np.stack(rows).reshape((int(height), int(width), channels))
    if color_type == 0:
        luma = pixels[:, :, 0].astype(np.float32)
    elif color_type == 3:
        if palette is None:
            raise ValueError(f"Indexed PNG missing palette: {path}")
        rgb = palette[pixels[:, :, 0]]
        if transparency is not None:
            alpha = np.ones((palette.shape[0],), dtype=np.float32) * 255.0
            alpha[: transparency.shape[0]] = transparency.astype(np.float32)
            a = alpha[pixels[:, :, 0]][:, :, None] / 255.0
            rgb = rgb.astype(np.float32) * a + 127.5 * (1.0 - a)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    else:
        rgb = pixels[:, :, :3].astype(np.float32)
        if color_type in (4, 6):
            alpha = pixels[:, :, -1:].astype(np.float32) / 255.0
            rgb = rgb * alpha + 127.5 * (1.0 - alpha)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return luma / 255.0


def render_patch_preview_png(target_pixels, pred_pixels, path, patch_side=4, scale=16, gap=2):
    """Render target/recovered/error patch contact sheet for AQ image codecs."""
    target = np.asarray(target_pixels, dtype=np.float32).reshape((-1, patch_side, patch_side))
    pred = np.asarray(pred_pixels, dtype=np.float32).reshape((-1, patch_side, patch_side))
    n = min(target.shape[0], pred.shape[0], 16)
    if n == 0:
        return False
    target = np.clip(target[:n], 0.0, 1.0)
    pred = np.clip(pred[:n], 0.0, 1.0)
    err = np.clip(np.abs(pred - target) * 4.0, 0.0, 1.0)

    patch_px = patch_side * scale
    width = n * patch_px + (n - 1) * gap
    height = 3 * patch_px + 2 * gap
    canvas = np.full((height, width), 245, dtype=np.uint8)
    for band, patches in enumerate((target, pred, err)):
        y0 = band * (patch_px + gap)
        for idx, patch in enumerate(patches):
            x0 = idx * (patch_px + gap)
            block = np.kron(patch, np.ones((scale, scale), dtype=np.float32))
            canvas[y0 : y0 + patch_px, x0 : x0 + patch_px] = np.rint(block * 255.0).astype(np.uint8)

    write_grayscale_png(path, canvas)
    return True


def assemble_patch_frames(pixels, frame_count, patch_grid=(4, 4), patch_side=4):
    patches = np.asarray(pixels, dtype=np.float32).reshape((-1, patch_side, patch_side))
    rows, cols = int(patch_grid[0]), int(patch_grid[1])
    frame_count = int(frame_count)
    expected = frame_count * rows * cols
    if patches.shape[0] < expected:
        raise ValueError(f"Need {expected} patches to assemble frames, got {patches.shape[0]}")
    frames = []
    for frame_idx in range(frame_count):
        start = frame_idx * rows * cols
        frame = np.zeros((rows * patch_side, cols * patch_side), dtype=np.float32)
        for patch_idx in range(rows * cols):
            pr = patch_idx // cols
            pc = patch_idx % cols
            patch = patches[start + patch_idx]
            y0 = pr * patch_side
            x0 = pc * patch_side
            frame[y0 : y0 + patch_side, x0 : x0 + patch_side] = patch
        frames.append(np.clip(frame, 0.0, 1.0))
    return frames


def reorder_patch_pixels_for_frames(pixels, patch_indices, frame_count, patch_grid=(4, 4), patch_side=4):
    patches = np.asarray(pixels, dtype=np.float32).reshape((-1, patch_side * patch_side))
    indices = np.asarray(patch_indices, dtype=np.int64).reshape(-1)
    rows, cols = int(patch_grid[0]), int(patch_grid[1])
    total = int(frame_count) * rows * cols
    ordered = np.zeros((total, patch_side * patch_side), dtype=np.float32)
    seen = np.zeros((total,), dtype=bool)
    for patch, idx in zip(patches, indices):
        if 0 <= idx < total and not seen[idx]:
            ordered[idx] = patch
            seen[idx] = True
    if not np.all(seen):
        missing = np.where(~seen)[0].tolist()
        raise ValueError(f"Missing preview patches for indices: {missing[:8]}")
    return ordered


def render_frame_png(path, frame, scale=16):
    frame = np.asarray(frame, dtype=np.float32)
    block = np.kron(np.clip(frame, 0.0, 1.0), np.ones((scale, scale), dtype=np.float32))
    write_grayscale_png(path, np.rint(block * 255.0).astype(np.uint8))


def render_frame_comparison_png(target_frame, pred_frame, path, scale=16, gap=4):
    target = np.asarray(target_frame, dtype=np.float32)
    pred = np.asarray(pred_frame, dtype=np.float32)
    err = np.clip(np.abs(pred - target) * 4.0, 0.0, 1.0)
    panels = [target, pred, err]
    panel_h, panel_w = target.shape
    canvas = np.full((panel_h * scale, 3 * panel_w * scale + 2 * gap), 245, dtype=np.uint8)
    for idx, frame in enumerate(panels):
        block = np.kron(np.clip(frame, 0.0, 1.0), np.ones((scale, scale), dtype=np.float32))
        x0 = idx * (panel_w * scale + gap)
        canvas[:, x0 : x0 + panel_w * scale] = np.rint(block * 255.0).astype(np.uint8)
    write_grayscale_png(path, canvas)


def write_frame_preview_pngs(target_pixels, pred_pixels, out_dir, prefix, frame_count, frame_names, patch_grid):
    written = []
    target_frames = assemble_patch_frames(target_pixels, frame_count, patch_grid)
    pred_frames = assemble_patch_frames(pred_pixels, frame_count, patch_grid)
    for idx, (target_frame, pred_frame) in enumerate(zip(target_frames, pred_frames)):
        frame_name = safe_filename_part(frame_names[idx] if idx < len(frame_names) else f"frame_{idx:02d}")
        base = f"{prefix}_{frame_name}"
        target_path = os.path.join(out_dir, f"{base}_target.png")
        recovered_path = os.path.join(out_dir, f"{base}_recovered.png")
        error_path = os.path.join(out_dir, f"{base}_error.png")
        compare_path = os.path.join(out_dir, f"{base}_compare.png")
        render_frame_png(target_path, target_frame)
        render_frame_png(recovered_path, pred_frame)
        render_frame_png(error_path, np.clip(np.abs(pred_frame - target_frame) * 4.0, 0.0, 1.0))
        render_frame_comparison_png(target_frame, pred_frame, compare_path)
        written.extend([target_path, recovered_path, error_path, compare_path])
    return written


def safe_filename_part(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def write_recovery_preview_pngs(records, out_dir):
    written = []
    for rec in records:
        depth = rec.get("mera_depth", "d")
        for noise_level, seed_map in rec.get("semantic_recovery", {}).items():
            for seed, ctrl_map in seed_map.items():
                for ctrl, metrics in ctrl_map.items():
                    tf = metrics.get("transform_field", {})
                    target_pixels = tf.get("preview_patch_pixels_target")
                    pred_pixels = tf.get("preview_patch_pixels_pred")
                    if target_pixels is None or pred_pixels is None:
                        continue
                    patch_indices = tf.get("preview_patch_indices", tf.get("patch_true_indices"))
                    name = (
                        f"aq_preview_depth{safe_filename_part(depth)}"
                        f"_noise{safe_filename_part(noise_level)}"
                        f"_seed{safe_filename_part(seed)}"
                        f"_{safe_filename_part(ctrl)}.png"
                    )
                    path = os.path.join(out_dir, name)
                    if render_patch_preview_png(target_pixels, pred_pixels, path):
                        written.append(path)
                    frame_count = int(tf.get("preview_frame_count", 0) or 0)
                    if frame_count > 0:
                        frame_target_pixels = target_pixels
                        frame_pred_pixels = pred_pixels
                        if patch_indices is not None:
                            frame_target_pixels = reorder_patch_pixels_for_frames(
                                target_pixels,
                                patch_indices,
                                frame_count,
                                tf.get("preview_patch_grid", [4, 4]),
                            )
                            frame_pred_pixels = reorder_patch_pixels_for_frames(
                                pred_pixels,
                                patch_indices,
                                frame_count,
                                tf.get("preview_patch_grid", [4, 4]),
                            )
                        prefix = (
                            f"aq_frame_depth{safe_filename_part(depth)}"
                            f"_noise{safe_filename_part(noise_level)}"
                            f"_seed{safe_filename_part(seed)}"
                            f"_{safe_filename_part(ctrl)}"
                        )
                        written.extend(
                            write_frame_preview_pngs(
                                frame_target_pixels,
                                frame_pred_pixels,
                                out_dir,
                                prefix,
                                frame_count,
                                tf.get("preview_frame_names", []),
                                tf.get("preview_patch_grid", [4, 4]),
                            )
                        )
    return written


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


def build_yinyang_frames(frame_size=16, supersample=4):
    sample_size = int(frame_size) * int(max(1, supersample))
    coords = (jnp.arange(sample_size, dtype=jnp.float32) + 0.5) / float(sample_size)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
    x = 2.0 * xx - 1.0
    y = 1.0 - 2.0 * yy
    circle = x * x + y * y <= 0.98 * 0.98

    upper_lobe = x * x + (y - 0.5) * (y - 0.5) <= 0.5 * 0.5
    lower_lobe = x * x + (y + 0.5) * (y + 0.5) <= 0.5 * 0.5
    upper_dot = x * x + (y - 0.5) * (y - 0.5) <= 0.13 * 0.13
    lower_dot = x * x + (y + 0.5) * (y + 0.5) <= 0.13 * 0.13

    base = jnp.where(y >= 0.0, 1.0, 0.0)
    symbol = jnp.where(upper_lobe, 0.0, base)
    symbol = jnp.where(lower_lobe, 1.0, symbol)
    symbol = jnp.where(upper_dot, 1.0, symbol)
    symbol = jnp.where(lower_dot, 0.0, symbol)
    symbol = jnp.where(circle, symbol, 0.5)
    inverted = jnp.where(circle, 1.0 - symbol, 0.5)
    symbol = symbol.reshape((frame_size, supersample, frame_size, supersample)).mean(axis=(1, 3))
    inverted = inverted.reshape((frame_size, supersample, frame_size, supersample)).mean(axis=(1, 3))
    return jnp.stack([symbol, inverted])


def write_yinyang_reference_pngs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frames = jax.device_get(build_yinyang_frames(frame_size=32, supersample=8))
    human_frames = jax.device_get(build_yinyang_frames(frame_size=256, supersample=2))
    names = ["frame_00_yin_yang", "frame_01_yang_yin"]
    written = []
    for name, frame in zip(names, frames):
        path = os.path.join(out_dir, f"{name}_reference.png")
        render_frame_png(path, frame)
        written.append(path)
    for name, frame in zip(names, human_frames):
        path = os.path.join(out_dir, f"{name}_human_reference.png")
        render_frame_png(path, frame, scale=1)
        written.append(path)
    return written


def find_yinyang_reference_dir():
    candidates = [
        os.environ.get("YINYANG_REFERENCE_DIR", ""),
        os.path.join(os.getcwd(), "yinyang_reference_v2"),
        os.path.join(os.getcwd(), "tpu_previews", "yinyang_reference_v2"),
        os.path.join(os.path.dirname(os.getcwd()), "tpu_previews", "yinyang_reference_v2"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def load_yinyang_reference_frames(frame_size=32, frame_count=2, frame_start=1):
    ref_dir = find_yinyang_reference_dir()
    if ref_dir is None:
        return None, None
    prefix = f"taichi_{int(frame_size)}x{int(frame_size)}_frame_"
    names = sorted(name for name in os.listdir(ref_dir) if name.startswith(prefix) and name.lower().endswith(".png"))
    if not names:
        return None, None
    start = max(0, int(frame_start) - 1)
    selected_names = names[start : start + int(frame_count)]
    if len(selected_names) < int(frame_count):
        raise ValueError(f"Requested {frame_count} frame(s) from frame_start={frame_start}, but only found {len(names)} matching PNGs")
    frames = []
    for name in selected_names:
        frame = read_png_luma(os.path.join(ref_dir, name))
        if frame.shape != (int(frame_size), int(frame_size)):
            raise ValueError(f"Reference frame {name} has shape {frame.shape}, expected {(int(frame_size), int(frame_size))}")
        frames.append(jnp.asarray(frame, dtype=jnp.float32))
    return jnp.stack(frames), selected_names


def build_yinyang_fft_patch_codec_payload(
    seq_len,
    top_k=6,
    residual_k=4,
    direct_complex=True,
    dna_instruction=True,
    frame_count=2,
):
    payload_seqs, source_features, graph_features, layer_features, transform_info = build_fft_patch_codec_payload(
        seq_len,
        top_k=top_k,
        patch_limit=16,
        residual_k=residual_k,
        direct_complex=direct_complex,
        dna_instruction=dna_instruction,
    )

    frames = build_yinyang_frames()[: int(frame_count)]
    frame_count, frame_size, _ = frames.shape
    patch_side = 4
    patch_rows = frame_size // patch_side
    patch_cols = frame_size // patch_side
    patches = []
    for frame_idx in range(int(frame_count)):
        frame = frames[frame_idx]
        for row in range(int(patch_rows)):
            for col in range(int(patch_cols)):
                patch = frame[
                    row * patch_side : (row + 1) * patch_side,
                    col * patch_side : (col + 1) * patch_side,
                ].reshape(-1)
                patches.append(patch)
    patch_pixels = jnp.stack(patches)

    motif_pixels = transform_info["motif_pixel_catalog"]
    patch_motif_indices = jnp.argmin(
        jnp.mean((patch_pixels[:, None, :] - motif_pixels[None, :, :]) ** 2, axis=2),
        axis=1,
    ).astype(jnp.int32)
    payload_seqs = jnp.rint(patch_pixels * 3.0).astype(jnp.int32)
    motif_features = transform_info["motif_code_catalog"][patch_motif_indices]

    patches_2d = patch_pixels.reshape((patch_pixels.shape[0], patch_side, patch_side))
    fft_full = jnp.fft.fft2(patches_2d).reshape((patch_pixels.shape[0], 16))
    magnitudes = jnp.abs(fft_full)
    top_k = int(max(1, min(16, int(top_k))))
    residual_k = int(max(0, min(16 - top_k, int(residual_k))))
    basis_order = jnp.array([0, 1, 4, 5, 2, 8, 6, 9, 10, 3, 12, 7, 13, 11, 14, 15], dtype=jnp.int32)
    core_idx = basis_order[:top_k]
    core_mask = jnp.sum(jax.nn.one_hot(core_idx, 16, dtype=jnp.float32), axis=0)
    residual_magnitudes = magnitudes * (1.0 - core_mask[None, :])
    residual_idx = jnp.argsort(residual_magnitudes, axis=1)[:, -residual_k:] if residual_k > 0 else jnp.zeros((patch_pixels.shape[0], 0), dtype=jnp.int32)
    top_idx = (
        jnp.concatenate([jnp.tile(core_idx[None, :], (patch_pixels.shape[0], 1)), residual_idx], axis=1)
        if residual_k > 0
        else jnp.tile(core_idx[None, :], (patch_pixels.shape[0], 1))
    )
    top_mask = jnp.clip(jnp.sum(jax.nn.one_hot(top_idx, 16, dtype=jnp.float32), axis=1), 0.0, 1.0)
    residual_mask = jnp.clip(jnp.sum(jax.nn.one_hot(residual_idx, 16, dtype=jnp.float32), axis=1), 0.0, 1.0)
    sparse_fft = fft_full * top_mask
    sparse_mag = jnp.abs(sparse_fft)
    sparse_phase = jnp.angle(sparse_fft)
    sparse_phase_cos = jnp.cos(sparse_phase) * top_mask
    sparse_phase_sin = jnp.sin(sparse_phase) * top_mask
    sparse_fft_complex_features = jnp.concatenate([jnp.real(sparse_fft), jnp.imag(sparse_fft)], axis=1)
    full_fft_features = jnp.concatenate([jnp.real(fft_full), jnp.imag(fft_full)], axis=1)
    low_freq_idx = jnp.array([0, 1, 4, 5], dtype=jnp.int32)
    low_freq_mag = sparse_mag[:, low_freq_idx]
    low_freq_phase = jnp.concatenate([sparse_phase_cos[:, low_freq_idx], sparse_phase_sin[:, low_freq_idx]], axis=1)
    dc_mag = jnp.abs(fft_full[:, 0:1])
    total_energy = jnp.sum(magnitudes * magnitudes, axis=1, keepdims=True) + 1e-8
    dc_energy = (jnp.abs(fft_full[:, 0:1]) ** 2) / total_energy
    lowfreq_energy = jnp.sum((jnp.abs(fft_full[:, low_freq_idx]) ** 2), axis=1, keepdims=True) / total_energy
    residual_energy = jnp.sum((magnitudes * magnitudes) * residual_mask, axis=1, keepdims=True) / total_energy
    phase_coherence = jnp.sum(sparse_phase_cos * top_mask, axis=1, keepdims=True) / (
        jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8
    )
    support_prob = top_mask / (jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8)
    support_entropy = -jnp.sum(
        jnp.where(support_prob > 0.0, support_prob * jnp.log(support_prob + 1e-8), 0.0),
        axis=1,
        keepdims=True,
    ) / jnp.log(jnp.float32(16.0))
    motif_consistency = jnp.max(motif_features, axis=1, keepdims=True)
    shape_charges = jnp.concatenate(
        [dc_energy, lowfreq_energy, phase_coherence, residual_energy, support_entropy, motif_consistency],
        axis=1,
    )
    mag_norm = sparse_mag / (jnp.sum(sparse_mag, axis=1, keepdims=True) + 1e-8)
    freq_axis = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)[None, :]
    checksum = jnp.mod(
        jnp.sum(top_mask * (freq_axis + 0.125) + mag_norm * 0.5 + sparse_phase_cos * 0.25, axis=1, keepdims=True),
        1.0,
    )
    instruction_features = jnp.concatenate([top_mask, residual_mask, mag_norm, sparse_phase_cos, sparse_phase_sin, checksum], axis=1)
    source_features = jnp.concatenate([motif_features, sparse_fft_complex_features, instruction_features], axis=1)
    graph_features = jnp.concatenate([sparse_fft_complex_features, instruction_features, motif_features], axis=1)
    layer_features = {
        "L1_occupation": dc_mag,
        "L2_transition": low_freq_mag,
        "L3_fft_phase": low_freq_phase,
        "L4_recurrence": sparse_mag,
        "L5_transform": instruction_features,
    }
    oracle_recon = jnp.real(jnp.fft.ifft2(sparse_fft.reshape((patch_pixels.shape[0], 4, 4))).reshape((patch_pixels.shape[0], 16)))
    oracle_recon = jnp.clip(oracle_recon, 0.0, 1.0)
    oracle_mse = jnp.mean((oracle_recon - patch_pixels) ** 2)
    transform_info.update(
        {
            "pair_names": [
                f"yin_yang_f{frame_idx}_patch_{patch_idx:02d}"
                for frame_idx in range(int(frame_count))
                for patch_idx in range(int(patch_rows * patch_cols))
            ],
            "source_features": source_features,
            "target_features": full_fft_features,
            "transform_code": instruction_features,
            "delta_graph": graph_features,
            "delta_phase": low_freq_phase,
            "delta_recurrence": top_mask,
            "patch_pixels": patch_pixels,
            "patch_motif_indices": patch_motif_indices,
            "fft_full": fft_full,
            "fft_sparse": sparse_fft,
            "fft_sparse_features": sparse_fft_complex_features,
            "fft_sparse_complex_features": sparse_fft_complex_features,
            "fft_complex_x0_table": sparse_fft,
            "fft_top_indices": top_idx,
            "fft_top_mask": top_mask,
            "fft_top_k": top_k,
            "fft_core_k": top_k,
            "fft_residual_k": residual_k,
            "fft_total_k": top_k + residual_k,
            "fft_basis_indices": core_idx.tolist(),
            "fft_residual_indices": residual_idx.tolist(),
            "fft_instruction_features": instruction_features,
            "fft_shape_charges": shape_charges,
            "fft_direct_complex_x0": True,
            "fft_codec_format": "complex_ri",
            "fft_codec_variant": "yinyang_direct_complex_oscillator",
            "fft_oracle_psnr": float(-10.0 * jnp.log10(oracle_mse + 1e-8)),
            "source_names": transform_info["source_names"],
            "patch_grid": [4, 4],
            "patch_shape": [4, 4],
            "frame_count": int(frame_count),
            "frame_names": ["frame_00_yin_yang", "frame_01_yang_yin"][: int(frame_count)],
            "frame_shape": [int(frame_size), int(frame_size)],
            "yinyang_frame_codec": True,
        }
    )
    return payload_seqs, source_features, graph_features, layer_features, transform_info


def build_yinyang_raw_patch_payload(seq_len, frame_count=1, direct_complex=True, frame_size=32, frame_start=1):
    del seq_len
    frames, reference_names = load_yinyang_reference_frames(frame_size=frame_size, frame_count=frame_count, frame_start=frame_start)
    if frames is None:
        frames = build_yinyang_frames(frame_size=int(frame_size), supersample=8)[: int(frame_count)]
        reference_names = ["frame_00_yin_yang", "frame_01_yang_yin"][: int(frame_count)]
        reference_source = "generated"
    else:
        reference_names = [os.path.splitext(name)[0] for name in reference_names]
        reference_source = "taichi_reference_png"
    frame_count, frame_size, _ = frames.shape
    patch_side = 4
    patch_rows = frame_size // patch_side
    patch_cols = frame_size // patch_side
    patches = []
    for frame_idx in range(int(frame_count)):
        frame = frames[frame_idx]
        for row in range(int(patch_rows)):
            for col in range(int(patch_cols)):
                patch = frame[
                    row * patch_side : (row + 1) * patch_side,
                    col * patch_side : (col + 1) * patch_side,
                ].reshape(-1)
                patches.append(patch)
    patch_pixels = jnp.stack(patches)
    payload_seqs = jnp.rint(patch_pixels * 3.0).astype(jnp.int32)

    pixel_mean = jnp.mean(patch_pixels, axis=1, keepdims=True)
    pixel_std = jnp.std(patch_pixels, axis=1, keepdims=True)
    pixel_min = jnp.min(patch_pixels, axis=1, keepdims=True)
    pixel_max = jnp.max(patch_pixels, axis=1, keepdims=True)
    pixel_energy = jnp.mean(patch_pixels * patch_pixels, axis=1, keepdims=True)
    patch_stats = jnp.concatenate([pixel_mean, pixel_std, pixel_min, pixel_max, pixel_energy], axis=1)
    position_feature = jax.nn.one_hot(jnp.arange(patch_pixels.shape[0], dtype=jnp.int32), patch_pixels.shape[0], dtype=jnp.float32)

    source_features = jnp.concatenate([patch_pixels, patch_stats, position_feature], axis=1)
    graph_features = jnp.concatenate([patch_stats, patch_pixels, position_feature], axis=1)
    base_layer_features = build_layer_feature_table(payload_seqs)
    layer_features = {
        "L1_occupation": pixel_mean,
        "L2_transition": patch_pixels[:, :4],
        "L3_fft_phase": patch_pixels[:, 4:8],
        "L4_recurrence": patch_pixels[:, 8:16],
        "L5_transform": jnp.concatenate([patch_stats, position_feature], axis=1),
    }
    transform_info = {
        "image_patch_codec": True,
        "raw_yinyang_patch_codec": True,
        "pair_names": [
            f"raw_yinyang_f{frame_idx}_patch_{patch_idx:02d}"
            for frame_idx in range(int(frame_count))
            for patch_idx in range(int(patch_rows * patch_cols))
        ],
        "source_features": source_features,
        "target_features": source_features,
        "transform_code": jnp.concatenate([patch_pixels, patch_stats], axis=1),
        "delta_graph": graph_features,
        "delta_phase": base_layer_features["L3_fft_phase"],
        "delta_recurrence": base_layer_features["L4_recurrence"],
        "patch_pixels": patch_pixels,
        "motif_pixel_catalog": patch_pixels,
        "motif_code_catalog": source_features,
        "patch_motif_indices": jnp.arange(patch_pixels.shape[0], dtype=jnp.int32),
        "fft_top_indices": jnp.arange(patch_pixels.shape[0] * 0, dtype=jnp.int32).reshape((patch_pixels.shape[0], 0)),
        "residuals": jnp.zeros((patch_pixels.shape[0],), dtype=jnp.float32),
        "residual_vectors": jnp.zeros((patch_pixels.shape[0], 5), dtype=jnp.float32),
        "residual_vector_codec": False,
        "source_names": [f"raw_patch_{i:02d}" for i in range(int(patch_pixels.shape[0]))],
        "patch_grid": [int(patch_rows), int(patch_cols)],
        "patch_shape": [patch_side, patch_side],
        "frame_count": int(frame_count),
        "frame_names": reference_names,
        "frame_shape": [int(frame_size), int(frame_size)],
        "frame_reference_source": reference_source,
        "direct_complex_x0": bool(direct_complex),
        "direct_complex_x0_table": patch_pixels.astype(jnp.complex64),
        "direct_complex_x0_kind": "raw_pixels",
        "fft_patch_codec": False,
        "fast_codec_probe": True,
    }
    return payload_seqs, source_features, graph_features, layer_features, transform_info


def token_mutual_information(labels_a, labels_b, n_a, n_b):
    labels_a = labels_a.astype(jnp.int32).reshape(-1)
    labels_b = labels_b.astype(jnp.int32).reshape(-1)
    joint_idx = labels_a * int(n_b) + labels_b
    joint = jnp.bincount(joint_idx, length=int(n_a) * int(n_b)).astype(jnp.float32)
    joint = joint.reshape((int(n_a), int(n_b)))
    joint = joint / (jnp.sum(joint) + 1e-8)
    pa = jnp.sum(joint, axis=1, keepdims=True)
    pb = jnp.sum(joint, axis=0, keepdims=True)
    ratio = joint / (pa * pb + 1e-8)
    return jnp.sum(jnp.where(joint > 0.0, joint * (jnp.log(ratio + 1e-8) / jnp.log(2.0)), 0.0))


def token_mi_against_roll_null(labels_a, labels_b, n_a, n_b):
    observed = token_mutual_information(labels_a, labels_b, n_a, n_b)
    shifts = jnp.arange(1, 9, dtype=jnp.int32)
    nulls = jnp.stack(
        [token_mutual_information(labels_a, jnp.roll(labels_b, int(shift)), n_a, n_b) for shift in shifts]
    )
    return observed, jnp.mean(nulls), jnp.mean((nulls >= observed).astype(jnp.float32))


def ttn_fold_instruction_packets(packet_features, slot_charges):
    current = packet_features
    charge_current = slot_charges
    bond_messages = []
    charge_messages = []
    while current.shape[1] > 1:
        left = current[:, 0::2, :]
        right = current[:, 1::2, :]
        parent = jnp.concatenate([(left + right) * 0.5, jnp.abs(left - right), left * right], axis=2)
        charge_parent = charge_current[:, 0::2, :] + charge_current[:, 1::2, :]
        bond_messages.append(jnp.mean(parent, axis=1))
        charge_messages.append(jnp.mean(charge_parent, axis=1))
        current = parent
        charge_current = charge_parent
    root_tensor = current[:, 0, :]
    root_charge = charge_current[:, 0, :]
    internal_bonds = jnp.concatenate(bond_messages[:-1] if len(bond_messages) > 1 else bond_messages, axis=1)
    charge_bonds = jnp.concatenate(charge_messages[:-1] if len(charge_messages) > 1 else charge_messages, axis=1)
    return root_tensor, internal_bonds, root_charge, charge_bonds


def hermitian_project_fft4(fft_values):
    fft_grid = fft_values.reshape((-1, 4, 4))
    rows = jnp.arange(4, dtype=jnp.int32)
    cols = jnp.arange(4, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    partner = jnp.conj(fft_grid[:, (-rr) % 4, (-cc) % 4])
    projected = 0.5 * (fft_grid + partner)
    self_mask = (((2 * rr) % 4 == 0) & ((2 * cc) % 4 == 0))[None, :, :]
    projected = jnp.where(self_mask, jnp.real(projected) + 0j, projected)
    return projected.reshape((-1, 16))


def hermitian_error_fft4(fft_values):
    fft_grid = fft_values.reshape((-1, 4, 4))
    rows = jnp.arange(4, dtype=jnp.int32)
    cols = jnp.arange(4, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    partner = jnp.conj(fft_grid[:, (-rr) % 4, (-cc) % 4])
    pair_error = jnp.abs(fft_grid - partner) ** 2
    denom = jnp.sum(jnp.abs(fft_grid) ** 2, axis=(1, 2)) + 1e-8
    return jnp.mean(jnp.sum(pair_error, axis=(1, 2)) / denom)


def self_conjugate_imag_energy_fft4(fft_values):
    self_idx = jnp.array([0, 2, 8, 10], dtype=jnp.int32)
    self_values = fft_values[:, self_idx]
    return jnp.mean(jnp.sum(jnp.imag(self_values) ** 2, axis=1))


def build_fft_patch_codec_payload(
    seq_len,
    top_k=6,
    patch_limit=16,
    residual_k=0,
    direct_complex=False,
    dna_instruction=False,
    dna_tokenized=False,
    dna_checksum_mode=False,
    ttn_folded=False,
    ttn_unfold_decoder=False,
):
    del seq_len
    payload_seqs, _source_features, _graph_features, _layer_features, base_info = build_image_patch_codec_payload(
        16,
        residual_vector=False,
        patch_limit=patch_limit,
    )
    patch_pixels = base_info["patch_pixels"]
    patches_2d = patch_pixels.reshape((patch_limit, 4, 4))
    fft_full = jnp.fft.fft2(patches_2d).reshape((patch_limit, 16))
    magnitudes = jnp.abs(fft_full)
    top_k = int(max(1, min(16, int(top_k))))
    residual_k = int(max(0, min(16 - top_k, int(residual_k))))
    # Fixed low-frequency-first basis ordering (4x4 rfft-style zig-zag-ish scan).
    basis_order = jnp.array([0, 1, 4, 5, 2, 8, 6, 9, 10, 3, 12, 7, 13, 11, 14, 15], dtype=jnp.int32)
    core_idx = basis_order[:top_k]
    core_mask = jnp.sum(jax.nn.one_hot(core_idx, 16, dtype=jnp.float32), axis=0)
    residual_magnitudes = magnitudes * (1.0 - core_mask[None, :])
    if residual_k > 0:
        residual_idx = jnp.argsort(residual_magnitudes, axis=1)[:, -residual_k:]
        top_idx = jnp.concatenate([jnp.tile(core_idx[None, :], (patch_limit, 1)), residual_idx], axis=1)
    else:
        residual_idx = jnp.zeros((patch_limit, 0), dtype=jnp.int32)
        top_idx = jnp.tile(core_idx[None, :], (patch_limit, 1))
    top_mask = jnp.sum(jax.nn.one_hot(top_idx, 16, dtype=jnp.float32), axis=1)
    top_mask = jnp.clip(top_mask, 0.0, 1.0)
    residual_mask = jnp.sum(jax.nn.one_hot(residual_idx, 16, dtype=jnp.float32), axis=1)
    residual_mask = jnp.clip(residual_mask, 0.0, 1.0)
    sparse_fft = fft_full * top_mask
    sparse_mag = jnp.abs(sparse_fft)
    sparse_phase = jnp.angle(sparse_fft)
    sparse_phase_cos = jnp.cos(sparse_phase) * top_mask
    sparse_phase_sin = jnp.sin(sparse_phase) * top_mask
    sparse_fft_mag_phase_features = jnp.concatenate([sparse_mag, sparse_phase_cos, sparse_phase_sin], axis=1)
    sparse_fft_complex_features = jnp.concatenate([jnp.real(sparse_fft), jnp.imag(sparse_fft)], axis=1)
    sparse_fft_features = sparse_fft_complex_features if direct_complex else sparse_fft_mag_phase_features
    full_fft_features = jnp.concatenate([jnp.real(fft_full), jnp.imag(fft_full)], axis=1)
    low_freq_idx = jnp.array([0, 1, 4, 5], dtype=jnp.int32)
    low_freq_mag = sparse_mag[:, low_freq_idx]
    low_freq_phase = jnp.concatenate([sparse_phase_cos[:, low_freq_idx], sparse_phase_sin[:, low_freq_idx]], axis=1)
    dc_mag = jnp.abs(fft_full[:, 0:1])
    total_energy = jnp.sum(magnitudes * magnitudes, axis=1, keepdims=True) + 1e-8
    dc_energy = (jnp.abs(fft_full[:, 0:1]) ** 2) / total_energy
    lowfreq_energy = jnp.sum((jnp.abs(fft_full[:, low_freq_idx]) ** 2), axis=1, keepdims=True) / total_energy
    residual_energy = jnp.sum((magnitudes * magnitudes) * residual_mask, axis=1, keepdims=True) / total_energy
    phase_coherence = jnp.sum(sparse_phase_cos * top_mask, axis=1, keepdims=True) / (
        jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8
    )
    support_prob = top_mask / (jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8)
    support_entropy = -jnp.sum(
        jnp.where(support_prob > 0.0, support_prob * jnp.log(support_prob + 1e-8), 0.0),
        axis=1,
        keepdims=True,
    ) / jnp.log(jnp.float32(16.0))

    motif_features = base_info["motif_code_catalog"][base_info["patch_motif_indices"]]
    motif_consistency = jnp.max(motif_features, axis=1, keepdims=True)
    shape_charges = jnp.concatenate(
        [dc_energy, lowfreq_energy, phase_coherence, residual_energy, support_entropy, motif_consistency],
        axis=1,
    )
    mag_norm = sparse_mag / (jnp.sum(sparse_mag, axis=1, keepdims=True) + 1e-8)
    freq_axis = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)[None, :]
    checksum = jnp.mod(
        jnp.sum(top_mask * (freq_axis + 0.125) + mag_norm * 0.5 + sparse_phase_cos * 0.25, axis=1, keepdims=True),
        1.0,
    )
    instruction_features = jnp.concatenate(
        [top_mask, residual_mask, mag_norm, sparse_phase_cos, sparse_phase_sin, checksum],
        axis=1,
    )
    packet_features = jnp.stack(
        [
            top_mask,
            residual_mask,
            mag_norm,
            sparse_phase_cos,
            sparse_phase_sin,
            jnp.tile(freq_axis, (patch_limit, 1)),
            jnp.tile(checksum, (1, 16)),
        ],
        axis=2,
    )
    dc_slot = jax.nn.one_hot(jnp.array([0], dtype=jnp.int32), 16, dtype=jnp.float32)[0]
    low_slot = jnp.sum(jax.nn.one_hot(low_freq_idx, 16, dtype=jnp.float32), axis=0)
    slot_energy = (magnitudes * magnitudes) / total_energy
    slot_charges = jnp.stack(
        [
            slot_energy * dc_slot[None, :],
            slot_energy * low_slot[None, :],
            top_mask * sparse_phase_cos / (jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8),
            residual_mask,
            top_mask * motif_consistency / (jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8),
            top_mask * checksum / (jnp.sum(top_mask, axis=1, keepdims=True) + 1e-8),
        ],
        axis=2,
    )
    ttn_root_tensor, ttn_internal_bonds, ttn_root_charge, ttn_charge_bonds = ttn_fold_instruction_packets(
        packet_features, slot_charges
    )
    ttn_grammar_features = jnp.concatenate(
        [ttn_root_tensor, ttn_internal_bonds, ttn_root_charge, ttn_charge_bonds],
        axis=1,
    )
    retained_mask = top_mask.astype(jnp.float32)
    mag_mean = jnp.mean(sparse_mag)
    mag_std = jnp.std(sparse_mag) + 1e-8
    mag_state = jnp.where(sparse_mag > mag_mean + 0.35 * mag_std, 2, jnp.where(sparse_mag > mag_mean, 1, 0))
    phase_raw = jnp.floor((jnp.mod(sparse_phase + jnp.pi, 2.0 * jnp.pi) / (2.0 * jnp.pi)) * 4.0).astype(jnp.int32)
    phase_state = jnp.clip(phase_raw, 0, 3)
    residual_mag = sparse_mag * residual_mask
    residual_threshold = jnp.mean(jnp.where(residual_mask > 0.0, residual_mag, 0.0))
    residual_state = jnp.where(
        core_mask[None, :] > 0.0,
        1,
        jnp.where(residual_mask > 0.0, jnp.where(residual_mag > residual_threshold, 3, 2), 0),
    )
    dominant_idx = jnp.argmax(sparse_mag, axis=1)
    dominant_mask = jax.nn.one_hot(dominant_idx, 16, dtype=jnp.float32)
    support_rank_state = jnp.where(
        dominant_mask > 0.0,
        3,
        jnp.where(core_mask[None, :] > 0.0, 2, jnp.where(residual_mask > 0.0, 1, 0)),
    )
    mag_tokens = jax.nn.one_hot(mag_state, 3, dtype=jnp.float32).reshape((patch_limit, -1))
    phase_tokens = jax.nn.one_hot(phase_state, 4, dtype=jnp.float32).reshape((patch_limit, -1))
    residual_tokens = jax.nn.one_hot(residual_state, 4, dtype=jnp.float32).reshape((patch_limit, -1))
    support_tokens = jax.nn.one_hot(support_rank_state, 4, dtype=jnp.float32).reshape((patch_limit, -1))
    dna_token_features = jnp.concatenate(
        [mag_tokens, phase_tokens, residual_tokens, support_tokens, motif_features, checksum],
        axis=1,
    )
    grammar_features = (
        ttn_grammar_features
        if ttn_folded
        else (dna_token_features if dna_tokenized else (instruction_features if dna_instruction else sparse_fft_features))
    )
    if ttn_folded:
        source_features = jnp.concatenate([motif_features, ttn_grammar_features], axis=1)
        graph_features = jnp.concatenate([ttn_grammar_features, motif_features], axis=1)
    else:
        source_features = jnp.concatenate([motif_features, sparse_fft_features, grammar_features], axis=1)
        graph_features = jnp.concatenate([sparse_fft_features, grammar_features, motif_features], axis=1)
    layer_features = {
        "L1_occupation": dc_mag,
        "L2_transition": low_freq_mag,
        "L3_fft_phase": low_freq_phase,
        "L4_recurrence": sparse_mag,
        "L5_transform": grammar_features,
    }
    oracle_recon = jnp.real(jnp.fft.ifft2(sparse_fft.reshape((patch_limit, 4, 4))).reshape((patch_limit, 16)))
    oracle_recon = jnp.clip(oracle_recon, 0.0, 1.0)
    oracle_mse = jnp.mean((oracle_recon - patch_pixels) ** 2)
    transform_info = {
        "fft_patch_codec": True,
        "image_patch_codec": False,
        "pair_names": [name.replace("patch_", "fft_patch_") for name in base_info["pair_names"]],
        "source_features": source_features,
        "target_features": full_fft_features,
        "transform_code": grammar_features,
        "delta_graph": graph_features,
        "delta_phase": low_freq_phase,
        "delta_recurrence": top_mask,
        "patch_pixels": patch_pixels,
        "motif_code_catalog": base_info["motif_code_catalog"],
        "motif_pixel_catalog": base_info["motif_pixel_catalog"],
        "patch_motif_indices": base_info["patch_motif_indices"],
        "fft_full": fft_full,
        "fft_sparse": sparse_fft,
        "fft_sparse_features": sparse_fft_features,
        "fft_sparse_mag_phase_features": sparse_fft_mag_phase_features,
        "fft_sparse_complex_features": sparse_fft_complex_features,
        "fft_complex_x0_table": sparse_fft,
        "fft_top_indices": top_idx,
        "fft_top_mask": top_mask,
        "fft_top_k": top_k,
        "fft_core_k": top_k,
        "fft_residual_k": residual_k,
        "fft_total_k": top_k + residual_k,
        "fft_basis_indices": core_idx.tolist(),
        "fft_residual_indices": residual_idx.tolist(),
        "fft_instruction_features": instruction_features,
        "dna_token_features": dna_token_features,
        "dna_mag_state": mag_state,
        "dna_phase_state": phase_state,
        "dna_residual_state": residual_state,
        "dna_support_rank_state": support_rank_state,
        "dna_retained_mask": retained_mask,
        "dna_tokenized": bool(dna_tokenized),
        "dna_checksum_mode": bool(dna_checksum_mode),
        "dna_token_layout": {
            "mag": [0, 48],
            "phase": [48, 112],
            "residual": [112, 176],
            "support": [176, 240],
        },
        "fft_triplet_packet_format": "[support_or_residual_slot, normalized_magnitude, phase_sincos]",
        "fft_shape_charges": ttn_root_charge if ttn_folded else shape_charges,
        "fft_shape_charge_names": [
            "Q_DC",
            "Q_lowfreq_energy",
            "Q_phase_coherence",
            "Q_residual_energy",
            "Q_support_entropy",
            "Q_motif_consistency",
        ],
        "fft_direct_complex_x0": bool(direct_complex),
        "fft_codec_format": "complex_ri" if direct_complex else "mag_phase_sincos",
        "fft_codec_variant": (
            "direct_complex_oscillator"
            if direct_complex
            else "ttn_folded_instruction_genome"
            if ttn_folded
            else ("dna_token_genome" if dna_tokenized else ("core_residual_genome" if dna_instruction else "fixed_core"))
        ),
        "as_0_ttn": bool(ttn_folded),
        "as_ttn_unfold_decoder": bool(ttn_unfold_decoder),
        "as_ttn_root_tensor": ttn_root_tensor,
        "as_ttn_root_dim": int(ttn_root_tensor.shape[1]),
        "as_ttn_internal_bond_dim": int(ttn_internal_bonds.shape[1]),
        "as_ttn_charge_dim": int(ttn_root_charge.shape[1]),
        "as_flat_baseline_psnr": 16.9995,
        "as_flat_baseline_oracle_psnr": 25.0480,
        "fft_oracle_psnr": float(-10.0 * jnp.log10(oracle_mse + 1e-8)),
        "fast_codec_probe": True,
        "source_names": base_info["source_names"],
        "patch_grid": [4, 4],
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
    if getattr(args, "aq_yinyang_4", False) and getattr(args, "yinyang_ordered_recovery", False):
        idx = jnp.arange(n_samples) % n_states
        return jax.nn.one_hot(idx, n_states, dtype=jnp.float32)
    if (
        getattr(args, "aq_1a_tf", False)
        or getattr(args, "aq_1a_tf_s29", False)
        or getattr(args, "aq_1b", False)
        or getattr(args, "aq_1c", False)
        or getattr(args, "aq_seq_0", False)
        or getattr(args, "as_0b", False)
        or getattr(args, "as_0", False)
        or getattr(args, "aq_yinyang_4", False)
        or getattr(args, "aq_yinyang_3", False)
        or getattr(args, "aq_yinyang_2", False)
        or getattr(args, "aq_yinyang_1", False)
        or getattr(args, "aq_yinyang_0", False)
        or getattr(args, "aq_hybrid_0", False)
        or getattr(args, "aq_dna_0", False)
        or getattr(args, "aq_yinyang_1", False)
        or getattr(args, "aq_yinyang_0", False)
        or getattr(args, "aq_fft_7", False)
        or getattr(args, "aq_fft_6", False)
        or getattr(args, "aq_fft_5", False)
        or getattr(args, "aq_fft_4", False)
        or getattr(args, "aq_fft_3", False)
        or getattr(args, "aq_fft_2", False)
        or getattr(args, "aq_fft_1", False)
        or getattr(args, "aq_fft_0", False)
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


def shape_charge_node_field(nodes, charges, sequence_length):
    centers = (nodes[:, 0] + nodes[:, 1]).astype(jnp.float32) / (2.0 * float(sequence_length))
    spans = (nodes[:, 1] - nodes[:, 0]).astype(jnp.float32) / float(sequence_length)
    generators = jnp.stack(
        [
            jnp.ones_like(centers),
            centers,
            jnp.sin(2.0 * jnp.pi * centers),
            jnp.cos(2.0 * jnp.pi * centers),
            spans,
            jnp.sin(4.0 * jnp.pi * centers),
        ],
        axis=0,
    )
    charge_centered = charges - jnp.mean(charges)
    field = charge_centered @ generators
    return field / (jnp.std(field) + 1e-8)


def signatures_for_weights(
    args,
    weights_batch,
    nodes,
    motif_entropy,
    ctrl_cfg,
    control_name,
    key_control,
    dynamics_bundle=None,
    shape_charge_table=None,
    complex_x0_table=None,
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
        if ctrl_cfg.get("shape_charge_control", False) and shape_charge_table is not None:
            charges = weights @ shape_charge_table
            charge_field = shape_charge_node_field(nodes, charges, args.sequence_length)
            eta = jnp.float32(ctrl_cfg.get("shape_charge_eta", 0.28))
            semantic_phase = semantic_phase + eta * charge_field
            omega = omega * (1.0 + 0.04 * eta * charge_field)
        if control_name == "entangled_phase_noise":
            semantic_phase = semantic_phase + args.recovery_noise * jax.random.normal(
                jax.random.fold_in(key_control, 30_000 + sample_idx),
                semantic_phase.shape,
            )

        x0 = jnp.sqrt(crofton + 1e-3) * jnp.exp(1j * semantic_phase)
        x0 = x0.astype(jnp.complex64)
        payload_mag = None
        if complex_x0_table is not None:
            complex_payload = (weights @ complex_x0_table).astype(jnp.complex64)
            if ctrl_cfg.get("complex_pair_norm", False):
                payload_mag = jnp.abs(complex_payload).astype(jnp.float32)
                complex_payload = complex_payload / (payload_mag + 1e-8)
            complex_payload = complex_payload / (jnp.linalg.norm(complex_payload) + 1e-8)
            payload_width = complex_x0_table.shape[1]
            x0 = x0.at[:payload_width].set(complex_payload)
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
        if complex_x0_table is not None:
            payload_width = complex_x0_table.shape[1]
            recovered_slots = xs[-1, :payload_width]
            direct_complex_signature = jnp.concatenate([jnp.real(recovered_slots), jnp.imag(recovered_slots)], axis=0)
            if payload_mag is not None:
                payload_mag = payload_mag / (jnp.linalg.norm(payload_mag) + 1e-8)
                direct_complex_signature = jnp.concatenate([direct_complex_signature, payload_mag], axis=0)
            base_signature = filament_signature(xs, crofton, args.top_frac)
            signature = jnp.concatenate([base_signature, direct_complex_signature], axis=0)
            return signature / (jnp.linalg.norm(signature) + 1e-8)
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
    shape_charge_table = None
    if transform_info is not None and "fft_shape_charges" in transform_info:
        shape_charge_table = transform_info["fft_shape_charges"]
    complex_x0_table = None
    if transform_info is not None and transform_info.get("fft_direct_complex_x0", False):
        complex_x0_table = transform_info["fft_complex_x0_table"]
    elif transform_info is not None and transform_info.get("direct_complex_x0", False):
        complex_x0_table = transform_info["direct_complex_x0_table"]
    train_x = signatures_for_weights(
        args, train_w, nodes, clean_entropy, ctrl_cfg, "structured", key_train, dynamics_bundle, shape_charge_table, complex_x0_table
    )
    clean_test_x = signatures_for_weights(
        args, test_w, nodes, clean_entropy, ctrl_cfg, "structured", key_test, dynamics_bundle, shape_charge_table, complex_x0_table
    )
    old_noise = args.recovery_noise
    args.recovery_noise = noise_level
    noisy_test_x = signatures_for_weights(
        args, test_w, nodes, clean_entropy, ctrl_cfg, "entangled_phase_noise", key_noise, dynamics_bundle, shape_charge_table, complex_x0_table
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
            args, test_w, nodes, clean_entropy, ctrl_cfg, "phase_scramble", key_phase, dynamics_bundle, shape_charge_table, complex_x0_table
        )
        shuffled_seqs = apply_sequence_control(seqs, "markov_shuffle", key_shuffle, args.control_block_size)
        shuffled_entropy = motif_interval_entropy_table(shuffled_seqs, nodes)
        topology_shuffle_x = signatures_for_weights(
            args, test_w, nodes, shuffled_entropy, ctrl_cfg, "structured", key_shuffle, dynamics_bundle, shape_charge_table, complex_x0_table
        )
        block_seqs = apply_sequence_control(seqs, "block_permute_size2", key_block, args.control_block_size)
        block_entropy = motif_interval_entropy_table(block_seqs, nodes)
        block_permute_x = signatures_for_weights(
            args, test_w, nodes, block_entropy, ctrl_cfg, "structured", key_block, dynamics_bundle, shape_charge_table, complex_x0_table
        )
        block_inner_seqs = apply_sequence_control(seqs, "block_permute_size2_within_random", key_block_inner, args.control_block_size)
        block_inner_entropy = motif_interval_entropy_table(block_inner_seqs, nodes)
        block_permute_inner_x = signatures_for_weights(
            args, test_w, nodes, block_inner_entropy, ctrl_cfg, "structured", key_block_inner, dynamics_bundle, shape_charge_table, complex_x0_table
        )
        crofton_random_x = signatures_for_weights(
            args, test_w, nodes, clean_entropy, ctrl_cfg, "random_crofton", key_crofton, dynamics_bundle, shape_charge_table, complex_x0_table
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
        if transform_info.get("fft_patch_codec", False):
            patch_pixel_target = test_w @ transform_info["patch_pixels"]
            sparse_fft_target = test_w @ transform_info["fft_sparse_features"]
            if transform_info.get("as_ttn_unfold_decoder", False):
                fft_pred = ridge_decode_jax(
                    transform_y,
                    train_w @ transform_info["fft_sparse_features"],
                    transform_pred,
                    args.decoder_ridge,
                )
            else:
                fft_pred = ridge_decode_jax(
                    train_x,
                    train_w @ transform_info["fft_sparse_features"],
                    noisy_test_x,
                    args.decoder_ridge,
                )
            shape_charge_pred = None
            shape_charge_target = None
            if "fft_shape_charges" in transform_info:
                shape_charge_y = train_w @ transform_info["fft_shape_charges"]
                shape_charge_target = test_w @ transform_info["fft_shape_charges"]
                shape_charge_pred = ridge_decode_jax(train_x, shape_charge_y, noisy_test_x, args.decoder_ridge)
            codec_format = str(transform_info.get("fft_codec_format", "complex_ri"))
            if codec_format == "mag_phase_sincos":
                pred_mag = jnp.clip(fft_pred[:, :16], 0.0)
                pred_phase_cos = fft_pred[:, 16:32]
                pred_phase_sin = fft_pred[:, 32:48]
                pred_phase_norm = jnp.sqrt(pred_phase_cos * pred_phase_cos + pred_phase_sin * pred_phase_sin + 1e-8)
                pred_phase_cos = pred_phase_cos / pred_phase_norm
                pred_phase_sin = pred_phase_sin / pred_phase_norm
                fft_pred_complex = pred_mag * (pred_phase_cos + 1j * pred_phase_sin)
            else:
                fft_pred_complex = fft_pred[:, :16] + 1j * fft_pred[:, 16:32]
                if ctrl_cfg.get("complex_phase_locked", False):
                    sparse_mag_y = train_w @ jnp.abs(transform_info["fft_sparse"])
                    pred_mag_locked = ridge_decode_jax(train_x, sparse_mag_y, noisy_test_x, args.decoder_ridge)
                    pred_mag_locked = jnp.clip(pred_mag_locked, 0.0)
                    pred_dir = fft_pred_complex / (jnp.abs(fft_pred_complex) + 1e-8)
                    fft_pred_complex = pred_mag_locked * pred_dir
                if ctrl_cfg.get("complex_gaugefix", False):
                    anchor_mag = jnp.abs(fft_pred_complex[:, 0])
                    pred_mag_tmp = jnp.abs(fft_pred_complex)
                    top1_anchor = jnp.take_along_axis(
                        fft_pred_complex,
                        jnp.argmax(pred_mag_tmp, axis=1, keepdims=True),
                        axis=1,
                    )[:, 0]
                    anchor = jnp.where(anchor_mag > 1e-6, fft_pred_complex[:, 0], top1_anchor)
                    gauge_rot = jnp.exp(-1j * jnp.angle(anchor))[:, None]
                    fft_pred_complex = fft_pred_complex * gauge_rot
                if ctrl_cfg.get("complex_hermitian", False):
                    fft_pred_complex = hermitian_project_fft4(fft_pred_complex)
                fft_pred = jnp.concatenate([jnp.real(fft_pred_complex), jnp.imag(fft_pred_complex)], axis=1)
            patch_pred = jnp.real(jnp.fft.ifft2(fft_pred_complex.reshape((-1, 4, 4))).reshape((-1, 16)))
            patch_pred = jnp.clip(patch_pred, 0.0, 1.0)
            pixel_mse = jnp.mean((patch_pred - patch_pixel_target) ** 2)
            pixel_psnr = -10.0 * jnp.log10(pixel_mse + 1e-8)
            pixel_direct_y = train_w @ transform_info["patch_pixels"]
            pixel_direct_pred = ridge_decode_jax(train_x, pixel_direct_y, noisy_test_x, args.decoder_ridge)
            pixel_direct_pred = jnp.clip(pixel_direct_pred, 0.0, 1.0)
            pixel_direct_mse = jnp.mean((pixel_direct_pred - patch_pixel_target) ** 2)
            pixel_direct_psnr = -10.0 * jnp.log10(pixel_direct_mse + 1e-8)
            palette = jnp.array([0.0, 2.0 / 3.0, 1.0], dtype=jnp.float32)
            palette_idx = jnp.argmin(jnp.abs(pixel_direct_pred[:, :, None] - palette[None, None, :]), axis=2)
            palette_pred = palette[palette_idx]
            palette_mse = jnp.mean((palette_pred - patch_pixel_target) ** 2)
            palette_psnr = -10.0 * jnp.log10(palette_mse + 1e-8)

            pred_mag = jnp.abs(fft_pred_complex)
            pred_top_idx = jnp.argsort(pred_mag, axis=1)[:, -int(transform_info.get("fft_total_k", transform_info["fft_top_k"])) :]
            pred_mask = jnp.sum(jax.nn.one_hot(pred_top_idx, 16, dtype=jnp.float32), axis=1)
            true_mask = transform_info["fft_top_mask"][pair_idx]
            core_idx_local = jnp.array(transform_info.get("fft_basis_indices", []), dtype=jnp.int32)
            core_mask = jnp.sum(jax.nn.one_hot(core_idx_local, 16, dtype=jnp.float32), axis=0)
            freq_index_recovery = jnp.mean(jnp.sum(pred_mask * true_mask, axis=1) / (jnp.sum(true_mask, axis=1) + 1e-8))
            coeff_mae = jnp.mean(jnp.abs(fft_pred - sparse_fft_target))
            coeff_cos = jnp.mean(graph_cosine_batch_jax(fft_pred, sparse_fft_target))
            true_sparse_complex = transform_info["fft_sparse"][pair_idx]
            true_sparse_mag = jnp.abs(true_sparse_complex)
            coeff_mag_cos = jnp.mean(graph_cosine_batch_jax(pred_mag, true_sparse_mag))
            coeff_real_cos = jnp.mean(graph_cosine_batch_jax(jnp.real(fft_pred_complex), jnp.real(true_sparse_complex)))
            coeff_imag_cos = jnp.mean(graph_cosine_batch_jax(jnp.imag(fft_pred_complex), jnp.imag(true_sparse_complex)))
            phase_target = jnp.angle(transform_info["fft_sparse"][pair_idx])
            phase_pred = jnp.angle(fft_pred_complex)
            phase_weight = true_mask
            phase_recovery = jnp.mean(jnp.cos(phase_pred - phase_target) * phase_weight) / (jnp.mean(phase_weight) + 1e-8)
            gauge_inner = jnp.sum(true_sparse_complex * jnp.conj(fft_pred_complex) * true_mask, axis=1)
            gauge_theta = jnp.angle(gauge_inner)
            fft_pred_global_aligned = fft_pred_complex * jnp.exp(1j * gauge_theta)[:, None]
            phase_pred_global_aligned = jnp.angle(fft_pred_global_aligned)
            global_phase_aligned_imag_cos = jnp.mean(
                graph_cosine_batch_jax(jnp.imag(fft_pred_global_aligned), jnp.imag(true_sparse_complex))
            )
            global_phase_aligned_phase_cos = (
                jnp.mean(jnp.cos(phase_pred_global_aligned - phase_target) * phase_weight)
                / (jnp.mean(phase_weight) + 1e-8)
            )
            slot_inner = jnp.sum(true_sparse_complex * jnp.conj(fft_pred_complex) * true_mask, axis=0)
            slot_theta = jnp.angle(slot_inner)
            fft_pred_slot_aligned = fft_pred_complex * jnp.exp(1j * slot_theta)[None, :]
            per_slot_phase_aligned_coeff_cos = jnp.mean(
                graph_cosine_batch_jax(
                    jnp.concatenate([jnp.real(fft_pred_slot_aligned), jnp.imag(fft_pred_slot_aligned)], axis=1),
                    jnp.concatenate([jnp.real(true_sparse_complex), jnp.imag(true_sparse_complex)], axis=1),
                )
            )
            complex_gauge_error = jnp.mean(1.0 - jnp.cos(gauge_theta))
            true_top1_idx = jnp.argmax(true_sparse_mag, axis=1)
            pred_top1_idx = jnp.argmax(pred_mag, axis=1)
            top1_freq_acc = jnp.mean((pred_top1_idx == true_top1_idx).astype(jnp.float32))
            true_energy = true_sparse_mag * true_sparse_mag
            topk_energy_recall = jnp.mean(jnp.sum(true_energy * pred_mask, axis=1) / (jnp.sum(true_energy, axis=1) + 1e-8))
            true_phase_unit = jnp.exp(1j * phase_target)
            pred_phase_unit = jnp.exp(1j * phase_pred)
            retained_weight = true_mask
            retained_den = jnp.sum(retained_weight) + 1e-8
            angle_error = 1.0 - jnp.cos(phase_pred - phase_target)
            phase_cos_energy_weighted = jnp.sum(true_energy * jnp.cos(phase_pred - phase_target)) / (
                jnp.sum(true_energy) + 1e-8
            )
            support_weighted_angle_error = jnp.sum(angle_error * retained_weight) / retained_den
            energy_den = jnp.sum(true_energy) + 1e-8
            energy_weighted_complex_mse = jnp.sum((jnp.abs(fft_pred_complex - true_sparse_complex) ** 2) * true_energy) / energy_den
            true_imag_energy = (jnp.imag(true_sparse_complex) ** 2) * retained_weight
            pred_imag_energy = (jnp.imag(fft_pred_complex) ** 2) * retained_weight
            quadrature_energy_recall = jnp.sum(pred_imag_energy) / (jnp.sum(true_imag_energy) + 1e-8)
            imaginary_energy_recall = jnp.sum(jnp.minimum(pred_imag_energy, true_imag_energy)) / (jnp.sum(true_imag_energy) + 1e-8)
            complex_pair_norm_error = jnp.sum(jnp.abs(pred_mag - true_sparse_mag) * retained_weight) / retained_den
            phase_error_by_slot = jnp.sum(angle_error * retained_weight, axis=0) / (
                jnp.sum(retained_weight, axis=0) + 1e-8
            )
            hermitian_projected_pred = hermitian_project_fft4(fft_pred_complex)
            hermitian_error_before = hermitian_error_fft4(fft_pred_complex)
            hermitian_error_after = hermitian_error_fft4(hermitian_projected_pred)
            self_conjugate_imag_energy = self_conjugate_imag_energy_fft4(fft_pred_complex)
            pairwise_conjugate_consistency = 1.0 / (1.0 + hermitian_error_before)
            dc_mae = jnp.mean(jnp.abs(fft_pred_complex[:, 0] - true_sparse_complex[:, 0]))
            dc_relative_error = jnp.mean(
                jnp.abs(fft_pred_complex[:, 0] - true_sparse_complex[:, 0])
                / (jnp.abs(true_sparse_complex[:, 0]) + 1e-8)
            )
            low_freq_idx = jnp.array([0, 1, 4, 5], dtype=jnp.int32)
            mid_freq_idx = jnp.array([2, 3, 6, 7, 8, 9, 12], dtype=jnp.int32)
            low_freq_mae = jnp.mean(jnp.abs(fft_pred_complex[:, low_freq_idx] - true_sparse_complex[:, low_freq_idx]))
            mid_freq_mae = jnp.mean(jnp.abs(fft_pred_complex[:, mid_freq_idx] - true_sparse_complex[:, mid_freq_idx]))

            def psnr_from_fft(fft_values):
                recon = jnp.real(jnp.fft.ifft2(fft_values.reshape((-1, 4, 4))).reshape((-1, 16)))
                recon = jnp.clip(recon, 0.0, 1.0)
                mse = jnp.mean((recon - patch_pixel_target) ** 2)
                return -10.0 * jnp.log10(mse + 1e-8)

            psnr_true_support = psnr_from_fft(fft_pred_complex * true_mask)
            psnr_pred_support_true_coeffs = psnr_from_fft(true_sparse_complex * pred_mask)
            psnr_true_real_pred_imag = psnr_from_fft(
                (jnp.real(true_sparse_complex) + 1j * jnp.imag(fft_pred_complex)) * true_mask
            )
            psnr_pred_real_true_imag = psnr_from_fft(
                (jnp.real(fft_pred_complex) + 1j * jnp.imag(true_sparse_complex)) * true_mask
            )
            psnr_after_global_alignment = psnr_from_fft(fft_pred_global_aligned)
            psnr_true_magnitude = psnr_from_fft(true_sparse_mag * pred_phase_unit * true_mask)
            psnr_true_phase = psnr_from_fft(pred_mag * true_phase_unit * true_mask)
            shape_charge_error = jnp.float32(0.0)
            shape_charge_cos = jnp.float32(0.0)
            if shape_charge_pred is not None and shape_charge_target is not None:
                shape_charge_error = jnp.mean(jnp.linalg.norm(shape_charge_pred - shape_charge_target, axis=1))
                shape_charge_cos = jnp.mean(graph_cosine_batch_jax(shape_charge_pred, shape_charge_target))
            as_metrics = {}
            if transform_info.get("as_0_ttn", False):
                root_dim = int(transform_info.get("as_ttn_root_dim", 0))
                root_target = transform_target[:, :root_dim]
                root_pred = transform_pred[:, :root_dim]
                root_cos = jnp.mean(graph_cosine_batch_jax(root_pred, root_target))
                flat_baseline_psnr = jnp.float32(transform_info.get("as_flat_baseline_psnr", 0.0))
                as_metrics = {
                    "as_0_ttn_active": True,
                    "as_root_tensor_cosine": float(root_cos),
                    "as_charge_conservation_error": float(shape_charge_error),
                    "as_charge_cosine": float(shape_charge_cos),
                    "as_unfolded_coeff_psnr": float(pixel_psnr),
                    "as_flat_baseline_psnr": float(flat_baseline_psnr),
                    "as_flat_baseline_oracle_psnr": float(transform_info.get("as_flat_baseline_oracle_psnr", 0.0)),
                    "as_ttn_lift_over_flat_psnr": float(pixel_psnr - flat_baseline_psnr),
                    "as_decoder_path": "ttn_unfold" if transform_info.get("as_ttn_unfold_decoder", False) else "direct_signature",
                    "as_ttn_root_dim": int(transform_info.get("as_ttn_root_dim", 0)),
                    "as_ttn_internal_bond_dim": int(transform_info.get("as_ttn_internal_bond_dim", 0)),
                    "as_ttn_charge_dim": int(transform_info.get("as_ttn_charge_dim", 0)),
                }

            motif_chosen = nearest_class_indices_jax(
                source_pred[:, : transform_info["motif_code_catalog"].shape[1]],
                transform_info["motif_code_catalog"],
            )
            true_motif = transform_info["patch_motif_indices"][pair_idx]
            motif_exact = jnp.mean((motif_chosen == true_motif).astype(jnp.float32))
            patch_luma_target = jnp.mean(patch_pixel_target, axis=1)
            patch_luma_pred = jnp.mean(patch_pred, axis=1)
            dna_metrics = {}
            if transform_info.get("dna_checksum_mode", False):
                mag_mean_pred = jnp.mean(pred_mag, axis=1, keepdims=True)
                mag_std_pred = jnp.std(pred_mag, axis=1, keepdims=True) + 1e-8
                mag_pred = jnp.where(
                    pred_mag > mag_mean_pred + 0.35 * mag_std_pred,
                    2,
                    jnp.where(pred_mag > mag_mean_pred, 1, 0),
                ).astype(jnp.int32)
                phase_pred_state = jnp.floor(
                    (jnp.mod(phase_pred + jnp.pi, 2.0 * jnp.pi) / (2.0 * jnp.pi)) * 4.0
                ).astype(jnp.int32)
                phase_pred_state = jnp.clip(phase_pred_state, 0, 3)
                pred_residual_mask = jnp.clip(pred_mask - core_mask[None, :], 0.0, 1.0)
                pred_residual_mag = pred_mag * pred_residual_mask
                pred_residual_threshold = jnp.mean(
                    jnp.where(pred_residual_mask > 0.0, pred_residual_mag, 0.0), axis=1, keepdims=True
                )
                residual_pred_state = jnp.where(
                    core_mask[None, :] > 0.0,
                    1,
                    jnp.where(
                        pred_residual_mask > 0.0,
                        jnp.where(pred_residual_mag > pred_residual_threshold, 3, 2),
                        0,
                    ),
                ).astype(jnp.int32)
                pred_dominant_idx = jnp.argmax(pred_mag, axis=1)
                pred_dominant_mask = jax.nn.one_hot(pred_dominant_idx, 16, dtype=jnp.float32)
                support_pred_state = jnp.where(
                    pred_dominant_mask > 0.0,
                    3,
                    jnp.where(core_mask[None, :] > 0.0, 2, jnp.where(pred_residual_mask > 0.0, 1, 0)),
                ).astype(jnp.int32)
                mag_true = transform_info["dna_mag_state"][pair_idx]
                phase_true = transform_info["dna_phase_state"][pair_idx]
                residual_true = transform_info["dna_residual_state"][pair_idx]
                support_true = transform_info["dna_support_rank_state"][pair_idx]
                retained = transform_info["dna_retained_mask"][pair_idx]
                retained_den = jnp.sum(retained) + 1e-8

                mag_acc = jnp.sum(((mag_pred == mag_true).astype(jnp.float32) * retained)) / retained_den
                phase_acc = jnp.sum(((phase_pred_state == phase_true).astype(jnp.float32) * retained)) / retained_den
                residual_acc = jnp.sum(((residual_pred_state == residual_true).astype(jnp.float32) * retained)) / retained_den
                support_acc = jnp.sum(((support_pred_state == support_true).astype(jnp.float32) * retained)) / retained_den
                packet_match_slot = (
                    (mag_pred == mag_true)
                    & (phase_pred_state == phase_true)
                    & (residual_pred_state == residual_true)
                    & (support_pred_state == support_true)
                ).astype(jnp.float32)
                packet_exact = jnp.sum(packet_match_slot * retained) / retained_den

                motif_flat = jnp.repeat(motif_chosen.astype(jnp.int32), 16)
                support_flat = support_pred_state.reshape(-1).astype(jnp.int32)
                residual_flat = residual_pred_state.reshape(-1).astype(jnp.int32)
                phase_flat = phase_pred_state.reshape(-1).astype(jnp.int32)
                slot_flat = jnp.tile(jnp.arange(16, dtype=jnp.int32), (phase_pred_state.shape[0],))
                patch_err = jnp.mean(jnp.abs(patch_pred - patch_pixel_target), axis=1)
                err_state = (patch_err > jnp.mean(patch_err)).astype(jnp.int32)
                err_flat = jnp.repeat(err_state, 16)

                mi_header_support, mi_header_support_null, mi_header_support_p = token_mi_against_roll_null(
                    motif_flat, support_flat, 8, 4
                )
                mi_motif_residual, mi_motif_residual_null, mi_motif_residual_p = token_mi_against_roll_null(
                    motif_flat, residual_flat, 8, 4
                )
                mi_phase_error, mi_phase_error_null, mi_phase_error_p = token_mi_against_roll_null(
                    phase_flat, err_flat, 4, 2
                )
                mi_slot_class, mi_slot_class_null, mi_slot_class_p = token_mi_against_roll_null(
                    slot_flat, motif_flat, 16, 8
                )
                dna_metrics = {
                    "dna_token_active": True,
                    "dna_mag_state_acc": float(mag_acc),
                    "dna_phase_state_acc": float(phase_acc),
                    "dna_residual_state_acc": float(residual_acc),
                    "dna_support_rank_acc": float(support_acc),
                    "dna_instruction_packet_exact_match": float(packet_exact),
                    "dna_mi_header_support": float(mi_header_support),
                    "dna_mi_header_support_null": float(mi_header_support_null),
                    "dna_mi_header_support_gap": float(mi_header_support - mi_header_support_null),
                    "dna_mi_header_support_p_value": float(mi_header_support_p),
                    "dna_mi_motif_residual": float(mi_motif_residual),
                    "dna_mi_motif_residual_null": float(mi_motif_residual_null),
                    "dna_mi_motif_residual_gap": float(mi_motif_residual - mi_motif_residual_null),
                    "dna_mi_motif_residual_p_value": float(mi_motif_residual_p),
                    "dna_mi_phase_error": float(mi_phase_error),
                    "dna_mi_phase_error_null": float(mi_phase_error_null),
                    "dna_mi_phase_error_gap": float(mi_phase_error - mi_phase_error_null),
                    "dna_mi_phase_error_p_value": float(mi_phase_error_p),
                    "dna_mi_slot_class": float(mi_slot_class),
                    "dna_mi_slot_class_null": float(mi_slot_class_null),
                    "dna_mi_slot_class_gap": float(mi_slot_class - mi_slot_class_null),
                    "dna_mi_slot_class_p_value": float(mi_slot_class_p),
                }
            transform_metrics = {
                "fft_patch_motif_recovery": float(motif_exact),
                "fft_patch_psnr": float(pixel_psnr),
                "fft_oracle_psnr": float(transform_info.get("fft_oracle_psnr", 0.0)),
                "fft_coefficient_cosine": float(coeff_cos),
                "fft_coeff_mag_cosine": float(coeff_mag_cos),
                "fft_coeff_real_cosine": float(coeff_real_cos),
                "fft_coeff_imag_cosine": float(coeff_imag_cos),
                "fft_coefficient_mae": float(coeff_mae),
                "fft_frequency_index_recovery": float(freq_index_recovery),
                "fft_phase_recovery": float(phase_recovery),
                "fft_top1_frequency_accuracy": float(top1_freq_acc),
                "fft_topk_energy_recall": float(topk_energy_recall),
                "fft_support_weighted_angle_error": float(support_weighted_angle_error),
                "fft_energy_weighted_complex_mse": float(energy_weighted_complex_mse),
                "fft_quadrature_energy_recall": float(quadrature_energy_recall),
                "fft_imaginary_energy_recall": float(imaginary_energy_recall),
                "fft_complex_pair_norm_error": float(complex_pair_norm_error),
                "fft_phase_error_by_coefficient_slot": phase_error_by_slot.tolist(),
                "fft_hermitian_error_before": float(hermitian_error_before),
                "fft_hermitian_error_after": float(hermitian_error_after),
                "fft_self_conjugate_imag_energy": float(self_conjugate_imag_energy),
                "fft_pairwise_conjugate_consistency": float(pairwise_conjugate_consistency),
                "fft_phase_cos_energy_weighted": float(phase_cos_energy_weighted),
                "fft_dc_mae": float(dc_mae),
                "fft_dc_relative_error": float(dc_relative_error),
                "fft_low_freq_mae": float(low_freq_mae),
                "fft_mid_freq_edge_mae": float(mid_freq_mae),
                "fft_global_phase_aligned_imag_cos": float(global_phase_aligned_imag_cos),
                "fft_global_phase_aligned_phase_cos": float(global_phase_aligned_phase_cos),
                "fft_per_slot_phase_aligned_coeff_cos": float(per_slot_phase_aligned_coeff_cos),
                "fft_complex_gauge_error": float(complex_gauge_error),
                "fft_psnr_before_alignment": float(pixel_psnr),
                "fft_psnr_after_alignment": float(psnr_after_global_alignment),
                "fft_psnr_if_true_support": float(psnr_true_support),
                "fft_psnr_pred_support_true_coeffs": float(psnr_pred_support_true_coeffs),
                "fft_psnr_true_real_pred_imag": float(psnr_true_real_pred_imag),
                "fft_psnr_pred_real_true_imag": float(psnr_pred_real_true_imag),
                "fft_psnr_if_true_magnitudes": float(psnr_true_magnitude),
                "fft_psnr_if_true_phases": float(psnr_true_phase),
                "fft_shape_charge_error": float(shape_charge_error),
                "fft_shape_charge_cosine": float(shape_charge_cos),
                "fft_shape_charge_control": bool(ctrl_cfg.get("shape_charge_control", False)),
                "fft_top_k": int(transform_info["fft_top_k"]),
                "fft_core_k": int(transform_info.get("fft_core_k", transform_info["fft_top_k"])),
                "fft_residual_k": int(transform_info.get("fft_residual_k", 0)),
                "fft_total_k": int(transform_info.get("fft_total_k", transform_info["fft_top_k"])),
                "fft_codec_variant": str(transform_info.get("fft_codec_variant", "unknown")),
                "image_patch_motif_recovery": float(motif_exact),
                "image_patch_psnr": float(pixel_psnr),
                "image_patch_pixel_mse": float(pixel_mse),
                "image_patch_pixel_ridge_psnr": float(pixel_direct_psnr),
                "image_patch_pixel_ridge_mse": float(pixel_direct_mse),
                "image_patch_palette_prior_psnr": float(palette_psnr),
                "image_patch_palette_prior_mse": float(palette_mse),
                "image_patch_catalog_psnr": float(transform_info.get("fft_oracle_psnr", 0.0)),
                "image_patch_residual_mae": float(coeff_mae),
                "image_patch_luma_mae": float(jnp.mean(jnp.abs(patch_luma_pred - patch_luma_target))),
                "source_motif_recovery": float(motif_exact),
                "target_motif_recovery": float(motif_exact),
                "derived_next_accuracy": 0.0,
                "transform_class_recovery": float(freq_index_recovery),
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
            if getattr(args, "save_preview_png", False):
                preview_pred = pixel_direct_pred if getattr(args, "aq_yinyang_3", False) else patch_pred
                transform_metrics.update(
                    {
                        "preview_patch_pixels_target": patch_pixel_target.tolist(),
                        "preview_patch_pixels_pred": preview_pred.tolist(),
                        "preview_patch_pixels_pred_ifft": patch_pred.tolist(),
                        "preview_patch_pixels_pred_pixel_ridge": pixel_direct_pred.tolist(),
                        "preview_patch_pixels_pred_palette": palette_pred.tolist(),
                        "preview_patch_indices": pair_idx.tolist(),
                        "preview_frame_count": int(transform_info.get("frame_count", 0)),
                        "preview_frame_names": transform_info.get("frame_names", []),
                        "preview_patch_grid": transform_info.get("patch_grid", [4, 4]),
                    }
                )
            transform_metrics.update(dna_metrics)
            transform_metrics.update(as_metrics)
        elif transform_info.get("image_patch_codec", False):
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
            if getattr(args, "save_preview_png", False):
                transform_metrics.update(
                    {
                        "preview_patch_pixels_target": patch_pixel_target.tolist(),
                        "preview_patch_pixels_pred": pixel_pred.tolist(),
                        "preview_patch_indices": pair_idx.tolist(),
                        "preview_frame_count": int(transform_info.get("frame_count", 0)),
                        "preview_frame_names": transform_info.get("frame_names", []),
                        "preview_patch_grid": transform_info.get("patch_grid", [4, 4]),
                    }
                )
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
    if getattr(args, "aq_yinyang_4", False):
        return "semantic_yinyang_raw_pixel_state_transport_aq_yinyang_4"
    if getattr(args, "aq_yinyang_3", False):
        return "semantic_yinyang_one_frame_pixel_decoder_aq_yinyang_3"
    if getattr(args, "aq_yinyang_2", False):
        return "semantic_yinyang_one_frame_all_coeff_transport_aq_yinyang_2"
    if getattr(args, "aq_yinyang_1", False):
        return "semantic_yinyang_one_frame_transport_aq_yinyang_1"
    if getattr(args, "aq_yinyang_0", False):
        return "semantic_yinyang_two_frame_transport_aq_yinyang_0"
    if getattr(args, "as_0b", False):
        return "hierarchical_instruction_genome_unfold_decoder_as_0b"
    if getattr(args, "as_0", False):
        return "hierarchical_instruction_genome_transport_as_0"
    if getattr(args, "aq_hybrid_0", False):
        return "semantic_kspace_hybrid_continuous_checksum_aq_hybrid_0"
    if getattr(args, "aq_dna_0", False):
        return "semantic_kspace_dna_token_genome_aq_dna_0"
    if getattr(args, "aq_fft_2", False):
        return "semantic_kspace_shape_charge_control_aq_fft_2"
    if getattr(args, "aq_fft_7", False):
        return "semantic_kspace_residual_capacity_scaling_aq_fft_7"
    if getattr(args, "aq_fft_6", False):
        return "semantic_kspace_hermitian_complex_decoder_aq_fft_6"
    if getattr(args, "aq_fft_5", False):
        return "semantic_kspace_gauge_aligned_complex_decoding_aq_fft_5"
    if getattr(args, "aq_fft_4", False):
        return "semantic_kspace_complex_orientation_preservation_aq_fft_4"
    if getattr(args, "aq_fft_3", False):
        return "semantic_kspace_direct_complex_oscillator_aq_fft_3"
    if getattr(args, "aq_fft_1", False):
        return "semantic_kspace_genome_codec_aq_fft_1"
    if getattr(args, "aq_fft_0", False):
        return "semantic_kspace_patch_codec_aq_fft_0"
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
    if getattr(args, "aq_fft_7", False):
        args.aq_fft_residual_k = 4
    if getattr(args, "aq_yinyang_0", False):
        args.aq_fft_residual_k = 4
    if getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False):
        args.aq_fft_top_k = 16
        args.aq_fft_residual_k = 0
    if getattr(args, "aq_yinyang_1", False):
        args.aq_fft_residual_k = 4

    use_aq_payload = (
        getattr(args, "aq_0b", False)
        or getattr(args, "aq_1a_tf", False)
        or getattr(args, "aq_1a_tf_s29", False)
        or getattr(args, "aq_1b", False)
        or getattr(args, "aq_1c", False)
        or getattr(args, "aq_seq_0", False)
        or getattr(args, "as_0b", False)
        or getattr(args, "as_0", False)
        or getattr(args, "aq_yinyang_4", False)
        or getattr(args, "aq_yinyang_3", False)
        or getattr(args, "aq_yinyang_2", False)
        or getattr(args, "aq_yinyang_1", False)
        or getattr(args, "aq_yinyang_0", False)
        or getattr(args, "aq_hybrid_0", False)
        or getattr(args, "aq_dna_0", False)
        or getattr(args, "aq_fft_7", False)
        or getattr(args, "aq_fft_6", False)
        or getattr(args, "aq_fft_5", False)
        or getattr(args, "aq_fft_4", False)
        or getattr(args, "aq_fft_3", False)
        or getattr(args, "aq_fft_2", False)
        or getattr(args, "aq_fft_1", False)
        or getattr(args, "aq_fft_0", False)
        or getattr(args, "aq_img_0", False)
        or getattr(args, "aq_img_1", False)
        or getattr(args, "aq_img_1_lite", False)
    )
    motifs, _ = build_semantic_motifs(args.sequence_length, aq_payload=use_aq_payload)
    motif_names = list(motifs.keys())
    seqs = jnp.stack([motifs[name] for name in motif_names])
    features = jnp.stack([motif_feature(seq) for seq in seqs])
    graph_features = jnp.stack([motif_transition_graph(seq) for seq in seqs])
    layer_features = build_layer_feature_table(seqs)
    transform_info = None
    if (
        getattr(args, "aq_hybrid_0", False)
        or getattr(args, "aq_yinyang_4", False)
        or getattr(args, "aq_yinyang_3", False)
        or getattr(args, "aq_yinyang_2", False)
        or getattr(args, "aq_yinyang_1", False)
        or getattr(args, "aq_yinyang_0", False)
        or getattr(args, "as_0b", False)
        or getattr(args, "as_0", False)
        or getattr(args, "aq_dna_0", False)
        or getattr(args, "aq_fft_7", False)
        or getattr(args, "aq_fft_6", False)
        or getattr(args, "aq_fft_5", False)
        or getattr(args, "aq_fft_4", False)
        or getattr(args, "aq_fft_3", False)
        or getattr(args, "aq_fft_2", False)
        or getattr(args, "aq_fft_1", False)
        or getattr(args, "aq_fft_0", False)
    ):
        if getattr(args, "aq_yinyang_4", False):
            seqs, features, graph_features, layer_features, transform_info = build_yinyang_raw_patch_payload(
                args.sequence_length,
                frame_count=args.yinyang_frame_count,
                direct_complex=True,
                frame_size=16,
                frame_start=args.yinyang_frame_start,
            )
        elif getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False) or getattr(args, "aq_yinyang_1", False) or getattr(args, "aq_yinyang_0", False):
            seqs, features, graph_features, layer_features, transform_info = build_yinyang_fft_patch_codec_payload(
                args.sequence_length,
                top_k=args.aq_fft_top_k,
                residual_k=args.aq_fft_residual_k,
                direct_complex=True,
                dna_instruction=True,
                frame_count=1
                if (getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False) or getattr(args, "aq_yinyang_1", False))
                else 2,
            )
        else:
            seqs, features, graph_features, layer_features, transform_info = build_fft_patch_codec_payload(
                args.sequence_length,
                top_k=args.aq_fft_top_k,
                residual_k=args.aq_fft_residual_k
                if (
                    getattr(args, "aq_hybrid_0", False)
                    or getattr(args, "as_0b", False)
                    or getattr(args, "as_0", False)
                    or getattr(args, "aq_dna_0", False)
                    or getattr(args, "aq_fft_7", False)
                    or getattr(args, "aq_fft_6", False)
                    or getattr(args, "aq_fft_5", False)
                    or getattr(args, "aq_fft_4", False)
                    or getattr(args, "aq_fft_3", False)
                    or getattr(args, "aq_fft_2", False)
                    or getattr(args, "aq_fft_1", False)
                )
                else 0,
                direct_complex=getattr(args, "aq_fft_7", False) or getattr(args, "aq_fft_6", False) or getattr(args, "aq_fft_5", False) or getattr(args, "aq_fft_4", False) or getattr(args, "aq_fft_3", False),
                dna_instruction=getattr(args, "aq_hybrid_0", False)
                or getattr(args, "as_0b", False)
                or getattr(args, "as_0", False)
                or getattr(args, "aq_dna_0", False)
                or getattr(args, "aq_fft_7", False)
                or getattr(args, "aq_fft_6", False)
                or getattr(args, "aq_fft_5", False)
                or getattr(args, "aq_fft_4", False)
                or getattr(args, "aq_fft_3", False)
                or getattr(args, "aq_fft_2", False)
                or getattr(args, "aq_fft_1", False),
                dna_tokenized=getattr(args, "aq_dna_0", False),
                dna_checksum_mode=getattr(args, "aq_hybrid_0", False) or getattr(args, "aq_dna_0", False),
                ttn_folded=getattr(args, "as_0", False) or getattr(args, "as_0b", False),
                ttn_unfold_decoder=getattr(args, "as_0b", False),
                patch_limit=16,
            )
        if getattr(args, "as_0b", False):
            print("AS-0b ACTIVE:")
            print("  payload = AQ-FFT-1 instruction packets")
            print("  geometry = binary TTN over coefficient/residual packets")
            print("  transported_object = root tensor + selected internal bonds + charge sectors")
            print("  decoder = recovered TTN state -> coefficient field unfold")
            print("  baseline = flat AQ-FFT-1 free path")
            print("  controller = free")
        if getattr(args, "as_0", False):
            print("AS-0 ACTIVE:")
            print("  payload = AQ-FFT-1 instruction packets")
            print("  geometry = binary TTN over coefficient/residual packets")
            print("  transported_object = root tensor + selected internal bonds + charge sectors")
            print("  baseline = flat AQ-FFT-1 free path")
            print("  controller = free")
        if getattr(args, "aq_hybrid_0", False):
            print("AQ-HYBRID-0 ACTIVE:")
            print("  primary_payload = continuous k-space geometry")
            print("  token_side_channel = checksum/consistency")
            print("  patch_order = Hilbert-4x4")
            print("  kspace_order = zigzag/local fixed core + adaptive residual")
            print("  token_metrics = enabled")
            print("  permutation_nulls = enabled")
        if getattr(args, "aq_dna_0", False):
            print("AQ-DNA-0 ACTIVE:")
            print("  tokenization = deterministic GMM-lite bins")
            print("  patch_order = Hilbert-4x4")
            print("  kspace_order = zigzag/local fixed core + adaptive residual")
            print("  token_metrics = enabled")
            print("  permutation_nulls = enabled")
            print("  runtime_envelope != AQ-FFT-1")
        if getattr(args, "aq_yinyang_0", False):
            print("AQ-YINYANG-0 ACTIVE:")
            print("  frames = two 16x16 8-bit tai-chi-tu polarity frames")
            print("  tiling = 4x4 patches per frame, 32 patch states total")
            print("  codec = direct complex k-space with residual_K=4")
            print("  previews = reference/recovered/error frame PNGs")
        if getattr(args, "aq_yinyang_1", False):
            print("AQ-YINYANG-1 ACTIVE:")
            print("  frames = one 16x16 8-bit tai-chi-tu frame")
            print("  tiling = 4x4 patches, 16 patch states total")
            print("  codec = direct complex k-space with residual_K=4")
            print("  purpose = isolate codec ceiling before temporal polarity sequence")
        if getattr(args, "aq_yinyang_2", False):
            print("AQ-YINYANG-2 ACTIVE:")
            print("  frames = one 16x16 8-bit tai-chi-tu frame")
            print("  tiling = 4x4 patches, 16 patch states total")
            print("  codec = direct complex k-space with all 16 FFT coefficients per patch")
            print("  purpose = one-image lossless-codec transport gate")
        if getattr(args, "aq_yinyang_3", False):
            print("AQ-YINYANG-3 ACTIVE:")
            print("  frames = one 16x16 8-bit tai-chi-tu frame")
            print("  tiling = 4x4 patches, 16 patch states total")
            print("  codec = all 16 FFT coefficients per patch")
            print("  decoder = direct pixel-space ridge + palette prior diagnostics")
        if getattr(args, "aq_yinyang_4", False):
            print("AQ-YINYANG-4 ACTIVE:")
            print(
                f"  frames = {args.yinyang_frame_count} 16x16 tai-chi-tu reference PNG frame(s) "
                f"starting at {args.yinyang_frame_start}"
            )
            print(f"  tiling = 4x4 raw pixel patches, {args.yinyang_frame_count * 16} patch states total")
            print("  codec = no FFT/no folding")
            print("  transport = direct raw-pixel oscillator slots")
            print("  controller = complex_pair_norm")
            print(f"  recovery_order = {'ordered' if args.yinyang_ordered_recovery else 'permuted'}")
        if getattr(args, "aq_fft_3", False):
            print("AQ-FFT-3 ACTIVE:")
            print("  codec = AQ-FFT-1 fixed core + residual K-space")
            print("  coefficient_encoding = direct complex oscillator slots")
            print("  direct_complex_x0 = true")
            print("  decoder_target = sparse FFT Re/Im coefficients")
            print("  baseline = AQ-FFT-1 flat path at ~17.00 dB")
        if getattr(args, "aq_fft_4", False):
            print("AQ-FFT-4 ACTIVE:")
            print("  codec = AQ-FFT-1 fixed core K=6 + residual K=2")
            print("  coefficient_encoding = direct complex oscillator slots")
            print("  controllers = free, complex_pair_norm, complex_phase_locked")
            print("  diagnostics = complex orientation + error budget")
            print("  alpha065 = excluded")
        if getattr(args, "aq_fft_5", False):
            print("AQ-FFT-5 ACTIVE:")
            print("  codec = AQ-FFT-4 complex_pair_norm reference")
            print("  coefficient_encoding = direct complex oscillator slots")
            print("  controllers = complex_pair_norm, complex_pair_norm_gaugefix")
            print("  diagnostics = oracle gauge alignment + anchor gauge-fix decoder")
            print("  alpha065 = excluded")
        if getattr(args, "aq_fft_6", False):
            print("AQ-FFT-6 ACTIVE:")
            print("  codec = AQ-FFT-4 complex_pair_norm reference")
            print("  coefficient_encoding = direct complex oscillator slots")
            print("  controllers = complex_pair_norm, complex_pair_norm_hermitian")
            print("  diagnostics = Hermitian symmetry + energy-weighted phase")
            print("  alpha065 = excluded")
        if getattr(args, "aq_fft_7", False):
            print("AQ-FFT-7 ACTIVE:")
            print("  codec = complex_pair_norm residual capacity scaling")
            print("  coefficient_encoding = direct complex oscillator slots")
            print("  controller = complex_pair_norm")
            print("  residual_K = 4")
            print("  baseline = residual_K=2 at ~23.18 dB realized / 25.05 dB oracle")
            print("  alpha065 = excluded")
        print("payload_seqs sample:", jax.device_get(seqs[0]).tolist())
        print("fft basis indices:", transform_info.get("fft_basis_indices", []))
        print("fft residual-k:", transform_info.get("fft_residual_k", 0))
        fft_top_sample = transform_info.get("fft_top_indices")
        if fft_top_sample is not None and jnp.asarray(fft_top_sample).size > 0:
            print("fft top-k sample:", jax.device_get(fft_top_sample[0]).tolist())
        else:
            print("fft top-k sample:", [])
        motif_names = transform_info["pair_names"]
    elif getattr(args, "aq_img_0", False) or getattr(args, "aq_img_1", False) or getattr(args, "aq_img_1_lite", False):
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
        "complex_pair_norm": {
            "damping": args.damping,
            "feedback": 0.00,
            "tunnel": 0.00,
            "complex_pair_norm": True,
            "gate_cost": 0.02,
            "lane": "complex_geometry",
        },
        "complex_phase_locked": {
            "damping": args.damping,
            "feedback": 0.00,
            "tunnel": 0.00,
            "complex_phase_locked": True,
            "gate_cost": 0.02,
            "lane": "complex_geometry",
        },
        "complex_pair_norm_gaugefix": {
            "damping": args.damping,
            "feedback": 0.00,
            "tunnel": 0.00,
            "complex_pair_norm": True,
            "complex_gaugefix": True,
            "gate_cost": 0.03,
            "lane": "complex_geometry",
        },
        "complex_pair_norm_hermitian": {
            "damping": args.damping,
            "feedback": 0.00,
            "tunnel": 0.00,
            "complex_pair_norm": True,
            "complex_hermitian": True,
            "gate_cost": 0.03,
            "lane": "complex_geometry",
        },
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
        "shape_charge_control": {
            "damping": args.damping,
            "feedback": 0.00,
            "tunnel": 0.00,
            "shape_charge_control": True,
            "shape_charge_eta": 0.32,
            "gate_cost": 0.04,
            "lane": "shape_charge",
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
        or getattr(args, "as_0b", False)
        or getattr(args, "as_0", False)
        or getattr(args, "aq_yinyang_4", False)
        or getattr(args, "aq_yinyang_3", False)
        or getattr(args, "aq_yinyang_2", False)
        or getattr(args, "aq_yinyang_1", False)
        or getattr(args, "aq_yinyang_0", False)
        or getattr(args, "aq_hybrid_0", False)
        or getattr(args, "aq_dna_0", False)
        or getattr(args, "aq_fft_7", False)
        or getattr(args, "aq_fft_6", False)
        or getattr(args, "aq_fft_5", False)
        or getattr(args, "aq_fft_4", False)
        or getattr(args, "aq_fft_3", False)
        or getattr(args, "aq_fft_2", False)
        or getattr(args, "aq_fft_1", False)
        or getattr(args, "aq_fft_0", False)
        or getattr(args, "aq_img_0", False)
        or getattr(args, "aq_img_1", False)
        or getattr(args, "aq_img_1_lite", False)
    ):
        if (
            getattr(args, "as_0b", False)
            or getattr(args, "as_0", False)
            or getattr(args, "aq_hybrid_0", False)
            or getattr(args, "aq_dna_0", False)
        ):
            controllers = {"free": controllers["free"]}
        elif getattr(args, "aq_fft_2", False):
            controllers = {
                "free": controllers["free"],
                "shape_charge_control": controllers["shape_charge_control"],
            }
        elif getattr(args, "aq_fft_4", False):
            controllers = {
                "free": controllers["free"],
                "complex_pair_norm": controllers["complex_pair_norm"],
                "complex_phase_locked": controllers["complex_phase_locked"],
            }
        elif getattr(args, "aq_fft_5", False):
            controllers = {
                "complex_pair_norm": controllers["complex_pair_norm"],
                "complex_pair_norm_gaugefix": controllers["complex_pair_norm_gaugefix"],
            }
        elif getattr(args, "aq_fft_6", False):
            controllers = {
                "complex_pair_norm": controllers["complex_pair_norm"],
                "complex_pair_norm_hermitian": controllers["complex_pair_norm_hermitian"],
            }
        elif (
            getattr(args, "aq_yinyang_4", False)
            or getattr(args, "aq_yinyang_3", False)
            or getattr(args, "aq_yinyang_2", False)
            or getattr(args, "aq_yinyang_1", False)
            or getattr(args, "aq_yinyang_0", False)
            or getattr(args, "aq_fft_7", False)
        ):
            controllers = {
                "complex_pair_norm": controllers["complex_pair_norm"],
            }
        elif getattr(args, "aq_fft_3", False) or getattr(args, "aq_fft_1", False) or getattr(args, "aq_fft_0", False):
            controllers = {
                "free": controllers["free"],
                "tunnel_eigen_residual_alpha065": controllers["tunnel_eigen_residual_alpha065"],
            }
        else:
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

    preview_pngs = []
    if getattr(args, "save_preview_png", False):
        preview_pngs = write_recovery_preview_pngs(records, args.out_dir)
        for preview_path in preview_pngs:
            print(f"[PREVIEW] wrote {preview_path}")

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
        "preview_pngs": preview_pngs,
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
            or getattr(args, "as_0b", False)
            or getattr(args, "as_0", False)
            or getattr(args, "aq_yinyang_3", False)
            or getattr(args, "aq_yinyang_2", False)
            or getattr(args, "aq_yinyang_1", False)
            or getattr(args, "aq_yinyang_0", False)
            or getattr(args, "aq_fft_7", False)
            or getattr(args, "aq_fft_6", False)
            or getattr(args, "aq_fft_5", False)
            or getattr(args, "aq_fft_4", False)
            or getattr(args, "aq_fft_3", False)
            or getattr(args, "aq_fft_2", False)
            or getattr(args, "aq_fft_1", False)
            or getattr(args, "aq_fft_0", False)
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
                or getattr(args, "aq_yinyang_3", False)
                or getattr(args, "aq_yinyang_2", False)
                or getattr(args, "aq_yinyang_1", False)
                or getattr(args, "aq_yinyang_0", False)
                or getattr(args, "aq_fft_7", False)
                or getattr(args, "aq_fft_6", False)
                or getattr(args, "aq_fft_5", False)
                or getattr(args, "aq_fft_4", False)
                or getattr(args, "aq_fft_3", False)
                or getattr(args, "aq_fft_1", False)
                or getattr(args, "aq_fft_0", False)
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
        if (
            getattr(args, "as_0b", False)
            or getattr(args, "as_0", False)
            or getattr(args, "aq_hybrid_0", False)
            or getattr(args, "aq_dna_0", False)
            or getattr(args, "aq_yinyang_3", False)
            or getattr(args, "aq_yinyang_2", False)
            or getattr(args, "aq_yinyang_1", False)
            or getattr(args, "aq_yinyang_0", False)
            or getattr(args, "aq_fft_7", False)
            or getattr(args, "aq_fft_6", False)
            or getattr(args, "aq_fft_5", False)
            or getattr(args, "aq_fft_4", False)
            or getattr(args, "aq_fft_3", False)
            or getattr(args, "aq_fft_2", False)
            or getattr(args, "aq_fft_1", False)
            or getattr(args, "aq_fft_0", False)
        ):
            print("\n" + "=" * 112)
            if getattr(args, "as_0b", False):
                title = "AS-0b TTN unfold-decoder instruction genome readout"
            elif getattr(args, "as_0", False):
                title = "AS-0 TTN folded instruction genome readout"
            elif getattr(args, "aq_hybrid_0", False):
                title = "AQ-HYBRID-0 continuous-geometry + token-checksum readout"
            elif getattr(args, "aq_dna_0", False):
                title = "AQ-DNA-0 locality-preserving token genome readout"
            elif getattr(args, "aq_yinyang_3", False):
                title = "AQ-YINYANG-3 one-frame pixel-space decoder readout"
            elif getattr(args, "aq_yinyang_2", False):
                title = "AQ-YINYANG-2 one-frame all-coefficient visual transport readout"
            elif getattr(args, "aq_yinyang_1", False):
                title = "AQ-YINYANG-1 one-frame visual transport readout"
            elif getattr(args, "aq_yinyang_0", False):
                title = "AQ-YINYANG-0 two-frame visual transport readout"
            elif getattr(args, "aq_fft_2", False):
                title = "AQ-FFT-2 shape-charge controlled k-space codec readout"
            elif getattr(args, "aq_fft_7", False):
                title = "AQ-FFT-7 residual-capacity scaling readout"
            elif getattr(args, "aq_fft_6", False):
                title = "AQ-FFT-6 Hermitian-constrained complex decoder readout"
            elif getattr(args, "aq_fft_5", False):
                title = "AQ-FFT-5 gauge-aligned complex decoder readout"
            elif getattr(args, "aq_fft_4", False):
                title = "AQ-FFT-4 complex-orientation preservation codec readout"
            elif getattr(args, "aq_fft_3", False):
                title = "AQ-FFT-3 direct-complex oscillator codec readout"
            elif getattr(args, "aq_fft_1", False):
                title = "AQ-FFT-1 core+residual genome codec readout"
            else:
                title = "AQ-FFT-0 k-space image patch codec readout"
            print(f"{title:^112}")
            for rec in records:
                for _noise_level, seed_map in rec.get("semantic_recovery", {}).items():
                    for _seed, ctrl_map in seed_map.items():
                        for ctrl, metrics in ctrl_map.items():
                            tf = metrics.get("transform_field", {})
                            print(
                                f"{ctrl:>23} "
                                f"K={tf.get('fft_total_k', tf.get('fft_top_k', 0)):>2} "
                                f"core={tf.get('fft_core_k', tf.get('fft_top_k', 0)):>2} "
                                f"resid={tf.get('fft_residual_k', 0):>2} "
                                f"motif={tf.get('fft_patch_motif_recovery', 0.0):.4f}"
                            )
                            print(
                                f"{'':>23} "
                                f"ifft_psnr={tf.get('fft_patch_psnr', 0.0):.4f} "
                                f"oracle_psnr={tf.get('fft_oracle_psnr', 0.0):.4f}"
                            )
                            if getattr(args, "aq_yinyang_3", False):
                                print(
                                    f"{'':>23} "
                                    f"pixel_ridge_psnr={tf.get('image_patch_pixel_ridge_psnr', 0.0):.4f} "
                                    f"palette_prior_psnr={tf.get('image_patch_palette_prior_psnr', 0.0):.4f} "
                                    f"dc_mae={tf.get('fft_dc_mae', 0.0):.4f} "
                                    f"low_mae={tf.get('fft_low_freq_mae', 0.0):.4f} "
                                    f"mid_mae={tf.get('fft_mid_freq_edge_mae', 0.0):.4f}"
                                )
                            print(
                                f"{'':>23} "
                                f"coeff_mag_cos={tf.get('fft_coeff_mag_cosine', 0.0):.4f} "
                                f"coeff_real_cos={tf.get('fft_coeff_real_cosine', 0.0):.4f} "
                                f"coeff_imag_cos={tf.get('fft_coeff_imag_cosine', 0.0):.4f}"
                            )
                            print(
                                f"{'':>23} "
                                f"phase_cos={tf.get('fft_phase_recovery', 0.0):.4f} "
                                f"freq_idx_overlap={tf.get('fft_frequency_index_recovery', 0.0):.4f} "
                                f"top1_freq_acc={tf.get('fft_top1_frequency_accuracy', 0.0):.4f} "
                                f"topK_energy_recall={tf.get('fft_topk_energy_recall', 0.0):.4f}"
                            )
                            if getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False) or getattr(args, "aq_yinyang_1", False) or getattr(args, "aq_yinyang_0", False) or getattr(args, "aq_fft_7", False) or getattr(args, "aq_fft_6", False) or getattr(args, "aq_fft_5", False) or getattr(args, "aq_fft_4", False):
                                print(
                                    f"{'':>23} "
                                    f"angle_err={tf.get('fft_support_weighted_angle_error', 0.0):.4f} "
                                    f"energy_mse={tf.get('fft_energy_weighted_complex_mse', 0.0):.4f} "
                                    f"quad_recall={tf.get('fft_quadrature_energy_recall', 0.0):.4f} "
                                    f"imag_recall={tf.get('fft_imaginary_energy_recall', 0.0):.4f} "
                                    f"pair_norm_err={tf.get('fft_complex_pair_norm_error', 0.0):.4f}"
                                )
                            if getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False) or getattr(args, "aq_yinyang_1", False) or getattr(args, "aq_yinyang_0", False) or getattr(args, "aq_fft_7", False) or getattr(args, "aq_fft_6", False):
                                print(
                                    f"{'':>23} "
                                    f"herm_before={tf.get('fft_hermitian_error_before', 0.0):.4f} "
                                    f"herm_after={tf.get('fft_hermitian_error_after', 0.0):.4f} "
                                    f"self_imag_E={tf.get('fft_self_conjugate_imag_energy', 0.0):.4f} "
                                    f"pair_consistency={tf.get('fft_pairwise_conjugate_consistency', 0.0):.4f} "
                                    f"phase_Ew={tf.get('fft_phase_cos_energy_weighted', 0.0):.4f}"
                                )
                            if getattr(args, "aq_fft_5", False):
                                print(
                                    f"{'':>23} "
                                    f"aligned_imag_cos={tf.get('fft_global_phase_aligned_imag_cos', 0.0):.4f} "
                                    f"aligned_phase_cos={tf.get('fft_global_phase_aligned_phase_cos', 0.0):.4f} "
                                    f"slot_aligned_coeff_cos={tf.get('fft_per_slot_phase_aligned_coeff_cos', 0.0):.4f} "
                                    f"gauge_err={tf.get('fft_complex_gauge_error', 0.0):.4f}"
                                )
                                print(
                                    f"{'':>23} "
                                    f"psnr_before_align={tf.get('fft_psnr_before_alignment', 0.0):.4f} "
                                    f"psnr_after_align={tf.get('fft_psnr_after_alignment', 0.0):.4f}"
                                )
                            if getattr(args, "aq_fft_4", False):
                                print(
                                    f"{'':>23} "
                                    f"budget oracle={tf.get('fft_oracle_psnr', 0.0):.4f} "
                                    f"true_support_pred_coeffs={tf.get('fft_psnr_if_true_support', 0.0):.4f} "
                                    f"pred_support_true_coeffs={tf.get('fft_psnr_pred_support_true_coeffs', 0.0):.4f} "
                                    f"true_real_pred_imag={tf.get('fft_psnr_true_real_pred_imag', 0.0):.4f} "
                                    f"pred_real_true_imag={tf.get('fft_psnr_pred_real_true_imag', 0.0):.4f} "
                                    f"final={tf.get('fft_patch_psnr', 0.0):.4f}"
                                )
                            if getattr(args, "aq_yinyang_3", False) or getattr(args, "aq_yinyang_2", False) or getattr(args, "aq_yinyang_1", False) or getattr(args, "aq_yinyang_0", False) or getattr(args, "aq_fft_7", False) or getattr(args, "aq_fft_6", False) or getattr(args, "aq_fft_5", False) or getattr(args, "aq_fft_4", False) or getattr(args, "aq_fft_3", False):
                                print(
                                    f"{'':>23} "
                                    f"codec_variant={tf.get('fft_codec_variant', 'unknown')} "
                                    f"direct_complex_x0=1"
                                )
                            if getattr(args, "aq_fft_2", False):
                                print(
                                    f"{'':>23} "
                                    f"shape_err={tf.get('fft_shape_charge_error', 0.0):.4f} "
                                    f"shape_cos={tf.get('fft_shape_charge_cosine', 0.0):.4f} "
                                    f"psnr_true_support={tf.get('fft_psnr_if_true_support', 0.0):.4f} "
                                    f"psnr_true_mag={tf.get('fft_psnr_if_true_magnitudes', 0.0):.4f} "
                                    f"psnr_true_phase={tf.get('fft_psnr_if_true_phases', 0.0):.4f}"
                                )
                            if getattr(args, "as_0b", False) or getattr(args, "as_0", False):
                                print(
                                    f"{'':>23} "
                                    f"root_cos={tf.get('as_root_tensor_cosine', 0.0):.4f} "
                                    f"charge_err={tf.get('as_charge_conservation_error', 0.0):.4f} "
                                    f"charge_cos={tf.get('as_charge_cosine', 0.0):.4f}"
                                )
                                print(
                                    f"{'':>23} "
                                    f"flat_psnr={tf.get('as_flat_baseline_psnr', 0.0):.4f} "
                                    f"ttn_lift={tf.get('as_ttn_lift_over_flat_psnr', 0.0):.4f} "
                                    f"decoder={tf.get('as_decoder_path', 'unknown')} "
                                    f"root_dim={tf.get('as_ttn_root_dim', 0)} "
                                    f"bond_dim={tf.get('as_ttn_internal_bond_dim', 0)}"
                                )
                            if getattr(args, "aq_hybrid_0", False) or getattr(args, "aq_dna_0", False):
                                print(
                                    f"{'':>23} "
                                    f"packet_exact={tf.get('dna_instruction_packet_exact_match', 0.0):.4f} "
                                    f"mag_acc={tf.get('dna_mag_state_acc', 0.0):.4f} "
                                    f"phase_acc={tf.get('dna_phase_state_acc', 0.0):.4f} "
                                    f"resid_acc={tf.get('dna_residual_state_acc', 0.0):.4f} "
                                    f"rank_acc={tf.get('dna_support_rank_acc', 0.0):.4f}"
                                )
                                print(
                                    f"{'':>23} "
                                    f"MI_header_support_gap={tf.get('dna_mi_header_support_gap', 0.0):.4f} "
                                    f"MI_motif_resid_gap={tf.get('dna_mi_motif_residual_gap', 0.0):.4f} "
                                    f"MI_phase_error_gap={tf.get('dna_mi_phase_error_gap', 0.0):.4f} "
                                    f"MI_slot_class_gap={tf.get('dna_mi_slot_class_gap', 0.0):.4f}"
                                )
                                print(
                                    f"{'':>23} "
                                    f"p_header_support={tf.get('dna_mi_header_support_p_value', 1.0):.4f} "
                                    f"p_motif_resid={tf.get('dna_mi_motif_residual_p_value', 1.0):.4f} "
                                    f"p_phase_error={tf.get('dna_mi_phase_error_p_value', 1.0):.4f} "
                                    f"p_slot_class={tf.get('dna_mi_slot_class_p_value', 1.0):.4f}"
                                )
            print("=" * 112)
        if (
            getattr(args, "aq_yinyang_4", False)
            or getattr(args, "aq_img_0", False)
            or getattr(args, "aq_img_1", False)
            or getattr(args, "aq_img_1_lite", False)
        ):
            print("\n" + "=" * 112)
            if getattr(args, "aq_yinyang_4", False):
                title = "AQ-YINYANG-4 raw-pixel image-state transport readout"
            elif getattr(args, "aq_img_1_lite", False):
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
        "--save-preview-png",
        action="store_true",
        help="Save compact target/recovered/error PNG contact sheets for AQ image and FFT patch codecs.",
    )
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
        "--aq-fft-0",
        dest="aq_fft_0",
        action="store_true",
        help="Run Program AQ-FFT-0: noise-free fixed-basis FFT/k-space image patch codec gate with sin/cos phase channels.",
    )
    parser.add_argument(
        "--aq-fft-1",
        dest="aq_fft_1",
        action="store_true",
        help="Run Program AQ-FFT-1: fixed core plus sparse residual k-space genome codec gate.",
    )
    parser.add_argument(
        "--aq-fft-2",
        dest="aq_fft_2",
        action="store_true",
        help="Run Program AQ-FFT-2: shape-charge controlled fixed-core plus residual k-space codec gate.",
    )
    parser.add_argument(
        "--aq-fft-3",
        dest="aq_fft_3",
        action="store_true",
        help="Run Program AQ-FFT-3: direct complex FFT coefficients injected into D-LinOSS oscillator slots.",
    )
    parser.add_argument(
        "--aq-fft-4",
        dest="aq_fft_4",
        action="store_true",
        help="Run Program AQ-FFT-4: complex-orientation preservation gate for direct complex k-space transport.",
    )
    parser.add_argument(
        "--aq-fft-5",
        dest="aq_fft_5",
        action="store_true",
        help="Run Program AQ-FFT-5: gauge-aligned complex decoding gate for direct complex k-space transport.",
    )
    parser.add_argument(
        "--aq-fft-6",
        dest="aq_fft_6",
        action="store_true",
        help="Run Program AQ-FFT-6: Hermitian-constrained complex decoder gate for real-image k-space transport.",
    )
    parser.add_argument(
        "--aq-fft-7",
        dest="aq_fft_7",
        action="store_true",
        help="Run Program AQ-FFT-7: residual capacity scaling gate for complex_pair_norm k-space transport.",
    )
    parser.add_argument(
        "--aq-yinyang-0",
        dest="aq_yinyang_0",
        action="store_true",
        help="Run Program AQ-YINYANG-0: two-frame tai-chi-tu visual transport and unfolding gate.",
    )
    parser.add_argument(
        "--aq-yinyang-1",
        dest="aq_yinyang_1",
        action="store_true",
        help="Run Program AQ-YINYANG-1: one-frame tai-chi-tu visual transport and codec ceiling gate.",
    )
    parser.add_argument(
        "--aq-yinyang-2",
        dest="aq_yinyang_2",
        action="store_true",
        help="Run Program AQ-YINYANG-2: one-frame tai-chi-tu visual transport with all 16 FFT coefficients per patch.",
    )
    parser.add_argument(
        "--aq-yinyang-3",
        dest="aq_yinyang_3",
        action="store_true",
        help="Run Program AQ-YINYANG-3: one-frame all-coefficient visual transport with direct pixel-space ridge decoding.",
    )
    parser.add_argument(
        "--aq-yinyang-4",
        dest="aq_yinyang_4",
        action="store_true",
        help="Run Program AQ-YINYANG-4: one-frame raw-pixel image-state transport with direct complex oscillator slots.",
    )
    parser.add_argument(
        "--yinyang-frame-start",
        type=int,
        default=1,
        help="One-based starting frame index for AQ-YINYANG-4 taichi reference PNG input selection.",
    )
    parser.add_argument(
        "--yinyang-frame-count",
        type=int,
        default=1,
        help="Number of consecutive 16x16 taichi reference PNG frames for AQ-YINYANG-4.",
    )
    parser.add_argument(
        "--yinyang-ordered-recovery",
        action="store_true",
        help="Recover AQ-YINYANG-4 patch states in raster/frame order instead of a randomized state permutation.",
    )
    parser.add_argument(
        "--write-yinyang-reference-pngs",
        action="store_true",
        help="Write local AQ-YINYANG-0 reference frames to --out-dir and exit.",
    )
    parser.add_argument(
        "--aq-dna-0",
        dest="aq_dna_0",
        action="store_true",
        help="Run Program AQ-DNA-0: true locality-preserving discrete token genome gate.",
    )
    parser.add_argument(
        "--aq-hybrid-0",
        dest="aq_hybrid_0",
        action="store_true",
        help="Run Program AQ-HYBRID-0: continuous geometry payload with token checksum diagnostics.",
    )
    parser.add_argument(
        "--as-0",
        dest="as_0",
        action="store_true",
        help="Run Program AS-0: TTN folded instruction genome ceiling against the flat AQ-FFT-1 baseline.",
    )
    parser.add_argument(
        "--as-0b",
        dest="as_0b",
        action="store_true",
        help="Run Program AS-0b: recover TTN folded state, then unfold it into the coefficient field.",
    )
    parser.add_argument(
        "--aq-fft-top-k",
        type=int,
        default=6,
        help="Number of fixed low-frequency 2D FFT basis slots retained per patch for AQ-FFT.",
    )
    parser.add_argument(
        "--aq-fft-residual-k",
        type=int,
        default=2,
        help="Number of adaptive residual k-space instruction slots used by AQ-FFT-1/AQ-FFT-2.",
    )
    parser.add_argument(
        "--aq0-include-adaptive-v2",
        action="store_true",
        help="Optional AQ-0 comparison arm: include tunnel_eigen_residual_adaptive_v2_shifted.",
    )
    args = parser.parse_args()

    if args.write_yinyang_reference_pngs:
        written = write_yinyang_reference_pngs(args.out_dir)
        for path in written:
            print(f"[YINYANG REF] wrote {path}")
        if (
            not args.aq_yinyang_0
            and not args.aq_yinyang_1
            and not args.aq_yinyang_2
            and not args.aq_yinyang_3
            and not args.aq_yinyang_4
            and not args.require_tpu
        ):
            raise SystemExit(0)

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

    if args.aq_dna_0:
        args.sequence_length = 16
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
        print(
            f"[AQ-DNA-0] TRUE token-genome gate with core-K={args.aq_fft_top_k} "
            f"residual-K={args.aq_fft_residual_k}"
        )

    if args.as_0b:
        args.sequence_length = 16
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
        print(
            f"[AS-0b] TTN unfold-decoder ceiling with core-K={args.aq_fft_top_k} "
            f"residual-K={args.aq_fft_residual_k}"
        )

    if args.as_0:
        args.sequence_length = 16
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
        print(
            f"[AS-0] TTN folded instruction genome ceiling with core-K={args.aq_fft_top_k} "
            f"residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_hybrid_0:
        args.sequence_length = 16
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
        print(
            f"[AQ-HYBRID-0] continuous-geometry + token-checksum gate with core-K={args.aq_fft_top_k} "
            f"residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_yinyang_0:
        args.sequence_length = 16
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
        args.aq_fft_residual_k = 4
        args.save_preview_png = True
        print(
            f"[AQ-YINYANG-0] two-frame visual transport gate "
            f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_yinyang_1:
        args.sequence_length = 16
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
        args.aq_fft_residual_k = 4
        args.save_preview_png = True
        print(
            f"[AQ-YINYANG-1] one-frame visual transport codec-ceiling gate "
            f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_yinyang_2:
        args.sequence_length = 16
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
        args.aq_fft_top_k = 16
        args.aq_fft_residual_k = 0
        args.save_preview_png = True
        print(
            f"[AQ-YINYANG-2] one-frame all-coefficient visual transport gate "
            f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_yinyang_3:
        args.sequence_length = 16
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
        args.aq_fft_top_k = 16
        args.aq_fft_residual_k = 0
        args.save_preview_png = True
        print(
            f"[AQ-YINYANG-3] one-frame all-coefficient pixel decoder gate "
            f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
        )

    if args.aq_yinyang_4:
        yinyang_state_count = max(1, int(args.yinyang_frame_count)) * 16
        args.sequence_length = 16
        args.depth_sweep = [3]
        args.seeds = []
        args.dlinoss_steps = min(args.dlinoss_steps, 16)
        args.include_controls = False
        args.run_heldout_transport = False
        args.heldout_include_controls = False
        args.run_semantic_recovery = True
        args.recovery_train_states = yinyang_state_count
        args.recovery_test_states = yinyang_state_count
        args.recovery_noise_sweep = [0.00]
        args.recovery_seeds = [11]
        args.control_block_size = 2
        args.save_preview_png = True
        print(
            f"[AQ-YINYANG-4] {args.yinyang_frame_count}-frame 16x16 reference PNG raw-pixel "
            f"direct oscillator transport gate"
        )

    if args.aq_fft_7 or args.aq_fft_6 or args.aq_fft_5 or args.aq_fft_4 or args.aq_fft_3 or args.aq_fft_2 or args.aq_fft_1 or args.aq_fft_0:
        args.sequence_length = 16
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
        if args.aq_fft_7:
            args.aq_fft_residual_k = 4
            print(
                f"[AQ-FFT-7] noise-free residual capacity scaling gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_6:
            print(
                f"[AQ-FFT-6] noise-free Hermitian-constrained complex decoder gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_5:
            print(
                f"[AQ-FFT-5] noise-free gauge-aligned complex decoding gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_4:
            print(
                f"[AQ-FFT-4] noise-free complex-orientation preservation k-space gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_3:
            print(
                f"[AQ-FFT-3] noise-free direct-complex oscillator k-space gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_2:
            print(
                f"[AQ-FFT-2] noise-free shape-charge controlled k-space gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        elif args.aq_fft_1:
            print(
                f"[AQ-FFT-1] noise-free core+residual k-space genome gate "
                f"with core-K={args.aq_fft_top_k} residual-K={args.aq_fft_residual_k}"
            )
        else:
            print(f"[AQ-FFT-0] noise-free 4x4 k-space patch codec gate with top-K={args.aq_fft_top_k}")

    if args.smoke_test:
        aq_tf_mode = args.aq_1a_tf or args.aq_1a_tf_s29
        aq_l5_mode = (
            aq_tf_mode
            or args.aq_1b
            or args.aq_1c
            or args.aq_seq_0
            or args.as_0
            or args.aq_hybrid_0
            or args.aq_dna_0
            or args.aq_fft_7
            or args.aq_fft_6
            or args.aq_fft_5
            or args.aq_fft_4
            or args.aq_fft_3
            or args.aq_fft_2
            or args.aq_fft_1
            or args.aq_fft_0
            or args.aq_yinyang_4
            or args.aq_img_0
            or args.aq_img_1
            or args.aq_img_1_lite
        )
        args.sequence_length = 16 if (args.as_0b or args.as_0 or args.aq_hybrid_0 or args.aq_dna_0 or args.aq_yinyang_4 or args.aq_fft_7 or args.aq_fft_6 or args.aq_fft_5 or args.aq_fft_4 or args.aq_fft_3 or args.aq_fft_2 or args.aq_fft_1 or args.aq_fft_0) else (8 if (args.aq_0b or aq_l5_mode) else 12)
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
        args.recovery_train_states = 16 if (args.aq_1c or args.as_0b or args.as_0 or args.aq_hybrid_0 or args.aq_dna_0 or args.aq_yinyang_4 or args.aq_fft_7 or args.aq_fft_6 or args.aq_fft_5 or args.aq_fft_4 or args.aq_fft_3 or args.aq_fft_2 or args.aq_fft_1 or args.aq_fft_0 or args.aq_img_0 or args.aq_img_1) else (8 if args.aq_img_1_lite else (4 if aq_l5_mode else (2 if args.aq_0b else 3)))
        args.recovery_test_states = 16 if (args.aq_1c or args.as_0b or args.as_0 or args.aq_hybrid_0 or args.aq_dna_0 or args.aq_yinyang_4 or args.aq_fft_7 or args.aq_fft_6 or args.aq_fft_5 or args.aq_fft_4 or args.aq_fft_3 or args.aq_fft_2 or args.aq_fft_1 or args.aq_fft_0 or args.aq_img_0 or args.aq_img_1) else (8 if args.aq_img_1_lite else (4 if aq_l5_mode else (1 if args.aq_0b else 2)))
        args.recovery_noise_sweep = [0.30] if aq_l5_mode else ([0.00] if args.aq_0b else [0.18])
        args.recovery_noise_sweep = [0.00] if (args.aq_1c or args.aq_seq_0 or args.as_0b or args.as_0 or args.aq_hybrid_0 or args.aq_dna_0 or args.aq_yinyang_4 or args.aq_fft_7 or args.aq_fft_6 or args.aq_fft_5 or args.aq_fft_4 or args.aq_fft_3 or args.aq_fft_2 or args.aq_fft_1 or args.aq_fft_0 or args.aq_img_0 or args.aq_img_1 or args.aq_img_1_lite) else args.recovery_noise_sweep
        args.recovery_seeds = [11] if (args.aq_0b or aq_l5_mode) else args.recovery_seeds
        args.control_block_size = 2 if (args.aq_0b or aq_l5_mode) else 3
        print("[SMOKE TEST]")

    run_program_ap(args)
