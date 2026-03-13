from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from time_dilation_stream_dataset import (
    is_time_stream_dataset_path,
    load_time_stream_dataset_manifest,
    load_time_stream_dataset_records,
    time_stream_dataset_manifest_path,
)


StreamKey = tuple[tuple[str, Any], ...]
TIME_STREAM_CACHE_VERSION = 1
STREAM_GRANULARITY_CHOICES = ("gpu", "gpu_pid", "pid", "stream_id")


def ensure_torch_runtime_env() -> None:
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def load_torch_components():
    ensure_torch_runtime_env()
    import torch
    import torch.nn.functional as F
    from dlinoss.dlinoss_torch import DLinOSSModel
    return torch, F, DLinOSSModel


def load_entangled_memory_components():
    ensure_torch_runtime_env()
    import torch
    from entangled_camkii_lattice import EntangledCaMKIILattice
    return torch, EntangledCaMKIILattice


def sanitize_label_token(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text) or "unknown"


def coerce_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def infer_stream_id(record: dict[str, Any]) -> str:
    raw_stream_id = record.get("stream_id")
    if raw_stream_id is not None and str(raw_stream_id).strip():
        return str(raw_stream_id)
    gpu_id = coerce_int(record.get("gpu_id", -1), default=-1)
    device_worker_index = record.get("device_worker_index")
    if device_worker_index is not None:
        worker_index = coerce_int(device_worker_index, default=-1)
        if worker_index >= 0:
            return f"gpu{gpu_id}_worker{worker_index}"
    pid = coerce_int(record.get("pid", -1), default=-1)
    if pid >= 0:
        return f"gpu{gpu_id}_pid{pid}"
    return f"gpu{gpu_id}"


def record_stream_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "stream_id": infer_stream_id(record),
        "stream_global_index": coerce_int(record.get("stream_global_index", -1), default=-1),
        "device_worker_index": coerce_int(record.get("device_worker_index", -1), default=-1),
        "workers_per_gpu": max(coerce_int(record.get("workers_per_gpu", 1), default=1), 1),
        "gpu_id": coerce_int(record.get("gpu_id", -1), default=-1),
        "gpu_name": str(record.get("gpu_name", "unknown")),
        "pid": coerce_int(record.get("pid", -1), default=-1),
    }


def make_stream_key(record: dict[str, Any], granularity: str) -> StreamKey:
    identity = record_stream_identity(record)
    gpu_id = int(identity["gpu_id"])
    gpu_name = str(identity["gpu_name"])
    pid = int(identity["pid"])
    if granularity == "gpu":
        fields = (("gpu_id", gpu_id), ("gpu_name", gpu_name))
    elif granularity == "pid":
        fields = (("pid", pid),)
    elif granularity == "stream_id":
        fields = (("stream_id", str(identity["stream_id"])),)
    else:
        fields = (("gpu_id", gpu_id), ("gpu_name", gpu_name), ("pid", pid))
    return tuple(fields)


def stream_key_dict(key: StreamKey) -> dict[str, Any]:
    return {name: value for name, value in key}


def stream_label(key: StreamKey) -> str:
    parts = []
    for name, value in key:
        parts.extend([sanitize_label_token(name), sanitize_label_token(value)])
    return "_".join(parts)


def summarize_clock_array(clock_data: Any) -> dict[str, float]:
    arr = np.asarray(clock_data, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "range", "slope", "diff_std", "diff_mean_abs")}
    diffs = np.diff(arr) if arr.size > 1 else np.zeros(1, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "slope": float((arr[-1] - arr[0]) / max(arr.size - 1, 1)),
        "diff_std": float(diffs.std()),
        "diff_mean_abs": float(np.abs(diffs).mean()),
    }


def packet_record_sort_key(record: dict[str, Any]) -> tuple[float, int, int]:
    identity = record_stream_identity(record)
    return (
        float(record.get("host_time_mid", 0.0)),
        int(identity["gpu_id"]),
        int(identity["stream_global_index"]),
        int(identity["device_worker_index"]),
        int(record.get("packet_index", 0)),
    )


def filter_gpu_clock_packet_records(items: list[Any], source_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("record_type") == "gpu_clock_packet":
            records.append(item)
    if not records:
        raise ValueError(f"No gpu_clock_packet records found in {source_label}")
    records.sort(key=packet_record_sort_key)
    return records


def load_packet_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if is_time_stream_dataset_path(path):
        records, _ = load_time_stream_dataset_records(path)
        return records
    raw = np.load(path, allow_pickle=True)
    return filter_gpu_clock_packet_records(raw.tolist(), str(path))


def serialize_packet_record(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in record.items():
        if key == "clock_data":
            payload[key] = np.asarray(value, dtype=np.uint64).reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, np.generic):
            payload[key] = value.item()
        else:
            payload[key] = value
    return payload


def deserialize_packet_record(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["clock_data"] = np.asarray(payload.get("clock_data", []), dtype=np.uint64)
    return record


def time_stream_cache_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def source_file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stats = resolved.stat()
    return {
        "source_path": str(resolved),
        "source_size_bytes": int(stats.st_size),
        "source_mtime_ns": int(stats.st_mtime_ns),
    }


def load_time_stream_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    return json.loads(time_stream_cache_manifest_path(cache_dir).read_text(encoding="utf-8"))


def validate_time_stream_cache_manifest(manifest: dict[str, Any], input_path: Path, cache_dir: Path) -> None:
    signature = source_file_signature(input_path)
    if int(manifest.get("cache_version", -1)) != TIME_STREAM_CACHE_VERSION:
        raise ValueError("Time-stream cache version mismatch.")
    for field in ("source_path", "source_size_bytes", "source_mtime_ns"):
        if manifest.get(field) != signature[field]:
            raise ValueError(f"Time-stream cache field mismatch: {field}")
    missing_chunks = [name for name in manifest.get("chunk_files", []) if not (cache_dir / str(name)).exists()]
    if missing_chunks:
        raise ValueError(f"Time-stream cache is missing chunk files: {missing_chunks[:3]}")


def prepare_time_stream_cache(
    input_path: Path,
    cache_dir: Path,
    chunk_size: int = 2048,
    force: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    chunk_size = max(int(chunk_size), 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = time_stream_cache_manifest_path(cache_dir)
    existing_manifest = None
    if manifest_path.exists() and not force:
        try:
            existing_manifest = load_time_stream_cache_manifest(cache_dir)
            validate_time_stream_cache_manifest(existing_manifest, input_path, cache_dir)
            return {**existing_manifest, "cache_status": "reused", "cache_dir": str(cache_dir)}
        except Exception:
            existing_manifest = None
    had_manifest = manifest_path.exists()
    records = load_packet_records(input_path)
    for stale_chunk in cache_dir.glob("records_*.jsonl"):
        stale_chunk.unlink()
    chunk_files: list[str] = []
    for offset in range(0, len(records), chunk_size):
        chunk_name = f"records_{offset // chunk_size:05d}.jsonl"
        chunk_path = cache_dir / chunk_name
        chunk_lines = [json.dumps(serialize_packet_record(record), sort_keys=True) for record in records[offset: offset + chunk_size]]
        chunk_path.write_text("\n".join(chunk_lines) + ("\n" if chunk_lines else ""), encoding="utf-8")
        chunk_files.append(chunk_name)
    manifest = {
        "cache_version": TIME_STREAM_CACHE_VERSION,
        **source_file_signature(input_path),
        "record_count": int(len(records)),
        "record_type": "gpu_clock_packet",
        "sort_key": ["host_time_mid", "gpu_id", "packet_index"],
        "chunk_size": int(chunk_size),
        "chunk_files": chunk_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "cache_status": "rebuilt" if force or had_manifest or existing_manifest is not None else "created", "cache_dir": str(cache_dir)}


def load_cached_packet_records(
    input_path: Path,
    cache_dir: Path,
    max_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir = Path(cache_dir)
    manifest = load_time_stream_cache_manifest(cache_dir)
    validate_time_stream_cache_manifest(manifest, input_path, cache_dir)
    return list(iter_cached_packet_records(input_path, cache_dir, max_records=max_records)), manifest


def iter_cached_packet_records(
    input_path: Path,
    cache_dir: Path,
    max_records: int = 0,
):
    cache_dir = Path(cache_dir)
    manifest = load_time_stream_cache_manifest(cache_dir)
    validate_time_stream_cache_manifest(manifest, input_path, cache_dir)
    target_count = max(int(max_records), 0)
    emitted = 0
    for chunk_name in manifest.get("chunk_files", []):
        lines = (cache_dir / str(chunk_name)).read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            yield deserialize_packet_record(json.loads(line))
            emitted += 1
            if target_count > 0 and emitted >= target_count:
                return


def iter_time_moments(
    ordered_records,
    bin_width: float,
    global_start: float | None = None,
):
    iterator = iter(ordered_records)
    first_record = next(iterator, None)
    if first_record is None:
        return
    bin_width = max(float(bin_width), 1e-6)
    start = float(first_record.get("host_time_mid", 0.0)) if global_start is None else float(global_start)
    current_records = [first_record]
    current_bin = int(round((float(first_record.get("host_time_mid", 0.0)) - start) / bin_width))
    for record in iterator:
        mid = float(record.get("host_time_mid", 0.0))
        time_bin = int(round((mid - start) / bin_width))
        if time_bin != current_bin:
            yield build_time_moment(current_bin, current_records)
            current_bin = time_bin
            current_records = []
        current_records.append(record)
    if current_records:
        yield build_time_moment(current_bin, current_records)


def build_time_moment(time_bin: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_records = sorted(records, key=packet_record_sort_key)
    return {
        "time_bin": int(time_bin),
        "host_time_mid": float(np.mean([float(record.get("host_time_mid", 0.0)) for record in ordered_records])),
        "records": ordered_records,
    }


class TimeMomentStreamingService:
    def __init__(self, records, bin_width: float, global_start: float | None = None, source_mode: str = "replay"):
        self.bin_width = max(float(bin_width), 1e-6)
        if global_start is not None:
            self.records = records
            self.global_start = float(global_start)
        elif isinstance(records, (list, tuple)):
            self.records = records
            self.global_start = float(records[0].get("host_time_mid", 0.0)) if records else 0.0
        else:
            iterator = iter(records)
            first_record = next(iterator, None)
            if first_record is None:
                self.records = ()
                self.global_start = 0.0
            else:
                self.records = itertools.chain([first_record], iterator)
                self.global_start = float(first_record.get("host_time_mid", 0.0))
        self.source_mode = str(source_mode)

    @classmethod
    def from_cached_replay(
        cls,
        input_path: Path,
        cache_dir: Path,
        bin_width: float,
        max_records: int = 0,
    ):
        return cls(
            iter_cached_packet_records(input_path, cache_dir, max_records=max_records),
            bin_width=bin_width,
            global_start=None,
            source_mode="cached_replay",
        )

    def iter_moments(self):
        yield from iter_time_moments(self.records, self.bin_width, global_start=self.global_start)


def build_feature_sequences(records: list[dict[str, Any]], bin_width: float, stream_granularity: str = "stream_id") -> dict[str, Any]:
    if not records:
        raise ValueError("No packet records were provided for feature construction.")
    feature_names = [
        "rel_time_sec", "delta_host_sec", "elapsed_host_sec", "packet_gap",
        "clock_mean", "clock_std", "clock_range", "clock_slope",
        "clock_diff_std", "clock_diff_mean_abs", "log_sample_count",
        "log_total_threads", "log_buffer_size", "stream_index", "peer_count",
        "clock_mean_vs_peers", "time_mid_vs_peers",
    ]
    stream_keys = sorted({make_stream_key(r, stream_granularity) for r in records}, key=stream_label)
    stream_to_index = {key: idx for idx, key in enumerate(stream_keys)}
    gpu_keys = sorted({(int(r.get("gpu_id", -1)), str(r.get("gpu_name", "unknown"))) for r in records})
    service = TimeMomentStreamingService(records, bin_width)
    global_start = service.global_start
    sequences: dict[StreamKey, list[np.ndarray]] = defaultdict(list)
    metadata: dict[StreamKey, list[dict[str, float]]] = defaultdict(list)
    stream_info: dict[StreamKey, dict[str, Any]] = {}
    prev_by_stream: dict[StreamKey, dict[str, float]] = {}
    for moment in service.iter_moments() or []:
        entries = []
        for record in moment["records"]:
            mid = float(record.get("host_time_mid", 0.0))
            key = make_stream_key(record, stream_granularity)
            stream_info.setdefault(key, record_stream_identity(record))
            entries.append({
                "record": record,
                "gpu_key": key,
                "mid": mid,
                "clock": summarize_clock_array(record.get("clock_data", [])),
                "bin": int(moment["time_bin"]),
            })
        for entry in entries:
            record = entry["record"]
            key = entry["gpu_key"]
            peers = [peer for peer in entries if peer["gpu_key"] != key]
            prev = prev_by_stream.get(key)
            mid = entry["mid"]
            clock = entry["clock"]
            if peers:
                peer_clock_mean = float(np.mean([peer["clock"]["mean"] for peer in peers]))
                peer_time_mean = float(np.mean([peer["mid"] for peer in peers]))
            else:
                peer_clock_mean = clock["mean"]
                peer_time_mean = mid
            row = np.array([
                mid - global_start,
                0.0 if prev is None else mid - prev["mid"],
                float(record.get("elapsed_host_seconds", 0.0)),
                0.0 if prev is None else float(record.get("packet_index", 0)) - prev["packet_index"],
                clock["mean"], clock["std"], clock["range"], clock["slope"],
                clock["diff_std"], clock["diff_mean_abs"],
                float(np.log1p(max(float(record.get("sample_count", 0.0)), 0.0))),
                float(np.log1p(max(float(record.get("total_threads", 0.0)), 0.0))),
                float(np.log1p(max(float(record.get("buffer_size", 0.0)), 0.0))),
                float(stream_to_index[key]), float(len(peers)),
                clock["mean"] - peer_clock_mean,
                mid - peer_time_mean,
            ], dtype=np.float32)
            sequences[key].append(row)
            metadata[key].append({
                "packet_index": int(record.get("packet_index", 0)),
                "host_time_mid": mid,
                "elapsed_host_seconds": float(record.get("elapsed_host_seconds", 0.0)),
                "time_bin": int(entry["bin"]),
            })
            prev_by_stream[key] = {"mid": mid, "packet_index": float(record.get("packet_index", 0))}
    return {
        "feature_names": feature_names,
        "sequences": {key: np.stack(rows).astype(np.float32) for key, rows in sequences.items()},
        "metadata": metadata,
        "stream_info": stream_info,
        "multi_processor_observed": len(stream_keys) > 1,
        "multi_gpu_observed": len(gpu_keys) > 1,
        "stream_granularity": stream_granularity,
        "stream_labels": {stream_label(key): stream_info.get(key, stream_key_dict(key)) for key in stream_keys},
        "bin_width": bin_width,
        "global_start": global_start,
    }


def normalize_sequences(sequences: dict[StreamKey, np.ndarray]) -> tuple[dict[StreamKey, np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.vstack(list(sequences.values())).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = {key: ((seq - mean) / std).astype(np.float32) for key, seq in sequences.items()}
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def parse_int_list(text: str) -> list[int]:
    values = sorted({max(int(token.strip()), 0) for token in str(text).split(",") if token.strip()})
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def centered_rolling_mean_matrix(values: np.ndarray, radius: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if radius <= 0 or len(matrix) == 0:
        return matrix.copy()
    n = len(matrix)
    cumsum = np.cumsum(np.pad(matrix, ((1, 0), (0, 0)), mode='constant'), axis=0)
    out = np.zeros_like(matrix)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        out[i] = (cumsum[end] - cumsum[start]) / (end - start)
    return out


def centered_rolling_mean_vector(values: np.ndarray, radius: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if radius <= 0 or len(vector) == 0:
        return vector.copy()
    n = len(vector)
    cumsum = np.cumsum(np.pad(vector, (1, 0), mode='constant'))
    out = np.zeros_like(vector)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        out[i] = (cumsum[end] - cumsum[start]) / (end - start)
    return out


def make_now_stream_agent_key(base_key: StreamKey, radius: int) -> StreamKey:
    return tuple([*base_key, ("now_agent_kind", "temporal_scale"), ("now_scale_radius", int(radius))])


def build_now_stream_agents(
    raw_sequences: dict[StreamKey, np.ndarray],
    normalized_sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    temporal_radii: list[int],
) -> dict[str, Any]:
    ordered_base_keys = sorted(raw_sequences.keys(), key=stream_label)
    agent_raw_sequences: dict[StreamKey, np.ndarray] = {}
    agent_normalized_sequences: dict[StreamKey, np.ndarray] = {}
    agent_metadata: dict[StreamKey, list[dict[str, float]]] = {}
    manifest = []
    for base_key in ordered_base_keys:
        base_label = stream_label(base_key)
        raw_sequence = raw_sequences[base_key]
        normalized_sequence = normalized_sequences[base_key]
        host_times = np.array([float(item.get("host_time_mid", 0.0)) for item in metadata[base_key]], dtype=np.float64)
        for radius in temporal_radii:
            agent_key = make_now_stream_agent_key(base_key, radius)
            agent_raw_sequences[agent_key] = centered_rolling_mean_matrix(raw_sequence, radius)
            agent_normalized_sequences[agent_key] = centered_rolling_mean_matrix(normalized_sequence, radius)
            believed_now = centered_rolling_mean_vector(host_times, radius)
            agent_rows = []
            for meta, believed_now_sec in zip(metadata[base_key], believed_now):
                row = dict(meta)
                row["agent_believed_now_host_time_mid"] = float(believed_now_sec)
                row["agent_now_offset_seconds"] = float(believed_now_sec - float(meta.get("host_time_mid", 0.0)))
                row["source_stream_label"] = base_label
                row["now_scale_radius"] = int(radius)
                agent_rows.append(row)
            agent_metadata[agent_key] = agent_rows
            manifest.append({
                "agent_label": stream_label(agent_key),
                "source_stream_label": base_label,
                "now_agent_kind": "temporal_scale",
                "now_scale_radius": int(radius),
                "packet_count": int(len(normalized_sequence)),
            })
    return {
        "raw_sequences": agent_raw_sequences,
        "normalized_sequences": agent_normalized_sequences,
        "metadata": agent_metadata,
        "manifest": manifest,
        "temporal_radii": [int(v) for v in temporal_radii],
        "agent_count": int(len(agent_raw_sequences)),
        "base_stream_count": int(len(ordered_base_keys)),
    }


def build_aligned_stream_bins(
    raw_sequences: dict[StreamKey, np.ndarray],
    normalized_sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    extra_meta_fields: tuple[str, ...] = (),
) -> tuple[list[StreamKey], dict[int, dict[StreamKey, dict[str, Any]]], int]:
    if not raw_sequences:
        return [], {}, 0
    ordered_keys = sorted(raw_sequences.keys(), key=stream_label)
    feature_dim = int(next(iter(raw_sequences.values())).shape[1])
    bins: dict[int, dict[StreamKey, dict[str, Any]]] = defaultdict(dict)
    for key in ordered_keys:
        grouped: dict[int, dict[str, list[Any]]] = defaultdict(
            lambda: {"raw": [], "normalized": [], "host_time_mid": [], **{field: [] for field in extra_meta_fields}}
        )
        for raw_row, norm_row, meta in zip(raw_sequences[key], normalized_sequences[key], metadata[key]):
            grouped[int(meta["time_bin"])]["raw"].append(raw_row)
            grouped[int(meta["time_bin"])]["normalized"].append(norm_row)
            grouped[int(meta["time_bin"])]["host_time_mid"].append(float(meta["host_time_mid"]))
            for field in extra_meta_fields:
                if field in meta and meta[field] is not None:
                    grouped[int(meta["time_bin"])][field].append(float(meta[field]))
        for time_bin, payload in grouped.items():
            entry = {
                "raw": np.mean(np.stack(payload["raw"]).astype(np.float32), axis=0).astype(np.float32),
                "normalized": np.mean(np.stack(payload["normalized"]).astype(np.float32), axis=0).astype(np.float32),
                "host_time_mid": float(np.mean(payload["host_time_mid"])),
            }
            for field in extra_meta_fields:
                values = payload[field]
                entry[field] = float(np.mean(values)) if values else float("nan")
            bins[time_bin][key] = entry
    return ordered_keys, bins, feature_dim


def make_windows(sequences: dict[StreamKey, np.ndarray], window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for sequence in sequences.values():
        if len(sequence) < window_size:
            continue
        for start in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[start:start + window_size]
            inputs.append(window[:-1])
            targets.append(window[1:])
    if not inputs:
        raise ValueError("No windows were created. Try a smaller --window-size or provide more records.")
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_present_vectors(present_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    consensus = present_vectors.mean(axis=0).astype(np.float32)
    deviations = np.linalg.norm(present_vectors - consensus, axis=1).astype(np.float32)
    if len(present_vectors) > 1:
        diffs = present_vectors[:, None, :] - present_vectors[None, :, :]
        pairwise = np.linalg.norm(diffs, axis=2)
        upper = pairwise[np.triu_indices(len(present_vectors), k=1)]
        mean_pairwise = float(upper.mean()) if upper.size else 0.0
        max_pairwise = float(upper.max()) if upper.size else 0.0
    else:
        mean_pairwise = 0.0
        max_pairwise = 0.0
    return consensus, deviations, mean_pairwise, max_pairwise


def extract_padded_window(sequence: np.ndarray, center: int, radius: int) -> np.ndarray:
    width = 2 * radius + 1
    out = np.zeros((width, sequence.shape[1]), dtype=np.float32)
    start = center - radius
    end = center + radius + 1
    src_start = max(0, start)
    src_end = min(sequence.shape[0], end)
    dst_start = src_start - start
    out[dst_start:dst_start + (src_end - src_start)] = sequence[src_start:src_end]
    return out


def find_peak_indices(scores: np.ndarray, z_threshold: float, min_separation: int, max_events: int, fallback_top_k: int) -> tuple[list[int], np.ndarray]:
    if scores.size == 0:
        return [], np.zeros(0, dtype=np.float32)
    z_scores = ((scores - scores.mean()) / max(float(scores.std()), 1e-6)).astype(np.float32)
    local_maxima = []
    for idx, score in enumerate(scores):
        left = scores[idx - 1] if idx > 0 else -np.inf
        right = scores[idx + 1] if idx + 1 < len(scores) else -np.inf
        if score >= left and score >= right:
            local_maxima.append(idx)
    ranked = [idx for idx in local_maxima if z_scores[idx] >= z_threshold]
    if not ranked:
        ranked = sorted(local_maxima, key=lambda idx: float(scores[idx]), reverse=True)[:fallback_top_k]
    else:
        ranked = sorted(ranked, key=lambda idx: float(scores[idx]), reverse=True)
    selected: list[int] = []
    for idx in ranked:
        if all(abs(idx - existing) >= min_separation for existing in selected):
            selected.append(idx)
        if len(selected) >= max_events:
            break
    selected.sort()
    return selected, z_scores


def cluster_phase_shift_events(events: list[dict[str, Any]], similarity_threshold: float) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (-float(item["distortion_score_z"]), -float(item["distortion_score"]))):
        signature = np.asarray(event["signature"], dtype=np.float32)
        best_cluster = None
        best_similarity = -1.0
        for cluster in clusters:
            similarity = cosine_similarity(signature, cluster["prototype"])
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_cluster = cluster
                best_similarity = similarity
        if best_cluster is None:
            cluster_id = len(clusters)
            event["class_id"] = cluster_id
            event["class_similarity"] = 1.0
            clusters.append({
                "class_id": cluster_id,
                "prototype": signature.copy(),
                "signature_sum": signature.astype(np.float64).copy(),
                "members": [event],
                "similarities": [1.0],
            })
        else:
            event["class_id"] = int(best_cluster["class_id"])
            event["class_similarity"] = float(best_similarity)
            best_cluster["members"].append(event)
            best_cluster["similarities"].append(float(best_similarity))
            best_cluster["signature_sum"] += signature
            best_cluster["prototype"] = (best_cluster["signature_sum"] / len(best_cluster["members"])).astype(np.float32)
    class_summaries = []
    repeated_event_count = 0
    for cluster in clusters:
        members = sorted(cluster["members"], key=lambda item: (str(item["processor_label"]), float(item["host_time_mid"])))
        if len(members) > 1:
            repeated_event_count += len(members)
        times = [float(member["host_time_mid"]) for member in members]
        time_gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        class_summaries.append({
            "class_id": int(cluster["class_id"]),
            "member_count": int(len(members)),
            "repeats": bool(len(members) > 1),
            "processor_labels": sorted({str(member["processor_label"]) for member in members}),
            "mean_similarity": float(np.mean(cluster["similarities"])),
            "mean_distortion_score": float(np.mean([member["distortion_score"] for member in members])),
            "mean_distortion_score_z": float(np.mean([member["distortion_score_z"] for member in members])),
            "mean_recurrence_seconds": float(np.mean(time_gaps)) if time_gaps else None,
            "members": [
                {
                    "processor_label": str(member["processor_label"]),
                    "transition_index": int(member["transition_index"]),
                    "packet_index": int(member["packet_index"]),
                    "host_time_mid": float(member["host_time_mid"]),
                    "distortion_score": float(member["distortion_score"]),
                    "distortion_score_z": float(member["distortion_score_z"]),
                    "class_similarity": float(member["class_similarity"]),
                }
                for member in members
            ],
        })
    class_summaries.sort(key=lambda item: (-int(item["member_count"]), -float(item["mean_distortion_score"])))
    serializable_events = [
        {
            "processor_label": str(event["processor_label"]),
            "transition_index": int(event["transition_index"]),
            "packet_index": int(event["packet_index"]),
            "host_time_mid": float(event["host_time_mid"]),
            "distortion_score": float(event["distortion_score"]),
            "distortion_score_z": float(event["distortion_score_z"]),
            "class_id": int(event["class_id"]),
            "class_similarity": float(event["class_similarity"]),
        }
        for event in sorted(events, key=lambda item: (str(item["processor_label"]), float(item["host_time_mid"])))
    ]
    return {
        "events": serializable_events,
        "classes": class_summaries,
        "class_count": int(len(class_summaries)),
        "repeated_class_count": int(sum(1 for item in class_summaries if item["repeats"])),
        "one_off_class_count": int(sum(1 for item in class_summaries if not item["repeats"])),
        "repeated_event_count": int(repeated_event_count),
    }


def analyze_phase_shift_classes(processor_details: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    processor_event_counts: dict[str, int] = {}
    for label, detail in processor_details.items():
        peak_indices, z_scores = find_peak_indices(
            detail["rmse"],
            z_threshold=args.phase_z_threshold,
            min_separation=args.phase_min_separation,
            max_events=args.phase_max_events_per_processor,
            fallback_top_k=args.phase_fallback_top_k,
        )
        processor_event_counts[label] = int(len(peak_indices))
        observed = detail["observed"]
        residual = detail["residual"]
        for peak_idx in peak_indices:
            observed_window = extract_padded_window(observed, peak_idx, args.phase_context_radius)
            residual_window = extract_padded_window(residual, peak_idx, args.phase_context_radius)
            signature = np.concatenate([
                observed_window.reshape(-1),
                residual_window.reshape(-1),
                np.array([detail["rmse"][peak_idx], z_scores[peak_idx]], dtype=np.float32),
            ]).astype(np.float32)
            meta = detail["metadata"][peak_idx]
            events.append({
                "processor_label": label,
                "transition_index": int(peak_idx),
                "packet_index": int(meta["packet_index"]),
                "host_time_mid": float(meta["host_time_mid"]),
                "distortion_score": float(detail["rmse"][peak_idx]),
                "distortion_score_z": float(z_scores[peak_idx]),
                "signature": signature,
            })
    clustered = cluster_phase_shift_events(events, similarity_threshold=args.phase_similarity_threshold)
    clustered["processor_event_counts"] = processor_event_counts
    clustered["parameters"] = {
        "z_threshold": float(args.phase_z_threshold),
        "min_separation": int(args.phase_min_separation),
        "context_radius": int(args.phase_context_radius),
        "similarity_threshold": float(args.phase_similarity_threshold),
        "max_events_per_processor": int(args.phase_max_events_per_processor),
        "fallback_top_k": int(args.phase_fallback_top_k),
    }
    clustered["repeated_classes"] = [item for item in clustered["classes"] if item["repeats"]]
    return clustered


def fit_linear_operator(previous_states: np.ndarray, next_states: np.ndarray, ridge: float) -> np.ndarray:
    prev64 = previous_states.astype(np.float64)
    next64 = next_states.astype(np.float64)
    gram = prev64.T @ prev64
    cross = prev64.T @ next64
    operator = np.linalg.solve(gram + ridge * np.eye(gram.shape[0], dtype=np.float64), cross)
    return operator.astype(np.float32)


def nearest_orthogonal_operator(operator: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(operator.astype(np.float64), full_matrices=False)
    return (u @ vh).astype(np.float32)


def split_canonical_state_matrix(
    state_matrix: np.ndarray,
    pairing: str = "half_split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_count = int(state_matrix.shape[1] // 2)
    if pair_count <= 0:
        empty = np.zeros((state_matrix.shape[0], 0), dtype=np.float32)
        return empty, empty, empty, state_matrix.astype(np.float32)
    if pairing == "interleaved":
        canonical_slice = state_matrix[:, :2 * pair_count]
        q = canonical_slice[:, 0::2]
        p = canonical_slice[:, 1::2]
        auxiliary = state_matrix[:, 2 * pair_count:]
    else:
        q = state_matrix[:, :pair_count]
        p = state_matrix[:, pair_count:2 * pair_count]
        auxiliary = state_matrix[:, 2 * pair_count:]
    canonical_state = np.concatenate([q, p], axis=1).astype(np.float32)
    return canonical_state, q.astype(np.float32), p.astype(np.float32), auxiliary.astype(np.float32)


def build_symplectic_form(dim: int) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError(f"Symplectic form requires even dimension, got {dim}")
    half = dim // 2
    form = np.zeros((dim, dim), dtype=np.float32)
    form[:half, half:] = np.eye(half, dtype=np.float32)
    form[half:, :half] = -np.eye(half, dtype=np.float32)
    return form


def measure_symplectic_defect(operator: np.ndarray, symplectic_form: np.ndarray) -> float:
    residual = operator.T.astype(np.float64) @ symplectic_form.astype(np.float64) @ operator.astype(np.float64) - symplectic_form.astype(np.float64)
    return float(np.linalg.norm(residual) / max(np.linalg.norm(symplectic_form.astype(np.float64)), 1e-6))


def project_generator_to_hamiltonian(generator: np.ndarray, symplectic_form: np.ndarray) -> np.ndarray:
    return (0.5 * (generator.astype(np.float64) + symplectic_form.astype(np.float64) @ generator.astype(np.float64).T @ symplectic_form.astype(np.float64))).astype(np.float32)


def cayley_transform_hamiltonian(generator: np.ndarray) -> np.ndarray:
    eye = np.eye(generator.shape[0], dtype=np.float64)
    left = eye - 0.5 * generator.astype(np.float64)
    right = eye + 0.5 * generator.astype(np.float64)
    try:
        return np.linalg.solve(left, right).astype(np.float32)
    except np.linalg.LinAlgError:
        return right.astype(np.float32)


def summarize_linear_operator_fit(
    state_matrix: np.ndarray,
    ridge: float,
    array_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(state_matrix) < 2:
        return {
            "status": "insufficient_states",
            "message": "Need at least two aligned latent states to estimate a transition operator.",
        }, {}
    previous_states = state_matrix[:-1].astype(np.float32)
    next_states = state_matrix[1:].astype(np.float32)
    operator = fit_linear_operator(previous_states, next_states, ridge=ridge)
    orthogonal = nearest_orthogonal_operator(operator)
    pred_empirical = previous_states @ operator
    pred_orthogonal = previous_states @ orthogonal
    empirical_rmse = np.sqrt(((pred_empirical - next_states) ** 2).mean(axis=1)).astype(np.float32)
    orthogonal_rmse = np.sqrt(((pred_orthogonal - next_states) ** 2).mean(axis=1)).astype(np.float32)
    singular_values = np.linalg.svd(operator.astype(np.float64), compute_uv=False).astype(np.float32)
    operator_gap = float(np.linalg.norm(operator - orthogonal) / max(np.linalg.norm(operator), 1e-6))
    return {
        "status": "ok",
        "joint_state_dim": int(state_matrix.shape[1]),
        "transition_count": int(previous_states.shape[0]),
        "ridge": float(ridge),
        "operator_distance_to_orthogonal": operator_gap,
        "empirical_transition_rmse_mean": float(empirical_rmse.mean()),
        "empirical_transition_rmse_p95": float(np.quantile(empirical_rmse, 0.95)),
        "orthogonal_transition_rmse_mean": float(orthogonal_rmse.mean()),
        "orthogonal_transition_rmse_p95": float(np.quantile(orthogonal_rmse, 0.95)),
        "singular_value_min": float(singular_values.min()) if singular_values.size else 0.0,
        "singular_value_max": float(singular_values.max()) if singular_values.size else 0.0,
        "singular_value_mean": float(singular_values.mean()) if singular_values.size else 0.0,
    }, {
        f"{array_prefix}empirical_transition_rmse": empirical_rmse,
        f"{array_prefix}orthogonal_transition_rmse": orthogonal_rmse,
        f"{array_prefix}empirical_operator": operator.astype(np.float32),
        f"{array_prefix}nearest_orthogonal_operator": orthogonal.astype(np.float32),
        f"{array_prefix}operator_singular_values": singular_values,
    }


def summarize_canonical_oscillator_fit(
    state_matrix: np.ndarray,
    ridge: float,
    pairing: str = "half_split",
    array_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(state_matrix) < 2:
        return {
            "status": "insufficient_states",
            "message": "Need at least two latent states to estimate canonical oscillator dynamics.",
        }, {}
    canonical_state, q_matrix, p_matrix, auxiliary = split_canonical_state_matrix(state_matrix.astype(np.float32), pairing=pairing)
    canonical_dim = int(canonical_state.shape[1])
    pair_count = canonical_dim // 2
    if pair_count <= 0:
        return {
            "status": "insufficient_canonical_pairs",
            "message": "Need at least two latent dimensions to construct a q/p canonical state.",
        }, {}
    previous_states = canonical_state[:-1].astype(np.float32)
    next_states = canonical_state[1:].astype(np.float32)
    empirical_operator = fit_linear_operator(previous_states, next_states, ridge=ridge)
    orthogonal_operator = nearest_orthogonal_operator(empirical_operator)
    empirical_generator = (empirical_operator - np.eye(canonical_dim, dtype=np.float32)).astype(np.float32)
    symplectic_form = build_symplectic_form(canonical_dim)
    hamiltonian_generator = project_generator_to_hamiltonian(empirical_generator, symplectic_form)
    symplectic_operator = cayley_transform_hamiltonian(hamiltonian_generator)
    pred_empirical = previous_states @ empirical_operator
    pred_orthogonal = previous_states @ orthogonal_operator
    pred_symplectic = previous_states @ symplectic_operator
    empirical_rmse = np.sqrt(((pred_empirical - next_states) ** 2).mean(axis=1)).astype(np.float32)
    orthogonal_rmse = np.sqrt(((pred_orthogonal - next_states) ** 2).mean(axis=1)).astype(np.float32)
    symplectic_rmse = np.sqrt(((pred_symplectic - next_states) ** 2).mean(axis=1)).astype(np.float32)
    empirical_symplectic_defect = measure_symplectic_defect(empirical_operator, symplectic_form)
    projected_symplectic_defect = measure_symplectic_defect(symplectic_operator, symplectic_form)
    generator_projection_gap = float(np.linalg.norm(empirical_generator - hamiltonian_generator) / max(np.linalg.norm(empirical_generator), 1e-6))
    hamiltonian_energy_matrix = (-symplectic_form.astype(np.float64) @ hamiltonian_generator.astype(np.float64))
    hamiltonian_energy_matrix = 0.5 * (hamiltonian_energy_matrix + hamiltonian_energy_matrix.T)
    canonical_energy = (0.5 * np.einsum("ti,ij,tj->t", canonical_state.astype(np.float64), hamiltonian_energy_matrix, canonical_state.astype(np.float64))).astype(np.float32)
    energy_drift = np.abs(np.diff(canonical_energy)).astype(np.float32)
    reciprocity_matrix = (-symplectic_form.astype(np.float64) @ empirical_generator.astype(np.float64)).astype(np.float32)
    reciprocity_gap = float(np.linalg.norm(reciprocity_matrix - reciprocity_matrix.T) / max(np.linalg.norm(reciprocity_matrix), 1e-6))
    k_block = reciprocity_matrix[:pair_count, :pair_count]
    mass_block = reciprocity_matrix[pair_count:, pair_count:]
    coupling_block = reciprocity_matrix[:pair_count, pair_count:]
    dual_coupling_block = reciprocity_matrix[pair_count:, :pair_count].T
    spring_reciprocity_gap = float(np.linalg.norm(k_block - k_block.T) / max(np.linalg.norm(k_block), 1e-6))
    mass_reciprocity_gap = float(np.linalg.norm(mass_block - mass_block.T) / max(np.linalg.norm(mass_block), 1e-6))
    coupling_reciprocity_gap = float(np.linalg.norm(coupling_block - dual_coupling_block) / max(np.linalg.norm(coupling_block), 1e-6))
    methodology_scores = {
        "orthogonal": float(orthogonal_rmse.mean()),
        "symplectic": float(symplectic_rmse.mean()),
    }
    preferred_reference = min(methodology_scores, key=methodology_scores.get)
    return {
        "status": "ok",
        "canonical_dim": canonical_dim,
        "canonical_pair_count": int(pair_count),
        "auxiliary_dim_dropped": int(auxiliary.shape[1]),
        "pairing": pairing,
        "transition_count": int(previous_states.shape[0]),
        "ridge": float(ridge),
        "orthogonal_transition_rmse_mean": float(orthogonal_rmse.mean()),
        "orthogonal_transition_rmse_p95": float(np.quantile(orthogonal_rmse, 0.95)),
        "symplectic_transition_rmse_mean": float(symplectic_rmse.mean()),
        "symplectic_transition_rmse_p95": float(np.quantile(symplectic_rmse, 0.95)),
        "empirical_operator_symplectic_defect": empirical_symplectic_defect,
        "projected_symplectic_defect": projected_symplectic_defect,
        "hamiltonian_projection_gap": generator_projection_gap,
        "hamiltonian_generator_norm": float(np.linalg.norm(hamiltonian_generator)),
        "canonical_energy_mean": float(canonical_energy.mean()) if canonical_energy.size else 0.0,
        "canonical_energy_drift_mean_abs": float(energy_drift.mean()) if energy_drift.size else 0.0,
        "canonical_energy_drift_p95_abs": float(np.quantile(energy_drift, 0.95)) if energy_drift.size else 0.0,
        "reciprocity_gap": reciprocity_gap,
        "spring_reciprocity_gap": spring_reciprocity_gap,
        "mass_reciprocity_gap": mass_reciprocity_gap,
        "coupling_reciprocity_gap": coupling_reciprocity_gap,
        "preferred_reference": preferred_reference,
        "methodology_comparison": methodology_scores,
    }, {
        f"{array_prefix}canonical_state_matrix": canonical_state.astype(np.float32),
        f"{array_prefix}canonical_q_matrix": q_matrix.astype(np.float32),
        f"{array_prefix}canonical_p_matrix": p_matrix.astype(np.float32),
        f"{array_prefix}canonical_auxiliary_matrix": auxiliary.astype(np.float32),
        f"{array_prefix}canonical_energy": canonical_energy.astype(np.float32),
        f"{array_prefix}canonical_energy_drift": energy_drift.astype(np.float32),
        f"{array_prefix}canonical_empirical_operator": empirical_operator.astype(np.float32),
        f"{array_prefix}canonical_orthogonal_operator": orthogonal_operator.astype(np.float32),
        f"{array_prefix}canonical_symplectic_operator": symplectic_operator.astype(np.float32),
        f"{array_prefix}canonical_hamiltonian_generator": hamiltonian_generator.astype(np.float32),
        f"{array_prefix}canonical_orthogonal_transition_rmse": orthogonal_rmse.astype(np.float32),
        f"{array_prefix}canonical_symplectic_transition_rmse": symplectic_rmse.astype(np.float32),
    }


def project_lattice_to_markov_latent(lattice_tensor) -> np.ndarray:
    return lattice_tensor.detach().cpu().float().mean(dim=(0, 2)).numpy().astype(np.float32)


def make_collapsed_wave_state(torch, resolved_state: np.ndarray, output_dim: int, lattice_size: int):
    state = torch.from_numpy(resolved_state.astype(np.float32))
    expanded = state.view(1, 1, state.shape[0], 1).expand(1, output_dim, -1, lattice_size).contiguous()
    return torch.complex(expanded, torch.zeros_like(expanded))


def analyze_joint_system(
    raw_sequences: dict[StreamKey, np.ndarray],
    normalized_sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    feature_names: list[str],
    ridge: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not raw_sequences:
        return {"status": "no_streams"}, {}
    ordered_keys, bins, feature_dim = build_aligned_stream_bins(raw_sequences, normalized_sequences, metadata)
    stream_count = int(len(ordered_keys))
    clock_idx = feature_names.index("clock_mean") if "clock_mean" in feature_names else None
    decoherence_rows = []
    complete_states = []
    complete_times = []
    for time_bin in sorted(bins):
        present_keys = sorted(bins[time_bin], key=stream_label)
        present_vectors = np.stack([bins[time_bin][key]["normalized"] for key in present_keys]).astype(np.float32)
        _, deviations, mean_pairwise, max_pairwise = summarize_present_vectors(present_vectors)
        host_times = [bins[time_bin][key]["host_time_mid"] for key in present_keys]
        row = {
            "time_bin": int(time_bin),
            "host_time_mid": float(np.mean(host_times)),
            "stream_count_present": int(len(present_keys)),
            "decoherence_score": float(deviations.mean()) if deviations.size else 0.0,
            "max_stream_deviation": float(deviations.max()) if deviations.size else 0.0,
            "mean_pairwise_distance": mean_pairwise,
            "max_pairwise_distance": max_pairwise,
            "host_time_spread": float(max(host_times) - min(host_times)) if len(host_times) > 1 else 0.0,
        }
        if clock_idx is not None:
            clock_values = [float(bins[time_bin][key]["raw"][clock_idx]) for key in present_keys]
            row["clock_mean_spread"] = float(max(clock_values) - min(clock_values)) if len(clock_values) > 1 else 0.0
        decoherence_rows.append(row)
        if len(present_keys) == len(ordered_keys):
            complete_times.append(float(np.mean(host_times)))
            complete_states.append(np.concatenate([bins[time_bin][key]["normalized"] for key in ordered_keys]).astype(np.float32))
    arrays: dict[str, np.ndarray] = {
        "coherence_host_time_mid": np.array([row["host_time_mid"] for row in decoherence_rows], dtype=np.float64),
        "coherence_time_bin": np.array([row["time_bin"] for row in decoherence_rows], dtype=np.int32),
        "coherence_stream_count_present": np.array([row["stream_count_present"] for row in decoherence_rows], dtype=np.int32),
        "decoherence_score": np.array([row["decoherence_score"] for row in decoherence_rows], dtype=np.float32),
        "max_stream_deviation": np.array([row["max_stream_deviation"] for row in decoherence_rows], dtype=np.float32),
        "mean_pairwise_distance": np.array([row["mean_pairwise_distance"] for row in decoherence_rows], dtype=np.float32),
        "max_pairwise_distance": np.array([row["max_pairwise_distance"] for row in decoherence_rows], dtype=np.float32),
        "host_time_spread": np.array([row["host_time_spread"] for row in decoherence_rows], dtype=np.float32),
    }
    if decoherence_rows and "clock_mean_spread" in decoherence_rows[0]:
        arrays["clock_mean_spread"] = np.array([row["clock_mean_spread"] for row in decoherence_rows], dtype=np.float32)
    summary = {
        "stream_count": stream_count,
        "feature_dim_per_stream": feature_dim,
        "aligned_bin_count": int(len(decoherence_rows)),
        "complete_bin_count": int(len(complete_states)),
        "complete_bin_fraction": float(len(complete_states) / max(len(decoherence_rows), 1)),
        "mean_streams_per_bin": float(np.mean(arrays["coherence_stream_count_present"])) if decoherence_rows else 0.0,
        "mean_decoherence_score": float(arrays["decoherence_score"].mean()) if decoherence_rows else 0.0,
        "peak_decoherence_score": float(arrays["decoherence_score"].max()) if decoherence_rows else 0.0,
        "peak_decoherence_host_time_mid": float(arrays["coherence_host_time_mid"][int(arrays["decoherence_score"].argmax())]) if decoherence_rows else None,
        "mean_pairwise_distance": float(arrays["mean_pairwise_distance"].mean()) if decoherence_rows else 0.0,
        "peak_pairwise_distance": float(arrays["max_pairwise_distance"].max()) if decoherence_rows else 0.0,
    }
    if len(complete_states) < 2:
        summary["operator_fit"] = {
            "status": "insufficient_complete_bins",
            "message": "Need at least two fully aligned joint-state bins to estimate a transition operator.",
        }
        return summary, arrays
    state_matrix = np.stack(complete_states).astype(np.float32)
    operator_summary, operator_arrays = summarize_linear_operator_fit(state_matrix, ridge=ridge)
    arrays.update({
        "complete_state_host_time_mid": np.array(complete_times, dtype=np.float64),
        **operator_arrays,
    })
    summary["operator_fit"] = operator_summary
    return summary, arrays


def analyze_entangled_shared_now(
    raw_sequences: dict[StreamKey, np.ndarray],
    normalized_sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    ridge: float,
    lattice_size: int,
    memory_mix: float,
    event_z_threshold: float,
    event_min_separation: int,
    event_top_k: int,
    canonical_pairing: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if not raw_sequences:
        return {"status": "no_streams"}, {}, {"events": [], "parameters": {}}
    torch, EntangledCaMKIILattice = load_entangled_memory_components()
    ordered_keys, bins, feature_dim = build_aligned_stream_bins(raw_sequences, normalized_sequences, metadata)
    if not ordered_keys:
        return {"status": "no_streams"}, {}, {"events": [], "parameters": {}}
    stream_labels = [stream_label(key) for key in ordered_keys]
    shared_lattice = EntangledCaMKIILattice(feature_dim, feature_dim, lattice_size=lattice_size)
    per_stream_deviation_rows: dict[str, list[float]] = {label: [] for label in stream_labels}
    
    # Pre-allocate rows for faster appending
    rows = []
    
    # Sort bins once
    sorted_bins = sorted(bins)
    num_bins = len(sorted_bins)
    
    # Extract structural metadata upfront
    host_time_mids = np.zeros(num_bins, dtype=np.float64)
    time_bins = np.zeros(num_bins, dtype=np.int32)
    stream_counts = np.zeros(num_bins, dtype=np.int32)
    
    # Vectorize the present vectors processing
    max_streams = len(ordered_keys)
    present_vectors_all = np.zeros((num_bins, max_streams, feature_dim), dtype=np.float32)
    present_mask = np.zeros((num_bins, max_streams), dtype=bool)
    
    key_to_idx = {key: i for i, key in enumerate(ordered_keys)}
    
    for i, time_bin in enumerate(sorted_bins):
        host_times = []
        for key in bins[time_bin]:
            idx = key_to_idx[key]
            present_vectors_all[i, idx] = bins[time_bin][key]["normalized"]
            present_mask[i, idx] = True
            host_times.append(bins[time_bin][key]["host_time_mid"])
            
        time_bins[i] = time_bin
        host_time_mids[i] = np.mean(host_times)
        stream_counts[i] = len(host_times)

    # Perform the sequential memory geometry updates
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared_lattice = shared_lattice.to(device)
    
    # Output arrays
    shared_now_states = np.zeros((num_bins, feature_dim), dtype=np.float32)
    shared_now_dev_mean = np.zeros(num_bins, dtype=np.float32)
    shared_now_dev_max = np.zeros(num_bins, dtype=np.float32)
    mean_pairwise_distaa = np.zeros(num_bins, dtype=np.float32)
    max_pairwise_distaa = np.zeros(num_bins, dtype=np.float32)
    coherence_strengths = np.zeros(num_bins, dtype=np.float32)
    memory_context_norms = np.zeros(num_bins, dtype=np.float32)
    memory_consensus_gaps = np.zeros(num_bins, dtype=np.float32)
    memory_write_scores = np.zeros(num_bins, dtype=np.float32)
    memory_deltas = np.zeros(num_bins, dtype=np.float32)
    per_stream_devs = np.full((num_bins, max_streams), np.nan, dtype=np.float32)

    memory_before_np = project_lattice_to_markov_latent(shared_lattice.get_lattice())
    
    for i in range(num_bins):
        mask_i = present_mask[i]
        present_vectors = present_vectors_all[i, mask_i]
        
        consensus, _, mean_pairwise, max_pairwise = summarize_present_vectors(present_vectors)
        
        shared_now = ((1.0 - memory_mix) * consensus + memory_mix * memory_before_np).astype(np.float32)
        
        # Calculate deviations
        shared_now_deviations = np.linalg.norm(present_vectors - shared_now, axis=1).astype(np.float32)
        
        coherence_strength = float(1.0 / (1.0 + mean_pairwise))
        memory_consensus_gap = float(np.linalg.norm(consensus - memory_before_np))
        memory_write_score = float(memory_consensus_gap * coherence_strength)
        
        # Push to lattice
        collapsed_state = make_collapsed_wave_state(torch, shared_now, output_dim=feature_dim, lattice_size=lattice_size).to(device)
        shared_lattice.write_silhouette(collapsed_state)
        memory_after_np = project_lattice_to_markov_latent(shared_lattice.get_lattice())
        
        memory_delta = float(np.linalg.norm(memory_after_np - memory_before_np))
        memory_context_norms[i] = float(np.linalg.norm(memory_before_np))
        
        # Store results
        shared_now_states[i] = shared_now
        shared_now_dev_mean[i] = shared_now_deviations.mean() if shared_now_deviations.size else 0.0
        shared_now_dev_max[i] = shared_now_deviations.max() if shared_now_deviations.size else 0.0
        mean_pairwise_distaa[i] = mean_pairwise
        max_pairwise_distaa[i] = max_pairwise
        coherence_strengths[i] = coherence_strength
        memory_consensus_gaps[i] = memory_consensus_gap
        memory_write_scores[i] = memory_write_score
        memory_deltas[i] = memory_delta
        
        per_stream_devs[i, mask_i] = shared_now_deviations
        
        memory_before_np = memory_after_np

    # Build final structures
    rows = [
        {
            "time_bin": int(time_bins[i]),
            "host_time_mid": float(host_time_mids[i]),
            "stream_count_present": int(stream_counts[i]),
            "shared_now_deviation_mean": float(shared_now_dev_mean[i]),
            "shared_now_deviation_max": float(shared_now_dev_max[i]),
            "mean_pairwise_distance": float(mean_pairwise_distaa[i]),
            "max_pairwise_distance": float(max_pairwise_distaa[i]),
            "coherence_strength": float(coherence_strengths[i]),
            "memory_context_norm": float(memory_context_norms[i]),
            "shared_now_norm": float(np.linalg.norm(shared_now_states[i])),
            "memory_consensus_gap": float(memory_consensus_gaps[i]),
            "memory_write_score": float(memory_write_scores[i]),
            "memory_delta": float(memory_deltas[i]),
        }
        for i in range(num_bins)
    ]

    arrays: dict[str, np.ndarray] = {
        "shared_now_host_time_mid": host_time_mids,
        "shared_now_time_bin": time_bins,
        "shared_now_stream_count_present": stream_counts,
        "shared_now_deviation_mean": shared_now_dev_mean,
        "shared_now_deviation_max": shared_now_dev_max,
        "shared_now_mean_pairwise_distance": mean_pairwise_distaa,
        "shared_now_max_pairwise_distance": max_pairwise_distaa,
        "shared_now_coherence_strength": coherence_strengths,
        "memory_context_norm": memory_context_norms,
        "memory_consensus_gap": memory_consensus_gaps,
        "memory_write_score": memory_write_scores,
        "memory_delta": memory_deltas,
        "shared_now_state_matrix": shared_now_states,
    }
    for label, values in per_stream_deviation_rows.items():
        arrays[f"shared_now_deviation_{label}"] = np.array(values, dtype=np.float32)
    peak_indices, z_scores = find_peak_indices(
        arrays["memory_write_score"],
        z_threshold=event_z_threshold,
        min_separation=event_min_separation,
        max_events=max(event_top_k, 1),
        fallback_top_k=max(event_top_k, 1),
    )
    events = [
        {
            "event_index": int(idx),
            "time_bin": int(rows[idx]["time_bin"]),
            "host_time_mid": float(rows[idx]["host_time_mid"]),
            "memory_write_score": float(rows[idx]["memory_write_score"]),
            "memory_write_score_z": float(z_scores[idx]),
            "memory_consensus_gap": float(rows[idx]["memory_consensus_gap"]),
            "shared_now_deviation_mean": float(rows[idx]["shared_now_deviation_mean"]),
            "stream_count_present": int(rows[idx]["stream_count_present"]),
        }
        for idx in peak_indices
    ]
    operator_summary, operator_arrays = summarize_linear_operator_fit(
        arrays["shared_now_state_matrix"],
        ridge=ridge,
        array_prefix="shared_now_",
    )
    oscillator_summary, oscillator_arrays = summarize_canonical_oscillator_fit(
        arrays["shared_now_state_matrix"],
        ridge=ridge,
        pairing=canonical_pairing,
        array_prefix="shared_now_",
    )
    arrays.update(operator_arrays)
    arrays.update(oscillator_arrays)
    summary = {
        "stream_count": int(len(ordered_keys)),
        "latent_dim": feature_dim,
        "aligned_bin_count": int(len(rows)),
        "memory_lattice_size": int(lattice_size),
        "memory_mix": float(memory_mix),
        "mean_shared_now_deviation": float(arrays["shared_now_deviation_mean"].mean()) if rows else 0.0,
        "peak_shared_now_deviation": float(arrays["shared_now_deviation_max"].max()) if rows else 0.0,
        "mean_memory_consensus_gap": float(arrays["memory_consensus_gap"].mean()) if rows else 0.0,
        "peak_memory_consensus_gap": float(arrays["memory_consensus_gap"].max()) if rows else 0.0,
        "mean_memory_write_score": float(arrays["memory_write_score"].mean()) if rows else 0.0,
        "peak_memory_write_score": float(arrays["memory_write_score"].max()) if rows else 0.0,
        "peak_memory_write_host_time_mid": float(arrays["shared_now_host_time_mid"][int(arrays["memory_write_score"].argmax())]) if rows else None,
        "shared_now_event_count": int(len(events)),
        "per_stream_mean_shared_now_deviation": {
            label: float(np.nanmean(values)) if np.isfinite(values).any() else None
            for label, values in ((label, arrays[f"shared_now_deviation_{label}"]) for label in stream_labels)
        },
        "operator_fit": operator_summary,
        "hamiltonian_qp_analysis": oscillator_summary,
        "methodology_comparison": oscillator_summary.get("methodology_comparison", {}),
        "projection": "The shared CaMKII lattice is projected into a Markovian latent state by averaging over lattice geometry and output channels, yielding a shared-now trajectory in feature space.",
    }
    event_payload = {
        "events": events,
        "parameters": {
            "z_threshold": float(event_z_threshold),
            "min_separation": int(event_min_separation),
            "top_k": int(event_top_k),
        },
    }
    return summary, arrays, event_payload


def split_canonical_tensor(torch, state_tensor, pairing: str = "half_split"):
    pair_count = int(state_tensor.shape[-1] // 2)
    if pair_count <= 0:
        empty = state_tensor[..., :0]
        return empty, empty, empty, state_tensor
    if pairing == "interleaved":
        canonical_slice = state_tensor[..., :2 * pair_count]
        q = canonical_slice[..., 0::2]
        p = canonical_slice[..., 1::2]
        auxiliary = state_tensor[..., 2 * pair_count:]
    else:
        q = state_tensor[..., :pair_count]
        p = state_tensor[..., pair_count:2 * pair_count]
        auxiliary = state_tensor[..., 2 * pair_count:]
    canonical_state = torch.cat([q, p], dim=-1)
    return canonical_state, q, p, auxiliary


def compute_oscillator_regularization_terms(torch, sequence_batch, pairing: str = "half_split") -> dict[str, Any]:
    zero = sequence_batch.new_zeros(())
    if sequence_batch.shape[-1] < 2 or sequence_batch.shape[-2] < 2:
        return {"qp": zero, "energy": zero, "spring": zero, "canonical_pair_count": 0}
    _, q, p, _ = split_canonical_tensor(torch, sequence_batch, pairing=pairing)
    if q.shape[-1] == 0 or q.shape[-2] < 2:
        return {"qp": zero, "energy": zero, "spring": zero, "canonical_pair_count": 0}
    q_delta = q[..., 1:, :] - q[..., :-1, :]
    p_mid = 0.5 * (p[..., 1:, :] + p[..., :-1, :])
    qp_loss = (q_delta - p_mid).pow(2).mean()
    canonical_energy = 0.5 * (q.pow(2) + p.pow(2)).sum(dim=-1)
    energy_loss = (canonical_energy[..., 1:] - canonical_energy[..., :-1]).pow(2).mean()
    q_prev = q[..., :-1, :]
    p_delta = p[..., 1:, :] - p[..., :-1, :]
    denom = q_prev.pow(2).mean(dim=-2, keepdim=True).clamp_min(1e-6)
    spring_constant = -(p_delta * q_prev).mean(dim=-2, keepdim=True) / denom
    spring_residual = p_delta + q_prev * spring_constant
    spring_loss = spring_residual.pow(2).mean()
    return {
        "qp": qp_loss,
        "energy": energy_loss,
        "spring": spring_loss,
        "canonical_pair_count": int(q.shape[-1]),
    }


def train_model(inputs: np.ndarray, targets: np.ndarray, args: argparse.Namespace, device) -> tuple[Any, list[float], dict[str, list[float]]]:
    torch, F, DLinOSSModel = load_torch_components()
    model = DLinOSSModel(
        input_dim=inputs.shape[-1],
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        output_dim=targets.shape[-1],
        num_blocks=args.num_blocks,
        classification=False,
        drop_rate=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    x_all = torch.from_numpy(inputs)
    y_all = torch.from_numpy(targets)
    history: list[float] = []
    training_terms = {
        "prediction_loss": [],
        "qp_regularizer": [],
        "energy_regularizer": [],
        "spring_regularizer": [],
        "total_regularizer": [],
    }
    from tqdm import tqdm
    for _ in tqdm(range(args.epochs), desc="Training epochs", leave=False):
        model.train()
        permutation = torch.randperm(x_all.shape[0])
        total_loss = 0.0
        total_prediction_loss = 0.0
        total_qp_loss = 0.0
        total_energy_loss = 0.0
        total_spring_loss = 0.0
        total_regularizer = 0.0
        total_examples = 0
        for start in range(0, x_all.shape[0], args.batch_size):
            idx = permutation[start:start + args.batch_size]
            xb = x_all[idx].to(device)
            yb = y_all[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            prediction_loss = F.mse_loss(pred, yb)
            regularization_terms = compute_oscillator_regularization_terms(torch, pred, pairing=args.canonical_pairing)
            regularizer = (
                args.qp_regularization_weight * regularization_terms["qp"]
                + args.energy_drift_regularization_weight * regularization_terms["energy"]
                + args.spring_regularization_weight * regularization_terms["spring"]
            )
            loss = prediction_loss + regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            total_prediction_loss += prediction_loss.item() * xb.shape[0]
            total_qp_loss += regularization_terms["qp"].item() * xb.shape[0]
            total_energy_loss += regularization_terms["energy"].item() * xb.shape[0]
            total_spring_loss += regularization_terms["spring"].item() * xb.shape[0]
            total_regularizer += regularizer.item() * xb.shape[0]
            total_examples += xb.shape[0]
        history.append(total_loss / max(total_examples, 1))
        training_terms["prediction_loss"].append(total_prediction_loss / max(total_examples, 1))
        training_terms["qp_regularizer"].append(total_qp_loss / max(total_examples, 1))
        training_terms["energy_regularizer"].append(total_energy_loss / max(total_examples, 1))
        training_terms["spring_regularizer"].append(total_spring_loss / max(total_examples, 1))
        training_terms["total_regularizer"].append(total_regularizer / max(total_examples, 1))
    return model, history, training_terms


def evaluate_model(model, sequences: dict[StreamKey, np.ndarray], metadata: dict[StreamKey, list[dict[str, float]]], device, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    torch, _, _ = load_torch_components()
    metrics: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    processor_details: dict[str, dict[str, Any]] = {}
    
    model.eval()
    cuda_sync = getattr(args, "cuda_sync", False)
    
    # Sort keys for deterministic output
    ordered_keys = sorted(sequences.keys(), key=stream_label)
    
    # We use a thread pool for I/O and GPU launch management
    # Note: Torch models are thread-safe for evaluation if no state is updated
    from concurrent.futures import ThreadPoolExecutor
    
    def process_one_stream(key):
        sequence = sequences[key]
        if len(sequence) < 2:
            return None
        
        label = stream_label(key)
        
        # Each thread can use its own stream if we wanted even more parallelism
        # but for evaluation, serial launch on one GPU is often bottlenecked by CPU/Memory anyway.
        # However, keeping it thread-safe.
        with torch.no_grad():
            x = torch.from_numpy(sequence[:-1]).unsqueeze(0).to(device)
            y = sequence[1:]
            
            if cuda_sync and device.type == "cuda":
                torch.cuda.synchronize()
            
            pred = model(x).squeeze(0).cpu().numpy()
            
            if cuda_sync and device.type == "cuda":
                torch.cuda.synchronize()
            
            residual = pred - y
            mse = (residual ** 2).mean(axis=1)
            rmse = np.sqrt(mse)
            peak_idx = int(rmse.argmax())
            meta = metadata[key][peak_idx + 1]
            
            res_metrics = {
                "packet_count": int(len(sequence)),
                "transition_count": int(len(rmse)),
                "distortion_score_mean": float(rmse.mean()),
                "distortion_score_std": float(rmse.std()),
                "distortion_score_p95": float(np.quantile(rmse, 0.95)),
                "peak_distortion_score": float(rmse[peak_idx]),
                "peak_packet_index": int(meta["packet_index"]),
                "peak_host_time_mid": float(meta["host_time_mid"]),
                "peak_elapsed_host_seconds": float(meta["elapsed_host_seconds"]),
            }
            return label, res_metrics, rmse.astype(np.float32), {
                "rmse": rmse.astype(np.float32),
                "residual": residual.astype(np.float32),
                "observed": y.astype(np.float32),
                "metadata": metadata[key][1:],
            }

    # Parallelize across workers
    max_workers = min(len(ordered_keys), 8) if ordered_keys else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_one_stream, ordered_keys))
        
    for res in results:
        if res:
            label, res_metrics, rmse, detail = res
            metrics[label] = res_metrics
            score_arrays[label] = rmse
            processor_details[label] = detail # Flattened later anyway

    # Fix nesting if necessary (evaluate_model should return flattened processor_details)
    flattened_details = {}
    for label, res_metrics in metrics.items():
        # Find detail for this label
        for res in results:
            if res and res[0] == label:
                flattened_details[label] = res[3]
                break

    return metrics, score_arrays, flattened_details


def train_observed_stream_models(
    sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    stream_info: dict[StreamKey, dict[str, Any]],
    args: argparse.Namespace,
    device,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    summaries: dict[str, Any] = {}
    metrics_by_stream: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    processor_details: dict[str, dict[str, Any]] = {}
    spectral_summaries: dict[str, dict[str, Any]] = {}
    model_states: dict[str, Any] = {}
    from tqdm import tqdm
    print("Training observed stream models...")
    for key in tqdm(sorted(sequences.keys(), key=stream_label), desc="Observed streams"):
        label = stream_label(key)
        sequence = sequences[key]
        identity = dict(stream_info.get(key, stream_key_dict(key)))
        if len(sequence) < args.window_size:
            summaries[label] = {
                "status": "insufficient_packets",
                **identity,
                "packet_count": int(len(sequence)),
                "window_size": int(args.window_size),
            }
            continue
        inputs, targets = make_windows({key: sequence}, window_size=args.window_size, stride=args.stride)
        model, history, training_terms = train_model(inputs, targets, args, device)
        metrics, local_scores, local_details = evaluate_model(model, {key: sequence}, {key: metadata[key]}, device, args)
        metric = metrics.get(label)
        detail = local_details.get(label)
        if metric is None or detail is None:
            summaries[label] = {
                "status": "evaluation_missing",
                **identity,
                "packet_count": int(len(sequence)),
                "window_count": int(inputs.shape[0]),
            }
            continue
        summaries[label] = {
            "status": "ok",
            **identity,
            "packet_count": int(len(sequence)),
            "window_count": int(inputs.shape[0]),
            "transition_count": int(metric["transition_count"]),
            "final_loss": float(history[-1]),
            "final_prediction_loss": float(training_terms["prediction_loss"][-1]),
            "final_total_regularizer": float(training_terms["total_regularizer"][-1]),
            "mean_distortion_score": float(metric["distortion_score_mean"]),
            "peak_distortion_score": float(metric["peak_distortion_score"]),
            "peak_host_time_mid": float(metric["peak_host_time_mid"]),
        }
        metrics_by_stream[label] = metric
        score_arrays[label] = local_scores[label]
        processor_details[label] = detail
        spectral_summaries[label] = {
            name: float(value) if isinstance(value, (int, float)) else value
            for name, value in model.spectral_summary().items()
        }
        model_states[label] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return summaries, metrics_by_stream, score_arrays, processor_details, spectral_summaries, model_states


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    if len(vectors) <= 1:
        return 1.0 if len(vectors) == 1 else 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms < 1e-8, 1.0, norms)
    normalized = vectors / safe
    sims = normalized @ normalized.T
    upper = sims[np.triu_indices(len(vectors), k=1)]
    return float(upper.mean()) if upper.size else 1.0


def train_now_stream_agent_models(
    agent_sequences: dict[StreamKey, np.ndarray],
    agent_metadata: dict[StreamKey, list[dict[str, float]]],
    args: argparse.Namespace,
    device,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    details: dict[str, dict[str, Any]] = {}
    for key in sorted(agent_sequences.keys(), key=stream_label):
        label = stream_label(key)
        sequence = agent_sequences[key]
        key_info = stream_key_dict(key)
        if len(sequence) < args.window_size:
            summaries[label] = {
                "status": "insufficient_packets",
                "packet_count": int(len(sequence)),
                "window_size": int(args.window_size),
                "now_scale_radius": int(key_info.get("now_scale_radius", 0)),
            }
            continue
        inputs, targets = make_windows({key: sequence}, window_size=args.window_size, stride=args.stride)
        model, history, training_terms = train_model(inputs, targets, args, device)
        metrics, _, processor_details = evaluate_model(model, {key: sequence}, {key: agent_metadata[key]}, device, args)
        metric = metrics.get(label)
        detail = processor_details.get(label)
        if metric is None or detail is None:
            summaries[label] = {
                "status": "evaluation_missing",
                "packet_count": int(len(sequence)),
                "window_count": int(inputs.shape[0]),
                "now_scale_radius": int(key_info.get("now_scale_radius", 0)),
            }
            continue
        summaries[label] = {
            "status": "ok",
            "packet_count": int(len(sequence)),
            "window_count": int(inputs.shape[0]),
            "transition_count": int(metric["transition_count"]),
            "now_scale_radius": int(key_info.get("now_scale_radius", 0)),
            "final_loss": float(history[-1]),
            "final_prediction_loss": float(training_terms["prediction_loss"][-1]),
            "mean_distortion_score": float(metric["distortion_score_mean"]),
            "peak_distortion_score": float(metric["peak_distortion_score"]),
            "peak_host_time_mid": float(metric["peak_host_time_mid"]),
        }
        details[label] = detail
    return summaries, details


def analyze_now_stream_mesh(
    raw_sequences: dict[StreamKey, np.ndarray],
    normalized_sequences: dict[StreamKey, np.ndarray],
    metadata: dict[StreamKey, list[dict[str, float]]],
    temporal_radii: list[int],
    ridge: float,
    lattice_size: int,
    memory_mix: float,
    event_z_threshold: float,
    event_min_separation: int,
    event_top_k: int,
    canonical_pairing: str,
    agent_processor_details: dict[str, dict[str, Any]] | None = None,
    agent_training_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if not raw_sequences:
        return {"status": "no_streams"}, {}, {"events": [], "parameters": {}}
    agent_payload = build_now_stream_agents(raw_sequences, normalized_sequences, metadata, temporal_radii)
    if not agent_payload["raw_sequences"]:
        return {"status": "no_agents"}, {}, {"events": [], "parameters": {}}
    torch, EntangledCaMKIILattice = load_entangled_memory_components()
    ordered_keys, bins, feature_dim = build_aligned_stream_bins(
        agent_payload["raw_sequences"],
        agent_payload["normalized_sequences"],
        agent_payload["metadata"],
        extra_meta_fields=("agent_believed_now_host_time_mid", "agent_now_offset_seconds"),
    )
    if not ordered_keys:
        return {"status": "no_aligned_bins"}, {}, {"events": [], "parameters": {}}
    agent_labels = [stream_label(key) for key in ordered_keys]
    shared_lattice = EntangledCaMKIILattice(feature_dim, feature_dim, lattice_size=lattice_size)
    qpc_rows: dict[str, list[float]] = {label: [] for label in agent_labels}
    now_rows: dict[str, list[float]] = {label: [] for label in agent_labels}
    offset_rows: dict[str, list[float]] = {label: [] for label in agent_labels}
    adapt_rows: dict[str, list[float]] = {label: [] for label in agent_labels}
    agent_sequences = agent_payload["normalized_sequences"]
    fallback_adaptation = {
        stream_label(key): {
            int(meta["time_bin"]): float(np.linalg.norm(sequence[idx] - sequence[idx - 1])) if idx > 0 else 0.0
            for idx, meta in enumerate(agent_payload["metadata"][key])
        }
        for key, sequence in agent_sequences.items()
    }
    trained_adaptation: dict[str, dict[int, float]] = {}
    for label, detail in (agent_processor_details or {}).items():
        by_bin: dict[int, list[float]] = defaultdict(list)
        for rmse, meta in zip(detail.get("rmse", []), detail.get("metadata", [])):
            by_bin[int(meta["time_bin"])].append(float(rmse))
        trained_adaptation[label] = {time_bin: float(np.mean(values)) for time_bin, values in by_bin.items()}
    rows = []
    qpc_states = []
    memory_states = []
    complete_mesh_states = []
    previous_qpc_state = None
    for time_bin in sorted(bins):
        present_keys = sorted(bins[time_bin], key=stream_label)
        present_vectors = np.stack([bins[time_bin][key]["normalized"] for key in present_keys]).astype(np.float32)
        consensus, _, mean_pairwise_distance, max_pairwise_distance = summarize_present_vectors(present_vectors)
        host_times = [float(bins[time_bin][key]["host_time_mid"]) for key in present_keys]
        believed_now = [float(bins[time_bin][key]["agent_believed_now_host_time_mid"]) for key in present_keys]
        memory_before = project_lattice_to_markov_latent(shared_lattice.get_lattice())
        shared_qpc_now = ((1.0 - memory_mix) * consensus + memory_mix * memory_before).astype(np.float32)
        coherence_values = np.array([cosine_similarity(vector, shared_qpc_now) for vector in present_vectors], dtype=np.float32)
        gregorian_now = float(np.mean(host_times))
        offset_values = np.array([value - gregorian_now for value in believed_now], dtype=np.float32)
        believed_now_spread = float(max(believed_now) - min(believed_now)) if believed_now else 0.0
        memory_consensus_gap = float(np.linalg.norm(consensus - memory_before))
        memory_write_score = float(memory_consensus_gap * max(float(coherence_values.mean()) if coherence_values.size else 0.0, 0.0))
        collapsed_state = make_collapsed_wave_state(torch, shared_qpc_now, output_dim=feature_dim, lattice_size=lattice_size)
        shared_lattice.write_silhouette(collapsed_state)
        memory_after = project_lattice_to_markov_latent(shared_lattice.get_lattice())
        qpc_state_delta_norm = float(np.linalg.norm(shared_qpc_now - previous_qpc_state)) if previous_qpc_state is not None else 0.0
        previous_qpc_state = shared_qpc_now.copy()
        mean_pairwise_coherence = mean_pairwise_cosine(present_vectors)
        memory_qpc_coherence = cosine_similarity(shared_qpc_now, memory_after)
        qpc_states.append(shared_qpc_now)
        memory_states.append(memory_after)
        if len(present_keys) == len(ordered_keys):
            complete_mesh_states.append(np.concatenate([bins[time_bin][key]["normalized"] for key in ordered_keys]).astype(np.float32))
        coherence_map = {stream_label(key): float(value) for key, value in zip(present_keys, coherence_values)}
        believed_map = {stream_label(key): float(value) for key, value in zip(present_keys, believed_now)}
        offset_map = {stream_label(key): float(value) for key, value in zip(present_keys, offset_values)}
        adapt_map = {
            stream_label(key): float(trained_adaptation.get(stream_label(key), fallback_adaptation.get(stream_label(key), {})).get(int(time_bin), fallback_adaptation.get(stream_label(key), {}).get(int(time_bin), 0.0)))
            for key in present_keys
        }
        for label in agent_labels:
            qpc_rows[label].append(coherence_map.get(label, np.nan))
            now_rows[label].append(believed_map.get(label, np.nan))
            offset_rows[label].append(offset_map.get(label, np.nan))
            adapt_rows[label].append(adapt_map.get(label, np.nan))
        rows.append({
            "time_bin": int(time_bin),
            "host_time_mid": gregorian_now,
            "agent_count_present": int(len(present_keys)),
            "believed_now_spread_seconds": believed_now_spread,
            "mean_agent_qpc_coherence": float(coherence_values.mean()) if coherence_values.size else 0.0,
            "mean_pairwise_agent_coherence": float(mean_pairwise_coherence),
            "mean_pairwise_distance": float(mean_pairwise_distance),
            "max_pairwise_distance": float(max_pairwise_distance),
            "memory_consensus_gap": memory_consensus_gap,
            "memory_write_score": memory_write_score,
            "memory_qpc_coherence": float(memory_qpc_coherence),
            "qpc_state_delta_norm": qpc_state_delta_norm,
            "mean_agent_adaptation_delta": float(np.nanmean(np.array(list(adapt_map.values()), dtype=np.float32))) if adapt_map else 0.0,
        })
    arrays: dict[str, np.ndarray] = {
        "shared_qpc_now_host_time_mid": np.array([row["host_time_mid"] for row in rows], dtype=np.float64),
        "shared_qpc_now_time_bin": np.array([row["time_bin"] for row in rows], dtype=np.int32),
        "shared_qpc_now_agent_count_present": np.array([row["agent_count_present"] for row in rows], dtype=np.int32),
        "shared_qpc_believed_now_spread_seconds": np.array([row["believed_now_spread_seconds"] for row in rows], dtype=np.float32),
        "shared_qpc_mean_agent_coherence": np.array([row["mean_agent_qpc_coherence"] for row in rows], dtype=np.float32),
        "shared_qpc_mean_pairwise_agent_coherence": np.array([row["mean_pairwise_agent_coherence"] for row in rows], dtype=np.float32),
        "shared_qpc_mean_pairwise_distance": np.array([row["mean_pairwise_distance"] for row in rows], dtype=np.float32),
        "shared_qpc_max_pairwise_distance": np.array([row["max_pairwise_distance"] for row in rows], dtype=np.float32),
        "shared_qpc_memory_consensus_gap": np.array([row["memory_consensus_gap"] for row in rows], dtype=np.float32),
        "shared_qpc_memory_write_score": np.array([row["memory_write_score"] for row in rows], dtype=np.float32),
        "shared_qpc_memory_coherence": np.array([row["memory_qpc_coherence"] for row in rows], dtype=np.float32),
        "shared_qpc_state_delta_norm": np.array([row["qpc_state_delta_norm"] for row in rows], dtype=np.float32),
        "shared_qpc_mean_agent_adaptation_delta": np.array([row["mean_agent_adaptation_delta"] for row in rows], dtype=np.float32),
        "shared_qpc_now_state_matrix": np.stack(qpc_states).astype(np.float32),
        "shared_qpc_memory_estimate_matrix": np.stack(memory_states).astype(np.float32),
    }
    for label in agent_labels:
        arrays[f"agent_believed_now_{label}"] = np.array(now_rows[label], dtype=np.float64)
        arrays[f"agent_believed_now_offset_{label}"] = np.array(offset_rows[label], dtype=np.float32)
        arrays[f"agent_qpc_coherence_{label}"] = np.array(qpc_rows[label], dtype=np.float32)
        arrays[f"agent_adaptation_delta_{label}"] = np.array(adapt_rows[label], dtype=np.float32)
    if complete_mesh_states:
        arrays["complete_mesh_state_matrix"] = np.stack(complete_mesh_states).astype(np.float32)
    peak_indices, z_scores = find_peak_indices(
        arrays["shared_qpc_state_delta_norm"],
        z_threshold=event_z_threshold,
        min_separation=event_min_separation,
        max_events=max(event_top_k, 1),
        fallback_top_k=max(event_top_k, 1),
    )
    recurrence_events = []
    for idx in peak_indices:
        recurrence_events.append({
            "processor_label": "mesh_observer",
            "transition_index": int(idx),
            "packet_index": int(rows[idx]["time_bin"]),
            "host_time_mid": float(rows[idx]["host_time_mid"]),
            "distortion_score": float(rows[idx]["qpc_state_delta_norm"]),
            "distortion_score_z": float(z_scores[idx]),
            "signature": np.concatenate([
                arrays["shared_qpc_now_state_matrix"][idx],
                arrays["shared_qpc_memory_estimate_matrix"][idx],
                np.array([
                    arrays["shared_qpc_believed_now_spread_seconds"][idx],
                    arrays["shared_qpc_mean_agent_coherence"][idx],
                    arrays["shared_qpc_mean_pairwise_agent_coherence"][idx],
                ], dtype=np.float32),
            ]).astype(np.float32),
        })
    clustered = cluster_phase_shift_events(recurrence_events, similarity_threshold=0.9)
    operator_summary, operator_arrays = summarize_linear_operator_fit(arrays["shared_qpc_now_state_matrix"], ridge=ridge, array_prefix="shared_qpc_now_")
    oscillator_summary, oscillator_arrays = summarize_canonical_oscillator_fit(
        arrays["shared_qpc_now_state_matrix"],
        ridge=ridge,
        pairing=canonical_pairing,
        array_prefix="shared_qpc_now_",
    )
    mesh_operator_summary, mesh_operator_arrays = summarize_linear_operator_fit(
        arrays.get("complete_mesh_state_matrix", arrays["shared_qpc_now_state_matrix"]),
        ridge=ridge,
        array_prefix="mesh_",
    )
    arrays.update(operator_arrays)
    arrays.update(oscillator_arrays)
    arrays.update(mesh_operator_arrays)
    summary = {
        "status": "ok",
        "base_stream_count": int(agent_payload["base_stream_count"]),
        "agent_count": int(len(ordered_keys)),
        "agent_temporal_radii": [int(v) for v in temporal_radii],
        "agentization": "temporal_scale_centered_rolling_mean",
        "aligned_bin_count": int(len(rows)),
        "complete_agent_bin_count": int(len(complete_mesh_states)),
        "shared_qpc_state_dim": int(feature_dim),
        "memory_lattice_size": int(lattice_size),
        "memory_mix": float(memory_mix),
        "mean_believed_now_spread_seconds": float(arrays["shared_qpc_believed_now_spread_seconds"].mean()) if rows else 0.0,
        "peak_believed_now_spread_seconds": float(arrays["shared_qpc_believed_now_spread_seconds"].max()) if rows else 0.0,
        "mean_agent_qpc_coherence": float(arrays["shared_qpc_mean_agent_coherence"].mean()) if rows else 0.0,
        "mean_pairwise_agent_coherence": float(arrays["shared_qpc_mean_pairwise_agent_coherence"].mean()) if rows else 0.0,
        "mean_memory_write_score": float(arrays["shared_qpc_memory_write_score"].mean()) if rows else 0.0,
        "peak_memory_write_score": float(arrays["shared_qpc_memory_write_score"].max()) if rows else 0.0,
        "mean_memory_qpc_coherence": float(arrays["shared_qpc_memory_coherence"].mean()) if rows else 0.0,
        "mean_qpc_state_delta_norm": float(arrays["shared_qpc_state_delta_norm"].mean()) if rows else 0.0,
        "recurrence_candidate_count": int(len(clustered["events"])),
        "recurring_state_class_count": int(clustered["repeated_class_count"]),
        "per_agent_mean_qpc_coherence": {
            label: float(np.nanmean(arrays[f"agent_qpc_coherence_{label}"])) if np.isfinite(arrays[f"agent_qpc_coherence_{label}"]).any() else None
            for label in agent_labels
        },
        "per_agent_mean_believed_now_offset_seconds": {
            label: float(np.nanmean(arrays[f"agent_believed_now_offset_{label}"])) if np.isfinite(arrays[f"agent_believed_now_offset_{label}"]).any() else None
            for label in agent_labels
        },
        "per_agent_mean_adaptation_delta": {
            label: float(np.nanmean(arrays[f"agent_adaptation_delta_{label}"])) if np.isfinite(arrays[f"agent_adaptation_delta_{label}"]).any() else None
            for label in agent_labels
        },
        "agent_manifest": agent_payload["manifest"],
        "agent_model_training": agent_training_summary or {},
        "shared_qpc_operator_fit": operator_summary,
        "shared_qpc_hamiltonian_qp_analysis": oscillator_summary,
        "mesh_operator_fit": mesh_operator_summary,
        "projection": "The mesh observer aligns agent states by Gregorian host_time_mid, compares each agent's believed-now offset to the others, and measures directional ghost-basin coherence to a universal shared QPC-now state plus the system's shared memory estimate.",
    }
    event_payload = {
        "events": [
            {
                "event_index": int(event["transition_index"]),
                "time_bin": int(event["packet_index"]),
                "host_time_mid": float(event["host_time_mid"]),
                "qpc_state_delta_norm": float(event["distortion_score"]),
                "qpc_state_delta_norm_z": float(event["distortion_score_z"]),
                "class_id": int(event["class_id"]),
                "class_similarity": float(event["class_similarity"]),
            }
            for event in clustered["events"]
        ],
        "classes": clustered["classes"],
        "class_count": int(clustered["class_count"]),
        "repeated_class_count": int(clustered["repeated_class_count"]),
        "one_off_class_count": int(clustered["one_off_class_count"]),
        "repeated_event_count": int(clustered["repeated_event_count"]),
        "parameters": {
            "z_threshold": float(event_z_threshold),
            "min_separation": int(event_min_separation),
            "top_k": int(event_top_k),
            "similarity_threshold": 0.9,
        },
    }
    return summary, arrays, event_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn packet-level time-dilation patterns with local D-LinOSS.")
    parser.add_argument("--input", type=Path, default=Path("time_dilation_data.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("python/resonance_results/time_dilation_dlinoss"))
    parser.add_argument("--time-stream-cache-dir", type=Path, default=None, help="Optional cache directory for deterministic ordered packet records.")
    parser.add_argument("--time-stream-cache-chunk-size", type=int, default=2048, help="Chunk size used when preparing the deterministic time-stream cache.")
    parser.add_argument("--rebuild-time-stream-cache", action="store_true", help="Force rebuilding the time-stream cache from the source .npy file.")
    parser.add_argument("--prepare-time-stream-cache-only", action="store_true", help="Prepare the deterministic time-stream cache, print its manifest summary, and exit.")
    parser.add_argument("--stream-granularity", choices=STREAM_GRANULARITY_CHOICES, default="stream_id")
    parser.add_argument("--window-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--phase-z-threshold", type=float, default=2.5)
    parser.add_argument("--phase-min-separation", type=int, default=8)
    parser.add_argument("--phase-context-radius", type=int, default=2)
    parser.add_argument("--phase-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--phase-max-events-per-processor", type=int, default=32)
    parser.add_argument("--phase-fallback-top-k", type=int, default=8)
    parser.add_argument("--operator-ridge", type=float, default=1e-3)
    parser.add_argument("--entangled-lattice-size", type=int, default=9)
    parser.add_argument("--shared-now-memory-mix", type=float, default=0.35)
    parser.add_argument("--shared-now-event-z-threshold", type=float, default=1.5)
    parser.add_argument("--shared-now-event-min-separation", type=int, default=4)
    parser.add_argument("--shared-now-event-top-k", type=int, default=8)
    parser.add_argument(
        "--now-agent-temporal-radii",
        type=parse_int_list,
        default=[0],
        help="Comma-separated centered rolling radii used to instantiate concurrent now-stream agents.",
    )
    parser.add_argument("--canonical-pairing", choices=("half_split", "interleaved"), default="half_split")
    parser.add_argument("--qp-regularization-weight", type=float, default=0.02)
    parser.add_argument("--energy-drift-regularization-weight", type=float, default=0.01)
    parser.add_argument("--spring-regularization-weight", type=float, default=0.02)
    parser.add_argument("--cuda-sync", action="store_true", help="Perform explicit torch.cuda.synchronize() during evaluation to offset asynchronous jitter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_is_stream_dataset = is_time_stream_dataset_path(args.input)
    cache_dir = None if input_is_stream_dataset else (args.time_stream_cache_dir or (args.output_dir / "time_stream_cache" if args.prepare_time_stream_cache_only else None))
    cache_manifest: dict[str, Any] | None = None
    dataset_manifest: dict[str, Any] | None = None
    if input_is_stream_dataset:
        dataset_manifest = load_time_stream_dataset_manifest(args.input)
        if args.prepare_time_stream_cache_only:
            print(json.dumps({
                "prepared_only": True,
                "input_is_stream_dataset": True,
                "input_path": str(args.input),
                "time_stream_cache_dir": None,
                "time_stream_cache_status": "not_needed",
                "record_count": int(dataset_manifest.get("packet_count", 0)),
                "chunk_count": int(len(dataset_manifest.get("chunk_files", []))),
            }, indent=2))
            return 0
    if cache_dir is not None:
        cache_manifest = prepare_time_stream_cache(
            args.input,
            cache_dir,
            chunk_size=args.time_stream_cache_chunk_size,
            force=args.rebuild_time_stream_cache,
        )
        if args.prepare_time_stream_cache_only:
            print(json.dumps({
                "prepared_only": True,
                "input_path": str(args.input),
                "time_stream_cache_dir": str(cache_dir),
                "time_stream_cache_status": cache_manifest["cache_status"],
                "record_count": int(cache_manifest["record_count"]),
                "chunk_count": int(len(cache_manifest["chunk_files"])),
            }, indent=2))
            return 0
    torch, _, _ = load_torch_components()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if input_is_stream_dataset and dataset_manifest is not None:
        records, dataset_manifest = load_time_stream_dataset_records(args.input, max_records=args.max_records)
        total_records = int(dataset_manifest.get("packet_count", len(records)))
        manifest_path = time_stream_dataset_manifest_path(args.input)
        dataset_size = int(manifest_path.stat().st_size + sum((args.input / str(name)).stat().st_size for name in dataset_manifest.get("chunk_files", [])))
        time_stream_source = {
            "mode": "stream_dataset",
            "cache_dir": None,
            "cache_status": "not_needed",
            "chunk_count": int(len(dataset_manifest.get("chunk_files", []))),
            "source_path": str(args.input.resolve()),
            "source_size_bytes": dataset_size,
            "dataset_type": dataset_manifest.get("dataset_type", "unknown"),
            "stream_count": int(dataset_manifest.get("stream_count", 0)),
            "workers_per_gpu": int(dataset_manifest.get("workers_per_gpu", 1)),
        }
    elif cache_dir is not None and cache_manifest is not None:
        records, loaded_manifest = load_cached_packet_records(args.input, cache_dir, max_records=args.max_records)
        total_records = int(loaded_manifest["record_count"])
        time_stream_source = {
            "mode": "time_stream_cache",
            "cache_dir": str(cache_dir),
            "cache_status": cache_manifest["cache_status"],
            "chunk_count": int(len(loaded_manifest.get("chunk_files", []))),
            "source_path": loaded_manifest["source_path"],
            "source_size_bytes": int(loaded_manifest["source_size_bytes"]),
        }
    else:
        records = load_packet_records(args.input)
        total_records = len(records)
        if args.max_records > 0:
            records = records[:args.max_records]
        resolved_input = args.input.resolve()
        time_stream_source = {
            "mode": "raw_npy",
            "cache_dir": None,
            "cache_status": "disabled",
            "chunk_count": 0,
            "source_path": str(resolved_input),
            "source_size_bytes": int(resolved_input.stat().st_size),
        }
    payload = build_feature_sequences(records, bin_width=args.bin_width, stream_granularity=args.stream_granularity)
    sequences = payload["sequences"]
    normalized, mean, std = normalize_sequences(sequences)
    inputs, targets = make_windows(normalized, window_size=args.window_size, stride=args.stride)
    observed_stream_training, metrics, score_arrays, processor_details, spectral_summaries, model_states = train_observed_stream_models(
        normalized,
        payload["metadata"],
        payload.get("stream_info", {}),
        args,
        device,
    )
    phase_shift_analysis = analyze_phase_shift_classes(processor_details, args)
    joint_system_analysis, joint_arrays = analyze_joint_system(sequences, normalized, payload["metadata"], payload["feature_names"], ridge=args.operator_ridge)
    entangled_shared_now_summary, entangled_arrays, shared_now_events = analyze_entangled_shared_now(
        sequences,
        normalized,
        payload["metadata"],
        ridge=args.operator_ridge,
        lattice_size=args.entangled_lattice_size,
        memory_mix=args.shared_now_memory_mix,
        event_z_threshold=args.shared_now_event_z_threshold,
        event_min_separation=args.shared_now_event_min_separation,
        event_top_k=args.shared_now_event_top_k,
        canonical_pairing=args.canonical_pairing,
    )
    now_agent_temporal_radii = [int(value) for value in args.now_agent_temporal_radii]
    now_agent_payload = build_now_stream_agents(sequences, normalized, payload["metadata"], now_agent_temporal_radii)
    agent_training_summary, agent_processor_details = train_now_stream_agent_models(
        now_agent_payload["normalized_sequences"],
        now_agent_payload["metadata"],
        args,
        device,
    )
    now_stream_mesh_summary, now_stream_mesh_arrays, now_stream_mesh_events = analyze_now_stream_mesh(
        sequences,
        normalized,
        payload["metadata"],
        temporal_radii=now_agent_temporal_radii,
        ridge=args.operator_ridge,
        lattice_size=args.entangled_lattice_size,
        memory_mix=args.shared_now_memory_mix,
        event_z_threshold=args.shared_now_event_z_threshold,
        event_min_separation=args.shared_now_event_min_separation,
        event_top_k=args.shared_now_event_top_k,
        canonical_pairing=args.canonical_pairing,
        agent_processor_details=agent_processor_details,
        agent_training_summary=agent_training_summary,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed_model_dir = args.output_dir / "observed_stream_models"
    observed_model_dir.mkdir(parents=True, exist_ok=True)
    observed_model_paths: dict[str, str] = {}
    for label, state_dict in model_states.items():
        model_path = observed_model_dir / f"{sanitize_label_token(label)}.pt"
        torch.save(state_dict, model_path)
        observed_model_paths[label] = str(model_path)
    np.savez_compressed(
        args.output_dir / "time_dilation_transition_scores.npz",
        feature_names=np.array(payload["feature_names"], dtype=object),
        feature_mean=mean,
        feature_std=std,
        **{f"scores_{label}": arr for label, arr in score_arrays.items()},
    )
    (args.output_dir / "phase_shift_events.json").write_text(json.dumps(phase_shift_analysis, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "joint_system_analysis.npz", **joint_arrays)
    (args.output_dir / "shared_now_events.json").write_text(json.dumps(shared_now_events, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "entangled_shared_now_analysis.npz", **entangled_arrays)
    (args.output_dir / "now_stream_mesh_events.json").write_text(json.dumps(now_stream_mesh_events, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "now_stream_mesh_analysis.npz", **now_stream_mesh_arrays)
    processor_summaries = []
    for key, sequence in sequences.items():
        label = stream_label(key)
        key_dict = dict(payload["stream_labels"].get(label, stream_key_dict(key)))
        meta = payload["metadata"][key]
        processor_summaries.append({
            "label": label,
            **key_dict,
            "packet_count": int(len(sequence)),
            "host_time_start": float(meta[0]["host_time_mid"]),
            "host_time_end": float(meta[-1]["host_time_mid"]),
            "training_status": str(observed_stream_training.get(label, {}).get("status", "missing")),
            "model_path": observed_model_paths.get(label),
        })
    trained_streams = [entry for entry in observed_stream_training.values() if entry.get("status") == "ok"]
    final_losses = [float(entry["final_loss"]) for entry in trained_streams]
    prediction_losses = [float(entry["final_prediction_loss"]) for entry in trained_streams]
    regularizer_losses = [float(entry["final_total_regularizer"]) for entry in trained_streams]
    training_status_counts = dict(sorted(Counter(str(entry.get("status", "unknown")) for entry in observed_stream_training.values()).items()))
    notes = [
        "Only one grouped stream was observed in this dataset, so exactly one observed stream-model was trained and the shared observer layer stayed local-only even though the outputs are ready for future multi-stream captures."
        if len(sequences) == 1 else
        "Multiple grouped streams were observed, so one observed stream-model was trained per real stream while peer-relative timing features and the shared observer analyses were activated across streams.",
        "The empirical transition operator is estimated from aligned observed states and compared to its nearest orthogonal operator; no unitary structure is assumed a priori.",
        "The entangled shared-now analysis writes a shared CaMKII memory silhouette over time and projects that geometry into a Markovian latent trajectory so local stream evidence, shared memory, and a moving QPC-now anchor can be compared rather than conflated.",
        "The Hamiltonian q/p analysis compares orthogonal and symplectic reference evolutions on a canonical pairing of the shared-now latent while optional weak oscillator regularizers encourage q/p consistency, spring reciprocity, and bounded energy drift without hard-coding the result.",
        "The now-stream mesh stage remains a secondary temporal-scale probe layered on top of the real observed stream models; it is not the definition of the real stream count.",
    ]
    if time_stream_source["mode"] == "time_stream_cache":
        notes.append(
            "The time-stream service prepares deterministic JSONL cache chunks once, validates them against the source file signature, and then replays ordered packet moments through the same feature and mesh pipeline so bounded runs can reuse the same chronology without reloading the full object-array container."
        )
    elif time_stream_source["mode"] == "stream_dataset":
        notes.append(
            "The input was already captured as a stream-native chunked dataset, so the learner loaded ordered packet records directly from binary replay chunks and fed them into the same moment/mesh path without a separate cache-conversion step."
        )
    summary = {
        "input_path": str(args.input),
        "records_loaded": int(total_records),
        "records_used": int(len(records)),
        "time_stream_source": {
            **time_stream_source,
            "records_used": int(len(records)),
        },
        "stream_granularity": payload["stream_granularity"],
        "stream_count": int(len(sequences)),
        "observed_stream_model_count": int(len(model_states)),
        "gpu_count": int(len({(int(r.get('gpu_id', -1)), str(r.get('gpu_name', 'unknown'))) for r in records})),
        "processor_count": int(len(sequences)),
        "multi_processor_observed": bool(payload["multi_processor_observed"]),
        "multi_gpu_observed": bool(payload["multi_gpu_observed"]),
        "cross_processor_features_active": bool(payload["multi_processor_observed"]),
        "feature_names": payload["feature_names"],
        "bin_width_seconds": float(payload["bin_width"]),
        "window_count": int(inputs.shape[0]),
        "window_size": int(args.window_size),
        "train_sequence_length": int(inputs.shape[1]),
        "device": str(device),
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "mode": "one_model_per_observed_stream",
            "tuning_methodology": "oscillator_regularized" if any(weight > 0.0 for weight in (args.qp_regularization_weight, args.energy_drift_regularization_weight, args.spring_regularization_weight)) else "prediction_only",
            "canonical_pairing": args.canonical_pairing,
            "qp_regularization_weight": float(args.qp_regularization_weight),
            "energy_drift_regularization_weight": float(args.energy_drift_regularization_weight),
            "spring_regularization_weight": float(args.spring_regularization_weight),
            "aggregate_window_count": int(inputs.shape[0]),
            "trained_stream_count": int(len(trained_streams)),
            "stream_status_counts": training_status_counts,
            "mean_final_loss": float(np.mean(final_losses)) if final_losses else None,
            "mean_final_prediction_loss": float(np.mean(prediction_losses)) if prediction_losses else None,
            "mean_final_total_regularizer": float(np.mean(regularizer_losses)) if regularizer_losses else None,
        },
        "observed_stream_models": observed_stream_training,
        "observed_stream_model_dir": str(observed_model_dir),
        "observed_stream_spectral_summaries": spectral_summaries,
        "observer_layer": {
            "moving_qpc_anchor": "host_time_mid",
            "shared_memory_analysis": "entangled_shared_now",
            "joint_observer_analysis": "joint_system_analysis",
            "temporal_probe_analysis": "now_stream_mesh_analysis",
        },
        "processors": processor_summaries,
        "streams": processor_summaries,
        "per_processor_metrics": metrics,
        "phase_shift_event_path": str(args.output_dir / "phase_shift_events.json"),
        "joint_system_analysis_path": str(args.output_dir / "joint_system_analysis.npz"),
        "shared_now_event_path": str(args.output_dir / "shared_now_events.json"),
        "entangled_shared_now_analysis_path": str(args.output_dir / "entangled_shared_now_analysis.npz"),
        "now_stream_mesh_event_path": str(args.output_dir / "now_stream_mesh_events.json"),
        "now_stream_mesh_analysis_path": str(args.output_dir / "now_stream_mesh_analysis.npz"),
        "phase_shift_analysis": {
            "event_count": int(len(phase_shift_analysis["events"])),
            "class_count": int(phase_shift_analysis["class_count"]),
            "repeated_class_count": int(phase_shift_analysis["repeated_class_count"]),
            "one_off_class_count": int(phase_shift_analysis["one_off_class_count"]),
            "repeated_event_count": int(phase_shift_analysis["repeated_event_count"]),
            "processor_event_counts": phase_shift_analysis["processor_event_counts"],
            "top_classes": phase_shift_analysis["classes"][:5],
        },
        "joint_system_analysis": joint_system_analysis,
        "entangled_shared_now_analysis": {
            **entangled_shared_now_summary,
            "top_events": shared_now_events["events"][:5],
        },
        "now_stream_mesh_analysis": {
            **now_stream_mesh_summary,
            "top_events": now_stream_mesh_events["events"][:5],
        },
        "notes": notes,
    }
    (args.output_dir / "time_dilation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary_path": str(args.output_dir / "time_dilation_summary.json"),
        "processor_count": summary["processor_count"],
        "observed_stream_model_count": summary["observed_stream_model_count"],
        "multi_processor_observed": summary["multi_processor_observed"],
        "window_count": summary["window_count"],
        "time_stream_mode": summary["time_stream_source"]["mode"],
        "phase_shift_event_count": summary["phase_shift_analysis"]["event_count"],
        "repeated_phase_classes": summary["phase_shift_analysis"]["repeated_class_count"],
        "now_stream_agent_count": summary["now_stream_mesh_analysis"]["agent_count"],
        "mesh_recurrence_candidates": summary["now_stream_mesh_analysis"]["recurrence_candidate_count"],
        "mean_observed_stream_final_loss": summary["training"]["mean_final_loss"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())