from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


STREAM_DATASET_SCHEMA_VERSION = 1
STREAM_DATASET_TYPE = "gpu_clock_time_stream_dataset"


def time_stream_dataset_manifest_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "manifest.json"


def load_time_stream_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads(time_stream_dataset_manifest_path(dataset_dir).read_text(encoding="utf-8"))


def validate_time_stream_dataset_manifest(manifest: dict[str, Any], dataset_dir: Path) -> None:
    dataset_dir = Path(dataset_dir)
    if int(manifest.get("schema_version", -1)) != STREAM_DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported time-stream dataset schema version.")
    if manifest.get("dataset_type") != STREAM_DATASET_TYPE:
        raise ValueError("Unsupported time-stream dataset type.")
    missing_chunks = [name for name in manifest.get("chunk_files", []) if not (dataset_dir / str(name)).exists()]
    if missing_chunks:
        raise ValueError(f"Time-stream dataset is missing chunk files: {missing_chunks[:3]}")


def is_time_stream_dataset_path(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    manifest_path = time_stream_dataset_manifest_path(path)
    if not manifest_path.exists():
        return False
    try:
        manifest = load_time_stream_dataset_manifest(path)
    except Exception:
        return False
    return manifest.get("dataset_type") == STREAM_DATASET_TYPE


def iter_time_stream_dataset_records(dataset_dir: Path, max_records: int = 0):
    dataset_dir = Path(dataset_dir)
    manifest = load_time_stream_dataset_manifest(dataset_dir)
    validate_time_stream_dataset_manifest(manifest, dataset_dir)
    target_count = max(int(max_records), 0)
    emitted = 0
    for chunk_name in manifest.get("chunk_files", []):
        chunk_path = dataset_dir / str(chunk_name)
        with np.load(chunk_path, allow_pickle=False) as chunk:
            gpu_id = np.asarray(chunk["gpu_id"], dtype=np.int32)
            gpu_name = np.asarray(chunk["gpu_name"]).astype(str)
            pid = np.asarray(chunk["pid"], dtype=np.int64)
            packet_index = np.asarray(chunk["packet_index"], dtype=np.int64)
            host_time_start = np.asarray(chunk["host_time_start"], dtype=np.float64)
            host_time_end = np.asarray(chunk["host_time_end"], dtype=np.float64)
            host_time_mid = np.asarray(chunk["host_time_mid"], dtype=np.float64)
            elapsed_host_seconds = np.asarray(chunk["elapsed_host_seconds"], dtype=np.float64)
            threads_per_block = np.asarray(chunk["threads_per_block"], dtype=np.int32)
            blocks = np.asarray(chunk["blocks"], dtype=np.int32)
            samples_per_thread = np.asarray(chunk["samples_per_thread"], dtype=np.int32)
            total_threads = np.asarray(chunk["total_threads"], dtype=np.int32)
            buffer_size = np.asarray(chunk["buffer_size"], dtype=np.int32)
            sample_count = np.asarray(chunk["sample_count"], dtype=np.int32)
            offsets = np.asarray(chunk["clock_data_offset"], dtype=np.int64)
            flat_clock = np.asarray(chunk["clock_data_flat"], dtype=np.uint64)
            packet_count = max(len(offsets) - 1, 0)
            if "stream_id" in chunk.files:
                stream_id = np.asarray(chunk["stream_id"]).astype(str)
            else:
                stream_id = np.asarray(
                    [f"gpu{int(gpu_id[index])}_pid{int(pid[index])}" for index in range(packet_count)]
                ).astype(str)
            device_worker_index = (
                np.asarray(chunk["device_worker_index"], dtype=np.int32)
                if "device_worker_index" in chunk.files
                else np.full(packet_count, -1, dtype=np.int32)
            )
            stream_global_index = (
                np.asarray(chunk["stream_global_index"], dtype=np.int32)
                if "stream_global_index" in chunk.files
                else np.full(packet_count, -1, dtype=np.int32)
            )
            workers_per_gpu = (
                np.asarray(chunk["workers_per_gpu"], dtype=np.int32)
                if "workers_per_gpu" in chunk.files
                else np.full(packet_count, int(manifest.get("workers_per_gpu", 1)), dtype=np.int32)
            )
            for index in range(packet_count):
                start = int(offsets[index])
                end = int(offsets[index + 1])
                yield {
                    "schema_version": STREAM_DATASET_SCHEMA_VERSION,
                    "record_type": "gpu_clock_packet",
                    "gpu_id": int(gpu_id[index]),
                    "gpu_name": str(gpu_name[index]),
                    "pid": int(pid[index]),
                    "stream_id": str(stream_id[index]),
                    "device_worker_index": int(device_worker_index[index]),
                    "stream_global_index": int(stream_global_index[index]),
                    "workers_per_gpu": max(int(workers_per_gpu[index]), 1),
                    "packet_index": int(packet_index[index]),
                    "host_time_start": float(host_time_start[index]),
                    "host_time_end": float(host_time_end[index]),
                    "host_time_mid": float(host_time_mid[index]),
                    "elapsed_host_seconds": float(elapsed_host_seconds[index]),
                    "threads_per_block": int(threads_per_block[index]),
                    "blocks": int(blocks[index]),
                    "samples_per_thread": int(samples_per_thread[index]),
                    "total_threads": int(total_threads[index]),
                    "buffer_size": int(buffer_size[index]),
                    "sample_count": int(sample_count[index]),
                    "clock_data": flat_clock[start:end].copy(),
                }
                emitted += 1
                if target_count > 0 and emitted >= target_count:
                    return


def load_time_stream_dataset_records(dataset_dir: Path, max_records: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    manifest = load_time_stream_dataset_manifest(dataset_dir)
    validate_time_stream_dataset_manifest(manifest, dataset_dir)
    return list(iter_time_stream_dataset_records(dataset_dir, max_records=max_records)), manifest