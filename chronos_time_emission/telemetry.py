from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import jax
import numpy as np


def make_balanced_labels(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray([0, 1] * (n_trials // 2), dtype=np.int32)
    if n_trials % 2:
        labels = np.append(labels, np.int32(rng.integers(0, 2)))
    rng.shuffle(labels)
    return labels.astype(np.int32)


def run_trial(alice_fn: Callable, bob_fn: Callable, alice_input, bob_input, barrier=None) -> tuple[int, int, int]:
    # Chronos cares about condition-separated work profiles rather than strict
    # simultaneous host-side execution, so run Alice then Bob sequentially.
    # This keeps one pmap at a time on the TPU backend and avoids replica-group
    # collisions when probing the two roles repeatedly.
    _ = alice_fn(alice_input)
    jax.block_until_ready(_)
    start_ns = time.perf_counter_ns()
    out = bob_fn(bob_input)
    jax.block_until_ready(out)
    end_ns = time.perf_counter_ns()
    return int(end_ns - start_ns), int(start_ns), int(end_ns)


def warmup(workloads: dict, bob_probe, alice_input, bob_input, iterations: int = 8):
    warmup_names = ["dense_full", "block_sparse", "equal_flop_tiled", "equal_flop_permuted", "noop"]
    for _ in range(iterations):
        for name in warmup_names:
            run_trial(workloads[name], bob_probe, alice_input, bob_input)


def run_persistent_trials(
    schedule: list[dict],
    workloads: dict,
    bob_probe: Callable,
    alice_input,
    bob_input,
    progress_every: int = 100,
) -> list[dict]:
    jobs: queue.Queue = queue.Queue()
    barrier = threading.Barrier(2)
    stop = object()

    def alice_worker():
        while True:
            job = jobs.get()
            if job is stop:
                break
            barrier.wait()
            out = workloads[job["actual_workload"]](alice_input)
            jax.block_until_ready(out)
            job["done"].set()

    worker = threading.Thread(target=alice_worker, daemon=True)
    worker.start()
    records: list[dict] = []
    try:
        for trial_global, job in enumerate(schedule):
            done = threading.Event()
            job = {**job, "done": done}
            jobs.put(job)
            barrier.wait()
            start_ns = time.perf_counter_ns()
            out = bob_probe(bob_input)
            jax.block_until_ready(out)
            end_ns = time.perf_counter_ns()
            done.wait()
            record = {
                **{k: v for k, v in job.items() if k != "done"},
                "trial_global": int(trial_global),
                "latency_ns": int(end_ns - start_ns),
                "timestamp_start_ns": int(start_ns),
                "timestamp_end_ns": int(end_ns),
            }
            records.append(record)
            if trial_global % progress_every == 0:
                print(
                    f"  trial {trial_global:5d} condition={record['condition']:>10} "
                    f"label={record['label_name']:>20} lat={record['latency_ns']:9d} ns"
                )
    finally:
        jobs.put(stop)
        worker.join(timeout=10)
    return records
