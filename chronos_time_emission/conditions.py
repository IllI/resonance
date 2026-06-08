from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

from .classifiers import block_feature_table, evaluate_block_classifiers
from .spectral import spectral_summary
from .telemetry import make_balanced_labels, run_persistent_trials, run_trial


def condition_spec(name: str) -> dict:
    specs = {
        "active": {
            "display": "Active randomized dense/sparse",
            "label_names": ["dense_full", "block_sparse"],
            "workloads": ["dense_full", "block_sparse"],
        },
        "severed": {
            "display": "Control severed Alice no-op",
            "label_names": ["dense_full", "block_sparse"],
            "workloads": ["noop", "noop"],
        },
        "constant": {
            "display": "Control constant dense Alice",
            "label_names": ["dense_full", "block_sparse"],
            "workloads": ["dense_full", "dense_full"],
        },
        "shuffled": {
            "display": "Control active with shuffled labels",
            "label_names": ["dense_full", "block_sparse"],
            "workloads": ["dense_full", "block_sparse"],
            "shuffle_labels": True,
        },
        "equal_flop": {
            "display": "Equal-FLOP topology control",
            "label_names": ["equal_flop_tiled", "equal_flop_permuted"],
            "workloads": ["equal_flop_tiled", "equal_flop_permuted"],
        },
    }
    if name not in specs:
        raise ValueError(f"Unknown condition: {name}")
    return specs[name]


def run_condition(
    name: str,
    workloads: dict,
    bob_probe,
    alice_input,
    bob_input,
    blocks: int,
    block_size: int,
    rng: np.random.Generator,
    progress_every: int = 100,
) -> Dict:
    spec = condition_spec(name)
    records: List[dict] = []
    trial_global = 0
    import threading

    barrier = threading.Barrier(2)
    print(f"\n-- {spec['display']} --")
    for block in range(blocks):
        labels = make_balanced_labels(block_size, rng)
        if spec.get("shuffle_labels", False):
            observed_labels = rng.permutation(labels)
        else:
            observed_labels = labels.copy()
        for trial_in_block, actual_label in enumerate(labels):
            workload_name = spec["workloads"][int(actual_label)]
            latency_ns, start_ns, end_ns = run_trial(
                workloads[workload_name], bob_probe, alice_input, bob_input, barrier
            )
            label = int(observed_labels[trial_in_block])
            records.append(
                {
                    "condition": name,
                    "block": int(block),
                    "trial": int(trial_in_block),
                    "trial_global": int(trial_global),
                    "label": label,
                    "actual_label": int(actual_label),
                    "label_name": spec["label_names"][label],
                    "actual_workload": workload_name,
                    "latency_ns": int(latency_ns),
                    "timestamp_start_ns": int(start_ns),
                    "timestamp_end_ns": int(end_ns),
                }
            )
            if trial_global % progress_every == 0:
                print(
                    f"  trial {trial_global:5d} block={block:02d} "
                    f"label={spec['label_names'][label]:>20} lat={latency_ns:9d} ns"
                )
            trial_global += 1
    return analyze_condition(name, spec, records, int(rng.integers(0, 1_000_000_000)))


def run_chronos0b_interleaved(
    workloads: dict,
    bob_probe,
    alice_input,
    bob_input,
    trials_per_condition: int,
    block_size: int,
    rng: np.random.Generator,
    discard: int = 50,
    progress_every: int = 100,
) -> Dict[str, Dict]:
    condition_names = ["active", "severed", "shuffled", "equal_flop"]
    schedule: List[dict] = []
    for condition in condition_names:
        spec = condition_spec(condition)
        actual_labels = make_balanced_labels(trials_per_condition + discard, rng)
        if spec.get("shuffle_labels", False):
            observed_labels = rng.permutation(actual_labels)
        else:
            observed_labels = actual_labels.copy()
        for trial_local, actual_label in enumerate(actual_labels):
            label = int(observed_labels[trial_local])
            actual_label = int(actual_label)
            schedule.append(
                {
                    "condition": condition,
                    "block": int(trial_local // max(1, block_size)),
                    "trial": int(trial_local),
                    "label": label,
                    "actual_label": actual_label,
                    "label_name": spec["label_names"][label],
                    "actual_workload": spec["workloads"][actual_label],
                    "discard": bool(trial_local < discard),
                }
            )
    rng.shuffle(schedule)
    print("\n-- CHRONOS-0b persistent interleaved correction --")
    print(
        f"  conditions={','.join(condition_names)} "
        f"trials_per_condition={trials_per_condition} discard_per_condition={discard}"
    )
    records = run_persistent_trials(
        schedule,
        workloads,
        bob_probe,
        alice_input,
        bob_input,
        progress_every=progress_every,
    )
    kept = [row for row in records if not row.get("discard", False)]
    results: Dict[str, Dict] = {}
    for condition in condition_names:
        spec = condition_spec(condition)
        condition_records = [{k: v for k, v in row.items() if k != "discard"} for row in kept if row["condition"] == condition]
        results[condition] = analyze_condition(condition, spec, condition_records, int(rng.integers(0, 1_000_000_000)))
    return results


def analyze_condition(name: str, spec: dict, records: List[dict], seed: int = 0) -> Dict:
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int32)
    latencies = np.asarray([float(row["latency_ns"]) for row in records], dtype=np.float64)
    dense = latencies[labels == 0]
    sparse = latencies[labels == 1]
    if dense.size > 1 and sparse.size > 1:
        u_stat, mw_p = mannwhitneyu(dense, sparse, alternative="two-sided")
        ks_stat, ks_p = ks_2samp(dense, sparse)
    else:
        u_stat = mw_p = ks_stat = ks_p = float("nan")

    spec_dense = spectral_summary(dense)
    spec_sparse = spectral_summary(sparse)
    features = block_feature_table(records)
    classifier_metrics = evaluate_block_classifiers(features, seed=seed)
    result = {
        "condition": name,
        "display": spec["display"],
        "n_trials": int(latencies.size),
        "label_0": spec["label_names"][0],
        "label_1": spec["label_names"][1],
        "mean_label0_ns": float(np.mean(dense)) if dense.size else 0.0,
        "mean_label1_ns": float(np.mean(sparse)) if sparse.size else 0.0,
        "std_label0_ns": float(np.std(dense)) if dense.size else 0.0,
        "std_label1_ns": float(np.std(sparse)) if sparse.size else 0.0,
        "delta_mean_ns": float(np.mean(dense) - np.mean(sparse)) if dense.size and sparse.size else 0.0,
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(mw_p),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "df_fund_label0": float(spec_dense["df_fund"]),
        "df_fund_label1": float(spec_sparse["df_fund"]),
        "sideband_asym_label0": float(spec_dense["sideband_asym"]),
        "sideband_asym_label1": float(spec_sparse["sideband_asym"]),
        "okl_mean_label0": float(spec_dense["okl_mean"]),
        "okl_mean_label1": float(spec_sparse["okl_mean"]),
        **classifier_metrics,
        "records": records,
        "spectral_features": features,
    }
    print(
        f"  delta={result['delta_mean_ns']:+.0f} ns "
        f"MW_p={result['mann_whitney_p']:.4g} KS_p={result['ks_p']:.4g} "
        f"mean_acc={result['mean_acc']:.3f} spectral_acc={result['sideband_acc']:.3f} "
        f"combined_acc={result['combined_acc']:.3f}"
    )
    return result


def write_raw_csv(path: Path, results: Dict[str, Dict]):
    rows = []
    for result in results.values():
        rows.extend(result.get("records", []))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_spectral_csv(path: Path, results: Dict[str, Dict]):
    rows = []
    for result in results.values():
        for row in result.get("spectral_features", []):
            rows.append({"condition": result["condition"], **row})
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
