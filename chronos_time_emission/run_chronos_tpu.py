#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from chronos_time_emission.conditions import run_chronos0b_interleaved, run_condition, write_raw_csv, write_spectral_csv
from chronos_time_emission.telemetry import warmup
from chronos_time_emission.workloads import build_inputs, build_workloads, setup_devices


def parse_args():
    parser = argparse.ArgumentParser(description="CHRONOS time-emission TPU workload fingerprinting")
    parser.add_argument("--mode", choices=["chronos0", "chronos0b"], default="chronos0")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--trials-per-condition", type=int, default=400)
    parser.add_argument("--discard-trials-per-condition", type=int, default=50)
    parser.add_argument("--trials", type=int, default=None, help="Overrides block-size as total trials split across blocks.")
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--sparse-block", type=int, default=32)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument(
        "--condition",
        default="all",
        choices=["all", "active", "severed", "constant", "shuffled", "equal_flop"],
    )
    parser.add_argument("--out", default="chronos_time_emission/results/chronos0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--require-tpu", action="store_true")
    return parser.parse_args()


def requested_conditions(condition: str) -> list[str]:
    if condition == "all":
        return ["active", "severed", "constant", "shuffled", "equal_flop"]
    return [condition]


def write_summary(path: Path, args, backend: str, n_devices: int, results: dict):
    lines = [
        "# CHRONOS Time-Emission Summary",
        "",
        f"- backend: `{backend}`",
        f"- devices: `{n_devices}`",
        f"- D: `{args.d}`",
        f"- sparse_block: `{args.sparse_block}`",
        f"- layers: `{args.layers}`",
        f"- blocks: `{args.blocks}`",
        f"- block_size: `{args.block_size}`",
        "",
        "| condition | delta_mean_ns | MW_p | KS_p | mean_acc | sideband_acc | combined_acc | shuffled_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            "| {name} | {delta:+.0f} | {mw:.4g} | {ks:.4g} | {mean:.3f} | {side:.3f} | {comb:.3f} | {shuf:.3f} |".format(
                name=name,
                delta=result.get("delta_mean_ns", 0.0),
                mw=result.get("mann_whitney_p", float("nan")),
                ks=result.get("ks_p", float("nan")),
                mean=result.get("mean_acc", 0.5),
                side=result.get("sideband_acc", 0.5),
                comb=result.get("combined_acc", 0.5),
                shuf=result.get("shuffled_acc", 0.5),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation gates:",
            "",
            "- Physical coupling: active significant while severed and shuffled are null.",
            "- Beyond mean latency: sideband or combined classifier improves over mean latency.",
            "- Topology specificity: equal-FLOP topology control separates with matched or near-matched mean latency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in {"records", "spectral_features"}}


def main():
    args = parse_args()
    if args.trials is not None:
        args.block_size = max(1, int(np.ceil(args.trials / args.blocks)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend, devices, alice_devs, bob_devs = setup_devices()
    if args.require_tpu and backend != "tpu":
        raise RuntimeError(f"--require-tpu was set, but backend is {backend}")

    print("=" * 80)
    print("CHRONOS-0/1 TPU workload-state temporal emission")
    print("=" * 80)
    print(f"backend={backend} devices={len(devices)} alice={len(alice_devs)} bob={len(bob_devs)}")
    print(f"mode={args.mode} D={args.d} sparse_block={args.sparse_block} layers={args.layers}")
    print(f"blocks={args.blocks} block_size={args.block_size} seed={args.seed}")

    workloads, bob_probe = build_workloads(alice_devs, bob_devs, args.d, args.sparse_block, args.layers)
    alice_input, bob_input = build_inputs(alice_devs, bob_devs, args.d, args.seed)

    print("\nWarming up JAX/XLA...")
    warmup(workloads, bob_probe, alice_input, bob_input, iterations=args.warmup_iters)
    print("Warm-up complete.")

    rng = np.random.default_rng(args.seed)
    if args.mode == "chronos0b":
        results = run_chronos0b_interleaved(
            workloads,
            bob_probe,
            alice_input,
            bob_input,
            args.trials_per_condition,
            args.block_size,
            rng,
            discard=args.discard_trials_per_condition,
        )
    else:
        results = {}
        for condition in requested_conditions(args.condition):
            results[condition] = run_condition(
                condition,
                workloads,
                bob_probe,
                alice_input,
                bob_input,
                args.blocks,
                args.block_size,
                rng,
            )

    raw_csv = out_dir / "latencies.csv"
    spectral_csv = out_dir / "spectral_features.csv"
    results_json = out_dir / "results.json"
    summary_md = out_dir / "summary.md"

    write_raw_csv(raw_csv, results)
    write_spectral_csv(spectral_csv, results)
    payload = {
        "args": vars(args),
        "backend": backend,
        "n_devices": len(devices),
        "device_ids": [str(device) for device in devices],
        "pid": os.getpid(),
        "results": {name: compact_result(result) for name, result in results.items()},
    }
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_md, args, backend, len(devices), results)

    print("\n" + "=" * 80)
    print("CHRONOS SUMMARY")
    print("=" * 80)
    for name, result in results.items():
        print(
            f"{name:12s} delta={result['delta_mean_ns']:+.0f} ns "
            f"MW_p={result['mann_whitney_p']:.4g} KS_p={result['ks_p']:.4g} "
            f"mean_acc={result['mean_acc']:.3f} sideband_acc={result['sideband_acc']:.3f} "
            f"combined_acc={result['combined_acc']:.3f}"
        )
    print(f"\nWrote {results_json}")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {spectral_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
