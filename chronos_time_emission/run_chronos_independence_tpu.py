#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


SCHUMANN_HZ = np.asarray([7.83, 14.3, 20.8, 27.3, 33.8], dtype=np.float64)


@jax.jit
def probe_kernel(x, phase):
    y = x
    for _ in range(4):
        y = jnp.tanh(y @ y.T + phase)
        y = y / (jnp.linalg.norm(y) + jnp.float32(1e-6))
    return jnp.sum(jnp.sin(y + phase))


def robust_norm(x, clip=5.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.clip((x - med) / mad, -clip, clip)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 4:
        return 0.0
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def fft_band_filter(x, fs_hz, lo_hz=None, hi_hz=None, mode="pass"):
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs_hz)
    spec = np.fft.rfft(centered)
    if mode == "full":
        mask = np.ones_like(freqs, dtype=bool)
        mask[0] = False
    else:
        lo = float(lo_hz)
        hi = min(float(hi_hz), fs_hz / 2.0 - 1e-6)
        if hi <= lo:
            return np.zeros_like(centered)
        mask = (freqs >= lo) & (freqs <= hi)
        if mode == "stop":
            mask = ~mask
            mask[0] = False
    return np.fft.irfft(spec * mask, n=centered.size)


def lag_corr(a, b, max_lag=8):
    best = {"lag": 0, "r": pearson(a, b)}
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag == 0:
            continue
        if lag > 0:
            aa, bb = a[lag:], b[:-lag]
        else:
            aa, bb = a[:lag], b[-lag:]
        r = pearson(aa, bb)
        if abs(r) > abs(best["r"]):
            best = {"lag": int(lag), "r": float(r)}
    return best


def diagnose_scenario(r_full):
    if r_full >= 0.95:
        return "same_host_same_timer"
    if r_full >= 0.70:
        return "marginal_high_common_mode"
    if r_full >= 0.20:
        return "shared_infrastructure_possible"
    return "independent_or_weakly_coupled"


def gate_minus_one(r_full):
    passed = bool(r_full < 0.90)
    return {
        "passed": passed,
        "r_full": float(r_full),
        "scenario": diagnose_scenario(r_full),
        "verdict": "PASS" if passed else "FAIL_COMMON_MODE",
    }


def wait_for_start(start_epoch):
    if start_epoch is None:
        return
    while True:
        remaining = float(start_epoch) - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def collect_q(args):
    backend = jax.default_backend()
    if args.require_tpu and backend != "tpu":
        raise RuntimeError(f"--require-tpu was set, but backend is {backend!r}")

    out = Path(args.out or "chronos_time_emission/results/schumann0b")
    out.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(int(args.seed))
    x = jax.random.normal(key, (int(args.size), int(args.size)), dtype=jnp.float32)
    for idx in range(int(args.warmup)):
        probe_kernel(x, jnp.float32(idx * 0.001)).block_until_ready()
        x = jnp.roll(x, 1, axis=0)

    print("CHRONOS-SCHUMANN-0b collector")
    print(f"label={args.node_label} backend={backend} devices={len(jax.devices())}")
    if args.start_epoch is not None:
        print(f"waiting_for_start_epoch={args.start_epoch:.3f}")
    wait_for_start(args.start_epoch)

    samples = []
    stamps = []
    period = 1.0 / float(args.sample_hz)
    next_t = time.perf_counter()
    started_epoch = time.time()
    for idx in range(int(args.samples)):
        phase = jnp.float32((idx + 1) * 0.0031)
        t0 = time.perf_counter_ns()
        probe_kernel(x, phase).block_until_ready()
        t1 = time.perf_counter_ns()
        stamps.append(t0)
        samples.append(t1 - t0)
        x = jnp.roll(x, 1, axis=1)
        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        if (idx + 1) % max(1, int(args.samples) // 4) == 0:
            print(f"progress {idx + 1}/{args.samples}")

    stamps = np.asarray(stamps, dtype=np.int64)
    latency = np.asarray(samples, dtype=np.float64)
    duration = max(1e-9, float(stamps[-1] - stamps[0]) / 1e9) if stamps.size > 1 else 1.0
    fs_actual = float((stamps.size - 1) / duration) if stamps.size > 1 else float(args.sample_hz)
    q = robust_norm(latency)

    payload = {
        "program": "CHRONOS-SCHUMANN-0b-collect",
        "node_label": args.node_label,
        "backend": backend,
        "n_devices": len(jax.devices()),
        "seed": int(args.seed),
        "samples": int(args.samples),
        "sample_hz_requested": float(args.sample_hz),
        "fs_actual_hz": float(fs_actual),
        "size": int(args.size),
        "started_epoch": float(started_epoch),
        "ended_epoch": float(time.time()),
        "latency_mean_ns": float(np.mean(latency)),
        "latency_std_ns": float(np.std(latency)),
        "q_latency": [round(float(v), 6) for v in q],
        "pid": os.getpid(),
    }
    q_path = out / f"{args.node_label}_q.json"
    q_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    summary = [
        "# CHRONOS-SCHUMANN-0b Collector",
        "",
        f"- node_label: `{args.node_label}`",
        f"- backend: `{backend}`",
        f"- devices: `{len(jax.devices())}`",
        f"- samples: `{args.samples}`",
        f"- fs_actual_hz: `{fs_actual:.3f}`",
        f"- latency_mean_ns: `{np.mean(latency):.3f}`",
        f"- latency_std_ns: `{np.std(latency):.3f}`",
        f"- q_file: `{q_path}`",
    ]
    (out / f"{args.node_label}_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"collector_complete q_file={q_path}")


def load_q(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, np.asarray(payload["q_latency"], dtype=np.float64)


def evaluate_pair(alice, bob, fs_hz, max_lag):
    modes = {
        "full": (fft_band_filter(alice, fs_hz, mode="full"), fft_band_filter(bob, fs_hz, mode="full")),
        "schumann": (fft_band_filter(alice, fs_hz, 6.0, 40.0, "pass"), fft_band_filter(bob, fs_hz, 6.0, 40.0, "pass")),
        "anti": (fft_band_filter(alice, fs_hz, 6.0, 40.0, "stop"), fft_band_filter(bob, fs_hz, 6.0, 40.0, "stop")),
        "grid": (fft_band_filter(alice, fs_hz, 58.0, 62.0, "pass"), fft_band_filter(bob, fs_hz, 58.0, 62.0, "pass")),
    }
    results = {}
    for name, (a_sig, b_sig) in modes.items():
        lag = lag_corr(a_sig, b_sig, max_lag=max_lag)
        results[name] = {
            "r_zero": pearson(a_sig, b_sig),
            "r_best": lag["r"],
            "best_lag": lag["lag"],
            "rms_alice": float(np.sqrt(np.mean(a_sig * a_sig))),
            "rms_bob": float(np.sqrt(np.mean(b_sig * b_sig))),
        }
    gate = gate_minus_one(results["full"]["r_zero"])
    results["gate_minus_one"] = gate
    results["schumann_minus_anti"] = float(results["schumann"]["r_zero"] - results["anti"]["r_zero"])
    results["gate_one_passed"] = bool(gate["passed"] and results["schumann_minus_anti"] > 0.10)
    return results


def analyze_pair(args):
    alice_meta, alice_q = load_q(args.alice_q)
    bob_meta, bob_q = load_q(args.bob_q)
    n = min(alice_q.size, bob_q.size)
    fs = min(float(alice_meta["fs_actual_hz"]), float(bob_meta["fs_actual_hz"]))
    results = evaluate_pair(alice_q[:n], bob_q[:n], fs, args.max_lag)
    payload = {
        "program": "CHRONOS-SCHUMANN-0b-analyze",
        "alice": {k: alice_meta[k] for k in ("node_label", "fs_actual_hz", "samples", "started_epoch", "ended_epoch")},
        "bob": {k: bob_meta[k] for k in ("node_label", "fs_actual_hz", "samples", "started_epoch", "ended_epoch")},
        "paired_samples": int(n),
        "fs_hz_used": float(fs),
        "modes": results,
    }
    out = Path(args.out or "chronos_time_emission/results/schumann0b_pair")
    out.mkdir(parents=True, exist_ok=True)
    (out / "independence_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# CHRONOS-SCHUMANN-0b Independence Summary",
        "",
        f"- gate_minus_one: `{results['gate_minus_one']['verdict']}`",
        f"- scenario: `{results['gate_minus_one']['scenario']}`",
        f"- schumann_minus_anti: `{results['schumann_minus_anti']:.4f}`",
        f"- gate_one_passed: `{results['gate_one_passed']}`",
        "",
        "| mode | r_zero | r_best | best_lag |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("full", "schumann", "anti", "grid"):
        row = results[name]
        lines.append(f"| {name} | {row['r_zero']:.4f} | {row['r_best']:.4f} | {row['best_lag']} |")
    (out / "independence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="CHRONOS-SCHUMANN-0b independence collector/analyzer")
    parser.add_argument("--out", default=None)
    parser.add_argument("--require-tpu", action="store_true")
    sub = parser.add_subparsers(dest="mode", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--node-label", required=True)
    collect.add_argument("--seed", type=int, default=11)
    collect.add_argument("--samples", type=int, default=32768)
    collect.add_argument("--sample-hz", type=float, default=128.0)
    collect.add_argument("--size", type=int, default=96)
    collect.add_argument("--warmup", type=int, default=12)
    collect.add_argument("--start-epoch", type=float, default=None)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--alice-q", required=True)
    analyze.add_argument("--bob-q", required=True)
    analyze.add_argument("--max-lag", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "collect":
        collect_q(args)
    elif args.mode == "analyze":
        analyze_pair(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
