#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BANDS = {
    "full": ("full", None, None),
    "low_drift": ("pass", 0.1, 6.0),
    "schumann": ("pass", 6.0, 40.0),
    "anti": ("stop", 6.0, 40.0),
    "grid": ("pass", 58.0, 62.0),
    "high_residual": ("pass", 40.0, 63.0),
}


def load_q(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    q = np.asarray(payload["q_latency"], dtype=np.float64)
    fs = float(payload.get("fs_actual_hz", payload.get("sample_hz_requested", 1.0)))
    return payload, q, fs


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 4:
        return 0.0
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def robust_norm(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.clip((x - med) / mad, -clip, clip)


def fft_band_filter(x: np.ndarray, fs_hz: float, mode: str = "pass", lo_hz=None, hi_hz=None) -> np.ndarray:
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


def evaluate_pair(a_q: np.ndarray, b_q: np.ndarray, fs_hz: float):
    a_q = robust_norm(a_q)
    b_q = robust_norm(b_q)
    out = {}
    for name, (mode, lo, hi) in BANDS.items():
        a_band = fft_band_filter(a_q, fs_hz, mode=mode, lo_hz=lo, hi_hz=hi)
        b_band = fft_band_filter(b_q, fs_hz, mode=mode, lo_hz=lo, hi_hz=hi)
        out[name] = float(pearson(a_band, b_band))
    out["schumann_minus_anti"] = float(out["schumann"] - out["anti"])
    return out


def classify_same_vs_shifted(same: dict, shifted: dict):
    same_full = same["full"]
    shifted_full = shifted["full"]
    drop = same_full - shifted_full
    if abs(shifted_full - same_full) <= 0.10:
        return "hardware_structural_or_extractor_similarity"
    if abs(shifted_full) <= 0.15 and drop > 0.25:
        return "time_window_dependent_coupling"
    return "mixed_band_specific_drift"


def run(args):
    alice_meta, alice_q, alice_fs = load_q(args.alice)
    same_meta, same_q, same_fs = load_q(args.same_bob)
    shifted_meta, shifted_q, shifted_fs = load_q(args.shifted_bob)
    fs = min(alice_fs, same_fs, shifted_fs)
    n_same = min(alice_q.size, same_q.size)
    n_shifted = min(alice_q.size, shifted_q.size)

    same = evaluate_pair(alice_q[:n_same], same_q[:n_same], fs)
    shifted = evaluate_pair(alice_q[:n_shifted], shifted_q[:n_shifted], fs)
    deltas = {name: float(same[name] - shifted[name]) for name in BANDS}
    verdict = classify_same_vs_shifted(same, shifted)

    result = {
        "program": "CHRONOS-MARGINAL-DRIFT-1",
        "alice": {k: alice_meta.get(k) for k in ("node_label", "seed", "fs_actual_hz", "started_epoch", "ended_epoch")},
        "same_bob": {k: same_meta.get(k) for k in ("node_label", "seed", "fs_actual_hz", "started_epoch", "ended_epoch")},
        "shifted_bob": {k: shifted_meta.get(k) for k in ("node_label", "seed", "fs_actual_hz", "started_epoch", "ended_epoch")},
        "same_window": same,
        "shifted_window": shifted,
        "same_minus_shifted": deltas,
        "verdict": verdict,
        "policy": "correlation is diagnostic; heldout transition gain remains the evidence gate",
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chronos_marginal_drift1_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# CHRONOS-MARGINAL-DRIFT-1 Summary",
        "",
        f"- verdict: `{verdict}`",
        f"- r_full_same_window: `{same['full']:.4f}`",
        f"- r_full_shifted_window: `{shifted['full']:.4f}`",
        f"- delta_full: `{deltas['full']:.4f}`",
        "",
        "| band | same_window | shifted_window | same_minus_shifted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in BANDS:
        lines.append(f"| {name} | {same[name]:.4f} | {shifted[name]:.4f} | {deltas[name]:.4f} |")
    lines.extend(
        [
            "",
            "Correlation remains diagnostic only. The D-LinOSS heldout transition endpoint is still the truth gate.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (out_dir / "chronos_marginal_drift1_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser(description="CHRONOS-MARGINAL-DRIFT-1 asynchronous diagnostic")
    parser.add_argument("--alice", required=True)
    parser.add_argument("--same-bob", required=True)
    parser.add_argument("--shifted-bob", required=True)
    parser.add_argument("--out", default="chronos_time_emission/results0b_independence_remote/marginal_drift1")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
