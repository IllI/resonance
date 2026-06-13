#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def load_q(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    q = np.asarray(payload["q_latency"], dtype=np.float64)
    fs = float(payload.get("fs_actual_hz", payload.get("sample_hz_requested", 1.0)))
    return payload, q, fs


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


def robust_norm(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.clip((x - med) / mad, -5.0, 5.0)


def fft_band_filter(x, fs_hz, mode="pass", lo_hz=None, hi_hz=None):
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


def band_map(q, fs):
    out = {}
    for name, (mode, lo, hi) in BANDS.items():
        out[name] = robust_norm(fft_band_filter(q, fs, mode=mode, lo_hz=lo, hi_hz=hi))
    return out


def align_pair(a, b, lag):
    if lag > 0:
        return a[lag:], b[:-lag]
    if lag < 0:
        return a[:lag], b[-lag:]
    return a, b


def scan_carrier(alice_bands, bob_bands, max_lag):
    best = None
    rows = []
    for band in alice_bands:
        for lag in range(-int(max_lag), int(max_lag) + 1):
            aa, bb = align_pair(alice_bands[band], bob_bands[band], lag)
            r = pearson(aa, bb)
            sign = 1.0 if r >= 0 else -1.0
            row = {"band": band, "lag": int(lag), "sign": sign, "r": float(r), "abs_r": abs(float(r))}
            rows.append(row)
            if best is None or row["abs_r"] > best["abs_r"]:
                best = row
    return best, rows


def lock_gain(a, b, alpha):
    n = min(a.size, b.size)
    err = np.angle(np.exp(1j * float(alpha) * (a[:n] - b[:n])))
    rms = float(np.sqrt(np.mean(err * err)))
    coherence = float(np.mean(np.cos(err)))
    gain = 120.0 if rms < 1e-12 else max(0.0, -20.0 * math.log10(rms + 1e-12) - 3.0)
    return {"phase_error_rms": rms, "phase_coherence": coherence, "lock_gain_proxy_db": gain}


def phase_surrogate(x, rng):
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x - np.mean(x))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.size)
    phases[0] = 0.0
    if spec.size > 1 and x.size % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=x.size)


def evaluate_payload(best, alice_payload_bands, bob_payload_bands, alpha, seed):
    rng = np.random.default_rng(seed)
    band = best["band"]
    a, b = align_pair(alice_payload_bands[band], bob_payload_bands[band] * best["sign"], best["lag"])
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    controls = {
        "true": b,
        "shuffled": rng.permutation(b),
        "wrong_time": np.roll(b, max(1, n // 4)),
        "severed_phase": robust_norm(phase_surrogate(b, rng)),
        "random": robust_norm(rng.normal(size=n)),
    }
    out = {}
    for name, series in controls.items():
        out[name] = {
            "r": pearson(a, series[:n]),
            **lock_gain(a, series[:n], alpha),
        }
    true_gain = out["true"]["lock_gain_proxy_db"]
    for name in list(out):
        out[name]["gain_delta_from_true_db"] = true_gain - out[name]["lock_gain_proxy_db"]
    return out


def run_one(alice_path, bob_path, out_path, cal_fraction, max_lag, alpha, seed):
    alice_meta, alice_q, alice_fs = load_q(alice_path)
    bob_meta, bob_q, bob_fs = load_q(bob_path)
    n = min(alice_q.size, bob_q.size)
    split = int(n * float(cal_fraction))
    fs = min(alice_fs, bob_fs)
    alice_cal, alice_payload = alice_q[:split], alice_q[split:n]
    bob_cal, bob_payload = bob_q[:split], bob_q[split:n]
    alice_cal_bands = band_map(alice_cal, fs)
    bob_cal_bands = band_map(bob_cal, fs)
    best, rows = scan_carrier(alice_cal_bands, bob_cal_bands, max_lag)
    payload = evaluate_payload(best, band_map(alice_payload, fs), band_map(bob_payload, fs), alpha, seed + 101)
    pass_gate = bool(
        payload["true"]["r"] > 0.10
        and payload["true"]["gain_delta_from_true_db"] == 0.0
        and payload["shuffled"]["gain_delta_from_true_db"] > 0.25
        and payload["wrong_time"]["gain_delta_from_true_db"] > 0.25
        and payload["severed_phase"]["gain_delta_from_true_db"] > 0.25
    )
    result = {
        "program": "CHRONOS-CARRIER-2-split-calibration",
        "alice": {"path": str(alice_path), "node_label": alice_meta.get("node_label"), "fs_actual_hz": alice_fs},
        "bob": {"path": str(bob_path), "node_label": bob_meta.get("node_label"), "fs_actual_hz": bob_fs},
        "n_samples": int(n),
        "calibration_samples": int(split),
        "payload_samples": int(n - split),
        "fs_hz_used": float(fs),
        "alpha": float(alpha),
        "max_lag": int(max_lag),
        "selected_carrier": best,
        "payload": payload,
        "pass_gate": pass_gate,
        "top_calibration_candidates": sorted(rows, key=lambda row: row["abs_r"], reverse=True)[:8],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="CHRONOS-CARRIER-2 calibration/payload carrier test")
    parser.add_argument("--alice", required=True)
    parser.add_argument("--bob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--max-lag", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    result = run_one(args.alice, args.bob, args.out, args.cal_fraction, args.max_lag, args.alpha, args.seed)
    best = result["selected_carrier"]
    print("# CHRONOS-CARRIER-2 split-calibration")
    print(f"alice={result['alice']['node_label']} bob={result['bob']['node_label']}")
    print(f"selected band={best['band']} lag={best['lag']} sign={best['sign']:+.0f} cal_r={best['r']:+.4f}")
    for name, row in result["payload"].items():
        print(
            f"{name:13s} r={row['r']:+.4f} gain={row['lock_gain_proxy_db']:.4f} "
            f"true_minus_control={row['gain_delta_from_true_db']:.4f}"
        )
    print(f"pass_gate={result['pass_gate']}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
