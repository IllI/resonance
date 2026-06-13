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
def telemetry_kernel(a, b, c, phase):
    x = jnp.tanh(a @ b + c)
    x = jnp.sin(x + phase) * 0.5 + jnp.cos(x - phase) * 0.25
    return jnp.sum(x)


def robust_norm(x, clip=3.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.clip((x - med) / mad, -clip, clip)


def fft_band_filter(x, fs_hz, lo_hz=None, hi_hz=None, mode="pass"):
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs_hz)
    spec = np.fft.rfft(centered)
    if mode == "full":
        mask = np.ones_like(freqs, dtype=bool)
        mask[0] = False
    else:
        mask = (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
        if mode == "stop":
            mask = ~mask
            mask[0] = False
    filtered = np.fft.irfft(spec * mask, n=centered.size)
    return filtered


def entropy_q_series(signal, windows=64, feature_scales=8):
    x = robust_norm(signal)
    if x.size < 64:
        return robust_norm(x)
    win = max(64, x.size // int(windows))
    step = max(1, win // 2)
    rows = []
    for start in range(0, max(1, x.size - win), step):
        seg = x[start : start + win]
        feats = []
        cur = seg
        for _ in range(int(feature_scales)):
            if cur.size < 8:
                feats.append(0.0)
                continue
            n = (cur.size // 4) * 4
            blocks = cur[:n].reshape(-1, 4)
            var = blocks.var(axis=1)
            hist, _ = np.histogram(robust_norm(var), bins=16, range=(-3, 3), density=False)
            p = hist.astype(np.float64) + 1e-12
            p = p / p.sum()
            feats.append(float(-np.sum(p * np.log(p))))
            cur = blocks.mean(axis=1)
        rows.append(feats)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return robust_norm(x[: min(x.size, 64)])
    return robust_norm(arr.reshape(-1))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


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


def lock_gain_proxy(q_alice, q_bob, alpha):
    n = min(len(q_alice), len(q_bob))
    err = np.angle(np.exp(1j * float(alpha) * (q_alice[:n] - q_bob[:n])))
    rms = float(np.sqrt(np.mean(err * err)))
    coherence = float(np.mean(np.cos(err)))
    if rms < 1e-12:
        gain = 120.0
    else:
        gain = max(0.0, -20.0 * math.log10(rms + 1e-12) - 3.0)
    return {"phase_error_rms": rms, "phase_coherence": coherence, "lock_gain_proxy_db": gain}


def collect_tpu_signal(seed, samples, sample_hz, size, warmup):
    key = jax.random.PRNGKey(int(seed))
    k1, k2, k3 = jax.random.split(key, 3)
    a = jax.random.normal(k1, (int(size), int(size)), dtype=jnp.float32)
    b = jax.random.normal(k2, (int(size), int(size)), dtype=jnp.float32)
    c = jax.random.normal(k3, (int(size), int(size)), dtype=jnp.float32)
    for idx in range(int(warmup)):
        telemetry_kernel(a, b, c, jnp.float32(idx * 0.001)).block_until_ready()
        a = jnp.roll(a, 1, axis=0)
        b = jnp.roll(b, 1, axis=1)

    alice = []
    bob = []
    stamps = []
    period = 1.0 / float(sample_hz)
    next_t = time.perf_counter()
    for idx in range(int(samples)):
        phase = jnp.float32((idx + 1) * 0.0065)
        stamps.append(time.perf_counter_ns())
        av = telemetry_kernel(a + phase, b - phase, c, phase).block_until_ready()
        bv = telemetry_kernel(a + phase * 1.01, b - phase * 0.99, c, phase * 1.005).block_until_ready()
        alice.append(float(av))
        bob.append(float(bv))
        a = jnp.roll(a, 1, axis=0)
        b = jnp.roll(b, 1, axis=1)
        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
    stamps = np.asarray(stamps, dtype=np.int64)
    duration = max(1e-9, float(stamps[-1] - stamps[0]) / 1e9) if stamps.size > 1 else 1.0
    fs_actual = float((stamps.size - 1) / duration) if stamps.size > 1 else float(sample_hz)
    return np.asarray(alice), np.asarray(bob), fs_actual


def schumann_detection(x, fs_hz):
    z = robust_norm(x)
    freqs = np.fft.rfftfreq(z.size, d=1.0 / fs_hz)
    psd = np.abs(np.fft.rfft(z - z.mean())) ** 2
    out = {}
    for f0 in SCHUMANN_HZ:
        peak_mask = (freqs >= f0 - 0.5) & (freqs <= f0 + 0.5)
        noise_mask = (freqs >= f0 - 5.0) & (freqs <= f0 + 5.0) & ~peak_mask
        peak = float(psd[peak_mask].max()) if peak_mask.any() else 0.0
        noise = float(np.median(psd[noise_mask])) if noise_mask.any() else float(np.median(psd) + 1e-30)
        snr = 10.0 * math.log10((peak + 1e-30) / (noise + 1e-30))
        out[f"{f0:g}Hz"] = {"snr_db": snr, "detected": bool(snr > 3.0)}
    return out


def evaluate_modes(alice, bob, fs_hz, alpha_values, max_lag):
    rng = np.random.default_rng(11)
    modes = {
        "schumann": (fft_band_filter(alice, fs_hz, 6.0, 40.0, "pass"), fft_band_filter(bob, fs_hz, 6.0, 40.0, "pass")),
        "anti": (fft_band_filter(alice, fs_hz, 6.0, 40.0, "stop"), fft_band_filter(bob, fs_hz, 6.0, 40.0, "stop")),
        "grid": (fft_band_filter(alice, fs_hz, 58.0, min(62.0, fs_hz / 2.0 - 1.0), "pass"), fft_band_filter(bob, fs_hz, 58.0, min(62.0, fs_hz / 2.0 - 1.0), "pass")),
        "full": (fft_band_filter(alice, fs_hz, mode="full"), fft_band_filter(bob, fs_hz, mode="full")),
    }
    results = {}
    for name, (a_sig, b_sig) in modes.items():
        qa = entropy_q_series(a_sig)
        qb = entropy_q_series(b_sig)
        shuffled = rng.permutation(qb)
        random_q = robust_norm(rng.normal(size=qb.size))
        lag = lag_corr(qa, qb, max_lag=max_lag)
        mode_result = {
            "q_len": int(min(len(qa), len(qb))),
            "signal_rms_alice": float(np.sqrt(np.mean(a_sig * a_sig))),
            "signal_rms_bob": float(np.sqrt(np.mean(b_sig * b_sig))),
            "r_zero": pearson(qa, qb),
            "r_best": lag["r"],
            "best_lag": lag["lag"],
            "r_shuffled": pearson(qa, shuffled),
            "r_random": pearson(qa, random_q),
            "alpha": {},
        }
        for alpha in alpha_values:
            local = lock_gain_proxy(qa, qb, alpha)
            shuf = lock_gain_proxy(qa, shuffled, alpha)
            rand = lock_gain_proxy(qa, random_q, alpha)
            mode_result["alpha"][str(alpha)] = {
                **local,
                "gain_vs_shuffled_db": local["lock_gain_proxy_db"] - shuf["lock_gain_proxy_db"],
                "gain_vs_random_db": local["lock_gain_proxy_db"] - rand["lock_gain_proxy_db"],
            }
        results[name] = mode_result
    return results


def write_summary(path, payload):
    lines = [
        "# CHRONOS-SCHUMANN-0a Summary",
        "",
        f"- backend: `{payload['backend']}`",
        f"- devices: `{payload['n_devices']}`",
        f"- samples: `{payload['samples']}`",
        f"- fs_actual_hz: `{payload['fs_actual_hz']:.3f}`",
        "",
        "| mode | r_zero | r_best | best_lag | rms_alice | alpha=0.5 gain proxy | vs shuffled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["modes"].items():
        alpha = row["alpha"].get("0.5") or next(iter(row["alpha"].values()))
        lines.append(
            f"| {name} | {row['r_zero']:.4f} | {row['r_best']:.4f} | {row['best_lag']} | "
            f"{row['signal_rms_alice']:.4f} | {alpha['lock_gain_proxy_db']:.4f} | "
            f"{alpha['gain_vs_shuffled_db']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="CHRONOS Schumann-band TPU telemetry screening")
    parser.add_argument("--out", default="chronos_time_emission/results/schumann0a")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--sample-hz", type=float, default=128.0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--require-tpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = jax.default_backend()
    if args.require_tpu and backend != "tpu":
        raise RuntimeError(f"--require-tpu was set, but backend is {backend!r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("CHRONOS-SCHUMANN-0a TPU telemetry screening")
    print(f"backend={backend} devices={len(jax.devices())} samples={args.samples} sample_hz={args.sample_hz}")
    alice, bob, fs_actual = collect_tpu_signal(args.seed, args.samples, args.sample_hz, args.size, args.warmup)
    modes = evaluate_modes(alice, bob, fs_actual, args.alphas, args.max_lag)
    payload = {
        "program": "CHRONOS-SCHUMANN-0a",
        "backend": backend,
        "n_devices": len(jax.devices()),
        "seed": int(args.seed),
        "samples": int(args.samples),
        "sample_hz_requested": float(args.sample_hz),
        "fs_actual_hz": float(fs_actual),
        "size": int(args.size),
        "schumann_detection_alice": schumann_detection(alice, fs_actual),
        "schumann_detection_bob": schumann_detection(bob, fs_actual),
        "modes": modes,
        "pid": os.getpid(),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out / "summary.md", payload)
    print("CHRONOS-SCHUMANN-0a COMPLETE")
    for name, row in modes.items():
        alpha = row["alpha"].get("0.5") or next(iter(row["alpha"].values()))
        print(
            f"{name:9s} r0={row['r_zero']:+.4f} rb={row['r_best']:+.4f} lag={row['best_lag']:+d} "
            f"gain05={alpha['lock_gain_proxy_db']:.4f} vs_shuf={alpha['gain_vs_shuffled_db']:.4f}"
        )
    print(f"results={out / 'results.json'}")


if __name__ == "__main__":
    main()
