#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_q(path: str | Path):
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


def modes(q, fs):
    return {
        "full": fft_band_filter(q, fs, mode="full"),
        "schumann": fft_band_filter(q, fs, 6.0, 40.0, "pass"),
        "anti": fft_band_filter(q, fs, 6.0, 40.0, "stop"),
        "grid": fft_band_filter(q, fs, 58.0, 62.0, "pass"),
    }


def lag_scan(a, b, max_lag):
    rows = []
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag > 0:
            aa, bb = a[lag:], b[:-lag]
        elif lag < 0:
            aa, bb = a[:lag], b[-lag:]
        else:
            aa, bb = a, b
        rows.append({"lag": lag, "r": pearson(aa, bb)})
    return rows


def phase_surrogate(x, rng):
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x - np.mean(x))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.size)
    phases[0] = 0.0
    if spec.size > 1 and x.size % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=x.size)


def block_permute(x, block_size, rng):
    x = np.asarray(x, dtype=np.float64)
    n_blocks = x.size // int(block_size)
    if n_blocks < 2:
        return rng.permutation(x)
    trimmed = x[: n_blocks * int(block_size)].reshape(n_blocks, int(block_size))
    permuted = trimmed[rng.permutation(n_blocks)].reshape(-1)
    if permuted.size < x.size:
        permuted = np.concatenate([permuted, x[permuted.size :]])
    return permuted


def p_ge(values, observed):
    values = np.asarray(values, dtype=np.float64)
    return float((np.sum(values >= observed) + 1.0) / (values.size + 1.0))


def summarize_surrogates(alice_modes, bob_q, fs, observed, permutations, block_size, seed):
    rng = np.random.default_rng(seed)
    controls = {
        "shuffle": [],
        "block": [],
        "phase": [],
    }
    for _ in range(int(permutations)):
        controls["shuffle"].append(rng.permutation(bob_q))
        controls["block"].append(block_permute(bob_q, block_size, rng))
        controls["phase"].append(phase_surrogate(bob_q, rng))

    out = {}
    for control, series_list in controls.items():
        schumann_r = []
        delta_r = []
        for surrogate in series_list:
            b_modes = modes(surrogate, fs)
            r_sc = pearson(alice_modes["schumann"], b_modes["schumann"])
            r_anti = pearson(alice_modes["anti"], b_modes["anti"])
            schumann_r.append(r_sc)
            delta_r.append(r_sc - r_anti)
        out[control] = {
            "schumann_p_ge": p_ge(schumann_r, observed["schumann_r"]),
            "delta_p_ge": p_ge(delta_r, observed["schumann_minus_anti"]),
            "schumann_mean": float(np.mean(schumann_r)),
            "schumann_std": float(np.std(schumann_r)),
            "delta_mean": float(np.mean(delta_r)),
            "delta_std": float(np.std(delta_r)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="CHRONOS Schumann 0b significance and lag analysis")
    parser.add_argument("--alice", required=True)
    parser.add_argument("--bob", required=True)
    parser.add_argument("--out", default="chronos_time_emission/results0b_independence_remote/significance_results.json")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--delta-threshold", type=float, default=0.10)
    parser.add_argument("--p-threshold", type=float, default=0.01)
    parser.add_argument("--max-primary-lag", type=int, default=0)
    args = parser.parse_args()

    alice_meta, alice_q, alice_fs = load_q(args.alice)
    bob_meta, bob_q, bob_fs = load_q(args.bob)
    n = min(alice_q.size, bob_q.size)
    fs = min(alice_fs, bob_fs)
    alice_q = alice_q[:n]
    bob_q = bob_q[:n]
    alice_modes = modes(alice_q, fs)
    bob_modes = modes(bob_q, fs)

    mode_rows = {}
    for name in ("full", "schumann", "anti", "grid"):
        lags = lag_scan(alice_modes[name], bob_modes[name], args.max_lag)
        best = max(lags, key=lambda row: abs(row["r"]))
        mode_rows[name] = {
            "r_zero": pearson(alice_modes[name], bob_modes[name]),
            "best_lag": int(best["lag"]),
            "r_best": float(best["r"]),
            "best_lag_seconds": float(best["lag"] / fs),
        }

    observed = {
        "schumann_r": mode_rows["schumann"]["r_zero"],
        "schumann_minus_anti": mode_rows["schumann"]["r_zero"] - mode_rows["anti"]["r_zero"],
    }
    z_approx = observed["schumann_r"] * math.sqrt(max(1, n - 3))
    lag_delta = mode_rows["schumann"]["r_best"] - mode_rows["schumann"]["r_zero"]
    controls = summarize_surrogates(
        alice_modes,
        bob_q,
        fs,
        observed,
        args.permutations,
        args.block_size,
        args.seed,
    )
    primary_pass = bool(
        observed["schumann_minus_anti"] >= args.delta_threshold
        and controls["phase"]["delta_p_ge"] <= args.p_threshold
    )
    secondary_pass = bool(
        mode_rows["full"]["r_zero"] < 0.90
        and observed["schumann_r"] > 0.0
        and mode_rows["anti"]["r_zero"] < observed["schumann_r"]
    )
    diagnostic_lag_only = bool(abs(mode_rows["schumann"]["best_lag"]) > args.max_primary_lag)

    payload = {
        "program": "CHRONOS-SCHUMANN-0b-significance",
        "alice": {
            "node_label": alice_meta.get("node_label"),
            "started_epoch": alice_meta.get("started_epoch"),
            "ended_epoch": alice_meta.get("ended_epoch"),
            "fs_actual_hz": alice_fs,
        },
        "bob": {
            "node_label": bob_meta.get("node_label"),
            "started_epoch": bob_meta.get("started_epoch"),
            "ended_epoch": bob_meta.get("ended_epoch"),
            "fs_actual_hz": bob_fs,
        },
        "n_samples": int(n),
        "fs_hz_used": float(fs),
        "mode_rows": mode_rows,
        "schumann_minus_anti": observed["schumann_minus_anti"],
        "schumann_z_approx_raw_samples": float(z_approx),
        "schumann_best_minus_zero": float(lag_delta),
        "permutations": int(args.permutations),
        "block_size": int(args.block_size),
        "preregistered_gates": {
            "primary_endpoint": "zero_lag_schumann_minus_anti",
            "delta_threshold": float(args.delta_threshold),
            "p_threshold": float(args.p_threshold),
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "diagnostic_lag_only": diagnostic_lag_only,
        },
        "controls": controls,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("# CHRONOS-SCHUMANN-0b Significance")
    print(f"n_samples={n} fs_hz={fs:.6f} permutations={args.permutations}")
    for name in ("full", "schumann", "anti", "grid"):
        row = mode_rows[name]
        print(
            f"{name:9s} r0={row['r_zero']:+.4f} "
            f"best={row['r_best']:+.4f} lag={row['best_lag']:+d} "
            f"lag_s={row['best_lag_seconds']:+.4f}"
        )
    print(f"schumann_minus_anti={observed['schumann_minus_anti']:+.4f}")
    print(f"schumann_best_minus_zero={lag_delta:+.4f}")
    for name, row in controls.items():
        print(
            f"{name:7s} p_sc={row['schumann_p_ge']:.4f} "
            f"p_delta={row['delta_p_ge']:.4f} "
            f"delta_null={row['delta_mean']:+.4f}+/-{row['delta_std']:.4f}"
        )
    print(f"primary_pass={primary_pass} secondary_pass={secondary_pass} diagnostic_lag_only={diagnostic_lag_only}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
