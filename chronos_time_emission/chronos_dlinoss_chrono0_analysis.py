#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


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


def robust_norm(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.clip((x - med) / mad, -clip, clip)


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


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 4:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    denom = np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12
    return float(np.dot(aa, bb) / denom)


def transition_geometry_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    if n < 4:
        return 0.0
    a = a[:n]
    b = b[:n]
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_std = np.where(np.std(a, axis=0) < 1e-8, 1.0, np.std(a, axis=0))
    b_std = np.where(np.std(b, axis=0) < 1e-8, 1.0, np.std(b, axis=0))
    a_level = (a - a_mean) / a_std
    b_level = (b - b_mean) / b_std
    level = cosine_score(a_level, b_level)
    a_delta = np.diff(a_level, axis=0)
    b_delta = np.diff(b_level, axis=0)
    delta = cosine_score(a_delta, b_delta) if min(a_delta.shape[0], b_delta.shape[0]) >= 3 else 0.0
    return float(0.35 * level + 0.65 * delta)


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


def bandpass_named(x: np.ndarray, fs_hz: float, name: str) -> np.ndarray:
    mode, lo, hi = BANDS[name]
    return fft_band_filter(x, fs_hz, mode=mode, lo_hz=lo, hi_hz=hi)


def band_energy(window: np.ndarray, fs_hz: float, name: str) -> float:
    filtered = bandpass_named(window, fs_hz, name)
    return float(np.mean(filtered * filtered))


def histogram_entropy(x: np.ndarray, bins: int = 16) -> float:
    hist, _ = np.histogram(robust_norm(x), bins=bins, density=False)
    probs = hist.astype(np.float64) + 1e-12
    probs /= np.sum(probs)
    return float(-np.sum(probs * np.log2(probs)))


def fit_ridge(x: np.ndarray, y: np.ndarray, reg: float = 1e-3):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    gram = x_aug.T @ x_aug
    gram += reg * np.eye(gram.shape[0], dtype=np.float64)
    beta = np.linalg.solve(gram, x_aug.T @ y)
    return beta[:-1], beta[-1]


def predict_ridge(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def fit_transition_residual(z_base: np.ndarray, cal_len: int):
    usable = max(3, min(cal_len, z_base.shape[0]) - 1)
    train_len = max(2, int(np.floor(0.10 * usable)))
    x_train = z_base[:train_len]
    y_train = z_base[1 : train_len + 1]
    w, b = fit_ridge(x_train, y_train, reg=1e-2)
    residual = np.zeros(z_base.shape[0], dtype=np.float64)
    if z_base.shape[0] > 1:
        pred = predict_ridge(z_base[:-1], w, b)
        residual[1:] = np.linalg.norm(z_base[1:] - pred, axis=1)
    return residual, w, b


def build_feature_matrix(q: np.ndarray, fs_hz: float, window: int, hop: int):
    q = robust_norm(q)
    schumann_full = bandpass_named(q, fs_hz, "schumann")
    phase_full = np.unwrap(np.angle(hilbert(schumann_full)))
    starts = list(range(0, max(1, q.size - window + 1), hop))
    rows = []
    centers = []
    for start in starts:
        stop = start + window
        if stop > q.size:
            break
        center = start + (window // 2)
        centers.append(center)
        window_t = q[start:stop]
        phase_here = phase_full[center]
        phase_prev = phase_full[max(0, center - hop)]
        phase_next = phase_full[min(q.size - 1, center + hop)]
        row = [
            float(np.mean(window_t)),
            float(np.std(window_t)),
            float(np.median(window_t)),
            float(np.median(np.abs(window_t - np.median(window_t)))),
            band_energy(window_t, fs_hz, "schumann"),
            band_energy(window_t, fs_hz, "anti"),
            band_energy(window_t, fs_hz, "grid"),
            band_energy(window_t, fs_hz, "low_drift"),
            band_energy(window_t, fs_hz, "high_residual"),
            float(phase_here),
            float(phase_here - phase_prev),
            float(phase_next - 2.0 * phase_here + phase_prev),
            histogram_entropy(window_t, bins=16),
        ]
        rows.append(row)
    z_base = np.asarray(rows, dtype=np.float64)
    return z_base, np.asarray(centers, dtype=np.int64)


def add_transition_residual(z_base: np.ndarray, cal_len: int):
    residual, w, b = fit_transition_residual(z_base, cal_len=cal_len)
    z = np.concatenate([z_base, residual[:, None]], axis=1)
    return z, residual, w, b


def fit_standardizer(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std


def align_lag(a: np.ndarray, b: np.ndarray, lag: int):
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        return a[-lag:], b[:lag]
    return a, b


def choose_best_transition_lag(a_cal: np.ndarray, b_cal: np.ndarray, max_lag: int):
    best = None
    for lag in range(-int(max_lag), int(max_lag) + 1):
        ax, bx = align_lag(a_cal, b_cal, lag)
        if min(ax.shape[0], bx.shape[0]) < 4:
            continue
        w, bias = fit_ridge(ax, bx, reg=1e-2)
        pred = predict_ridge(ax, w, bias)
        score = transition_geometry_score(pred, bx)
        record = {"lag": int(lag), "score": float(score), "w": w, "b": bias}
        if best is None or score > best["score"]:
            best = record
    return best


def choose_best_external_band(a_q: np.ndarray, b_q: np.ndarray, fs_hz: float, window: int, hop: int, max_lag: int):
    starts = list(range(0, max(1, a_q.size - window + 1), hop))
    best = None
    for band in ("schumann", "anti", "grid", "low_drift", "high_residual"):
        a_series = []
        b_series = []
        for start in starts:
            stop = start + window
            if stop > a_q.size or stop > b_q.size:
                break
            a_series.append(band_energy(a_q[start:stop], fs_hz, band))
            b_series.append(band_energy(b_q[start:stop], fs_hz, band))
        a_series = np.asarray(a_series, dtype=np.float64)[:, None]
        b_series = np.asarray(b_series, dtype=np.float64)[:, None]
        cal_len = max(4, a_series.shape[0] // 2)
        for lag in range(-int(max_lag), int(max_lag) + 1):
            ax, bx = align_lag(a_series[:cal_len], b_series[:cal_len], lag)
            if min(ax.shape[0], bx.shape[0]) < 4:
                continue
            w, bias = fit_ridge(ax, bx, reg=1e-2)
            pred = predict_ridge(ax, w, bias)
            score = transition_geometry_score(pred, bx)
            record = {"band": band, "lag": int(lag), "score": float(score), "w": w, "b": bias}
            if best is None or score > best["score"]:
                best = record
    return best


def compute_tau(a_z: np.ndarray, b_z: np.ndarray):
    combined = np.concatenate([a_z, b_z], axis=0)
    mean, std = fit_standardizer(combined)
    a_std = apply_standardizer(a_z, mean, std)
    b_std = apply_standardizer(b_z, mean, std)
    _, _, vt = np.linalg.svd(combined := np.concatenate([a_std, b_std], axis=0), full_matrices=False)
    if vt.shape[0] < 2:
        vt = np.pad(vt, ((0, 2 - vt.shape[0]), (0, 0)), mode="constant")
    basis = vt[:2].T
    a_proj = a_std @ basis
    b_proj = b_std @ basis
    a_tau = np.unwrap(np.angle(a_proj[:, 0] + 1j * a_proj[:, 1]))
    b_tau = np.unwrap(np.angle(b_proj[:, 0] + 1j * b_proj[:, 1]))
    return a_tau[:, None], b_tau[:, None], mean, std


def block_shuffle_rows(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [x[i : i + block] for i in range(0, x.shape[0], block)]
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)[: x.shape[0]]


def estimate_ar1_params(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=0)
    xc = x - mu
    if x.shape[0] < 2:
        phi = np.zeros(x.shape[1], dtype=np.float64)
        sigma = np.std(x, axis=0)
        return mu, phi, sigma
    denom = np.sum(xc[:-1] * xc[:-1], axis=0) + 1e-12
    phi = np.sum(xc[1:] * xc[:-1], axis=0) / denom
    phi = np.clip(phi, -0.98, 0.98)
    resid = xc[1:] - phi * xc[:-1]
    sigma = np.std(resid, axis=0) + 1e-6
    return mu, phi, sigma


def make_cross_seed_control(reference: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    mu, phi, sigma = estimate_ar1_params(reference)
    out = np.zeros_like(reference, dtype=np.float64)
    out[0] = mu + rng.normal(scale=np.maximum(sigma, 1e-3), size=reference.shape[1])
    for idx in range(1, reference.shape[0]):
        noise = rng.normal(scale=sigma, size=reference.shape[1])
        out[idx] = mu + phi * (out[idx - 1] - mu) + noise
    return out


def phase_surrogate_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(x - np.mean(x))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.size)
    phases[0] = 0.0
    if spec.size > 1 and x.size % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=x.size)


def phase_surrogate_matrix(x: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    cols = [phase_surrogate_1d(x[:, idx], rng) for idx in range(x.shape[1])]
    return np.stack(cols, axis=1)


def payload_controls(b_payload: np.ndarray, lag: int, seed: int):
    rng = np.random.default_rng(seed)
    controls = {}
    perm = rng.permutation(b_payload.shape[0])
    controls["shuffled_B"] = b_payload[perm]
    shift = max(4, abs(int(lag)) + 4)
    controls["wrong_time_B"] = np.roll(b_payload, shift=shift, axis=0)
    controls["severed_same_distribution_B"] = block_shuffle_rows(b_payload, block=4, rng=rng)
    controls["cross_seed_B"] = make_cross_seed_control(b_payload, seed=seed + 17)
    return controls


def score_phase_surrogates(pred: np.ndarray, truth: np.ndarray, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    true_score = transition_geometry_score(pred, truth)
    surrogate_scores = []
    for _ in range(int(n_perm)):
        surrogate = np.stack([phase_surrogate_1d(truth[:, idx], rng) for idx in range(truth.shape[1])], axis=1)
        surrogate_scores.append(transition_geometry_score(pred, surrogate))
    surrogate_scores = np.asarray(surrogate_scores, dtype=np.float64)
    p_value = float((1.0 + np.sum(surrogate_scores >= true_score)) / (surrogate_scores.size + 1.0))
    return true_score, surrogate_scores, p_value


def run_analysis(args):
    alice_meta, alice_q, alice_fs = load_q(args.alice)
    bob_meta, bob_q, bob_fs = load_q(args.bob)
    fs_hz = min(alice_fs, bob_fs)
    n = min(alice_q.size, bob_q.size)
    alice_q = robust_norm(alice_q[:n])
    bob_q = robust_norm(bob_q[:n])

    a_base, _ = build_feature_matrix(alice_q, fs_hz, window=args.window, hop=args.hop)
    b_base, _ = build_feature_matrix(bob_q, fs_hz, window=args.window, hop=args.hop)
    n_rows = min(a_base.shape[0], b_base.shape[0])
    a_base = a_base[:n_rows]
    b_base = b_base[:n_rows]
    cal_len = max(4, n_rows // 2)

    a_z, a_residual, _, _ = add_transition_residual(a_base, cal_len=cal_len)
    b_z, b_residual, _, _ = add_transition_residual(b_base, cal_len=cal_len)
    a_z = a_z[:n_rows]
    b_z = b_z[:n_rows]

    full_zero_lag = pearson(bandpass_named(alice_q, fs_hz, "full"), bandpass_named(bob_q, fs_hz, "full"))
    independence_pass = bool(full_zero_lag < 0.90)

    transition = choose_best_transition_lag(a_z[:cal_len], b_z[:cal_len], max_lag=args.max_lag)
    a_payload = a_z[cal_len:]
    b_payload = b_z[cal_len:]
    a_payload_aligned, b_payload_true = align_lag(a_payload, b_payload, transition["lag"])
    z_pred = predict_ridge(a_payload_aligned, transition["w"], transition["b"])
    payload_true_score = transition_geometry_score(z_pred, b_payload_true)
    payload_true_mse = float(np.mean((z_pred - b_payload_true) ** 2))

    controls = payload_controls(b_payload_true, lag=transition["lag"], seed=args.seed)
    control_scores = {name: transition_geometry_score(z_pred, target) for name, target in controls.items()}
    heldout_gain = float(payload_true_score - max(control_scores.values()))

    a_tau, b_tau, tau_mean, tau_std = compute_tau(a_z[:cal_len], b_z[:cal_len])
    tau_model = choose_best_transition_lag(a_tau, b_tau, max_lag=args.max_lag)
    a_tau_payload = apply_standardizer(a_z[cal_len:], tau_mean, tau_std)
    b_tau_payload = apply_standardizer(b_z[cal_len:], tau_mean, tau_std)
    tau_basis = np.linalg.svd(np.concatenate([apply_standardizer(a_z[:cal_len], tau_mean, tau_std), apply_standardizer(b_z[:cal_len], tau_mean, tau_std)], axis=0), full_matrices=False)[2][:2].T
    a_tau_payload = np.unwrap(np.angle((a_tau_payload @ tau_basis)[:, 0] + 1j * (a_tau_payload @ tau_basis)[:, 1]))[:, None]
    b_tau_payload = np.unwrap(np.angle((b_tau_payload @ tau_basis)[:, 0] + 1j * (b_tau_payload @ tau_basis)[:, 1]))[:, None]
    a_tau_payload_aligned, b_tau_payload_aligned = align_lag(a_tau_payload, b_tau_payload, tau_model["lag"])
    tau_pred = predict_ridge(a_tau_payload_aligned, tau_model["w"], tau_model["b"])
    relational_clock_score = transition_geometry_score(tau_pred, b_tau_payload_aligned)

    external = choose_best_external_band(alice_q, bob_q, fs_hz, window=args.window, hop=args.hop, max_lag=args.max_lag)
    starts = list(range(0, max(1, n - args.window + 1), args.hop))
    a_band = []
    b_band = []
    for start in starts:
        stop = start + args.window
        if stop > n:
            break
        a_band.append(band_energy(alice_q[start:stop], fs_hz, external["band"]))
        b_band.append(band_energy(bob_q[start:stop], fs_hz, external["band"]))
    a_band = np.asarray(a_band, dtype=np.float64)[:, None]
    b_band = np.asarray(b_band, dtype=np.float64)[:, None]
    a_band_payload, b_band_payload = align_lag(a_band[cal_len:], b_band[cal_len:], external["lag"])
    band_pred = predict_ridge(a_band_payload, external["w"], external["b"])
    external_band_score = transition_geometry_score(band_pred, b_band_payload)

    _, surrogate_scores, phase_p = score_phase_surrogates(z_pred, b_payload_true, n_perm=args.permutations, seed=args.seed + 101)

    result = {
        "program": "AQ-DLINOSS-CHRONO-0-analyze",
        "alice": {k: alice_meta.get(k) for k in ("node_label", "seed", "samples", "fs_actual_hz", "started_epoch", "ended_epoch")},
        "bob": {k: bob_meta.get(k) for k in ("node_label", "seed", "samples", "fs_actual_hz", "started_epoch", "ended_epoch")},
        "window": int(args.window),
        "hop": int(args.hop),
        "paired_samples": int(n),
        "paired_windows": int(n_rows),
        "calibration_windows": int(cal_len),
        "payload_windows": int(b_payload_true.shape[0]),
        "gate_minus_one": {
            "passed": independence_pass,
            "r_full_zero_lag": float(full_zero_lag),
        },
        "true_transition": {
            "selected_lag": int(transition["lag"]),
            "calibration_score": float(transition["score"]),
            "payload_true_score": float(payload_true_score),
            "payload_true_mse": float(payload_true_mse),
            "payload_null_scores": {k: float(v) for k, v in control_scores.items()},
            "heldout_cross_system_transition_gain": float(heldout_gain),
            "transition_operator_norm": float(np.linalg.norm(transition["w"])),
            "transition_operator_condition": float(np.linalg.cond(transition["w"])) if transition["w"].size else 0.0,
        },
        "external_band_carrier": {
            "selected_band": external["band"],
            "selected_lag": int(external["lag"]),
            "calibration_score": float(external["score"]),
            "payload_score": float(external_band_score),
        },
        "relational_clock_carrier": {
            "selected_lag": int(tau_model["lag"]),
            "calibration_score": float(tau_model["score"]),
            "payload_score": float(relational_clock_score),
        },
        "phase_surrogate_test": {
            "p_value": float(phase_p),
            "surrogate_score_mean": float(np.mean(surrogate_scores)),
            "surrogate_score_std": float(np.std(surrogate_scores)),
        },
        "gates": {
            "gate_1_passed": bool(heldout_gain >= 0.10 and payload_true_score > max(control_scores.values())),
            "gate_2_passed": bool(phase_p <= 0.01),
            "gate_3_passed": bool(max(relational_clock_score, external_band_score) > max(control_scores.values())),
            "relational_beats_external": bool((relational_clock_score - external_band_score) >= 0.05),
        },
        "null_prior_note": "If Gate 1 fails, treat the result as evidence that cross-system temporal geometry is not stable enough for calibration-to-payload prediction under this hardware/window configuration.",
        "feature_notes": {
            "transition_residual_mean_alice": float(np.mean(a_residual)),
            "transition_residual_mean_bob": float(np.mean(b_residual)),
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aq_dlinoss_chrono0_results.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary_lines = [
        "# AQ-DLINOSS-CHRONO-0 Summary",
        "",
        f"- gate_minus_one_passed: `{result['gate_minus_one']['passed']}`",
        f"- r_full_zero_lag: `{result['gate_minus_one']['r_full_zero_lag']:.4f}`",
        f"- heldout_cross_system_transition_gain: `{heldout_gain:.4f}`",
        f"- payload_true_score: `{payload_true_score:.4f}`",
        f"- payload_null_score_max: `{max(control_scores.values()):.4f}`",
        f"- external_band: `{external['band']}`",
        f"- external_band_payload_score: `{external_band_score:.4f}`",
        f"- relational_clock_payload_score: `{relational_clock_score:.4f}`",
        f"- phase_surrogate_p_value: `{phase_p:.4f}`",
        f"- gate_1_passed: `{result['gates']['gate_1_passed']}`",
        f"- gate_2_passed: `{result['gates']['gate_2_passed']}`",
        f"- gate_3_passed: `{result['gates']['gate_3_passed']}`",
    ]
    (out_dir / "aq_dlinoss_chrono0_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="AQ-DLINOSS-CHRONO-0 analyzer")
    parser.add_argument("--alice", required=True, help="Path to Alice compact q JSON")
    parser.add_argument("--bob", required=True, help="Path to Bob compact q JSON")
    parser.add_argument("--out", default="chronos_time_emission/results_chrono0")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=14)
    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
