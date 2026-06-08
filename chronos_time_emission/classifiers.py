from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .spectral import spectral_summary


def _safe_acc(pred: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0:
        return 0.5
    return float(np.mean(pred.astype(np.int32) == y.astype(np.int32)))


def _threshold_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    x0 = train_x[train_y == 0]
    x1 = train_x[train_y == 1]
    if x0.size == 0 or x1.size == 0:
        return np.zeros_like(test_x, dtype=np.int32)
    m0 = float(np.mean(x0))
    m1 = float(np.mean(x1))
    threshold = 0.5 * (m0 + m1)
    return (test_x > threshold).astype(np.int32) if m1 > m0 else (test_x < threshold).astype(np.int32)


def block_feature_table(records: list[dict]) -> list[dict]:
    rows = []
    if not records:
        return rows
    blocks = sorted({int(row["block"]) for row in records})
    labels = sorted({int(row["label"]) for row in records})
    for block in blocks:
        for label in labels:
            subset = [row for row in records if int(row["block"]) == block and int(row["label"]) == label]
            if not subset:
                continue
            latencies = np.asarray([float(row["latency_ns"]) for row in subset], dtype=np.float64)
            spec = spectral_summary(latencies)
            rows.append(
                {
                    "block": block,
                    "label": label,
                    "n": int(latencies.size),
                    "mean_latency_ns": float(np.mean(latencies)),
                    "std_latency_ns": float(np.std(latencies)),
                    "df_fund": float(spec["df_fund"]),
                    "sideband_asym": float(spec["sideband_asym"]),
                    "okl_mean": float(spec["okl_mean"]),
                }
            )
    return rows


def _split_features(feature_rows: list[dict], train_fraction: float = 0.75) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not feature_rows:
        return (
            np.zeros((0, 5), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 5), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
        )
    max_block = max(int(row["block"]) for row in feature_rows)
    cutoff = int(np.floor((max_block + 1) * train_fraction))
    cols = ["mean_latency_ns", "std_latency_ns", "df_fund", "sideband_asym", "okl_mean"]

    def to_xy(rows: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
        return x, y

    train_rows = [row for row in feature_rows if int(row["block"]) < cutoff]
    test_rows = [row for row in feature_rows if int(row["block"]) >= cutoff]
    if not test_rows:
        train_rows = feature_rows[::2]
        test_rows = feature_rows[1::2]
    train_x, train_y = to_xy(train_rows)
    test_x, test_y = to_xy(test_rows)
    return train_x, train_y, test_x, test_y


def evaluate_block_classifiers(feature_rows: list[dict], seed: int = 0) -> Dict[str, float]:
    train_x, train_y, test_x, test_y = _split_features(feature_rows)
    if train_y.size < 2 or test_y.size < 1 or len(np.unique(train_y)) < 2:
        return {
            "mean_acc": 0.5,
            "sideband_acc": 0.5,
            "combined_acc": 0.5,
            "shuffled_acc": 0.5,
        }

    mean_pred = _threshold_predict(train_x[:, 0], train_y, test_x[:, 0])
    sideband_pred = _threshold_predict(train_x[:, 3], train_y, test_x[:, 3])

    mu = np.mean(train_x, axis=0, keepdims=True)
    sigma = np.std(train_x, axis=0, keepdims=True) + 1e-9
    xtr = (train_x - mu) / sigma
    xte = (test_x - mu) / sigma
    xtr_aug = np.concatenate([xtr, np.ones((xtr.shape[0], 1))], axis=1)
    xte_aug = np.concatenate([xte, np.ones((xte.shape[0], 1))], axis=1)
    y_signed = 2.0 * train_y.astype(np.float64) - 1.0
    ridge = 1e-3 * np.eye(xtr_aug.shape[1])
    w = np.linalg.solve(xtr_aug.T @ xtr_aug + ridge, xtr_aug.T @ y_signed)
    combined_pred = (xte_aug @ w > 0.0).astype(np.int32)

    rng = np.random.default_rng(seed)
    shuffled_y = rng.permutation(train_y)
    shuffled_mean_pred = _threshold_predict(train_x[:, 0], shuffled_y, test_x[:, 0])

    return {
        "mean_acc": _safe_acc(mean_pred, test_y),
        "sideband_acc": _safe_acc(sideband_pred, test_y),
        "combined_acc": _safe_acc(combined_pred, test_y),
        "shuffled_acc": _safe_acc(shuffled_mean_pred, test_y),
    }

