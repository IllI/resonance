from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.special import rel_entr


DF_FUND_REF = 0.001147


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return x
    std = float(np.std(x))
    if std <= 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.mean(x))) / std


def extract_df_fund(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.size < 64:
        return float("nan")
    nperseg = min(z.size // 4, 512)
    freqs, power = welch(z, fs=1.0, nperseg=nperseg, window="hann")
    peaks, _ = find_peaks(power, height=np.percentile(power, 80))
    if peaks.size < 2:
        return float("nan")
    peak_freqs = np.sort(freqs[peaks])
    diffs = np.diff(peak_freqs)
    diffs = diffs[diffs > 1e-6]
    return float(np.median(diffs)) if diffs.size else float("nan")


def sideband_asymmetry(z: np.ndarray, df: float = DF_FUND_REF, bw: float = 0.0005) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.size < 64:
        return float("nan")
    nperseg = min(z.size // 4, 512)
    freqs, power = welch(z, fs=1.0, nperseg=nperseg, window="hann")
    lo = (freqs > df - 1.5 * bw) & (freqs < df - 0.5 * bw)
    hi = (freqs > df + 0.5 * bw) & (freqs < df + 1.5 * bw)
    p_lo = float(np.trapz(power[lo], freqs[lo])) if lo.any() else 0.0
    p_hi = float(np.trapz(power[hi], freqs[hi])) if hi.any() else 0.0
    denom = p_lo + p_hi
    return float((p_hi - p_lo) / denom) if denom > 1e-30 else 0.0


def okl_scan(z: np.ndarray, radius: int = 30) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    values = []
    for center in range(radius, z.size - radius):
        back = z[center - radius : center]
        fwd = z[center : center + radius]
        back_hist, edges = np.histogram(back, bins=16, density=True)
        fwd_hist, _ = np.histogram(fwd, bins=edges, density=True)
        eps = 1e-12
        back_hist = (back_hist + eps) / np.sum(back_hist + eps)
        fwd_hist = (fwd_hist + eps) / np.sum(fwd_hist + eps)
        values.append(float(np.sum(rel_entr(fwd_hist, back_hist))))
    return np.asarray(values, dtype=np.float64)


def spectral_summary(latencies: np.ndarray) -> Dict[str, float]:
    z = zscore(latencies)
    okl = okl_scan(z)
    return {
        "df_fund": extract_df_fund(z),
        "sideband_asym": sideband_asymmetry(z),
        "okl_mean": float(np.nanmean(okl)) if okl.size else 0.0,
        "okl_gt02_fraction": float(np.mean(okl > 0.2)) if okl.size else 0.0,
    }

