"""
ptm_corpus_gen.py — TPU Program A: Full PTM Spectral Dataset
=============================================================
Generates D = {chi_t, Gamma, N, T_ij, lambda_k, V_Q, F_avg, kappa_Q, rank}

Grid:
  N      : [4, 6, 8]
  chi_t  : 60 points in [0.01, pi]
  Gamma  : 12 points in [0.0, 0.4]
  noise  : dephasing (Lindblad, analytic)
Total    : 3 * 60 * 12 = 2160 points (fast sweep, no V_Q/kappa_Q)
           + 200 selected points with full V_Q + kappa_Q

Runtime  : ~3 min on CPU (fast mode), ~25 min for full V_Q+kQ subset

Output   : ptm_corpus.npz  (arrays)
           ptm_corpus_meta.json  (all field values as list-of-dicts)

Usage:
  python ptm_corpus_gen.py                    # default
  python ptm_corpus_gen.py --full-vq          # include V_Q + kappa_Q on all pts
  python ptm_corpus_gen.py --N 4 --chi-pts 30 # smaller test run
"""

import argparse
import json
import time
import numpy as np

from ptm_features import PTMFeatures, PTMFeatureVector, rho2_exact, compute_V_Q, compute_kappa_Q

# ── Noise models ──────────────────────────────────────────────────────────────
NOISE_MODELS = {
    "dephasing": "analytic_lindblad",
    "amp_damp":  "amplitude_damping",
    "coherent":  "coherent_overrotation",
}


def apply_amplitude_damping(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Single-qubit amplitude damping on each qubit independently.
    Kraus operators: K0 = [[1,0],[0,sqrt(1-p)]], K1 = [[0,sqrt(p)],[0,0]]
    Applied to each qubit in the 2-qubit boundary state.
    p = damping probability (p=0: no noise, p=1: full decay to |0>)
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    result = np.zeros((4, 4), dtype=complex)
    for ka in [K0, K1]:
        for kb in [K0, K1]:
            K = np.kron(ka, kb)
            result += K @ rho @ K.conj().T
    return result


def apply_coherent_overrotation(rho: np.ndarray, eps: float) -> np.ndarray:
    """
    Coherent overrotation: apply R_z(eps) to each qubit.
    Models systematic angle errors in gate calibration.
    """
    Rz = np.array([[np.exp(-1j * eps / 2), 0],
                   [0, np.exp(1j * eps / 2)]], dtype=complex)
    U = np.kron(Rz, Rz)
    return U @ rho @ U.conj().T


# ── V_Q subset selection: pick points spanning the phase diagram ───────────────
def select_vq_subset(chi_t_grid, Gamma_grid, N_vals, n_per_N=50):
    """Select ~n_per_N points per N for full V_Q + kappa_Q computation."""
    subset = []
    for N in N_vals:
        # Always include: product state, optimal, singularity, and dephased boundary
        key_chi = [0.05, 0.3, 0.716 * np.pi, np.pi * 0.9, np.pi * 0.99]
        key_Gamma = [0.0, 0.05, 0.15]
        for g in key_Gamma:
            for c in key_chi:
                subset.append((N, c, g))
        # Fill remaining slots with uniform random sample
        rng = np.random.default_rng(42)
        n_remain = max(0, n_per_N - len(subset))
        rand_chi = rng.choice(chi_t_grid, size=n_remain)
        rand_Gam = rng.choice(Gamma_grid, size=n_remain)
        for c, g in zip(rand_chi, rand_Gam):
            subset.append((N, c, g))
    return list(set(subset))


# ── Main corpus generation ────────────────────────────────────────────────────
def generate_corpus(N_vals, chi_pts, Gamma_pts, full_vq=False,
                    noise_models=("dephasing",), verbose=True):
    chi_t_grid = np.linspace(0.01, np.pi, chi_pts)
    Gamma_grid  = np.linspace(0.0, 0.4, Gamma_pts)

    all_rows = []
    all_meta = []

    for N in N_vals:
        feat = PTMFeatures(N=N, V_Q_samples=150, kQ_restarts=6, fast=not full_vq)

        # Determine V_Q subset for this N
        if not full_vq:
            vq_subset = set()
            # Key diagnostic points always get full V_Q
            for c in [0.05, 0.716 * np.pi, 1.189, np.pi - 0.01]:
                vq_subset.add((round(c, 4), 0.0))
        else:
            vq_subset = None  # all points

        total = chi_pts * Gamma_pts
        k = 0
        t0 = time.time()

        for noise in noise_models:
            for Gamma in Gamma_grid:
                for chi_t in chi_t_grid:
                    k += 1

                    # Base rho from Proposition 1 (analytic, includes Lindblad dephasing)
                    rho = rho2_exact(chi_t, N, Gamma if noise == "dephasing" else 0.0)

                    # Apply additional noise model
                    if noise == "amp_damp" and Gamma > 0:
                        rho = apply_amplitude_damping(rho, p=Gamma * 0.5)
                    elif noise == "coherent" and Gamma > 0:
                        rho = apply_coherent_overrotation(rho, eps=Gamma * 0.3)

                    # Decide whether to compute V_Q + kappa_Q
                    use_full = full_vq or (vq_subset is not None and
                                           (round(chi_t, 4), round(Gamma, 4)) in vq_subset)

                    if use_full:
                        feat.fast = False
                    else:
                        feat.fast = True

                    fv = feat.compute(chi_t, Gamma, seed=k)
                    fv_dict = {
                        **{f: getattr(fv, f) for f in PTMFeatureVector.feature_names()},
                        "noise_model": noise,
                        "N": N,
                    }
                    all_rows.append(fv.to_array())
                    all_meta.append(fv_dict)

                    if verbose and k % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  N={N} {noise:10s}  {k}/{total}  "
                              f"chi_t={chi_t:.3f}  Gamma={Gamma:.3f}  "
                              f"F={fv.F_avg:.4f}  rank={fv.T_rank}  "
                              f"[{elapsed:.0f}s]")

        if verbose:
            print(f"  N={N} done — {total} points × {len(noise_models)} models "
                  f"in {time.time()-t0:.1f}s")

    return np.array(all_rows), all_meta


# ── Phase labels (post-hoc, never used in training) ──────────────────────────
def assign_phase_labels(meta: list) -> list:
    """
    Assign ground-truth phase labels for validation only.
    Labels are NOT fed to D-LinOSS during training.
    """
    labels = []
    for m in meta:
        chi_t = m["chi_t"]
        Gamma = m["Gamma"]
        F = m["F_avg"]
        T_rank = m["T_rank"]
        # Singularity: near chi_t=pi and low fidelity
        if abs(chi_t - np.pi) < 0.05 and F < 0.51:
            labels.append("FLAT")
        elif F > 2 / 3 and T_rank >= 3 and Gamma < 0.15:
            labels.append("QUANTUM")
        else:
            labels.append("CLASSICAL")
    return labels


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate PTM corpus")
    parser.add_argument("--N", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--chi-pts", type=int, default=60)
    parser.add_argument("--gamma-pts", type=int, default=12)
    parser.add_argument("--full-vq", action="store_true",
                        help="Compute V_Q + kappa_Q on ALL points (slow)")
    parser.add_argument("--noise", nargs="+", default=["dephasing"],
                        choices=["dephasing", "amp_damp", "coherent"])
    parser.add_argument("--out", default="ptm_corpus")
    args = parser.parse_args()

    print("=" * 65)
    print("PTM CORPUS GENERATOR — TPU Program A")
    print("=" * 65)
    print(f"  N = {args.N}")
    print(f"  chi_t: {args.chi_pts} points in [0.01, pi]")
    print(f"  Gamma: {args.gamma_pts} points in [0.0, 0.4]")
    print(f"  Noise models: {args.noise}")
    print(f"  Full V_Q: {args.full_vq}")
    total = len(args.N) * args.chi_pts * args.gamma_pts * len(args.noise)
    print(f"  Total points: {total}")
    print()

    t0 = time.time()
    rows, meta = generate_corpus(
        N_vals=args.N,
        chi_pts=args.chi_pts,
        Gamma_pts=args.gamma_pts,
        full_vq=args.full_vq,
        noise_models=args.noise,
        verbose=True,
    )
    labels = assign_phase_labels(meta)

    # Save
    npz_path = f"{args.out}.npz"
    np.savez(npz_path,
             X=rows,
             labels=np.array(labels),
             feature_names=np.array(PTMFeatureVector.feature_names()))
    print(f"\nSaved: {npz_path}  shape={rows.shape}")

    json_path = f"{args.out}_meta.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "labels": labels,
                   "feature_names": PTMFeatureVector.feature_names()}, f, indent=2)
    print(f"Saved: {json_path}  ({len(meta)} records)")

    # Quick summary
    label_arr = np.array(labels)
    print(f"\nPhase distribution (post-hoc labels):")
    for lab in ["QUANTUM", "CLASSICAL", "FLAT"]:
        n = (label_arr == lab).sum()
        print(f"  {lab:<12} {n:5d} ({100*n/len(labels):.1f}%)")

    print(f"\nCorpus generated in {time.time()-t0:.1f}s")
    print(f"Feature shape: {rows.shape}  ({rows.shape[1]} features per point)")


if __name__ == "__main__":
    main()
