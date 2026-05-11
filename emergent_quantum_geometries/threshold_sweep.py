"""
threshold_sweep.py
==================
Tests K_eff stability as a function of SVD truncation threshold and maximum
allowed rank K_max. Also computes BIC-penalized rank selection.

Critical question: Is K_eff=6 for SYK/XXZ a physical result or a threshold artifact?

Run: python threshold_sweep.py
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

# ── Matrix pencil core (inlined to allow threshold control) ───────────────────
def matrix_pencil(y, tau, threshold, K_max=12):
    """Matrix pencil decomposition with explicit threshold and rank cap."""
    M = len(y)
    L = M // 3
    # Hankel matrices
    Y1 = np.array([[y[i + j] for j in range(L)] for i in range(M - L)])
    Y2 = np.array([[y[i + j + 1] for j in range(L)] for i in range(M - L)])

    U, s, Vh = np.linalg.svd(Y1, full_matrices=False)

    # Threshold-based truncation
    cutoff = threshold * s[0]
    K_thresh = int(np.sum(s > cutoff))

    # BIC: penalise each additional mode by log(M)
    # Residual for K modes: approximate as sigma_{K+1}^2 * (M-L) * L
    bic_scores = []
    for k in range(1, min(K_thresh, K_max) + 1):
        Uk = U[:, :k]
        sk_diag = np.diag(s[:k])
        Vhk = Vh[:k, :]
        Y1_recon = Uk @ sk_diag @ Vhk
        residual = np.sum((Y1 - Y1_recon)**2)
        bic = M * np.log(residual / (M * L) + 1e-15) + k * np.log(M)
        bic_scores.append((k, bic))

    K_bic = min(bic_scores, key=lambda x: x[1])[0] if bic_scores else 1
    K_eff_thresh = min(K_thresh, K_max)

    return {
        'K_thresh': K_eff_thresh,
        'K_bic': K_bic,
        'singular_values': s[:min(K_max + 4, len(s))],
        'sigma_ratio': float(s[0] / s[1]) if len(s) > 1 else float('inf'),
    }


def sweep_library(library_path, thresholds, K_max_values):
    data = np.load(library_path, allow_pickle=True)

    print("=" * 80)
    print("K_eff THRESHOLD SENSITIVITY SWEEP")
    print("=" * 80)
    print(f"\nLibrary: {library_path}")
    print(f"Thresholds: {thresholds}")
    print(f"K_max values: {K_max_values}")
    print()

    results = {}

    for sys_name in data.files:
        recs = data[sys_name]
        print(f"\n{'─'*60}")
        print(f"SYSTEM: {sys_name} ({len(recs)} records)")
        print(f"{'─'*60}")

        # Header
        header = f"{'Record':<25}"
        for th in thresholds:
            header += f" K@{th:.0e}"
        header += "  K_BIC"
        print(header)

        system_results = []
        for r in recs:
            r = r.item() if hasattr(r, 'item') else r
            y = np.array(r.get('y', []))
            tau = np.linspace(0.01, 200.0, len(y))
            if len(y) < 10:
                continue

            # Record label
            if sys_name == 'OAT':
                label = f"N={r['N']:2d} G={r['Gamma']:.3f}"
            elif sys_name == 'SYK4':
                label = f"Nf={r['Nf']} b={r['beta']}"
            elif sys_name == 'XXZ':
                label = f"L={r['L']} W={r['W']}"
            elif sys_name == 'Dicke':
                label = f"N={r['N_spins']} g={r['g_ratio']}"
            else:
                label = str(r)[:20]

            row = f"  {label:<23}"
            rec_res = {}
            for th in thresholds:
                res = matrix_pencil(y, tau, threshold=th, K_max=max(K_max_values))
                k = res['K_thresh']
                row += f"  {k:4d}"
                rec_res[f'K@{th:.0e}'] = k

            # BIC at default threshold
            res_default = matrix_pencil(y, tau, threshold=1e-3, K_max=16)
            k_bic = res_default['K_bic']
            sv = res_default['singular_values']
            row += f"  {k_bic:5d}"
            print(row)
            rec_res['K_bic'] = k_bic
            rec_res['sigma_ratio'] = res_default['sigma_ratio']
            rec_res['label'] = label
            system_results.append(rec_res)

        results[sys_name] = system_results

    # ── Summary: is K_eff stable? ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)
    print(f"{'System':<10} {'K@1e-2 range':>15} {'K@1e-3 range':>15} {'K@1e-4 range':>15} {'K_BIC range':>12}")
    print("-" * 70)

    for sys_name, recs in results.items():
        if not recs:
            continue
        k2  = [r.get('K@1e-02', r.get('K@1e-2', '?')) for r in recs]
        k3  = [r.get('K@1e-03', r.get('K@1e-3', '?')) for r in recs]
        k4  = [r.get('K@1e-04', r.get('K@1e-4', '?')) for r in recs]
        kb  = [r.get('K_bic', '?') for r in recs]

        def rng_str(lst):
            nums = [x for x in lst if isinstance(x, int)]
            if not nums: return '?'
            return f"{min(nums)}–{max(nums)}"

        print(f"  {sys_name:<10} {rng_str(k2):>14} {rng_str(k3):>15} {rng_str(k4):>15} {rng_str(kb):>11}")

    print()
    print("INTERPRETATION:")
    print("  If K_eff(1e-4) << K_eff(1e-3) for SYK/XXZ → saturation ARTIFACT")
    print("  If K_eff(1e-4) ≈ K_eff(1e-3) for SYK/XXZ → likely PHYSICAL")
    print("  If K_BIC << K_thresh for SYK/XXZ → threshold is too permissive")
    print()

    return results


if __name__ == '__main__':
    lib = 'dlinoss_training/training_library.npz'
    thresholds = [1e-2, 1e-3, 1e-4, 1e-5]
    K_max_values = [4, 8, 12, 16]

    if not os.path.exists(lib):
        print(f"ERROR: {lib} not found. Run tpu_dlinoss_training_gen.py first.")
        sys.exit(1)

    results = sweep_library(lib, thresholds, K_max_values)
    np.savez('dlinoss_training/threshold_sweep.npz', **{k: v for k, v in results.items()})
    print("Saved to dlinoss_training/threshold_sweep.npz")
