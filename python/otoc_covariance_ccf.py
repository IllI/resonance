"""
otoc_covariance_ccf.py
======================
Computes the OTOC-analog cross-correlation between Alice's and Bob's
4x4 covariance matrices over lag τ ∈ [-120s, +120s].

F(τ) = mean_t [ <C_alice(t), C_bob(t+τ)>_F ]
     = mean_t [ Frobenius inner product of normalized covariance matrices ]

Usage:
    python python/otoc_covariance_ccf.py \
        --alice chronos_v2_alice_scramble/ \
        --bob   chronos_v2_bob_passive/
"""
import argparse, json, pathlib, sys
import numpy as np
from scipy import stats

def load_chunks(d):
    d = pathlib.Path(d)
    m = json.loads((d / 'manifest.json').read_text())
    records = []
    for cf in m['chunk_files']:
        npz = np.load(str(d / cf))
        timing = npz['stream_timing'].astype(np.float64)
        # Auto-recover JAX absolute timestamps
        if timing.mean() > 10000:
            dt = np.diff(timing, axis=1)
            cov = np.cov(dt)
        else:
            cov = npz['covariance_matrix'].astype(np.float64)
        
        records.append({
            't':   float(npz['host_time_mid']),
            'lam': float(npz['lambda_val']),
            'cov': cov,
        })
    records.sort(key=lambda r: r['t'])
    return records, m

def normalise_cov(C):
    """Frobenius-normalise so ||C||_F = 1."""
    n = np.linalg.norm(C, 'fro')
    return C / n if n > 1e-12 else C

def frob_inner(A, B):
    return float(np.sum(A * B))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alice', required=True)
    p.add_argument('--bob',   required=True)
    p.add_argument('--lag-max', type=float, default=120.0)
    p.add_argument('--lag-step', type=float, default=5.0)
    args = p.parse_args()

    alice_chunks, ma = load_chunks(args.alice)
    bob_chunks,   mb = load_chunks(args.bob)

    print('='*60)
    print('OTOC COVARIANCE CROSS-CORRELATION  F(τ)')
    print('='*60)
    print(f'Alice chunks: {len(alice_chunks)}  t0={ma["t0_utc"]:.1f}')
    print(f'Bob   chunks: {len(bob_chunks)}    t0={mb["t0_utc"]:.1f}')

    t0_offset = mb['t0_utc'] - ma['t0_utc']
    print(f'Bob started {t0_offset:+.1f}s relative to Alice')
    print()

    # Build time-indexed arrays
    alice_t   = np.array([r['t'] for r in alice_chunks])
    alice_cov = np.array([normalise_cov(r['cov']) for r in alice_chunks])  # (N,4,4)
    alice_lam = np.array([r['lam'] for r in alice_chunks])

    bob_t   = np.array([r['t'] for r in bob_chunks])
    bob_cov = np.array([normalise_cov(r['cov']) for r in bob_chunks])
    bob_lam = np.array([r['lam'] for r in bob_chunks])

    lags = np.arange(-args.lag_max, args.lag_max + args.lag_step, args.lag_step)
    F    = np.zeros(len(lags))
    N_   = np.zeros(len(lags), dtype=int)

    for i, lag in enumerate(lags):
        scores = []
        for j, at in enumerate(alice_t):
            target_t = at + lag
            # Find nearest Bob chunk
            idx = np.argmin(np.abs(bob_t - target_t))
            if abs(bob_t[idx] - target_t) <= args.lag_step * 1.5:
                scores.append(frob_inner(alice_cov[j], bob_cov[idx]))
        if scores:
            F[i]  = np.mean(scores)
            N_[i] = len(scores)

    best_i   = np.argmax(F)
    best_lag = lags[best_i]
    best_F   = F[best_i]
    zero_F   = F[np.argmin(np.abs(lags))]

    print(f'F(τ) summary:')
    print(f'  Null (lag=0):    F = {zero_F:.4f}')
    print(f'  Peak:            F = {best_F:.4f}  at lag = {best_lag:+.0f}s')
    print(f'  Peak/Null ratio: {best_F/zero_F:.3f}x' if abs(zero_F) > 1e-9 else '  zero_F ~ 0')
    print()

    # Hayden-Preskill predicted lag (4 streams, ~matmul ops)
    hp_lag = 40.0
    hp_F   = F[np.argmin(np.abs(lags - hp_lag))]
    hp_neg = F[np.argmin(np.abs(lags + hp_lag))]
    print(f'Hayden-Preskill predicted lag (±{hp_lag:.0f}s):')
    print(f'  F(+{hp_lag:.0f}s) = {hp_F:.4f}')
    print(f'  F(-{hp_lag:.0f}s) = {hp_neg:.4f}')
    print()

    # Condition on Alice ON (λ > 0.5)
    on_idx  = np.where(alice_lam > 0.5)[0]
    off_idx = np.where(alice_lam <= 0.5)[0]
    F_on, F_off = [], []
    for j in on_idx:
        idx = np.argmin(np.abs(bob_t - (alice_t[j] + best_lag)))
        if abs(bob_t[idx] - (alice_t[j] + best_lag)) <= args.lag_step * 1.5:
            F_on.append(frob_inner(alice_cov[j], bob_cov[idx]))
    for j in off_idx:
        idx = np.argmin(np.abs(bob_t - (alice_t[j] + best_lag)))
        if abs(bob_t[idx] - (alice_t[j] + best_lag)) <= args.lag_step * 1.5:
            F_off.append(frob_inner(alice_cov[j], bob_cov[idx]))

    if F_on and F_off:
        t_stat, p_val = stats.ttest_ind(F_on, F_off)
        print(f'F at best lag — conditioned on Alice schedule:')
        print(f'  ON  (λ>0.5):  mean={np.mean(F_on):.4f}  n={len(F_on)}')
        print(f'  OFF (λ≤0.5):  mean={np.mean(F_off):.4f}  n={len(F_off)}')
        print(f'  t={t_stat:.3f}  p={p_val:.4f}  '
              f'{"SIGNIFICANT ***" if p_val < 0.05 else "not significant"}')

    # Save results
    out = {
        'lags': lags.tolist(),
        'F': F.tolist(),
        'N_pairs': N_.tolist(),
        'best_lag_s': float(best_lag),
        'best_F': float(best_F),
        'zero_F': float(zero_F),
        'hp_F_pos': float(hp_F),
        'hp_F_neg': float(hp_neg),
        'F_on_mean':  float(np.mean(F_on))  if F_on  else None,
        'F_off_mean': float(np.mean(F_off)) if F_off else None,
    }
    out_path = pathlib.Path('resonance_results/otoc_covariance_ccf.json')
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved: {out_path}')

if __name__ == '__main__':
    main()
