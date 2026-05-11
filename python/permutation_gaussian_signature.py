"""
permutation_gaussian_signature.py
===================================
Tests whether Bob's 4x4 covariance matrices show stronger Gaussian
off-diagonal structure during Alice's ON phase (λ > 0.5) vs OFF phase.

The Gaussian off-diagonal signature is:
    G_score(C) = ||C_offdiag||_F / ||C_diag||_F
where C_offdiag has zeros on the diagonal and C_diag has zeros off-diagonal.

If Alice's Gaussian injection propagates to Bob, G_score should be higher
during Alice's ON windows than OFF windows.

Usage:
    python python/permutation_gaussian_signature.py \
        --bob   chronos_v2_bob_passive/ \
        --alice-manifest chronos_v2_alice_scramble/manifest.json
"""
import argparse, json, pathlib
import numpy as np
from scipy import stats

def gaussian_off_diagonal_score(C):
    """Ratio of off-diagonal to diagonal Frobenius norm — the 'Bell shape' signature."""
    diag_mask = np.eye(C.shape[0], dtype=bool)
    off  = C.copy(); off[diag_mask]  = 0.0
    diag = C.copy(); diag[~diag_mask] = 0.0
    n_off  = np.linalg.norm(off,  'fro')
    n_diag = np.linalg.norm(diag, 'fro')
    return n_off / n_diag if n_diag > 1e-12 else 0.0

def load_bob(d):
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

def alice_lambda_at(t, alice_t0, cycle_s, ramp_up_s, on_s, ramp_down_s):
    """Reconstruct Alice's λ(t) from her schedule parameters."""
    elapsed = t - alice_t0
    if elapsed < 0: return 0.0
    phase = elapsed % cycle_s
    if phase < ramp_up_s:
        return 0.5 * (1 - np.cos(np.pi * phase / ramp_up_s))
    elif phase < ramp_up_s + on_s:
        return 1.0
    elif phase < ramp_up_s + on_s + ramp_down_s:
        p = (phase - ramp_up_s - on_s) / ramp_down_s
        return 0.5 * (1 + np.cos(np.pi * p))
    return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bob',              required=True)
    p.add_argument('--alice-manifest',   required=True)
    p.add_argument('--n-perm', type=int, default=10000)
    args = p.parse_args()

    alice_m = json.loads(pathlib.Path(args.alice_manifest).read_text())
    bob_chunks, bob_m = load_bob(args.bob)

    # Reconstruct Alice's λ at each Bob chunk timestamp
    alice_t0    = alice_m['t0_utc']
    cycle_s     = alice_m['cycle_s']
    ramp_up_s   = alice_m['ramp_up_s']
    on_s        = alice_m['on_s']
    ramp_down_s = alice_m['ramp_down_s']

    # Compute G_score for every Bob chunk
    scores = []
    alice_lams = []
    for r in bob_chunks:
        g = gaussian_off_diagonal_score(r['cov'])
        lam_a = alice_lambda_at(r['t'], alice_t0, cycle_s, ramp_up_s, on_s, ramp_down_s)
        scores.append(g)
        alice_lams.append(lam_a)

    scores     = np.array(scores)
    alice_lams = np.array(alice_lams)

    on_mask  = alice_lams > 0.5
    off_mask = ~on_mask

    g_on  = scores[on_mask]
    g_off = scores[off_mask]

    print('='*60)
    print('GAUSSIAN OFF-DIAGONAL SIGNATURE TEST')
    print('='*60)
    print(f'Bob chunks:  {len(scores)}')
    print(f'Alice ON  (λ>0.5): {on_mask.sum()} chunks')
    print(f'Alice OFF (λ≤0.5): {off_mask.sum()} chunks')
    print()
    print(f'G_score (off-diag/diag ratio):')
    print(f'  ON  windows: mean={g_on.mean():.4f}  std={g_on.std():.4f}')
    print(f'  OFF windows: mean={g_off.mean():.4f}  std={g_off.std():.4f}')
    print(f'  Delta (ON - OFF): {g_on.mean() - g_off.mean():.4f}')
    print()

    # t-test
    t_stat, p_t = stats.ttest_ind(g_on, g_off, equal_var=False)
    print(f't-test: t={t_stat:.3f}  p={p_t:.4f}  '
          f'{"SIGNIFICANT ***" if p_t < 0.05 else "not significant"}')

    # Permutation test — shuffle the ON/OFF labels
    obs_delta = g_on.mean() - g_off.mean()
    rng = np.random.default_rng(42)
    perm_deltas = np.zeros(args.n_perm)
    for i in range(args.n_perm):
        shuffled = rng.permutation(alice_lams)
        on_s_p   = scores[shuffled > 0.5]
        off_s_p  = scores[shuffled <= 0.5]
        if len(on_s_p) > 0 and len(off_s_p) > 0:
            perm_deltas[i] = on_s_p.mean() - off_s_p.mean()

    p_perm = (perm_deltas >= obs_delta).mean()
    print()
    print(f'Permutation test (N={args.n_perm}):')
    print(f'  Observed delta:  {obs_delta:.4f}')
    print(f'  Perm mean delta: {perm_deltas.mean():.4f} ± {perm_deltas.std():.4f}')
    print(f'  p-value:         {p_perm:.4f}  '
          f'{"SIGNIFICANT ***" if p_perm < 0.05 else "not significant"}')
    print()

    # Also check: does the G_score track λ(t) over time?
    if len(scores) > 10:
        r_pearson, p_pearson = stats.pearsonr(alice_lams, scores)
        print(f'Pearson r(λ_alice, G_score_bob): r={r_pearson:.4f}  p={p_pearson:.4f}')
        print(f'  {"λ tracks G_score *** — injection propagated" if p_pearson < 0.05 else "no λ-G_score correlation"}')

    # Save
    out = {
        'n_chunks': int(len(scores)),
        'n_on':     int(on_mask.sum()),
        'n_off':    int(off_mask.sum()),
        'g_on_mean':  float(g_on.mean()),
        'g_off_mean': float(g_off.mean()),
        'obs_delta':  float(obs_delta),
        'p_ttest':    float(p_t),
        'p_perm':     float(p_perm),
        'pearson_r':  float(r_pearson) if len(scores) > 10 else None,
        'pearson_p':  float(p_pearson) if len(scores) > 10 else None,
        'significant': bool(p_perm < 0.05),
    }
    out_path = pathlib.Path('resonance_results/gaussian_signature_test.json')
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved: {out_path}')

if __name__ == '__main__':
    main()
