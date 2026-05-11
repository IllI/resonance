"""
analyze_v4_experiment.py
========================
Full analysis for Chronos v4 — Shared Coherent State Protocol.

Primary observable: eigenvector alignment between Alice and Bob's 
principal EM modes (dominant eigenvector of 4x4 timing covariance).

Usage:
    python python/analyze_v4_experiment.py \
        --alice "mismo tiempo/holaMundo/chronos_v4_alice_scramble" \
        --bob   "mismo tiempo/holaMundo/chronos_v4_bob_passive"
"""
import argparse, json, pathlib
import numpy as np
from scipy import stats

def load_v4(d):
    d = pathlib.Path(d)
    m = json.loads((d / 'manifest.json').read_text())
    recs = []
    for cf in m['chunk_files']:
        npz = np.load(str(d / cf))
        timing = npz['stream_timing'].astype(np.float64)
        if timing.mean() > 10000:
            timing = np.diff(timing, axis=1) * 1000.0
        cov = np.cov(timing).astype(np.float64) if 'principal_eigvec' not in npz.files \
              else npz['covariance_matrix'].astype(np.float64)
        # Principal eigenvec: use saved if present, else compute
        if 'principal_eigvec' in npz.files:
            pv = npz['principal_eigvec'].astype(np.float64)
        else:
            _, vecs = np.linalg.eigh(cov)
            pv = vecs[:, -1]
        # Eigenvalue entropy
        if 'eigenvalue_entropy' in npz.files:
            H = float(npz['eigenvalue_entropy'])
        else:
            ev = np.linalg.eigvalsh(cov)
            ev = ev[ev > 1e-12]; ev /= ev.sum()
            H = float(-np.sum(ev * np.log(ev + 1e-12)))
        recs.append({
            't':   float(npz['host_time_mid']),
            'lam': float(npz['lambda_val']),
            'cov': cov, 'pv': pv, 'H': H,
            'timing': timing,
        })
    recs.sort(key=lambda r: r['t'])
    return recs, m

def alice_lambda_at(t, m):
    t0   = m['t0_utc']
    C    = m['cycle_s'];  R = m['ramp_up_s']
    O    = m['on_s'];     D = m['ramp_down_s']
    el   = t - t0
    if el < 0: return 0.0
    ph = el % C
    if ph < R:     return 0.5*(1-np.cos(np.pi*ph/R))
    elif ph < R+O: return 1.0
    elif ph < R+O+D:
        p = (ph-R-O)/D; return 0.5*(1+np.cos(np.pi*p))
    return 0.0

def alignment(v1, v2):
    return float(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))**2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alice', required=True)
    p.add_argument('--bob',   required=True)
    p.add_argument('--n-perm', type=int, default=10000)
    args = p.parse_args()

    alice, ma = load_v4(args.alice)
    bob,   mb = load_v4(args.bob)

    print('='*65)
    print('CHRONOS v4 — SHARED COHERENT STATE ANALYSIS')
    print('='*65)

    # ── Data quality ────────────────────────────────────────────────────────
    alice_hz = len(alice) / ma.get('duration_s', 900)
    bob_hz   = len(bob)   / mb.get('duration_s', 900)
    print(f'\nAlice: {len(alice)} chunks  natural_hz={ma.get("natural_hz","?")}  tz={ma["infrastructure"].get("timezone")}  ip={ma["infrastructure"].get("ip")}')
    print(f'Bob:   {len(bob)} chunks    natural_hz={mb.get("natural_hz","?")}  tz={mb["infrastructure"].get("timezone")}  ip={mb["infrastructure"].get("ip")}')
    dt_t0 = abs(ma['t0_utc'] - mb['t0_utc'])
    print(f'Session start offset: {dt_t0:.2f}s  ({"OK" if dt_t0<30 else "WARNING >30s"})')
    print(f'Null alignment: Alice={ma.get("null_alignment","?")}  Bob={mb.get("null_alignment","?")}')

    bob_t   = np.array([r['t'] for r in bob])
    bob_pv  = np.array([r['pv'] for r in bob])
    bob_H   = np.array([r['H'] for r in bob])
    bob_lam_from_alice = np.array([alice_lambda_at(r['t'], ma) for r in bob])

    # Cross-alignment between Alice and Bob principal eigenvectors
    alice_t  = np.array([r['t'] for r in alice])
    alice_pv = np.array([r['pv'] for r in alice])
    alice_lam = np.array([r['lam'] for r in alice])

    # Interpolate Alice principal eigenvec at each Bob timestamp
    cross_alignments = []
    cross_t = []
    for r in bob:
        idx = np.argmin(np.abs(alice_t - r['t']))
        if abs(alice_t[idx] - r['t']) < 10.0:  # within 10s
            a = alignment(r['pv'], alice_pv[idx])
            cross_alignments.append(a)
            cross_t.append(r['t'])
    cross_alignments = np.array(cross_alignments)
    cross_t = np.array(cross_t)
    cross_lam = np.array([alice_lambda_at(t, ma) for t in cross_t])

    print(f'\n{"="*65}')
    print('EIGENVECTOR ALIGNMENT TEST (primary)')
    print('='*65)
    print(f'Cross-aligned pairs: {len(cross_alignments)}')
    on_mask  = cross_lam > 0.5
    off_mask = ~on_mask
    a_on  = cross_alignments[on_mask]
    a_off = cross_alignments[off_mask]
    print(f'Alignment ON  (lam>0.5): mean={a_on.mean():.4f}  std={a_on.std():.4f}  n={len(a_on)}')
    print(f'Alignment OFF (lam<=0.5): mean={a_off.mean():.4f}  std={a_off.std():.4f}  n={len(a_off)}')
    print(f'Delta (ON-OFF): {a_on.mean()-a_off.mean():.4f}')
    t_s, p_t = stats.ttest_ind(a_on, a_off, equal_var=False)
    print(f't-test: t={t_s:.3f}  p={p_t:.4f}  {"SIGNIFICANT ***" if p_t<0.05 else "not significant"}')

    # Permutation test
    rng = np.random.default_rng(42)
    obs = a_on.mean() - a_off.mean()
    perms = np.zeros(args.n_perm)
    for i in range(args.n_perm):
        sh = rng.permutation(cross_lam)
        on_p = cross_alignments[sh>0.5]; off_p = cross_alignments[sh<=0.5]
        perms[i] = on_p.mean()-off_p.mean() if len(on_p)>0 and len(off_p)>0 else 0
    p_perm = (perms >= obs).mean()
    print(f'Permutation test: p={p_perm:.4f}  {"SIGNIFICANT ***" if p_perm<0.05 else "not significant"}')

    r_p, p_p = stats.pearsonr(cross_lam, cross_alignments)
    print(f'Pearson r(lam_alice, alignment): r={r_p:.4f}  p={p_p:.4f}  {"*** tracking" if p_p<0.05 else "no correlation"}')

    # ── Eigenvalue entropy vs lambda ─────────────────────────────────────────
    print(f'\n{"="*65}')
    print('EIGENVALUE ENTROPY vs LAMBDA (system ordering)')
    print('='*65)
    bob_H_on  = bob_H[bob_lam_from_alice > 0.5]
    bob_H_off = bob_H[bob_lam_from_alice <= 0.5]
    print(f'Bob eigenvalue entropy ON:  mean={bob_H_on.mean():.4f}  std={bob_H_on.std():.4f}')
    print(f'Bob eigenvalue entropy OFF: mean={bob_H_off.mean():.4f}  std={bob_H_off.std():.4f}')
    t_H, p_H = stats.ttest_ind(bob_H_on, bob_H_off, equal_var=False)
    print(f't-test: t={t_H:.3f}  p={p_H:.4f}  {"SIGNIFICANT ***" if p_H<0.05 else "not significant"}')
    direction = "DECREASED (more ordered)" if bob_H_on.mean() < bob_H_off.mean() else "INCREASED (more disordered)"
    print(f'Entropy during Alice ON: {direction}')

    # ── Retrocausal CCF ──────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('RETROCAUSAL CCF: alignment vs FUTURE lambda')
    print('='*65)
    lags = np.arange(-120, 125, 5)
    CCF  = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        pairs = []
        for j, t_b in enumerate(cross_t):
            target = t_b - lag  # shift Bob forward by lag to test retrocausal
            idx = np.argmin(np.abs(alice_t - target))
            if abs(alice_t[idx] - target) < 7.5:
                pairs.append((cross_alignments[j], alice_lam[idx]))
        if len(pairs) > 3:
            a_arr = np.array([x[0] for x in pairs])
            l_arr = np.array([x[1] for x in pairs])
            if a_arr.std() > 1e-9 and l_arr.std() > 1e-9:
                CCF[i] = np.corrcoef(a_arr, l_arr)[0,1]
    best_lag_i = np.argmax(np.abs(CCF))
    best_lag   = lags[best_lag_i]
    print(f'CCF peak: r={CCF[best_lag_i]:.4f}  at lag={best_lag:+.0f}s')
    print(f'CCF at lag=0 (classical expected): {CCF[np.argmin(np.abs(lags))]:.4f}')
    if best_lag < 0:
        print(f'>>> RETROCAUSAL SIGNATURE: Bob alignment peak at lag={best_lag:.0f}s (Bob sees Alice BEFORE she injects)')
    else:
        print(f'>>> Classical or null: peak at lag={best_lag:.0f}s >= 0')

    # ── Recoherence edges ────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('RECOHERENCE EDGE ANALYSIS')
    print('='*65)
    for i in range(1, len(alice_lam)):
        if alice_lam[i-1] > 0.9 and alice_lam[i] < 0.9:
            t_edge = alice_t[i]
            rel    = t_edge - ma['t0_utc']
            pre  = [cross_alignments[j] for j,t in enumerate(cross_t) if -60<=t-t_edge<0]
            post = [cross_alignments[j] for j,t in enumerate(cross_t) if 0<=t-t_edge<=60]
            pre_H  = [bob_H[j] for j in range(len(bob)) if -60<=bob_t[j]-t_edge<0]
            post_H = [bob_H[j] for j in range(len(bob)) if 0<=bob_t[j]-t_edge<=60]
            print(f'\n  ON->OFF edge at t={rel:.0f}s ({rel/60:.1f}min)')
            if pre and post:
                da = np.mean(post)-np.mean(pre)
                print(f'    Alignment pre:{np.mean(pre):.4f} -> post:{np.mean(post):.4f}  delta={da:+.4f}')
                print(f'    {"(+) RECOHERENCE: alignment INCREASED after OFF edge ***" if da>0 else "(-) alignment decreased (expected)"}')
            if pre_H and post_H:
                dH = np.mean(post_H)-np.mean(pre_H)
                print(f'    Entropy pre:{np.mean(pre_H):.4f} -> post:{np.mean(post_H):.4f}  delta={dH:+.4f}')
                print(f'    {"(-) Entropy DECREASED after OFF -> system self-orders ***" if dH<0 else "(+) entropy increased"}')

    # ── Save ─────────────────────────────────────────────────────────────────
    pathlib.Path('resonance_results').mkdir(exist_ok=True)
    out = {
        'schema': 'v4_analysis',
        'alice_chunks': len(alice), 'bob_chunks': len(bob),
        'start_offset_s': float(dt_t0),
        'cross_pairs': int(len(cross_alignments)),
        'alignment_on': float(a_on.mean()) if len(a_on)>0 else None,
        'alignment_off': float(a_off.mean()) if len(a_off)>0 else None,
        'alignment_delta': float(a_on.mean()-a_off.mean()) if len(a_on)>0 and len(a_off)>0 else None,
        'p_ttest': float(p_t), 'p_perm': float(p_perm),
        'pearson_r': float(r_p), 'pearson_p': float(p_p),
        'entropy_on': float(bob_H_on.mean()), 'entropy_off': float(bob_H_off.mean()),
        'p_entropy': float(p_H),
        'ccf_best_lag': float(best_lag), 'ccf_best_r': float(CCF[best_lag_i]),
        'ccf_lags': lags.tolist(), 'ccf_vals': CCF.tolist(),
        'retrocausal': bool(best_lag < 0 and abs(CCF[best_lag_i]) > 0.1),
    }
    pathlib.Path('resonance_results/v4_full_analysis.json').write_text(json.dumps(out, indent=2))
    print(f'\nSaved: resonance_results/v4_full_analysis.json')
    print('\n' + '='*65)
    print('ANALYSIS COMPLETE')
    print('='*65)

if __name__ == '__main__':
    main()
