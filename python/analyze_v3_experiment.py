"""deep_inspect_v3.py — Inspect the v3 experiment data and run full analysis."""
import json, pathlib, numpy as np
from scipy import stats

ALICE_DIR = pathlib.Path(r"mismo tiempo\holaMundo\chronos_v3_alice_scramble")
BOB_DIR   = pathlib.Path(r"mismo tiempo\holaMundo\chronos_v3_bob_passive")

def load_dataset(d):
    d = pathlib.Path(d)
    m = json.loads((d / 'manifest.json').read_text())
    records = []
    for cf in m['chunk_files']:
        npz = np.load(str(d / cf))
        timing = npz['stream_timing'].astype(np.float64)
        # Auto-recover JAX absolute timestamps if needed
        if timing.mean() > 10000:
            dt = np.diff(timing, axis=1)
            cov = np.cov(dt)
        else:
            cov = npz['covariance_matrix'].astype(np.float64)
        records.append({
            't':   float(npz['host_time_mid']),
            'lam': float(npz['lambda_val']),
            'cov': cov,
            'timing': timing,
            'sizes': npz.get('stream_sizes'),
        })
    records.sort(key=lambda r: r['t'])
    return records, m

print("=" * 65)
print("V3 DATA INSPECTION")
print("=" * 65)

alice_chunks, ma = load_dataset(ALICE_DIR)
bob_chunks,   mb = load_dataset(BOB_DIR)

for label, chunks, m in [("ALICE", alice_chunks, ma), ("BOB", bob_chunks, mb)]:
    timing_all = np.vstack([r['timing'] for r in chunks])
    lams = np.array([r['lam'] for r in chunks])
    print(f"\n[{label}]")
    print(f"  Role:       {m['role']}")
    print(f"  Schema:     {m.get('schema_version','?')}")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  t0 UTC:     {m['t0_utc']:.1f}")
    print(f"  IP:         {m['infrastructure'].get('ip')}")
    print(f"  TZ:         {m['infrastructure'].get('timezone')}")
    print(f"  Zone:       {m['infrastructure'].get('gcp_zone')}")
    print(f"  Timing shape (first chunk): {chunks[0]['timing'].shape}")
    print(f"  Timing mean: {timing_all.mean():.3f}ms  std: {timing_all.std():.3f}ms")
    loop_hz = 1000.0 / timing_all.mean() if timing_all.mean() > 0 else 0
    print(f"  Loop rate:  ~{loop_hz:.1f} Hz")
    print(f"  Lambda range: [{lams.min():.3f}, {lams.max():.3f}]  ON frac: {(lams>0.5).mean()*100:.0f}%")

dt = abs(ma['t0_utc'] - mb['t0_utc'])
print(f"\nAlignment: |t0_alice - t0_bob| = {dt:.2f}s  ({'OK' if dt < 30 else 'WARNING: >30s'})")

# ── Gaussian off-diagonal score ─────────────────────────────────────────────
def g_score(C):
    diag_mask = np.eye(C.shape[0], dtype=bool)
    off  = C.copy(); off[diag_mask]   = 0
    diag = C.copy(); diag[~diag_mask] = 0
    nd = np.linalg.norm(diag, 'fro')
    return np.linalg.norm(off, 'fro') / nd if nd > 1e-12 else 0.0

def frobenius_norm(C):
    return np.linalg.norm(C, 'fro')

def normalise_cov(C):
    n = frobenius_norm(C)
    return C / n if n > 1e-12 else C

alice_t0 = ma['t0_utc']
cycle_s     = ma['cycle_s']
ramp_up_s   = ma['ramp_up_s']
on_s        = ma['on_s']
ramp_down_s = ma['ramp_down_s']

def alice_lambda_at(t):
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

print("\n" + "=" * 65)
print("GAUSSIAN OFF-DIAGONAL SIGNATURE TEST")
print("=" * 65)

bob_scores = []
bob_lams_from_alice = []
for r in bob_chunks:
    g = g_score(r['cov'])
    lam_a = alice_lambda_at(r['t'])
    bob_scores.append(g)
    bob_lams_from_alice.append(lam_a)

scores  = np.array(bob_scores)
a_lams  = np.array(bob_lams_from_alice)
on_mask = a_lams > 0.5

g_on  = scores[on_mask]
g_off = scores[~on_mask]

print(f"Bob chunks: {len(scores)}  ON: {on_mask.sum()}  OFF: {(~on_mask).sum()}")
print(f"G_score ON  mean={g_on.mean():.4f}  std={g_on.std():.4f}")
print(f"G_score OFF mean={g_off.mean():.4f}  std={g_off.std():.4f}")
print(f"Delta (ON-OFF): {g_on.mean()-g_off.mean():.4f}")

t_stat, p_t = stats.ttest_ind(g_on, g_off, equal_var=False)
print(f"t-test: t={t_stat:.3f}  p={p_t:.4f}  {'SIGNIFICANT ***' if p_t<0.05 else 'not significant'}")

# Permutation test
rng = np.random.default_rng(42)
obs_delta = g_on.mean() - g_off.mean()
perm_deltas = np.zeros(10000)
for i in range(10000):
    sh = rng.permutation(a_lams)
    on_p  = scores[sh > 0.5]
    off_p = scores[sh <= 0.5]
    if len(on_p) > 0 and len(off_p) > 0:
        perm_deltas[i] = on_p.mean() - off_p.mean()
p_perm = (perm_deltas >= obs_delta).mean()
print(f"Permutation test: p={p_perm:.4f}  {'SIGNIFICANT ***' if p_perm<0.05 else 'not significant'}")

r_pearson, p_pearson = stats.pearsonr(a_lams, scores)
print(f"Pearson r(lambda_alice, G_score_bob): r={r_pearson:.4f}  p={p_pearson:.4f}  {'*** tracking' if p_pearson<0.05 else 'no correlation'}")

# ── OTOC Cross-Correlation F(tau) ────────────────────────────────────────────
print("\n" + "=" * 65)
print("OTOC FROBENIUS CROSS-CORRELATION F(tau)")
print("=" * 65)

alice_t_arr   = np.array([r['t'] for r in alice_chunks])
alice_cov_arr = np.array([normalise_cov(r['cov']) for r in alice_chunks])
alice_lam_arr = np.array([r['lam'] for r in alice_chunks])
bob_t_arr     = np.array([r['t'] for r in bob_chunks])
bob_cov_arr   = np.array([normalise_cov(r['cov']) for r in bob_chunks])

lags = np.arange(-120, 125, 5)
F    = np.zeros(len(lags))
for i, lag in enumerate(lags):
    scores_lag = []
    for j, at in enumerate(alice_t_arr):
        target_t = at + lag
        idx = np.argmin(np.abs(bob_t_arr - target_t))
        if abs(bob_t_arr[idx] - target_t) <= 7.5:
            scores_lag.append(float(np.sum(alice_cov_arr[j] * bob_cov_arr[idx])))
    F[i] = np.mean(scores_lag) if scores_lag else 0.0

best_i   = np.argmax(F)
best_lag = lags[best_i]
zero_F   = F[np.argmin(np.abs(lags))]
best_F   = F[best_i]
hp_F_pos = F[np.argmin(np.abs(lags - 40))]
hp_F_neg = F[np.argmin(np.abs(lags + 40))]

print(f"F(0) null baseline:      {zero_F:.4f}")
print(f"Peak F:                  {best_F:.4f}  at tau={best_lag:+.0f}s")
print(f"Peak/Null ratio:         {best_F/zero_F:.4f}x" if abs(zero_F)>1e-9 else "zero_F~0")
print(f"F(+40s) HP predicted:    {hp_F_pos:.4f}")
print(f"F(-40s) HP predicted:    {hp_F_neg:.4f}")

# Condition F on Alice ON vs OFF
F_on_vals, F_off_vals = [], []
for j, at in enumerate(alice_t_arr):
    target_t = at + best_lag
    idx = np.argmin(np.abs(bob_t_arr - target_t))
    if abs(bob_t_arr[idx] - target_t) <= 7.5:
        val = float(np.sum(alice_cov_arr[j] * bob_cov_arr[idx]))
        (F_on_vals if alice_lam_arr[j] > 0.5 else F_off_vals).append(val)

if F_on_vals and F_off_vals:
    t_f, p_f = stats.ttest_ind(F_on_vals, F_off_vals)
    print(f"F at best lag, ON:  mean={np.mean(F_on_vals):.4f}  n={len(F_on_vals)}")
    print(f"F at best lag, OFF: mean={np.mean(F_off_vals):.4f}  n={len(F_off_vals)}")
    print(f"t={t_f:.3f}  p={p_f:.4f}  {'SIGNIFICANT ***' if p_f<0.05 else 'not significant'}")

# ── Recoherence windows ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RECOHERENCE WINDOW ANALYSIS")
print("  (Moments where lambda transitions from ON->OFF: decoherence edge)")
print("=" * 65)

# Find ramp-down windows (lambda falling from ~1 to ~0) = decoherence edge
alice_t_rel = alice_t_arr - alice_t0
lam_arr_t   = alice_lam_arr

recoherence_events = []
for i in range(1, len(lam_arr_t)):
    prev_lam = lam_arr_t[i-1]
    curr_lam = lam_arr_t[i]
    # Transition: ON->ramp-down (lambda dropping from >0.9)
    if prev_lam > 0.9 and curr_lam < 0.9:
        t_event = alice_t_arr[i]
        # Measure Bob G-score in window around this moment
        window_scores = []
        for r in bob_chunks:
            if abs(r['t'] - t_event) < 60:  # 60s window
                window_scores.append((r['t'] - t_event, g_score(r['cov'])))
        if window_scores:
            recoherence_events.append({'t': t_event, 'window': window_scores})
            print(f"  Recoherence edge at t+{alice_t_rel[i]:.0f}s  (Alice lam: {prev_lam:.2f}->{curr_lam:.2f})")
            pre  = [s for dt_e, s in window_scores if -60 <= dt_e < 0]
            post = [s for dt_e, s in window_scores if 0 <= dt_e <= 60]
            if pre and post:
                print(f"    Bob G-score pre:  {np.mean(pre):.4f}")
                print(f"    Bob G-score post: {np.mean(post):.4f}")
                print(f"    Delta post-pre:   {np.mean(post)-np.mean(pre):.4f}  {'(+) Bob INCREASED coherence after OFF edge' if np.mean(post)>np.mean(pre) else '(-) Bob decreased after OFF edge'}")

if not recoherence_events:
    print("  No ON->OFF transitions detected in Alice chunks.")
    print(f"  (Alice only had {len(alice_chunks)} chunks, lambda range [{lam_arr_t.min():.2f},{lam_arr_t.max():.2f}])")

# ── Save all results ─────────────────────────────────────────────────────────
pathlib.Path('resonance_results').mkdir(exist_ok=True)
results = {
    'schema': 'v3_analysis',
    'alice_chunks': len(alice_chunks),
    'bob_chunks': len(bob_chunks),
    'alice_tz': ma['infrastructure'].get('timezone'),
    'bob_tz':   mb['infrastructure'].get('timezone'),
    'alice_ip': ma['infrastructure'].get('ip'),
    'bob_ip':   mb['infrastructure'].get('ip'),
    'start_offset_s': float(dt),
    'bob_loop_hz': float(1000.0 / np.vstack([r['timing'] for r in bob_chunks]).mean()),
    'alice_loop_hz': float(1000.0 / np.vstack([r['timing'] for r in alice_chunks]).mean()),
    'g_score_on': float(g_on.mean()) if len(g_on)>0 else None,
    'g_score_off': float(g_off.mean()) if len(g_off)>0 else None,
    'g_delta': float(g_on.mean()-g_off.mean()) if len(g_on)>0 and len(g_off)>0 else None,
    'p_ttest': float(p_t),
    'p_perm': float(p_perm),
    'pearson_r': float(r_pearson),
    'pearson_p': float(p_pearson),
    'otoc_best_lag': float(best_lag),
    'otoc_best_F': float(best_F),
    'otoc_null_F': float(zero_F),
    'otoc_ratio': float(best_F/zero_F) if abs(zero_F)>1e-9 else None,
    'otoc_hp_pos': float(hp_F_pos),
    'otoc_hp_neg': float(hp_F_neg),
    'F_lags': lags.tolist(),
    'F_vals': F.tolist(),
    'recoherence_n_events': len(recoherence_events),
}
pathlib.Path('resonance_results/v3_full_analysis.json').write_text(
    json.dumps(results, indent=2))
print(f"\nSaved: resonance_results/v3_full_analysis.json")
print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
