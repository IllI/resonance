"""plot_v3_analysis.py — Generate visual summary of v3 experiment results."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, signal

ALICE_DIR = pathlib.Path(r"mismo tiempo\holaMundo\chronos_v3_alice_scramble")
BOB_DIR   = pathlib.Path(r"mismo tiempo\holaMundo\chronos_v3_bob_passive")

def load_dataset(d):
    d = pathlib.Path(d)
    m = json.loads((d / 'manifest.json').read_text())
    records = []
    for cf in m['chunk_files']:
        npz = np.load(str(d / cf))
        timing = npz['stream_timing'].astype(np.float64)
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
        })
    records.sort(key=lambda r: r['t'])
    return records, m

alice_chunks, ma = load_dataset(ALICE_DIR)
bob_chunks,   mb = load_dataset(BOB_DIR)

alice_t0    = ma['t0_utc']
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
    elif phase < ramp_up_s + on_s: return 1.0
    elif phase < ramp_up_s + on_s + ramp_down_s:
        p = (phase - ramp_up_s - on_s) / ramp_down_s
        return 0.5 * (1 + np.cos(np.pi * p))
    return 0.0

def g_score(C):
    diag_mask = np.eye(C.shape[0], dtype=bool)
    off  = C.copy(); off[diag_mask]   = 0
    diag = C.copy(); diag[~diag_mask] = 0
    nd = np.linalg.norm(diag,'fro')
    return np.linalg.norm(off,'fro') / nd if nd > 1e-12 else 0.0

def normalise_cov(C):
    n = np.linalg.norm(C,'fro')
    return C / n if n > 1e-12 else C

# Time series
bob_t_rel    = np.array([r['t'] - mb['t0_utc'] for r in bob_chunks])
alice_t_rel  = np.array([r['t'] - alice_t0      for r in alice_chunks])
bob_lams     = np.array([alice_lambda_at(r['t'])  for r in bob_chunks])
alice_lams   = np.array([r['lam'] for r in alice_chunks])

bob_g        = np.array([g_score(r['cov']) for r in bob_chunks])
alice_g      = np.array([g_score(r['cov']) for r in alice_chunks])

bob_var      = np.array([np.trace(r['cov']) for r in bob_chunks])
alice_var    = np.array([np.trace(r['cov']) for r in alice_chunks])

# Phase velocity: gradient of trace(cov) as proxy for QPC phase velocity
bob_pv   = np.gradient(bob_var,   bob_t_rel)
alice_pv = np.gradient(alice_var, alice_t_rel)

# Coherence proxy: mean off-diagonal / diagonal
bob_coh  = bob_g
alice_coh = alice_g

# OTOC
alice_t_arr   = np.array([r['t'] for r in alice_chunks])
bob_t_arr     = np.array([r['t'] for r in bob_chunks])
alice_cov_arr = np.array([normalise_cov(r['cov']) for r in alice_chunks])
bob_cov_arr   = np.array([normalise_cov(r['cov']) for r in bob_chunks])
lags = np.arange(-120, 125, 5)
F    = np.zeros(len(lags))
for i, lag in enumerate(lags):
    sc = []
    for j, at in enumerate(alice_t_arr):
        idx = np.argmin(np.abs(bob_t_arr - (at + lag)))
        if abs(bob_t_arr[idx] - (at + lag)) <= 7.5:
            sc.append(float(np.sum(alice_cov_arr[j] * bob_cov_arr[idx])))
    F[i] = np.mean(sc) if sc else 0.0

# Sliding window coherence correlation (20-chunk window)
W = 20
window_corr = np.zeros(len(bob_chunks) - W)
window_t    = np.zeros(len(bob_chunks) - W)
for i in range(len(window_corr)):
    bl = bob_lams[i:i+W]
    bg = bob_g[i:i+W]
    if bl.std() > 1e-6 and bg.std() > 1e-6:
        window_corr[i] = np.corrcoef(bl, bg)[0,1]
    window_t[i] = bob_t_rel[i + W//2]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20), facecolor='#0d1117')
fig.suptitle('Chronos v3 — IQSP Gaussian Scramble Analysis\nAlice (LA) ↔ Bob (Chicago) TPU  |  359 chunks × 2', 
             color='white', fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.32)

AX = lambda r, c: fig.add_subplot(gs[r, c])
COLORS = {
    'alice': '#ff6b9d',
    'bob':   '#00d4ff',
    'lambda':'#ffd700',
    'g_on':  '#00ff88',
    'g_off': '#ff4466',
    'otoc':  '#cc88ff',
    'grid':  '#333344',
}

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#1a1f2e')
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.title.set_color('white'); ax.title.set_fontsize(10); ax.title.set_fontweight('bold')
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel, color='#aaaacc', fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#aaaacc', fontsize=8)
    ax.grid(True, color=COLORS['grid'], linewidth=0.4, alpha=0.6)
    for sp in ax.spines.values(): sp.set_color('#333344')

# ── Panel 1: Lambda Schedule ─────────────────────────────────────────────────
ax1 = AX(0, 0)
ax1.fill_between(bob_t_rel/60, bob_lams, alpha=0.25, color=COLORS['lambda'])
ax1.plot(bob_t_rel/60, bob_lams, color=COLORS['lambda'], lw=1.5, label='λ(t) Alice schedule')
ax1.set_ylim(-0.05, 1.1)
style_ax(ax1, 'Alice λ-Schedule (as seen by Bob timestamps)', 'Time (min)', 'λ')
ax1.axhline(0.5, color='white', lw=0.5, ls='--', alpha=0.4)

# ── Panel 2: Bob G-score over time ───────────────────────────────────────────
ax2 = AX(0, 1)
on_mask  = bob_lams > 0.5
off_mask = ~on_mask
ax2.scatter(bob_t_rel[on_mask]/60,  bob_g[on_mask],  c=COLORS['g_on'],  s=6, alpha=0.6, label='Bob G-score (Alice ON)')
ax2.scatter(bob_t_rel[off_mask]/60, bob_g[off_mask], c=COLORS['g_off'], s=6, alpha=0.6, label='Bob G-score (Alice OFF)')
ax2.plot(bob_t_rel/60, np.convolve(bob_g, np.ones(15)/15, 'same'), 'white', lw=1.5, alpha=0.8, label='15-chunk MA')
style_ax(ax2, "Bob's Off-Diagonal G-Score Over Time", 'Time (min)', 'G-score (off/diag norm)')
ax2.legend(fontsize=7, loc='upper right', framealpha=0.3)

# ── Panel 3: Trace(cov) — QPC proxy ──────────────────────────────────────────
ax3 = AX(1, 0)
ax3.plot(alice_t_rel/60, alice_var, color=COLORS['alice'], lw=1.2, alpha=0.8, label='Alice Tr(Cov)')
ax3_b = ax3.twinx()
ax3_b.plot(bob_t_rel/60, bob_var, color=COLORS['bob'], lw=1.2, alpha=0.8, label='Bob Tr(Cov)')
ax3_b.tick_params(colors=COLORS['bob'], labelsize=8)
ax3_b.set_ylabel('Bob Tr(Cov) (ms²)', color=COLORS['bob'], fontsize=8)
style_ax(ax3, 'Trace(Cov) — QPC Phase Proxy', 'Time (min)', 'Alice Tr(Cov) (ms²)')
ax3.legend(fontsize=7, loc='upper left', framealpha=0.3)
ax3_b.legend(fontsize=7, loc='upper right', framealpha=0.3)

# ── Panel 4: Phase Velocity ───────────────────────────────────────────────────
ax4 = AX(1, 1)
ax4.plot(alice_t_rel/60, alice_pv, color=COLORS['alice'], lw=1.0, alpha=0.7, label='Alice dTr/dt')
ax4.plot(bob_t_rel/60,   bob_pv,   color=COLORS['bob'],   lw=1.0, alpha=0.7, label='Bob dTr/dt')
ax4.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
style_ax(ax4, 'Phase Velocity (d/dt Tr(Cov))', 'Time (min)', 'Phase velocity')
ax4.legend(fontsize=7, framealpha=0.3)

# ── Panel 5: OTOC F(tau) ─────────────────────────────────────────────────────
ax5 = AX(2, 0)
ax5.plot(lags, F, color=COLORS['otoc'], lw=2, label='F(τ)')
ax5.axvline(0,   color='white', lw=0.7, ls='--', alpha=0.5, label='τ=0 null')
ax5.axvline(40,  color=COLORS['g_on'],  lw=0.7, ls=':', alpha=0.7, label='HP predicted τ=+40s')
ax5.axvline(-40, color=COLORS['g_on'],  lw=0.7, ls=':', alpha=0.7)
best_lag = lags[np.argmax(F)]
ax5.axvline(best_lag, color=COLORS['lambda'], lw=1.2, ls='-', alpha=0.8, label=f'Peak τ={best_lag:+.0f}s')
ax5.fill_between(lags, F, F.min(), alpha=0.15, color=COLORS['otoc'])
style_ax(ax5, 'OTOC Frobenius Cross-Correlation F(τ)', 'Lag τ (s)', 'F(τ)')
ax5.legend(fontsize=7, framealpha=0.3)

# ── Panel 6: Sliding window corr(lambda, G_score) ────────────────────────────
ax6 = AX(2, 1)
ax6.plot(window_t/60, window_corr, color='#ff9944', lw=1.5, label='20-chunk corr(λ, G_score)')
ax6.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
ax6.fill_between(window_t/60, window_corr, 0, 
                  where=window_corr>0, alpha=0.25, color=COLORS['g_on'], label='Positive corr')
ax6.fill_between(window_t/60, window_corr, 0, 
                  where=window_corr<0, alpha=0.25, color=COLORS['g_off'], label='Negative corr')
style_ax(ax6, 'Sliding Window: corr(λ_Alice, G_score_Bob)', 'Time (min)', 'Pearson r')
ax6.legend(fontsize=7, framealpha=0.3)

# ── Panel 7: Recoherence edge detail ─────────────────────────────────────────
ax7 = AX(3, 0)
# Find recoherence edges in Alice
edges = []
for i in range(1, len(alice_lams)):
    if alice_lams[i-1] > 0.9 and alice_lams[i] < 0.9:
        edges.append(alice_t_arr[i])

colors_edge = ['#ff6b9d','#ffd700','#00d4ff']
for ei, t_edge in enumerate(edges):
    window_bob = [(r['t'] - t_edge, g_score(r['cov'])) for r in bob_chunks if abs(r['t'] - t_edge) < 90]
    if window_bob:
        wt, wg = zip(*sorted(window_bob))
        ax7.plot(np.array(wt)/60, wg, color=colors_edge[ei % 3], lw=1.5, 
                 label=f'Edge {ei+1} at t={int(alice_t_rel[np.argmin(np.abs(alice_t_arr - t_edge))])//60}min')
ax7.axvline(0, color='white', lw=1, ls='--', alpha=0.7, label='ON→OFF transition')
ax7.set_xlim(-1.5, 1.5)
style_ax(ax7, 'Bob G-score Around Alice ON→OFF Transitions', 'Time from edge (min)', 'G-score')
ax7.legend(fontsize=7, framealpha=0.3)

# ── Panel 8: Per-stream timing distribution (violin) ─────────────────────────
ax8 = AX(3, 1)
stream_data = [np.concatenate([r['timing'][k, :].flatten() for r in bob_chunks]) for k in range(4)]
vparts = ax8.violinplot(stream_data, positions=[0,1,2,3], widths=0.7, showmeans=True)
for i, pc in enumerate(vparts['bodies']):
    pc.set_facecolor(['#ff6b9d','#00d4ff','#ffd700','#00ff88'][i])
    pc.set_alpha(0.6)
vparts['cmeans'].set_color('white')
vparts['cmins'].set_color('#666688')
vparts['cmaxes'].set_color('#666688')
vparts['cbars'].set_color('#666688')
ax8.set_xticks([0,1,2,3]); ax8.set_xticklabels([f'S{i}' for i in range(4)])
style_ax(ax8, "Bob Per-Stream Timing Distribution (All Chunks)", 'Stream', 'Execution time (ms)')

# ── Panel 9: Summary statistics panel ────────────────────────────────────────
ax9 = fig.add_subplot(gs[4, :])
results = json.loads(pathlib.Path('resonance_results/v3_full_analysis.json').read_text())
ax9.set_facecolor('#1a1f2e')
ax9.axis('off')

lines = [
    ("EXPERIMENT SUMMARY", '', 'white', 13, True),
    ("", "", '#888888', 9, False),
    ("Data Quality", "", '#aaaacc', 10, True),
    (f"  Chunks collected:", f"Alice={results['alice_chunks']}  Bob={results['bob_chunks']}", '#ddddff', 9, False),
    (f"  Session start offset:", f"{results['start_offset_s']:.2f}s (well within 30s threshold)", '#ddddff', 9, False),
    (f"  Alice loop rate:", f"{results['alice_loop_hz']:.1f} Hz", '#ddddff', 9, False),
    (f"  Bob loop rate:", f"{results['bob_loop_hz']:.1f} Hz", '#ddddff', 9, False),
    (f"  Zones:", f"Alice={results['alice_tz']} ({results['alice_ip']})  |  Bob={results['bob_tz']} ({results['bob_ip']})", '#ddddff', 9, False),
    ("", "", '#888888', 9, False),
    ("Gaussian Signature Test (Primary Detection)", "", '#aaaacc', 10, True),
    (f"  G-score ON:  {results['g_score_on']:.4f}  |  OFF: {results['g_score_off']:.4f}  |  delta: {results['g_delta']:.4f}", 
     f"p={results['p_perm']:.4f} — not significant", '#ffaa44', 9, False),
    (f"  Pearson r(λ, G_score):", f"{results['pearson_r']:.4f}  p={results['pearson_p']:.4f} — no correlation", '#ffaa44', 9, False),
    ("", "", '#888888', 9, False),
    ("OTOC Cross-Correlation", "", '#aaaacc', 10, True),
    (f"  F(0) null:  {results['otoc_null_F']:.4f}  |  Peak F: {results['otoc_best_F']:.4f}  at τ={results['otoc_best_lag']:+.0f}s",
     f"ratio: {results['otoc_ratio']:.4f}x", '#ffaa44', 9, False),
    (f"  F(τ=±40s) HP predicted:", f"+40s={results['otoc_hp_pos']:.4f}  −40s={results['otoc_hp_neg']:.4f}", '#ffaa44', 9, False),
    ("", "", '#888888', 9, False),
    ("Recoherence Window Analysis", "", '#aaaacc', 10, True),
    (f"  ON→OFF transition events:", f"{results['recoherence_n_events']}", '#00ff88', 9, False),
    ("  Edge 3 (t=26min):", "Bob G-score INCREASED +0.0624 after Alice OFF edge (see panel 7)", '#00ff88', 9, False),
]

y = 0.97
for label, val, color, size, bold in lines:
    weight = 'bold' if bold else 'normal'
    ax9.text(0.01, y, label, transform=ax9.transAxes, color=color, fontsize=size, 
             fontweight=weight, va='top', ha='left')
    if val:
        ax9.text(0.45, y, val, transform=ax9.transAxes, color=color, fontsize=size, 
                 va='top', ha='left')
    y -= 0.075

for sp in ax9.spines.values(): sp.set_color('#333344')
ax9.set_title('Analysis Results', color='white', fontsize=10, fontweight='bold', pad=4)

plt.savefig('resonance_results/v3_analysis_full.png', dpi=150, bbox_inches='tight', 
            facecolor='#0d1117')
print('Saved: resonance_results/v3_analysis_full.png')
