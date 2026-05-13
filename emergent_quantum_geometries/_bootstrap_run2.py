"""
Residual analysis + bootstrap confidence intervals for Run 2 fit.
Outputs: ibm_run2_bootstrap.json, prints residual structure.
"""
import numpy as np, json
from scipy.optimize import curve_fit

with open('ibm_run2_results.json') as f:
    r = json.load(f)
with open('ibm_run2_fit.json') as f:
    fit0 = json.load(f)

pts    = r['results']
chi_ts = np.array([p['chi_t_pi'] for p in pts]) * np.pi
T_mit  = np.array([p['T_mit']    for p in pts])
shots  = np.array([p['shots']    for p in pts])
sigma  = 1.0/(2.0*np.sqrt(shots))   # shot-noise sigma per point

# Include 4000-shot null
null = r['null']
chi_ts = np.append(chi_ts, np.pi)
T_mit  = np.append(T_mit,  null['T_mit'])
shots  = np.append(shots,  4000)
sigma  = np.append(sigma,  null['sigma'])

T_anal = np.cos(chi_ts/2)**2 / 2
A_fit  = fit0['A_attenuation']

# ── Residuals ─────────────────────────────────────────────────────────────
residuals = T_mit - A_fit * T_anal

print("RESIDUAL ANALYSIS")
print("="*55)
print(f"{'chi_t/pi':>9}  {'T_mit':>8}  {'fit':>8}  {'resid':>8}  {'resid/sig':>10}")
print("-"*55)
for c, tm, ta, res, sig in zip(chi_ts/np.pi, T_mit, T_anal, residuals, sigma):
    nsig = res/sig
    print(f"  {c:7.4f}  {tm:8.4f}  {A_fit*ta:8.4f}  {res:8.5f}  {nsig:10.2f}")

print()
print(f"Residual mean:    {residuals.mean():.5f}  (should be ~0)")
print(f"Residual std:     {residuals.std():.5f}")
print(f"Max |residual|:   {np.abs(residuals).max():.5f}")
print(f"Shapiro-Wilk test of residuals:")
from scipy import stats
stat, p = stats.shapiro(residuals)
print(f"  W={stat:.4f}, p={p:.4f}  ({'Gaussian OK' if p>0.05 else 'non-Gaussian'})")
print(f"Run (sign) test: {'PASS' if sum(residuals>0)==5 or sum(residuals<0)==5 else 'structured?'}")

# ── Bootstrap CI ─────────────────────────────────────────────────────────
print("\nBOOTSTRAP CONFIDENCE INTERVALS (N=10000)")
print("="*55)
N_boot = 10000
rng    = np.random.default_rng(42)

A_boots, offset_boots = [], []
mask = T_anal > 0.001   # exclude exact-zero analytic points

def model_offset(x, a, b):
    """Allow small offset: T_xx = a*cos^2 + b"""
    return a * x + b

for _ in range(N_boot):
    # Resample with replacement, weighted by shot counts
    idx = rng.choice(len(T_mit[mask]), size=len(T_mit[mask]), replace=True)
    T_s = T_mit[mask][idx]
    A_s = T_anal[mask][idx]
    try:
        popt, _ = curve_fit(model_offset, A_s, T_s, p0=[0.9, 0.0])
        A_boots.append(popt[0])
        offset_boots.append(popt[1])
    except Exception:
        pass

A_boots = np.array(A_boots)
offset_boots = np.array(offset_boots)

print(f"A (attenuation):  {A_boots.mean():.4f} ± {A_boots.std():.4f}  "
      f"95% CI [{np.percentile(A_boots,2.5):.4f}, {np.percentile(A_boots,97.5):.4f}]")
print(f"Offset:           {offset_boots.mean():.5f} ± {offset_boots.std():.5f}  "
      f"95% CI [{np.percentile(offset_boots,2.5):.5f}, {np.percentile(offset_boots,97.5):.5f}]")
print(f"  (Offset compatible with zero: {abs(offset_boots.mean()) < 2*offset_boots.std()})")

# ── Save ─────────────────────────────────────────────────────────────────
out = {
    "A_bootstrap_mean":     float(A_boots.mean()),
    "A_bootstrap_std":      float(A_boots.std()),
    "A_CI_95":              [float(np.percentile(A_boots,2.5)),
                             float(np.percentile(A_boots,97.5))],
    "offset_mean":          float(offset_boots.mean()),
    "offset_std":           float(offset_boots.std()),
    "offset_CI_95":         [float(np.percentile(offset_boots,2.5)),
                             float(np.percentile(offset_boots,97.5))],
    "residual_mean":        float(residuals.mean()),
    "residual_std":         float(residuals.std()),
    "residual_max_abs":     float(np.abs(residuals).max()),
    "shapiro_W":            float(stat),
    "shapiro_p":            float(p),
    "gaussian_residuals":   bool(p > 0.05),
    "N_bootstrap":          N_boot,
}
with open("ibm_run2_bootstrap.json","w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: ibm_run2_bootstrap.json")
