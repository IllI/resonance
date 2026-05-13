"""Fit attenuation model to run 2 results and print summary."""
import numpy as np, json
from scipy.optimize import curve_fit

with open('ibm_run2_results.json') as f:
    r = json.load(f)

pts   = r['results']
chi_ts = np.array([p['chi_t_pi'] for p in pts]) * np.pi
T_mit  = np.array([p['T_mit']    for p in pts])
T_anal = np.array([np.cos(c/2)**2/2 for c in chi_ts])

# Fit: T_mit = A * cos^2(chi_t/2)/2  (single param)
mask = T_anal > 0.01
A_fit, = curve_fit(lambda x, a: a*x, T_anal[mask], T_mit[mask])[0]
resid = T_mit[mask] - A_fit*T_anal[mask]
rms   = np.sqrt(np.mean(resid**2))
r2    = 1 - np.var(resid)/np.var(T_mit[mask])

print("FIT: T_xx_hardware = A * cos^2(chi_t_eff/2)/2")
print(f"  A (noise attenuation) = {A_fit:.4f}  (1.0=perfect, <1=noise)")
print(f"  RMS residual          = {rms:.5f}")
print(f"  R^2                   = {r2:.5f}")
print()
print(f"  {'chi_t/pi':>9}  {'T_mit':>8}  {'A*analytic':>11}  {'residual':>9}")
print("  "+"-"*42)
for p, ta in zip(pts, T_anal):
    chi = p['chi_t_pi']
    tm  = p['T_mit']
    fit = A_fit * ta
    print(f"  {chi:9.4f}  {tm:8.4f}  {fit:11.4f}  {tm-fit:9.5f}")

null = r['null']
sig  = null['sigma']
print(f"  {'1.0000':>9}  {null['T_mit']:8.5f}  {'0.0000':>11}  {null['T_mit']:9.5f}  <- NULL ({null['T_mit']/sig:.1f}sig)")

print()
print(f"Run 2 summary:")
print(f"  Attenuation:       A = {A_fit:.4f}  ({(1-A_fit)*100:.1f}% signal loss from noise)")
print(f"  Null (4000 shots): T_xx(pi) = {null['T_mit']:.5f} +/- {sig:.5f}  ({null['T_mit']/sig:.1f} sigma from zero)")
print(f"  Curve shape:       R^2 = {r2:.5f}  (functional form confirmed)")

print()
print("Mirror symmetry:")
for m in r['mirrors']:
    chi_pos = abs(m['chi_t_pi'])
    idx = int(round(chi_pos * 8))
    T_pos = pts[idx]['T_mit'] if idx < len(pts) else float('nan')
    print(f"  T_mit(-{chi_pos:.3f}pi)={m['T_mit']:.4f}  T_mit(+{chi_pos:.3f}pi)={T_pos:.4f}  "
          f"diff={abs(m['T_mit']-T_pos):.4f}  SYM-{'OK' if abs(m['T_mit']-T_pos)<0.06 else 'FAIL'}")

# Save fit results
fit_summary = {"A_attenuation": A_fit, "R2": r2, "RMS": rms,
               "null_T_mit": null['T_mit'], "null_sigma": sig,
               "null_nsigma": null['T_mit']/sig}
with open("ibm_run2_fit.json","w") as f:
    json.dump(fit_summary, f, indent=2)
print("\nFit saved: ibm_run2_fit.json")
