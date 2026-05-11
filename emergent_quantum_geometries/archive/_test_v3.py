"""Full cross-validation N=2..16 + fidelity smoke test."""
import math, numpy as np, sys
sys.path.insert(0, '.')
from oat_teleport_v3_tpu import (oat_mps, extract_boundary_rho, cross_validate,
                                  teleport_fidelity, avg_fidelity, haar_qubit)
from jila_oat_exact_tpu import concurrence

print("=== Cross-validation N=2..16 ===")
xv_all_pass = True
for N in [2, 4, 6, 8, 10, 12, 14, 16]:
    C_ex, C_tn, diff, rho2 = cross_validate(N, math.pi / 3)
    ok = diff < 1e-5
    if not ok: xv_all_pass = False
    print(f"N={N:>2d}: C_exact={C_ex:.6f}  C_tn={C_tn:.6f}  diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")

print(f"\nCross-val: {'ALL PASS' if xv_all_pass else 'SOME FAIL'}")

print("\n=== Fidelity smoke test N=4 (K=500) ===")
N = 4; n = 2
chi_t_opt = 1.043
tensors = oat_mps(N, chi_t_opt)
rho2 = extract_boundary_rho(tensors, n)
Co = concurrence(rho2)
Fs, ss = avg_fidelity(rho2, K=500, schmidt_rotate=True)
Fn, sn = avg_fidelity(rho2, K=500, schmidt_rotate=False)
gain_pred = (1 - Co) / 6
print(f"C={Co:.4f}  F_sch={Fs:.4f}  F_naive={Fn:.4f}  gain={Fs-Fn:.4f}  pred={gain_pred:.4f}")
print(f"H1 ok: {abs(Fs-(2+Co)/3)<0.01}  Gain ok: {abs((Fs-Fn)-gain_pred)<0.01}")
