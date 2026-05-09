"""
oat_teleport_v5c_model_compare.py — D-LinOSS as model comparison engine.

Instead of Prony decomposition on a single-mode exponential (which is
numerically ill-conditioned), fit each framework's predicted C(Gamma*t)
shape directly to the data and compare residuals.

Framework predictions:
  Lindblad:   C(x) = C0 * exp(-b*x)            [single exponential]
  MBL:        C(x) = C0 / (1 + b*x)^alpha      [power law]
  SYK:        C(x) = C0 * exp(-b*sqrt(x))       [sub-exponential, chaos bound]
  Penrose OR: C(x) = C0 * exp(-b*x) * H(x<xOR) + C0*exp(-b*xOR)*exp(-c*(x-xOR))
              [exponential then kink at OR collapse time]
  Heisenberg: C(x) = C0                         [no decay, null baseline]

Winner = framework with lowest normalized residual.

Also: corrected null_residual = C (for dominant eigenstate) via det(M).
Winding = total arc length on Bloch sphere (not winding number — correct for dephasing).
"""
import math, json, sys, os
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence

BELLS = [
    np.array([1,0,0,1],  dtype=complex)/math.sqrt(2),
    np.array([1,0,0,-1], dtype=complex)/math.sqrt(2),
    np.array([0,1,1,0],  dtype=complex)/math.sqrt(2),
    np.array([0,1,-1,0], dtype=complex)/math.sqrt(2),
]


# ── Framework model functions ──────────────────────────────────────────────

def model_lindblad(x, b):
    return np.exp(-b * x)

def model_mbl(x, b, alpha):
    return 1.0 / (1 + b * x) ** alpha

def model_syk(x, b):
    return np.exp(-b * np.sqrt(x))

def model_or(x, b, x_OR, c):
    y = np.where(x < x_OR,
                 np.exp(-b * x),
                 np.exp(-b * x_OR) * np.exp(-c * (x - x_OR)))
    return y

def model_heis(x):
    return np.ones_like(x)


def fit_framework(x, y_norm, name):
    """
    Fit y_norm (C/C0, normalized) to framework model.
    Returns (params, residual_rms, fitted_curve).
    """
    eps = 1e-12
    try:
        if name == "Lindblad":
            p, _ = curve_fit(model_lindblad, x, y_norm, p0=[4.0],
                             bounds=([0],[50]), maxfev=2000)
            yfit = model_lindblad(x, *p)
        elif name == "MBL":
            p, _ = curve_fit(model_mbl, x, y_norm, p0=[1.0,1.0],
                             bounds=([0,0.1],[20,5]), maxfev=2000)
            yfit = model_mbl(x, *p)
        elif name == "SYK":
            p, _ = curve_fit(model_syk, x, y_norm, p0=[2.0],
                             bounds=([0],[20]), maxfev=2000)
            yfit = model_syk(x, *p)
        elif name == "Penrose_OR":
            x_OR_guess = x[np.argmax(np.diff(y_norm) < -0.05)] if \
                         np.any(np.diff(y_norm) < -0.05) else x[-1]/2
            p, _ = curve_fit(model_or, x, y_norm,
                             p0=[2.0, float(x_OR_guess), 10.0],
                             bounds=([0,x[1],0],[20,x[-1],100]), maxfev=5000)
            yfit = model_or(x, *p)
        elif name == "Heisenberg":
            p = []
            yfit = model_heis(x)
        else:
            return None, 1e9, None

        resid = float(np.sqrt(np.mean((y_norm - yfit)**2)))
        return list(p), resid, yfit.tolist()
    except Exception:
        return None, 1.0, None


# ── Bi-twistor: det(M) formulation ────────────────────────────────────────

def bitwistor_null(rho2):
    """null_res = 2|det(M)|/||M||_F^2, where M = dominant_eigvec.reshape(2,2)."""
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    nrm = float(np.real(np.trace(M @ M.conj().T)))
    return float(2*abs(det)/nrm) if nrm > 1e-14 else 0.0


# ── Bloch arc length ────────────────────────────────────────────────────────

def bloch_arc(rho_series):
    """Total arc length of Alice's Bloch vector over the decay series."""
    sz_op = np.array([[1,0],[0,-1]], dtype=complex)
    sx_op = np.array([[0,1],[1,0]], dtype=complex)
    sy_op = np.array([[0,-1j],[1j,0]], dtype=complex)
    arc = 0.0
    prev = None
    for rho2 in rho_series:
        rho_A = np.trace(np.array(rho2).reshape(2,2,2,2), axis1=1, axis2=3)
        bvec = np.array([float(np.real(np.trace(rho_A @ op)))
                         for op in [sx_op, sy_op, sz_op]])
        if prev is not None:
            arc += float(np.linalg.norm(bvec - prev))
        prev = bvec
    return arc


def dephase(rho2, gamma_t):
    basis = [(0,0),(0,1),(1,0),(1,1)]
    dH = np.array([[abs(a-c)+abs(b-d) for c,d in basis] for a,b in basis])
    rho = rho2 * np.exp(-gamma_t * dH**2)
    return rho / max(float(np.trace(rho).real), 1e-14)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  v5c: D-LinOSS as Direct Model Comparison")
    print("  Framework fit residuals on C(Γt) decay curves")
    print("="*70)

    sweep = [(2,2.985),(4,1.091),(6,0.713),(8,0.523),(12,0.334),(16,0.239)]
    FRAMEWORKS = ["Lindblad","MBL","SYK","Penrose_OR","Heisenberg"]
    N_GAMMA, GT_MAX = 80, 5.0
    x_grid = np.linspace(0, GT_MAX, N_GAMMA)

    all_results = []
    for N, chi_t in sweep:
        n = N//2
        tensors = oat_mps(N, chi_t)
        rho0 = extract_boundary_rho(tensors, n)
        C0 = float(concurrence(rho0))

        # Build C(x) series
        C_series = np.array([float(concurrence(dephase(rho0, x)))
                             for x in x_grid])
        rho_series = [dephase(rho0, x) for x in x_grid]

        # Normalize by C0 for fitting
        y_norm = C_series / (C0 + 1e-14)

        # Fit each framework
        fit_results = {}
        for fw in FRAMEWORKS:
            params, resid, yfit = fit_framework(x_grid, y_norm, fw)
            fit_results[fw] = {"params": params, "residual": resid, "fit": yfit}

        # Winner = lowest residual
        resids = {fw: fit_results[fw]["residual"] for fw in FRAMEWORKS}
        winner = min(resids, key=resids.get)
        conf = 1 - resids[winner] / (sum(resids.values()) + 1e-14)

        # Infer Gamma_mb from Lindblad fit (the physically meaningful rate)
        lindblad_b = (fit_results["Lindblad"]["params"][0]
                      if fit_results["Lindblad"]["params"] else 4.0)
        # C(x) = C0*exp(-b*x) where x = Gamma*t
        # b relates to dephasing: C ~ exp(-4*Gamma_actual*t) → b = 4*Gamma_actual*t_unit
        # Since x is already Gamma*t, b should be ~4 for pure Markovian
        Gamma_mb = lindblad_b / 4.0  # in units of the reference Gamma

        # Bi-twistor and Bloch arc
        null_res = bitwistor_null(rho0)
        arc = bloch_arc(rho_series[:20])  # first 20 steps for efficiency

        # Witness
        W0 = 0.25 - max(float(np.real(b.conj() @ rho0 @ b)) for b in BELLS)

        # OR collapse time estimate
        or_x_OR = (fit_results["Penrose_OR"]["params"][1]
                   if fit_results["Penrose_OR"]["params"] and
                   len(fit_results["Penrose_OR"]["params"]) > 1 else None)

        print(f"\n  N={N:>3d}  C₀={C0:.4f}  Winner={winner}")
        print(f"    Residuals: " + "  ".join(
              f"{fw[:4]}={resids[fw]:.4f}" for fw in FRAMEWORKS))
        print(f"    Lindblad b={lindblad_b:.3f} (expect 4.0 for Markovian)")
        print(f"    Γ_mb = b/4 = {Gamma_mb:.3f}χ")
        if or_x_OR:
            print(f"    OR collapse x_OR = {or_x_OR:.3f}  → "
                  f"τ_OR = {or_x_OR:.3f}/Γ  E_G = ħ/τ_OR")
        print(f"    null_res(det)={null_res:.4f}  Bloch arc={arc:.4f}")
        print(f"    Witness={W0:.4f}  N_shots(3σ)≈{int(9/max(W0**2,1e-6)*1.04):d}")

        all_results.append({
            "N": N, "C0": C0, "winner": winner, "confidence": conf,
            "residuals": resids,
            "lindblad_b": lindblad_b, "Gamma_mb_units_chi": Gamma_mb,
            "OR_x_collapse": or_x_OR,
            "null_residual": null_res, "bloch_arc": arc,
            "witness_at_0": W0,
            "fit_params": {fw: fit_results[fw]["params"] for fw in FRAMEWORKS},
        })

    print("\n"+"="*70)
    print("  FRAMEWORK DISCRIMINATION SUMMARY")
    print("="*70)
    print(f"  {'N':>4}  {'Winner':>12}  {'Conf':>5}  "
          f"{'Lind_b':>7}  {'null_res':>9}  {'Arc':>7}  {'Witness':>8}")
    for r in all_results:
        print(f"  {r['N']:>4}  {r['winner']:>12}  {r['confidence']:>5.2f}  "
              f"{r['lindblad_b']:>7.3f}  {r['null_residual']:>9.4f}  "
              f"{r['bloch_arc']:>7.4f}  {r['witness_at_0']:>8.4f}")

    winners = [r["winner"] for r in all_results]
    print(f"\n  b≈4 across all N → Markovian dephasing confirmed: "
          f"{all(abs(r['lindblad_b']-4)<1 for r in all_results)}")
    print(f"  null_res=C₀ for pure state (N=2): "
          f"{abs(all_results[0]['null_residual']-all_results[0]['C0'])<0.01}")

    # JILA recommendation
    n4 = next(r for r in all_results if r["N"]==4)
    print(f"\n  JILA N=4 summary:")
    print(f"    b={n4['lindblad_b']:.3f} vs 4.0 (Markovian prediction)")
    print(f"    If b=4: Γ_mb = 1χ (OAT rate limited)")
    print(f"    If b>4: additional many-body dephasing present")
    print(f"    Witness Tr[Wρ] = {n4['witness_at_0']:.4f}")
    print(f"    Recommended N_shots ≈ {int(9/max(n4['witness_at_0']**2,1e-6)*1.04)}")

    dom = max(set(winners), key=winners.count)
    print(f"\n  Dominant framework: {dom}")
    interpret = {
        "Lindblad":   "Single-exponential decay. Measure J_ex. Γ_mb = J_ex × N.",
        "MBL":        "Power-law decay. System more robust than v4 assumes.",
        "SYK":        "Sub-exponential decay. Measure OTOC at JILA for J.",
        "Penrose_OR": "Kink in decay. τ_OR → E_G = ħ/τ_OR (gravitational collapse).",
        "Heisenberg": "No decay signal. Increase Γ range.",
    }
    print(f"  → {interpret.get(dom,'')}")

    with open("v5c_results.json","w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n[DONE] v5c_results.json")


if __name__=="__main__":
    main()
