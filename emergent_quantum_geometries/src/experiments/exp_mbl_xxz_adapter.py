"""
exp_mbl_xxz_adapter.py — MBL Probe Calibration via Disordered XXZ Chain

Generates the power-law correlation decay that D-LinOSS needs to calibrate
the MBL framework probe.

Physics:
  H = J Σ_i σ_z^i σ_z^{i+1} + Σ_i h_i σ_z^i
  where h_i ~ U[-W, W] (random on-site fields)

  MBL phase: W/J > 3.5
  Observable: imbalance I(t) = (1/L) Σ_i (-1)^i <σ_z^i(t)>
  MBL signature: I(t) ~ t^{-α} for 0 < α < 1 (power law, not exponential)

  At long times the imbalance saturates to a finite value I(∞) > 0,
  which is the MBL order parameter. The approach to that value is power-law.

D-LinOSS input: I(t) time series (L=14, W/J=5, 3 disorder realizations averaged)
Expected probe result: MBL wins (power-law residual < Lindblad/SYK residual)
"""
import math, json, sys, os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.linalg import hankel, svd as scipy_svd
sys.path.insert(0, os.path.dirname(__file__))


# ── XXZ chain exact diagonalization ────────────────────────────────────────

def build_xxz_hamiltonian(L, J=1.0, W=5.0, seed=0):
    """
    Build disordered XXZ Hamiltonian in the σ_z basis.

    H = J Σ_i [σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} + σ_z^i σ_z^{i+1}]
      + Σ_i h_i σ_z^i

    Equivalent to: H = 2J Σ_i [S+^i S-^{i+1} + h.c.] + J Σ_i σ_z^i σ_z^{i+1}
                      + Σ_i h_i σ_z^i

    For MBL we only need the Ising part + disorder (sufficient for imbalance):
    H = J Σ_i σ_z^i σ_z^{i+1} + Σ_i h_i σ_z^i   (Ising + disorder)

    This is diagonal in the σ_z basis — no matrix exponentiation needed,
    just eigenphases. Fast and exact.
    """
    dim = 2 ** L
    rng = np.random.default_rng(seed)
    h = rng.uniform(-W, W, L)  # disorder fields

    # Build diagonal Hamiltonian in σ_z basis
    H_diag = np.zeros(dim)
    for state in range(dim):
        bits = [(state >> i) & 1 for i in range(L)]  # 0=up, 1=down
        spins = [1 - 2*b for b in bits]               # +1=up, -1=down
        # ZZ coupling
        for i in range(L-1):
            H_diag[state] += J * spins[i] * spins[i+1]
        # Disorder
        for i in range(L):
            H_diag[state] += h[i] * spins[i]

    return H_diag, h


def compute_imbalance(L, J=1.0, W=5.0, t_max=50.0, N_t=60, seed=0):
    """
    Compute the charge imbalance I(t) = (1/L) Σ_i (-1)^i <σ_z^i(t)>

    Initial state: Néel state |↑↓↑↓...⟩ (maximally imbalanced)
    Under MBL dynamics, I(t) decays slowly as a power law to I(∞) > 0.

    Uses the diagonal representation: fast and exact.
    """
    dim = 2**L
    H_diag, h = build_xxz_hamiltonian(L, J, W, seed)

    # Néel initial state |↑↓↑↓...⟩: site i=up for even i, down for odd i
    # In binary representation: bit i=0 (up) for even i, bit i=1 (down) for odd i
    neel_state = sum((1 << i) for i in range(1, L, 2))  # bits set for odd sites
    psi0 = np.zeros(dim, dtype=complex)
    psi0[neel_state] = 1.0

    # σ_z^i eigenvalues for each basis state
    sz_i = np.zeros((dim, L))
    for state in range(dim):
        for i in range(L):
            bit = (state >> i) & 1
            sz_i[state, i] = 1 - 2*bit   # +1=up, -1=down

    # Staggered density operator: n_stag[state] = (1/L) Σ_i (-1)^i sz_i
    stagger = np.array([(-1)**i for i in range(L)])
    n_stag = sz_i @ stagger / L  # shape (dim,)

    # Time evolution in diagonal basis: ψ(t) = e^{-iHt} |Néel⟩
    # |ψ(t)|² = |ψ(t)[state]|² = |ψ(0)[state]|² (diagonal → no spreading!)
    # Wait — diagonal H means no hopping, so this is pure Ising + disorder.
    # For MBL imbalance we need the hopping term too.
    # Solution: use only disorder + Ising but add small XX hopping perturbatively,
    # OR: work in a reduced sector with exact XY + Ising + disorder.
    # Simplest correct approach: full Hamiltonian with XX+YY terms.
    # For L=12 (dim=4096): build full sparse matrix.

    # Build full XXZ Hamiltonian (sparse-like, but dense for L≤14)
    H_full = np.diag(H_diag.astype(complex))  # start with diagonal

    # Add XX+YY hopping: (1/2)(σ+_i σ-_{i+1} + h.c.) for each pair
    # σ+ |↓⟩ = |↑⟩  →  clears bit i
    # σ- |↑⟩ = |↓⟩  →  sets bit i
    for i in range(L-1):
        mask_i  = 1 << i
        mask_i1 = 1 << (i+1)
        for state in range(dim):
            bit_i  = (state >> i)  & 1
            bit_i1 = (state >> (i+1)) & 1
            # σ+_i σ-_{i+1}: flips i from ↓→↑ and i+1 from ↑→↓
            if bit_i == 1 and bit_i1 == 0:   # site i is ↓, site i+1 is ↑
                new_state = state ^ mask_i ^ mask_i1  # flip both
                H_full[new_state, state] += J  # off-diagonal element
            # σ-_i σ+_{i+1}: reverse
            if bit_i == 0 and bit_i1 == 1:
                new_state = state ^ mask_i ^ mask_i1
                H_full[new_state, state] += J

    # Diagonalize full H
    print(f"  Diagonalizing full XXZ H (dim={dim})...", end="", flush=True)
    evals, evecs = np.linalg.eigh(H_full)
    print(" done")

    # ψ(t) in eigenbasis: coefficients c_n = <n|ψ(0)>
    c = evecs.conj().T @ psi0   # shape (dim,)

    t_grid = np.linspace(0, t_max, N_t)
    imbalance = np.zeros(N_t)

    for it, t in enumerate(t_grid):
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (c * phase)       # ψ(t)
        # <σ_z^i(t)> = <ψ(t)|σ_z^i|ψ(t)>
        probs = np.abs(psi_t)**2           # |<state|ψ(t)>|²
        # Imbalance
        imbalance[it] = float(np.dot(probs, n_stag))

    return t_grid, imbalance


# ── D-LinOSS and framework scoring ─────────────────────────────────────────

def dlinoss_modes(x, y, K=8):
    """Matrix pencil decomposition on imbalance decay."""
    # Shift and normalize: focus on decay toward long-time value
    y_inf = float(np.mean(y[-5:]))   # I(∞) estimate
    yn = (y - y_inf) / max(abs(y[0] - y_inf), 1e-10)

    N = len(yn)
    L = N // 2
    K = min(K, L-1)
    Hmat = hankel(yn[:L], yn[L-1:])
    U, s, _ = scipy_svd(Hmat)
    K_eff = max(1, int(np.sum(s > s[0]*0.01)))
    K_eff = min(K_eff, K)
    U_k = U[:, :K_eff]
    Z = np.linalg.pinv(U_k[:-1,:]) @ U_k[1:,:]
    eigs = np.linalg.eigvals(Z)
    eigs = eigs[np.abs(eigs) <= 1.0+1e-6]
    if len(eigs) == 0:
        eigs = np.array([0.95+0j])
    dt = (x[-1]-x[0])/(len(x)-1)
    gamma_k = np.clip(-np.real(np.log(np.abs(eigs)+1e-15))/dt, 0, 100)
    omega_k = np.imag(np.log(eigs+1e-15*(eigs==0)))/dt
    V = np.vander(eigs, N, increasing=True).T
    A_k, _, _, _ = np.linalg.lstsq(V, yn.astype(complex), rcond=None)
    A_k_abs = np.abs(A_k)
    y_recon = np.real(V @ A_k) + y_inf
    err = float(np.sqrt(np.mean((y - y_recon)**2)))
    return gamma_k, omega_k, A_k_abs, err, K_eff, float(y_inf)


def score_frameworks(t, I_t):
    """Fit MBL, Lindblad, SYK models to I(t).

    Restricts to t in [t_min, t_window] for power-law regime where
    finite-size oscillations haven't dominated the signal.
    """
    I_inf = float(np.mean(I_t[-5:]))
    # Restrict to early regime where power law holds (before saturation)
    # Use t in [0.5, 12] to avoid finite-size oscillations at late times
    t_min, t_win = 0.5, 12.0
    mask = (t >= t_min) & (t <= t_win)
    t_w = t[mask]
    if len(t_w) < 4:
        t_w = t[t >= t_min][:8]
        mask = np.zeros(len(t), dtype=bool)
        mask[np.where(t >= t_min)[0][:8]] = True

    # Subtract saturation value and normalize
    dy = I_t[mask] - I_inf
    dy0 = max(abs(dy[0]), 1e-10)
    yn = dy / dy0   # decays from ~1 toward 0

    scores = {}

    # MBL: power law yn ~ (t/t0)^{-alpha}
    t0 = t_w[0]
    try:
        p, _ = curve_fit(lambda tt, a: (tt/t0)**(-a),
                         t_w[1:], yn[1:], p0=[0.4],
                         bounds=([0.01], [1.5]), maxfev=3000)
        yf_vals = (t_w[1:]/t0)**(-p[0])
        scores["MBL"] = {
            "alpha": float(p[0]),
            "I_inf": I_inf,
            "residual": float(np.sqrt(np.mean((yn[1:] - yf_vals)**2))),
            "note": f"I(t)-I(inf) ~ t^{{-{p[0]:.3f}}}"
        }
    except Exception as e:
        scores["MBL"] = {"alpha": 0.5, "residual": 1.0, "note": str(e)}

    # Lindblad: pure exponential (should fail for MBL)
    try:
        p, _ = curve_fit(lambda tt, b: np.exp(-b*(tt-t0)),
                         t_w, yn, p0=[0.1], bounds=([0],[5]), maxfev=2000)
        yf_vals = np.exp(-p[0]*(t_w-t0))
        scores["Lindblad"] = {
            "b": float(p[0]),
            "residual": float(np.sqrt(np.mean((yn - yf_vals)**2)))
        }
    except Exception:
        scores["Lindblad"] = {"b": 0.1, "residual": 1.0}

    # SYK: stretched exponential exp(-b*sqrt(t))
    try:
        p, _ = curve_fit(lambda tt, b: np.exp(-b*np.sqrt(tt-t0+1e-6)),
                         t_w, yn, p0=[0.2], bounds=([0],[5]), maxfev=2000)
        yf_vals = np.exp(-p[0]*np.sqrt(t_w-t0+1e-6))
        scores["SYK"] = {
            "b": float(p[0]),
            "residual": float(np.sqrt(np.mean((yn - yf_vals)**2)))
        }
    except Exception:
        scores["SYK"] = {"b": 0.1, "residual": 1.0}

    winner = min(scores, key=lambda fw: scores[fw]["residual"])
    return scores, winner, I_inf


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*68)
    print("  MBL Probe Calibration — Disordered XXZ Chain")
    print("  H = J Σ σ_z^i σ_z^{i+1} + XX/YY + Σ h_i σ_z^i")
    print("="*68)

    L     = 12      # chain length (dim=4096, feasible for ED)
    J     = 1.0     # exchange coupling
    W     = 5.0     # disorder strength (W/J=5 >> 3.5 → deep MBL)
    T_MAX = 30.0    # time in units of 1/J  (shorter: stay in power-law regime)
    N_T   = 60      # time points
    N_REAL = 5      # disorder realizations (more = smoother power-law)

    print(f"\n  L={L}, J={J}, W={W} (W/J={W/J:.1f} >> 3.5 = MBL threshold)")
    print(f"  Time: 0 to {T_MAX}/J, {N_T} points, {N_REAL} disorder realizations")
    print(f"  Hilbert space: {2**L} states")
    print(f"\n  Expected MBL signature: I(t)-I(∞) ~ t^{{-α}}, 0 < α < 1")
    print(f"  Expected I(∞) > 0  (memory of initial state = MBL order parameter)")

    # Average over disorder realizations
    t_grid = None
    imbalance_avg = None
    for seed in range(N_REAL):
        print(f"\n  Realization {seed+1}/{N_REAL} (seed={seed}):")
        t, I = compute_imbalance(L, J, W, T_MAX, N_T, seed)
        if imbalance_avg is None:
            t_grid = t
            imbalance_avg = I / N_REAL
        else:
            imbalance_avg += I / N_REAL

    I_inf = float(np.mean(imbalance_avg[-5:]))
    print(f"\n  I(t=0)   = {imbalance_avg[0]:.4f}  (expect ~1 for Néel start)")
    print(f"  I(t={T_MAX:.0f}) = {imbalance_avg[-1]:.4f}  (expect > 0 for MBL, → 0 for ergodic)")
    print(f"  I(∞) est = {I_inf:.4f}  (MBL order parameter: > 0 = localized)")

    mbl_confirmed = I_inf > 0.05
    print(f"  MBL phase: {'✓ CONFIRMED  (I(∞) > 0)' if mbl_confirmed else '⚠ NOT MBL (I(∞) ≈ 0)'}")

    # D-LinOSS fit
    print(f"\n  Running D-LinOSS on imbalance decay...")
    gamma_k, omega_k, A_k, recon_err, K_eff, I_inf_fit = dlinoss_modes(t_grid, imbalance_avg)

    print(f"  K_eff={K_eff}  recon_err={recon_err:.4f}")
    print(f"  γ_k (top 4): {np.sort(gamma_k)[:4].round(5)}")
    print(f"  Note: MBL → slow power-law → small γ_k values (near zero)")

    # Framework scoring
    scores, winner, I_inf = score_frameworks(t_grid, imbalance_avg)

    print(f"\n  Framework model fits:")
    for fw, s in scores.items():
        note = s.get("note", "")
        print(f"    {fw:10s}: residual={s['residual']:.4f}  {note}")
    print(f"\n  Winner: {winner}")
    print(f"  MBL probe calibration: {'✓ CALIBRATED' if winner=='MBL' else '⚠ CHECK'}")

    if winner == "MBL":
        alpha = scores["MBL"]["alpha"]
        print(f"  Power-law exponent α = {alpha:.3f}  (expected 0.2-0.8 for deep MBL)")
        print(f"  If OAT data returns MBL winner: Γ_mb follows power-law, not exp(-Γt)")
        print(f"  That would indicate disorder-induced localization in Sr-87 — a surprise.")

    print(f"\n  Comparison residuals (want MBL << Lindblad, SYK):")
    print(f"    MBL:      {scores['MBL']['residual']:.4f}")
    print(f"    Lindblad: {scores['Lindblad']['residual']:.4f}  "
          f"({scores['Lindblad']['residual']/max(scores['MBL']['residual'],1e-6):.1f}× worse)")
    print(f"    SYK:      {scores['SYK']['residual']:.4f}  "
          f"({scores['SYK']['residual']/max(scores['MBL']['residual'],1e-6):.1f}× worse)")

    # Save
    output = {
        "experiment": "mbl_xxz_disordered_chain",
        "L": L, "J": J, "W": W, "W_over_J": W/J,
        "N_realizations": N_REAL, "t_max": T_MAX, "N_t": N_T,
        "t_grid": t_grid.tolist(),
        "imbalance_avg": imbalance_avg.tolist(),
        "I_inf": I_inf, "mbl_phase_confirmed": mbl_confirmed,
        "dlinoss": {
            "gamma_k": gamma_k.tolist(), "omega_k": omega_k.tolist(),
            "A_k": A_k.tolist(), "recon_err": recon_err, "K_eff": K_eff,
        },
        "framework_scores": scores,
        "framework_winner": winner,
        "calibration": {
            "mbl_calibrated": winner == "MBL",
            "alpha": scores["MBL"].get("alpha", None),
            "I_inf": I_inf,
            "jila_relevance": (
                "MBL requires disorder W/J > 3.5. Sr-87 in ordered optical lattice "
                "has no such disorder — MBL winning on OAT data would be a surprise. "
                "Probe calibrated for completeness: if it fires, the physics is real."
            )
        }
    }
    with open("mbl_probe_result.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[DONE] mbl_probe_result.json")


if __name__ == "__main__":
    main()
