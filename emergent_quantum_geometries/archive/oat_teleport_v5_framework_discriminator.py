"""
oat_teleport_v5_framework_discriminator.py
Experiment v5: D-LinOSS Framework Discrimination

Uses damped-sinusoid decomposition (Prony/ESPRIT) to extract learned modes
(omega_k, gamma_k, A_k) from C(Gamma*t) decay series, then scores each
quantum framework by comparing mode distributions to expected signatures.

Frameworks tested:
  1. Lindblad (Markovian)     -- single mode, gamma_k = 4, omega_k = 0
  2. MBL (power-law)          -- many small gamma_k, Poisson level spacing r~0.386
  3. SYK (holographic)        -- flat amplitude, uniform gamma_k, GOE spacing r~0.536
  4. Penrose OR               -- bimodal gamma_k distribution (pre/post collapse)
  5. Heisenberg (null)        -- gamma_k = 0, baseline

Also computes corrected bi-twistor null residual:
  null_residual = |Tr[Z @ Z]| / ||Z||_F^2  (NOT normalized before storing)
  Winding number from properly phase-unwrapped spinor trajectory.
"""
import math, json, sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import hankel, svd
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence


# ── Prony / ESPRIT decomposition ───────────────────────────────────────────────

def prony_fit(signal, K=6):
    """
    Fit signal as sum of K complex exponentials using the matrix pencil method.
    signal[n] ≈ Σ_k A_k * z_k^n
    Returns: (omega_k, gamma_k, A_k) arrays where z_k = exp((-gamma_k + i*omega_k)*dt)
    """
    N = len(signal)
    L = N // 2  # pencil parameter

    # Build Hankel matrix
    H = hankel(signal[:L], signal[L-1:])  # L x (N-L+1)

    # SVD truncation to K modes
    U, s, Vh = svd(H)
    U_k = U[:, :K]
    Vh_k = Vh[:K, :]

    # Matrix pencil: shift by 1 and solve generalized eigenvalue problem
    Z = np.linalg.pinv(U_k[:-1, :]) @ U_k[1:, :]
    eigs = np.linalg.eigvals(Z)

    # Extract frequencies and decay rates (dt=1 step in x=Gamma*t space)
    gamma_k = -np.real(np.log(np.abs(eigs) + 1e-15))
    omega_k = np.imag(np.log(eigs + 1e-15 * (eigs == 0)))

    # Least-squares amplitudes
    V = np.vander(eigs, N, increasing=True).T  # N x K
    A_k, _, _, _ = np.linalg.lstsq(V, signal, rcond=None)

    return omega_k, gamma_k, A_k


def reconstruct(omega_k, gamma_k, A_k, N):
    """Reconstruct signal from mode parameters."""
    x = np.arange(N)
    eigs = np.exp((-gamma_k + 1j * omega_k))
    V = np.vander(eigs, N, increasing=True).T
    return np.real(V @ A_k)


# ── Framework signature functions ──────────────────────────────────────────────

def level_spacing_ratio(omega_k):
    """
    Level-spacing ratio r for the learned ω_k spectrum.
    r → 0.386 Poisson (MBL), r → 0.536 GOE (SYK/chaotic)
    """
    freqs = np.sort(np.abs(omega_k))
    if len(freqs) < 3:
        return float('nan')
    spacings = np.diff(freqs)
    spacings = spacings[spacings > 1e-10]
    if len(spacings) < 2:
        return float('nan')
    ratios = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1])
              for i in range(len(spacings)-1)]
    return float(np.mean(ratios))


def framework_scores(omega_k, gamma_k, A_k, K=6):
    """
    Score each framework by how well the learned modes match predictions.
    Lower score = better match.
    """
    amp = np.abs(A_k)
    amp_norm = amp / (amp.sum() + 1e-14)
    gamma_pos = np.abs(gamma_k)
    
    r_level = level_spacing_ratio(omega_k)
    
    # Amplitude entropy (high = flat = SYK/MBL, low = peaked = Lindblad)
    ent_amp = float(-np.sum(amp_norm * np.log(amp_norm + 1e-14)))
    max_ent = math.log(K)

    # 1. Lindblad: one dominant mode with gamma~4, omega~0
    # Score = deviation from single-mode structure
    dominant_idx = np.argmax(amp)
    lindblad_gamma_score = abs(gamma_k[dominant_idx] - 4.0) / 4.0
    lindblad_omega_score = abs(omega_k[dominant_idx]) / math.pi
    lindblad_concentration = 1 - amp_norm[dominant_idx]  # 0 = fully concentrated
    lindblad_score = lindblad_gamma_score + lindblad_omega_score + lindblad_concentration

    # 2. MBL: many small gamma, broad flat amplitude, Poisson r~0.386
    mbl_gamma_score = float(np.mean(gamma_pos)) / 4.0  # small gamma preferred
    mbl_spacing_score = abs(r_level - 0.386) if not math.isnan(r_level) else 1.0
    mbl_flat_score = (max_ent - ent_amp) / max_ent  # want high entropy (flat)
    mbl_score = mbl_gamma_score + mbl_spacing_score + mbl_flat_score

    # 3. SYK: uniform gamma, flat amplitude, GOE r~0.536
    syk_gamma_var = float(np.std(gamma_pos)) / (float(np.mean(gamma_pos)) + 1e-14)
    syk_spacing_score = abs(r_level - 0.536) if not math.isnan(r_level) else 1.0
    syk_flat_score = (max_ent - ent_amp) / max_ent
    syk_score = syk_gamma_var + syk_spacing_score + syk_flat_score * 0.5

    # 4. Penrose OR: bimodal gamma — two clusters of gamma_k
    # Fit a 2-component Gaussian mixture to gamma_pos
    g_sort = np.sort(gamma_pos)
    g_mid = (g_sort[0] + g_sort[-1]) / 2
    g_low = gamma_pos[gamma_pos < g_mid]
    g_high = gamma_pos[gamma_pos >= g_mid]
    # Bimodal = two well-separated clusters
    if len(g_low) > 0 and len(g_high) > 0:
        separation = (np.mean(g_high) - np.mean(g_low))
        within_spread = (np.std(g_low) + np.std(g_high)) / 2
        bimodality = separation / (within_spread + 1e-14)
        or_score = 1.0 / (1.0 + bimodality)  # 0=bimodal, 1=unimodal
    else:
        or_score = 1.0

    # 5. Heisenberg (null): all gamma~0
    heisenberg_score = float(np.mean(gamma_pos))

    return {
        "Lindblad":   float(lindblad_score),
        "MBL":        float(mbl_score),
        "SYK":        float(syk_score),
        "Penrose_OR": float(or_score),
        "Heisenberg": float(heisenberg_score),
        "r_level":    float(r_level) if not math.isnan(r_level) else None,
        "amp_entropy": float(ent_amp),
        "dominant_gamma": float(gamma_k[dominant_idx]),
        "dominant_omega": float(omega_k[dominant_idx]),
        "dominant_amp":   float(amp_norm[dominant_idx]),
    }


def winner_and_confidence(scores_dict):
    """Find winning framework and confidence (entropy of score distribution)."""
    fw = ["Lindblad","MBL","SYK","Penrose_OR","Heisenberg"]
    raw = np.array([scores_dict[f] for f in fw])
    # Convert to probabilities: lower score = higher probability
    inv = 1.0 / (raw + 1e-6)
    probs = inv / inv.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-14)))
    winner = fw[np.argmax(probs)]
    confidence = float(probs.max())
    return winner, confidence, entropy, dict(zip(fw, probs.tolist()))


# ── Bi-twistor null residual (corrected) ──────────────────────────────────────

def construct_spinor(rho2):
    """
    Construct 2-component spinor from the dominant eigenvector of ρ₂.
    Diagonalize ρ₂, take the largest eigenstate (2-qubit), project to
    Alice's single-qubit spinor by partial trace eigendecomposition.
    """
    eigs, vecs = np.linalg.eigh(rho2)
    # Dominant eigenvector (2-qubit, 4-component)
    psi_dominant = vecs[:, -1]  # largest eigenvalue
    # Reshape to 2x2 (Alice x Bob) and take Alice's reduced state
    psi_mat = psi_dominant.reshape(2, 2)
    rho_A = psi_mat @ psi_mat.conj().T
    # Alice's spinor from dominant eigenvector of rho_A
    ea, va = np.linalg.eigh(rho_A)
    spinor = va[:, -1]  # shape (2,)
    return spinor


def bitwistor_Z(spinor_A, spinor_B=None):
    """
    Construct bi-twistor Z^{AB} from two spinors (Alice and Bob sides).
    If spinor_B is None, use identity for Bob.
    Z^{AB} = spinor_A ⊗ spinor_B (outer product, 2x2 matrix)
    """
    if spinor_B is None:
        spinor_B = np.array([1.0, 0.0], dtype=complex)
    Z = np.outer(spinor_A, spinor_B)  # 2x2 complex matrix
    return Z


def null_residual(Z):
    """
    Corrected bi-twistor null residual.
    
    For a 2x2 skew-symmetric bi-twistor Z^{AB}:
      Null condition: Z^{AB} Z_{AB} = 0 (Penrose null condition)
    
    For a general (not necessarily skew) 2x2 matrix, use the Pfaffian:
      Pf = Z[0,1] - Z[1,0] (for 2x2 skew part)
    
    Corrected formula:
      Z_norm_sq = ||Z||_F^2 = Tr[Z @ Z†]
      null_prod  = |Tr[Z @ Z]|  (contraction with itself, NOT conjugate)
      null_res   = null_prod / Z_norm_sq
    
    For separable states:   null_res → 0
    For maximally entangled: null_res → 1
    """
    Z_norm_sq = float(np.real(np.trace(Z @ Z.conj().T)))
    null_prod  = abs(np.trace(Z @ Z))  # NOT conjugate: Z @ Z, not Z @ Z†
    if Z_norm_sq < 1e-14:
        return 0.0
    return float(null_prod / Z_norm_sq)


def spinor_winding(spinor_series):
    """
    Winding number of the spinor trajectory over the chi_t sweep.
    
    spinor_series: list of 2-component complex spinors at each chi_t step.
    
    Correct approach:
      1. Compute Berry phase: phi(t) = arg(<psi(t)|psi(t+dt)>)
      2. Accumulate phase using unwrapping (not zero-crossing counting)
      3. Winding = total accumulated phase / (2*pi)
    """
    phases = []
    for i in range(len(spinor_series) - 1):
        psi_t  = spinor_series[i]
        psi_t1 = spinor_series[i+1]
        overlap = np.dot(psi_t.conj(), psi_t1)
        phase_step = np.angle(overlap)  # in (-pi, pi)
        phases.append(phase_step)
    
    if not phases:
        return 0.0
    
    # Unwrap and sum
    phases = np.array(phases)
    phases_unwrapped = np.unwrap(phases)
    total_phase = float(np.sum(phases_unwrapped))
    winding = total_phase / (2 * math.pi)
    return float(winding)


# ── Main experiment ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Experiment v5: D-LinOSS Framework Discrimination")
    print("  Prony decomposition + quantum framework scoring")
    print("=" * 72)

    K_MODES = 6          # number of Prony modes
    N_GAMMA  = 60        # points in Gamma*t series
    GAMMA_T_MAX = 4.0    # maximum dimensionless decoherence

    sweep = [
        (2,  2.985), (4,  1.091), (6,  0.713),
        (8,  0.523), (10, 0.429), (12, 0.334), (16, 0.239),
    ]

    all_results = []

    for N, chi_t_opt in sweep:
        n = N // 2
        tensors = oat_mps(N, chi_t_opt)
        rho2_ideal = extract_boundary_rho(tensors, n)
        C0 = float(concurrence(rho2_ideal))

        # ── Build decay series C(x) where x = Gamma*t ──────────────────────
        x_grid = np.linspace(0, GAMMA_T_MAX, N_GAMMA)
        C_series = np.zeros(N_GAMMA)
        witness_series = np.zeros(N_GAMMA)
        spinor_series = []

        # Apply dephasing analytically
        basis = [(0,0),(0,1),(1,0),(1,1)]
        dH = np.zeros((4,4))
        for i, (a,b) in enumerate(basis):
            for j, (c,d) in enumerate(basis):
                dH[i,j] = abs(a-c) + abs(b-d)

        for idx, x in enumerate(x_grid):
            decay = np.exp(-x * dH**2)
            rho_x = rho2_ideal * decay
            rho_x /= np.trace(rho_x).real
            C_series[idx] = float(concurrence(rho_x))

            # Witness: Tr[W rho] = 0.25 - max Bell frac
            bells = [
                np.array([1,0,0,1],dtype=complex)/math.sqrt(2),
                np.array([1,0,0,-1],dtype=complex)/math.sqrt(2),
                np.array([0,1,1,0],dtype=complex)/math.sqrt(2),
                np.array([0,1,-1,0],dtype=complex)/math.sqrt(2),
            ]
            f_vals = [float(np.real(b.conj() @ rho_x @ b)) for b in bells]
            witness_series[idx] = 0.25 - max(f_vals)

            # Spinor for winding number
            spinor_series.append(construct_spinor(rho_x))

        # ── Prony decomposition on C(x) ─────────────────────────────────────
        omega_k, gamma_k, A_k = prony_fit(C_series, K=K_MODES)
        C_reconstructed = reconstruct(omega_k, gamma_k, A_k, N_GAMMA)
        recon_error = float(np.mean((C_reconstructed - C_series)**2)**0.5)

        # ── Framework scoring ────────────────────────────────────────────────
        scores = framework_scores(omega_k, gamma_k, A_k, K=K_MODES)
        winner, confidence, score_entropy, probs = winner_and_confidence(scores)

        # Inferred Gamma_mb from dominant mode gamma_k
        dominant_gamma_k = scores["dominant_gamma"]
        # dominant_gamma_k ≈ 4 * Gamma_mb * t* (in dimensionless units per step)
        # x_step = GAMMA_T_MAX / N_GAMMA
        x_step = GAMMA_T_MAX / N_GAMMA
        # gamma_k per step, convert: actual decay rate per unit x = gamma_k/x_step
        decay_rate_per_x = dominant_gamma_k / x_step if x_step > 0 else 0
        # C(x) = C0 * exp(-4*Gamma*t) = C0 * exp(-4*x)
        # So decay_rate_per_x ≈ 4 for pure Markovian
        Gamma_mb_inferred = decay_rate_per_x / 4  # in units of chi (Hz if chi=1Hz)

        # ── Bi-twistor analysis ──────────────────────────────────────────────
        # Build spinor at ideal state
        spinor_A = construct_spinor(rho2_ideal)
        spinor_B = construct_spinor(rho2_ideal)  # same boundary structure
        Z = bitwistor_Z(spinor_A, spinor_B)
        null_res = null_residual(Z)
        winding  = spinor_winding(spinor_series)

        # ── Print results ────────────────────────────────────────────────────
        print(f"\n  N={N:>3d}  C₀={C0:.4f}  Prony recon error={recon_error:.4f}")
        print(f"    Dominant mode: γ_k={dominant_gamma_k:.3f}  ω_k={scores['dominant_omega']:.3f}  "
              f"amp_frac={scores['dominant_amp']:.3f}")
        print(f"    Level-spacing r={scores['r_level']}  (MBL=0.386, SYK=0.536)")
        print(f"    Amplitude entropy={scores['amp_entropy']:.3f}  "
              f"(max={math.log(K_MODES):.3f})")
        print(f"    Framework scores: " + 
              "  ".join(f"{k}={v:.3f}" for k,v in scores.items()
                        if k in ["Lindblad","MBL","SYK","Penrose_OR","Heisenberg"]))
        print(f"    WINNER: {winner}  confidence={confidence:.3f}  entropy={score_entropy:.3f}")
        print(f"    Γ_mb_inferred = {Gamma_mb_inferred:.4f} χ")
        print(f"    Bi-twistor null_res={null_res:.4f}  "
              f"winding={winding:.3f}")
        print(f"    Witness crosses 0 at x=" +
              (f"{x_grid[np.where(witness_series > 0)[0][0]]:.2f}"
               if np.any(witness_series > 0) else "never (entangled throughout)"))

        all_results.append({
            "N": N, "chi_t_opt": chi_t_opt, "C_ideal": C0,
            "prony_modes": {
                "omega_k": omega_k.tolist(),
                "gamma_k": gamma_k.tolist(),
                "A_k_abs": np.abs(A_k).tolist(),
            },
            "prony_recon_error": recon_error,
            "framework_scores": scores,
            "framework_winner": winner,
            "framework_confidence": confidence,
            "framework_entropy": score_entropy,
            "framework_probs": probs,
            "Gamma_mb_inferred": Gamma_mb_inferred,
            "bitwistor": {
                "null_residual": null_res,
                "winding_number": winding,
                "Z_matrix": [list(row) for row in Z.tolist()],
            },
            "witness_zero_crossing_x": (
                float(x_grid[np.where(witness_series > 0)[0][0]])
                if np.any(witness_series > 0) else None
            ),
        })

    # ── Cross-N summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FRAMEWORK DISCRIMINATION SUMMARY")
    print("=" * 72)
    print(f"  {'N':>4}  {'Winner':>12}  {'Conf':>6}  {'Γ_mb':>8}  "
          f"{'null_res':>9}  {'winding':>8}  {'r_level':>8}")
    for r in all_results:
        print(f"  {r['N']:>4}  {r['framework_winner']:>12}  "
              f"{r['framework_confidence']:>6.3f}  "
              f"{r['Gamma_mb_inferred']:>8.4f}χ  "
              f"{r['bitwistor']['null_residual']:>9.4f}  "
              f"{r['bitwistor']['winding_number']:>8.3f}  "
              f"{str(r['framework_scores']['r_level'])[:6]:>8}")

    # Detect phase transition in framework winner across N
    winners = [r["framework_winner"] for r in all_results]
    transitions = [(all_results[i]["N"], winners[i], all_results[i+1]["N"], winners[i+1])
                   for i in range(len(winners)-1) if winners[i] != winners[i+1]]
    if transitions:
        print(f"\n  Framework transitions detected:")
        for t in transitions:
            print(f"    N={t[0]}→{t[2]}: {t[1]} → {t[3]} "
                  f"(possible dephasing mechanism phase transition)")
    else:
        print(f"\n  Consistent framework: {winners[0]} for all N")

    print(f"\n  Physical interpretation:")
    dominant_winner = max(set(winners), key=winners.count)
    if dominant_winner == "Lindblad":
        print("  → Markovian single-particle dephasing dominates.")
        print("    Next step: measure exchange coupling J_ex in Sr-87.")
        print("    Γ_mb ≈ J_ex × N. Threshold condition: J_ex < Γ*(N)/N = 0.55/N² Hz.")
    elif dominant_winner == "MBL":
        print("  → Many-body localization signature: power-law decay.")
        print("    v4 threshold is CONSERVATIVE — real system more robust.")
        print("    Next step: measure dynamical structure factor S(q,ω) in lattice.")
    elif dominant_winner == "SYK":
        print("  → Holographic scrambling signature: uniform mode density.")
        print("    Γ_mb ~ J√N. For Sr-87: J ~ 0.01-0.10 Hz → Γ_mb(N=4) ~ 0.02-0.20 Hz.")
        print("    Next step: OTOC measurement at JILA to bound J directly.")
    elif dominant_winner == "Penrose_OR":
        print("  → Objective Reduction signature: bimodal gamma distribution.")
        print("    Collapse time τ_OR separates pre/post-collapse dynamics.")
        print("    Next step: map τ_OR to state complexity E_G (gravitational energy).")

    with open("v5_framework_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n[DONE] v5_framework_results.json")


if __name__ == "__main__":
    main()
