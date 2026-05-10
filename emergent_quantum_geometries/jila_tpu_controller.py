"""
jila_tpu_controller.py — Quantum-Classical Hybrid Loop Controller

The TPU is the classical controller in a quantum-classical hybrid loop:
  - JILA provides quantum hardware (Sr-87 OAT + Bell measurement)
  - TPU provides intelligence (D-LinOSS fit, framework ID, parameter update)

Five-step loop (< 1s on TPU, ~6s per JILA experimental batch):

  1. Ingest JILA shots → Tr[W·ρ(τ)] per delay point
  2. D-LinOSS fit → learned (ω_k, γ_k, A_k) spectrum
  3. Framework scoring → Lindblad / MBL / SYK / OR winner
  4. Extract Γ_eff → compute Γ_mb = Γ_eff - Γ₁
  5. Feed back → optimal θ_A, θ_B, χt*, predicted F

Usage:
  controller = JILATPUController(N=4, chi_Hz=1.0)
  feedback = controller.run_loop(shots_array, tau_array)
  # shots_array: shape (n_tau_points, n_shots) with +1/-1 values
  # tau_array:   delay times in seconds
"""
import math, json, sys, os
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.linalg import hankel, svd as scipy_svd
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence

GAMMA_1_SR87 = 1.0 / 118.0   # s⁻¹, single-particle dephasing (Rey 2025)

BELLS = [
    np.array([1,0,0,1],  dtype=complex)/math.sqrt(2),
    np.array([1,0,0,-1], dtype=complex)/math.sqrt(2),
    np.array([0,1,1,0],  dtype=complex)/math.sqrt(2),
    np.array([0,1,-1,0], dtype=complex)/math.sqrt(2),
]
SIGMAS = [
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
]


class JILATPUController:
    """
    Classical controller for the JILA-TPU quantum-classical hybrid loop.

    Responsibilities:
    - Reconstruct Tr[W·ρ(τ)] from raw JILA shots
    - Fit D-LinOSS observer modes (ω_k, γ_k, A_k)
    - Score framework hypotheses (Lindblad, MBL, SYK, OR)
    - Extract Γ_eff(N) and compute Γ_mb = Γ_eff - Γ₁
    - Compute optimal Schmidt angles θ_A, θ_B for next run
    - Return feedback packet to JILA
    """

    def __init__(self, N, chi_Hz=1.0, chi_t_opt=None):
        self.N = N
        self.n = N // 2
        self.chi_Hz = chi_Hz

        # Load optimal chi_t from simulation
        CHI_T_TABLE = {2: 2.985, 4: 1.091, 6: 0.713, 8: 0.523,
                       10: 0.429, 12: 0.334, 16: 0.239}
        self.chi_t_opt = chi_t_opt or CHI_T_TABLE.get(N, 1.0)
        self.t_opt = self.chi_t_opt / chi_Hz

        # Build ideal MPS state and witness
        print(f"  [Controller] Building MPS for N={N}, chi_t={self.chi_t_opt:.3f}...")
        tensors = oat_mps(N, self.chi_t_opt)
        self.rho2_ideal = extract_boundary_rho(tensors, self.n)
        self.C_ideal = float(concurrence(self.rho2_ideal))
        self.F_ideal = (2 + self.C_ideal) / 3

        # Best witness operator
        f_vals = [float(np.real(b.conj() @ self.rho2_ideal @ b)) for b in BELLS]
        k_opt = np.argmax(f_vals)
        self.beta_opt = BELLS[k_opt]
        self.W_op = np.eye(4, dtype=complex)/4 - np.outer(self.beta_opt, self.beta_opt.conj())
        self.W0_ideal = float(np.real(np.trace(self.W_op @ self.rho2_ideal)))

        # Schmidt angles from SVD of ideal rho2
        _, vecs = np.linalg.eigh(self.rho2_ideal)
        psi_dom = vecs[:, -1].reshape(2, 2)
        U, _, Vh = np.linalg.svd(psi_dom)
        self.theta_A_ideal = float(np.angle(U[0, 0]))
        self.theta_B_ideal = float(np.angle(Vh[0, 0]))

        print(f"  [Controller] C₀={self.C_ideal:.4f}  F_ideal={self.F_ideal:.4f}")
        print(f"  [Controller] W₀={self.W0_ideal:.4f}")
        print(f"  [Controller] θ_A={math.degrees(self.theta_A_ideal):.1f}°  "
              f"θ_B={math.degrees(self.theta_B_ideal):.1f}°")

    # ── Step 1: Reconstruct witness from shots ──────────────────────────────

    def reconstruct_witness(self, shots_array):
        """
        Invert shot encoding to recover Tr[W·ρ(τ)].

        Shot encoding (in synthetic_jila_shots and IBM circuit):
          +1 = separable outcome, -1 = entangled outcome
          p_entangled = P(-1) = (0.25 - Tr[W·ρ]) / 0.5
          mean(shots) = P(+1) - P(-1) = 1 - 2·p_entangled

        Inversion:
          p_entangled = (1 - mean) / 2
          Tr[W·ρ] = 0.25 - 0.5·p_entangled = mean / 4

        Verification: mean=-0.68 → witness=-0.17 ✓ (entangled, negative)
        """
        n_shots = shots_array.shape[1]
        witness_vals = np.mean(shots_array, axis=1) / 4  # CORRECTED: was * (-1)
        uncertainties = 0.5 / np.sqrt(n_shots) * np.ones(len(witness_vals))  # σ_W = σ_mean/4
        return witness_vals.tolist(), uncertainties.tolist()

    # ── Step 2: D-LinOSS fit ────────────────────────────────────────────────

    def dlinoss_fit(self, tau_arr, witness_arr, K=6):
        """
        Fit witness decay via matrix pencil method.
        Returns (omega_k, gamma_k, A_k, recon_err).
        """
        y = np.array(witness_arr)
        W0 = y[0] if abs(y[0]) > 1e-6 else self.W0_ideal
        yn = y / W0  # normalize to start at 1

        N = len(yn)
        L = N // 2
        K = min(K, L - 1)

        H = hankel(yn[:L], yn[L-1:])
        U, s, _ = scipy_svd(H)

        # Adaptive rank: keep modes with singular value > 2% of max
        K_eff = max(1, int(np.sum(s > s[0] * 0.02)))
        K_eff = min(K_eff, K)

        U_k = U[:, :K_eff]
        Z = np.linalg.pinv(U_k[:-1, :]) @ U_k[1:, :]
        eigs = np.linalg.eigvals(Z)
        eigs = eigs[np.abs(eigs) <= 1.0 + 1e-6]

        # Convert from per-step to per-second rates
        dt = (tau_arr[-1] - tau_arr[0]) / (len(tau_arr) - 1)
        gamma_k = np.clip(-np.real(np.log(np.abs(eigs) + 1e-15)) / dt, 0, 1000)
        omega_k = np.imag(np.log(eigs + 1e-15*(eigs==0))) / dt

        V = np.vander(eigs, N, increasing=True).T
        A_k, _, _, _ = np.linalg.lstsq(V, yn.astype(complex), rcond=None)
        A_k = np.abs(A_k)

        y_recon = np.real(V @ A_k)
        recon_err = float(np.sqrt(np.mean((yn - y_recon)**2)))

        return omega_k, gamma_k, A_k, recon_err

    # ── Step 3: Framework scoring ───────────────────────────────────────────

    def score_frameworks(self, tau_arr, witness_arr):
        """
        Fit each framework model to witness decay. Return scores dict.
        """
        tau = np.array(tau_arr)
        y = np.array(witness_arr)
        W0 = y[0] if abs(y[0]) > 1e-6 else self.W0_ideal
        yn = y / W0

        scores = {}

        # Lindblad: exp(-b·τ)
        try:
            p, _ = curve_fit(lambda t, b: np.exp(-b*t), tau, yn,
                            p0=[4*GAMMA_1_SR87], bounds=([0],[10]), maxfev=5000)
            yf = np.exp(-p[0]*tau)
            scores["Lindblad"] = {
                "b": float(p[0]),
                "Gamma_eff": float(p[0]/4),  # Tr[W] ~ exp(-4Γt)
                "residual": float(np.sqrt(np.mean((yn-yf)**2)))
            }
        except Exception:
            scores["Lindblad"] = {"b": 4*GAMMA_1_SR87, "Gamma_eff": GAMMA_1_SR87,
                                  "residual": 1.0}

        # MBL: (1 + b·τ)^{-α}
        try:
            p, _ = curve_fit(lambda t, b, a: 1/(1+b*t)**a, tau, yn,
                            p0=[0.1, 1.0], bounds=([0,0.1],[5,5]), maxfev=5000)
            yf = 1/(1+p[0]*tau)**p[1]
            scores["MBL"] = {
                "b": float(p[0]), "alpha": float(p[1]),
                "residual": float(np.sqrt(np.mean((yn-yf)**2)))
            }
        except Exception:
            scores["MBL"] = {"b": 0.1, "alpha": 1.0, "residual": 1.0}

        # SYK: exp(-b·√τ)
        try:
            p, _ = curve_fit(lambda t, b: np.exp(-b*np.sqrt(t+1e-10)), tau, yn,
                            p0=[0.1], bounds=([0],[5]), maxfev=5000)
            yf = np.exp(-p[0]*np.sqrt(tau+1e-10))
            scores["SYK"] = {
                "b": float(p[0]),
                "J_scramble": float(p[0]**2 / self.N),  # Γ_mb ~ J√N
                "residual": float(np.sqrt(np.mean((yn-yf)**2)))
            }
        except Exception:
            scores["SYK"] = {"b": 0.1, "residual": 1.0}

        # Heisenberg: constant (null)
        scores["Heisenberg"] = {"residual": float(np.sqrt(np.mean((yn-1)**2)))}

        winner = min(scores, key=lambda fw: scores[fw]["residual"])
        return scores, winner

    # ── Step 4: Extract Γ_mb ────────────────────────────────────────────────

    def extract_gamma_mb(self, scores, winner):
        """Extract many-body dephasing rate Γ_mb from the winning framework."""
        if winner == "Lindblad":
            Gamma_eff = scores["Lindblad"]["Gamma_eff"]
            Gamma_mb = max(0, Gamma_eff - GAMMA_1_SR87)
            J_ex = Gamma_mb / self.N if self.N > 0 else 0
            return {"Gamma_eff": Gamma_eff, "Gamma_mb": Gamma_mb, "J_ex": J_ex}
        elif winner == "SYK":
            J_sc = scores["SYK"].get("J_scramble", 0)
            Gamma_mb = J_sc * math.sqrt(self.N)
            return {"Gamma_eff": GAMMA_1_SR87 + Gamma_mb, "Gamma_mb": Gamma_mb,
                    "J_scramble": J_sc}
        elif winner == "MBL":
            # MBL: slower than exponential — use effective rate at t=t_opt
            b = scores["MBL"]["b"]
            alpha = scores["MBL"]["alpha"]
            Gamma_eff_at_t = alpha * b / (1 + b * self.t_opt)  # instantaneous rate
            Gamma_mb = max(0, Gamma_eff_at_t - GAMMA_1_SR87)
            return {"Gamma_eff_at_t": Gamma_eff_at_t, "Gamma_mb": Gamma_mb,
                    "note": "MBL: power law, rate decreases with time"}
        return {"Gamma_eff": GAMMA_1_SR87, "Gamma_mb": 0.0}

    # ── Step 5: Compute feedback ────────────────────────────────────────────

    def compute_feedback(self, Gamma_mb_data, winner):
        """
        Given measured Γ_mb, compute:
        - Predicted F under actual decoherence
        - Optimal χt* accounting for decoherence
        - Updated Schmidt angles
        - Quantum advantage flag
        """
        Gamma_eff = Gamma_mb_data.get("Gamma_eff", GAMMA_1_SR87)

        # C after decoherence at t_opt
        C_dephased = self.C_ideal * math.exp(-4 * Gamma_eff * self.t_opt)
        F_predicted = (2 + C_dephased) / 3
        quantum_advantage = F_predicted > 2/3

        # Threshold check
        Gamma_star = 0.550 / self.N  # from v4 scaling law
        margin = Gamma_star / max(Gamma_eff, 1e-10)

        # Schmidt angles: same as ideal (U(1) symmetry makes them independent of Γ)
        theta_A = self.theta_A_ideal
        theta_B = self.theta_B_ideal

        return {
            "C_predicted": float(C_dephased),
            "F_predicted": float(F_predicted),
            "quantum_advantage": bool(quantum_advantage),
            "Gamma_star": float(Gamma_star),
            "safety_margin": float(margin),
            "theta_A_rad": float(theta_A),
            "theta_B_rad": float(theta_B),
            "theta_A_deg": float(math.degrees(theta_A)),
            "theta_B_deg": float(math.degrees(theta_B)),
            "chi_t_opt": float(self.chi_t_opt),
            "t_opt_sec": float(self.t_opt),
            "n_shots_recommended": int(9 / max((F_predicted - 2/3)**2, 1e-6)),
            "framework": winner,
        }

    # ── Main loop ────────────────────────────────────────────────────────────

    def run_loop(self, shots_array, tau_array):
        """
        Full 5-step hybrid loop.

        shots_array: np.ndarray (n_tau, n_shots) with values +1 or -1
        tau_array:   list/array of delay times in seconds
        Returns: feedback dict for JILA
        """
        tau = np.array(tau_array)

        print(f"\n  === Hybrid Loop: N={self.N} ===")

        # Step 1: Reconstruct witness
        witness_vals, uncertainties = self.reconstruct_witness(shots_array)
        print(f"  Step 1: W(0) = {witness_vals[0]:.4f} ± {uncertainties[0]:.4f}  "
              f"(ideal: {self.W0_ideal:.4f})")

        # Step 2: D-LinOSS fit
        omega_k, gamma_k, A_k, recon_err = self.dlinoss_fit(tau, witness_vals)
        dom = np.argmax(A_k)
        dom_gamma = float(gamma_k[dom]) if len(gamma_k) > 0 else 0.0
        print(f"  Step 2: D-LinOSS K_eff={len(gamma_k)}  "
              f"dom_γ={dom_gamma:.4f} s⁻¹  recon_err={recon_err:.4f}")
        expected_gamma = 4 * GAMMA_1_SR87
        print(f"          Expected (Markovian): γ_k = 4Γ₁ = {expected_gamma:.5f} s⁻¹")
        # Adaptive tau recommendation: need tau_max ≥ 3/dom_gamma for clean decay
        tau_max_current = tau[-1]
        tau_max_recommended = 3.0 / max(dom_gamma, expected_gamma, 1e-10)
        if tau_max_current < tau_max_recommended * 0.5:
            print(f"  [⚠] tau_max={tau_max_current:.4f}s < 3/γ={tau_max_recommended:.4f}s"
                  f" — extend range for reliable framework ID")

        # Step 3: Framework scoring
        scores, winner = self.score_frameworks(tau, witness_vals)
        print(f"  Step 3: Winner={winner}")
        for fw, s in scores.items():
            print(f"          {fw:12s}: residual={s['residual']:.4f}")

        # Step 4: Extract Γ_mb
        Gamma_mb_data = self.extract_gamma_mb(scores, winner)
        print(f"  Step 4: Γ_eff={Gamma_mb_data.get('Gamma_eff',0):.5f} s⁻¹  "
              f"Γ_mb={Gamma_mb_data.get('Gamma_mb',0):.5f} s⁻¹  "
              f"(Γ₁={GAMMA_1_SR87:.5f} s⁻¹)")

        # Step 5: Feedback
        feedback = self.compute_feedback(Gamma_mb_data, winner)
        print(f"  Step 5: F_predicted={feedback['F_predicted']:.4f}  "
              f"{'QUANTUM ADVANTAGE ✓' if feedback['quantum_advantage'] else 'NO ADVANTAGE ✗'}  "
              f"margin={feedback['safety_margin']:.1f}×")
        print(f"          θ_A={feedback['theta_A_deg']:.1f}°  θ_B={feedback['theta_B_deg']:.1f}°")
        print(f"          Recommended shots for next run: {feedback['n_shots_recommended']}")

        # Assemble full output
        result = {
            "N": self.N, "chi_Hz": self.chi_Hz,
            "witness_reconstructed": witness_vals,
            "tau_array": tau.tolist(),
            "uncertainties": uncertainties,
            "dlinoss": {
                "omega_k": omega_k.tolist(), "gamma_k": gamma_k.tolist(),
                "A_k": A_k.tolist(), "recon_err": recon_err,
            },
            "framework_scores": {fw: scores[fw] for fw in scores},
            "framework_winner": winner,
            "Gamma_mb_data": Gamma_mb_data,
            "feedback": feedback,
            "diagnostics": {
                "tau_max_used": float(tau[-1]),
                "tau_max_recommended": float(3.0 / max(dom_gamma, expected_gamma, 1e-10)),
                "tau_range_adequate": tau_max_current >= tau_max_recommended * 0.5,
                "sign_convention": "witness = mean(shots)/4  [corrected 2026-05-10]",
            }
        }
        return result


# ── Synthetic test (Monte Carlo shots from ideal ρ₂) ────────────────────────

def synthetic_jila_shots(rho2, W_op, n_shots=280, Gamma_eff=None,
                          tau_array=None, t_opt=1.0):
    """
    Generate synthetic JILA shots from the MPS-predicted ρ₂ under dephasing.
    Each shot: +1 if separable result, -1 if entangled result.
    (Sign convention: Tr[Wρ] < 0 for entangled, so entangled → -1)
    """
    if tau_array is None:
        tau_array = np.linspace(0, t_opt * 3, 20)
    if Gamma_eff is None:
        Gamma_eff = GAMMA_1_SR87  # use only known single-particle rate

    basis = [(0,0),(0,1),(1,0),(1,1)]
    dH = np.array([[abs(a-c)+abs(b-d) for c,d in basis] for a,b in basis], float)

    rng = np.random.default_rng(42)
    shots_array = np.zeros((len(tau_array), n_shots))

    for i, tau in enumerate(tau_array):
        rho_tau = rho2 * np.exp(-Gamma_eff * dH**2 * tau)
        rho_tau /= max(float(np.trace(rho_tau).real), 1e-14)
        W_expect = float(np.real(np.trace(W_op @ rho_tau)))
        # P(entangled outcome) = (1/4 - W_expect) / (1/2) — from projector structure
        p_entangled = (0.25 - W_expect) / 0.5
        p_entangled = np.clip(p_entangled, 0, 1)
        # +1 = separable, -1 = entangled
        outcomes = rng.choice([1, -1], size=n_shots,
                              p=[1-p_entangled, p_entangled])
        shots_array[i] = outcomes

    return shots_array, tau_array


def main():
    print("="*68)
    print("  JILA-TPU Controller: Synthetic Validation")
    print("  Monte Carlo shots from ideal ρ₂ + known Γ₁")
    print("="*68)

    # Test at N=4 (optimal experimental target)
    N, chi_Hz = 4, 1.0
    controller = JILATPUController(N=N, chi_Hz=chi_Hz)

    # Generate synthetic shots at 20 delay points
    tau_array = np.linspace(0.1, controller.t_opt * 2, 20)
    shots, tau = synthetic_jila_shots(
        controller.rho2_ideal, controller.W_op,
        n_shots=280, Gamma_eff=GAMMA_1_SR87,
        tau_array=tau_array, t_opt=controller.t_opt
    )

    # Run full loop
    result = controller.run_loop(shots, tau)

    print("\n" + "="*68)
    print("  VALIDATION SUMMARY")
    print("="*68)
    fb = result["feedback"]
    print(f"  N=4 synthetic shots (Γ_eff = Γ₁ = {GAMMA_1_SR87:.5f} s⁻¹)")
    print(f"  Framework identified: {result['framework_winner']}")
    print(f"  F_predicted = {fb['F_predicted']:.4f}  "
          f"(ideal: {controller.F_ideal:.4f})")
    print(f"  Quantum advantage: {fb['quantum_advantage']}")
    print(f"  Safety margin: {fb['safety_margin']:.1f}× over Γ₁")
    print(f"  θ_A = {fb['theta_A_deg']:.1f}°  θ_B = {fb['theta_B_deg']:.1f}°")

    with open("controller_validation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[DONE] controller_validation.json")
    print(f"\n  Ready for real JILA data integration.")
    print(f"  Replace synthetic shots_array with JILA measurement stream.")


if __name__ == "__main__":
    main()
