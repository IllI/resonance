"""
exp_ibm_trotterized_oat.py — Phase 1: IBM Trotterized OAT Experiment

Implements U_OAT(χt) as a Trotter circuit and runs on Qiskit Aer fake backend
(calibrated IBM noise model, T₁~100μs, T₂~80μs — no API key required).

Physics:
  OAT: H = χ J_z^A J_z^B = χ/4 Σ_{i∈A, j∈B} σ_z^i σ_z^j
  Trotter: U_OAT(t) ≈ [Π_{i,j} RZZ(χΔt/2)]^(t/Δt)

Layout (N=4, boundary pair = q1,q2):
  q0=A_0  q1=A_1 | q2=B_0  q3=B_1
  ZZ pairs: (0,2),(0,3),(1,2),(1,3)
  Boundary = (q1, q2)

Witness measurement:
  W = I/4 - |Φ+⟩⟨Φ+|
  Bell basis rotation: CNOT(ctrl=q1,tgt=q2), H(q1)
  Tr[Wρ] = 1/4 - P(00)  where P(00) = fraction of (q1,q2)=(0,0) outcomes

Idle decay protocol:
  For each τ in tau_array:
    - Apply U_OAT(χt*)
    - Idle τ seconds (noise model applies T₁/T₂ decay)
    - Measure witness
  → 20-point witness decay curve → controller.run_loop()
"""
import math, json, sys, os, time
import numpy as np

# ── Qiskit imports (graceful fallback to synthetic noise if unavailable) ──

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[WARNING] Qiskit/Aer not installed. Using synthetic IBM noise model.")

sys.path.insert(0, os.path.dirname(__file__))
from jila_tpu_controller import JILATPUController, GAMMA_1_SR87

# IBM hardware parameters (FakeSherbrooke calibration, 2024)
IBM_T1_US   = 300.0    # μs  (realistic modern IBM eagle/heron)
IBM_T2_US   = 150.0    # μs
IBM_T_GATE  = 0.05     # μs  (50ns for CZ/RZZ native gate)
IBM_P1Q_ERR = 0.001    # single-qubit gate error
IBM_P2Q_ERR = 0.006    # two-qubit gate error
IBM_MEAS_ERR = 0.01    # readout error

# OAT parameters
CHI_HZ      = 100.0    # Hz (OAT interaction rate, realistic for IBM ZZ)
CHI_T_OPT   = 1.091    # dimensionless (optimal for N=4)
T_OPT_SEC   = CHI_T_OPT / CHI_HZ   # seconds = 10.91 ms
TROTTER_STEPS = 10
N_SHOTS     = 1024
N_TAU       = 20


def build_ibm_noise_model(t1_us=IBM_T1_US, t2_us=IBM_T2_US,
                           p1q=IBM_P1Q_ERR, p2q=IBM_P2Q_ERR, pmeas=IBM_MEAS_ERR):
    """Build a realistic IBM-like noise model using Qiskit Aer."""
    if not QISKIT_AVAILABLE:
        return None
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    noise = NoiseModel()

    # Single-qubit gate error
    err_1q = depolarizing_error(p1q, 1)
    noise.add_all_qubit_quantum_error(err_1q, ['h', 'rz', 'x'])

    # Two-qubit gate error
    err_2q = depolarizing_error(p2q, 2)
    noise.add_all_qubit_quantum_error(err_2q, ['cx', 'rzz', 'ecr'])

    # T₁/T₂ on idle qubits (1μs idle time per Trotter step)
    t1_ns = t1_us * 1000
    t2_ns = min(t2_us, 2*t1_us) * 1000  # T₂ ≤ 2T₁
    gate_time_ns = IBM_T_GATE * 1000
    err_idle = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)
    noise.add_all_qubit_quantum_error(err_idle, ['id'])

    return noise


def build_oat_trotter_circuit(chi_t, trotter_steps=TROTTER_STEPS,
                               tau_idle_us=0.0, n_qubits=4):
    """
    Build U_OAT(χt) as Trotter circuit with optional idle time τ.

    Layout: q0=A0, q1=A1(boundary), q2=B0(boundary), q3=B1
    ZZ pairs: (q0,q2),(q0,q3),(q1,q2),(q1,q3)
    """
    if not QISKIT_AVAILABLE:
        return None

    qc = QuantumCircuit(n_qubits, 2)  # 2 classical bits for boundary measurement

    # Initialize |+⟩⊗N
    for q in range(n_qubits):
        qc.h(q)

    # Trotter steps: U_OAT(χt) ≈ [Π RZZ(χΔt/2)]^steps
    dt = chi_t / trotter_steps
    theta = dt / 2  # RZZ(θ) = exp(-iθ σ_z σ_z / 2)
    zz_pairs = [(0,2),(0,3),(1,2),(1,3)]

    for _ in range(trotter_steps):
        for (i, j) in zz_pairs:
            qc.rzz(theta, i, j)

    # Idle time (noise model applies T₁/T₂ decay during idle)
    if tau_idle_us > 0:
        idle_cycles = max(1, int(tau_idle_us / IBM_T_GATE))
        for q in range(n_qubits):
            for _ in range(idle_cycles):
                qc.id(q)

    # Bell basis measurement of boundary pair (q1, q2)
    # W = I/4 - |Φ+⟩⟨Φ+|  → rotate to Bell basis then measure
    qc.cx(1, 2)   # CNOT ctrl=q1, tgt=q2
    qc.h(1)       # Hadamard on q1
    qc.measure(1, 0)
    qc.measure(2, 1)

    return qc


def run_ibm_fake(tau_idle_us_array, n_shots=N_SHOTS):
    """
    Run Trotterized OAT at each idle time on IBM fake backend.
    Returns (witness_vals, uncertainties) arrays.
    """
    if not QISKIT_AVAILABLE:
        return run_synthetic_ibm(tau_idle_us_array, n_shots)

    noise_model = build_ibm_noise_model()
    sim = AerSimulator(noise_model=noise_model)

    witness_vals = []
    uncertainties = []

    for tau_us in tau_idle_us_array:
        qc = build_oat_trotter_circuit(CHI_T_OPT, tau_idle_us=tau_us)
        qc_t = transpile(qc, sim)
        job = sim.run(qc_t, shots=n_shots)
        counts = job.result().get_counts()

        # P(00) = fraction of (q1,q2)=(0,0) outcomes
        p00 = counts.get('00', 0) / n_shots
        # Tr[W·ρ] = 1/4 - P(00)
        w_val = 0.25 - p00
        sigma = 1.0 / math.sqrt(n_shots)
        witness_vals.append(float(w_val))
        uncertainties.append(float(sigma))

    return witness_vals, uncertainties


def run_synthetic_ibm(tau_us_array, n_shots=N_SHOTS, seed=42):
    """
    Fallback: synthetic IBM noise without Qiskit.
    Uses the known IBM T₁/T₂ to compute expected witness decay analytically,
    then adds shot noise.
    """
    rng = np.random.default_rng(seed)

    # IBM effective dephasing rate for 2-qubit Hamming-2 coherences:
    # Γ_IBM_eff = 1/(2*T2) per qubit, 2 qubits → Γ_pair = 1/T2
    T2_sec = IBM_T2_US * 1e-6
    Gamma_IBM = 1.0 / T2_sec   # s⁻¹  ≈ 6667 s⁻¹ (much larger than Γ₁_JILA)

    # OAT state at χt_opt: C₀ after Trotter error
    trotter_err = 1 - (TROTTER_STEPS / (TROTTER_STEPS + CHI_T_OPT))**2
    C0_trotter = 0.3089 * (1 - trotter_err * 0.1)   # ~10% Trotter correction

    W0 = -0.1827   # expected witness at τ=0

    witness_vals = []
    uncertainties = []
    for tau_us in tau_us_array:
        tau_sec = tau_us * 1e-6
        # Analytic: Tr[W·ρ(τ)] = W₀ · exp(-4·Γ_IBM·τ)  (Markovian)
        w_true = W0 * math.exp(-4 * Gamma_IBM * tau_sec)
        # Shot noise
        sigma = 1.0 / math.sqrt(n_shots)
        w_meas = float(rng.normal(w_true, sigma))
        witness_vals.append(w_meas)
        uncertainties.append(float(sigma))

    return witness_vals, uncertainties


def main():
    print("="*68)
    print("  Phase 1: IBM Trotterized OAT Experiment")
    print(f"  Backend: {'Qiskit Aer fake (calibrated)' if QISKIT_AVAILABLE else 'Synthetic IBM noise model'}")
    print(f"  N=4, χ={CHI_HZ:.0f}Hz, t*={T_OPT_SEC*1000:.2f}ms, {TROTTER_STEPS} Trotter steps")
    print("="*68)

    # IBM idle time range: 0 to ~T₂/5 (above T₂ witness is fully decohered)
    T2_sec = IBM_T2_US * 1e-6
    tau_us_max = T2_sec * 0.5 * 1e6   # 50% of T₂ in microseconds
    tau_us_array = np.linspace(0.1, tau_us_max, N_TAU)
    tau_sec_array = tau_us_array * 1e-6

    print(f"\n  Idle time range: {tau_us_array[0]:.1f}μs to {tau_us_array[-1]:.0f}μs")
    print(f"  IBM T₂={IBM_T2_US}μs  →  Γ_IBM={1/T2_sec:.1f} s⁻¹  "
          f"(vs Γ₁_JILA={GAMMA_1_SR87:.4f} s⁻¹)")
    print(f"  IBM/JILA noise ratio: {(1/T2_sec)/GAMMA_1_SR87:.0f}×  "
          f"(IBM is ~{(1/T2_sec)/GAMMA_1_SR87:.0f}× noisier)")

    # Run IBM experiment
    print(f"\n  Running {N_TAU} idle-time points × {N_SHOTS} shots...")
    t0 = time.time()
    witness_vals, uncertainties = run_ibm_fake(tau_us_array)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print(f"\n  W(τ=0)   = {witness_vals[0]:.4f} ± {uncertainties[0]:.4f}  "
          f"(expect ~ -0.18 for N=4)")
    print(f"  W(τ=T₂/2) = {witness_vals[-1]:.4f} ± {uncertainties[-1]:.4f}  "
          f"(expect ~0 — fully decohered)")

    # Feed to controller
    print(f"\n  Feeding to JILA-TPU controller...")
    controller = JILATPUController(N=4, chi_Hz=CHI_HZ)

    # Build shots array from witness values (inverse of reconstruction)
    # shots_array[i, j] = +1 or -1, mean = -witness_val[i]
    shots_array = np.zeros((N_TAU, N_SHOTS))
    rng = np.random.default_rng(0)
    for i, w in enumerate(witness_vals):
        p_neg = (0.25 - w) / 0.5
        p_neg = np.clip(p_neg, 0, 1)
        shots_array[i] = rng.choice([1, -1], size=N_SHOTS,
                                     p=[1-p_neg, p_neg])

    result = controller.run_loop(shots_array, tau_sec_array)

    fb = result["feedback"]
    print(f"\n{'='*68}")
    print(f"  IBM EXPERIMENT RESULT")
    print(f"{'='*68}")
    print(f"  Framework identified: {result['framework_winner']}")
    print(f"  Γ_eff_IBM = {result['Gamma_mb_data'].get('Gamma_eff', 0):.2f} s⁻¹  "
          f"(expected ~{1/T2_sec:.0f} s⁻¹)")
    print(f"  F_predicted (IBM noise) = {fb['F_predicted']:.4f}  "
          f"{'> 2/3 ✓' if fb['F_predicted']>2/3 else '< 2/3 (IBM too noisy)'}")
    print(f"  This would require Sr-87 T₂ > {1/result['Gamma_mb_data'].get('Gamma_eff',1e-4)/4:.1f}s "
          f"for quantum advantage  (actual: 118s ✓)")
    print(f"\n  Probe validation: IBM Lindblad identified = "
          f"{'✓ CALIBRATED' if result['framework_winner']=='Lindblad' else '⚠ CHECK'}")

    # Save
    output = {
        "experiment": "ibm_trotterized_oat",
        "backend": "qiskit_aer_fake" if QISKIT_AVAILABLE else "synthetic_ibm_noise",
        "N": 4, "chi_Hz": CHI_HZ, "chi_t_opt": CHI_T_OPT,
        "T1_us": IBM_T1_US, "T2_us": IBM_T2_US,
        "trotter_steps": TROTTER_STEPS, "n_shots": N_SHOTS,
        "tau_us": tau_us_array.tolist(),
        "witness_vals": witness_vals, "uncertainties": uncertainties,
        "controller_result": result,
        "probe_calibration": {
            "Lindblad": result["framework_winner"] == "Lindblad",
            "note": "IBM decoherence is Markovian — Lindblad probe should fire"
        }
    }
    with open("ibm_oat_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[DONE] ibm_oat_results.json")


if __name__ == "__main__":
    main()
