# OAT Quantum Teleportation -- Research Findings

## Branch: `quantum-teleportation-results`
**Last updated:** 2026-05-10  
**Latest commit:** `51a56fe`  
**Hardware:** Google Cloud TPU v6e-8, europe-west4-a (TRC allocation)

---

## Summary

We establish that OAT boundary entanglement is a viable teleportation resource for JILA's $^{87}\text{Sr}$ apparatus. The digital twin (MPS on TPU) is exact, the hybrid loop controller is ready, and all three framework probes (Lindblad, SYK, MBL) are calibrated. The only remaining gap before JILA contact is the measurement of the many-body dephasing rate $\Gamma_{\text{mb}}$.

---

## Confirmed Results

### 1. MPS Digital Twin (v3)
The Matrix Product State (MPS) construction is numerically exact for OAT states.
- **Accuracy:** $\max |\rho_2^{\text{MPS}} - \rho_2^{\text{exact}}|_F < 3 \times 10^{-8}$ for all $N=2$ to $16$.
- **Performance:** $N=64$ simulated in seconds on TPU.

### 2. $F_{\text{opt}} = (2+C)/3$ is Exact for OAT (not an approximation)
The $U(1)$ symmetry of the OAT Hamiltonian forces the Bell fractions to satisfy $f(\Phi^+) = f(\Psi^+)$ and $f(\Phi^-) = f(\Psi^-)$ exactly, which implies $f_{\text{max}} = (1+C)/2$ and therefore $F_{\text{opt}} = (2+C)/3$ **exactly** for this Hamiltonian family. This is proved, not assumed.

### 3. Quantum Advantage Survives JILA Dephasing (v4)
At $\Gamma_1 = 1/118$ s$^{-1}$ (Sr-87, Dr. Rey arXiv:2505.06444):
- **N=4:** Peak $F \approx 0.761$ after decoherence, safety margin $17\times$ over threshold $\Gamma^*$.
- **Threshold scaling:** $\Gamma^*(N) = 0.550 N^{-1.03}$ s$^{-1}$ ($R^2=0.993$).

### 4. Entanglement Witness is the Correct Observable (v5d)
- Wootters concurrence $C(t)$ is nonlinear in $\rho$, producing artifacts for mixed states.
- Entanglement witness $W = I/4 - |\Psi^+\rangle\langle\Psi^+|$ is linear, showing exact $\exp(-4\Gamma t)$ decay.
- **D-LinOSS on witness:** Returns $b = 4.000 \pm 0.001$ for all $N=2$ to $16$. **Lindblad confirmed.**

### 5. Hybrid Loop Controller Validated (`jila_tpu_controller.py`)
Five-step loop (< 1s on TPU):
1. Ingest JILA shots $\to$ $\text{Tr}[W \rho(\tau)]$ via **mean(shots)/4** (sign-corrected).
2. D-LinOSS fit $\to$ $(\omega_k, \gamma_k, A_k)$.
3. Framework scoring: Lindblad / MBL / SYK residuals.
4. Extract $\Gamma_{\text{mb}} = \Gamma_{\text{eff}} - \Gamma_1$.
5. Feedback: $\theta_A$, $\theta_B$, $\chi t^*$, $F_{\text{predicted}}$, quantum advantage.

**Synthetic validation ($N=4$):**
- $F_{\text{predicted}} = 0.7687$ vs $F_{\text{ideal}} = 0.7696$ (0.1% error).
- $\theta_A = 164.4^\circ$, $\theta_B = 0.0^\circ$ (exact).
- Framework correctly identified as **Lindblad**.

---

## Hardware Probe Calibration

### Lindblad Probe: CONFIRMED
- **IBM Qiskit Aer circuit:** $W(0)_{\text{IBM}} = -0.160$ (12.3% error from Trotter + gates), correct decay rate.
- **JILA prediction:** $W(0)_{\text{JILA}} \approx -0.181$ (within 1% of ideal).

### SYK Probe: CALIBRATED (composite=0.790)
- **Input:** SYK retarded Green's function $G_R(t)$ from exact diagonalization.
- **Fingerprint:** $K_{\text{eff}} = 2$, amplitude flatness = 1.000.
- SYK is clearly distinguishable from Lindblad (residual 0.045).

### MBL Probe: CALIBRATED
- **Experiment:** disordered XXZ chain ($L=12, W/J=5$).
- **Observable:** charge imbalance $I(t)$.
- **Fingerprint:** $\gamma_{k,dom} = 0.015 J$ (near-zero) and $I(\infty) = 0.613$.

---

## Pipeline Validation Status

| Component | Status | Evidence |
|-----------|--------|----------|
| MPS digital twin | OK Exact | $<3 \times 10^{-8}$ error |
| $F_{\text{opt}} = (2+C)/3$ | OK Proved | $U(1)$ symmetry |
| Witness decay (v5d) | OK Confirmed | $b=4.000$ all $N$ |
| Sign convention | OK Fixed | mean/4, $\sigma = 0.5/\sqrt{N}$ |
| Lindblad probe | OK Confirmed | controller + v5d |
| SYK probe | OK Calibrated | ED $G_R(t)$, composite=0.790 |
| MBL probe | OK Calibrated | $I(\infty)=0.613$, $\gamma_{k,dom}$ discriminator |
| JILA readiness | 1 gap | $\Gamma_{\text{mb}}$ unmeasured |

---

## Code Artifacts

| File | Description |
|------|-------------|
| `src/simulation/oat_teleport_v3_tpu.py` | MPS construction, N=64 capable |
| `src/controller/jila_tpu_controller.py` | 5-step hybrid loop, sign-corrected |
| `src/experiments/exp_syk_ed_adapter.py` | SYK ED adapter |
| `src/experiments/exp_mbl_xxz_adapter.py` | MBL probe via disordered XXZ ED |
| `docs/PAPER_DRAFT_v1.md` | Full scientific draft with figures |
| `docs/CODE_GUIDE.md` | Developer manual |
