# OAT Quantum Teleportation + MoQE Framework Discriminator
## Experiment Summary & Findings
**Branch:** `quantum-teleportation-results`
**Date:** 2026-05-09
**Hardware:** Google Cloud TPU v6e-8, europe-west4-a (TRC allocation, zero-cost)

---

## What We Did

### Experiment 1 — Exact OAT Teleportation Simulation (VALID)
Exact statevector simulation of the one-axis twisting (OAT) Hamiltonian:

```
H = χ J_z^A J_z^B
```

using diagonal unitary evolution (no Trotter error) on TPU v6e-8.

**Protocol:** Prepare |+⟩^⊗N, evolve under H, extract boundary pair (A_{N/2-1}, B_0),
compute Wootters concurrence C, teleportation fidelity F = (2+C)/3.

**Results (N = 2..14):**

| N | F_max | C_max | χt* | Above 2/3? |
|---|-------|-------|-----|------------|
| 2 | 0.976 | 0.928 | 3.14 | YES |
| 4 | 0.889 | 0.667 | 2.21 | YES |
| 6 | 0.819 | 0.458 | 1.74 | YES |
| 8 | 0.771 | 0.313 | 1.45 | YES |
| 10 | 0.740 | 0.220 | 1.24 | YES |
| 12 | 0.718 | 0.154 | 1.09 | YES |
| 14 | 0.701 | 0.102 | 0.97 | YES |

**Validated physics:**
- F > 2/3 (classical threshold) for all N ∈ [2, 14] ✓
- Concurrence scaling: C ~ N^{-1.43} ✓
- Optimal coupling time: χt* ~ N^{-1.316} ✓
- Fidelity formula F = (2+C)/3 (Bowen-Bose 2001) verified ✓

**Scientific status: CONFIRMED.** Quantum teleportation advantage demonstrated
through exact simulation. Reproducible from the committed code.

---

### Experiment 2 — MoQE Framework Discriminator (EXPLORATORY, NOT VALID YET)

**Goal:** Build a self-supervised "organic learner" that, given time-series
dynamics data, identifies which quantum framework (Heisenberg, SYK, MBL,
Lindblad, OAT, Penrose OR) the data naturally inhabits.

**Architecture (v3):**
- Layer 1 (Learner): D-LiNOSS damped oscillator SSM, reconstruction-only,
  completely blind to quantum physics. Learns latent z(t), ω_k, γ_k.
- Layer 2 (Observer): 7 pluggable formalism probes. Each asks:
  "Does z(t) satisfy my mathematical laws?"

**Probes implemented:**
1. Hamiltonian — Heisenberg EOM consistency: |dz/dt| / (Ω·|z|·dt) ~ 1
2. SYK — OTOC decay rate + Lyapunov exponent check
3. MBL — Level spacing ratio r (Poisson=0.386 localized vs GOE=0.53 chaotic)
4. Lindblad — BLP Markovianity measure (N_BLP = information backflow)
5. OAT — raw_mean > 2/3 (F > 2/3 unique positive offset, no other framework)
6. Penrose bi-twistor — Pfaffian Z^{AB} null surface residual (topological probe)
7. Novel — SVD effective dimension, Lyapunov, time-reversal, LLM narrative

**TPU Results (600 epochs, B=128, T=48, F=12):**
- OAT correctly identified: dominant score 0.752-0.840 ✓
- All other sources: not cleanly separated (Penrose OR / Lindblad over-generalizing)

## What Went Wrong

**Critical data integrity issue:** The MoQE discriminator is not using data
from the actual OAT simulation. The 6 data generators are hand-crafted
analytical approximations:

```
gen_oat:       F(t) ≈ (2 + |sin(χt)| * N^{-1.43}) / 3   [APPROXIMATION]
gen_syk:       exp(-λ_L t) * cos(ωt)                      [CARTOON]
gen_mbl:       random ω, small γ                           [CARTOON]
gen_lindblad:  uniform γ                                   [CARTOON]
gen_penrose_or: cos(ωt) * sudden_drop                      [CARTOON]
```

These are mathematically motivated but not derived from actual simulations
or experimental data. Testing a discriminator on patterns we designed to
be different is circular — it tells us nothing about real quantum systems.

**Conclusion on MoQE so far:** The architecture is sound and the OAT probe
works (raw_mean > 2/3 is a genuine unique signature), but the experiment
is not yet scientifically valid because the input data is synthetic.

---

## Hypothesis for Next Experiment

**H₀ (Null):** The D-LiNOSS learner, trained on real OAT simulation output
(F(χt), C(χt) for N=2..14), will produce a latent representation z(t) that
the observer cannot assign to any known quantum framework with confidence > 0.7.

**H₁ (Alternative):** The learner will produce z(t) that the OAT probe
identifies with confidence > 0.8, and the Penrose bi-twistor null surface
residual will be measurably higher than for non-entangled data, indicating
departure from the null surface consistent with F > 2/3 entanglement.

**Scientific threshold for confirmation:**
- OAT probe fit > 0.80
- All other probes < 0.50
- Penrose bi-twistor null residual > 0 for OAT data, ≈ 0 for separable data
- Result reproducible across 3 independent random seeds

**What we need:**
1. Feed real F(χt), C(χt) output from `jila_oat_exact_tpu.py` into the learner
2. Compare OAT probe score against same learner trained on synthetic non-OAT data
3. Measure bi-twistor null residual for entangled (OAT) vs separable (random) states

---

## Data Gap — Path to Real Experimental Data

For the discriminator to be scientifically meaningful, the data sources
need to be physically grounded. In order of accessibility:

| Source | Data | Maps to | Status |
|--------|------|---------|--------|
| OAT exact simulation | F(χt), C(χt) for N=2..14 | Already computed | **USE THIS FIRST** |
| SYK ED | G(t,t') exact diagonalization of H_SYK | C(t) = -Im[G(t,t')] | Implementable |
| MBL | Disordered XXZ chain, ED | ⟨σ_i(t)⟩ auto-correlator | Implementable |
| Lindblad | Lindblad master equation (qutip) | ρ(t) trace | Implementable |
| Rey Sr-87 | Spin squeezing F(t) measurement | Direct F(t), C(t) | Gold standard |
| Sycamore | Bitstring 2-pt + 4-pt correlators | C(t) | Gold standard |

The Sr-87 (Rey group) data is the cleanest external target: they literally
measure F(t) and spin squeezing parameters that map directly to the OAT
teleportation fidelity. That is the natural first real-data run.

---

## Files

| File | Status | Description |
|------|--------|-------------|
| `jila_oat_exact_tpu.py` | ✓ Valid | Exact OAT teleportation simulation |
| `generate_figures.py` | ✓ Valid | Publication figures from TPU results |
| `quantum_teleportation_experiment/REPORT.md` | ✓ Valid | OAT scientific report |
| `oat_moqe_tpu_run.py` | ⚠ Superseded | v1: pure physics scoring, degenerate |
| `oat_moqe_v3_tpu.py` | ⚠ Exploratory | v3: correct architecture, wrong data |
| `oat_moqe_v4_real_data.py` | 🔲 Next | v4: real OAT sim data + proper framework sims |

---

## Next Steps

1. **`oat_moqe_v4_real_data.py`** — Wire actual `jila_oat_exact_tpu.py` output
   into the learner. Test H₁ against H₀.
2. **SYK ED generator** — Replace cartoon gen_syk with exact diagonalization
   of H_SYK = i Σ_{i<j<k<l} J_{ijkl} χ_i χ_j χ_k χ_l
3. **MBL ED generator** — Replace cartoon gen_mbl with disordered XXZ chain
4. **Penrose bi-twistor calibration** — Establish what null residual value
   corresponds to a fully entangled vs separable 2-qubit state from the
   actual simulation boundary pair
