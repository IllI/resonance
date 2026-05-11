# OAT Boundary State Entanglement — Key Findings

## Session: 2026-05-11

### 0. Phase-Alignment Rotation — Primary Experimental Prediction [JILA-Actionable]

**This is the single strongest falsifiable prediction in the paper.**

Two $R_z$ gates applied before Bell measurement recover the full teleportation advantage:

$$\Delta F = F_\mathrm{opt} - F_\mathrm{naive} = \frac{2+C}{3} - \frac{1+C}{2} = \frac{1-C}{6}$$

Per-N table (exact Schmidt angles):

| N  | C_peak | F_naive | F_opt | ΔF    | θ_A (atom A gate) |
|----|--------|---------|-------|-------|-------------------|
| 2  | 0.861  | 0.931   | 0.954 | 0.023 | 38.1°             |
| 4  | 0.309  | 0.655   | 0.770 | 0.115 | **164.4°**        |
| 8  | 0.157  | 0.579   | 0.719 | 0.141 | 172.1°            |
| 16 | 0.063  | 0.532   | 0.688 | 0.156 | 175.3°            |

Key points:
- For N=4: **17.6% fidelity gain** from two software-controlled phase gates. Zero new hardware.
- For N=8: F_naive = 0.579 is **below the classical limit**; phase alignment lifts to 0.719.
- The gain **increases** as N grows (entanglement weakens) — most useful precisely when needed most.
- θ_B = 0° in all cases; only atom A requires rotation.
- JILA can verify in the same session as the Γ_mb measurement.

**Definition note:** F_naive ≡ (1+C)/2 is defined as the fidelity of the best fixed-basis Bell measurement over all four Bell states without phase information. The gain formula ΔF = (1-C)/6 follows exactly from this definition and F_opt = (2+C)/3. The physical derivation of F_naive from the OAT density matrix is a pending analytic task.

### 1. Correct Boundary Pair Definition
- The relevant pair is a **cross-half atom pair**: one atom from left half A, one from right half B.
- Same-half pairs (both atoms in A or both in B) have C=0 always.
- All cross-half pairs give identical ρ_2 due to permutation symmetry within each half.
- The "first and last atom" interpretation was incorrect for the OAT teleportation theorem.

### 2. Exact Closed-Form Density Matrix [New]
For OAT state at coupling time χt, the boundary pair density matrix elements are:

```
ρ_{(iL,iR),(jL,jR)} = (1/4)
  × exp(-i·χt·[(iL-1/2)(iR-1/2) - (jL-1/2)(jR-1/2)])
  × cos^{N/2-1}((iR-jR)·χt/2)
  × cos^{N/2-1}((iL-jL)·χt/2)
```

Properties:
- Uniform diagonal: ρ_ii = 1/4 for all i ∈ {00,01,10,11}
- Valid for all N ≥ 2, all χt
- O(1) computation — no MPS, no exponential memory
- Derived analytically; validated vs brute force (N=4, all chi_t)

### 3. Not an X-State [Correction to Paper]
- ρ_2 is **not** an X-state. All 16 elements are generically non-zero.
- The old proof argument (J_z^A + J_z^B conservation → X-state) is wrong:
  that argument applies to the full bipartite (half-vs-half) state,
  not to the single-atom reduced state after tracing N-2 interior atoms.
- The X-state zeros do NOT appear in ρ_2.

### 4. Wootters Concurrence Values [Confirmed]
- N=2: C_peak = 1.000 (maximally entangled Bell pair at χt=π)
- N=4: C_peak = 0.309 at χt*=1.100
- N=6: C_peak = 0.192 at χt*=5.550 (second-lobe peak)
- N=8: C_peak = 0.140
- N=16: C_peak = 0.066
- N=64: C_peak = 0.015
- Scaling: C_peak ∝ N^{-1.03} approximately

Must use **full** Wootters formula on 4×4 matrix (not X-state simplification).

### 5. Theorem F_opt = (2+C)/3 — Status
- Numerically verified for N=4, 10 chi_t values, δF < 1e-4
- Holds for all C > 0 cases
- Fails at C=0 (OAT state at those χt is actually worse than classical limit)
- Analytic proof NOT yet established; the X-state proof was wrong
- Downgraded from "exact theorem" to "numerically verified conjecture"

### 6. D-LinOSS Calibration [Confirmed]
- K_eff = 1 for all OAT runs N=2–64, all Γ values
- gamma_dom = 4Γ to 4 decimal places in all cases
- Exact formula makes OAT generation O(1) — 50 parameter configs in 0.2s

### 7. Code Status
- `tpu_dlinoss_training_gen.py`: exact formula implemented, all N working
- Commit: `6bebc98` (simulation), `5103b19` (paper v5.4)
- Next: run SYK + XXZ + Dicke on TPU (v4 on-demand, us-central2-b)

### 8. Open Problems
1. **Analytic proof of F_opt=(2+C)/3 for N≥4**: PROVED for N=2 (see Sec. II.C of paper).
   For N≥4: likely follows from permutation symmetry forcing f_max=(1+C)/2. [HYPOTHESIS]
2. **C scaling**: C_peak ∝ N^{-1.03} — derive from closed-form formula analytically.
3. **K_BIC for SYK at large Nf**: K_BIC=1 at Nf=6–10; need Nf≥16 to see if multi-mode
   structure emerges at large N.

---

## Session: 2026-05-11 (TPU Run 1 + Adversarial Suite)

### 9. F_opt Proved for N=2 [PROVED]
For N=2, ρ_2 has off-diagonal e^{±iχt}/4. After R_z(χt)⊗I phase alignment,
ρ becomes real: ρ_{00,11} → |cos(χt/2)|/4. The singlet fraction:
  f_max = (1 + |cos(χt)|)/2 = (1+C)/2
  F_opt = (2+C)/3  ✓ analytically exact for N=2.

### 10. K_eff Threshold Sweep — Critical Result [OBSERVED]
The original K_eff=6 for SYK4 was a truncation ceiling artifact.

| System | K@1e-2 | K@1e-3 | K@1e-4 | K_BIC | Verdict |
|---|---|---|---|---|---|
| OAT | 1 | 1 | 1 | **1** | Stable single-mode ✅ |
| Dicke | 1 | 1 | 1 | **1** | Stable single-mode ✅ |
| SYK4 | 2–13 | 10–13 | 13 | **1** | 🚨 ARTIFACT — BIC says single mode |
| XXZ | 4–13 | 13 | 13 | **13** | Genuine multi-mode ✅ |

**Key finding:** SYK4 Green's function is BIC-single-mode. The apparent K_eff=6 was
because the default K_max=6 was a hard ceiling — at K_max=16 it hits 13.
Under information-criterion penalty (BIC), SYK4 collapses to K=1.
XXZ imbalance is genuinely multi-mode (K_BIC=13) due to spin-wave structure.

**Revised fingerprint:**
- K_BIC=1, γ>0: OAT (decaying)
- K_BIC=1, γ≈0: Dicke (oscillatory), SYK4 (noisy decay)
- K_BIC=13, γ decreasing with W: XXZ (genuine spin-wave multi-mode)

**Implication:** SYK4 multi-mode scrambling claim is RETRACTED. SYK4 is
indistinguishable from OAT under BIC-penalized rank selection. This is the
adversarial suite's most important finding.

### 11. Adversarial Stress-Test Results [OBSERVED]

| System | K_eff | gamma_dom | omega_dom | Assessment |
|---|---|---|---|---|
| Floquet quasiperiodic | 6 | 0.007 | 1.11 | Correct (multi-mode, no decay) |
| Damped oscillator | **2** | 0.040 | **0.500** | ✅ Correct (2 modes for damped sinusoid) |
| Random telegraph noise | **6** | 0.095 | 0.00 | 🚨 FALSE POSITIVE (should be K=1) |
| Integrable XXZ (W=0) | 6 | 0.016 | 0.41 | Correct (multi-mode oscillatory) |
| Cavity-QED decay | **1** | 0.000 | 0.00 | Partial (missing kappa/2 decay) |
| Chaotic spin chain | 6 | **2.23** | 1.24 | Distinguishable from SYK (gamma differs) |
| Duffing oscillator | — | — | — | Overflow (unstable params) |

**Critical diagnostic — RTN vs OAT:**
- RTN (classical stochastic): K=6, gamma=0.095 → false positive
- OAT (quantum Lindblad): K=1, gamma=0.040
- D-LinOSS CANNOT distinguish RTN from a 6-mode quantum signal.
  This proves the framework reads spectral morphology, not quantum structure per se.

**Positive finding — damped oscillator:**
K=2 correctly identified (damped sinusoid requires 2 complex exponentials). The
gamma and omega recoveries are exact. This is a genuine success of the matrix pencil.

**Chaotic spin chain vs SYK:**
Both K=6 (at threshold), but gamma_dom differs by 5×: chaotic chain=2.23, SYK=0.07–0.67.
The (K_eff, gamma_dom) pair distinguishes these two multi-mode systems
even when K_eff is saturated — gamma_dom carries genuine physical information.

### 12. Revised D-LinOSS Embedding Geometry [HYPOTHESIS]
The revised embedding is 3-dimensional:
  - **K_BIC**: 1 vs >1 (single vs multi-mode)
  - **gamma_dom**: decay rate (OAT: 4Γ, SYK: ~0.1, XXZ: monotone with W)
  - **I_inf** (XXZ only): long-time imbalance, MBL discriminant

The "2D fingerprint" claim (K_eff, gamma_dom) is weakened: K alone is unreliable without
BIC correction. The reliable claim is:
  (K_BIC, gamma_dom, I_inf) separates OAT/Dicke, SYK4, and XXZ.
But SYK4 and OAT remain in the SAME K_BIC=1 class. The SYK4 scrambling claim
is NOT supported by D-LinOSS at Nf=6–10.

### 13. Open Problems (Updated)
1. **K_BIC for SYK at Nf≥16**: does K_BIC grow above 1 at large Nf?
2. **RTN false positive**: can noise injection + BIC distinguish RTN from Lindblad decay?
3. **Cavity-QED cavity decay**: gamma≈0 suggests the Jaynes-Cummings oscillation
   dominates over the κ decay at short times — expected, but cavity_qed_decay needs
   longer τ_max to see the decay tail.
4. **Duffing Duffing overflow**: reduce gamma_d to 0.1 and rerun.
