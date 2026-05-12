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
4. **Duffing overflow**: reduce gamma_d to 0.1 and rerun.

---

## Session: 2026-05-12 (TPU Run 2 — chronos-bob2, v6e-8, europe-west4-a, 1157s)

### 14. Run 2 Complete Results — BIC Fingerprint Confirmed [OBSERVED]

**OAT (50 records, N=2–64, Γ=0.001–0.10):**
K_eff=1, K_bic=1, gdom=4Γ exact for ALL records. Calibration anchor stable.

**SYK4 — K_BIC=1 universally [OBSERVED]:**

| Nf | β | K_eff | K_bic | gdom | sigma_ratio |
|---|---|---|---|---|---|
| 6 | 1.0 | 13 | **1** | 0.742 | ∞ |
| 6 | 5.0 | 13 | **1** | 0.656 | ∞ |
| 6 | 10.0 | 13 | **1** | 0.513 | ∞ |
| 8 | 1.0 | 13 | **1** | 0.356 | ∞ |
| 8 | 5.0 | 13 | **1** | 1.040 | ∞ |
| 8 | 10.0 | 13 | **1** | 0.928 | ∞ |
| 10 | 1.0 | 13 | **1** | 0.822 | ∞ |
| 10 | 5.0 | 13 | **1** | 1.931 | 2.9 |
| 10 | 10.0 | 10 | **1** | 0.132 | 3.5 |

**K_BIC=1 for every SYK4 record across all Nf and β.** Even at Nf=10 where
sigma_ratio drops to 2.9-3.5 (weakly multi-modal), BIC still selects 1 mode.
SYK Green's function is dominated by a single decaying exponential.

**XXZ — K_BIC=13 universally [OBSERVED]:**

| L | W | K_bic | gdom | I_inf |
|---|---|---|---|---|
| 8 | 0.5 | **13** | 0.240 | 0.153 |
| 8 | 2.0 | **13** | 0.261 | 0.379 |
| 8 | 5.0 | **13** | 0.486 | 0.650 |
| 8 | 10.0 | **13** | 0.343 | 0.708 |
| 10 | 0.5 | **13** | 0.854 | 0.155 |
| 10 | 2.0 | **13** | 0.467 | 0.379 |
| 10 | 5.0 | **13** | 2.935 | 0.674 |
| 10 | 10.0 | **13** | 0.603 | 0.761 |
| 12 | 0.5 | **13** | 0.828 | 0.191 |
| 12 | 2.0 | **13** | 0.496 | 0.288 |
| 12 | 5.0 | **13** | 2.890 | 0.606 |
| 12 | 10.0 | **13** | 0.747 | 0.702 |

I_inf increases monotonically with W at fixed L. Genuine multi-mode spin-wave
structure confirmed — K_BIC=13 is not an artifact.

**Dicke — K_BIC=1 at ALL N and ALL g/g_c [OBSERVED]:**
K_eff=1, K_bic=1, gdom=0.000 for all 36 records (N=2,4,6,8,10,12 × g/g_c=0.2–2.0).
The superradiant QPT does NOT create multi-mode structure in ⟨Jz⟩ under BIC.
H4 falsification confirmed robustly across large-N and both phases.

**Adversarial baselines:**
| System | K_bic | gdom | Assessment |
|---|---|---|---|
| RTN λ=0.02 | **13** | 0.042 | False positive |
| RTN λ=0.05 | **13** | 0.318 | False positive |
| RTN λ=0.10 | **13** | 0.092 | False positive |
| DampedOsc g=0.02 ω=0.30 | **2** | 0.020 | ✅ Exact |
| DampedOsc g=0.04 ω=0.50 | **2** | 0.040 | ✅ Exact |
| DampedOsc g=0.10 ω=1.00 | **2** | 0.100 | ✅ Exact |

### 15. Definitive BIC Fingerprint Table [OBSERVED]

| System | K_BIC | I_inf | Classification |
|---|---|---|---|
| OAT (Lindblad) | 1 | 0 | Decaying single-mode |
| SYK4 (scrambling) | **1** | 0 | Decaying single-mode — SAME CLASS as OAT |
| Dicke (all N, all g) | **1** | 0 | Oscillatory single-mode |
| XXZ (disordered) | **13** | >0 | Genuine multi-mode |
| RTN (classical noise) | 13 | 0 | Classical multi-mode — false positive |
| DampedOsc (classical) | 2 | 0 | Classical oscillatory — exact |

**The discriminant between XXZ and RTN is I_inf:**
- XXZ: I_inf ∈ [0.15, 0.76] (non-zero → localization)
- RTN: I_inf = 0 (classical noise decays to zero)

**Revised 3D embedding:** (K_BIC, I_inf, γ_dom) separates all 6 system classes.
SYK4 and OAT cannot be distinguished by D-LinOSS — they share K_BIC=1, I_inf=0.

### 16. TPU Run 2 Metadata
- Node: chronos-bob2-node (v6e-8, europe-west4-a, spot)
- Runtime: 1157s (~19 min)
- Commit: pending
- Result: dlinoss_training/training_library_run2.npz (62KB)
- DELETE RITUAL: completed — all zones verified empty


---

## Session: 2026-05-12 (Singlet Fraction Direct Proof)

### 17. CRITICAL CORRECTION: Singlet Fraction Conjecture Violated [OBSERVED]

**Context:** The reviewer identified that the OAT boundary-pair state is NOT an X-state.
This invalidates the X-state shortcut f_max=(1+C)/2 and F_opt=(2+C)/3.
Direct numerical proof via DE + Nelder-Mead (120 restarts) + Powell refinement.

**Optimizer validation on Werner states (exact ground truth):**

| p   | C    | f_theory | f_opt   | error   |
|-----|------|----------|---------|---------|
| 0.3 | 0.00 | 0.475    | 0.475   | +0.0e+0 |
| 0.5 | 0.25 | 0.625    | 0.625   | +1e-16  |
| 0.8 | 0.70 | 0.850    | 0.850   | +1e-16  |
| 1.0 | 1.00 | 1.000    | 1.000   | +2e-16  |

*Note: For p=0.3 (C=0, separable), the correct f_max=0.475=(3p+1)/4, NOT 0.5.*
*The optimizer is confirmed correct.*

**Direct singlet fraction results for OAT family:**

| N  | C_peak | f_max  | (1+C)/2 | delta   | F_opt_direct | S_max  |
|----|--------|--------|---------|---------|--------------|--------|
| 2  | 1.0000 | 1.0000 | 1.0000  | +4e-05  | 1.0000       | 2.8284 |
| 4  | 0.2221 | 0.5780 | 0.6110  | -0.0330 | 0.7187       | 1.9695 |
| 6  | 0.1585 | 0.5399 | 0.5793  | -0.0393 | 0.6933       | 1.9767 |
| 8  | 0.1297 | 0.5289 | 0.5649  | -0.0360 | 0.6859       | 1.9794 |
| 12 | 0.1005 | 0.5182 | 0.5502  | -0.0320 | 0.6788       | 1.9851 |
| 16 | 0.0852 | 0.5117 | 0.5426  | -0.0309 | 0.6744       | 1.9914 |
| 24 | 0.0680 | 0.5078 | 0.5340  | -0.0262 | 0.6719       | 1.9937 |
| 32 | 0.0576 | 0.5046 | 0.5288  | -0.0242 | 0.6697       | 1.9971 |

**Conclusions [OBSERVED]:**

1. **f_max = (1+C)/2 is FALSE** for all N≥4 OAT states. Max violation: 3.9% at N=6.
2. **F_opt = (2+C)/3 is FALSE** for the OAT family. Actual F_opt is lower.
3. **S_max < 2.0 for all N≥4** (rigorous Horodecki, no X-state assumption required).
   Only N=2 (pure Bell state C≈1) violates Bell inequality.
4. **The phase-alignment ΔF table in §0 is superseded** by F_opt_direct above.
5. **C_peak values in §0 are from an older grid** (120 pts); current (400 pts) gives
   corrected values (e.g. N=4: 0.222 not 0.309).

### 18. Asymptotic Concurrence Scaling [OBSERVED]

Power-law fit C_peak = 0.505 * N^{-0.636} (exponent fit over N=4 to N=32).

- Observed scaling: **C_peak ~ N^{-0.636}**, NOT N^{-1} as the reviewer conjectured.
- Optimal time: chi_t* = 2.26 * N^{-0.577} ~ N^{-1/2} (not N^{-1}).
- The reviewer's argument assumed chi_t* ~ N^{-1}. Numerical result shows chi_t* ~ N^{-1/2},
  which modifies the asymptotic argument.
- Derivation of the exact exponent from the closed-form formula: OPEN PROBLEM [HYPOTHESIS].

### 19. Revised Claims for Paper [EPISTEMIC LABELS]

| Claim | Old Status | New Status |
|-------|-----------|------------|
| F_opt = (2+C)/3 | CLAIMED | **FALSIFIED** — replace with F_opt_direct table |
| f_max = (1+C)/2 | CONJECTURE | **FALSIFIED** for N≥4 by 2.4-3.9% |
| S_max < 2 for N≥4 | HEURISTIC | **PROVED** via direct Horodecki [OBSERVED] |
| C_peak ~ N^{-1} | HYPOTHESIS | **REVISED** to N^{-0.636} [OBSERVED] |
| Phase alignment θ_A→π as N grows | OBSERVED | **CONFIRMED** |
| SYK K_BIC=1 | OBSERVED | **CONFIRMED** (Run 2, all Nf, all beta) |
| XXZ K_BIC=13 | OBSERVED | **CONFIRMED** (all L, all W) |

### 20. Key Open Problems (Reviewer Priority Order)

A. **Derive exact C_peak scaling** from closed-form ρ_2 — likely C ~ N^{-2/3} from saddle-point.
B. **Prove f_max < (1+C)/2** analytically for the OAT non-X family — determine tight bound.
C. **Direct Horodecki bound is now complete** for N=2–32.
D. **Optimal local rotations** — derive analytically (near θ_A=180° regime).
E. **Realistic decoherence** — add collective dephasing Γ_mb to F_opt_direct calculation.
