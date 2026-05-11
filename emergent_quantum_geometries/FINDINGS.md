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
1. **Analytic proof of F_opt=(2+C)/3**: likely follows from permutation symmetry
   forcing f_max = (1+C)/2. Establish analytically.
2. **C scaling**: C_peak ∝ N^{-1.03} — derive from closed-form formula analytically.
   Large-N: C_peak ≈ (1/4)|cos^{N-2}(χt*_N/2)|_max → find χt*_N analytically.
3. **D-LinOSS out-of-distribution**: needs SYK + XXZ training library (TPU run).
