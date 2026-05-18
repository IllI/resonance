# Program O — Encoded-State Recoverability

**Builds on:** Program L / N-series (transport recoverability)  
**Branch:** `quantum-teleportation-results`  
**Status:** Design phase (post N10a)

---

## Motivation

Program N demonstrated a clean control-phase hierarchy:

| Regime | Adaptive behavior | η_ctrl |
|--------|-------------------|--------|
| XXZ    | strongly beneficial | ~+0.56 |
| W1     | beneficial, sparse | ~+0.37 |
| W3     | intervention-neutral boundary | ~−0.10 |
| OAT    | learned abstention | undefined |

The current metric `τ_transport` (recoverability lifetime) answers:
> "Does the system stay in a transport-capable state longer?"

Program O asks the next question:
> "Does that actually preserve transmitted quantum information?"

This is the bridge to information-theoretic language and IBM hardware relevance.

---

## Pre-Registered Falsification Criteria (FROZEN before first run)

```
TH_F_REC     = 2/3      inherited from Program L
TH_DF_AVG    = 0.02     minimum meaningful ΔF_avg (above measurement noise)
TH_PROBE_ADD = 0.01     probe must add > 1% F improvement over no-probe baseline
```

Primary claim (falsifiable):
- `ΔF_avg > TH_DF_AVG` in XXZ and W1 under agnostic controller
- `ΔF_avg ~ 0` in W3 (transition) and OAT (stable basin)
- `F_probe+adaptive > F_adaptive_only` in at least 1 transport-capable regime

---

## Phase 1 — Response Characterization (Chirp Probe)

**What:** Short chirped pulse sequence (4–6 pulses) to estimate local response Jacobian  
**Why:** Replaces fixed `_CLIFF_OBS_DELTAS` with measured J_ij = ∂x_i/∂u_j

### Chirp sequence
```
For each Clifford g in {X, Y, Z, H, S}:
    Apply g at step k
    Measure x_{k+1} - x_k  →  J[g, :]
```

### Output
- Measured J_ij per regime instance
- Regime classification: stable / transport / transition / localized
- η_probe_add = F_probe+adaptive − F_adaptive_only  (falsification metric)

---

## Phase 2 — Probe State Families

Replace Néel initial state `|↑↓↑↓...⟩` with structured probe states.

### State families (in order of sensitivity)

| State | Label | Phase sensitivity | Purpose |
|-------|-------|------------------|---------|
| `\|0⟩^⊗N`       | Z_basis   | low  | canonical baseline |
| `\|+⟩^⊗N`       | X_basis   | high | phase noise sensitive |
| `\|R⟩^⊗N`       | Y_basis   | high | T2 sensitive |
| Equatorial sweep | EQ_k     | max  | ΔF_avg most informative |
| Magic state `T\|+⟩` | Magic | max  | scrambling sensitive |

### Equatorial sweep
```
|ψ_k⟩ = cos(θ_k/2)|0⟩ + e^{iφ_k} sin(θ_k/2)|1⟩
θ ∈ {π/4, π/2, 3π/4},  φ ∈ {0, π/2, π, 3π/2}  → 12 states
```

---

## Phase 3 — Controlled Transport

### Controllers
| Controller | Role |
|------------|------|
| `static`   | lower bound (no adaptation) |
| `dd_xy8`   | engineering baseline |
| `random`   | adaptive-structure null |
| `agnostic` | test (observable-only D-LinOSS) |
| `agnostic_probe` | test + Phase 1 chirp calibration |

### Metric
```
F(ρ_out, ρ_target) = Tr(ρ_target ρ_out) + corrections
```
Computed as: average state fidelity over probe family.

```
ΔF_avg = mean_F(agnostic) − mean_F(static)
```

---

## Phase 4 — Recovery Analysis

### Primary metrics
- `ΔF_avg` per regime per state family
- `ΔF_probe` = F_probe+adaptive − F_adaptive (probe stage value)
- Fidelity distribution across equatorial sweep (phase sensitivity map)

### Secondary metrics
- Gate-conditioned fidelity drop (replaces gate-conditioned Δτ from N10a)
- Regime-sorted ΔF_avg (should reproduce N-series hierarchy)

---

## Architectural Changes from Program L

| Component | Program L | Program O |
|-----------|-----------|-----------|
| Initial state | Néel `\|↑↓↑↓⟩` | Structured probe |
| Controller metric | τ_transport | F(ρ_out, ρ_target) |
| Jacobian | Fixed `_CLIFF_OBS_DELTAS` | Measured per-run (Phase 1) |
| Bootstrap | B seeds, same noise | B seeds × P probe states |
| Output | `tau_transport` | `F_avg`, `ΔF_avg` per family |

---

## IBM Translation Path

```
1. TPU tensor-network learns adaptive Clifford policies
2. IBM hardware provides calibration observables (same 12 channels)
3. Phase 1 chirp estimates local J_ij on real hardware
4. Controller selects from: Clifford layers / DD families / sparse SU(2) pulses
5. Post-hoc tomography measures recovered fidelity
```

The key: controller never sees the encoded state — only observable transport signatures.

---

## Files

| File | Purpose |
|------|---------|
| `program_o_tpu.py` | Main experiment (Phases 1–4) |
| `analyze_program_o.py` | ΔF_avg analysis, fidelity maps |
| `PROGRAM_O_PLAN.md` | This document |

---

*Written: 2026-05-18. Pre-registered before first run.*
