# Research Log: Recoverable Operator Transport
## `emergent_quantum_geometries/`

**Branch:** `quantum-teleportation-results`
**Started:** 2026-05 session
**Language:** "Recoverable operator transport sufficient for above-classical fidelity."
NOT: "teleportation," "emergent spacetime," "nonlocal communication."

---

## Pre-Registered Thresholds (FROZEN — do not change between runs)

```
th_MI  = 0.03    excess above t=0 baseline
th_EE  = 0.04    excess above t=0 baseline
th_OS  = 0.015   excess above t=0 baseline
th_dF  = 0.015   excess above t=0 baseline
th_rec = 2/3     absolute (classical communication limit)
```

These are locked in `program_k_tpu.py` v2 header. Changing them post-hoc invalidates
the causal fraction and cross-model comparisons.

---

## Experimental Chain

### Program H3 — Concurrence Lifetime
**File:** `program_h3_tpu.py`
**Key result:** ΔF outlives pairwise concurrence C. Recovery geometry is not reducible to entanglement.
**Confound eliminated:** "It's just pairwise entanglement."

### Program I0 — MI vs dF Separation
**File:** `program_i0_tpu.py`
**Key result:** Ising: MI=1.9 bits at j=3, ΔF≈0. High MI without transport.
**Confound eliminated:** "It's just mutual information spreading."

### Program I+ — Multi-Representation Stability
**File:** `program_iplus_tpu.py`
**Key result:** XXZ CV=0.084 across PTM/EE/OS/MI representations. Ising: all null.
**Confound eliminated:** "It's just a PTM coordinate artifact."

### Program J — Blind Prediction
**File:** `program_j_tpu.py`
**Key result:** XXZ blind precision ~1.0, Ising ~0.19. MI+EE insufficient for Ising.
**Confound eliminated:** "MI+EE already predict it."

### Program K v1 — Predictive Recovery Fronts
**File:** `program_k_tpu.py`
**Results:**
- N=10: cross-model prec=0.920 (`program_k_N10_results.json`)
- N=12 (XY/XXZ/Ising): cross-model prec=0.951 (`program_k_N12_results.json`)
- N=12 extra (TiltedIsing/OAT): both prec=0.000 (`program_k_N12_extra_results.json`)
**Confound eliminated:** "The predictor just says transport everywhere."
**New finding:** 4/5 models show null — only XXZ recovers. Not a generic quantum feature.

### Program K v3 — Clifford Scramble + Disorder W-Sweep (current)
**File:** `program_k_tpu.py` (same, updated)
**New in v3:**
1. **Local Clifford scramble** (`--clifford-scramble N`):
   Applies random discrete Clifford gates C_A⊗C_B (24-element group) before computing MI.
   Harder than continuous basis scramble: permutes X↔Y↔Z axes discretely, restructures stabilizers.
   If precision survives: the predictor tracks transport organization, not coordinate orientation.
   Smoke test (N=6): XXZ 0.963±0.000, Ising 0.500±0.000 (correct null/signal).
2. **Disorder W-sweep** models: W=1.0 (weak), W=3.0 (near MBL transition), W=5.0 (localized), W=8.0 (deep).
   Provides tunable control parameter and direct transport/localization phase boundary.
   Target: MI propagates at W=1, recoverability collapses before or simultaneously.

---

## Model Roster (v3)

| Model | Description | Expected prec | Observed (N=12) |
|---|---|---|---|
| XXZ | Antiferromagnetic Heisenberg, Dz=1 | ~0.94 | 0.941 |
| XY | XX+YY only, Dz=0 | ~0.00 | 0.000 |
| Ising | ZZ + transverse field | ~0.00 | 0.000 |
| TiltedIsing | Non-integrable Ising (hx=0.9045, hz=0.809) | ~0.00 | 0.000 |
| OAT | One-axis twisting, collective | ~0.50 | 0.513 |
| MBL_XXZ | Disordered XXZ, W=5 | ~0.00 | TBD (N=14) |
| DisorderedXXZ_W1 | Weak disorder, transport likely | ~0.70 | TBD (N=14) |
| DisorderedXXZ_W3 | Near MBL transition (~W=3.7) | ~0.20 | TBD (N=14) |
| DisorderedXXZ_W5 | Clearly localized | ~0.00 | TBD (N=14) |
| DisorderedXXZ_W8 | Deep localization | ~0.00 | TBD (N=14) |

---

## Framing (LOCKED 2026-05-15)

### Primary claim:
> "Predicting when recoverable quantum transport emerges."

### NOT:
- "Teleporting information through entangled space"
- "Emergent spacetime geometry"
- "Nonlocal communication"
- Speculative cosmology / Orch-OR / retrocausality

### Critical distinction: transport enhancement ≠ teleportation protocol
The geometry may be **robust** even when a naive teleportation protocol fails.
Program G2 showed this clearly: naive Bell basis → sub-classical; PTM-aligned → super-classical.
The emergent object is **recoverable transport structure**, not teleportation itself.

Operational fidelity still depends on: measurement basis, recovery unitaries, PTM alignment, protocol choice.
The predictor identifies *when* the structure is available — not *how* to exploit it.

### Strongest scientific asset (representation stability):
The predictor survives:
1. Continuous basis scrambling (Haar-random SU(2))
2. Local Clifford scrambling (discrete, axis-permuting)
3. Independent representations (PTM/EE/OS/MI)
4. Unseen Hamiltonians (cross-model)
5. Negative controls staying null (Ising, TiltedIsing, MBL_XXZ)

This is the hardest thing to fake accidentally. It is what "representation-stable" means.

---

## N=14 Run — Three Jobs

**Command (via `run_program_k_shotgun.ps1`):**
```
--N 14 --T-max 8.0 --n-coarse 40 --n-fine 100 --t-onset 3.5
--K 200 --B 5 --j-focus 4 5 6 7 8 9 10 11
--models XXZ Ising TiltedIsing MBL_XXZ DisorderedXXZ_W1 DisorderedXXZ_W3
--basis-scramble 5 --clifford-scramble 5
```

### Job 1: Causal hierarchy at large j
**Target:** t_MI < t_dF ≈ t_rec holds at j=7..11
**Success:** Mean dtt(rec-MI) > 0, causal fraction > 0.7 in XXZ
**Failure mode:** dF/recovery collapse → recoverability is just another entanglement proxy

### Job 2: Disorder phase boundary
**Target layered decay:**
- W=1: MI propagates, recoverability present
- W=3: MI propagates, recoverability marginal
- W=5/8: MI propagates, recoverability collapses

**Watch for nuanced separation:**
- MI survives longest (longest range)
- EE intermediate
- dF/recovery collapse together at lower W

If MI and recovery collapse simultaneously → less interesting.
If recovery collapses BEFORE MI → strong result (recovery ≠ information propagation).

### Job 3: Representation stress test
**Target:** Clifford-scramble precision ≥ basis-scramble precision in XXZ (both ≥ 0.90)
**All nulls:** Ising, TiltedIsing, MBL_XXZ Clifford prec ≈ 0.00

---

## Deployment Notes

### Running locally
```powershell
cd emergent_quantum_geometries
python program_k_tpu.py --N 12 --T-max 6.0 --n-coarse 30 --n-fine 80 `
  --t-onset 3.0 --K 200 --B 5 --j-focus 4 5 6 7 8 `
  --models XXZ Ising TiltedIsing MBL_XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 `
  --basis-scramble 5 --clifford-scramble 5 `
  --out program_k_N12_v3_results.json
```

### Running on TPU
Use `run_program_k_shotgun.ps1` — races 3 spot zones, SSH-verifies winner,
runs full analysis ON TPU before download, performs Bible delete ritual.

**Key TPU rules (from trc_tpu_bible.md):**
- Use `--spot` flag, NOT `--best-effort`
- Only use TRC-approved zones: us-central2-b, us-east1-d, europe-west4-a
- v6e spot in europe-west4-a is highly preempted — prefer v4 (us-central2-b)
- Always run delete ritual, verify all zones return 0 items

---

## Next Milestone: Blind Hardware Validation

After N=14:
1. Generate IBM hardware trajectories (real hardware noise)
2. Reconstruct observables from hardware shots
3. **Hide fidelities completely**
4. Run D-LinOSS blindly (no Hamiltonian labels, no fidelity exposure)
5. Predict recoverable regions
6. Reveal fidelities afterward

**Scientific significance:**
If recoverability prediction survives hardware noise + basis scrambling + unseen Hamiltonians:
> "Recoverable transport regions can be inferred operationally before fidelity is measured."
This moves the work from "interesting simulation study" to "hardware-relevant predictive transport geometry."

---

## Key Scientific Questions (N=14)

1. Does `t_MI < t_dF ≈ t_rec` hold at large j (j≥7)? (causal hierarchy resolution)
2. Does basis-scramble precision hold ≥ 0.90 in XXZ at N=14?
3. Does Clifford-scramble precision match or exceed basis-scramble precision?
4. Does MBL_XXZ give prec ≈ 0.00 with both scramble types?
5. At what disorder W does recoverability collapse? Before or with MI?
6. Does cross-model precision stay > 0.90 with disorder models in the pool?

---
*Log last updated: 2026-05-15. Branch: quantum-teleportation-results.*
