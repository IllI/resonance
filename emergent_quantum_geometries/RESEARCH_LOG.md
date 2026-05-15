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

### Program K v2 — With Adversarial Controls (current)
**File:** `program_k_tpu.py` (same, updated)
**New metrics:**
1. **Basis-scramble precision:** XXZ 0.941 ± 0.000 across 5 random basis rotations (N=12).
   Basis-independence confirmed. Ising: 0.000 ± 0.000.
2. **Causal fraction P(t_dF < t_rec):** computed per model with 95% Wilson CI.
   XXZ: 0.00 (n=5) — t_dF arrives simultaneously with or after t_rec.
   Interpretation: t_dF ≈ t_rec, not t_dF strictly precedes t_rec. The simultaneity
   IS the finding: transport geometry and recovery onset coincide.
3. **Raw trajectories:** F_rec(j,t), MI(j,t), EE(j,t), dF(j,t) saved in JSON.
4. **MBL-XXZ added:** W=5 disordered XXZ. Same ZZ structure as clean XXZ, zero transport.

**Results file (in progress):** `program_k_N12_v2_results.json`

---

## Model Roster

| Model | Description | Expected prec | Observed prec (N=12) |
|---|---|---|---|
| XXZ | Antiferromagnetic Heisenberg, Dz=1 | ~0.94 | 0.941 |
| XY | XX+YY only, Dz=0 | ~0.00 (j=4-8) | 0.000 |
| Ising | ZZ + transverse field | ~0.00 | 0.000 |
| TiltedIsing | Non-integrable Ising (hx=0.9045, hz=0.809) | ~0.00 | 0.000 |
| OAT | One-axis twisting, collective | ~0.50 (random) | 0.513 |
| MBL_XXZ | Disordered XXZ, W=5 (above MBL transition) | ~0.00 | TBD (N=14) |

---

## Deployment Notes

### Running locally
```powershell
cd emergent_quantum_geometries
python program_k_tpu.py --N 12 --T-max 6.0 --n-coarse 30 --n-fine 80 `
  --t-onset 3.0 --K 200 --B 5 --j-focus 4 5 6 7 8 `
  --models XY XXZ Ising TiltedIsing MBL_XXZ --basis-scramble 5 `
  --out program_k_N12_v2_results.json
```
Runtime: ~20-30 min locally at N=12. N=14 requires TPU (~30-45 min on VM).

### Running on TPU (next session)
Use `run_program_k_shotgun.ps1` — races 3 spot zones, SSH-verifies winner,
runs full analysis ON TPU before download, performs Bible delete ritual.

**Key TPU rules (from trc_tpu_bible.md):**
- Use `--spot` flag, NOT `--best-effort` (this version of gcloud)
- Only use TRC-approved zones: us-central2-b, us-east1-d, europe-west4-a
- v6e spot in europe-west4-a is highly preempted — prefer v4 (us-central2-b)
- Always run delete ritual, verify all zones return 0 items

---

## Next Session Checklist

1. [ ] Retrieve `program_k_N12_v2_results.json` (in-progress run)
2. [ ] Launch `run_program_k_shotgun.ps1` for N=14:
   ```
   --N 14 --T-max 7.0 --j-focus 5 6 7 8 9
   --models XY XXZ Ising TiltedIsing MBL_XXZ
   --basis-scramble 5
   ```
3. [ ] Full analysis on TPU before download
4. [ ] Verify: causal fraction P(t_dF < t_rec) in XXZ, all zones clean after

---

## Key Scientific Questions Remaining

1. **Does basis-scramble precision hold at N=14?** (XXZ ≥ 0.90, Ising = 0.00)
2. **Does MBL-XXZ give prec ≈ 0.00?** (same ZZ structure, zero transport)
3. **What is the causal fraction at N=14 with tighter CI?**
4. **Does cross-model precision stay > 0.90 with MBL_XXZ in the pool?**
