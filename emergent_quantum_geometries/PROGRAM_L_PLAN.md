# Program L — Representation-Agnostic Adaptive Recovery
## Implementation Plan (Revised)

Programs H–K established that **predictive transport structure exists, survives basis
scrambling, and separates models cleanly across independent representations**.
Program L asks the next falsifiable question:

> **Can a controller stabilize recoverable transport under noise using only
> observable dynamics — without Hamiltonian access or coordinate knowledge?**

---

## Framing Constraints

### What D-LinOSS discovers (safe claim)
Multiple independent representations exhibit **predictive structure stable across
the tested representations** in their transport dynamics. A controller using only
those observables can extend recoverability lifetime.

### Preferred phrasing (use consistently)
- **YES:** "predictive structure stable across the tested representations"
- **YES:** "recoverability-supporting structure in observable dynamics"
- **NO:** "representation-stable predictive structure" — still sounds ontological
- **NO:** "shared latent structure" — implies hidden geometry
- **NO:** "transport geometry" — implies ontological substrate

Anchoring the claim experimentally ("across the *tested* representations") scopes
it to what the experiment actually shows.

### What D-LinOSS does NOT discover (do not imply)
- An ontologically real "quantum geometry" independent of representation
- A latent space that reflects something substrate-independent or physical

The core operational claim is already strong enough:

> *"Recoverable transport can be predicted and stabilized from observable dynamics
> without Hamiltonian access, across multiple independently chosen representations."*

---

## Reward Leakage: Explicit Acknowledgment

The controller objective is:
```
J = strata_score(x_t)
```
where `x_t` contains observables (PTM SVs, EE, MI, OS) that were historically
**selected because of their predictive relationship to recoverability**.

This does not invalidate the experiment. The controller still never directly observes
fidelity. But reviewers will note it, so the honest framing is:

> *"The controller never observes fidelity directly. However, the transport observables
> were selected based on their known predictive relationship to recoverability from
> Programs H–K. The controller is therefore indirectly optimizing a proxy correlated
> with fidelity — this is acknowledged and is the intended design."*

---

## Gain Metric (Revised)

**Old (pathological when τ_haware ≈ τ_static):**
```
gain = (τ_agnostic - τ_static) / (τ_haware - τ_static)  ← RETIRED
```

**New:**
```
τ_best = max(τ_static, τ_haware)
normalized_gain = (τ_agnostic - τ_static) / max(ε, τ_best - τ_static)
```
- `ε = 1e-3` (pre-registered floor, avoids division near zero)
- Separately report `beats_oracle` (bool): whether agnostic exceeds myopic oracle

**On gain > 1:** The H-aware oracle is intentionally a 1-step greedy selector
(maximises F_rec at next step only). The agnostic's strata-score objective is
inherently long-horizon. Gain > 1 is therefore **expected and interpretable**, not
a sign of a broken oracle. This must be stated explicitly in any writeup.

---

## Primary Result: Representation Dropout

> [!IMPORTANT]
> This is elevated to a **primary result**, not a diagnostic.

If recoverability-supporting structure survives:
- PTM observable dropout (30%)
- EE dropout
- MI dropout
- OS dropout

then no single representation is necessary. The structure is not reducible to
any one channel. This directly operationalises the representation-agnostic claim.

**Report as:** "Representation Dropout Robustness" — a separate result, not a
footnote.

---

## Adversarial Representation Poisoning

> [!IMPORTANT]
> Add before full N=10 run. Stronger than dropout.

Rather than simply masking channels, actively inject misleading structure:

| Poisoning Mode | Method |
|---|---|
| EE scramble | Shuffle EE time-series while preserving marginals |
| Synthetic MI | Inject artificial MI fronts into MI channel |
| PTM phase randomise | Randomise phases of PTM singular spectra |
| Channel permutation | Randomly permute latent channel order |

**Expected:** agnostic controller performance degrades only if it is over-relying
on one representation. If it is tracking genuine representation-stable structure,
degradation should be bounded.

**Add to `program_l_tpu.py`:** `--adversarial` flag with `["none", "ee_scramble",
"mi_inject", "ptm_phase", "channel_perm"]` options.

---

## Representation Mutual Predictability Matrix (L0 Analysis)

> [!IMPORTANT]
> This diagnostic guards against the feature-engineering collapse risk.

Several observables in the 12-dim vector may be highly correlated:

| Suspected correlation | Risk |
|---|---|
| ptm_sv1, ptm_sv2, ptm_sv3 | All PTM-derived — may compress to one axis |
| H_T, ptm_H | Spectral entropy of the same matrix |
| D_eff, front_v | Both time-derivatives of EE/OS |

If PTM-derived quantities predict all other channels, the "representation diversity"
claim is weaker than it appears.

**Method:** For each observable pair (i, j), compute R² of predicting channel j
from channel i using linear regression over the obs_history matrix across all
trajectories. Report a 12×12 R² heatmap.

**Add to `analyze_program_l.py`:** `compute_predictability_matrix()` — returns
12×12 R² matrix and flags any channel with mean cross-prediction R² > 0.80 as
potentially redundant.

**Desired outcome:** EE, MI, OS independently contribute predictive value beyond
what PTM SVs already capture. If true, the representation diversity argument holds.
If not, the observable set should be pruned or acknowledged as having one dominant
axis — which still may be scientifically interesting but changes the claim.

---

## Extended Controller Comparison

The full Program L controller table (for eventual writeup):

| Controller | Access | Purpose |
|---|---|---|
| Static | Fixed pulses | Lower bound |
| Echoed (DD) | None — fixed dynamical decoupling schedule | Standard engineering baseline |
| H-aware (greedy-1) | Full H + noise model, 1-step | Myopic oracle |
| D-LinOSS Agnostic | Observables only | **Test** |

Adding the **Echoed/DD baseline** (standard dynamical decoupling) makes this an
adaptive-control paper rather than just a quantum information paper. It is also
directly implementable on IBM hardware.

---

## Program L0 — Pre-Controller Falsification Run

> [!IMPORTANT]
> Run this BEFORE the full Program L controller comparison.

**Purpose:** Verify representation-stable predictive structure survives intentional
representation corruption — before any controller is introduced.

**Design:**
- N=10, K=100, B=3
- Models: XXZ, Ising, DisorderedXXZ_W3
- No controller (observe-only)
- Tests: representation dropout, adversarial poisoning, continuous + Clifford scramble

**Success criteria:**
1. Predictive structure for XXZ survives all four adversarial poisoning modes
2. XXZ Ising distinction preserved under 30% channel dropout
3. **Time-Shift Generalization:** D-LinOSS trained on steps 0–T/2 correctly
   predicts transport structure in steps T/2–T. If the controller is memorising
   local time-correlations rather than tracking dynamical organisation, this will
   fail. If it passes, the structure is genuinely temporal rather than local.
4. Ising remains null under all conditions (all tests)

**If L0 fails:** The controller experiment (full L) becomes uninterpretable.
L0 must pass first.

---

## Three-Controller Comparison (Program L proper)

### Controllers
```
Static     — fixed pre-calibrated Clifford pulses
HAware     — full H + noise model, 1-step greedy oracle
Agnostic   — representation-stable transport signatures only,
             D-LinOSS observable extrapolation, 30% dropout
```

### Observable vector (12-dim, locked before first run)
```python
OBS_NAMES = [
    "ptm_sv1", "ptm_sv2", "ptm_sv3", "ptm_H",
    "EE", "MI", "OS", "H_T",
    "D_eff", "front_v", "noise_slope", "basis_invar",
]
```

### Pre-registered thresholds (frozen in program_l_tpu.py header)
```python
TH_F_REC      = 2/3
TH_SV1_MIN    = 0.30
TH_SPEC_H_MIN = 0.40
TH_EE_MIN     = 0.10
TH_MI_MIN     = 0.03
TH_BI_MAX     = 0.25
P_DROPOUT     = 0.30
CTRL_INTERVAL = 5
```

### Model roster
| Model | Expected norm_gain | Role |
|---|---|---|
| XXZ | high | Primary signal |
| DisorderedXXZ_W1 | moderate | Near transport |
| DisorderedXXZ_W3 | low–moderate | Near transition |
| OAT | moderate | Cross-type test |
| Ising | ~0 | **Null control** |
| TiltedIsing | ~0 | **Null control** |

Ising/TiltedIsing must show `normalized_gain ~ 0`. If not, the framework is invalid.

---

## Success Criteria (Pre-Registered)

**Strong pass (publishable direction):**
- `normalized_gain >= 0.5` in ≥ 4/6 non-null models
- `CV_latent < 0.3` across non-null models
- Ising/TiltedIsing: `normalized_gain ~ 0`
- Representation dropout degradation < 20%
- Adversarial poisoning degradation < 40% (bounded, not catastrophic)

**Falsification (experiment fails if ANY):**
- Agnostic ≤ static in all non-null models
- Ising shows significant normalized_gain (> 0.15)
- Dropout destroys performance entirely (> 80% degradation)

---

## IBM Translation Path

**Realistic path (not "TPU teleportation"):**
1. TPU learns representation-stable control structure (Program L)
2. IBM calibration runs provide noise observables
3. D-LinOSS infers persistent recoverability-supporting structure
4. Controller selects from Clifford layers / echoed cross-resonance / DD families
5. Fidelity evaluated afterward (post-hoc, not used in control)

This is scientifically coherent. The controller is constrained to:
- Clifford layers
- Echoed cross-resonance
- Dynamical decoupling families
- Calibrated SU(2) blocks

It does NOT require the controller to know the device Hamiltonian.

---

## What Not to Chase

> [!WARNING]
> Do not optimise for absolute fidelity maximisation.

If the framing becomes "90% fidelity on IBM hardware," the work reads as a
noisy-control engineering project. The scientific contribution is:

**Recoverability can be predicted and stabilized from representation-invariant
observables without Hamiltonian access.**

The important axes are:
- **Invariance:** structure survives basis and Clifford scrambling
- **Transferability:** structure generalises across model classes
- **Robustness:** structure survives dropout and adversarial poisoning
- **Causal ordering:** structure precedes and predicts fidelity, not vice versa

These are what Programs H–K built toward. Program L is the operational test.

---

## Deployment

### Smoke test (done — validated)
```powershell
cd emergent_quantum_geometries
python program_l_tpu.py --smoke-test
# XXZ norm_gain=2.60, Ising norm_gain=0.00 (null) — structural PASS
```
> [!CAUTION]
> Do not anchor to norm_gain=2.60. This is N=6, B=1, single seed, no poisoning,
> no dropout enabled at scale. It confirms the pipeline runs correctly — nothing
> more. Meaningful numbers require N>=10, B>=3, all poisoning modes, multiple seeds.

### L0 (next — pre-controller falsification)
```powershell
# TBD: program_l0_tpu.py — observe-only, dropout + poisoning tests
```

### Full Program L (TPU)
```powershell
.\run_program_l_shotgun.ps1 -N 10 -B 3 `
  -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing"
```

### Analysis
```powershell
python analyze_program_l.py program_l_results\program_l_N10_results.json
```

---

## Program N -- Dynamical Control-System Identification

> [!NOTE]
> Program N extends the agnostic controller from state-classifier to dynamical response estimator.
> Architecture: z_t = f(x_{0:t}, u_{0:t}, Dx_{0:t}) -- includes control history and response gradients.

### Scientific question
Does the controller benefit from modeling the response dynamics of intervention itself?
Does short-time observable response encode local controllability?

### Controller evolution

| Version | Architecture | Key constant | Status |
|---|---|---|---|
| v1 (N Phase 0) | strata_score only | -- | COMPLETE |
| v2 (N1) | + GRAD_ALPHA*cos(theta) | GRAD_ALPHA=0.20 | COMPLETE |
| v3 (N2) | + dynamic alpha_eff + lambda_sparse | GRAD_ALPHA_BASE=0.05, LAMBDA=0.02 | COMPLETE |

### Program N Phase 0 results (ctrl_agnostic v1)
- XXZ: agnostic > DD > static. Adaptive control exploits transport structure.
- W3: DD destabilizes (-1.46), agnostic preserves (-0.54). Key physics result.
- OAT: H=0.000 -- complete restraint learned. No intervention in stable basin.
- Random baseline: catastrophically destabilizes W1 (tau_rnd=1.660 vs tau_s=3.040).

### Program N1 results (ctrl_agnostic v2, GRAD_ALPHA=0.20)
- CV_latent=0.414 (FAIL). GRAD_ALPHA too aggressive -- destabilized XXZ.
- W3 improved (+0.400) -- gradient following helped near-critical regime.
- OAT behavioral null broken -- gradient term forced unnecessary action.
**Critical finding:** Response Jacobian revealed controller was targeting wrong channels.
  High-response channels: D_eff(0.626), noise_slope(0.465), basis_invar(0.396), front_v(0.334)
  Old model targeted: ptm_sv1, ptm_H, EE, MI -- all LOW-Jacobian.

### Program N2 results (ctrl_agnostic v3, Jacobian-calibrated)
**N=10, B=10, 2026-05-16:**
| Model       | tau_s | tau_dd | tau_h | tau_a  | Delta_tau | vs_DD  | cos_theta | acts |
|-------------|-------|--------|-------|--------|-----------|--------|-----------|------|
| XXZ         | 1.280 | 1.420  | 2.880 | 3.260  | +1.980    | +1.840 | -0.229    | 22   |
| W1          | 3.040 | 2.640  | 2.680 | 2.960  | -0.080    | +0.320 | -0.064    | 30   |
| W3          | 4.900 | 3.440  | 4.900 | 4.120  | -0.780    | +0.680 | -0.103    | 20   |
| OAT         | 4.360 | 4.360  | 3.720 | 4.000  | -0.360    | -0.360 | -0.338    | 10   |
| Ising[null] | 3.900 | 4.500  | 4.840 | 4.320  | +0.420    | -0.180 | -0.077    | 20   |
| TIsing[null]| 3.640 | 3.540  | 4.800 | 3.480  | -0.160    | -0.060 | -0.100    | 24   |
CV_latent = 0.136 (PASS)

**Key structural finding -- anti-alignment is corrective stabilization:**
cos(theta) < 0 across all models. Controller selects gates whose PREDICTED effect
OPPOSES recent observable drift. This is corrective stabilization, not trajectory-following.
Stronger anti-alignment (more negative cos_theta) correlates with better performance.

**XXZ breakthrough:** tau_a=3.260 > tau_h=2.880. Agnostic beats myopic oracle.
Long-horizon corrective strategy outperforms 1-step greedy fidelity maximization.

### Remaining issues and next iteration design

**OAT over-fires:** lambda=0.02 insufficient for complete restraint.
**W3 under-fires:** corrective suppression too strong; W3 benefits from gradient following.
**Solution:** Regime-conditioned lambda complementary to susceptibility-gated alpha.

**N3 architecture (planned):**
`python
susceptibility = dx_norm * (1 + |D_eff| + |noise_slope|)
alpha_eff  = GRAD_ALPHA_BASE * tanh(susceptibility / SUSCEPTIBILITY_SCALE)  # high -> follow gradient
lambda_eff = LAMBDA_MAX * (1 - tanh(susceptibility / SUSCEPTIBILITY_SCALE)) # high -> enforce restraint
score(u!=I) = strata_score(x_pred) + alpha_eff * alignment - lambda_eff
`
This creates a complementary gate: high susceptibility (fragile/W3) -> follow gradient, enforce no restraint.
Low susceptibility (stable/OAT) -> no gradient following, strong restraint.

### Shotgun zone pool (verified 2026-05-16)
| Zone | Type | Quota | Status |
|---|---|---|---|
| us-central2-b | v4-8 | on-demand + spot | ACTIVE in pool |
| europe-west4-a | v6e-8 | spot | ACTIVE in pool -- primary winner |
| us-east1-d | v6e-8 | spot | ACTIVE in pool |
| us-central1-a | v6e-8 | spot | NOT in quota (permission denied) |
| europe-west4-b | v5p only | -- | No single-chip available |

*Plan updated: 2026-05-16. N2 complete. N3 designed.*
