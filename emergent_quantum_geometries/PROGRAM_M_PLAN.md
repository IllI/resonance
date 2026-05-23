# Program M — Adaptive Transport Stabilization
## Implementation Plan

Programs H–L established that **recoverability-supporting structure exists in
observable transport dynamics and can be exploited by a representation-agnostic
controller without Hamiltonian access.**

Program M asks the next operational question:

> **Can adaptive control keep a quantum system inside a recoverable transport
> phase longer than passive evolution or standard dynamical decoupling baselines?**

---

## Scientific Framing

### What Program L showed
- Agnostic controller extends τ_transport in XXZ-like systems (Δτ > 0)
- Null models remain operationally null (Δτ ≈ 0 for Ising, TiltedIsing)
- DisorderedXXZ_W3 stalls — near transition, underresolved at B=3
- OAT at ceiling — static already optimal, consistent with rigid transport basin
- Oracle (haware) frequently underperforms static → greedy 1-step is insufficient

### Core insight
Recoverability is not equivalent to information presence.
MI and EE spread in many systems. Recoverability only emerges when
**operator organization remains stable** — and that structure is now controllable
without Hamiltonian knowledge.

### Program M hypothesis (falsifiable)
> Adaptive representation-agnostic control can prolong recoverability-supporting
> transport structure beyond static dynamical decoupling baselines in systems
> where transport is possible.

This is directly translatable to IBM hardware.

---

## What to Optimize

Do **not** optimize raw fidelity directly. Instead optimize simultaneously:

| Target | Physical meaning |
|---|---|
| τ_transport (Δτ primary) | operational lifetime in recoverable phase |
| PTM spectral rank persistence | keeps operator channels open |
| Front coherence stability | prevents fragmentation |
| Entropy growth suppression | delays decoherence |
| Basis-scramble invariance | avoids representation overfitting |
| Recovery variance reduction | stabilizes channel operationally |

### The controller's question
Not: "How do I maximize fidelity?"
But: **"How do I prevent operator fragmentation?"**

This is a more stable target and closer to what D-LinOSS can reliably track.

---

## Primary Metric Change (from Program L)

### Metric A — Absolute improvement (primary)
```
delta_tau = tau_agnostic - tau_static
```
Denominator-free, stable, interpretable.

### Metric B — Oracle-relative efficiency (secondary)
```python
if tau_haware > tau_static + eps:
    norm_gain = (tau_agnostic - tau_static) / (tau_haware - tau_static)
else:
    norm_gain = nan
    oracle_valid = False
```

Report: Model | Δτ | Oracle valid? | norm_gain

---

## Immediate Next Run (Program L B=10 Diagnostic)

Before Program M, stabilize Program L statistics:

```powershell
.\run_program_l_shotgun.ps1 -N 10 -B 10 `
  -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing" `
  -QueueMaxMin 30 -MaxMin 240 `
  -OutDir "program_l_b10_diagnostic"
```

**Additional flags to append to run command in program_l_tpu.py:**
- `--store-obs` — log obs_history for trajectory analysis
- `--adversarial none` — baseline; run separately with `ee_scramble`, `ptm_phase`

**Purpose:** Statistical stability before scaling N or claiming phase structure.

---

## Program M Experimental Design

### Models

| Model | Expected | Role |
|---|---|---|
| XXZ | Strong positive Δτ | Primary signal |
| DisorderedXXZ_W1 | Moderate Δτ | Near-transport |
| DisorderedXXZ_W3 | Critical test | Near transition |
| MBL_XXZ (new) | Low Δτ or null | MBL boundary probe |
| Ising | Δτ ≈ 0 | **Null control** |
| TiltedIsing | Δτ ≈ 0 | **Null control** |

### Controllers

| Controller | Access | Purpose |
|---|---|---|
| Static | Fixed pulses | Lower bound |
| DD (XY8/CPMG) | Fixed schedule | **Strong engineering baseline** |
| H-aware MPC | Full H + noise, receding horizon | Oracle |
| D-LinOSS Agnostic | Observables only, 30% dropout | **Test** |

> [!IMPORTANT]
> The DD (dynamical decoupling) baseline is the critical addition.
> Comparing agnostic to DD, not just static, is what makes this an
> adaptive-control paper rather than a quantum information paper.
> It is also directly implementable on IBM hardware.

---

## New Diagnostics

### 1. Δτ histogram (per model, per bootstrap)
Already implemented in `analyze_program_l.py` v2.

### 2. Control entropy
```
H_ctrl = -Σ p(cliff_i) log₂ p(cliff_i)   [non-identity actions]
```
Low entropy → controller converged to stable pulse families (strong evidence
of adaptive organization). High entropy → chaotic switching (weak signal).
Already implemented in `analyze_program_l.py` v2.

### 3. Representation contribution ablation

Remove one observable group at a time from agnostic controller input.
Run each in a separate shotgun pass:

```powershell
# Example: EE ablation
.\run_program_l_shotgun.ps1 ... --adversarial ee_scramble --OutDir "ablation_ee"
```

| Ablation | Observable removed | Predicted effect |
|---|---|---|
| `ee_scramble` | EE time-series | Moderate degradation in XXZ |
| `mi_inject` | MI signal | Mild degradation |
| `ptm_phase` | PTM SV phase structure | Strongest degradation |
| `channel_perm` | All channel ordering | Catastrophic |

If PTM ablation causes the largest performance drop → PTM is the dominant
information channel. This directly supports the representation-agnostic claim
while identifying the critical substrate.

### 4. Transport coherence half-life
Time until PTM spectral entropy collapses below TH_SPEC_H_MIN.
Complement to τ_transport (which is F_rec based).

### 5. Front fragmentation index
Whether transport front remains contiguous vs. fragmenting.
Currently approximated by OS (operator spreading). May need explicit
spatial profile tracking for larger N.

### 6. Recovery variance
Report std(τ_agnostic) across bootstrap samples alongside mean.
Already tracked in stats, needs highlighting in output.

---

## Pulse Strategy Upgrade

### Current (Program L)
6 Clifford gates (I, X, Y, Z, H, S) on site 0.
Selection via strata_score on predicted ptm_sv1 change.

### Program M additions

**1. Echo-protected transport pulses**
Add CPMG and XY8 echo sequences as fixed-schedule options.
These form the DD baseline and also expand the agnostic pulse library.

```python
# XY8 cycle (8 pulses per block)
XY8_SEQUENCE = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]

def ctrl_dd_xy8(step, N, **_):
    """Dynamical decoupling XY8 schedule — engineering baseline."""
    cycle = step % 8
    return (XY8_IDX[cycle], 0)
```

**2. Spectral steering**
Inject mild perturbative sweeps and measure transport response.
Controller learns which frequency regions stabilize transport.
(Phase 2 — after B=10 run confirms signal.)

**3. Floquet stabilization search**
Search periodic drive sequences U_F = exp(-iH₁t₁)exp(-iH₂t₂)...
that maximize transport persistence.
(Phase 3 — after ablation confirms mechanism.)

---

## DisorderedXXZ_W3 Priority

This model is now the most scientifically interesting unresolved case:

| W1 | W3 | Ising |
|---|---|---|
| Δτ > 0 | Δτ ≈ 0 | Δτ ≈ 0 |

If this pattern holds at B=10:
- W1 improves → W3 stalls → Ising nulls
- Program L resembles a **real transport-phase sensitivity test**

The controller is tracking the boundary between recoverable transport
and MBL-adjacent localization. That becomes much stronger physics.

---

## IBM Translation Path

1. TPU simulation learns noise-response manifold and control structure
2. IBM calibration runs provide hardware noise observables
3. D-LinOSS infers recoverability-supporting structure from IBM data
4. Controller selects from constrained pulse families:
   - Clifford layers
   - Echoed cross-resonance
   - Dynamical decoupling families (XY8/CPMG)
   - Calibrated SU(2) blocks
5. Fidelity evaluated post-hoc (not used in control loop)

Controller does NOT require knowledge of device Hamiltonian.
Noise observables replace Hamiltonian access entirely.

---

## Deepest Result So Far

> Recoverability is not equivalent to information presence.

Ising nulls demonstrated this cleanly at N=10, B=3.
And now:

> Recoverability-supporting structure may itself be stabilizable
> without Hamiltonian knowledge.

That is the core experimental claim of Program M.

---

## Success Criteria (Pre-Registered)

**Strong pass:**
- Δτ > 0 in ≥ 4/4 non-null models at B=10
- Agnostic > DD baseline in ≥ 2/4 non-null models
- Ising / TiltedIsing: |Δτ| < 0.5 (operationally null)
- Control entropy H_agnostic > H_static (adaptive, not random)
- PTM ablation causes largest performance drop vs. other channels

**Falsification:**
- Agnostic ≤ DD in all non-null models
- Ising/TiltedIsing show Δτ > 0.5
- Control entropy collapses to uniform (random switching, no adaptation)

---

## Deployment Sequence

```powershell
# Step 1: Statistical stability (run now)
.\run_program_l_shotgun.ps1 -N 10 -B 10 -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing" -QueueMaxMin 30 -MaxMin 240 -OutDir "program_l_b10_diagnostic"

# Step 2: Ablation studies (one per shotgun run)
# --adversarial ee_scramble / mi_inject / ptm_phase / channel_perm

# Step 3: Program M proper (with DD baseline added)
# N=10, B=10, all 6 models, 4 controllers, full diagnostics
```
