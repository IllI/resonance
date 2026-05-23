# Program M2 — Adaptive Recoverability Enhancement

> **Core claim (falsifiable):**
> Adaptive representation-agnostic control improves recoverability fidelity
> by stabilizing coherent operator transport structure under noise — without
> direct fidelity feedback and without Hamiltonian knowledge.

---

## Two-Layer Architecture

| Layer | Question | Metric |
|---|---|---|
| **Operational** | Does recoverability/fidelity improve? | F̄, FSI, τ_transport |
| **Mechanistic** | Why did it improve? | fragmentation, PTM rank, front_v, reversibility |

Programs H–L established the mechanistic layer:
- recoverability ≠ MI, ≠ pairwise concurrence
- representation stability exists and is observable
- adaptive control can prolong recoverability-supporting structure

Program M2 connects mechanism → operational benchmark without collapsing into
vague "AI optimizes pulses" territory.

---

## Experimental Design: Five Phases

### Phase 1 — Spectral Probing / Calibration

Weak pulse sweeps estimate the system's stability landscape.

**Measured quantities:**
- Entropy susceptibility: `dS/dt` under each Clifford family
- PTM fragmentation response: `d(ptm_sv1)/dt` under pulse perturbation
- Coherence decay curvature: second derivative of F_rec vs. time
- Echo reversibility: `F_rec(after_echo) / F_rec(before_echo)`
- Front stability: `std(front_v)` under perturbation

**Controller builds a local stability model from these observables.**

**Critical constraint:**
- No Hamiltonian labels
- No fidelity feedback during Phase 1
- No transport labels — only observables

This is the "chirp" concept formalized: spectroscopy, not cosmological resonance.

### Phase 2 — Entangling Evolution

Evolve the transport medium under each model Hamiltonian.

**Models:**
| Model | Expected behavior | Role |
|---|---|---|
| XXZ | Strong transport | Primary signal |
| DisorderedXXZ_W1 | Moderate transport | Near-transport |
| DisorderedXXZ_W3 | Critical test | Transport boundary |
| OAT | Rigid (ceiling) | Geometric constraint |
| Ising | No transport | **Null** |
| TiltedIsing | No transport | **Null** |

### Phase 3 — Encode Teleportation Benchmark States

This is where the experiment becomes **operationally meaningful**.

**Benchmark state classes:**
| State class | Purpose |
|---|---|
| Bloch tomography states {±X, ±Y, ±Z} | Standard benchmark |
| Equatorial phase states `\|0⟩ + e^{iφ}\|1⟩` | Phase sensitivity |
| Magic states `T\|+⟩` | Non-Clifford stress test |
| Randomized Haar states | Generalization test |
| Small GHZ/cat probes | Fragility test |

**Implementation:** Each benchmark state is encoded at one boundary, evolved
through the transport channel, recovered at the other boundary.

### Phase 4 — Adaptive Stabilization (Controller Acts)

The controller acts online during the transport window.

> [!IMPORTANT]
> The controller must NOT optimize fidelity directly.
> It minimizes: fragmentation, entropy growth, PTM collapse, front incoherence.
> Fidelity is computed post-hoc only.
> This preserves scientific integrity and prevents reward leakage.

**Controller action loop:**
```
observe x_t = [ptm_sv1, ptm_H, EE, MI, basis_invar, ...]
predict x_t' after each Clifford (gate physics model)
select Clifford maximizing strata_score(x_t')
apply Clifford, record action
```

### Phase 5 — Blind Post-Hoc Recovery Evaluation

Only after the transport window closes:
- Compute teleportation fidelity F for each benchmark state
- Compute process fidelity χ
- Compute recoverability lifetime τ_transport
- Compute fidelity variance Var(F)
- Compute Fidelity Stability Index (FSI)

---

## Metrics

### Primary: Fidelity Stability Index (FSI)

$$\text{FSI} = \frac{\mathbb{E}[F]}{\sqrt{\text{Var}(F) + \epsilon}}$$

**Why FSI over mean fidelity alone:**
A controller with slightly lower F̄ but dramatically lower Var(F) may be
*operationally superior* — practical transport channels fail through
instability, not just mean degradation.

### Full metric table:

| Metric | Symbol | Physical meaning |
|---|---|---|
| Mean fidelity | F̄ | Operational performance |
| Fidelity variance | Var(F) | Stability |
| Fidelity Stability Index | FSI | Combined operational quality |
| Recoverability lifetime | τ_transport | Persistence of transport |
| Fragmentation index | FI | Front coherence |
| PTM entropy growth rate | dS_PTM/dt | Operator channel decay |
| Front velocity stability | std(front_v) | Coherent propagation |
| Basis invariance | basis_invar | Representation robustness |
| Echo reversibility | R_echo | Mechanism probe |

### Mediation hypothesis (critical test)

Fidelity improvement should correlate with structural quantities:

| Structural quantity | Hypothesis |
|---|---|
| Lower fragmentation | → stronger recoverability |
| Slower PTM entropy growth | → improved transport |
| Stable front velocity | → coherent propagation |
| Higher reversibility score | → improved recovery |

Confirming this correlation turns the result from **empirical optimization**
into **mechanistic science**.

---

## Controllers

| Controller | Hamiltonian access | Purpose |
|---|---|---|
| Static | None | Lower bound baseline |
| DD (XY8/CPMG) | None (schedule-based) | **Engineering baseline** — critical addition |
| H-aware MPC | Full H + noise, receding horizon | Oracle upper bound |
| D-LinOSS Agnostic | Observables only, 30% dropout | **Test** |

> [!IMPORTANT]
> The DD baseline is the decisive comparison. "Adaptive > DD" is a paper.
> "Adaptive > Static" is a preliminary result. The DD controller requires
> no learning and is already deployed on IBM hardware — beating it is
> a non-trivial claim.

**XY8 pulse sequence:**
```python
XY8_SEQUENCE = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]  # 8-pulse cycle

def ctrl_dd_xy8(step, N, **_):
    """Dynamical decoupling XY8 — engineering baseline."""
    cliff_map = {"X": 1, "Y": 2, "Z": 3, "H": 4, "S": 5}
    cycle = step % 8
    gate_name = XY8_SEQUENCE[cycle]
    return (cliff_map[gate_name], 0)
```

**CPMG sequence:**
```python
def ctrl_dd_cpmg(step, N, **_):
    """CPMG — alternating X echoes."""
    return (1, 0) if step % 2 == 0 else (0, 0)  # X every other step
```

---

## Falsification Criteria (Pre-Registered)

### Strong pass:
- FSI(agnostic) > FSI(DD) in ≥ 2/4 non-null models
- FSI(agnostic) > FSI(static) in ≥ 4/4 non-null models
- Null models: |ΔF̄| < 0.05 (Ising, TiltedIsing operationally null)
- Mediation: Δτ_transport correlates with ΔF̄ (Pearson r > 0.6)
- Control entropy H_agnostic > H_static (adaptive, not trivial)

### Falsification:
- Agnostic ≤ DD in all non-null models (no advantage over engineering baseline)
- Ising / TiltedIsing show ΔF̄ > 0.05 (null models activated)
- Fidelity improvement uncorrelated with structural metrics (mechanism absent)
- Control entropy collapses to uniform or zero (random or trivial switching)

---

## Deployment Sequence

### Step 1 (now running): Program L B=10 fixed controller
```powershell
# Already launched as 5331ea36
.\run_program_l_shotgun.ps1 -N 10 -B 10 ... -OutDir "program_l_b10_fixed" -StoreObs
```
**Purpose:** Verify fixed controller makes non-identity choices; establish
cleaned baseline Δτ numbers.

### Step 2: Add DD controller to program_l_tpu.py
```python
def ctrl_dd_xy8(step, N, **_): ...
def ctrl_dd_cpmg(step, N, **_): ...
```
Register in `get_controller()`. No other changes.

### Step 3: Run 4-controller comparison at N=10, B=10
```powershell
.\run_program_l_shotgun.ps1 -N 10 -B 10 \
  -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing" \
  -Controllers "static dd_xy8 haware agnostic" \
  -OutDir "program_m2_baseline"
```

### Step 4: Add teleportation benchmark states + FSI metric
- Encode benchmark states in `simulate_trajectory`
- Compute post-hoc F per state class
- Add FSI to `analyze_program_l.py`

### Step 5: Add spectral probing phase (Phase 1)
- Pre-evolution calibration window
- Build local stability model from susceptibility measurements
- Feed into controller's initial prior

### Step 6: Full Program M2 run
- N=10, B=20, all 6 models, 4 controllers
- All 5 benchmark state classes
- Full mediation analysis

---

## Strongest Possible Outcome

If results show:
- Agnostic > DD baseline (non-trivial adaptive advantage)
- Agnostic preserves front coherence (mechanism confirmed)
- Fidelity improves without localization/freezing
- Improvements survive scrambling and dropout
- Null models remain null
- Δτ correlates with ΔF̄ (mechanistic mediation)

**The claim becomes:**

> The controller is not merely denoising. It is selectively stabilizing
> transport-capable dynamical structure — a genuinely different mechanism
> from dynamical decoupling.

That is publishable if true.

---

## What to Avoid

- Claims about "accessing nonlocal quantum space" or "outside spacetime"
- "AI discovered teleportation geometry"
- Optimizing directly for fidelity (destroys integrity)
- Reporting only mean F without variance (FSI is required)
- Scaling N before establishing mechanism at N=10

## Implementation Notes

### FSI implementation
```python
def fidelity_stability_index(fidelities, eps=1e-6):
    """FSI = mean / sqrt(var + eps)"""
    f = np.array(fidelities)
    return float(np.mean(f) / np.sqrt(np.var(f) + eps))
```

### Mediation correlation
```python
def mediation_correlation(delta_tau_list, delta_fidelity_list):
    """Pearson r between Δτ and ΔF across models and bootstraps."""
    from scipy import stats
    r, p = stats.pearsonr(delta_tau_list, delta_fidelity_list)
    return {"r": r, "p_value": p, "significant": p < 0.05}
```

### Echo reversibility score
```python
def echo_reversibility(f_rec_before, f_rec_after_echo):
    """R_echo = F_rec(after echo) / F_rec(before echo). >1 means echo helped."""
    return float(f_rec_after_echo / (f_rec_before + 1e-9))
```
