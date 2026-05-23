# Program N — Adaptive State-Transfer Fidelity

> **Core claim (falsifiable):**
> Adaptive representation-agnostic control increases operational state-transfer
> fidelity relative to standard dynamical decoupling schedules under identical
> noise, without Hamiltonian access.

---

## Scientific Grounding (What Programs H–M Established)

| Result | Source |
|---|---|
| Recoverability ≠ MI, ≠ EE | Programs H–J |
| PTM/front coherence tracks recoverability | Programs K–L |
| Adaptive control extends τ_transport in XXZ | Program L |
| Fixed DD destabilizes near-critical W3 | Program M |
| Agnostic controller learns restraint (OAT) | Program M |
| Control entropy correlates with gain | Program M |

The chirp/probe phase is now experimentally justified.
Programs H–M provide the operational anchor.

---

## Hypothesis

> Can adaptive representation-agnostic control increase operational
> state-transfer fidelity relative to standard DD schedules under identical
> noise?

**Safeguard:** The controller is learning dynamical response structure,
not discovering ontological quantum geometry.

---

## Experimental Structure: Three Phases

### Phase 1 — Online Coherence Spectroscopy (Chirp / Probe)

Before transport begins, apply a short probe window to initialize
the controller's latent state.

**Probe pulse families:**
- Small-angle X/Y rotations (coherence decay rate)
- Short echo pairs (echo recoverability)
- Phase ramps (phase susceptibility)
- Staggered Clifford perturbations (operator spreading response)

**Observable response measured:**
- PTM SV decay slope
- EE short-time growth rate
- MI front initiation speed
- Short-time F_rec slope (recoverability curvature)
- OTOC proxy (operator spreading)

**Controller outputs from probe:**
- Damping mode estimates (D-LinOSS decomposition)
- Stable frequency bands
- Control sensitivity profile
- Echo reversibility score

**What this is scientifically:**
Online coherence spectroscopy — exactly analogous to NMR calibration,
Ramsey characterization, Floquet spectroscopy, adaptive Hamiltonian learning.
Not "mapping hidden channels."

**Implementation:** Add `probe_window` parameter (default 5 steps).
During probe: apply fixed sweep, record obs_history.
After probe: compute latent state z_probe, seed agnostic controller.

### Phase 2 — Entangling Evolution + State Encoding

Evolve the transport medium, then encode benchmark states at the boundary.

**Models (same as Program M):**
- XXZ — primary signal
- DisorderedXXZ_W1 — near-transport
- DisorderedXXZ_W3 — near-critical (most important)
- OAT — geometric ceiling (control restraint expected)
- Ising — null
- TiltedIsing — null

**Benchmark state classes (ordered by priority):**

| State class | Encoding | Why |
|---|---|---|
| Cardinal states {±X, ±Y, ±Z} | Bloch tomography | Standard benchmark |
| Equatorial sweeps φ ∈ [0, 2π] | Phase sensitivity | Robustness test |
| Magic states T\|+⟩ | Non-Clifford stress | Hard case |

Avoid GHZ/cat states at N=10–14 — they primarily benchmark noise floors,
not recoverability structure.

### Phase 3 — Adaptive Recovery + Blind Fidelity Evaluation

**Controller acts online. Must NOT optimize fidelity directly.**

Controller objective:
- Maximize coherence persistence
- Maintain recoverability-supporting observables
- Preserve echo recoverability
- Suppress spectral fragmentation

**Fidelity computed post-hoc only** (blind evaluation).
This preserves causal integrity of the experiment.

---

## Controllers

| Controller | H access | Schedule | Role |
|---|---|---|---|
| Static | None | Fixed | Lower bound |
| DD (XY8) | None | Fixed 8-pulse | Engineering baseline |
| Random adaptive | None | Random Clifford | Stochastic control null |
| D-LinOSS agnostic | None | Adaptive | **Test** |
| H-aware MPC | Full H + noise | Optimal | Oracle upper bound |

**The decisive comparison:** D-LinOSS vs DD and Random.
Random adaptive is new — it controls for "any action beats inaction."

---

## Primary Metrics (τ_transport now secondary)

| Metric | Symbol | Definition |
|---|---|---|
| Average state-transfer fidelity | F_avg | Mean over tomography states and bootstraps |
| Worst-case fidelity | F_worst | Min over tomography states |
| Fidelity gain vs DD | ΔF_DD | F_avg(agnostic) − F_avg(DD) |
| Fidelity Stability Index | FSI | E[F] / √(Var[F] + ε) |
| Robustness under scramble | F_scramble | Fidelity under adversarial obs poisoning |
| Disorder sensitivity | ΔF(W1→W3) | Fidelity degradation under disorder |
| Control entropy | H_ctrl | Shannon entropy of Clifford selections |
| Transport lifetime | τ_transport | Supporting metric (from Programs L–M) |

### Fidelity Stability Index

```python
def fidelity_stability_index(fidelities, eps=1e-6):
    f = np.array(fidelities)
    return float(np.mean(f) / np.sqrt(np.var(f) + eps))
```

FSI rewards low-variance performance — operationally important because
channels fail through instability, not just mean degradation.

---

## Random Adaptive Controller (New Null)

```python
def ctrl_random(step, N, rng=None, **_):
    """
    Random adaptive controller — stochastic control null.
    Selects uniformly from all Clifford gates at each interval.
    Controls for: 'any action beats inaction.'
    If agnostic > random, adaptation matters (not just intervention).
    """
    if step % CTRL_INTERVAL != 0:
        return (0, 0)
    if rng is None:
        rng = np.random.default_rng(step)
    return (int(rng.integers(0, N_CLIFF)), 0)
```

---

## Falsification Criteria (Pre-Registered)

### Pass:
- ΔF_DD > 0 in ≥ 2/4 non-null models (agnostic beats DD)
- ΔF_DD > F_gain(random) in ≥ 2/4 non-null models (adaptation ≠ random intervention)
- Ising/TiltedIsing: |ΔF_DD| < 0.05 (null preserved)
- W3: agnostic > DD in fidelity (DD destabilizes, agnostic preserves)
- Control entropy H(agnostic) > H(random) in transport models (structured, not uniform)

### Falsification:
- Agnostic ≤ DD in all non-null models (no adaptive advantage)
- Agnostic ≤ Random in all models (random intervention explains result)
- Null models show ΔF > 0.05 (null violated)
- Control entropy collapses to uniform (unstructured switching)

---

## What a 5–15% Fidelity Lift Means

You do not need 90% fidelity for this work to matter.

A consistent 5–15% lift under disorder/noise, with preserved nulls,
is already scientifically valuable because it demonstrates:

1. **Mechanism specificity** — gains appear where transport structure exists (XXZ, W1), not where it doesn't (Ising, OAT)
2. **Adaptive advantage** — gains exceed DD and random, so adaptation itself is the cause
3. **Disorder sensitivity** — W3 pattern confirms controller is tracking phase structure, not generic noise

---

## Implementation Steps

### Step 1 — Add probe window to simulate_trajectory
```python
# In simulate_trajectory, before main loop:
if probe_steps > 0:
    obs_history, t_history = run_probe_sweep(psi, N, probe_steps, ...)
    # seeds agnostic controller's first latent state
```

### Step 2 — Add fidelity computation (post-hoc)
```python
def state_transfer_fidelity(psi_recovered, psi_target):
    """F = |<psi_target|psi_recovered>|^2"""
    return float(abs(np.dot(psi_target.conj(), psi_recovered))**2)
```

### Step 3 — Add random controller
```python
def ctrl_random(step, N, seed=None, **_): ...
```

### Step 4 — Update analyze_program_l.py with fidelity metrics
Add F_avg, F_worst, FSI, ΔF_DD alongside existing Δτ/AAR columns.

### Step 5 — Deploy at N=10, B=20, 5 controllers, all 6 models

---

## Deployment

```powershell
.\run_program_l_shotgun.ps1 `
  -N 10 -B 20 `
  -Controllers "static dd_xy8 random haware agnostic" `
  -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing" `
  -QueueMaxMin 30 -MaxMin 360 `
  -OutDir "program_n_fidelity" `
  -StoreObs
```

---

## Key Framing Constraint

At every step, keep this operationally grounded:

| Do say | Don't say |
|---|---|
| "learning dynamical response structure" | "discovering quantum geometry" |
| "online coherence spectroscopy" | "mapping hidden channels" |
| "preserving recoverability-supporting observables" | "accessing nonlocal quantum space" |
| "adaptive advantage over DD" | "AI discovered teleportation" |
| "5–15% fidelity lift under disorder" | "breakthrough in quantum communication" |

The current results are already stronger than the earlier speculative framing.
The grounded version is the publishable one.
