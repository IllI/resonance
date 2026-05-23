# Program N1 — Control-Response Gradient Estimation

> **Architectural shift:**
> z_t = f(x_{0:t}, u_{0:t}, Δx_{0:t})
> not just z_t = f(x_{0:t})
>
> The controller stops being a state classifier.
> It becomes a dynamical control-system identifier.

---

## The Gap in the Current Architecture

Current agnostic controller:
```
observe x_t → score Clifford effects → select best → apply
```

Missing: the controller does not know how **its own past actions changed the state**.

It may know "X pulse helped at step 10" without learning:
"small perturbations along the X direction **at this phase of transport**
stabilize MI while destabilizing PTM_sv1, and this tradeoff reverses at W3."

That second sentence is what a control-system identifier learns.

---

## What Gets Added

### Response gradient vector

At each control step s, compute:

```
Δx_s = x_s - x_{s - CTRL_INTERVAL}   # observable change since last action
```

This is a finite-difference estimate of the local response:
`∂x/∂u ≈ Δx_s / ||u_s||`

Store per-step alongside obs_history and ctrl_actions.

### Augmented latent state

Instead of scoring `strata_score(x_cur)`:

```python
score = strata_score(x_pred) + alpha * gradient_alignment(Δx_recent, predicted_delta)
```

Where `gradient_alignment` measures whether the Clifford's predicted effect
aligns with the recent positive response direction.

This makes the controller **gradient-following** in observable space —
preferring interventions that extend successful response directions.

### Controllability efficiency (κ_stab)

```
κ_stab = Δrecoverability_positive / n_actions_total
```

Measures: how efficiently control actions generate stabilizing structure.
High κ_stab = controller finds good interventions quickly.
Low κ_stab = controller wastes actions on neutral/bad interventions.

---

## Response Gradient Features

| Quantity | Symbol | Meaning |
|---|---|---|
| EE response | ΔEE/Δu | Entanglement sensitivity to pulse |
| MI response | ΔMI/Δu | Information-flow sensitivity |
| PTM response | ΔPTM_sv1/Δu | Transport susceptibility |
| OS response | ΔOS/Δu | Scrambling response |
| basis_invar response | ΔBI/Δu | Representation stability response |

These become **dynamic control features** — the controller learns which
response directions are constructive for each transport regime.

---

## Why This Explains W3

W3 near-critical regime likely has:
- Rapidly changing local response gradients
- Highly nonstationary susceptibility (controllability windows open/close)
- Fixed DD fails because it assumes ∂x/∂u is stationary

Agnostic controller with gradient tracking can:
- Detect when a control direction becomes constructive (gradient aligns)
- Apply intervention in that window
- Detect when the window closes and revert to restraint

This would explain the partial preservation at W3 (better than DD, worse than static)
without needing to know the Hamiltonian.

---

## Controller Architecture v2

### Phase 0 — Probe sweep (existing chirp concept, now justified)
```
Apply Clifford sweep: I, X, Z, H, S, X
Measure Δx for each gate at t=0..5
Build initial response Jacobian estimate J_0
```

### Phase 1 — Gradient-augmented selection

```python
def ctrl_agnostic_v2(step, obs_history, ctrl_actions, **_):
    x_cur = obs_history[-1]
    
    # Recent response gradient
    if len(obs_history) >= 2:
        dx_recent = obs_history[-1] - obs_history[-2]
    else:
        dx_recent = zeros(N_OBS)
    
    best_idx, best_score = 0, -inf
    for ci in range(N_CLIFF):
        x_pred   = _predict_obs_after_cliff(x_cur, ci)
        delta_pred = x_pred - x_cur
        
        # Strata score (existing)
        s_strata = strata_score(x_pred)
        
        # Gradient alignment bonus: prefer Cliffords whose effect
        # aligns with recent positive response direction
        if norm(dx_recent) > 1e-8 and norm(delta_pred) > 1e-8:
            alignment = dot(delta_pred, dx_recent) / (norm(delta_pred) * norm(dx_recent))
        else:
            alignment = 0.0
        
        # Combined score: strata fidelity + gradient-following bonus
        score = s_strata + GRAD_ALPHA * alignment
        
        if score > best_score:
            best_score = score
            best_idx   = ci
    
    return (best_idx, 0)
```

Where `GRAD_ALPHA = 0.2` (tunable — controls how much gradient-following
vs strata-preservation drives selection).

---

## New Metrics

### Controllability efficiency
```python
kappa_stab = sum(max(0, f_rec[t] - f_rec[t-1]) for t in action_steps) / len(action_steps)
```

### Response Jacobian estimate (per model, per bootstrap)
```python
J_est = {obs_name: [] for obs in OBS_NAMES}
for t in action_steps:
    dx = obs_history[t+1] - obs_history[t-1]   # response window
    for i, obs in enumerate(OBS_NAMES):
        J_est[obs].append(dx[i])
# Mean response per obs channel per model
```

### Gradient alignment score (per trajectory)
```python
alignment_series = [dot(dx_t, predicted_delta_t) / (norm + eps) for t in ...]
mean_alignment = mean(alignment_series)
# >0: controller interventions are gradient-aligned
# <0: controller is anti-aligned (fighting the response)
```

---

## Falsification Update

| Test | Criterion | Meaning |
|---|---|---|
| κ_stab(agnostic_v2) > κ_stab(agnostic_v1) | +10% | Gradient tracking improves efficiency |
| mean_alignment > 0 in XXZ | alignment > 0 | Controller follows constructive gradients |
| mean_alignment ≈ 0 in Ising | |alignment| < 0.1 | No constructive gradient direction in null |
| W3 alignment variance > XXZ | var(align) > var(XXZ) | W3 has nonstationary susceptibility |
| κ_stab(random) << κ_stab(agnostic_v2) | ratio > 2 | Adaptation, not random injection |

---

## What NOT to Do

- Do NOT optimize fidelity directly in the gradient loop
- Do NOT call it "learning quantum geometry"
- Do NOT parameterize continuous amplitudes yet (keep Clifford discrete)
- Do NOT use response gradients as reward signal (use as feature only)

Operationally: the controller is doing **black-box finite-difference estimation
of local control response** — exactly what adaptive NMR/ESR pulse sequences do.

---

## Implementation Steps

### Step 1 — Add response_gradients to simulate_trajectory output
```python
# After each step, store dx = x_t - x_{t-CTRL_INTERVAL}
# Shape: same as obs_history
```

### Step 2 — Update ctrl_agnostic with gradient-alignment bonus
```python
GRAD_ALPHA = 0.2   # pre-registered, frozen before run
```

### Step 3 — Add κ_stab to analyze_program_l.py

### Step 4 — Add gradient_alignment series to --store-obs output

### Step 5 — Deploy Program N1
```powershell
.\run_program_l_shotgun.ps1 `
  -N 10 -B 10 `
  -Controllers "static dd_xy8 random haware agnostic" `
  -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing" `
  -OutDir "program_n1_gradient" `
  -StoreObs
```

---

## Terminology Safeguard

| Internal concept | Safe public description |
|---|---|
| Local control manifold | Local response structure |
| Response Jacobian | Finite-difference control sensitivity |
| Control geometry | Control-response gradient field |
| Dynamical control identifier | Adaptive online system estimator |
| Gradient-following controller | Response-aligned adaptive pulse selection |

The physics is sound. The framing just needs to stay operational.
