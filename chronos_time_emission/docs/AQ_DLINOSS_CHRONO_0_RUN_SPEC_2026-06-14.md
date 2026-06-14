# AQ-DLINOSS-CHRONO-0 Run Spec - 2026-06-14

## Purpose

The next run should stop asking whether a specific physical band is the shared carrier. The first two-host CHRONOS run produced a strong Schumann-minus-anti result, but the preregistered `rep1` run did not replicate that endpoint. That means the next scientific question is not "is Schumann real here?" but:

```text
Do independent TPU host trajectories share a learnable temporal phase-transition geometry?
```

`AQ-DLINOSS-CHRONO-0` reframes Schumann, grid, cloud infrastructure timing, Page-Wootters-style relational clocks, and null drift as competing model classes. The winner is whichever model best predicts heldout cross-system phase-transition structure under hard controls.

## Paper-Safe Claim Target

If the run succeeds, the defensible claim is:

```text
D-LinOSS can test whether independent network timing trajectories share a latent temporal phase-transition geometry. When such a geometry exists, it can be used as a gauge coordinate for folded-state alignment.
```

This deliberately avoids claiming a Schumann carrier, objective collapse, or physical quantum teleportation.

## Data Collection

Use the existing two-node independence collector as the acquisition base:

- Alice: one TRC TPU VM in `us-east1-d`
- Bob: one TRC TPU VM in `europe-west4-a`
- TPU type: `v6e-8` spot
- Samples: `32768`
- Sample rate: `128 Hz`
- Duration: about `256 s`
- Seed: new seed, preferably `14`
- Download policy: compact Q/state summaries only; do not download raw timing traces

The collection artifact should remain small enough for TRC-safe transfer:

```text
q_latency
fs_actual_hz
node label
start/end wall-clock metadata
NTP/chrony metadata if available
```

## Feature Extraction

For each node, build a temporal state trajectory:

```text
z_A(t), z_B(t)
```

where `z(t)` is not raw latency. It should be a compact learned/engineered state vector built from:

- robust-normalized latency Q windows
- temporal entropy profile
- spectral band energies
- phase curvature
- transition residuals
- operator drift statistics
- clock-state entropy proxy

Minimum first-run feature vector:

```text
z(t) = [
  q_window_stats,
  schumann_band_energy,
  anti_band_energy,
  grid_band_energy,
  low_drift_energy,
  high_residual_energy,
  instantaneous_phase,
  phase_velocity,
  phase_curvature,
  local_entropy,
  transition_residual_norm
]
```

This is intentionally conservative. Add temporal MERA later only if this first gate shows a real heldout transition gain.

Lock the first implementation to the following formulas so the run cannot drift during analysis:

```text
window_t = q[t : t + W], with W = 256 samples and hop = 128 samples

q_window_stats =
  [mean(window_t), std(window_t), median(window_t), MAD(window_t)]

band_energy(band) =
  mean(|FFT_bandpass(window_t, band)|^2)

instantaneous_phase =
  angle(hilbert(FFT_bandpass(q, 6 Hz, 40 Hz)))[center_t]

phase_velocity =
  unwrap(phase)[center_t] - unwrap(phase)[center_t - hop]

phase_curvature =
  unwrap(phase)[center_t + hop]
  - 2 * unwrap(phase)[center_t]
  + unwrap(phase)[center_t - hop]

local_entropy =
  -sum_i p_i log2(p_i), where p_i is a 16-bin histogram of robust-normalized window_t

transition_residual_norm =
  ||z_base(t) - (W_1step z_base(t - 1) + b_1step)||_2
```

`z_base(t)` means all feature channels except `transition_residual_norm`. `W_1step` and `b_1step` are fit only on the first 10% of the calibration half, then frozen for all later windows.

The first pass should avoid implementation-dependent labels like "clock entropy proxy" unless they are one of the formulas above.

## Candidate Model Classes

### Model 0 - Null Independent Drift

Alice and Bob trajectories have no shared transition operator.

Expected:

```text
low cross-system predictability
no stable lag
no phase-transition alignment
```

### Model 1 - External Carrier

A shared external band drives both systems. This class includes Schumann, grid, cloud environmental rhythms, and other band-limited timing structure.

Expected:

```text
specific band improves heldout alignment
fixed preregistered lag or zero-lag relation
carrier projection beats null controls
```

### Model 2 - Infrastructure Clock

The shared reference comes from cloud timing, provider control-plane behavior, runtime scheduling, or related infrastructure.

Expected:

```text
band identity may drift across runs
same-provider or same-runtime pairs should outperform cross-provider pairs
alignment depends on infrastructure metadata
```

This is currently the most mundane plausible explanation and should be treated seriously.

### Model 3 - Relational Clock

The systems carry an internal clock-like coordinate. Alignment is best when conditioned on state-derived clock phase rather than wall-clock time.

Represent a history state as:

```text
|Psi> = sum_t |t>_C |psi(t)>_S
```

Compare Alice and Bob by clock-conditioned state distance:

```text
D_W(A_t, B_t') = arccos(|<psi_A(t) | psi_B(t')>|)
```

Expected:

```text
state-space clock alignment beats external-band alignment
wrong-clock and shuffled-clock controls fail
relational clock improves heldout reconstruction at matched resource quality
```

Important limitation:

```text
AQ-DLINOSS-CHRONO-0 uses two Google TRC TPU hosts, so it cannot conclusively distinguish Model 2 from Model 3. If the relational-clock arm wins, the result is a candidate internal-clock alignment under shared provider conditions, not evidence against infrastructure timing.
```

The Model 2 vs Model 3 follow-up must add one of:

```text
same-zone/different-host Google arm
cross-provider Bob arm, e.g. Kaggle or Colab GPU
cross-day repeated Google arm with identical endpoint
```

### Model 4 - Exotic Residual

Only considered if Models 1-3 fail under strong controls while the effect persists cross-provider, cross-region, cross-hardware, and cross-day.

Expected:

```text
pre-registered observables persist independently of cloud timing metadata
effect survives provider/hardware changes
external and infrastructure explanations fail
```

Do not promote this model in `AQ-DLINOSS-CHRONO-0`.

## Primary Endpoint

Do not use raw band correlation as the primary endpoint.

Use:

```text
heldout_cross_system_transition_gain
```

Defined as:

```text
G = score(T_A_to_B on heldout true pair)
  - score(T_A_to_B on shuffled/severed controls)
```

In plain language:

```text
Can a transition law learned from Alice predict Bob's heldout phase-transition geometry better than null controls?
```

Recommended score:

```text
score = cosine(z_B_true_heldout, z_B_pred_heldout)
```

with an optional secondary error:

```text
transition_mse = mean(||z_B_true_heldout - z_B_pred_heldout||^2)
```

## Calibration/Payload Split

Use a strict split:

```text
calibration: first 50%
payload:     second 50%
```

All model selection, lag selection, band selection, and relational-clock fitting happen on the calibration half. The payload half is held out.

## Required Arms

Run these arms on identical Alice/Bob payloads:

| Arm | Purpose |
| --- | --- |
| `true_A_B` | True Alice-to-Bob transition prediction |
| `shuffled_B` | Bob temporal order destroyed |
| `wrong_time_B` | Bob shifted outside selected lag/clock match |
| `severed_same_distribution_B` | Bob distribution preserved, phase relation severed |
| `cross_seed_B` | Different seed/control trajectory |
| `external_band_carrier` | Best calibration-selected band/lag carrier |
| `relational_clock_carrier` | State-derived clock-coordinate alignment |

The key comparison is:

```text
relational_clock_carrier
vs external_band_carrier
vs null controls
```

Define `cross_seed_B` precisely for this run:

```text
cross_seed_B = synthetic Bob control generated from Bob's empirical feature distribution
               using a different RNG seed, preserving per-channel mean/std and
               approximate AR(1) autocorrelation, but severing Alice/Bob phase relation.
```

It is not a second real collection unless explicitly labeled as `cross_run_B`.

## D-LinOSS Transition Models

Fit compact models on calibration windows and evaluate on payload windows.

### Baseline Linear Transition

```text
z_B(t + lag) ~= W z_A(t) + b
```

Use ridge regularization and report `||W||`, condition number, and payload score.

### Latent Generator Transition

Promote this if runtime permits:

```text
u_A(t) = B^H (z_A(t) - mu_A)
u_B(t) = B^H (z_B(t) - mu_B)
u_B(t + dt) ~= exp(A dt) u_A(t) + b
```

Keep `A` in latent space.

First variants:

```text
A_diag_phase = i Omega + D
A_hamiltonian = i H, H = H^H
A_hamiltonian_plus_scale = D_luma exp(i H dt)
```

Do not use a rank-1 one-sample map as the main operator. It can remain a diagnostic only.

## Relational Clock Construction

Compute a clock coordinate `tau(t)` from the calibration half:

```text
1. Robust-normalize each feature channel on calibration data.
2. Fit PCA on the calibration feature matrix.
3. Take the first two PCA coordinates: c1(t), c2(t).
4. Define tau(t) = unwrap(angle(c1(t) + i c2(t))).
```

This definition is fixed for `AQ-DLINOSS-CHRONO-0`. Do not switch to Hilbert phase, empirical-mode phase, or a complex-state eigenphase during analysis.

If a later run uses complex D-LinOSS states, it may preregister:

```text
tau(t) = angle(<v_clock, z(t)>)
```

Then compare alignments by:

```text
wall clock t
external lag t + Delta
relational clock tau_A(t) ~= tau_B(t')
```

The relational model wins only if it improves payload transition gain under controls.

## Gauge-Alignment Hook

If a shared `tau(t)` is learned, use it as the gauge coordinate:

```text
theta_A = alpha tau_A
theta_B = alpha tau_B
F_k_enc = F_k exp(i theta_A)
F_k_dec = Fhat_k exp(-i theta_B)
```

For `AQ-DLINOSS-CHRONO-0`, do not require image reconstruction improvement. Report it only as a secondary hook if cheap.

## Metrics

Primary:

```text
heldout_cross_system_transition_gain
payload_true_score
payload_null_score_mean
payload_null_score_max
relational_minus_external_gain
```

Secondary:

```text
selected_band
selected_lag
wall_clock_score
external_band_score
relational_clock_score
phase_transition_alignment
transition_operator_norm
transition_operator_stability
clock_conditioned_distance
```

Controls:

```text
shuffle_p_value
block_shuffle_p_value
phase_surrogate_p_value
wrong_time_score
severed_score
cross_seed_score
```

Infrastructure metadata:

```text
zone_A
zone_B
start_epoch_A
start_epoch_B
fs_actual_A
fs_actual_B
duration_A
duration_B
ntp_offset_A
ntp_offset_B
```

## Pass Gates

Gate -1: independence

```text
r_full_zero_lag < 0.90
```

Gate 0: no raw-correlation promotion

```text
Do not call the run successful from band correlation alone.
```

Gate 1: heldout transition gain

```text
heldout_cross_system_transition_gain >= 0.10
payload_true_score > payload_null_score_max
```

Gate 2: permutation support

```text
phase_surrogate_p_value <= 0.01
```

Gate 3: model selection

At least one structured model must beat the null controls:

```text
max(external_band_score, relational_clock_score)
  > payload_null_score_max
```

Relational-clock promotion requires:

```text
relational_clock_score - external_band_score >= 0.05
```

## Expected Outcomes

### Outcome A - Null Wins

No heldout transition gain. Interpretation:

```text
No stable cross-system temporal geometry was detected in this run.
```

This is a valid scientific result, not a tuning failure. Given the `rep1` failure, Outcome A is the prior-favored result. It means the two-host temporal geometry is not stable enough for calibration-to-payload prediction under this hardware/window configuration.

### Outcome B - External Carrier Wins

External band alignment beats null and relational clock. Interpretation:

```text
Cross-system structure exists but is best explained as band/carrier coupling.
```

Do not assume Schumann unless the selected band is stable across replications.

### Outcome C - Infrastructure Clock Wins

Band identity drifts but same-provider/runtime metadata predicts alignment. Interpretation:

```text
Cloud infrastructure timing remains the leading explanation.
```

### Outcome D - Relational Clock Wins

Relational state-clock alignment beats external carrier and controls. Interpretation:

```text
D-LinOSS found an internal temporal coordinate that predicts heldout cross-system state geometry better than wall-clock or fixed-band alignment.
```

This is the first result that would justify a Page-Wootters-oriented follow-up.

## Launch Discipline

Follow the TRC rules:

- Use only the allowed TRC TPU zones/resources.
- Use compact summaries only; do not download raw timing traces.
- Keep one `v6e-8` per site unless the protocol explicitly requires otherwise.
- Use `n` for one-time SSH host-key prompts instead of caching new keys.
- Delete TPU VMs and queued resources immediately after the run.

## Immediate Implementation Plan

1. Add `run_chronos_dlinoss_chrono_tpu.py`.
2. Reuse the existing `collect` path from `run_chronos_independence_tpu.py` for compact Q acquisition.
3. Add local analyzer `chronos_dlinoss_chrono0_analysis.py` that consumes compact Alice/Bob JSON files.
4. Implement the minimum feature vector first; skip temporal MERA in the first pass.
5. Run US/EU pair with seed `14`.
6. Analyze calibration/payload heldout transition gain.
7. Only if Gate 1 passes, add richer D-LinOSS latent generator variants.

## Result

The live US/EU pair was collected with:

```text
seed = 14
samples = 32768
sample_hz = 128
size = 96
```

Observed result:

```text
gate_minus_one_passed = True
r_full_zero_lag = 0.5744
heldout_cross_system_transition_gain = -0.0781
payload_true_score = 0.0111
payload_null_score_max = 0.0893
external_band = schumann
external_band_payload_score = -0.0084
relational_clock_payload_score = 0.0212
phase_surrogate_p_value = 0.3881
gate_1_passed = False
gate_2_passed = False
gate_3_passed = False
```

Interpretation:

```text
AQ-DLINOSS-CHRONO-0 produced Outcome A. No stable heldout cross-system temporal geometry was detected. Neither the external-band arm nor the relational-clock arm beat the null controls on the payload half.
```

This is a valid result, not a tuning miss. Under this hardware/window configuration, cross-system temporal structure was not stable enough for calibration-to-payload prediction.

## Bottom Line

`AQ-DLINOSS-CHRONO-0` should not chase another Schumann-band win. It should ask whether independent network trajectories share a learned phase-transition law, and whether that law is better explained by an external carrier, infrastructure clock, relational clock, or null drift.
