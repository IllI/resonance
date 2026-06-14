# CHRONOS-MARGINAL-DRIFT-1 Run Spec - 2026-06-14

## Purpose

`AQ-DLINOSS-CHRONO-0` produced a clean null on heldout transition gain, but its independence gate still showed a marginal broadband correlation:

```text
r_full_zero_lag ~= 0.5744
```

`CHRONOS-MARGINAL-DRIFT-1` tests whether that broadband similarity is persistent hardware/extractor structure or same-time collection coupling.

## Collection Plan

Use compact Q vectors only:

```text
samples = 32768
sample_hz = 128
duration ~= 256 s
size = 96
```

Required comparisons:

```text
A0 = Alice us-east1-d at time T
B0 = Bob europe-west4-a at time T
B1 = Bob europe-west4-a at T + 1 hour
```

Optional later comparisons:

```text
B2 = Bob europe-west4-a at T + 6 hours
A1 = Alice us-east1-d at T + 1 hour
```

Because concurrent TRC allocation is painful, the first diagnostic only keeps the Europe node active for `B1` and releases the US node after `A0`.

## Primary Endpoint

Do not use Schumann as primary.

Use:

```text
r_full_time_shifted
```

and compare:

```text
r_full_same_window vs r_full_time_shifted
```

## Interpretation

Case A:

```text
r_full(T, T+1h) ~= r_full(T, T)
```

Interpretation:

```text
The similarity is probably hardware-structural: same TPU type, workload, feature extractor, and normalized spectral/entropy envelope. r_full is not a useful coupling metric.
```

Case B:

```text
r_full(T, T+1h) -> near 0
```

Interpretation:

```text
The systems share a time-window-dependent reference, likely infrastructure or environmental. This is still diagnostic unless heldout transition gain passes.
```

Case C:

```text
full, grid, Schumann, anti, and residual bands change differently
```

Interpretation:

```text
Band-specific drift maps which feature families are hardware-structural versus time-window-specific.
```

## Evidence Policy

Correlation is diagnostic. The main evidence gate remains:

```text
heldout_cross_system_transition_gain > null controls
```

This prevents the experiment from over-reading another attractive carrier/correlation story.

## Current Launch

Use the completed `AQ-DLINOSS-CHRONO-0` pair as:

```text
A0 = alice_us_chrono0_q.json
B0 = bob_eu_chrono0_q.json
```

Launch `B1` on the Europe node with:

```text
node_label = bob_eu_chrono0_plus1h
seed = 15
samples = 32768
sample_hz = 128
size = 96
start_epoch = B0.ended_epoch + 3600
```

Run:

```powershell
python .\chronos_time_emission\chronos_marginal_drift_analysis.py `
  --alice .\chronos_time_emission\results0b_independence_remote\alice_us_chrono0_q.json `
  --same-bob .\chronos_time_emission\results0b_independence_remote\bob_eu_chrono0_q.json `
  --shifted-bob .\chronos_time_emission\results0b_independence_remote\bob_eu_chrono0_plus1h_q.json `
  --out .\chronos_time_emission\results0b_independence_remote\marginal_drift1
```

## Result

The `T + 1 hour` Bob capture completed as `bob_eu_chrono0_plus1h_q.json`.

Primary result:

```text
verdict = hardware_structural_or_extractor_similarity
r_full_same_window = 0.5744
r_full_shifted_window = 0.4957
delta_full = 0.0787
```

Band table:

| band | same_window | shifted_window | same_minus_shifted |
| --- | ---: | ---: | ---: |
| full | 0.5744 | 0.4957 | 0.0787 |
| low_drift | 0.1055 | 0.1862 | -0.0807 |
| schumann | 0.2096 | 0.0705 | 0.1391 |
| anti | 0.6528 | 0.5978 | 0.0551 |
| grid | 0.0398 | -0.0537 | 0.0935 |
| high_residual | 0.0272 | -0.0274 | 0.0546 |

Interpretation:

```text
The broadband similarity persists after a deliberate one-hour time shift. This makes r_full more consistent with hardware/extractor structure than same-window coupling. Band-specific movement remains useful diagnostically, but correlation should not be promoted as evidence. The D-LinOSS heldout transition endpoint remains the truth gate.
```
