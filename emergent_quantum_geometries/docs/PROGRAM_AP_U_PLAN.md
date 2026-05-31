# Program AP-U Plan

## Purpose

AP-U is a self-calibrating variant-promotion run. Its first job is to determine whether the APR reference path reproduces the AP-S-MECH/AP-S-Fidelity APR margins inside the current execution mode. Its second job is to rank adaptive-alpha variants only if that reference lock passes.

Run record:

- `PROGRAM_AP_U_REMOTE_ANALYSIS_2026-05-29.md`

Final outcome:

- Stage 0 passed after key-locking `APR_APT_path` to the AP-S reference recovery slot.
- Stage 1 did not promote an adaptive controller.
- `tunnel_eigen_residual_alpha065` remains the current operating point.

## Stage 0: Reference-Lock Calibration

Run APR twice under explicit key slots:

- `APR_reference_lock`: AP-S-Fidelity APR key slot (`key_ctrl_idx=1`)
- `APR_APT_path`: AP-T comparator path, key-locked to the AP-S reference slot (`key_ctrl_idx=1`)

Historical note:

- Running `APR_APT_path` with the old AP-T slot (`key_ctrl_idx=0`) reproduced the drift.
- Key-locking it to `key_ctrl_idx=1` made all calibration cells match exactly.

Fixed setup:

- depth `3`
- noise `0.30`, `0.45`
- seeds `11`, `29`, `47`
- adversary `block_permute_size2_within_random`

Reference APR margins for the hard gate:

| noise | seed 11 | seed 29 | seed 47 |
| --- | ---: | ---: | ---: |
| `0.30` | `+0.0803` | `+0.0762` | `+0.1007` |

Stage 0 is a hard gate. If any APR margin at `noise=0.30` differs from the reference by more than `0.020`, stop immediately and do not run AQ. This protects all later interpretation from comparator drift.

The `noise=0.45` cells may still be recorded for diagnostics, but promotion and AQ launch are gated by the `noise=0.30` reference lock above.

## Stage 1: Variant Promotion

Run only after Stage 0 passes:

- `tunnel_eigen_residual_alpha065`
- `tunnel_eigen_residual_adaptive`
- `tunnel_eigen_residual_adaptive_v2_shifted`

Adaptive rules:

```python
adaptive_v1 = 0.50 + 0.30 * noise
adaptive_v2 = 0.56 + 0.22 * noise
```

Resolved alphas:

| controller | noise=0.30 | noise=0.45 |
| --- | ---: | ---: |
| `adaptive_v1` | `0.590` | `0.635` |
| `adaptive_v2` | `0.626` | `0.659` |

## Required Metadata

Each semantic-recovery cell records:

- controller name
- resolved projection alpha
- noise and seed
- recovery seed and fold-in key slot
- adversary fold-in labels
- block size and within-block random flag
- scoring weights
- hard-control list
- min-lift aggregation rule
- top-k mode and chosen adaptive mode count
- damping spread
- git commit
- execution mode

## Promotion Rule

Do not promote any Stage 1 controller unless:

1. Stage 0 passes.
2. Margin is positive on all `6/6` seed-noise cells.
3. The controller beats APR on at least `4/6` cells.
4. The controller beats `alpha065` on at least `3/6` cells.
5. No noisy recovery cosine falls below `0.80`.

If no adaptive rule passes all criteria, keep `alpha065` as the current operating point and report the calibration failure or promotion miss explicitly.

## 2026-05-29 Result

| controller | positive cells | mean margin | min margin | decision |
| --- | ---: | ---: | ---: | --- |
| `alpha065` | `6/6` | `+0.0668` | `+0.0411` | keep |
| `adaptive_v1` | `5/6` | `+0.0644` | `-0.0031` | not promoted |
| `adaptive_v2_shifted` | `6/6` | `+0.0627` | `+0.0473` | comparison arm only |

Next experiment:

- AQ-0 noise-free ceiling
- `bond_dim=4`
- `8` explicit semantic motifs
- default controller `tunnel_eigen_residual_alpha065`
- optional comparison arm `adaptive_v2_shifted` only if runtime remains compact
