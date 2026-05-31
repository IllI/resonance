# Program AP-U Remote Analysis - 2026-05-29

## Scope

AP-U was run on the active U.S. TPU node:

- Node: `program-ap-us-node-v1`
- Zone: `us-east1-d`
- TPU: `v6e-4`
- Mode: `--ap-u`
- Backend: JAX on TPU
- Data policy: remote log/table inspection only; no result artifact download

The compact summary JSON exists remotely at:

- `/home/cityz/program_ap/program_ap_results/program_ap_summary.json`

It was not downloaded for this analysis pass. The conclusions below are from the final printed remote table.

## Stage 0: Reference Lock

Stage 0 passed after key-locking the AP-T comparator path to the AP-S reference recovery slot.

All `APR_reference_lock` and `APR_APT_path` cells matched exactly:

| noise | seed | margin |
| --- | ---: | ---: |
| `0.30` | `11` | `+0.0803` |
| `0.30` | `29` | `+0.0762` |
| `0.30` | `47` | `+0.1007` |
| `0.45` | `11` | `+0.0258` |
| `0.45` | `29` | `+0.0183` |
| `0.45` | `47` | `+0.0287` |

Interpretation:

- The AP-S APR reference is reproducible inside AP-U.
- The earlier AP-T/AP-U drift was caused by recovery key/fold-in construction, not by a genuine controller effect.
- Stage 1 rankings are now interpretable against a stable APR baseline.

## Stage 1 Results

Stage 1 compared:

- `tunnel_eigen_residual_alpha065`
- `tunnel_eigen_residual_adaptive`
- `tunnel_eigen_residual_adaptive_v2_shifted`

### Margin Summary

| controller | positive cells | mean margin | min margin | noise=0.30 mean | noise=0.45 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alpha065` | `6/6` | `+0.0668` | `+0.0411` | `+0.0586` | `+0.0750` |
| `adaptive_v1` | `5/6` | `+0.0644` | `-0.0031` | `+0.0905` | `+0.0382` |
| `adaptive_v2_shifted` | `6/6` | `+0.0627` | `+0.0473` | `+0.0646` | `+0.0608` |

### Fidelity Summary

| controller | min noisy cosine | noise=0.30 mean noisy | noise=0.45 mean noisy |
| --- | ---: | ---: | ---: |
| `alpha065` | `0.8773` | `0.9081` | `0.9363` |
| `adaptive_v1` | `0.8616` | `0.9039` | `0.9121` |
| `adaptive_v2_shifted` | `0.8515` | `0.9231` | `0.9185` |

## Promotion Outcome

The promotion rule required:

1. Stage 0 pass.
2. Positive margin on all `6/6` cells.
3. Beats APR on at least `4/6` cells.
4. Beats `alpha065` on at least `3/6` cells.
5. No noisy cosine below `0.80`.

Outcome:

| controller | result |
| --- | --- |
| `adaptive_v1` | Not promoted: one negative high-noise margin (`noise=0.45`, `seed=29`) and only `3/6` cells beat APR. |
| `adaptive_v2_shifted` | Not promoted: stable on `6/6` and beats APR on `4/6`, but beats `alpha065` on only `2/6`. |
| `alpha065` | Keep as current operating point. It is the safest Stage 1 controller and has the best overall/high-noise mean margin. |

## Scientific Interpretation

AP-U successfully separated calibration from promotion.

The important result is not that adaptive alpha failed broadly. The result is sharper:

- Key-locked AP-U fixed the comparator baseline.
- `adaptive_v1` can produce strong moderate-noise wins, but it is not robust at high noise.
- `adaptive_v2_shifted` is stable and competitive, but it does not beat fixed `alpha=0.65` strongly enough to promote.
- Fixed `alpha=0.65` remains the current best operating point for the next semantic transport run.

## Next Run Decision

Proceed to AQ-0 only with the AP-U key-lock retained.

AQ-0 default controller:

- `tunnel_eigen_residual_alpha065`

Optional comparison arm, if runtime allows:

- `adaptive_v2_shifted`

Do not use `adaptive_v1` as a default controller for AQ-0 because it failed the `6/6` positive-margin robustness requirement.

## Billing / Data Handling Note

This analysis used remote log/table inspection rather than downloading the `111K` summary JSON. Future runs should keep this policy:

- inspect remote logs with short `tail`
- download only tiny summaries when necessary
- never download result folders or raw artifacts
- delete the TPU immediately after extracting the needed tiny record

