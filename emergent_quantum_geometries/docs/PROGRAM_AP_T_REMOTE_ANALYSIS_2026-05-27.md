# Program AP-T Remote Analysis (2026-05-27)

## Scope

- Source: remote SSH reads only
- No SCP or artifact download used
- Summary analyzed in place on Europe TPU via `cat`
- Run: `Program AP - Tunnel-Eigen Guided Folded Recovery`
- Node: `program-ap-eu-node-v1`
- Zone: `europe-west4-a`

## Run configuration

- Mode: `--ap-t`
- Backend: TPU via JAX
- Runtime: `5494.3 s` (~91.6 minutes)
- Depth: `3`
- Noise: `0.30`, `0.45`
- Seeds: `11`, `29`, `47`
- Controllers:
  - `tunnel_eigen_residual_APR`
  - `tunnel_eigen_residual_alpha065`
  - `tunnel_eigen_residual_adaptive`
- Primary adversary:
  - `block_permute_size2_within_random`
- Primary KPI:
  - `worst_control_margin`

## Final AP-T results

### `noise=0.30`

`seed=11`

- `APR`: margin `+0.0533`, noisy cosine `0.9453`
- `alpha065`: margin `+0.0814`, noisy cosine `0.8449`
- `adaptive`: margin `+0.0368`, noisy cosine `0.9412`

`seed=29`

- `APR`: margin `+0.0527`, noisy cosine `0.9124`
- `alpha065`: margin `+0.0792`, noisy cosine `0.9418`
- `adaptive`: margin `+0.0699`, noisy cosine `0.8828`

`seed=47`

- `APR`: margin `+0.0260`, noisy cosine `0.9258`
- `alpha065`: margin `+0.1039`, noisy cosine `0.8908`
- `adaptive`: margin `+0.0667`, noisy cosine `0.9065`

### `noise=0.45`

`seed=11`

- `APR`: margin `+0.0856`, noisy cosine `0.8865`
- `alpha065`: margin `+0.0255`, noisy cosine `0.9439`
- `adaptive`: margin `+0.0477`, noisy cosine `0.9127`

`seed=29`

- `APR`: margin `+0.1117`, noisy cosine `0.9065`
- `alpha065`: margin `+0.0201`, noisy cosine `0.9329`
- `adaptive`: margin `+0.0763`, noisy cosine `0.9352`

`seed=47`

- `APR`: margin `+0.0217`, noisy cosine `0.8941`
- `alpha065`: margin `+0.0369`, noisy cosine `0.8752`
- `adaptive`: margin `+0.1043`, noisy cosine `0.9617`

## Compact readout

Mean worst-control margin by noise:

| controller | noise=0.30 mean | noise=0.30 min | noise=0.45 mean | noise=0.45 min |
| --- | ---: | ---: | ---: | ---: |
| `APR` | `0.044` | `+0.026` | `0.073` | `+0.022` |
| `alpha065` | `0.088` | `+0.079` | `0.027` | `+0.020` |
| `adaptive` | `0.058` | `+0.037` | `0.076` | `+0.048` |

Mean noisy recovery cosine by noise:

| controller | noise=0.30 mean | noise=0.45 mean |
| --- | ---: | ---: |
| `APR` | `0.928` | `0.896` |
| `alpha065` | `0.892` | `0.917` |
| `adaptive` | `0.910` | `0.937` |

## What AP-T proved

All three controllers stayed positive on all `6/6` seed-noise cells, so the residual tunnel-eigen pathway remained stable throughout the compact matrix.

The adaptive controller did not dominate both fixed points. Instead:

- `alpha065` won the moderate-noise margin regime at `noise=0.30`
- `adaptive` won the high-noise regime at `noise=0.45`
- `adaptive` also had the strongest high-noise fidelity

This means the AP-S-Fidelity crossover story was not reproduced in the same form. AP-T changed the picture:

- APR is no longer the low-noise margin winner in this matrix
- `alpha065` is the strongest moderate-noise control point
- `adaptive` is the strongest high-noise robustness-plus-fidelity point

## Interpretation

The adaptive rule still looks mechanistically useful, but for a narrower claim:

> A noise-adaptive residual tunnel-eigen controller preserves high-noise robustness while improving recovered fidelity, but fixed `alpha=0.65` remains the stronger moderate-noise margin controller in the tested AP-T matrix.

The standout AP-T cell is:

- `noise=0.45`, `seed=47`, `adaptive`
- margin `+0.1043`
- noisy cosine `0.9617`

That is the best joint high-noise robustness/fidelity point in the run.

## Implication for the next run

The next probe should no longer ask adaptive alpha to beat every fixed-alpha point everywhere. It should target a better floor at moderate noise while keeping the high-noise gains.

A cleaner next candidate is a shifted rule whose low-noise value stays closer to `0.65`, rather than dropping to `0.59` at `noise=0.30`.
