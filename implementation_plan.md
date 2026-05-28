# Implementation Plan - Program AP After AP-T

## Current status

Program AP-T completed successfully on the active Europe TPU node and was analyzed remotely without downloading the result artifact.

Remote summary source:

- `/home/cityz/program_ap/program_ap_results/program_ap_summary.json`

TRC-safe analysis method:

- `gcloud compute tpus tpu-vm ssh ... --command="cat ...summary.json"`

## Confirmed result

AP-T tested the adaptive-alpha hypothesis directly:

- depth `3`
- noise `0.30`, `0.45`
- seeds `11`, `29`, `47`
- adversary `block_permute_size2_within_random`
- controllers:
  - `tunnel_eigen_residual_APR` (`alpha=0.55`)
  - `tunnel_eigen_residual_alpha065`
  - `tunnel_eigen_residual_adaptive` (`alpha = 0.50 + 0.30 * noise`)

Runtime:

- `5494.3 s` (~91.6 minutes)

All three controllers stayed positive on all `6/6` seed-noise cells, and all noisy recovery cosines stayed above `0.84`.

## AP-T result

Mean worst-control margin by noise:

| controller | noise=0.30 mean | noise=0.30 min | noise=0.45 mean | noise=0.45 min |
| --- | ---: | ---: | ---: | ---: |
| `alpha055/APR` | `0.044` | `+0.026` | `0.073` | `+0.022` |
| `alpha065` | `0.088` | `+0.079` | `0.027` | `+0.020` |
| `adaptive` | `0.058` | `+0.037` | `0.076` | `+0.048` |

Mean noisy recovery cosine by noise:

| controller | noise=0.30 mean | noise=0.45 mean |
| --- | ---: | ---: |
| `alpha055/APR` | `0.928` | `0.896` |
| `alpha065` | `0.892` | `0.917` |
| `adaptive` | `0.910` | `0.937` |

## Interpretation

AP-T did not confirm the original crossover story. It revised it.

What happened:

- `alpha065` won the moderate-noise margin regime at `noise=0.30`.
- Adaptive won the high-noise regime at `noise=0.45`.
- APR remained stable, but it was no longer the margin leader at either noise level.

That means the stronger claim is not "alpha should scale up with noise because `0.55` wins low noise and `0.65` wins high noise." The stronger claim is:

> A noise-adaptive residual tunnel-eigen controller preserves the high-noise robustness gains of stronger projection while improving recovered fidelity, but the fixed `alpha=0.65` controller still gives the best moderate-noise control margin in this compact AP-T matrix.

This is still a real mechanism result, just a different one than the bracket suggested.

## What AP-T proved

Adaptive succeeded on the stability/fidelity side:

- positive margin on all `6/6` cells
- noisy cosine above `0.90` in `5/6` cells and `0.8828` in the remaining moderate-noise cell
- strongest high-noise mean margin (`0.076`) and strongest high-noise mean noisy cosine (`0.937`)
- strongest single-cell result in the whole AP-T table:
  - `noise=0.45`, `seed=47`
  - margin `+0.1043`
  - noisy cosine `0.9617`

Adaptive failed the original AP-T low-noise dominance criterion:

- at `noise=0.30`, adaptive mean margin `0.058` did not beat `alpha065` mean margin `0.088`

## Recommended next step

The best next experiment is no longer "prove adaptive dominates fixed alpha everywhere." It is to separate the two operating regimes explicitly:

1. `alpha065` as the best moderate-noise margin controller.
2. `adaptive` as the best high-noise robustness-plus-fidelity controller.
3. A follow-up compact run that tests whether a different adaptive rule can recover the low-noise margin edge without giving up the high-noise gains.

The cleanest next probe is likely a shifted rule such as:

```python
projection_alpha = 0.60 + scale * (noise_level - 0.30)
```

or a two-regime controller that bottoms out closer to `0.65` instead of `0.59` at `noise=0.30`.

## TRC / billing discipline

- Europe analysis stayed remote-first.
- No SCP result download was used.
- Only SSH `cat` and `tail` were used for logs and summaries.
- The same active node was reused throughout the AP-T cycle.

## Bottom line

AP-T completed and gave a stronger, more specific story than AP-S-Fidelity alone:

- fixed `alpha=0.65` is best on moderate-noise margin in this compact matrix
- adaptive alpha is best on high-noise robustness plus fidelity
- the residual tunnel-eigen mechanism remains stable across all tested cells
