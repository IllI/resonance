# Program AQ-0b Remote Analysis - 2026-05-29

## TRC-safe data handling

This record was created from the final stdout tables already inspected on the TPU VM. No result artifact, summary JSON, checkpoint, array dump, or remote directory was downloaded.

Run context:

- project: `time-emission`
- node: `program-aq0b-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-0b`
- backend: JAX TPU
- devices visible: `4`
- node retained after completion for follow-up experiments

## AQ-0b configuration

This was the first explicit AQ semantic ceiling diagnostic:

- payload: locked `8`-motif AQ alphabet
- noise: `0.00`
- depth: `3`
- seeds: `11`, `29`
- controllers: `free`, `tunnel_eigen_residual_alpha055`, `tunnel_eigen_residual_alpha065`
- hard controls: disabled as experiment arms
- recovery train states: `4`
- recovery test states: `2`
- D-LinOSS steps: capped at `24`

AQ-0b added the missing semantic diagnostics:

- untransported ridge sanity
- `L1` occupation recovery
- `L2` transition recovery
- `L3` FFT/phase recovery
- `L4` recurrence recovery
- inter-layer leakage proxy

## Final semantic recovery table

| seed | controller | margin | score | noisy cosine | block lift | topology lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `11` | `free` | `+0.0272` | `0.3319` | `0.8199` | `-0.0035` | `-0.0016` |
| `11` | `tunnel_eigen_residual_alpha055` | `+0.0044` | `0.3078` | `0.9079` | `+0.0028` | `+0.0090` |
| `11` | `tunnel_eigen_residual_alpha065` | `-0.0016` | `0.3028` | `0.8907` | `-0.0050` | `+0.0175` |
| `29` | `free` | `+0.0106` | `0.3283` | `0.8986` | `+0.0048` | `-0.0025` |
| `29` | `tunnel_eigen_residual_alpha055` | `+0.0134` | `0.3098` | `0.8656` | `-0.0086` | `-0.0039` |
| `29` | `tunnel_eigen_residual_alpha065` | `+0.0210` | `0.3108` | `0.8217` | `+0.0321` | `+0.0288` |

## Layer diagnostic table

| seed | controller | sanity | L1 | L2 | L3 | L4 | leak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `11` | `free` | `0.9404` | `0.9637` | `0.8199` | `0.9313` | `0.9518` | `0.0296` |
| `11` | `tunnel_eigen_residual_alpha055` | `0.9358` | `0.9857` | `0.9079` | `0.8793` | `0.9519` | `0.0276` |
| `11` | `tunnel_eigen_residual_alpha065` | `0.9639` | `0.9878` | `0.8907` | `0.9489` | `0.9715` | `0.0152` |
| `29` | `free` | `0.9459` | `0.9653` | `0.8986` | `0.9414` | `0.9511` | `0.0321` |
| `29` | `tunnel_eigen_residual_alpha055` | `0.9317` | `0.9533` | `0.8656` | `0.9234` | `0.9358` | `0.0173` |
| `29` | `tunnel_eigen_residual_alpha065` | `0.9158` | `0.9570` | `0.8217` | `0.9007` | `0.9257` | `0.0302` |

## Aggregate read

| controller | positive cells | mean margin | min margin | mean noisy cosine | mean L2 | mean L3 | mean L4 | mean leak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `2/2` | `+0.0189` | `+0.0106` | `0.8593` | `0.8593` | `0.9364` | `0.9514` | `0.0309` |
| `tunnel_eigen_residual_alpha055` | `2/2` | `+0.0089` | `+0.0044` | `0.8868` | `0.8868` | `0.9014` | `0.9439` | `0.0225` |
| `tunnel_eigen_residual_alpha065` | `1/2` | `+0.0097` | `-0.0016` | `0.8562` | `0.8562` | `0.9248` | `0.9486` | `0.0227` |

## Interpretation

AQ-0b changed the story in an important way: the feared fine-layer collapse did not happen.

What held up:

- untransported ridge sanity stayed strong for all controllers
- `L1` remained very strong across the board
- `L3` and `L4` remained high for all controllers
- inter-layer leakage stayed low for all controllers

What actually separated the controllers was not semantic ceiling failure, but the tradeoff between noise-free margin and transition-layer behavior:

- `free` had the best mean semantic margin in this noise-free ceiling run
- `alpha055` had the best mean aggregate cosine and best mean `L2` transition recovery
- `alpha065` did not dominate at the ceiling and went negative on seed `11` margin, even though its `L1`, `L3`, and `L4` values stayed strong

The seed `11` `alpha065` miss was a control-margin miss, not a semantic ceiling collapse: its negative cell came from the internal block-control comparison while the semantic-layer diagnostics stayed strong.

That means the earlier AQ-0 proxy worry was too pessimistic. The explicit `8`-motif AQ payload is recoverable at `noise=0.00`. The main open question is now controller choice under noise, not whether the semantic encoder/decoder path is fundamentally broken.

## Decision

AQ-0b is good enough to advance.

Do not spend the next run on damping micro-tuning yet. AQ-0b did not show the predicted `L3`/`L4` collapse under `alpha065`, so the immediate next question is comparative noisy transport.

Recommended next run: `AQ-1a`

- keep the explicit `8`-motif payload
- keep depth `3`
- keep seeds `11`, `29`
- run noise `0.30` and `0.45`
- keep controllers `free`, `tunnel_eigen_residual_alpha055`, and `tunnel_eigen_residual_alpha065`
- keep reporting `margin`, aggregate cosine, `L1`-`L4`, and leakage

Why this next step:

- `free` is the current noise-free margin baseline
- `alpha055` is the best current semantic fidelity and `L2` baseline
- `alpha065` remains the candidate for stronger robustness under noise

Only after AQ-1a shows a real noisy crossover should we spend TPU time on a projection-alpha or damping-spread micro-sweep.
