# Program AQ-1a-TF Gate Remote Analysis - 2026-05-29

## TRC-safe data handling

This record was created from the final stdout tables already inspected on the TPU VM. No result artifact, summary JSON, checkpoints, arrays, or remote directories were downloaded.

Run context:

- project: `time-emission`
- node: `program-aq0b-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-1a-tf`
- backend: JAX TPU
- devices visible: `4`
- runtime-constrained gate: seed `11` only

## Configuration

- payload: explicit `8`-motif AQ alphabet with transformation pairs
- transformation pairs:
  - `cyclic -> chirp`
  - `palindrome -> braided`
  - `sparse -> bifurcating`
  - `fractal_nested -> alternating`
- noise: `0.30`, `0.45`
- depth: `3`
- seeds: `11`
- controllers:
  - `free`
  - `tunnel_eigen_residual_alpha055`
  - `tunnel_eigen_residual_alpha065`
- hard controls: disabled as experiment arms

## Semantic recovery table

| noise | seed | controller | margin | score | noisy cosine | block lift | topology lift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `11` | `free` | `+0.3123` | `0.5109` | `0.9193` | `+0.5320` | `+0.3294` |
| `0.30` | `11` | `alpha055` | `+0.4457` | `0.5641` | `0.9016` | `+0.3073` | `+0.3900` |
| `0.30` | `11` | `alpha065` | `+0.3879` | `0.5358` | `0.9253` | `+0.2103` | `+0.2921` |
| `0.45` | `11` | `free` | `+0.3820` | `0.5527` | `0.9193` | `+0.2411` | `+0.2830` |
| `0.45` | `11` | `alpha055` | `+0.2698` | `0.4564` | `0.8950` | `+0.4551` | `+0.4489` |
| `0.45` | `11` | `alpha065` | `+0.3139` | `0.4893` | `0.9462` | `+0.3990` | `+0.2937` |

## Layer table (`L1`-`L5`)

| noise | seed | controller | sanity | L1 | L2 | L3 | L4 | L5 | leak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `11` | `free` | `0.9271` | `0.9913` | `0.9355` | `0.9152` | `0.9640` | `0.9226` | `0.0679` |
| `0.30` | `11` | `alpha055` | `0.9401` | `0.9892` | `0.9311` | `0.9060` | `0.9596` | `0.8972` | `0.0753` |
| `0.30` | `11` | `alpha065` | `0.9416` | `0.9926` | `0.9482` | `0.9166` | `0.9662` | `0.9201` | `0.0666` |
| `0.45` | `11` | `free` | `0.9271` | `0.9913` | `0.9355` | `0.9152` | `0.9640` | `0.9226` | `0.0679` |
| `0.45` | `11` | `alpha055` | `0.9401` | `0.9901` | `0.9201` | `0.8982` | `0.9574` | `0.8955` | `0.0751` |
| `0.45` | `11` | `alpha065` | `0.9416` | `0.9946` | `0.9725` | `0.9316` | `0.9719` | `0.9350` | `0.0608` |

## Transformation table

| noise | seed | controller | L5 transform recovery | transform exact match | source recovery | target recovery | class recovery | delta graph cosine | delta phase cosine | delta recurrence match |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `11` | `free` | `0.9226` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9277` | `0.9222` | `0.9341` |
| `0.30` | `11` | `alpha055` | `0.8972` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9074` | `0.8873` | `0.9159` |
| `0.30` | `11` | `alpha065` | `0.9201` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9311` | `0.9095` | `0.9313` |
| `0.45` | `11` | `free` | `0.9226` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9277` | `0.9222` | `0.9341` |
| `0.45` | `11` | `alpha055` | `0.8955` | `0.7500` | `1.0000` | `0.7500` | `1.0000` | `0.8987` | `0.8918` | `0.9112` |
| `0.45` | `11` | `alpha065` | `0.9350` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9442` | `0.9166` | `0.9454` |

## Interpretation

The seed `11` gate passed on its own terms. `L5` transformation recovery was robust and non-trivial at both noises, and transformation decoding did not collapse under noise for this seed.

At `noise=0.30`, all controllers reached `transform_exact_match = 1.0000`.
At `noise=0.45`, `free` and `alpha065` stayed at `1.0000`, while `alpha055` dropped to `0.7500` through target recovery.

The strongest high-noise transformation profile in this gate is `alpha065`:

- best noisy cosine (`0.9462`)
- best `L5` (`0.9350`)
- perfect transform exact match (`1.0000`)
- strongest delta-graph/recurrence cosines among the controllers

## Seed 29 follow-up

Seed `29` was later run as a constrained follow-up and changed the AQ-1a-TF conclusion.

Follow-up record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ1A_TF_S29_REMOTE_ANALYSIS_2026-05-29.md`

Seed `29` showed that `L5` remains measurable, but exact transformation decoding is not stable across seed/noise:

- best `transform_exact_match` at seed `29` was `0.5000`
- several controller/noise cells fell to `0.0000`
- `alpha065` preserved `L5 = 0.9148` at `noise=0.30`, but fell to `L5 = 0.5368` at `noise=0.45`
- `free` was steadier at high noise on seed `29`, but did not recover exact transforms reliably

The revised interpretation is that AQ-1a-TF likely recovered a fragile endpoint association or transform label, not a stable operator.

## Decision

Do not widen this into a longer AQ-1a-TF matrix.

Next experiment should shift objective from sequence recovery to operator-conditioned recovery:

- move to `AQ-1b`
- payload: `source + operator` (target held out from transported payload)
- recovery target: derive `target_hat = operator_hat(source_hat)`
- include anti-leak control to verify target is not directly encoded
