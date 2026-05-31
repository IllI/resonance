# Program AQ-1a-TF Seed 29 Remote Analysis - 2026-05-29

## TRC-safe data handling

This record was created from the final stdout tables inspected on the TPU VM. No result artifact, summary JSON, checkpoints, arrays, or remote directories were downloaded.

Run context:

- project: `time-emission`
- node: `program-aq1b-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-1a-tf-s29`
- backend: JAX TPU
- devices visible: `4`
- runtime-constrained follow-up: seed `29` only

## Configuration

- payload: explicit `8`-motif AQ alphabet with transformation pairs
- transformation pairs:
  - `cyclic -> chirp`
  - `palindrome -> braided`
  - `sparse -> bifurcating`
  - `fractal_nested -> alternating`
- noise: `0.30`, `0.45`
- depth: `3`
- seeds: `29`
- controllers:
  - `free`
  - `tunnel_eigen_residual_alpha055`
  - `tunnel_eigen_residual_alpha065`
- hard controls: disabled as experiment arms

## Semantic recovery table

| noise | seed | controller | margin | score | noisy cosine | block lift | topology lift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `29` | `free` | `+0.0779` | `0.3615` | `0.8092` | `+0.0791` | `+0.0665` |
| `0.30` | `29` | `alpha055` | `+0.0135` | `0.2930` | `0.6695` | `-0.0113` | `+0.0160` |
| `0.30` | `29` | `alpha065` | `+0.0855` | `0.3574` | `0.9353` | `+0.1882` | `+0.0947` |
| `0.45` | `29` | `free` | `+0.1878` | `0.4297` | `0.8377` | `+0.1362` | `+0.2112` |
| `0.45` | `29` | `alpha055` | `+0.0555` | `0.3328` | `0.8823` | `+0.0327` | `+0.0376` |
| `0.45` | `29` | `alpha065` | `+0.0044` | `0.2903` | `0.6957` | `+0.0159` | `+0.0057` |

## Layer table (`L1`-`L5`)

| noise | seed | controller | sanity | L1 | L2 | L3 | L4 | L5 | leak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `29` | `free` | `0.9070` | `0.9810` | `0.8615` | `0.9744` | `0.9750` | `0.7741` | `0.0356` |
| `0.30` | `29` | `alpha055` | `0.7282` | `0.9409` | `0.8280` | `0.8577` | `0.9017` | `0.4032` | `0.0382` |
| `0.30` | `29` | `alpha065` | `0.9599` | `0.9903` | `0.9612` | `0.9792` | `0.9861` | `0.9148` | `0.0462` |
| `0.45` | `29` | `free` | `0.9206` | `0.9927` | `0.8533` | `0.9311` | `0.9721` | `0.8805` | `0.0583` |
| `0.45` | `29` | `alpha055` | `0.8935` | `0.9807` | `0.9432` | `0.9173` | `0.9645` | `0.7529` | `0.0443` |
| `0.45` | `29` | `alpha065` | `0.7869` | `0.9396` | `0.7831` | `0.8716` | `0.9181` | `0.5368` | `0.0587` |

## Transformation table

| noise | seed | controller | L5 transform recovery | transform exact match | source recovery | target recovery | class recovery | delta graph cosine | delta phase cosine | delta recurrence match |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `29` | `free` | `0.7741` | `0.2500` | `0.5000` | `0.2500` | `0.5000` | `0.5517` | `0.9066` | `0.5516` |
| `0.30` | `29` | `alpha055` | `0.4032` | `0.0000` | `0.5000` | `0.0000` | `0.0000` | `-0.1792` | `0.8155` | `-0.3383` |
| `0.30` | `29` | `alpha065` | `0.9148` | `0.5000` | `0.5000` | `0.5000` | `0.5000` | `0.8229` | `0.9796` | `0.7193` |
| `0.45` | `29` | `free` | `0.8805` | `0.2500` | `0.5000` | `0.2500` | `1.0000` | `0.7214` | `0.9417` | `0.8694` |
| `0.45` | `29` | `alpha055` | `0.7529` | `0.0000` | `0.2500` | `0.5000` | `0.5000` | `0.7140` | `0.8700` | `0.4788` |
| `0.45` | `29` | `alpha065` | `0.5368` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0597` | `0.8323` | `-0.3294` |

## Interpretation

Seed `29` changed the AQ-1a-TF conclusion. The `L5` layer is still measurable, especially for `alpha065` at `noise=0.30` and `free` at `noise=0.45`, but exact transformation decoding did not generalize across seed and noise.

The seed split suggests AQ-1a-TF was recovering a fragile endpoint association or transform label, not a stable semantic operator. Seed `11` showed perfect or near-perfect transform exact match, but seed `29` dropped to `0.5000` at best and `0.0000` in several controller/noise cells.

The most useful signal is diagnostic rather than promotional:

- `L5` is not dead.
- direct transform exact match is unstable.
- `alpha065` can strongly preserve `L5` at moderate noise but collapses at high noise on seed `29`.
- `free` is more stable at high noise for seed `29`, but still has weak exact transform recovery.

## Decision

Do not broaden AQ-1a-TF into a larger sequence-pair matrix.

Next work should move to `AQ-1b`: operator-conditioned folded recovery.

Required AQ-1b shift:

- payload: `source motif + operator code`
- withhold direct target features from the transported payload
- derive `target_hat = operator_hat(source_hat)`
- measure `derived_target_accuracy` instead of direct target recovery
- add target leakage, operator swap, and source swap controls

Before the next TPU launch, run or implement compact diagnostics that distinguish:

- `MI(L5; source)`
- `MI(L5; target)`
- `MI(L5; transform_class)`
- transform confusion by seed, noise, and controller

If `MI(L5; target)` exceeds `MI(L5; transform_class)`, the current L5 channel is carrying target identity rather than an operator.
