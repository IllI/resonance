# Program AQ-0 Remote Analysis - 2026-05-29

## TRC-safe data handling

This record was created from the tiny final stdout table already inspected on the TPU VM. No result artifact, summary JSON, array dump, checkpoint, or recursive directory was downloaded.

Run context:

- project: `time-emission`
- node: `program-ap-eu-node-v2`
- zone: `europe-west4-a`
- mode: `--aq-0`
- backend: JAX TPU
- devices visible: `4`
- experiment label: `semantic_subspace_transport_aq0`

## AQ-0 proxy configuration

This was the fast noise-free semantic ceiling probe, optimized for quick turnover rather than durability:

- noise: `0.00`
- depth: `3`
- seeds: `11`, `29`
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- hard controls: disabled
- recovery train states: `4`
- recovery test states: `2`
- D-LinOSS steps: capped at `24`

This run should be treated as an AQ-0 proxy ceiling check, not the full AQ layer-wise proof. It did not yet report explicit `L1`-`L4` semantic layer metrics or the offline untransported ridge sanity check.

Important implementation caveat:

- this proxy still used the legacy compact motif generator, not the locked `8`-motif AQ alphabet
- AQ-0b must switch to the explicit motif set before it can count as the semantic payload ceiling test

## Final table

| seed | controller | margin | score | noisy cosine | block lift | topology lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `11` | `free` | `+0.0341` | `0.3456` | `0.9388` | `+0.0506` | `+0.0336` |
| `11` | `tunnel_eigen_residual_alpha065` | `+0.0860` | `0.3428` | `0.7335` | `+0.0803` | `+0.0890` |
| `29` | `free` | `-0.0064` | `0.3103` | `0.8020` | `-0.0064` | `-0.0045` |
| `29` | `tunnel_eigen_residual_alpha065` | `+0.0045` | `0.3065` | `0.8914` | `+0.0187` | `+0.0063` |

## Aggregate read

| controller | positive cells | mean margin | min margin | mean noisy cosine | min noisy cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `free` | `1/2` | `+0.0139` | `-0.0064` | `0.8704` | `0.8020` |
| `tunnel_eigen_residual_alpha065` | `2/2` | `+0.0453` | `+0.0045` | `0.8125` | `0.7335` |

Controller lift over `free`:

- seed `11`: `+0.0519` margin lift
- seed `29`: `+0.0109` margin lift

## Interpretation

The useful signal is that `tunnel_eigen_residual_alpha065` improved margin over `free` on both seeds and made the weak seed `29` positive. That means the controller path is still doing useful work in the noise-free AQ proxy.

The caution is the seed `11` noisy cosine: `alpha065` dropped to `0.7335` while `free` stayed at `0.9388`. At `noise=0.00`, that is a ceiling warning. It may mean the AP-style aggregate noisy cosine is not aligned with AQ's semantic layer objective, or it may mean the current proxy encoder/decoder path is not clean enough yet.

The most likely scientific hypothesis is controller-induced semantic reshaping: `alpha065` may preserve coarse folded topology and improve the margin while compressing away fine layer geometry, especially `L3` phase/FFT and `L4` recurrence/correlation content.

Because AQ is supposed to prove recoverable semantic layers rather than merely improve an AP-style margin, this is not enough to launch AQ-1 noise.

## Decision

Do not advance to AQ-1 yet.

Next run should be `AQ-0b`: still noise-free, still compact, but with explicit semantic-layer reporting and the offline sanity check wired into the record.

Required AQ-0b additions:

- use the explicit `8`-motif AQ alphabet
- add `tunnel_eigen_residual_alpha055` as a minimal projection-strength ablation
- report `L1` occupation recovery
- report `L2` transition graph recovery
- report `L3` FFT/phase recovery
- report `L4` recurrence/correlation recovery
- run ridge decoder on the untransported encoded payload before interpreting transport
- keep controllers to `free`, `tunnel_eigen_residual_alpha055`, and `tunnel_eigen_residual_alpha065`
- keep hard controls disabled
- keep `noise=0.00`

Only after AQ-0b shows a clean layer-wise ceiling should AQ-1 add `noise=0.30` and `0.45`.
