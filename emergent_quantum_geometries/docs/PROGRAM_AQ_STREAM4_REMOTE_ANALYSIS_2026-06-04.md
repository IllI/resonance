# Program AQ-STREAM-4 Remote Analysis - 2026-06-04

## Purpose

`AQ-STREAM-4` tested a compact operator-architecture correction for the entangled-pair streaming branch. The immediate question was not whether D-LinOSS could improve raw transport fidelity, which was already high, but whether the transition operator could be made more identifiable when learned in a stable compress/reconstruct pair.

The run used:

- mode: `--aq-stream-4`
- frame size: `32 x 32`
- seed: `11`
- noise: `0.00`
- depth: `3`
- latent dimension: `8`
- TPU backend: active JAX TPU runtime
- local artifact: `tpu_previews/aq_stream4_2026-06-04/program_ap_summary.json`

Only small text artifacts were fetched from the remote TPU node: the JSON summary and run log. No preview PNGs were emitted by this mode.

## Readout

| arm | worst control margin | folded recovery score | noisy graph cosine | block lift | topology lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pair_v2_teacher` | `0.3379` | `0.5287` | `0.9854` | `0.4192` | `0.4871` |
| `static_delta_operator` | `0.3456` | `0.5334` | `0.9854` | `0.3764` | `0.4302` |
| `latent_affine_operator_Bfit` | `0.7078` | `0.7507` | `0.9854` | `0.2807` | `0.4339` |
| `low_rank_delta_operator` | `0.3576` | `0.5406` | `0.9854` | `0.4312` | `0.5273` |
| `teacher_distilled_operator` | `0.3110` | `0.5126` | `0.9854` | `0.4684` | `0.5180` |

Layer recovery was effectively invariant across arms:

| layer | cosine |
| --- | ---: |
| `L1_occupation` | `0.9978` |
| `L2_transition` | `0.9854` |
| `L3_fft_phase` | `0.9880` |
| `L4_recurrence` | `0.9925` |

The leakage proxy stayed low at `0.0144`.

## Interpretation

The folded recovery score is still not high enough to call the operator problem solved. The score is a composite gate that includes adversarial control separation, so a `0.7507` best result is useful but not yet a full playback-quality operator claim.

The important positive signal is that `latent_affine_operator_Bfit` doubled the worst-control margin relative to the pair teacher and static delta arms while using the same recovered state channel. That means the run found evidence that the hidden useful structure is not in stronger transport fidelity, but in the coordinate system used to express the transition.

Methodologically, this supports the following compression picture:

```text
z_{t,p} = B^H (x_{t,p} - mu)
z_hat_{t+1,p} = T z_{t,p} + b
x_hat_{t+1,p} = mu + B z_hat_{t+1,p}
```

Here `x_{t,p}` is Alice's patch state at time `t` and patch position `p`, `B` is the learned reconstruction-aligned basis, and `T` is the learned latent transition generator. `AQ-STREAM-4` suggests that learning the operator in the `B/B^H` coordinate system makes the rule more identifiable than trying to promote pair-v2 or low-rank delta packets directly.

The pair-v2 result remains important as a teacher and channel stabilizer. Earlier STREAM runs showed that raw patch states plus mode labels plus auxiliary pair residuals can improve ordered two-frame recovery. `AQ-STREAM-4` adds that the next transition operator should be learned against a reconstruction-aligned basis, not just against pair correlation labels.

## Relation To The Greater Goal

The current simulator goal can be stated carefully:

Once Bob recovers a stable state representation and the stream has learned a reusable transition operator, Alice's local pixel states can be streamed into Bob's corresponding mode-labeled states so Bob reconstructs a real-time playback sequence from the recovered state stream.

In this experiment, "entangled pixels" means simulated pair-resource slots with shared time, patch, polarity, and residual structure. It is not a physical claim of instantaneous image transfer. The empirical question is whether the D-LinOSS transport plus mode-labeled pair resource can preserve enough state geometry that Bob can unfold the stream into the same ordered visual sequence Alice encoded.

The STREAM branch has now separated three effects:

- Ordered mode labels help frame recovery by preventing the decoder from rediscovering time and patch identity from a flat roster.
- Pair-v2 auxiliary residuals help visual recovery when they preserve the raw patch body rather than overwriting quadrature lanes.
- Reconstruction-aligned latent coordinates help operator identifiability more than static delta, low-rank delta, or direct teacher distillation.

## What Did Not Work

`teacher_distilled_operator` did not beat the pair teacher. This argues against simply copying the pair-v2 behavior as the transition rule.

`low_rank_delta_operator` did not produce a meaningful improvement over static delta. This suggests that the problem is not just reducing delta dimensionality.

`latent_affine_operator_Bfit` improved control separation but did not change the base noisy graph cosine, so the transport channel is probably already saturated at this scale. The remaining problem is making the recovered state actionable as a transition generator.

## Next Correction

The next run should promote `latent_affine_operator_Bfit` from a controller arm into the explicit derived-frame path. The metric should move from folded recovery score alone toward reconstruction metrics:

- frame 1 PSNR
- derived frame 2 PSNR
- frame delta PSNR
- operator cosine in `B/B^H` coordinates
- cross-frame leakage
- patch-position accuracy

The key gate is:

```text
latent_affine_operator_Bfit derived_frame_2_psnr > static_delta_operator derived_frame_2_psnr
```

If that passes, the STREAM branch can move from "recover both frames as state packets" toward "recover frame 1 plus a learned transition and derive frame 2 on Bob's side."
