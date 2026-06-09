# Program AQ-STREAM-6 Latent-Capacity Sweep - 2026-06-05

## Purpose

This capstone run tested whether the weak `AQ-STREAM-5` visual recovery was mainly caused by insufficient `B`-space capacity. The hypothesis was:

```text
If Adelta is learning the latent transition correctly but Bob cannot unfold the image,
then increasing latent_dim should improve derived frame-2 PSNR.
```

The run reused the `AQ-STREAM-5` implementation path as a compact latent sweep:

- frame size: `32 x 32`
- seed: `11`
- noise: `0.00`
- depth: `3`
- latent dimensions: `8`, `12`, `16`
- primary generator arm: `Adelta_hamiltonian_plus_luma`
- teacher ceiling: `pair_v2_gain_0.25`
- derived baseline: `static_delta_operator`
- local artifacts: `tpu_previews/aq_stream6_latent_sweep_2026-06-05/`

A final no-fast-path diagnostic was also launched at `latent_dim=16` to test whether the optimized payload-lane decoder was hiding an Adelta signal. That diagnostic was stopped after about `14m45s` because it had not progressed past the initial depth banner and was no longer a quick capstone run.

## Reference Frames

![Reference frame 1](../../tpu_previews/yinyang_reference_v2/taichi_32x32_frame_1.png)

![Reference frame 2](../../tpu_previews/yinyang_reference_v2/taichi_32x32_frame_2.png)

## Numeric Results

| latent dim | arm | mean PSNR | frame 2 PSNR | derived frame 2 PSNR | derived delta PSNR | operator recovery | `B`-space op cosine |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `pair_v2_gain_0.25` | `25.06` | `24.83` | `24.83` | `19.06` | `0.92` | `1.0000` |
| `8` | `static_delta_operator` | `22.79` | `23.27` | `23.27` | `17.53` | `0.90` | `0.0000` |
| `8` | `Adelta_hamiltonian_plus_luma` | `20.25` | `20.79` | `19.47` | `13.06` | `0.92` | `1.0000` |
| `12` | `pair_v2_gain_0.25` | `25.06` | `24.83` | `24.83` | `19.06` | `0.92` | `1.0000` |
| `12` | `static_delta_operator` | `22.79` | `23.27` | `23.27` | `17.53` | `0.90` | `0.0000` |
| `12` | `Adelta_hamiltonian_plus_luma` | `20.59` | `21.33` | `20.86` | `14.37` | `0.90` | `1.0000` |
| `16` | `pair_v2_gain_0.25` | `25.06` | `24.83` | `24.83` | `19.06` | `0.92` | `1.0000` |
| `16` | `static_delta_operator` | `22.79` | `23.27` | `23.27` | `17.53` | `0.90` | `0.0000` |
| `16` | `Adelta_hamiltonian_plus_luma` | `20.72` | `20.87` | `20.87` | `14.94` | `0.87` | `1.0000` |

Detailed Adelta diagnostics:

| latent dim | basis MSE | `B` isometry error | unitarity error | rollout consistency | latent transition error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `0.004888953` | `0.006388` | `0.007800` | `0.385` | `0.0000024` |
| `12` | `0.001108798` | `0.008335` | `0.013529` | `0.480` | `0.0000020` |
| `16` | `0.000002694` | `0.010294` | `0.013492` | `0.550` | `0.0000017` |

## Visual Readout

The compare PNGs show target, recovered, and error panels. The selected local PNGs were fetched from the TPU output directory after the completed fast-path `latent_dim=16` run.

### Pair-v2 Teacher

![Pair-v2 frame 1](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_pair_v2_gain_0.25_taichi_32x32_frame_1_compare.png)

![Pair-v2 frame 2](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_pair_v2_gain_0.25_taichi_32x32_frame_2_compare.png)

### Static Delta Baseline

![Static delta frame 1](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_static_delta_operator_taichi_32x32_frame_1_compare.png)

![Static delta frame 2](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_static_delta_operator_taichi_32x32_frame_2_compare.png)

### Adelta Hamiltonian Plus Luma

![Adelta frame 1](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_Adelta_hamiltonian_plus_luma_taichi_32x32_frame_1_compare.png)

![Adelta frame 2](../../tpu_previews/aq_stream6_latent_sweep_2026-06-05/latent_dim_16/selected_pngs/aq_frame_depth3_noise0.0_seed11_Adelta_hamiltonian_plus_luma_taichi_32x32_frame_2_compare.png)

## Interpretation

The latent-capacity hypothesis partially held. Increasing `latent_dim` from `8` to `12` lifted `Adelta_hamiltonian_plus_luma` derived frame-2 PSNR from `19.47 dB` to `20.86 dB`, and derived delta PSNR from `13.06 dB` to `14.37 dB`.

The improvement then saturated. Increasing `latent_dim` from `12` to `16` did not materially improve derived frame-2 PSNR, even though basis reconstruction MSE collapsed to nearly zero. That means the failure is no longer simply "the basis is too small."

The more precise diagnosis is:

```text
Adelta learns a clean latent flow,
but the recovered operator payload does not preserve enough actionable pixel detail
to beat static delta or pair-v2 playback.
```

This is visible in the images. `Adelta_hamiltonian_plus_luma` preserves some coarse structure, but its recovered panels are noisy and patch-local. Pair-v2 remains the best playback channel because it carries more direct pair-resource information, and static delta remains a stronger derived-frame baseline than the current generator.

## What This Means For The Streaming Hypothesis

The streaming hypothesis still has a viable shape:

```text
Alice streams local pixel-state packets.
Bob receives mode-labeled paired state slots.
If Bob has a learned transition law, he should derive subsequent frames from recovered state plus operator dynamics.
```

But this sequence shows that the current generator form is not yet the playback operator. It is identifiable in latent space, but it is not an adequate reconstruction operator after D-LinOSS recovery.

The hidden working mechanism remains pair-v2 style paired state transport. The less successful mechanism is trying to compress the frame transition into a low-dimensional global latent generator and then unfold it directly as pixels.

## Conclusion

This capstone closes the current Adelta sequence:

- `A_delta` is mathematically coherent.
- Increasing `B` capacity helps at first.
- The improvement saturates before reaching static delta.
- The no-fast-path diagnostic is too slow at this scale to use as a quick validation loop.
- The next useful move is not more latent dimension.

The next branch should test a patch-local or residual-assisted operator decoder:

```text
frame_1 state
+ local patch transition code
+ pair-v2 residual teacher
-> derived frame_2
```

That keeps the reusable-operator goal, but stops forcing all visual detail through one compact global latent generator.
