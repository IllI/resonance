# Program AQ-STREAM-2 Remote Analysis

## Scope

`AQ-STREAM-2` was the first combined gain-sweep plus transition-operator run on the stream branch. It used the raw patch-state transport path, the `32x32` yin-yang reference frames, and two families of arms:

- `pair_v2_gain_*`: direct frame recovery with auxiliary pair residual channels
- `delta_operator_gain_*`: frame-1 anchor plus decoded transition/delta operator

The TPU VM was deleted after the run. The full `program_ap_summary.json` did not finish copying locally before deletion, but the launcher log captured the readout and the key PNG artifacts were copied into:

```text
tpu_previews/aq_stream2_2026-06-04/program_ap_results/
```

This note is therefore based on:

- the captured remote log readout
- the preserved local preview PNGs

## Experimental Goal

The run was designed to answer two questions at once:

1. Does pair-resource gain matter in the current stream architecture?
2. Can Bob recover frame 2 through a decoded transition operator rather than only by direct frame reconstruction?

The clean success condition was not "perfect image recovery." It was the isolation of these two effects under the same launch envelope.

## Captured Readout

The launcher log captured the following `AQ-STREAM-2` readout at `noise=0.00`, `depth=3`, `seed=11`, `frame_size=32`:

| arm | gain | frame 1 PSNR | frame 2 PSNR | frame delta PSNR | derived PSNR | derived frame 2 PSNR | derived frame delta PSNR | order | patch | leakage | pair fidelity | delta operator recovery | anchor recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat_ordered_roster` | `0.000` | `10.1790` | `6.5712` | `2.4957` | `8.0107` | `6.5722` | `2.4953` | `0.1562` | `0.1953` | `0.2891` | `0.8506` | `0.1236` | `0.8049` |
| `separable_mode_stream` | `0.000` | `13.8668` | `11.9720` | `7.0101` | `12.8165` | `11.9715` | `7.0109` | `1.0000` | `1.0000` | `0.0000` | `0.8429` | `0.4854` | `0.8834` |
| `pair_v2_gain_0.125` | `0.125` | `25.3039` | `24.8344` | `19.0573` | `25.0663` | `24.8461` | `19.0612` | `1.0000` | `1.0000` | `0.0000` | `0.9794` | `0.4969` | `0.8867` |
| `pair_v2_gain_0.25` | `0.250` | `25.3039` | `24.8344` | `19.0573` | `25.0666` | `24.8464` | `19.0615` | `1.0000` | `1.0000` | `0.0000` | `0.9794` | `0.4969` | `0.8867` |
| `pair_v2_gain_0.50` | `0.500` | `25.3036` | `24.8342` | `19.0570` | `25.0663` | `24.8463` | `19.0612` | `1.0000` | `1.0000` | `0.0000` | `0.9794` | `0.4969` | `0.8867` |
| `delta_operator_gain_0.125` | `0.125` | `22.3658` | `23.2689` | `17.5343` | `22.7987` | `23.2868` | `17.5442` | `1.0000` | `1.0000` | `0.0000` | `0.9732` | `0.4972` | `0.8877` |
| `delta_operator_gain_0.25` | `0.250` | `22.3658` | `23.2689` | `17.5343` | `22.7987` | `23.2868` | `17.5442` | `1.0000` | `1.0000` | `0.0000` | `0.9732` | `0.4972` | `0.8877` |
| `delta_operator_gain_0.50` | `0.500` | `22.3659` | `23.2690` | `17.5344` | `22.7984` | `23.2862` | `17.5435` | `1.0000` | `1.0000` | `0.0000` | `0.9731` | `0.4972` | `0.8877` |

## Main Findings

### 1. The run achieved the isolation goal

`AQ-STREAM-2` successfully separated:

- direct pair-resource assisted frame recovery
- transition-operator assisted frame derivation

That is the main scientific success of the run.

### 2. Gain was not the active bottleneck

Within each family, `0.125`, `0.25`, and `0.50` were effectively tied. The differences were small enough to treat them as numerically indistinguishable at this stage.

That means the next bottleneck is not "find the magic gain."

### 3. Pair-v2 direct recovery still wins

The best `pair_v2_gain_*` arms reached about:

- `25.30 dB` on frame 1
- `24.83 dB` on frame 2
- `24.85 dB` on derived frame 2

The `delta_operator_gain_*` arms were clearly real and coherent, but lower:

- about `22.37 dB` on frame 1
- about `23.27 dB` on frame 2
- about `23.29 dB` on derived frame 2

So the operator path worked, but it did not surpass direct pair-v2 reconstruction.

### 4. Ordering stayed solved at 32x32

For every non-flat stream arm:

- `temporal_order_accuracy = 1.0000`
- `patch_position_accuracy = 1.0000`
- `cross_frame_leakage = 0.0000`

That matters. The 32x32 capacity increase reduced reconstruction fidelity relative to earlier `16x16` runs, but it did **not** break the stream ordering mechanism.

## Interpretation

This run gives a cleaner story than the raw PSNR numbers alone:

- the mode-labeled stream architecture remains robust at `32x32`
- auxiliary pair resources are still the strongest reconstruction mechanism
- the decoded delta/operator channel is real enough to derive frame 2, but not yet strong enough to overtake direct pair-aided recovery
- gain no longer looks like the lever to pull next

In other words, the branch has moved past "can the stream keep order straight?" and into "how do we improve the quality of the recovered instruction body at higher capacity?"

## Visual Artifacts

Preserved local artifact root:

```text
tpu_previews/aq_stream2_2026-06-04/program_ap_results/
```

### Pair-v2 gain 0.25, 32x32

Frame 1 compare:

![pair_v2_gain_0.25 frame 1 compare](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_frame_depth3_noise0.0_seed11_pair_v2_gain_0.25_taichi_32x32_frame_1_compare.png)

This panel shows the preserved target, the reconstructed frame, and the residual error for the strongest direct-recovery family.

### Delta-operator gain 0.25, 32x32

Frame 1 compare:

![delta_operator_gain_0.25 frame 1 compare](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_frame_depth3_noise0.0_seed11_delta_operator_gain_0.25_taichi_32x32_frame_1_compare.png)

Frame 2 compare:

![delta_operator_gain_0.25 frame 2 compare](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_frame_depth3_noise0.0_seed11_delta_operator_gain_0.25_taichi_32x32_frame_2_compare.png)

These are the most important operator-side images preserved locally. They show that the delta-operator path remained visually coherent, but softer than the pair-v2 direct-recovery path.

### Stream previews

Flat ordered roster preview:

![flat roster preview](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_preview_depth3_noise0.0_seed11_flat_ordered_roster.png)

Separable mode stream preview:

![separable mode stream preview](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_preview_depth3_noise0.0_seed11_separable_mode_stream.png)

Pair-v2 gain 0.25 preview:

![pair_v2_gain_0.25 preview](../../tpu_previews/aq_stream2_2026-06-04/program_ap_results/aq_preview_depth3_noise0.0_seed11_pair_v2_gain_0.25.png)

## Bottom Line

`AQ-STREAM-2` passed the architectural question it was meant to answer.

It showed that:

- the stream branch still holds ordered recovery at `32x32`
- pair-v2 auxiliary residuals remain the best reconstruction path
- a decoded transition operator can derive frame 2 coherently, but not yet at the same fidelity as direct pair-v2 recovery
- gain is not the active limiter in the current design

That gives the next run a much sharper target: improve the operator body or reconstruction method, not the gain.
