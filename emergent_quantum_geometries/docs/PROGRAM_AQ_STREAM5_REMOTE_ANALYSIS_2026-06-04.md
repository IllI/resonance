# Program AQ-STREAM-5 Remote Analysis - 2026-06-04

## Purpose

`AQ-STREAM-5` tested the next operator correction after `AQ-STREAM-4`: do not learn a one-sample rank-1 map, and do not treat the transition operator as a separate free artifact. Instead, learn a reusable latent generator in the D-LinOSS state coordinates:

```text
z_t = B^H (x_t - mu)
z_hat_{t+1} = exp(A delta_t) z_t + b
x_hat_{t+1} = mu + B z_hat_{t+1}
```

The run kept `A` in latent space. It did not compute `B^H exp(A delta_t) B`, because that would imply `A` lives in the full input space rather than in the learned compact state basis.

Run envelope:

- mode: `--aq-stream-5`
- frame size: `32 x 32`
- seed: `11`
- noise: `0.00`
- depth: `3`
- latent dimension: `8`
- TPU backend: active JAX TPU runtime
- local artifact: `tpu_previews/aq_stream5_2026-06-04/program_ap_summary.json`
- runtime: `25.5s`

## Arms

| arm | purpose |
| --- | --- |
| `pair_v2_gain_0.25` | teacher/direct pair-resource ceiling |
| `static_delta_operator` | old derived-frame baseline |
| `latent_affine_operator_Bfit` | current best reconstruction-aligned free affine baseline |
| `Adelta_diag_phase` | diagonal `A = i Omega + D` style mode-wise rotation/scale approximation |
| `Adelta_hamiltonian` | near-unitary latent shape operator |
| `Adelta_hamiltonian_plus_luma` | near-unitary shape operator plus positive luminance/scale channel |

## Readout

| arm | mean PSNR | frame 2 PSNR | derived frame 2 PSNR | derived delta PSNR | operator recovery | `B`-space operator cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pair_v2_gain_0.25` | `25.06` | `24.83` | `24.83` | `19.06` | `0.92` | `1.0000` |
| `static_delta_operator` | `22.79` | `23.27` | `23.27` | `17.53` | `0.90` | `0.0000` |
| `latent_affine_operator_Bfit` | `20.26` | `20.79` | `19.46` | `13.06` | `0.92` | `1.0000` |
| `Adelta_diag_phase` | `20.26` | `20.79` | `19.46` | `13.06` | `0.92` | `1.0000` |
| `Adelta_hamiltonian` | `20.26` | `20.79` | `19.48` | `13.07` | `0.92` | `1.0000` |
| `Adelta_hamiltonian_plus_luma` | `20.26` | `20.79` | `19.47` | `13.06` | `0.92` | `1.0000` |

The folded score was not a useful promotion signal in this run:

- folded recovery score stayed near `0.325`
- worst control margin was `0.0000`
- phase scramble and block-permute controls matched the noisy graph cosine

That does not mean the state channel failed. It means this control family is not separating the new latent generator payload from its nulls at the graph-summary level.

## Key Finding

The `A_delta` variants learned the compressed transition almost perfectly, but the compressed basis was too lossy for image reconstruction.

For `Adelta_hamiltonian_plus_luma`:

- `operator_cosine_Bspace = 0.999998`
- `latent_transition_error = 0.0000024`
- `B_isometry_error = 0.006388`
- `operator_unitarity_error = 0.007800`
- `latent_basis_reconstruction_mse = 0.004889`
- `derived_frame_2_psnr = 19.47 dB`

This says the latent generator is not the main failure. The generator maps `z_t` to `z_{t+1}` cleanly inside the learned `B` space. The failure is that `B` at latent dimension `8` does not preserve enough pixel-space information to reconstruct the `32 x 32` tai chi frames at the teacher level.

In plain terms:

```text
good latent flow + lossy basis = weak visual playback
```

## What Worked

The implementation correction was valid:

- `A` stayed in latent space.
- `B/B^H` remained nearly isometric.
- `Adelta_hamiltonian` and `Adelta_hamiltonian_plus_luma` had low unitarity error.
- The `B`-space operator cosine was essentially perfect for the generator arms.

This supports the idea that D-LinOSS can represent a reusable latent transition generator in a compressed state coordinate system.

## What Did Not Work

The generator did not beat the old static-delta or pair-v2 teacher arms on actual derived-frame reconstruction.

The likely reason is not that `A_delta` is mathematically wrong. It is that the latent basis is under-capacity for the image domain:

```text
latent_dim = 8
patch_dim = 16
frame_size = 32 x 32
patch_count = 64 per frame
```

The basis was transition-aware but still too small. It learned the transition in compressed space while discarding enough local pixel detail that Bob's unfolded frame degraded to about `19.5 dB`.

## Experimental Implication

The STREAM branch now has a cleaner separation:

- Pair-v2 is still the best direct recovery/playback teacher.
- Static delta remains a stronger derived-frame baseline than the current low-dimensional latent generator.
- `A_delta` is promising as a reusable operator, but only after the compress/reconstruct basis has enough capacity.

The current result does not invalidate the real-time playback framing. It says that Bob can only play back what survives Alice's state compression. If `B` throws away visual detail, a perfect latent flow still reconstructs a blurry or distorted frame.

## Next Correction

The next run should keep the `A_delta` formulation but raise or restructure `B`:

- run `latent_dim = 12` and `latent_dim = 16`
- keep `Adelta_hamiltonian_plus_luma` and `static_delta_operator`
- keep `pair_v2_gain_0.25` as the teacher ceiling
- promote `derived_frame_2_psnr`, `frame_delta_psnr`, and `pair_teacher_gap`

Recommended compact gate:

```text
AQ-STREAM-6:
  frame_size = 32
  latent_dim = 8, 12, 16
  arms = pair_v2_gain_0.25, static_delta_operator, Adelta_hamiltonian_plus_luma
  success = Adelta derived_frame_2_psnr improves with latent_dim
```

If `latent_dim = 16` closes the gap, then `A_delta` is viable and the problem was basis capacity. If it does not, the next correction should be a patch-local decoder or a separate high-frequency residual stream layered on top of the generator.
