# Program AQ-FFT-4 Run Spec: Complex Orientation Preservation

Date: 2026-06-02

## Purpose

AQ-FFT-3 showed that direct complex oscillator injection improves realized image reconstruction, but the imaginary and phase channels remain weak. AQ-FFT-4 keeps the same codec capacity and tests whether the remaining gap is caused by loss of complex orientation rather than missing payload capacity.

Core hypothesis:

```math
F_k = a_k + i b_k
```

should be transported as a coupled quadrature pair. If the D-LinOSS signature or decoder is not equivariant to rotations in the complex plane, it can preserve coefficient energy while flattening the imaginary component and phase orientation.

## Run Shape

- mode: `--aq-fft-4`
- noise: `0.00`
- seed: `11`
- depth: `3`
- codec: fixed core `K=6` plus `residual_K=2`
- controllers: `free`, `complex_pair_norm`, `complex_phase_locked`
- excluded: `alpha065`, noise sweep, larger `K`

## Controller Variants

### free

Same direct complex path as AQ-FFT-3. This anchors the expected clean result:

- PSNR near `20.65 dB`
- `coeff_imag_cos` near `0.51`
- `phase_cos` near `0.50`

### complex_pair_norm

Each sparse FFT coefficient is treated as a coupled 2D direction:

```math
z_k = [\operatorname{Re}(F_k), \operatorname{Im}(F_k)],
\qquad
\hat{z}_k = \frac{z_k}{\|z_k\|+\epsilon}
```

D-LinOSS receives the unit complex direction in the oscillator slot and keeps coefficient magnitude as a coupled side signal in the signature. The intent is to preserve orientation without returning to independently decoded magnitude and phase scalars.

### complex_phase_locked

The decoder enforces a geometric consistency correction:

```math
\|\hat{z}_k\|^2 \approx \hat{m}_k^2
```

It recovers a magnitude estimate and combines it with the decoded complex direction:

```math
\hat{F}_k = \hat{m}_k \frac{\hat{F}_k^{raw}}{|\hat{F}_k^{raw}|+\epsilon}
```

This tests whether the weak imaginary channel is a decoder consistency problem rather than a transport problem.

## New Diagnostics

AQ-FFT-4 adds:

- `fft_support_weighted_angle_error`
- `fft_energy_weighted_complex_mse`
- `fft_quadrature_energy_recall`
- `fft_imaginary_energy_recall`
- `fft_complex_pair_norm_error`
- `fft_phase_error_by_coefficient_slot`

It also prints an error budget:

- `PSNR_oracle`
- `PSNR_using_true_support_pred_coeffs`
- `PSNR_using_pred_support_true_coeffs`
- `PSNR_using_true_real_pred_imag`
- `PSNR_using_pred_real_true_imag`
- `PSNR_final`

## Interpretation

If `true_real_pred_imag` causes the largest PSNR penalty, the imaginary/quadrature channel is the main limiter.

If `pred_support_true_coeffs` is poor, support/index recovery is still limiting.

If `complex_pair_norm` improves phase and imaginary cosine, the issue was orientation preservation in the transported signature.

If `complex_phase_locked` improves PSNR without improving raw phase much, the decoder was violating pair-norm geometry.

## Pass Signal

Minimum useful lift:

- `fft_patch_psnr > 20.65 dB`
- `fft_coeff_imag_cosine > 0.65`
- `fft_phase_recovery > 0.65`
- motif/header recovery remains `1.0000`
- oracle stays near `25.05 dB`

Strong lift:

- PSNR in the `22-23 dB` range
- `coeff_imag_cosine > 0.75`
- `phase_recovery > 0.75`

## Decision

Do not move to video, noise, alpha tuning, or larger residual payload until this gate tells us whether the complex-field bottleneck is input encoding, decoder geometry, or D-LinOSS signature extraction.
