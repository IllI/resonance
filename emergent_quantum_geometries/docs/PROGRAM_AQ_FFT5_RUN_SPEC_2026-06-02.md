# Program AQ-FFT-5 Run Spec: Gauge-Aligned Complex Decoding

Date: 2026-06-02

## Purpose

AQ-FFT-4 showed that `complex_pair_norm` is the current reference codec:

- realized PSNR rose to `23.1818 dB`
- oracle stayed near `25.0480 dB`
- motif/header recovery stayed `1.0000`
- imaginary and phase cosine remained near `0.51`

AQ-FFT-5 tests whether that weak imaginary/phase score is a complex gauge mismatch rather than true quadrature information loss.

## Gauge Diagnostic

For each patch, compute the best oracle global phase rotation:

```math
\theta^*=\arg\min_\theta \|F_{\mathrm{true}}-e^{i\theta}F_{\mathrm{pred}}\|_2
```

Equivalently:

```math
\theta^*=\arg\left(\sum_k F_{\mathrm{true},k}\overline{F_{\mathrm{pred},k}}\right)
```

Then recompute:

- `fft_global_phase_aligned_imag_cos`
- `fft_global_phase_aligned_phase_cos`
- `fft_per_slot_phase_aligned_coeff_cos`
- `fft_complex_gauge_error`
- `fft_psnr_before_alignment`
- `fft_psnr_after_alignment`

If aligned imaginary and phase cosine jump, the recovered field contains the complex structure but Bob needs gauge fixing.

If aligned metrics stay low, the quadrature information is genuinely degraded.

## Decoder Variant

AQ-FFT-5 adds:

```text
complex_pair_norm_gaugefix
```

Pipeline:

1. recover sparse complex coefficients with the `complex_pair_norm` signature
2. estimate a canonical phase from the DC coefficient, with strongest-coefficient fallback
3. rotate the coefficient field so the anchor is real-positive
4. reconstruct with inverse FFT

This is an anchor-based decoder correction. It does not use the true target coefficients.

## Run Shape

- mode: `--aq-fft-5`
- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `complex_pair_norm`, `complex_pair_norm_gaugefix`

Excluded:

- larger `K`
- noise
- `alpha065`
- video

## Success Criteria

Useful result:

- PSNR at least matches `23.1818 dB`
- phase-aligned imaginary cosine improves materially
- gauge error is small and stable
- motif recovery remains `1.0000`

Strong result:

- PSNR reaches or exceeds `24 dB`
- aligned imaginary cosine exceeds `0.75`
- aligned phase cosine exceeds `0.75`

## Decision

If gauge fixing works, proceed to `AQ-FFT-6` with `residual_K=4`.

If gauge fixing fails, avoid increasing payload and move to a complex-equivariant decoder or a belief-propagation/charge decoder that treats the coefficient field as a coupled object.
