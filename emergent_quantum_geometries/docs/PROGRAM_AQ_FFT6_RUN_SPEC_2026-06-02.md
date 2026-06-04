# Program AQ-FFT-6 Run Spec: Hermitian-Constrained Complex Decoder

Date: 2026-06-02

## Purpose

AQ-FFT-5 showed that the weak imaginary and phase cosine are not explained by a global complex gauge rotation. The remaining failure is local quadrature under-recovery.

AQ-FFT-6 tests the most actionable local structure: real-image FFT Hermitian symmetry.

For a real-valued patch:

```math
F[-k] = \overline{F[k]}
```

If the decoder predicts each complex slot independently, it can reconstruct energy-dominant structure while violating local conjugate consistency. AQ-FFT-6 projects the recovered spectrum onto the Hermitian subspace before inverse FFT.

## Run Shape

- mode: `--aq-fft-6`
- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `complex_pair_norm`, `complex_pair_norm_hermitian`

Excluded:

- larger `K`
- noise
- `alpha065`
- video

## Decoder Variant

`complex_pair_norm` is the AQ-FFT-4 reference codec.

`complex_pair_norm_hermitian` applies:

```math
\hat{F}_{sym}[k] = \frac{1}{2}\left(\hat{F}[k] + \overline{\hat{F}[-k]}\right)
```

Self-conjugate slots are forced real.

For a `4 x 4` FFT, the self-conjugate flattened slots are:

```text
0, 2, 8, 10
```

## Metrics

AQ-FFT-6 adds:

- `fft_hermitian_error_before`
- `fft_hermitian_error_after`
- `fft_self_conjugate_imag_energy`
- `fft_pairwise_conjugate_consistency`
- `fft_phase_cos_energy_weighted`

The central check is:

```text
hermitian_error_before -> hermitian_error_after
```

## Interpretation

If Hermitian error drops and PSNR stays at or above `23.18 dB`, the codec has gained a physically valid reconstruction constraint.

If imaginary or phase cosine improves, the previous weakness was partly a local symmetry violation.

If energy-weighted phase is high while raw phase is low, the model may already preserve the phase that matters for reconstruction, and low-energy quadrature slots should not drive the next design decision.

If Hermitian constraints do not help, the next step should be a complex-equivariant decoder or a BP/charge decoder that enforces local consistency among coefficient nodes.
