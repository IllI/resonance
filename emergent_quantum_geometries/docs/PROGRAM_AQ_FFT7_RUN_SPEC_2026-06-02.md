# Program AQ-FFT-7 Run Spec: Residual Capacity Scaling

Date: 2026-06-02

## Purpose

AQ-FFT-6 showed that raw phase and imaginary cosine are not the next primary gate:

- Hermitian error was already tiny
- Hermitian projection did not move PSNR
- energy-weighted phase cosine was `0.9992`

AQ-FFT-7 moves from phase debugging to capacity scaling. The question is whether `complex_pair_norm` reconstruction improves when the sparse residual body increases from `residual_K=2` to `residual_K=4`.

## Run Shape

- mode: `--aq-fft-7`
- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=4`
- controller: `complex_pair_norm`

Excluded:

- `alpha065`
- noise
- video
- additional controller variants

## Baseline

The repeated residual_K=2 reference is:

- realized PSNR near `23.18 dB`
- oracle PSNR near `25.05 dB`
- motif/header recovery `1.0000`
- energy-weighted phase cosine near `0.9992`

## Primary Metrics

Promoted:

- `fft_patch_psnr`
- `fft_oracle_psnr`
- `fft_phase_cos_energy_weighted`
- `fft_energy_weighted_complex_mse`
- `fft_topk_energy_recall`
- `fft_patch_motif_recovery`

Debug-only:

- raw `fft_coeff_imag_cosine`
- raw `fft_phase_recovery`

## Success Criteria

Useful:

- oracle PSNR rises
- realized PSNR rises above `23.18 dB`
- motif/header recovery remains `1.0000`
- energy-weighted phase remains high

Strong:

- realized PSNR reaches or exceeds `24 dB`

## Decision

If residual_K=4 improves realized reconstruction, AQ-FFT-8 should add noise at `0.30` using the higher-capacity codec.

If residual_K=4 raises oracle but not realized PSNR, the next bottleneck is decoder capacity rather than codec capacity.
