# Program AQ-FFT-3 Run Spec - Direct Complex Oscillator Codec

## Purpose

AQ-FFT-3 tests whether the persistent imaginary-coefficient failure is caused by splitting complex FFT coefficients into separate magnitude and phase channels before transport.

The repeated AQ-FFT/AS pattern is:

| metric | typical value |
| --- | ---: |
| `coeff_real_cos` | `~0.861` |
| `coeff_imag_cos` | `~0.628` |
| `phase_cos` | `~0.893` |
| realized PSNR | `~17.0 dB` |
| oracle PSNR | `~25.05 dB` |

The working diagnosis is:

```math
\operatorname{Im}(F_k) = |F_k|\sin(\phi_k)
```

When magnitude and phase errors are correlated, reconstructing the imaginary component as

```math
|\hat{F}_k|\sin(\hat{\phi}_k)
```

can be worse than either magnitude or phase recovery alone. The old AQ-FFT representation transported magnitude and phase as separate real channels, then multiplied them back together at decode time.

AQ-FFT-3 removes that cross-term by injecting the sparse complex coefficient directly:

```math
x_0[k] = F_k
```

The recovered coefficient is read as:

```math
\hat{F}_k = \hat{x}[k]
```

## Implementation

Mode:

```bash
--aq-fft-3
```

Runtime:

- noise: `0.00`
- seed: `11`
- depth: `3`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- no noise sweep
- no payload expansion

Payload changes:

- keep the AQ-FFT-1 fixed-core plus sparse-residual support
- replace sparse coefficient features from `[|F|, cos(phi), sin(phi)]` to `[Re(F), Im(F)]`
- inject `F_k` directly into the first complex D-LinOSS oscillator slots
- expose recovered oscillator slots in the signature as `[Re(x_T), Im(x_T)]`
- keep the filament signature attached for semantic/header context

Expected startup banner:

```text
AQ-FFT-3 ACTIVE:
  codec = AQ-FFT-1 fixed core + residual K-space
  coefficient_encoding = direct complex oscillator slots
  direct_complex_x0 = true
  decoder_target = sparse FFT Re/Im coefficients
  baseline = AQ-FFT-1 flat path at ~17.00 dB
```

If that banner does not appear, the run is not testing AQ-FFT-3.

## Metrics

Primary:

- `fft_patch_psnr`
- `fft_oracle_psnr`
- `fft_coeff_real_cosine`
- `fft_coeff_imag_cosine`
- `fft_coeff_mag_cosine`
- `fft_phase_recovery`
- `fft_frequency_index_recovery`
- `fft_topk_energy_recall`
- `fft_top1_frequency_accuracy`
- motif/header recovery

Key comparison:

```text
AQ-FFT-3 coeff_imag_cosine > AQ-FFT-1 coeff_imag_cosine (~0.628)
AQ-FFT-3 ifft_psnr > AQ-FFT-1 ifft_psnr (~17.00 dB)
```

## Interpretation

If `coeff_imag_cosine` rises toward the real-channel range:

- the old magnitude/phase split was damaging the joint complex structure
- continue with direct complex coefficient transport
- then test noise and controller robustness

If `coeff_imag_cosine` stays near `0.63`:

- the loss is likely in the D-LinOSS dynamics or signature extraction, not only in representation
- inspect damping symmetry and whether imaginary components decay faster than real components

If PSNR improves but coefficient metrics do not:

- the direct complex slots may be improving the inverse FFT enough without globally improving cosine
- inspect per-frequency error and support-weighted energy recovery

## QMI Diagnostic Direction

The Sanz Larrarte et al. MPS/QMI idea is best used here as a diagnostic, not as another transport representation.

For future analysis, compute:

```math
\operatorname{QMI}(i,j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij})
```

where the two sites represent coefficient subchannels such as magnitude and phase or real and imaginary components.

Useful tests:

- QMI between magnitude and phase before transport
- QMI between magnitude and phase after transport
- QMI between real and imaginary coefficient slots after direct complex injection

Decision:

- if QMI drops during transport, repair transport dynamics or signature extraction
- if QMI is preserved but decoding fails, repair the decoder
- if QMI is low before transport, repair the encoder

This keeps the paper's contribution in the role it fits best: detecting lost higher-order dependency structure.

## AQ-FFT-3 Result

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT3_REMOTE_ANALYSIS_2026-06-02.md`

Observed result:

| controller | ifft PSNR | oracle PSNR | coeff real cos | coeff imag cos | phase cos | freq overlap | top1 freq | topK recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `20.6478 dB` | `25.0480 dB` | `0.8713` | `0.5086` | `0.4980` | `0.7266` | `1.0000` | `0.8738` |
| `alpha065` | `18.2753 dB` | `25.0480 dB` | `0.8639` | `0.5092` | `0.4913` | `0.7578` | `1.0000` | `0.8741` |

Motif/header recovery remained `1.0000`.

## Interpretation After the Run

AQ-FFT-3 produced a genuine reconstruction gain:

- AQ-FFT-1 `free`: `~16.9995 dB`
- AQ-FFT-3 `free`: `20.6478 dB`

So direct complex injection helped the unfold/reconstruction path materially.

But the result also showed that direct complex injection did **not** fix the full complex field:

- `coeff_imag_cosine` fell to about `0.509`
- `phase_cos` fell to about `0.49`

The mathematical reason the PSNR can improve while the imaginary channel remains weak is that image-domain reconstruction error is dominated by high-energy low-frequency content:

```math
\mathrm{MSE}(p,\hat{p}) \propto \sum_k |F_k-\hat{F}_k|^2
```

If the transport improves the dominant energy-bearing coefficients, especially the DC and low-frequency real-dominant directions, PSNR can rise even when lower-energy quadrature structure is not preserved well.

That means AQ-FFT-3 improved the transport of coefficient energy more than it improved the transport of complex orientation.

Methodological read:

- the old magnitude/phase split really was part of the reconstruction bottleneck
- the folded channel now carries enough information to reconstruct more of the unfolded field
- the next bottleneck is no longer codec capacity, and not mainly semantic/header stability
- the next bottleneck is preserving the full complex orientation of retained modes during transport and/or signature extraction
