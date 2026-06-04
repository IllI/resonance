# Program AQ-FFT-3 Remote Analysis - 2026-06-02

## Run Summary

AQ-FFT-3 completed successfully on TPU in `europe-west4-a` on `program-ap-eu-node-v1`.

Launch mode:

```text
--aq-fft-3
```

Runtime envelope:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `free`, `tunnel_eigen_residual_alpha065`

TRC-safe handling:

- result was read from TPU stdout only
- no result artifact, checkpoint, array, directory, or summary payload was downloaded

## Observed Result

| controller | motif | ifft PSNR | oracle PSNR | coeff mag cos | coeff real cos | coeff imag cos | phase cos | freq overlap | top1 freq | topK recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `1.0000` | `20.6478 dB` | `25.0480 dB` | `0.8709` | `0.8713` | `0.5086` | `0.4980` | `0.7266` | `1.0000` | `0.8738` |
| `alpha065` | `1.0000` | `18.2753 dB` | `25.0480 dB` | `0.8640` | `0.8639` | `0.5092` | `0.4913` | `0.7578` | `1.0000` | `0.8741` |

Layer/header readout stayed strong:

| controller | sanity | L1 | L2 | L3 | L4 | L5 | leak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `0.9965` | `0.8750` | `0.8741` | `0.9923` | `0.8714` | `0.9872` | `0.0178` |
| `alpha065` | `0.9923` | `0.8750` | `0.8726` | `0.9885` | `0.8643` | `0.9811` | `0.0240` |

## What Worked

- AQ-FFT-3 achieved a real reconstruction lift over AQ-FFT-1.
  - AQ-FFT-1 `free`: `~16.9995 dB`
  - AQ-FFT-3 `free`: `20.6478 dB`
- motif/header recovery stayed `1.0000`
- oracle PSNR stayed fixed at `25.0480 dB`, so the gain came from better realized reconstruction rather than a different codec ceiling
- real-channel recovery stayed in the same strong range as before

This is the first clear sign that the transport pipeline can carry a more reconstructable version of the coefficient body when the complex coefficient is kept joint at encoding time.

## What Did Not Work

- imaginary coefficient recovery did not improve; it fell to about `0.509`
- phase recovery also fell sharply to about `0.49`
- `alpha065` again underperformed `free` at the noise-free ceiling

So AQ-FFT-3 improved image-domain reconstruction, but it did not yet preserve the complex field in a phase-faithful way.

## Mathematical Read

The key apparent contradiction is:

```text
PSNR improved a lot
but
imaginary-channel cosine and phase cosine got worse
```

That is not actually inconsistent.

The inverse FFT reconstruction error is energy-weighted:

```math
\mathrm{MSE}(p,\hat{p})
\propto
\sum_k |F_k - \hat{F}_k|^2
```

and PSNR depends on that global energy error, not on per-channel cosine symmetry.

In these small `4 x 4` patches, most image energy lives in:

- the DC component
- the fixed low-frequency core
- the larger real-dominant coefficient directions

If AQ-FFT-3 improves those dominant components, PSNR can rise substantially even while weaker quadrature structure degrades.

More concretely:

```math
F_k = a_k + i b_k
```

AQ-FFT-3 appears to be improving the recovery of the higher-energy part of the coefficient field, especially the part most coupled to luminance and low-frequency structure, while still losing coherence in the lower-energy quadrature directions \(b_k\).

That is why the following can happen simultaneously:

- `coeff_real_cos` remains high
- `top1_freq_acc` stays perfect
- `topK_energy_recall` stays high
- PSNR jumps
- `coeff_imag_cos` and `phase_cos` remain weak

The direct-complex encoding removed the old multiplicative reconstruction error

```math
\operatorname{Im}(\hat{F}_k)=|\hat{F}_k|\sin(\hat{\phi}_k)
```

but the current signature/decoder still does not preserve rotational fidelity in the complex plane. In effect, AQ-FFT-3 improved the transport of coefficient energy without yet preserving the full complex orientation of each retained mode.

## Methodological Conclusion

AQ-FFT-3 is a meaningful positive result.

It shows that:

- the old magnitude/phase split really was hurting reconstruction
- better reconstruction of the unfolded field is possible without changing codec capacity
- the folded semantic/header channel and the reconstructive coefficient channel can coexist cleanly

At the same time, it shows the next bottleneck more clearly:

- the transport now carries enough information to reconstruct much more of the patch
- but it still does not preserve the complex quadrature structure well enough for faithful imaginary/phase recovery

That moves the methodology closer to reconstruction of unfolded semantic space in a concrete way:

```text
folded semantic/header state
  + transportable coefficient body
  -> unfold decoder
  -> reconstructed image-domain field
```

AQ-FFT-1 proved the codec ceiling.
AQ-FFT-3 now shows a stronger realized reconstruction path.

So the program is no longer just demonstrating semantic folded-state survival. It is beginning to demonstrate partial reconstruction of the unfolded field from the transported folded representation.

## Next Technical Question

The next question is no longer whether direct complex encoding helps. It does.

The next question is:

> why does the transport preserve coefficient energy better than complex orientation?

The likely suspects are:

- D-LinOSS dynamics or damping acting anisotropically on the complex payload
- signature extraction losing too much phase/orientation information
- ridge decoding favoring energy-dominant directions over quadrature-faithful reconstruction

The best follow-up diagnostics are:

- QMI on real/imaginary or magnitude/phase subchannels before and after transport
- support-weighted complex-angle error by coefficient slot
- direct comparison of final oscillator slots versus filament-signature-decoded coefficients
