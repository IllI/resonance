# Program AQ-YINYANG-1 Expected Findings

Date: 2026-06-03

## Purpose

`AQ-YINYANG-1` is a one-frame visual transport gate for the tai-chi-tu image experiment. It is meant to isolate the failure mode observed in `AQ-YINYANG-0`, where the recovered two-frame images preserved broad yin/yang polarity but did not reconstruct the crisp binary symbol.

The scientific question is:

```text
Did the simple image fail because D-LinOSS lost the folded semantic state,
or because the sparse k-space codec cannot represent a crisp binary image at this capacity?
```

## Baseline From AQ-YINYANG-0

The two-frame run used:

- `noise=0.00`
- `seed=11`
- `depth=3`
- `controller=free`
- `core_K=6`
- `residual_K=4`
- two alternating polarity frames

Observed result:

```text
motif/source/target recovery = 1.0000
ifft_psnr                  = 15.3806 dB
oracle_psnr                = 16.8198 dB
coeff_mag_cos              = 0.9920
coeff_real_cos             = 0.9930
coeff_imag_cos             = 0.9759
phase_cos                  = 0.8451
topK_energy_recall         = 1.0000
energy_weighted_phase_cos  = 0.9992
```

The important observation is the small realized/oracle gap:

```text
oracle_psnr - ifft_psnr ~= 1.44 dB
```

That suggests the recovered unfolded image was already close to the sparse-codec ceiling. The dominant weakness was likely representation capacity for a hard-edged binary symbol, not collapse of the transported semantic state.

## AQ-YINYANG-1 Setup

`AQ-YINYANG-1` keeps the same codec and transport settings, but removes the second frame:

```text
mode        = --aq-yinyang-1
noise       = 0.00
seed        = 11
depth       = 3
controller  = free
frame_count = 1
patch_grid  = 4 x 4
patch_size  = 4 x 4
core_K      = 6
residual_K  = 4
preview_png = enabled
```

The frame is semanticized as sixteen `4 x 4` patches. Each patch is encoded by fixed low-frequency k-space slots plus four sparse residual instructions, transported through D-LinOSS, unfolded into a recovered coefficient field, and reconstructed by inverse FFT.

## Expected Visual Outcome

If the prior diagnosis is right, the recovered PNG should:

- preserve the large-scale yin/yang lobe polarity,
- show a coarse boundary and central dot structure,
- remain visibly blurred or smeared at hard edges,
- fail to be pixel-identical to the original binary reference.

This would not be a negative transport result by itself. A crisp binary yin/yang frame has substantial high-frequency edge content, while the current codec carries only ten of sixteen FFT slots per patch.

## Decision Rules

### Codec Ceiling Confirmed

If:

```text
oracle_psnr <= ~18 dB
ifft_psnr within ~2 dB of oracle_psnr
motif/source/target recovery = 1.0000
coefficient cosines remain high
```

then the bottleneck is the current sparse image encoding. The next correction should be a denser image codec, not more D-LinOSS controller tuning.

Likely next variants:

- all-slot `4 x 4` FFT recovery for this visual test,
- binary/edge-constrained post-decoder,
- patch vocabulary designed for edges, dots, and circular boundaries.

### Transport Or Decoder Failure

If:

```text
oracle_psnr >= ~24 dB
ifft_psnr remains low
```

then the codec can represent the frame but D-LinOSS or the unfolding decoder is losing recoverable image detail. The next correction should target transport signature extraction or the coefficient decoder.

### Temporal/Sequence Interference

If one-frame PSNR and visual quality improve sharply over `AQ-YINYANG-0`, then the two-frame alternating polarity setup introduced a sequencing or mixed-frame recovery burden. The next test should recover two frames as independent single-frame states before adding sequential folding again.

## Mathematical Expectation

The sparse reconstruction ceiling is governed by the retained k-space energy:

```latex
F_\Omega(k) =
\begin{cases}
F(k), & k \in \Omega \\
0, & k \notin \Omega
\end{cases}
```

and:

```latex
\hat{x}_{oracle} = \operatorname{Re}\left(\mathcal{F}^{-1}(F_\Omega)\right)
```

For binary edge images, many perceptually important terms live in high-frequency slots outside `Omega`. Even perfect D-LinOSS recovery of `F_Omega` cannot reconstruct those omitted edge harmonics.

The transport test is therefore:

```latex
\hat{x}_{D\text{-}LinOSS}
= \operatorname{Re}\left(\mathcal{F}^{-1}(\hat{F}_\Omega)\right)
```

If:

```latex
\|\hat{x}_{D\text{-}LinOSS} - \hat{x}_{oracle}\|_2
\ll
\|\hat{x}_{oracle} - x\|_2
```

then D-LinOSS is recovering the semanticized sparse field, and the visible blur belongs to the semanticization/codec layer.

## Expected Scientific Claim If Confirmed

The conservative claim would be:

> D-LinOSS can recover the folded sparse k-space semantic state with high identity and coefficient fidelity at `noise=0.00`, but pixel-level reconstruction quality is bounded by the image-domain vocabulary and retained coefficient support. For crisp binary images, the current codec is a lossy semanticization layer rather than a full image-preserving representation.

