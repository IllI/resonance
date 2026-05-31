# Program AQ-IMG-0 Remote Analysis - 2026-05-30

## TRC-safe handling

This note was written from TPU stdout inspected over SSH. No TPU result artifacts, arrays, checkpoints, directories, images, or summary JSON files were downloaded.

Run context:

- project: `time-emission`
- node: `program-aqimg-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-img-0`
- backend: JAX TPU
- controller: `free`
- noise: `0.00`
- seed: `11`

## Purpose

AQ-IMG-0 is the first media-adjacent bridge for Program AQ. It does not attempt full image transport. It tests whether a tiny image-like field can be encoded into semantic patch states, transported through the folded D-LinOSS/tunnel-eigen pipeline, and decoded back into a coarse pixel representation.

The experiment separates two claims:

1. Semantic patch transport: can Bob recover the folded semantic patch identity?
2. Pixel reconstruction: can the recovered semantic state and residual geometry reconstruct pixel intensity with useful resolution?

This distinction matters because a perfect semantic label recovery does not automatically imply high-resolution pixel reconstruction.

## Methodology

AQ-IMG-0 uses a `4 x 4` grid of image patches. Each patch is a small `4 x 4` grayscale field. Each patch is assigned to one of eight image-domain semantic motifs:

| motif | image primitive |
| --- | --- |
| `flat` | uniform region |
| `h_edge` | horizontal edge |
| `v_edge` | vertical edge |
| `diagonal` | diagonal gradient |
| `checkerboard` | high-frequency texture |
| `gradient` | smooth ramp |
| `periodic` | periodic texture |
| `noise_burst` | sparse high-energy patch |

Each patch \(p_i\) is encoded as:

$$
p_i \mapsto (m_i, r_i, x_i, y_i)
$$

where:

- \(m_i\) is the nearest image-domain semantic motif
- \(r_i\) is a scalar residual/luminance correction
- \((x_i, y_i)\) is the patch position in the `4 x 4` grid

The TPU-side transported payload remains the semantic motif sequence:

$$
z_i = E(m_i)
$$

Residual and position information are carried as compact feature geometry:

$$
L_5(i) = [r_i, x_i, y_i].
$$

Bob receives the transported folded state:

$$
m_i
\xrightarrow{E}
z_i
\xrightarrow{\mathrm{transport}}
\tilde{z}_i
\xrightarrow{D}
\hat{S}_i
$$

and decodes:

$$
\hat{m}_i = C(\hat{S}_i),
\qquad
\hat{p}_i = R(\hat{S}_i, \hat{m}_i, \hat{r}_i).
$$

The important metric split is:

$$
\mathrm{MotifAcc}
= \Pr[\hat{m}_i = m_i],
$$

versus:

$$
\mathrm{PSNR}
= 10 \log_{10}\left(\frac{1}{\mathrm{MSE}(p_i,\hat{p}_i)}\right).
$$

## Result

Run shape:

- patch grid: `4 x 4`
- patch count: `16`
- patch size: `4 x 4`
- sequence length: `16`
- interval nodes: `136`
- depth: `3`

Core transport metrics:

| metric | value |
| --- | ---: |
| worst-control margin | `+0.3213` |
| folded score | `0.5022` |
| noisy graph cosine | `0.7429` |
| block lift | `+0.1818` |
| topology lift | `+0.3737` |
| untransported sanity | `0.9082` |
| L1 occupation recovery | `0.9789` |
| L2 transition recovery | `0.9454` |
| L3 FFT/phase recovery | `0.8269` |
| L4 recurrence recovery | `0.9588` |
| L5 residual/position recovery | `0.6592` |
| leakage proxy | `0.0789` |

Image codec metrics:

| metric | value |
| --- | ---: |
| motif recovery | `1.0000` |
| target recovery | `1.0000` |
| class recovery | `1.0000` |
| pixel PSNR | `16.8927 dB` |
| catalog PSNR | `35.9693 dB` |
| residual MAE | `0.0150` |
| luma MAE | `0.0252` |

Patch readout:

| row | true motif | predicted motif | target luma | predicted luma |
| ---: | --- | --- | ---: | ---: |
| 0 | `flat` | `flat` | `0.0000` | `0.0050` |
| 1 | `periodic` | `periodic` | `0.3733` | `0.4217` |
| 2 | `gradient` | `gradient` | `0.5005` | `0.4604` |
| 3 | `flat` | `flat` | `0.0000` | `0.0050` |
| 4 | `checkerboard` | `checkerboard` | `0.5200` | `0.5092` |
| 5 | `gradient` | `gradient` | `0.4854` | `0.4604` |
| 6 | `v_edge` | `v_edge` | `0.5050` | `0.4774` |
| 7 | `h_edge` | `h_edge` | `0.5150` | `0.5150` |
| 8 | `h_edge` | `h_edge` | `0.5100` | `0.5150` |
| 9 | `periodic` | `periodic` | `0.3027` | `0.4217` |
| 10 | `diagonal` | `diagonal` | `0.5884` | `0.5644` |
| 11 | `noise_burst` | `noise_burst` | `0.4656` | `0.4546` |
| 12 | `diagonal` | `diagonal` | `0.5955` | `0.5644` |
| 13 | `v_edge` | `v_edge` | `0.5150` | `0.4774` |
| 14 | `checkerboard` | `checkerboard` | `0.5100` | `0.5092` |
| 15 | `noise_burst` | `noise_burst` | `0.4412` | `0.4546` |

## Interpretation

The significant result is:

$$
\Pr[\hat{m}_i = m_i] = 1.0000
\quad \mathrm{for}\quad 16/16\ \mathrm{patches}.
$$

This supports the Program AQ transport claim at the image-domain semantic level:

> A folded semantic patch state can be encoded, transported, and decoded with perfect class recovery in the noise-free gate.

The layer metrics show that the recovered signal is not only a class-label artifact. The transport also preserved structured semantic layers:

$$
L_1 = 0.9789,\quad
L_2 = 0.9454,\quad
L_3 = 0.8269,\quad
L_4 = 0.9588.
$$

That means the recovered state retains occupation, transition, phase, and recurrence structure at useful fidelity.

The weaker result is pixel-level reconstruction:

$$
\mathrm{PSNR}_{pixel} = 16.8927\ \mathrm{dB}.
$$

This is not a transport failure. It means the current reconstruction/decoder has limited resolution. The semantic class survives cleanly, but the mapping:

$$
(\hat{S}_i,\hat{m}_i,\hat{r}_i) \rightarrow \hat{p}_i
$$

does not yet recover enough patch-internal variation to produce high-fidelity pixels.

## Encoding and decoder obstacle

The reconstruction limit may come from the construction/encoding methodology as much as from the decoder.

The current encoder collapses each patch into:

$$
p_i \mapsto (m_i, r_i)
$$

where \(m_i\) captures the nearest semantic primitive and \(r_i\) is only a scalar residual. This is too coarse for patches whose important error is not just mean luminance. For example, two `periodic` patches can share motif identity but differ in contrast, phase, duty cycle, or local amplitude. A scalar residual cannot express those differences.

That explains the observed pattern:

- motif recovery is perfect
- catalog PSNR is high
- residual MAE is low
- pixel PSNR remains modest
- some same-class patches receive nearly the same predicted luma

In information terms, the current code preserves:

$$
I(\tilde{z}; m_i)
$$

very well, but it does not yet preserve enough:

$$
I(\tilde{z}; p_i \mid m_i)
$$

to reconstruct fine pixel detail.

The likely bottleneck is conditional detail within a recovered semantic class:

$$
p_i = \mathrm{catalog}(m_i) + \epsilon_i,
$$

but the current \(\epsilon_i\) is represented by too few degrees of freedom.

## Claim boundary

AQ-IMG-0 supports this claim:

> Image-domain semantic patch identities and layer signatures can be transported through the folded semantic channel and recovered at the receiver in a noise-free gate.

AQ-IMG-0 does not yet support this stronger claim:

> Arbitrary image pixels can be reconstructed at high fidelity from the transported folded state.

The next result should target the gap between these two claims.

## Updated next run guidance

The original AQ-IMG-1 vector-residual idea was tested later through AQ-IMG-1-LITE and did not hold up in its first form.

See:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG1_LITE_REMOTE_ANALYSIS_2026-05-31.md`

What changed:

- the runtime gate worked after restricting the probe to `free` only and skipping heavy comparator signatures
- the vector-residual branch itself underperformed
- motif recovery collapsed from the AQ-IMG-0 baseline `1.0000` to `0.3750`
- pixel PSNR dropped from `16.8927 dB` to `9.7162 dB`

The current best deduction is that the lite probe mixed too many changes at once:

- fewer patches
- one exemplar per motif
- wider residual targets
- thinner decoder conditioning

So the next fair comparison is no longer:

$$
p_i \mapsto (m_i, r_i)
\quad \text{versus} \quad
p_i \mapsto (m_i, r_i^{(0:4)})
$$

on different patch rosters.

The next fair comparison should hold the patch roster fixed and compare:

$$
\text{small repeated patch set} + \text{scalar residual}
$$

against:

$$
\text{small repeated patch set} + \text{vector residual}
$$

with:

- `free` controller only
- the fast runtime guard
- repeated motif classes preserved
- pixel PSNR, motif accuracy, residual MAE, and per-class collapse pattern reported

Only after the vector-residual branch improves over the AQ-IMG-0 baseline on a controlled small repeated patch set should AQ-IMG move to larger grids, noise, or temporal frame sequencing.
