# Program AQ-FFT-1 Remote Analysis - 2026-06-01

## TRC-safe handling

This note was written from TPU stdout inspected over SSH plus remote file presence checks. No TPU result artifacts, arrays, checkpoints, directories, images, or summary JSON files were downloaded to the local machine.

Run context:

- project: `time-emission`
- node: `program-aqfft-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-fft-1 --aq-fft-top-k 6 --aq-fft-residual-k 2`
- backend: JAX TPU
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- noise: `0.00`
- seed: `11`

## Purpose

AQ-FFT-1 was the first patch-codec run designed to combine two things that AQ-FFT-0 had separated:

1. a stable semantic header that the folded-state transport already recovers well
2. a higher-capacity reconstruction body that can carry more patch detail than a fixed low-frequency basis alone

AQ-FFT-0 had already shown that:

- semantic patch identity and header-like transform fields were highly recoverable
- fixed low-frequency `K=6` was too small a payload for good image reconstruction

So AQ-FFT-1 was not a robustness run. It was a codec-capacity run:

> keep the semantic header stable, add a small adaptive residual channel, and see whether realized reconstruction starts to move toward the new oracle ceiling.

## Methodology

Each `4 x 4` patch is converted into a `2D` FFT:

$$
F = \mathcal{F}_2(p)
$$

AQ-FFT-1 then transports a mixed payload with a fixed low-frequency core plus a small adaptive residual set.

### 1. Fixed core

The fixed basis order in code is:

$$
[0,1,4,5,2,8,6,9,10,3,12,7,13,11,14,15]
$$

The first `6` entries are always carried:

$$
\Omega_c = \{0,1,4,5,2,8\}
$$

This gives a stable low-frequency scaffold.

### 2. Adaptive residual

From the remaining coefficients, AQ-FFT-1 picks the `2` largest residual magnitudes:

$$
\Omega_r = \operatorname{TopK}_2\left(|F_i| : i \notin \Omega_c\right)
$$

The sparse transported spectrum is therefore:

$$
\tilde{F}_i =
\begin{cases}
F_i, & i \in \Omega_c \cup \Omega_r \\
0, & \text{otherwise}
\end{cases}
$$

This is why the total payload is `K=8`, but only `6` of those slots are fixed across patches.

### 3. Phase-safe coefficient representation

Instead of raw complex coordinates or wrapped angles, each retained coefficient is encoded as:

$$
\left(|F_i|,\cos\phi_i,\sin\phi_i\right)
$$

with:

$$
\phi_i = \arg(F_i)
$$

This is exactly why AQ-FFT-1 was less phase-fragile than earlier top-K experiments: the codec avoids direct angle-wrap discontinuities.

### 4. DNA-like instruction grammar

AQ-FFT-1 also adds a compact instruction block:

$$
g_i =
[\text{top\_mask}_i,\text{residual\_mask}_i,\bar{m}_i,\cos\phi_i,\sin\phi_i,\text{checksum}]
$$

where:

$$
\bar{m}_i = \frac{|F_i|}{\sum_j |F_j| + \epsilon}
$$

and the checksum is a bounded summary over slot identity, normalized magnitude, and phase-cosine content.

This is "DNA-like" only in the engineering sense of:

- stable header
- modular instruction packets
- redundancy/checksum
- expression through a reconstruction rule

No biological interpretation is implied by the experiment.

### 5. Decoder path

On the evaluation side, the ridge decoder predicts sparse FFT coefficients from the recovered folded state, rebuilds complex coefficients, and applies inverse FFT:

$$
\hat{p} = \Re\left(\mathcal{F}^{-1}_2(\hat{F})\right)
$$

So AQ-FFT-1 can be judged in two separate ways:

1. **oracle ceiling**: how much information the codec could reconstruct if transported perfectly
2. **realized result**: how much of that sparse spectrum the folded-state transport plus decoder actually recovers

That distinction is the key to interpreting this run.

## Result

### `free`

| metric | value |
| --- | ---: |
| noisy cosine / score proxy | `0.9722` |
| L1 occupation | `0.8750` |
| L2 low-frequency magnitude | `0.8710` |
| L3 low-frequency phase | `0.9544` |
| L4 sparse magnitude | `0.8551` |
| L5 grammar recovery | `0.9608` |
| leakage proxy | `0.0296` |
| source motif recovery | `1.0000` |
| target motif recovery | `1.0000` |
| transform exact match | `1.0000` |
| transform class recovery | `0.7578` |
| IFFT PSNR | `16.9995 dB` |
| oracle PSNR | `25.0480 dB` |
| coefficient magnitude cosine | `0.8551` |
| coefficient real cosine | `0.8613` |
| coefficient imaginary cosine | `0.6284` |
| phase cosine | `0.8930` |
| frequency-index overlap | `0.7578` |
| top-1 frequency accuracy | `1.0000` |
| top-K energy recall | `0.8738` |

### `tunnel_eigen_residual_alpha065`

| metric | value |
| --- | ---: |
| noisy cosine / score proxy | `0.9684` |
| L1 occupation | `0.8750` |
| L2 low-frequency magnitude | `0.8699` |
| L3 low-frequency phase | `0.9577` |
| L4 sparse magnitude | `0.8510` |
| L5 grammar recovery | `0.9599` |
| leakage proxy | `0.0293` |
| source motif recovery | `1.0000` |
| target motif recovery | `1.0000` |
| transform exact match | `1.0000` |
| transform class recovery | `0.7578` |
| IFFT PSNR | `16.6004 dB` |
| oracle PSNR | `25.0480 dB` |
| coefficient magnitude cosine | `0.8511` |
| coefficient real cosine | `0.8582` |
| coefficient imaginary cosine | `0.6528` |
| phase cosine | `0.8913` |
| frequency-index overlap | `0.7578` |
| top-1 frequency accuracy | `1.0000` |
| top-K energy recall | `0.8738` |

## Methodological interpretation

This run produced a very coherent picture.

### 1. Why oracle PSNR jumped

AQ-FFT-0 used a fixed low-frequency `K=6` basis only. Its oracle ceiling was about `11.83 dB`, which meant the codec itself was underpowered even before transport error entered.

AQ-FFT-1 raised the oracle ceiling to:

$$
25.0480\ \mathrm{dB}
$$

because the codec now carries:

- the same stable low-frequency scaffold
- plus `2` adaptive residual coefficients that capture patch-specific energy outside the fixed basis

Methodologically, this means the run succeeded at the first goal:

> the codec gained capacity.

### 2. Why realized PSNR improved, but not all the way to oracle

Realized PSNR rose to about:

$$
17.0\ \mathrm{dB}
$$

for `free`, which is a meaningful jump over AQ-FFT-0's `~11 dB` realized reconstruction.

But the remaining gap:

$$
25.0480 - 16.9995 \approx 8.05\ \mathrm{dB}
$$

shows that transport plus decode is still failing to fully realize the codec's new capacity.

That gap is methodologically important:

- if oracle had stayed low, the codec design would still be the bottleneck
- because oracle rose and realized PSNR followed only partway, the bottleneck has moved into **residual instruction recovery**

In plain terms: AQ-FFT-1 knows what extra detail to keep, but the system does not yet recover all of that extra detail accurately enough on the other side.

### 3. Why motif recovery stayed perfect

The semantic header remained extremely robust:

$$
\text{motif recovery} = 1.0000
$$

for both controllers.

That happened because motif identity is still carried through a comparatively simple, structured code path:

- motif code catalog features
- the stable low-frequency scaffold
- the grammar/checksum packet

Those are easier for the folded-state transport and ridge decoder to recover than full patch-detail coefficients.

So AQ-FFT-1 did **not** damage the semantic identity channel while adding reconstruction capacity. That is a real positive result.

### 4. Why top-1 frequency accuracy hit `1.0000` while overlap stayed at `0.7578`

This is one of the most informative diagnostics in the run.

We got:

- `top1_freq_acc = 1.0000`
- `freq_idx_overlap = 0.7578`

That means the system almost always recovered the single dominant coefficient correctly, but only partially recovered the full retained set.

Methodologically, this says:

- the coarse spectral anchor is stable
- the added residual slots are only partially stable

So the problem is not "FFT broke." The problem is narrower:

> residual-slot selection/value recovery is still harder than recovering the dominant scaffold.

### 5. Why phase looked decent while reconstruction still lagged

Phase stayed solid:

$$
\text{phase\_cos} \approx 0.89
$$

This is a strong sign that the `\sin\phi, \cos\phi` encoding did its job. We are not primarily losing the run to gross phase-wrap failure.

But:

- `coeff_mag_cos` stayed in the mid-`0.85` range
- `coeff_real_cos` stayed in the mid-`0.86` range
- `coeff_imag_cos` was notably lower, around `0.63` to `0.65`

That pattern tells us the transport is better at recovering:

- coarse magnitude structure
- real-valued low-frequency content

than:

- the full complex coefficient geometry, especially the more fragile imaginary component

So reconstruction lagged not because phase collapsed, but because **the complete complex coefficient recovery is still uneven**.

### 6. Why `free` beat `alpha065` at noise `0.00`

This run was a codec-ceiling diagnostic, not a noisy robustness test.

At noise `0.00`, `free` slightly outperformed `alpha065` on realized PSNR:

- `free`: `16.9995 dB`
- `alpha065`: `16.6004 dB`

while the rest of the semantic metrics were very close.

The likely reason is straightforward:

- `alpha065` introduces extra projection structure that can help under noisy/adversarial conditions
- at a clean ceiling, that same projection can smooth or compress some patch-detail variability that the codec is now trying to preserve

So this result does **not** mean `alpha065` is bad overall. It means:

> for this specific noise-free codec-capacity gate, the unconstrained path kept slightly more reconstructive detail.

### 7. Why control margins were `0.0000`

The AQ-FFT fast probe keeps runtime down by shortcutting the expensive comparator machinery. That makes this run useful for:

- codec ceiling
- coefficient recovery
- phase/index diagnostics

but not for:

- controller promotion
- adversarial robustness ranking

So the `0.0000` margin is not a failure signal here. It is just a consequence of the runtime-bounded probe design.

## What AQ-FFT-1 proved

AQ-FFT-1 proved three important things at once.

1. The semantic header channel remains stable even when the reconstruction body becomes richer.
2. A fixed-core plus sparse-residual k-space codec can sharply raise the reconstruction ceiling.
3. The next bottleneck is not semantic transport identity. It is accurate realization of the adaptive residual instructions.

That is exactly the kind of bottleneck shift we wanted to see. It means the experiment moved forward rather than just trading one failure for another.

## Where we stand

AQ-FFT-1 is a real improvement over AQ-FFT-0.

Conceptually:

- AQ-FFT-0 said "the semantic header survives, but the body is too small"
- AQ-FFT-1 says "the body can be made larger, but the residual instruction channel is now the limiting factor"

That is progress.

The cleanest next step is not a larger TPU endurance run. It is a tighter codec follow-up that asks whether the residual channel should become:

- slightly larger (`residual_k = 4`)
- more redundant
- or more structured around fixed enhancement slots instead of partly adaptive ones

## Immediate next-run implication

If we want the next run to stay runtime-bounded, the best follow-up gate is:

- keep `noise=0.00`
- keep the same patch roster
- keep `free` and `alpha065`
- compare the current `residual_k=2` run against either:
  - `residual_k=4`, or
  - a more fixed enhancement basis with fewer adaptive slot decisions

The decision rule should stay simple:

- if oracle rises again but realized PSNR does not, residual expression is still the bottleneck
- if realized PSNR rises with oracle, the codec genome is scaling in the right direction
