# Program AQ-FFT-2 Remote Analysis - 2026-06-01

## TRC-safe handling

This note was written from TPU stdout inspected over SSH plus remote summary-file presence checks. No TPU result artifacts, arrays, checkpoints, directories, images, or summary JSON files were downloaded to the local machine.

Run context:

- project: `time-emission`
- node: `program-aqfft-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-fft-2 --aq-fft-top-k 6 --aq-fft-residual-k 2`
- backend: JAX TPU
- controllers: `free`, `shape_charge_control`
- noise: `0.00`
- seed: `11`

## Purpose

AQ-FFT-2 was designed to test a stronger transport-side hypothesis than AQ-FFT-1:

> if a small set of global semantic/frequency "shape charges" is actively stabilized during D-LinOSS signature formation, then residual coefficient recovery should improve.

The intended target was the AQ-FFT-1 bottleneck:

- semantic header stayed strong
- oracle codec capacity improved sharply
- realized reconstruction still lagged because residual instruction realization was weak

So AQ-FFT-2 was not asking whether the codec worked. AQ-FFT-1 had already shown that the codec was promising. AQ-FFT-2 asked whether a transport-time charge constraint could improve the recovery of the residual body.

## Methodology

AQ-FFT-2 kept the AQ-FFT-1 patch codec exactly:

- fixed low-frequency core: `K=6`
- adaptive residual set: `residual_K=2`
- sparse FFT payload with `sin(phi), cos(phi)` phase channels
- same `4 x 4` patch grid and `4 x 4` patch size

The new piece was the `shape_charge_control` arm.

For each patch, the code measured six low-dimensional observables:

$$
Q =
[Q_{DC}, Q_{low}, Q_{phase}, Q_{resid}, Q_{support}, Q_{motif}]
$$

where:

- \(Q_{DC}\): DC / luminance energy
- \(Q_{low}\): low-frequency energy
- \(Q_{phase}\): retained-coefficient phase coherence
- \(Q_{resid}\): adaptive residual energy
- \(Q_{support}\): support entropy
- \(Q_{motif}\): motif/header consistency

These charges were then mapped into a node-wise field:

$$
h_Q(n) = \mathrm{normalize}\left((Q - \bar{Q})G(n)\right)
$$

and used to perturb the D-LinOSS phase/frequency generator:

$$
\phi(n) \leftarrow \phi(n) + \eta h_Q(n)
$$

$$
\omega(n) \leftarrow \omega(n)\left(1 + 0.04\eta h_Q(n)\right)
$$

The key idea was to keep the folded state closer to a target shape manifold during transport.

AQ-FFT-2 also added a useful error budget:

- final PSNR
- PSNR if true support were known
- PSNR if true magnitudes were known
- PSNR if true phases were known
- shape-charge error and shape-charge cosine

This makes it possible to see whether the remaining reconstruction gap is mostly:

- support/index error
- magnitude error
- phase error
- or something more global

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
| motif recovery | `1.0000` |
| IFFT PSNR | `16.9995 dB` |
| oracle PSNR | `25.0480 dB` |
| coefficient magnitude cosine | `0.8551` |
| coefficient real cosine | `0.8613` |
| coefficient imaginary cosine | `0.6284` |
| phase cosine | `0.8930` |
| frequency-index overlap | `0.7578` |
| top-1 frequency accuracy | `1.0000` |
| top-K energy recall | `0.8738` |
| shape-charge error | `0.1221` |
| shape-charge cosine | `0.9956` |
| PSNR if true support known | `17.4428 dB` |
| PSNR if true magnitudes known | `22.9657 dB` |
| PSNR if true phases known | `17.8089 dB` |

### `shape_charge_control`

| metric | value |
| --- | ---: |
| noisy cosine / score proxy | `0.9717` |
| L1 occupation | `0.8750` |
| L2 low-frequency magnitude | `0.8715` |
| L3 low-frequency phase | `0.9528` |
| L4 sparse magnitude | `0.8557` |
| L5 grammar recovery | `0.9594` |
| leakage proxy | `0.0292` |
| motif recovery | `1.0000` |
| IFFT PSNR | `16.7852 dB` |
| oracle PSNR | `25.0480 dB` |
| coefficient magnitude cosine | `0.8558` |
| coefficient real cosine | `0.8619` |
| coefficient imaginary cosine | `0.6029` |
| phase cosine | `0.8708` |
| frequency-index overlap | `0.7578` |
| top-1 frequency accuracy | `1.0000` |
| top-K energy recall | `0.8738` |
| shape-charge error | `0.1233` |
| shape-charge cosine | `0.9956` |
| PSNR if true support known | `17.1629 dB` |
| PSNR if true magnitudes known | `22.0593 dB` |
| PSNR if true phases known | `17.6971 dB` |

## Conclusion

AQ-FFT-2 did **not** validate the shape-charge-control hypothesis.

The hypothesis was:

> stabilizing global shape charges during transport would improve the recovery of the residual coefficient body.

The observed result was the opposite:

- realized PSNR decreased slightly
- shape-charge error did not improve
- support overlap stayed unchanged
- top-K energy recall stayed unchanged
- imaginary-coefficient recovery got worse
- phase recovery got worse

So the bottleneck identified by AQ-FFT-1 did not move.

## Why the hypothesis failed

The failure is informative rather than random.

### 1. The controlled variables were too coarse

The six shape charges are global summaries. They say useful things about:

- total DC energy
- total low-frequency energy
- phase coherence
- residual energy
- support entropy
- motif consistency

But AQ-FFT-1's unresolved bottleneck was more specific than that:

> the system was failing to recover the detailed slotwise complex residual instructions.

Global charges do not uniquely determine:

- which residual slots are active
- their exact complex magnitudes
- their relative phase geometry

So even if the shape charges are preserved, the patch can still reconstruct poorly.

### 2. The control was open-loop regularization, not true adaptive feedback

The implemented perturbation changed \(\phi(n)\) and \(\omega(n)\), but it did not run an actual transport-time correction loop of the form:

$$
\lambda_k \leftarrow \lambda_k + \eta(Q_k^{target} - Q_k^{current})
$$

There was no iterative observation-and-correction during evolution. In practice, AQ-FFT-2 added a structured bias field rather than a genuine closed-loop stabilization rule.

That means the run tested:

- whether a charge-informed perturbation helps

not:

- whether adaptive conjugate-field control actually stabilizes the transport dynamics

### 3. The evidence says support/index error was untouched

The strongest diagnostic is that the following stayed unchanged:

- `freq_idx_overlap = 0.7578`
- `top1_freq_acc = 1.0000`
- `topK_energy_recall = 0.8738`

So the control did not improve the actual support-selection bottleneck.

That matters because AQ-FFT-1 had already pointed to residual-slot realization as the main unresolved issue. AQ-FFT-2 left that issue in place.

### 4. The charge metrics were already easy to decode

Both controllers achieved:

$$
\text{shape-charge cosine} = 0.9956
$$

That means the shape-charge vector itself was not the scarce variable. The system was already recovering those coarse observables almost perfectly.

So AQ-FFT-2 mainly confirmed:

> shape-charge summaries are not the limiting mathematical object.

The scarce object is still the finer coefficient-level reconstruction.

### 5. The perturbation slightly harmed the fragile complex channel

The `shape_charge_control` arm slightly reduced:

- `ifft_psnr`
- `phase_cos`
- `coeff_imag_cos`
- `psnr_if_true_magnitudes`
- `psnr_if_true_support`

That pattern suggests the extra field nudged the state away from the most useful coefficient geometry rather than toward it.

Methodologically, the added constraint was not aligned tightly enough with the actual decoder burden.

## What still holds

AQ-FFT-2 failing does **not** collapse the broader AQ picture. It narrows it.

What still looks mathematically solid from prior runs:

1. `AQ-0b`:
   the explicit `8`-motif semantic state is recoverable at the noise-free ceiling, with strong `L1`-`L4` recovery and low leakage.

2. `AQ-IMG-0`:
   image-domain semantic patch identity is recoverable with perfect motif accuracy. The reconstruction bottleneck is decoder/representation resolution, not semantic transport collapse.

3. `AQ-FFT-1`:
   the fixed-core plus sparse-residual k-space codec cleanly separates codec capacity from transport fidelity. This is a strong mathematical decomposition of the problem, even though the reconstruction is not solved.

4. `AQ-SEQ-0`:
   folded semantic source states can be recovered exactly, and a simple sequential gate can derive the next semantic state with `1.0000` accuracy even when direct target decoding is weaker.

These are the runs that currently support a defensible claim:

> folded semantic state recovery works for structured source states and simple sequential relations; richer reconstruction and operator-like transforms remain downstream problems.

## Updated decision

AQ should not treat shape-charge control as the current main path.

The better-supported mathematical foundation is:

- recover source semantic shape faithfully
- preserve explicit layer structure
- use sequential transition gates where the relation is simple and lawful
- treat media reconstruction as a separate codec/decoder problem

The strongest next work should stay close to those foundations rather than returning immediately to global charge-control hypotheses.
