# Program AQ-DNA-0 Remote Analysis - 2026-06-01

## TRC-safe handling

This note was written from TPU stdout inspected over SSH. No TPU result artifacts, arrays, checkpoints, images, summary JSON files, or result directories were downloaded locally.

Run context:

- project: `time-emission`
- node: `program-aqfft-eu-node-v1`
- zone: `europe-west4-a`
- launch mode: `--aq-dna-0`
- backend: JAX TPU
- controller: `free`
- noise: `0.00`
- seed: `11`
- runtime: true AQ-DNA-0 token branch, not the earlier AQ-FFT-1 alias

## What was actually tested

This was the first real AQ-DNA-0 gate.

The branch printed the expected activation banner:

```text
AQ-DNA-0 ACTIVE:
  tokenization = deterministic GMM-lite bins
  patch_order = Hilbert-4x4
  kspace_order = zigzag/local fixed core + adaptive residual
  token_metrics = enabled
  permutation_nulls = enabled
  runtime_envelope != AQ-FFT-1
```

Methodologically, the run did three new things relative to AQ-FFT-1:

- replaced the continuous residual instruction body with discrete token states
- measured token-packet recovery directly
- measured token dependency signal against permutation/null baselines

But the run also showed a limitation that matters for interpretation:

> the token branch changed the recoverable label space much more than it changed the underlying transport geometry.

So AQ-DNA-0 was a real new branch, but not yet a deep structural departure from the AQ-FFT codec path.

## Observed result

| metric | value |
| --- | ---: |
| motif/header recovery | `1.0000` |
| IFFT PSNR | `16.9995 dB` |
| oracle PSNR | `25.0480 dB` |
| coefficient magnitude cosine | `0.8551` |
| coefficient real cosine | `0.8613` |
| coefficient imaginary cosine | `0.6284` |
| phase cosine | `0.8930` |
| frequency-index overlap | `0.7578` |
| top-1 frequency accuracy | `1.0000` |
| top-K energy recall | `0.8738` |
| instruction packet exact match | `0.9062` |
| magnitude-state accuracy | `0.9688` |
| phase-state accuracy | `0.9453` |
| residual-state accuracy | `0.9531` |
| support-rank accuracy | `0.9531` |
| `MI(header; support)` gap | `-0.0022 bits` |
| `MI(motif; residual_state)` gap | `+0.0210 bits` |
| `MI(phase_state; reconstruction_error)` gap | `+0.0082 bits` |
| `MI(slot; local_patch_class)` gap | `0.0000 bits` |
| `p(header; support)` | `0.3750` |
| `p(motif; residual_state)` | `0.2500` |
| `p(phase_state; reconstruction_error)` | `0.1250` |
| `p(slot; local_patch_class)` | `1.0000` |

## What this proves

### 1. Token states are recoverable

AQ-DNA-0 passed the first practical implementation test:

- packet exact match reached `0.9062`
- all four token families were recovered strongly
- motif/header recovery stayed at `1.0000`

So the folded-state channel can support discrete coefficient-packet labels. This is a real result.

### 2. The semantic identity channel is still not the problem

The run kept the now-familiar stable pieces:

- motif/header transport solved
- top-1 spectral anchor solved
- phase still decent
- oracle ceiling still high

That keeps the main AQ media-side bottleneck sharply defined:

> residual coefficient instruction realization is still the limiting factor.

## What did not improve

### 1. Realized reconstruction did not move

Realized PSNR stayed at:

$$
16.9995\ \mathrm{dB}
$$

which is effectively the same plateau as AQ-FFT-1 under `free`.

So converting the residual body into discrete packet labels did **not** by itself close the gap to the:

$$
25.0480\ \mathrm{dB}
$$

oracle ceiling.

### 2. Dependency signal stayed weak

This is the most important outcome in the run.

The token labels decoded well, but the mutual-information gaps over permutation nulls stayed tiny:

- `-0.0022 bits`
- `+0.0210 bits`
- `+0.0082 bits`
- `0.0000 bits`

with weak p-values throughout.

That means the current AQ-DNA-0 token genome is not yet behaving like a structurally informative dependency code. Right now it behaves more like:

> a recoverable relabeling layer over the same underlying transport problem.

In plain terms: the decoder can read the categories, but those categories are not yet imposing enough locality/dependency structure to improve reconstruction.

## Methodological interpretation

AQ-DNA-0 gives a stable, repeatable failure mode:

- semantic/header transport works
- codec capacity is high
- discrete packet labels are recoverable
- but the token grammar is not yet structurally stronger than the null permutations
- and realized PSNR remains stuck at the old plateau

This is a cleaner failure than AQ-FFT-2.

AQ-FFT-2 told us that global shape-charge summaries were too coarse.
AQ-DNA-0 now tells us that:

> tokenizing the residual body is not enough if the tokenization does not materially change the transport geometry or dependency structure.

That is why the MI gaps stayed near zero even though token accuracies were high.

## Course correction

The next move should **not** be another controller sweep, more noise, or a larger payload.

AQ-DNA-0 says the next correction needs to be structural.

### What to change next

1. Make locality real, not just declared.
   The current token branch still reuses much of the old patch/FFT scaffold. The next version should explicitly change sequence/packet ordering and neighborhood structure, not only the label vocabulary.

2. Move from per-slot token recovery to local packet dependency recovery.
   Right now the branch mostly tests whether the decoder can classify each retained slot. The next gate should force packet neighborhoods or short coefficient windows to matter jointly.

3. Target the weak residual body directly.
   The stable failure is still in the weaker residual and complex-detail components, especially the parts not covered by the dominant spectral anchor.

### What not to do next

- do not add `alpha065`
- do not add a noise sweep
- do not expand `residual_K`
- do not interpret this as a controller problem

## Bottom line

AQ-DNA-0 succeeded as an implementation gate and failed as a transport-improvement gate.

It proved:

> discrete coefficient packets can be recovered.

It did **not** prove:

> locality-preserving tokenization improves residual instruction realization.

So the next scientific question is now sharper:

> how do we make the token grammar structurally active in the folded transport, rather than merely easy to decode after transport?
