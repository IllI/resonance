# Program AQ-DNA-0 Run Spec - 2026-06-01

## Purpose

AQ-DNA-0 is the next compact run after the AQ-FFT-2 negative result.

AQ-FFT-2 showed that global shape-charge control is too coarse. The shape summaries were already easy to recover, but they did not improve slotwise residual instruction recovery.

AQ-DNA-0 tests a different hypothesis:

> residual instructions should be encoded as local, quantized, locality-preserving tokens before folding, rather than as continuous coefficient vectors plus global summaries.

This keeps the strongest prior results:

- AQ-0b: source semantic layers are recoverable
- AQ-IMG-0: semantic patch identity is recoverable
- AQ-FFT-1: k-space codec capacity can be separated from transport fidelity
- AQ-SEQ-0: simple sequential semantic relations can be derived from recovered source states

and applies the attached tensor-network feedback:

```text
continuous signal
-> quantized instruction tokens
-> locality-preserving order
-> folded tensor-network state
-> token / dependency recovery
-> permutation validation
```

## Core hypothesis

AQ-FFT-1 failed to fully realize the sparse residual body because continuous residual coefficients are difficult to recover slot by slot.

AQ-DNA-0 replaces that burden with a discrete instruction grammar:

$$
\text{packet}_{p,k}
=
[
\text{patch}_p,
\text{slot}_k,
\text{magnitude state},
\text{phase state},
\text{residual state},
\text{rank/support state},
\text{motif header},
\text{checksum}
]
$$

The target is not higher PSNR alone. The target is statistically meaningful instruction recovery:

$$
I(\hat{T}_i;\hat{T}_j) > I(\hat{T}^{shuffle}_i;\hat{T}^{shuffle}_j)
$$

for related semantic/frequency token pairs.

## Encoding

Start from the AQ-FFT-1 codec:

- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed k-space core: `K=6`
- adaptive residual: `residual_K=2`
- noise: `0.00`
- seed: `11`
- controller: `free`

For every retained coefficient, build symbolic token states.

### Magnitude token

Replace raw magnitude regression with a small mixture-state code:

```text
mag_state in {low, medium, high}
```

A true GMM is ideal later, but the first TPU-safe gate can use deterministic mixture-like bins derived from the batch distribution:

$$
q_m(|F|) \in \{0,1,2\}
$$

using low/mid/high quantiles. This preserves the methodological idea without adding CPU-heavy fitting inside the TPU job.

### Phase token

Represent phase as sector tokens instead of raw angle:

```text
phase_state in {0, 1, 2, 3}
```

The decoder may still retain `sin(phi), cos(phi)` for reconstruction, but the token-recovery metric should measure phase-sector recovery.

### Residual token

Represent adaptive residual membership and energy:

```text
residual_state in {core, residual_low, residual_high}
```

This directly targets the AQ-FFT-1 / AQ-FFT-2 failure: residual slot realization.

### Support/rank token

Represent coefficient importance in the sparse packet:

```text
rank_state in {dominant, core, enhancement}
```

This separates the already-stable top-1 coefficient from the less stable residual slots.

## Locality ordering

AQ-DNA-0 should not feed tokens in arbitrary order.

Use a locality-preserving sequence:

1. patch order: `4 x 4` Hilbert curve
2. within-patch coefficient order: fixed low-frequency zigzag / k-space locality order
3. residual tokens placed immediately after their nearest fixed-core neighbor

For the initial `4 x 4` patch grid, a Hilbert order can be hard-coded.

This tests the tensor-network lesson directly:

> spend bond capacity on semantic/frequency dependency, not on undoing a bad ordering.

## Folded representation

Build the transported state from token features:

$$
Z =
[
E_{motif},
E_{slot},
E_{mag},
E_{phase},
E_{residual},
E_{rank},
E_{checksum}
]
$$

The TPU-side D-LinOSS job should remain compact:

- no hard controls as full arms
- no noise sweep
- no alpha sweep
- no residual_K expansion
- no global shape-charge arm

## Validation metrics

AQ-DNA-0 should report the usual AQ-FFT readout plus token/dependency metrics.

Core recovery:

- motif/header recovery
- token exact recovery
- magnitude-state accuracy
- phase-state accuracy
- residual-state accuracy
- support/rank-state accuracy
- checksum recovery

Reconstruction:

- realized PSNR
- oracle PSNR
- motif-only PSNR baseline
- AQ-FFT-1 comparison baseline: `17.00 dB`

Dependency validation:

- `MI(header; support)`
- `MI(motif; residual_state)`
- `MI(phase_state; reconstruction_error)`
- `MI(slot; local_patch_class)`
- `MI(residual_state; pixel_error)`

Permutation/null controls:

- shuffle coefficient slots
- shuffle phase tokens
- shuffle semantic headers
- shuffle locality order
- block-permute local tokens

Report:

$$
\Delta MI = MI_{observed} - MI_{shuffle}
$$

and, if cheap enough:

$$
p_{empirical}
=
\Pr(MI_{shuffle} \ge MI_{observed})
$$

## Success criteria

AQ-DNA-0 passes if:

- motif/header recovery stays at `1.0000`
- token exact recovery is above chance for all token classes
- residual-state recovery improves over AQ-FFT-1 residual realization diagnostics
- observed token MI beats shuffled/null MI
- realized PSNR beats the motif-only baseline and ideally approaches or exceeds AQ-FFT-1's `17.00 dB`

It is still useful if PSNR does not improve but token MI survives above null, because that would prove the folded state carries statistically meaningful semantic/frequency dependencies.

## Interpretation

If AQ-DNA-0 works, the defensible claim becomes:

> folded semantic transport is stronger when media payloads are encoded as locality-preserving instruction tokens rather than continuous residual coefficients.

If AQ-DNA-0 fails, then the next move is not another controller. It means the residual media payload needs either:

- a better token grammar
- a geometry-aware BP/message-passing decoder
- or a smaller reconstruction target before returning to image-like payloads

## Recommended launch shape

```bash
--aq-dna-0
```

Expected internal parameters:

- depth: `3`
- noise: `0.00`
- seed: `11`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- token ordering: Hilbert patch order plus k-space zigzag order
- validation: token recovery plus MI against shuffled controls
