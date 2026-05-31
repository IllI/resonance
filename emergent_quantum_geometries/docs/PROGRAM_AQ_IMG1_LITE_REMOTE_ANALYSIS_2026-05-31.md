# Program AQ-IMG-1-LITE Remote Analysis - 2026-05-31

## TRC-safe handling

This note was written from TPU stdout inspected over SSH. No TPU result artifacts, arrays, checkpoints, directories, images, or summary JSON files were downloaded.

Run context:

- project: `time-emission`
- node: `program-aqimg-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-img-1-lite`
- backend: JAX TPU
- controller: `free`
- noise: `0.00`
- seed: `11`

## Purpose

AQ-IMG-1-LITE was introduced as a fast gate after the full AQ-IMG-1 path proved too expensive for a first probe. The goal was not to beat AQ-IMG-0 on raw transport fidelity yet. The goal was to answer a narrower question quickly:

> If we replace the scalar residual with a compact vector residual grammar, does reconstruction move in the right direction without paying the full `16`-patch runtime?

This run therefore tested the encoding idea under a compressed budget:

- `8` patches instead of `16`
- `36` interval nodes instead of `136`
- `free` controller only
- no heavy comparator branches

## Methodology

AQ-IMG-1-LITE still uses the same `8` image-domain semantic motifs as AQ-IMG-0:

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

But the patch encoding changed from scalar residual:

$$
p_i \mapsto (m_i, r_i)
$$

to vector residual:

$$
p_i \mapsto (m_i, r_i^{(0)}, r_i^{(1)}, r_i^{(2)}, r_i^{(3)}, r_i^{(4)})
$$

where the residual vector modulates:

- a scalar offset
- motif-centered contrast
- a horizontal basis
- a vertical basis
- a diagonal low-frequency basis

In code terms, the patch pixels are synthesized as:

$$
\hat{p}_i =
\mathrm{catalog}(m_i)
+ r_i^{(0)}
+ r_i^{(1)} c_i
+ r_i^{(2)} h
+ r_i^{(3)} v
+ r_i^{(4)} d
$$

where \(c_i\) is the centered motif pattern and \(h, v, d\) are fixed spatial bases.

The lite payload also changed patch selection:

- only the first `8` patch states are used
- this means one representative patch per motif class

So the fast probe uses:

$$
\{m_i\}_{i=1}^8
$$

with one class instance each, instead of the `16`-patch AQ-IMG-0 layout that included repeated motif classes and repeated residual contexts.

## Result

Runtime fix:

- the original AQ-IMG-1-LITE timeout problem was solved by restricting the run to `free` only and skipping heavy comparator signatures for the fast probe
- the corrected lite run completed and printed final metrics

Observed metrics:

| metric | value |
| --- | ---: |
| worst-control margin | `0.0000` |
| folded score | `0.3009` |
| noisy graph cosine | `0.6362` |
| block lift | `0.0000` |
| topology lift | `0.0000` |
| untransported sanity | `0.7815` |
| L1 occupation recovery | `0.8987` |
| L2 transition recovery | `0.7661` |
| L3 FFT/phase recovery | `0.6712` |
| L4 recurrence recovery | `0.8391` |
| L5 residual/position recovery | `0.5798` |
| leakage proxy | `0.1047` |
| motif recovery | `0.3750` |
| pixel PSNR | `9.7162 dB` |
| catalog PSNR | `6.5688 dB` |
| residual MAE | `0.0234` |
| luma MAE | `0.1012` |

Patch readout:

| row | true motif | predicted motif | target luma | predicted luma |
| ---: | --- | --- | ---: | ---: |
| 0 | `periodic` | `v_edge` | `0.3767` | `0.4824` |
| 1 | `flat` | `flat` | `0.0000` | `0.2609` |
| 2 | `gradient` | `v_edge` | `0.4836` | `0.4905` |
| 3 | `v_edge` | `v_edge` | `0.5062` | `0.4906` |
| 4 | `h_edge` | `flat` | `0.5210` | `0.2609` |
| 5 | `diagonal` | `v_edge` | `0.5967` | `0.4882` |
| 6 | `checkerboard` | `v_edge` | `0.5114` | `0.4898` |
| 7 | `noise_burst` | `noise_burst` | `0.4396` | `0.4697` |

## Interpretation

The important positive result is operational:

> the fast probe now completes under the TPU runtime guard and produces a usable answer.

That matters because the earlier full AQ-IMG-1 branch timed out before producing valid metrics. The lite branch now does its job as a gate.

The scientific result is negative:

$$
\mathrm{MotifAcc} = 0.3750
$$

and:

$$
\mathrm{PSNR}_{pixel} = 9.7162\ \mathrm{dB}
$$

Both are materially worse than AQ-IMG-0:

- motif recovery dropped from `1.0000` to `0.3750`
- pixel PSNR dropped from `16.8927 dB` to `9.7162 dB`

So the vector-residual lite encoding did not merely fail to improve reconstruction. It degraded the semantic patch decode itself.

## What the patch errors say

The per-patch readout is not random. It shows a specific collapse pattern:

- `periodic`, `gradient`, `diagonal`, and `checkerboard` all collapsed toward `v_edge`
- `h_edge` collapsed toward `flat`
- `flat` and `noise_burst` survived

This means the decoder did not completely lose the signal. It retained coarse geometry and coarse energy classes, but not the finer class boundaries. In particular:

- patches with mid-level luminance and strong directional structure drifted toward the `v_edge` basin
- the zero-energy anchor (`flat`) remained a stable attractor
- the sparse/high-energy anchor (`noise_burst`) remained recoverable

That is a classic sign that the decoder is falling back to a small number of robust prototypes rather than recovering the full motif vocabulary.

## Why the lite patch construction hurt

The logs and code together point to three causes.

### 1. One patch per motif removed within-class structure

AQ-IMG-0 used `16` patch states with repeated motif classes. That gave the decoder multiple examples of:

- `periodic`
- `gradient`
- `diagonal`
- `checkerboard`
- edge motifs with different residuals

AQ-IMG-1-LITE uses only the first `8` patch states:

$$
[0,5,1,2,3,4,6,7]
$$

which is effectively one exemplar per motif.

That means the decoder no longer sees class variation. It sees only one point per class. The ridge decoder therefore has much less evidence to separate:

$$
m_i
\quad \text{from} \quad
(m_i,\text{position},\text{residual vector})
$$

The fast probe got cheaper partly by removing the very repetitions that helped AQ-IMG-0 stay stable.

### 2. The target feature width increased sharply without adding new transport states

AQ-IMG-0 source features were:

$$
\mathrm{source\_features} = [\mathrm{patch\_features}, r_i, \mathrm{pos}_i]
$$

AQ-IMG-1-LITE source features became:

$$
\mathrm{source\_features} = [\mathrm{patch\_features}, r_i^{(0:4)}, \mathrm{pos}_i]
$$

So the decoder was asked to recover a larger joint target from the same transport signatures, but without adding richer transport examples. In effect, the residual side-channel became more expressive while the training geometry stayed tiny.

This makes the source decode less well-conditioned. The motif code is still present at the front of `source_features`, but the ridge solve has to fit it jointly with several extra residual coordinates and patch position.

That is consistent with the drop in:

$$
\mathrm{source\_motif\ recovery}
$$

from `1.0000` to `0.3750`.

### 3. The fast probe stopped meaningfully testing control margins

To make AQ-IMG-1-LITE finish, the fast probe reused the noisy signature for the expensive control branches instead of compiling separate topology/block/crofton variants. That was the right runtime fix, but it means:

- `margin = 0.0000`
- `block lift = 0.0000`
- `topology lift = 0.0000`

These fields are not evidence of failure in the transport physics. They are neutralized by construction in the lite probe. AQ-IMG-1-LITE should therefore be interpreted mainly through:

- motif recovery
- PSNR
- residual MAE
- layer recovery

not through control-margin ranking.

## What we can safely conclude

AQ-IMG-1-LITE tells us four useful things.

1. The runtime fix worked.
   The lite branch now produces results quickly enough to use as a gate.

2. The vector residual idea is not validated in its current form.
   It degraded both semantic patch recovery and pixel reconstruction.

3. The failure is likely in the decoder conditioning and patch sampling, not in the basic AQ transport path.
   AQ-IMG-0 reran cleanly with the same stable numbers, so the underlying semantic patch transport remains intact.

4. The current lite dataset is too thin to judge the vector residual grammar fairly.
   One exemplar per motif plus a wider residual target creates a harder decoding problem than AQ-IMG-0, even before asking whether the richer residual basis is useful.

## Recommended next move

Do not promote the current AQ-IMG-1 branch.

The next image-codec attempt should keep the runtime discipline from AQ-IMG-1-LITE, but restore enough decoder conditioning to make the comparison fair:

- keep `free` only
- keep the fast timeout guard
- keep the smaller interval-node geometry
- restore repeated motif classes instead of one exemplar per class
- widen the training set before widening the residual grammar further

Concretely, the next fair gate is:

$$
\text{small repeated patch set}
+ \text{scalar residual baseline}
$$

versus:

$$
\text{small repeated patch set}
+ \text{vector residual grammar}
$$

with the patch roster held constant across both.

That would isolate the encoding change itself, instead of mixing:

- fewer patches
- less class repetition
- wider targets
- and different runtime shortcuts

all in one step.
