# Program AQ Plan - Semantic Subspace Transport

## Core Claim

Program AQ should not claim "teleport arbitrary meaning." The experimentally defensible claim is narrower:

> Encode a structured semantic payload into a compressed tensor/MPS-like subspace, transport it through the tunnel-eigen residual channel, and test whether Bob can reconstruct the semantic layers above free and hard-control baselines.

This aligns with the quantum-autoencoder analogy:

```text
compress -> transport -> decompress
```

The target is a semantically sufficient subspace, not the full Hilbert space.

## Current Foundation

After the operator and shape-charge detours, the current mathematically supported AQ claim is narrower than the most ambitious hypothesis:

> folded semantic source-state recovery is real; lawful sequential semantic reconstruction is promising; operator transport and high-resolution media reconstruction remain open problems.

The runs that currently provide the strongest foundation are:

1. `AQ-0b`:
   explicit `8`-motif semantic states are recoverable at the noise-free ceiling with strong `L1`-`L4` recovery and low leakage.

2. `AQ-IMG-0`:
   semantic image-patch identity is perfectly recoverable, while pixel fidelity is limited by encoder/decoder resolution rather than by collapse of semantic transport.

3. `AQ-FFT-1`:
   the fixed-core plus sparse-residual k-space codec cleanly separates codec capacity from transport fidelity, which gives AQ a solid mathematical diagnostic frame for media-like payloads.

4. `AQ-SEQ-0`:
   recovered source states plus a simple transition gate can derive the next semantic state exactly, even when direct next-state decoding is weaker.

The runs that do **not** currently support a strong claim are:

1. operator-family transport (`AQ-1b`, `AQ-1c`)
2. global shape-charge stabilization as the main missing ingredient (`AQ-FFT-2`)

## Runtime Principle

AQ is a rapid-prototyping line, not a durability run. Keep TPU runs small enough to turn over quickly:

- run a noise-free ceiling first
- keep AP-U calibration fail-fast
- save compact transport artifacts and metadata only
- move decoder comparisons to offline CPU
- avoid IBM-lite and new adversaries until the semantic payload works

## AQ-FFT-1 Codec Pivot

AQ-FFT-0 separated the problem cleanly: semantic headers recovered perfectly, but fixed low-frequency `K=6` had a low image reconstruction ceiling (`oracle_psnr` about `11.83 dB`). The next run should therefore test codec capacity, not controller robustness.

AQ-FFT-1 combines:

- fixed low-frequency k-space core
- sparse adaptive residual k-space coefficients
- phase-safe `sin(phi), cos(phi)` channels
- a DNA-like instruction grammar used as a codec analogy only

The useful idea from DNA-style encoding is modular reconstruction instruction, not biological consciousness. In AQ terms:

| analogy | AQ-FFT-1 field |
| --- | --- |
| base token | frequency slot |
| codon | coefficient instruction packet |
| gene | patch reconstruction program |
| regulatory marker | semantic header / checksum |
| expression | inverse FFT reconstruction |

First gate:

- mode: `--aq-fft-1`
- core: `--aq-fft-top-k 6`
- residual: `--aq-fft-residual-k 2`
- noise: `0.00`
- depth: `3`
- seed: `11`
- controllers: `free`, `tunnel_eigen_residual_alpha065`

Interpretation rule:

- if oracle PSNR rises and realized PSNR follows, the core+residual genome is the right direction
- if oracle PSNR rises but realized PSNR does not, recovery of residual instructions is the bottleneck
- if oracle PSNR does not rise, the patch codec still lacks capacity

AQ-FFT-1 result:

- run record: `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT1_REMOTE_ANALYSIS_2026-06-01.md`
- `oracle_psnr` rose from AQ-FFT-0's `~11.83 dB` to `25.05 dB`
- realized `ifft_psnr` rose to `17.00 dB` under `free`
- motif/header recovery stayed at `1.0000`
- `phase_cos` stayed around `0.89`
- `top1_freq_acc` reached `1.0000`, while full frequency-index overlap stayed at `0.7578`

Methodological read:

- the codec-capacity problem improved materially
- the new bottleneck is residual instruction realization, not semantic header transport
- `free` slightly beat `alpha065` at `noise=0.00`, which suggests projection helps less at the clean codec ceiling than it does under noisy conditions

## AQ-FFT-2 Shape-Charge Control

The next run applies the tensor-network feedback lesson from recent lattice/tensor-network work: do not ask a shallow decoder to rediscover geometry after transport. Build problem geometry and approximately conserved observables into the transport path.

AQ-FFT-2 keeps the AQ-FFT-1 codec:

- fixed core: `K=6`
- adaptive residual: `residual_K=2`
- phase-safe `sin(phi), cos(phi)`
- DNA-like instruction grammar as a codec analogy

The triplet/codon analogy is kept concrete: each coefficient packet is treated as a small three-part instruction,

$$
(\text{slot/support}, |F|, [\cos\phi,\sin\phi])
$$

whose combined effect is a recoverable complex "shape charge" for that frequency slot. This is not a claim about literal quarks or DNA chemistry; it is a compact way to test whether grouped support, magnitude, and phase channels behave better than isolated scalar residuals.

The new addition is a `shape_charge_control` arm. It stabilizes six explicit observables during D-LinOSS signature formation:

| charge | meaning |
| --- | --- |
| `Q_DC` | DC / luminance energy |
| `Q_lowfreq_energy` | low-frequency energy |
| `Q_phase_coherence` | retained-coefficient phase coherence |
| `Q_residual_energy` | adaptive residual energy |
| `Q_support_entropy` | coefficient support spread |
| `Q_motif_consistency` | semantic header consistency |

These charges act like semantic/frequency analogues of conserved quantities. The implementation maps the charge vector into a node-wise conjugate field:

$$
h_Q(n) = \mathrm{normalize}\left((Q - \bar{Q})G(n)\right)
$$

and perturbs the D-LinOSS phase/frequency generator:

$$
\phi(n) \leftarrow \phi(n) + \eta h_Q(n)
$$

$$
\omega(n) \leftarrow \omega(n)\left(1 + 0.04\eta h_Q(n)\right)
$$

This is the practical AQ analogue of adaptive conjugate-field control: the run tries to keep shape observables from drifting during transport instead of relying only on post-hoc regression.

First gate:

- mode: `--aq-fft-2`
- core: `--aq-fft-top-k 6`
- residual: `--aq-fft-residual-k 2`
- noise: `0.00`
- depth: `3`
- seed: `11`
- controllers: `free`, `shape_charge_control`

Success criteria:

- realized PSNR beats AQ-FFT-1's `17.00 dB`
- `fft_shape_charge_error` decreases versus `free`
- residual/imaginary coefficient recovery improves
- motif/header recovery remains `1.0000`

AQ-FFT-2 result:

- run record: `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT2_REMOTE_ANALYSIS_2026-06-01.md`
- the shape-charge-control hypothesis failed
- `shape_charge_control` did not improve `ifft_psnr`, shape-charge error, support overlap, or top-K energy recall
- `shape_charge_control` slightly worsened phase and imaginary-coefficient recovery

Methodological conclusion:

- the bottleneck was not failure to preserve low-dimensional global shape summaries
- the bottleneck remains coefficient-level residual instruction realization
- the AQ-FFT-2 field acted more like an open-loop bias than a true adaptive feedback controller
- AQ should not currently treat global shape-charge control as the main path forward

Updated read:

- AQ-FFT-1 remains the stronger mathematical decomposition of the media-side problem
- AQ-FFT-2 is useful mainly as a negative result: global charge summaries are too coarse to recover the missing slotwise complex detail

## AQ-DNA-0 Locality-Preserving Instruction Genome

The next run should not be another global charge-control variant. AQ-FFT-2 showed that low-dimensional shape summaries are already recoverable and are not the scarce object.

AQ-DNA-0 instead targets the actual bottleneck:

> recover slotwise residual instructions by converting continuous coefficient details into local, quantized, locality-preserving tokens before folding.

Run record/spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_DNA0_RUN_SPEC_2026-06-01.md`

Core encoding:

```text
coefficient packet =
  patch_id
  frequency_slot
  magnitude_state
  phase_state
  residual_state
  support_rank_state
  motif_header
  checksum
```

The important methodological changes are:

- replace raw residual regression with mixture-style token states
- preserve image locality with a `4 x 4` Hilbert patch order
- preserve k-space locality with fixed zigzag / local frequency order
- validate token dependencies with mutual information and permutation controls

First gate:

- mode: `--aq-dna-0`
- depth: `3`
- noise: `0.00`
- seed: `11`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- codec basis: AQ-FFT-1 `K=6`, `residual_K=2`

Primary metrics:

- motif/header recovery
- token exact recovery
- magnitude-state accuracy
- phase-state accuracy
- residual-state accuracy
- support/rank-state accuracy
- realized PSNR and oracle PSNR
- `MI(header; support)`
- `MI(motif; residual_state)`
- `MI(phase_state; reconstruction_error)`
- `MI(slot; local_patch_class)`
- observed MI minus shuffled/null MI

Success criteria:

- semantic header remains `1.0000`
- token classes recover above chance
- observed token MI beats shuffled/null MI
- realized PSNR beats motif-only reconstruction and ideally meets or exceeds AQ-FFT-1's `17.00 dB`

Interpretation:

- if token MI survives above null, AQ has evidence that the folded state carries meaningful semantic/frequency dependencies
- if PSNR also improves, the locality-preserving instruction genome becomes the next media-side path
- if token MI fails, repair the token grammar or decoder before adding noise, controllers, or larger payloads

AQ-DNA-0 true gate result:

- run record: `emergent_quantum_geometries/docs/PROGRAM_AQ_DNA0_REMOTE_ANALYSIS_2026-06-01.md`
- the run executed the real AQ-DNA-0 token branch
- token metrics and permutation/null diagnostics were enabled
- controller set stayed at `free` only

Observed result:

- motif/header recovery stayed `1.0000`
- `oracle_psnr` stayed `25.0480 dB`
- realized `ifft_psnr` stayed `16.9995 dB`
- `phase_cos` stayed about `0.893`
- `freq_idx_overlap` stayed `0.7578`
- `topK_energy_recall` stayed `0.8738`
- instruction packet exact match reached `0.9062`
- token accuracies were high: magnitude `0.9688`, phase `0.9453`, residual `0.9531`, support-rank `0.9531`
- token MI gaps over permutation nulls stayed weak: `-0.0022`, `+0.0210`, `+0.0082`, `0.0000` bits

Methodological conclusion:

- discrete coefficient-packet labels are recoverable
- but locality-preserving tokenization did not improve realized reconstruction or produce strong dependency signal over null permutations
- AQ-DNA-0 is therefore a successful implementation gate but not yet a successful transport-improvement gate
- the next correction should target structural locality/dependency in the transport path, not another controller or noise sweep

## AQ-HYBRID-0 Continuous Geometry + Token Checksum

The attached implementation feedback sharpens the next move:

> tokenization may be obscuring the continuous D-LinOSS/k-space geometry rather than improving it.

AQ-DNA-0 showed that hard packet labels are easy to recover, but they did not improve reconstruction or carry meaningful dependency signal above null. That suggests the token labels are not the right primary variables.

The next run should therefore keep continuous geometry primary and demote tokens to checksum/constraint status.

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_HYBRID0_RUN_SPEC_2026-06-01.md`

Core design:

```text
primary payload:
  continuous sparse k-space coefficient geometry
  sin/cos phase channels
  residual coefficient geometry
  semantic motif header

side channel:
  magnitude token
  phase token
  residual token
  support/rank token
  checksum agreement
```

The decoder reconstructs from continuous coefficients first:

$$
\hat{p} =
\Re\left(\mathcal{F}^{-1}_2(\hat{F})\right)
$$

Then token states check consistency:

$$
\Pr[q(\hat{F}_i)=c_i]
$$

Primary question:

> does continuous D-LinOSS geometry carry stronger recoverable dependency than hard token labels?

First gate:

- mode: `--aq-hybrid-0`
- noise: `0.00`
- seed: `11`
- controller: `free`
- fixed core: `K=6`
- residual: `residual_K=2`
- no alpha sweep
- no noise sweep
- no larger payload

Required comparison:

- continuous MI gap versus token MI gap
- packet-window consistency versus isolated packet consistency
- realized PSNR versus AQ-FFT-1/AQ-DNA-0 plateau of about `17.0 dB`

Decision rule:

- if continuous MI beats token MI and PSNR rises, continue hybrid geometry-first encoding
- if continuous MI beats token MI but PSNR does not rise, keep the hybrid direction and improve decoder/window modeling
- if neither continuous MI nor window consistency beats null, locality is still not structurally active in the transport path

Status update:

- AQ-HYBRID-0 is useful as a precursor idea, but it is no longer the immediate next run.
- The next run should test whether coefficient packets need a hierarchical tensor representation rather than another flat-sequence or checksum variant.

## Program AS: Hierarchical Instruction Genome Transport

Program AS keeps the AQ-FFT-1 instruction packets but changes their geometry:

```text
instruction genome
  -> TTN fold encoder
  -> root tensor + selected internal bonds + charge sectors
  -> D-LinOSS transport
  -> TTN unfold decoder
  -> coefficient field
  -> inverse FFT patch
```

The conceptual shift is:

```text
from: transport tokens, decode shape
to: fold tokens into shape, transport shape, unfold tokens/field
```

AS-0 starts with a binary TTN rather than a full MERA. TTN is cheaper, easier to inspect, and sufficient for the first ceiling gate.

Payload:

- semantic header
- fixed k-space core
- sparse residual instructions
- phase-safe `sin(phi), cos(phi)` channels
- checksum

TTN leaf tokens:

- coefficient packets
- residual packets

Internal TTN tensors encode:

- local packet correlations
- local packet differences
- local packet products

The root tensor is the folded semantic shape / channel state.

Each internal bond also carries a charge vector:

| charge | meaning |
| --- | --- |
| `Q_DC` | luminance/DC charge |
| `Q_lowfreq_energy` | low-frequency energy |
| `Q_phase_coherence` | phase coherence |
| `Q_residual_support` | sparse residual support |
| `Q_motif_consistency` | semantic motif consistency |
| `Q_checksum_parity` | checksum parity |

The TTN tests an approximate conservation rule:

```math
Q_{parent} \approx Q_{left} + Q_{right}
```

This is the practical multi-scale gauge: the transported object is not every packet independently, but a compressed folded state with selected bond messages.

## AS-0 Run Spec

Mode:

```bash
--as-0
```

Runtime:

- noise: `0.00`
- seed: `11`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- depth: `3`
- no alpha sweep
- no noise sweep
- no larger payload

Comparison:

- flat AQ-FFT-1 free baseline
- TTN-folded AQ-FFT genome

Baseline:

- flat realized PSNR: `~17.00 dB`
- flat oracle PSNR: `~25.05 dB`

Primary metrics:

- `fft_patch_psnr`
- `fft_oracle_psnr`
- `as_root_tensor_cosine`
- `as_charge_conservation_error`
- `as_charge_cosine`
- `as_unfolded_coeff_psnr`
- `as_ttn_lift_over_flat_psnr`
- `fft_coefficient_cosine`
- `fft_coeff_mag_cosine`
- `fft_coeff_real_cosine`
- `fft_coeff_imag_cosine`
- `fft_phase_recovery`
- `fft_topk_energy_recall`
- semantic header / motif recovery

Success condition:

- semantic header remains `1.0000`
- TTN realized PSNR exceeds the flat `~17.0 dB` baseline
- charge recovery remains coherent
- coefficient recovery improves enough to explain the PSNR lift

Interpretation:

- if TTN improves PSNR or coefficient recovery, the missing ingredient was multi-scale shape encoding rather than another controller
- if TTN preserves root/charge structure but PSNR stays flat, the unfold decoder is the likely bottleneck
- if root and charge recovery fail, the TTN representation is too compressed or poorly aligned with D-LinOSS signatures

Next step only if AS-0 helps:

- compare TTN against a small MERA-style disentangler
- use disentanglers to suppress local packet redundancy before coarse-graining
- keep runtime compact before adding noise or controllers

## AS-0 and AS-0b Results

Run record/spec:

- `emergent_quantum_geometries/docs/PROGRAM_AS0_RUN_SPEC_2026-06-01.md`

AS-0 result:

- `ifft_psnr = 16.9995 dB`
- `oracle_psnr = 25.0480 dB`
- `as_ttn_lift_over_flat_psnr ~= 0.0000 dB`
- `as_root_tensor_cosine = 0.9518`
- `as_charge_cosine = 0.9962`
- `as_charge_conservation_error = 0.1877`
- `coeff_imag_cos = 0.6284`
- motif/header recovery stayed `1.0000`

AS-0b corrected the decoder path so reconstruction actually used the recovered TTN state:

```text
noisy D-LinOSS signature
  -> recovered TTN root/bonds/charges
  -> TTN unfold decoder
  -> sparse FFT coefficients
  -> inverse FFT patch
```

AS-0b result:

- `ifft_psnr = 16.9932 dB`
- `flat_psnr = 16.9995 dB`
- `ttn_lift = -0.0063 dB`
- `root_cos = 0.9518`
- `charge_cos = 0.9962`
- `charge_err = 0.1877`
- `coeff_imag_cos = 0.6280`
- motif/header recovery stayed `1.0000`

Methodological conclusion:

- Program AS worked operationally: TTN root state, bond/charge summaries, and semantic headers all remained recoverable
- Program AS did not work as a reconstruction improvement: neither AS-0 nor AS-0b lifted realized PSNR above the flat AQ-FFT-1 baseline
- the corrected AS-0b result matters most because it rules out the simple decoder-wiring explanation
- the bottleneck is still slotwise residual instruction realization, especially the weak imaginary/residual coefficient channel
- TTN folding therefore appears transport-compatible but not reconstructively sufficient in the current AQ-FFT regime

Updated decision:

- do not treat TTN/MERA depth as the immediate next lever
- do not run a MERA follow-up just because AS-0 was implemented
- keep hierarchical folded states as a useful analysis representation, but shift the next correction back to coefficient-body representation and decoder design

What now looks solid:

- semantic/header transport is stable
- global folded summaries can survive transport
- the oracle-to-realized gap is not mainly a root/charge preservation problem

What did not work:

- hierarchical folding did not improve support overlap, energy recall, or realized PSNR
- unfolding the TTN state did not recover more fine coefficient detail than the direct flat decoder

## AQ-FFT-3 Direct Complex Oscillator Codec

The next correction is at the encoding level.

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT3_RUN_SPEC_2026-06-02.md`

The repeated AQ-FFT/AS failure pattern is:

| metric | value |
| --- | ---: |
| `coeff_real_cos` | `~0.861` |
| `coeff_imag_cos` | `~0.628` |
| `phase_cos` | `~0.893` |
| realized PSNR | `~17.0 dB` |
| oracle PSNR | `~25.05 dB` |

The likely representational bug is that AQ-FFT-1 split a complex coefficient into separate magnitude and phase channels:

```math
F_k = |F_k|e^{i\phi_k}
```

and decoded the imaginary component as:

```math
\operatorname{Im}(\hat{F}_k)=|\hat{F}_k|\sin(\hat{\phi}_k)
```

If magnitude and phase errors are correlated, this product can make the imaginary channel worse than either recovered component alone. AQ-FFT-3 therefore injects the complex coefficient directly into D-LinOSS:

```math
x_0[k] = F_k
```

and recovers:

```math
\hat{F}_k = \hat{x}[k]
```

First gate:

- mode: `--aq-fft-3`
- noise: `0.00`
- seed: `11`
- controller comparison: `free`, `tunnel_eigen_residual_alpha065`
- fixed core: `K=6`
- residual: `residual_K=2`
- no noise sweep
- no larger payload

Success target:

- lift `coeff_imag_cosine` above the AQ-FFT-1/AS plateau of `~0.628`
- lift realized PSNR above the flat AQ-FFT-1 baseline of `~17.0 dB`
- preserve motif/header recovery

Interpretation:

- if imaginary recovery improves, the magnitude/phase split was the main representational bug
- if imaginary recovery does not improve, inspect D-LinOSS damping and signature extraction for asymmetric loss of imaginary components
- use QMI/MPS diagnostics from Sanz Larrarte et al. to locate where magnitude-phase or real-imaginary dependency is lost, not as another transport representation

AQ-FFT-3 result:

- run record: `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT3_REMOTE_ANALYSIS_2026-06-02.md`
- `free` reached `ifft_psnr = 20.6478 dB`
- `alpha065` reached `ifft_psnr = 18.2753 dB`
- `oracle_psnr` stayed `25.0480 dB`
- motif/header recovery stayed `1.0000`
- `coeff_real_cos` stayed strong at about `0.87`
- `coeff_imag_cos` fell to about `0.51`
- `phase_cos` fell to about `0.49`

Methodological conclusion:

- AQ-FFT-3 is a real positive result for reconstruction
- it shows that direct complex injection improves realized reconstruction much more than AQ-FFT-1 did
- so the old magnitude/phase split really was harming the unfold path

But AQ-FFT-3 is not yet a full complex-field win:

- the imaginary channel stayed weak
- phase fidelity dropped sharply
- `free` again beat `alpha065` at the noise-free ceiling

The mathematical reason PSNR can jump while the imaginary channel remains weak is that PSNR is dominated by total coefficient energy error:

```math
\mathrm{MSE}(p,\hat{p}) \propto \sum_k |F_k-\hat{F}_k|^2
```

In these tiny `4 x 4` patches, most image energy lives in the DC and low-frequency core. If AQ-FFT-3 improves those dominant components, especially the real-dominant low-frequency directions, PSNR can improve substantially even while lower-energy quadrature structure remains poorly preserved.

So AQ-FFT-3 improved transport of the coefficient body in an energy-weighted reconstruction sense, but did not yet preserve the full complex orientation of the retained modes.

Updated read:

- AQ is moving closer to reconstruction of the unfolded semantic space
- the folded semantic/header state is stable
- the coefficient body is now reconstructable enough to materially lift image-domain PSNR
- the next bottleneck is preserving complex orientation, not merely coefficient support or semantic identity

The practical implication is important:

> AQ is no longer only showing that a folded semantic state survives transport. It is now showing that the transported folded state can support a stronger partial reconstruction of the unfolded field.

## Phase 0: AP-U Calibration

AP-U completed on `2026-05-29` and passed after key-locking the AP-T comparator path to the AP-S reference recovery slot.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AP_U_REMOTE_ANALYSIS_2026-05-29.md`

Calibration outcome:

- `APR_reference_lock` and `APR_APT_path` matched exactly on all `6/6` cells.
- The earlier AP-T drift was caused by recovery key/fold-in construction.
- Stage 1 did not promote an adaptive controller.
- Fixed `alpha=0.65` remains the AQ-0 default controller.

If AP-U must be rerun before a future AQ launch, use the small AP-U reference gate:

- `APR_reference`
- `alpha065`
- `adaptive_v2`
- noise `0.30`, `0.45`
- seeds `11`, `29`, `47`

Hard stop rule:

- compare APR at `noise=0.30` to the reference margins
- seed `11`: `+0.0803`
- seed `29`: `+0.0762`
- seed `47`: `+0.1007`
- stop if any absolute delta exceeds `0.020`
- do not run AQ if this gate fails

Purpose:

- confirm the reference path is stable
- prevent variant promotion against a shifted baseline
- stop early if comparator drift is detected

## Semantic Motif Alphabet

AQ uses an explicit `8`-motif semantic alphabet. Each motif must have a distinct transition graph and recurrence signature so layer-wise recovery remains interpretable.

| motif | purpose |
| --- | --- |
| `cyclic` | full orbit structure |
| `fractal_nested` | hierarchical recurrence using recursive reuse, e.g. `A B A C A B A D` |
| `chirp` | frequency sweep / phase drift |
| `braided` | out-of-order continuity |
| `palindrome` | mirror symmetry |
| `sparse` | rare-event transport |
| `bifurcating` | two-attractor structure |
| `alternating` | asymmetric dwell alternating, e.g. `A A B A A B B A` |

## Layered Encoding

Motif cleanup before coding:

- `fractal_nested` must be genuinely hierarchical, not merely longer periodic. It should reuse a lower-order `A B A *` unit while changing the higher-order branch token.
- `alternating` should not be plain `A B A B`. Use asymmetric dwell alternation so it differs from cyclic motifs in occupation statistics and transition timing.
- These two motifs should remain distinguishable across transition graph, recurrence structure, FFT/phase signature, and cross-motif correlation.

AQ-0 uses four locked semantic layers:

| layer | encodes | expected noise robustness |
| --- | --- | --- |
| `L1` | symbol occupation statistics | strongest |
| `L2` | transition graph | strong/moderate |
| `L3` | FFT/phase signature | moderate/fragile |
| `L4` | cross-motif correlation / recurrence topology | most fragile |

Prediction:

- `noise=0.00`: `L1`-`L4` should reconstruct well
- `noise=0.30`: `L1`-`L3` should mostly survive
- `noise=0.45`: `L1`-`L2` / core topology should survive while `L3`-`L4` degrade

If layers do not degrade in that order, AQ has learned something real about the transport channel rather than merely failing a scalar score.

## Phase 1: TPU Semantic Payload Transport

### AQ-0 fast proxy result

AQ-0 fast proxy completed on `2026-05-29` in `europe-west4-a` on `program-ap-eu-node-v2`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ0_REMOTE_ANALYSIS_2026-05-29.md`

TRC handling:

- used only the tiny final stdout table already observed from the TPU VM
- no result artifact or remote summary JSON was downloaded

Result:

| controller | positive cells | mean margin | min margin | mean noisy cosine | min noisy cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `free` | `1/2` | `+0.0139` | `-0.0064` | `0.8704` | `0.8020` |
| `tunnel_eigen_residual_alpha065` | `2/2` | `+0.0453` | `+0.0045` | `0.8125` | `0.7335` |

Interpretation:

- `alpha065` improved margin over `free` on both seeds.
- `alpha065` made the weak seed `29` positive.
- The seed `11` noisy cosine dropped to `0.7335`, which is below the desired `0.80` floor.
- Because this happened at `noise=0.00`, AQ-0 has not yet established a clean semantic ceiling.
- The proxy still used the legacy compact motif generator, so it should not be treated as the locked `8`-motif AQ ceiling.
- The leading hypothesis is controller-induced semantic reshaping: `alpha065` preserves coarse topology/margin while damaging fine layer geometry.

Decision:

- do not advance to AQ-1 noise yet
- run `AQ-0b` first with explicit layer-wise metrics and the untransported ridge sanity check

### AQ-0b next run

AQ-0b should stay compact and noise-free:

- noise: `0.00`
- depth: `3`
- seeds: `11`, `29`
- add seed `47` only if runtime remains tiny
- controllers: `free`, `tunnel_eigen_residual_alpha055`, `tunnel_eigen_residual_alpha065`
- hard controls: disabled
- no adaptive controller
- payload: explicit `8`-motif AQ alphabet

Required additions:

- report `L1` occupation recovery
- report `L2` transition graph recovery
- report `L3` FFT/phase recovery
- report `L4` recurrence/correlation recovery
- run ridge decoder on the untransported encoded payload before interpreting transport
- report aggregate cosine, semantic margin, and inter-layer leakage proxy

AQ-1 should remain blocked until AQ-0b proves the semantic layers are recoverable without transport noise.

Tuning rule:

- do not change D-LinOSS damping yet
- if `alpha065` preserves `L1`/`L2` but collapses `L3`/`L4`, then run a later micro-sweep over `projection_alpha = 0.55, 0.65`, `damping_spread = 0.30, 0.60`, and D-LinOSS steps `24, 36`
- if `L1`/`L2` fail at `noise=0.00`, prioritize encoder/decoder repair before controller tuning

### AQ-0b result

AQ-0b completed on `2026-05-29` in `europe-west4-a` on `program-aq0b-eu-node-v1`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ0B_REMOTE_ANALYSIS_2026-05-29.md`

TRC handling:

- used only the final stdout tables already inspected from the TPU VM
- no result artifact or remote summary JSON was downloaded
- the node was intentionally kept alive for follow-up experiments

Result:

| controller | positive cells | mean margin | min margin | mean noisy cosine | mean L2 | mean L3 | mean L4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `2/2` | `+0.0189` | `+0.0106` | `0.8593` | `0.8593` | `0.9364` | `0.9514` |
| `tunnel_eigen_residual_alpha055` | `2/2` | `+0.0089` | `+0.0044` | `0.8868` | `0.8868` | `0.9014` | `0.9439` |
| `tunnel_eigen_residual_alpha065` | `1/2` | `+0.0097` | `-0.0016` | `0.8562` | `0.8562` | `0.9248` | `0.9486` |

Interpretation:

- the explicit `8`-motif AQ payload is recoverable at `noise=0.00`
- untransported ridge sanity stayed strong across all controllers
- `L1`, `L3`, and `L4` did not collapse under `alpha065`
- the earlier AQ-0 proxy worry about fine-layer destruction was not confirmed
- the live decision has shifted from "repair the semantic ceiling" to "test which controller survives noise best"

Updated decision:

- AQ-1 is now unblocked
- skip damping micro-tuning for now
- run a compact `AQ-1a` first with `free`, `alpha055`, and `alpha065`

### AQ-1a-TF next run

AQ-1a-TF should stay compact while moving from static motif recovery to semantic transformation recovery. To stay under the runtime threshold, run it as a gate first:

- payload: explicit `8`-motif AQ alphabet
- transformation pairs:
  - `cyclic -> chirp`
  - `palindrome -> braided`
  - `sparse -> bifurcating`
  - `fractal_nested -> alternating`
- noise: `0.30`, `0.45`
- depth: `3`
- seed: `11`
- controllers: `free`, `tunnel_eigen_residual_alpha055`, `tunnel_eigen_residual_alpha065`
- hard controls: disabled as experiment arms

The payload should encode motif-to-motif transformation fields rather than only static motifs. This is the synthetic version of `frame_t -> frame_t+1`:

```text
motif_i -> motif_j
```

Add `L5_transform`:

| layer | encodes | expected behavior |
| --- | --- | --- |
| `L5` | compact transformation operator between motif states | most fragile, most semantically meaningful |

The `L5` code is built from:

- `delta_L1`: occupation change
- `delta_L2`: transition-graph change
- `delta_L3`: FFT/phase change
- `delta_L4`: recurrence-topology change

Report:

- semantic margin
- aggregate cosine
- `L1` occupation recovery
- `L2` transition recovery
- `L3` FFT/phase recovery
- `L4` recurrence recovery
- `L5` transformation recovery
- inter-layer leakage proxy
- source motif recovery
- target motif recovery
- transform class recovery
- transformation exact match
- delta graph cosine
- delta phase cosine
- delta recurrence match

Promotion question:

- does `alpha055` or `alpha065` beat the free baseline once real transport noise is added, and if so, on which semantic layers and transformation metrics?

Primary transformation criterion:

```text
transform_exact_match =
  source motif correct
  AND target motif correct
  AND L5 transform class correct
```

Success shape:

- at `noise=0.30`, `L1`-`L3` should survive and `L5` should beat `free` for at least one tunnel-eigen controller
- at `noise=0.45`, `L1`-`L2` should survive and `L5` should retain partial transform-class recovery
- a coherent crossover would be `alpha055` winning `L2`/`L5` at `0.30` while `alpha065` wins transform topology at `0.45`

Follow-up rule:

- only run seed `29` after the seed `11` gate finishes cleanly and produces interpretable `L5` metrics
- do not launch the full two-seed matrix as a single job during constrained TPU windows

### AQ-1a-TF gate result

AQ-1a-TF seed-11 gate completed on `2026-05-29` in `europe-west4-a` on `program-aq0b-eu-node-v1`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ1A_TF_GATE_REMOTE_ANALYSIS_2026-05-29.md`
- `emergent_quantum_geometries/docs/PROGRAM_AQ1A_TF_S29_REMOTE_ANALYSIS_2026-05-29.md`

Seed-11 gate outcome:

- `L5` transformation recovery stayed strong at both `noise=0.30` and `0.45`.
- At `noise=0.30`, all controllers reached `transform_exact_match = 1.0000`.
- At `noise=0.45`, `free` and `alpha065` remained at `1.0000`, while `alpha055` dropped to `0.7500`.
- `alpha065` produced the strongest high-noise transform profile (`L5`, noisy cosine, and delta-graph/recurrence consistency).

Seed-29 follow-up outcome:

- `L5` remained measurable, but exact transform recovery did not generalize.
- Best seed-29 `transform_exact_match` was `0.5000`; several controller/noise cells fell to `0.0000`.
- `alpha065` preserved `L5 = 0.9148` at `noise=0.30`, but collapsed to `L5 = 0.5368` at `noise=0.45`.
- `free` was steadier at seed-29 high noise but still weak on exact transform recovery.

Revised interpretation:

- AQ-1a-TF established that an `L5` transformation layer is measurable.
- AQ-1a-TF did not prove stable operator transport.
- The likely failure mode is endpoint/label association: the decoder may be reading source/target correlation rather than a generative operator.

Runtime decision:

- do not widen AQ-1a-TF into a longer sequence-pair matrix
- move to AQ-1b so the next cycle tests operator-conditioned recovery directly

### AQ-1b next run

AQ-1b should change objective from sequence recovery to operator-conditioned recovery.

Payload:

- encode `source motif + compressed operator T`
- hold out direct target features from the transported payload

Decoder target:

- recover `source_hat`
- recover `operator_hat`
- derive `target_hat = operator_hat(source_hat)`

Required metrics:

- `operator_recovery`
- `derived_target_accuracy`
- `operator_application_error`
- `source_recovery`
- `target_direct_leakage_test`
- `operator_swap_consistency`
- `source_swap_consistency`
- `L1`-`L5` recovery

Primary KPI:

- `derived_target_accuracy`, not direct target recovery

Pre-launch AQ-1a-TF diagnostics:

- estimate `MI(L5; source)`
- estimate `MI(L5; target)`
- estimate `MI(L5; transform_class)`
- compute transform confusion matrices by seed, noise, and controller
- if `MI(L5; target) > MI(L5; transform_class)`, treat AQ-1a-TF as target leakage rather than operator transport

Anti-leak and action controls:

- target leakage test: target prediction should stay near chance without the operator channel
- operator swap test: holding source fixed and swapping operator should change the derived target
- source swap test: holding operator fixed and swapping source should change the derived target according to the same rule

L5 operator encoding:

- replace pure delta summaries with a compact operator code over semantic layers
- represent each transform class as an action over `L1`, `L2`, `L3`, and `L4`
- use a low-rank factorization of the layer action rather than directly transporting target features

Success criterion:

- `operator_recovery` above chance
- `derived_target_accuracy` above direct target leakage
- high operator-swap consistency
- high source-swap consistency
- direct target leakage near chance

Strong first-pass target:

- `derived_target_accuracy >= 0.75` at `noise=0.30`
- derived target accuracy remains above chance at `noise=0.45`
- target leakage remains near chance

Implementation status:

- `--aq-1b` is implemented in `program_ap_tunnel_eigen_recovery.py`
- `--aq-1b` is allowed by `run_program_ap_ready_node.ps1`
- AQ-1b clamps `sequence_length = 8` to protect the runtime budget
- AQ-1b completed and showed strong layer survival but weak compositional operator recovery

### AQ-1c next run

AQ-1c should not broaden AQ-1b. It should repair the operator grammar before more noisy transport.

Goal:

- prove the operator family is identifiable and reusable in the noise-free semantic code
- require source and operator recovery before interpreting derived target accuracy

Run shape:

- sources: `4`
- operators: `4`
- pairs: all `16` source/operator combinations
- noise: `0.00`
- depth: `3`
- controller: `free`
- seed: `11`
- hard controls: disabled

Gate criteria:

- `source_recovery >= 0.75`
- `operator_recovery >= 0.75`
- `derived_target_accuracy >= 0.75` only after source/operator gates pass
- target leakage should remain low

Implementation status:

- `--aq-1c` is implemented in `program_ap_tunnel_eigen_recovery.py`
- `--aq-1c` is allowed by `run_program_ap_ready_node.ps1`
- AQ-1c uses one-hot recovery states so source/operator exact-match metrics are interpretable

### AQ-1c result

AQ-1c completed on `2026-05-29`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ1C_REMOTE_ANALYSIS_2026-05-29.md`
- `emergent_quantum_geometries/docs/PROGRAM_AQ1C_RECALIBRATION_SUMMARY_2026-05-30.md`
- `emergent_quantum_geometries/docs/PROGRAM_AQ1C_SOURCE_RECOVERY_AND_SEQUENCE_METHODOLOGY_2026-05-30.md`

Outcome:

- source recovery: `0.5625`, below the `0.75` gate
- operator recovery: `0.4375`, below the `0.75` gate
- derived target accuracy: `0.2500`, below the `0.75` gate
- target leakage: `0.1875`
- `L5` recovery: `0.6242`

Decision:

- AQ-1c failed the noise-free operator identifiability gate
- do not run AQ-1d/noisy transport yet
- repair source/operator separability and add confusion tables before the next TPU cycle

Course correction implemented:

- source recovery now decodes an explicit source semantic code instead of only the original motif feature vector
- AQ-1c payload now keeps source and operator positions separated instead of blending them arithmetically
- operator codes are more orthogonal through stronger one-hot and marker components
- summary metrics now include `operator_invariance_score`, `source_confusion`, and `operator_confusion`
- rerun `--aq-1c` before any noisy operator transport

Follow-up AQ-1c probes on `2026-05-30` showed that the source/operator marker direction is unstable:

| variant | source recovery | operator recovery | derived target | target leakage | L5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline AQ-1c | `0.5625` | `0.4375` | `0.2500` | `0.1875` | `0.6242` |
| source-marker repair | `0.6875` | `0.6875` | `0.5625` | `0.1875` | `0.7290` |
| operator-suffix strengthening | `0.3750` | `0.3750` | `0.2500` | `0.1875` | `0.6040` |

The source-marker repair was the best balanced result, but it still failed the `0.75` gate. The operator-suffix strengthening regressed sharply.

Post-run code review found the root cause: AQ-1c had stopped transporting source motifs. The payload used marker halves:

```python
payload_seqs.append(
    jnp.concatenate([source_marker_full[:split], marker_full[split:]], axis=0)
)
```

Therefore D-LinOSS processed marker interval statistics while the decoder was asked to recover source semantic layers computed from different motif sequences. The correct AQ-1c payload must keep the actual source motif in the transported sequence:

```python
payload_seqs.append(seqs[src_idx])
```

The operator belongs in the feature/action side, not as a replacement for semantic content:

```math
z = E(s; C_T)
```

where `s` is the source motif and `C_T` is the operator action code.

The previous failure mode can be written as:

```math
\max I(z; m_T) \not\Rightarrow \max I(z; T \mid s)
```

but the deeper bug was:

```math
z = E(m_s, m_T) \neq E(s, T)
```

where `m_s` and `m_T` are hand-crafted markers.

Decision:

- stop the current AQ-1c branch
- do not run AQ-1d/noisy transport yet
- do not tune alpha or add hard controls
- fix AQ-1c so `payload_seqs` contains the real source motif
- keep operator information in `pair_features` / `L5_transform`
- add a pre-run sanity print proving `payload_seqs[0]` matches the source motif
- rerun AQ-1c only after this sanity check passes

Corrected AQ-1c source-transport result:

| metric | value |
| --- | ---: |
| payload/source match | `True` |
| source recovery | `1.0000` |
| operator recovery | `0.2500` |
| derived target accuracy | `0.2500` |
| target leakage | `0.1875` |
| noisy graph cosine | `0.9106` |
| L1 occupation recovery | `0.9850` |
| L2 transition recovery | `0.9106` |
| L3 FFT/phase recovery | `0.9445` |
| L4 recurrence recovery | `0.9217` |
| L5 operator recovery cosine | `0.5272` |

The decoded readout showed perfect source identity recovery on all `16` source/operator cells, while operator decoding collapsed mostly to `SYMBOL_SHIFT`. This reframes AQ-1c:

- source semantic transport is working in the noise-free gate
- operator derivation is not yet configured
- the next branch should test sequential semantic state signaling before arbitrary operator labels

The source transport claim is:

```math
s \xrightarrow{E} z \xrightarrow{\mathrm{transport}} \tilde{z} \xrightarrow{D} \hat{s},
\qquad
\Pr[\hat{s}=s] = 1.0000
```

The failed operator claim is:

```math
\Pr[\hat{T}=T] = 0.2500,
\qquad
\Pr[\hat{T}(\hat{s})=T(s)] = 0.2500
```

So the obstacle is operator fidelity, not source recovery.

### AQ-SEQ-0 result

AQ-SEQ-0 was run after corrected AQ-1c to test sequential semantic signaling.

Goal:

- verify that recovered semantic states preserve ordered changes through the folded field
- avoid label-conditioned target derivation until lawful waveform operators are defined

Define the semantic state:

```math
S_t =
\left[
L_1(s_t),
L_2(s_t),
L_3(s_t),
L_4(s_t)
\right]
```

Recover each transported state:

```math
s_t
\xrightarrow{E}
z_t
\xrightarrow{\mathrm{transport}}
\tilde{z}_t
\xrightarrow{D}
\hat{S}_t
```

Then measure whether semantic increments survive:

```math
\Delta S_t = S_{t+1} - S_t,
\qquad
\Delta \hat{S}_t = \hat{S}_{t+1} - \hat{S}_t
```

Primary metric:

```math
\mathrm{SeqFid} =
\cos(\Delta \hat{S}_t,\Delta S_t)
```

Compact gate used:

- noise: `0.00`
- depth: `3`
- seed: `11`
- controller: `free`
- payload: real motif sequence only
- sequence length: `8`
- sources: `cyclic`, `fractal_nested`, `palindrome`, `sparse`
- report source recovery, `L1`-`L4`, decoded state readout, sequential delta cosine, transition confusion

The final transition-shape gate used:

```text
L5_transform = target_state_layers + delta_layers
```

Observed result:

| metric | value |
| --- | ---: |
| source recovery | `1.0000` |
| derived next accuracy | `1.0000` |
| transition class recovery | `1.0000` |
| L5 layer recovery | `0.8045` |
| direct target recovery | `0.7500` |
| exact direct match | `0.7500` |
| leakage proxy | `0.0460` |

Interpretation:

- the tensor network is reliably recovering the source semantic shape
- the transition gate can derive the next semantic state from recovered source state
- direct next-state decoding is weaker than derived next-state reconstruction
- this is a promising mode for sequential signaling, not yet a full media reconstruction system

The reconstruction path should be:

```math
s_t \xrightarrow{E} z_t \xrightarrow{\mathrm{transport}} \tilde{z}_t \xrightarrow{D} \hat{S}_t,
\qquad
\hat{S}_{t+1} \approx G_{\mathrm{seq}}(\hat{S}_t)
```

Next engineering step:

- save compact recovered semantic-state outputs from successful AQ source/sequence gates
- keep TPU runs focused on transport and recovered-state signatures
- move reconstruction heuristics, operator-space detection, and media decoding into a separate layer

For concrete media targets such as movie frames with audio channels, AQ still needs a richer semantic encoder and modality-aware decoder. The current result supports the prerequisite claim: the folded semantic shape and a simple sequential relation can be recovered.

### AQ-IMG-0 next run

AQ-IMG-0 is the first concrete bridge from abstract semantic motifs to image-domain semantic reconstruction.

Goal:

- encode a tiny image-like field as semantic patch states
- transport the patch semantic states through D-LinOSS
- decode recovered patch features back into grayscale pixels
- measure whether reconstruction beats a motif-only baseline

Scope:

- image size: `4 x 4` patch grid
- patch size: `4 x 4` grayscale pixels
- patch states: `16`
- vocabulary: `8` image-domain motifs
- noise: `0.00`
- controller: `free`
- depth: `3`

Patch vocabulary:

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

Encoding:

```math
p_i \rightarrow (m_i, r_i)
```

where \(p_i\) is a patch, \(m_i\) is the nearest image-domain motif, and \(r_i\) is a scalar residual.

TPU-side payload:

```math
z_i = E(m_i)
```

The transported sequence remains the semantic patch motif. The residual and patch position are carried as compact feature geometry:

```math
L_5 = [r_i, \mathrm{pos}_i]
```

Decoder metrics:

- motif recovery
- ridge pixel PSNR
- motif-catalog-plus-residual PSNR
- residual MAE
- patch luminance MAE
- per-patch true/predicted motif readout

This tests the small end-to-end chain:

```text
patch -> motif + residual -> D-LinOSS semantic transport -> recovered feature vector -> pixel reconstruction
```

It does not claim full image or video reconstruction. It is the smallest useful test of whether recovered semantic patch states can support pixel-level reconstruction.

### AQ-IMG-0 first result

AQ-IMG-0 completed on `2026-05-30` in `europe-west4-a` on `program-aqimg-eu-node-v1`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG0_REMOTE_ANALYSIS_2026-05-30.md`

TRC-safe handling:

- result was read from TPU stdout only
- no result artifact, checkpoint, array, directory, or summary JSON was downloaded

Run shape:

- mode: `--aq-img-0`
- noise: `0.00`
- seed: `11`
- controller: `free`
- sequence length: `16`
- interval nodes: `136`
- patch states: `16`
- patch grid: `4 x 4`
- patch size: `4 x 4`

Core result:

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
| motif recovery | `1.0000` |
| target recovery | `1.0000` |
| class recovery | `1.0000` |
| pixel PSNR | `16.8927 dB` |
| catalog PSNR | `35.9693 dB` |
| residual MAE | `0.0150` |
| luma MAE | `0.0252` |

The per-patch readout showed all `16/16` image motifs were decoded correctly. Predicted luminance was close for most patches, but some residual/luma errors remained. Examples:

| patch | motif | target luma | predicted luma |
| ---: | --- | ---: | ---: |
| `0` | `flat` | `0.0000` | `0.0050` |
| `1` | `periodic` | `0.3733` | `0.4217` |
| `9` | `periodic` | `0.3027` | `0.4217` |
| `10` | `diagonal` | `0.5884` | `0.5644` |
| `15` | `noise_burst` | `0.4412` | `0.4546` |

Interpretation:

- the image-domain semantic vocabulary is recoverable
- the patch motif identity channel works at `16/16`
- layer recovery remains strong enough to support a reconstruction decoder
- pixel reconstruction is only moderate at `16.8927 dB`, so the remaining work is residual/pixel reconstruction quality, not motif recovery
- the reconstruction limit may come from the encoding methodology as much as the decoder: the current patch code uses one scalar residual, which is too coarse for within-class phase, contrast, duty-cycle, and local amplitude variation
- the run was slower than AQ-SEQ-0 because `sequence_length=16` produced `136` interval nodes instead of `36`

Current claim boundary:

- supported: folded image-domain semantic patch identities and layer signatures can be transported and recovered in the noise-free gate
- not yet supported: high-fidelity arbitrary pixel reconstruction from the transported folded state

Next AQ-IMG direction:

- keep the `4 x 4` patch grid and `8` image motifs
- do not broaden to larger images yet
- replace the scalar residual with a tiny residual vector: mean, contrast, phase/alignment, and one low-frequency DCT-like coefficient
- require motif recovery to remain `1.0000`
- require pixel PSNR to improve over `16.8927 dB`

Runtime correction after aborted AQ-IMG-1:

- do not run full AQ-IMG-1 as the first vector-residual probe
- use `--aq-img-1-lite` first
- `AQ-IMG-1-LITE` uses `8` representative patches instead of `16`
- it reduces interval nodes from `136` to `36`
- it keeps D-LinOSS steps at `16`
- it uses two training exposures over the `8` patch states
- it keeps noise at `0.00` and controller `free`
- it has a launcher timeout guard of `900s` by default

Gate rule:

- if `AQ-IMG-1-LITE` does not finish quickly or does not beat AQ-IMG-0 pixel PSNR directionally, do not launch full AQ-IMG-1
- if it keeps motif recovery at `1.0000` and improves residual/pixel reconstruction, then run full `16` patch AQ-IMG-1

### AQ-IMG-1-LITE result

AQ-IMG-1-LITE completed on `2026-05-31` in `europe-west4-a` on `program-aqimg-eu-node-v1`.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG1_LITE_REMOTE_ANALYSIS_2026-05-31.md`

TRC-safe handling:

- result was read from TPU stdout only
- no result artifact, checkpoint, array, directory, or summary JSON was downloaded

Important runtime note:

- the first AQ-IMG-1-LITE attempt still fanned out across many controllers and hit the timeout guard during compilation
- the corrected AQ-IMG-1-LITE run restricted the probe to `free` only and skipped heavy comparator signatures
- the corrected lite run finished and is the only interpretable AQ-IMG-1-LITE result

Corrected lite result:

| metric | value |
| --- | ---: |
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

Patch collapse pattern from stdout:

- `periodic`, `gradient`, `diagonal`, and `checkerboard` collapsed toward `v_edge`
- `h_edge` collapsed toward `flat`
- only `flat`, `v_edge`, and `noise_burst` survived cleanly

Interpretation:

- the runtime correction worked: AQ-IMG-1-LITE now answers quickly enough to use as a gate
- the encoding change did not work: vector-residual lite degraded both motif recovery and pixel reconstruction relative to AQ-IMG-0
- AQ-IMG-1-LITE should not be interpreted through `margin`, `block_lift`, or `topology_lift`; those were neutralized by construction in the fast probe
- the most likely failure mode is decoder conditioning, not transport failure

Why the lite gate underperformed:

- it used one exemplar per motif instead of repeated motif classes
- it widened `source_features` and `transform_code` from scalar residual to a five-component residual vector without adding richer transport states
- the ridge decoder therefore had to solve a wider joint target from a much thinner class-conditioned dataset

Practical conclusion:

- AQ-IMG-0 remains the stable image-domain baseline
- do not promote the current AQ-IMG-1 vector-residual branch
- the next fair comparison should hold the patch roster constant and compare scalar-residual versus vector-residual coding on the same repeated small patch set

### AQ-FFT-0 k-space codec gate

The next representational correction is to stop asking a symbolic residual grammar to behave like a media codec.

AQ-IMG showed that motif identity is recoverable, but pixel reconstruction is bottlenecked by the residual code. The more natural representation for recoverable image content is k-space / FFT coefficients:

$$
p \leftrightarrow F = \mathcal{F}_2(p)
$$

D-LinOSS already evolves complex oscillator amplitudes and phases:

$$
\dot{x}(t) = A x(t),
\qquad
A_{ii} = -\gamma_i + i \omega_i.
$$

So the next compact gate should transport the frequency-domain object directly instead of routing image content through a symbolic residual.

Implemented mode:

- `--aq-fft-0`
- `--aq-fft-top-k` (default `6`)

Payload:

$$
[\text{motif header}, \operatorname{Re}(\tilde{F}), \operatorname{Im}(\tilde{F}), \Omega_K]
$$

where \(\tilde{F}\) keeps only the top-\(K\) coefficients and \(\Omega_K\) is the frequency-position mask.

Run shape:

- patch grid: `4 x 4`
- patch size: `4 x 4`
- patch count: `16`
- noise: `0.00`
- seed: `11`
- controllers: `free`, `tunnel_eigen_residual_alpha065`

Primary metrics:

- `fft_coefficient_cosine`
- `fft_frequency_index_recovery`
- `fft_phase_recovery`
- `fft_patch_psnr`
- `fft_oracle_psnr`
- motif recovery as a secondary interpretability check

Success criteria:

- beat AQ-IMG-1-LITE PSNR (`9.7162 dB`)
- ideally approach or exceed `22-25 dB` with `K=6`
- keep motif recovery from collapsing into one prototype basin
- show whether alpha065 helps or distorts k-space reconstruction at the noise-free ceiling

Design note:

AQ-FFT-0 is still a hybrid gate: it keeps the motif header and uses the existing folded recovery infrastructure. A later structural redesign can make kinematic nodes represent frequency positions directly rather than symbolic sequence intervals. That redesign should wait until the hybrid top-\(K\) coefficient path proves that k-space payloads improve reconstruction.

Future operator work should move to waveform/layer operators rather than arbitrary class operators:

- null subtraction: \(s - P_{\mathrm{null}}s\)
- nodal subtraction: \(s - P_{\mathrm{node}}s\)
- harmonic amplification: \(F^{-1}G_\omega F s\)
- graph diffusion: \(e^{-\tau L}s\)
- recurrence filtering: \(R_\lambda s\)

Only after sequential signaling works should AQ return to the stronger operator claim:

> The transported folded state contains a recoverable transformation operator that can unfold a second semantic state at recovery.

First AQ TPU pass:

- payload: `8`-motif semantic alphabet
- encoding: layered MPS / folded semantic graph
- bond dimension: `4`
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- depth: `3`
- seeds: `11`, `29`, `47`
- noise: `0.00`
- hard controls: disabled

Optional comparison arm, only if runtime remains compact:

- `tunnel_eigen_residual_adaptive_v2_shifted`

Do not include `adaptive_v1` in AQ-0. AP-U showed that it can win some moderate-noise cells but failed the `6/6` positive-margin robustness rule at high noise.

Do not include hard controls in AQ-0. The first ceiling test should compare encoder/decoder recoverability against the free transport baseline. Hard controls belong in AQ-1 after the noise-free ceiling is established.

The `noise=0.00` run is mandatory. It establishes the ceiling:

- if fidelity fails at `0.00`, the failure is in the encoder or decoder
- if fidelity succeeds at `0.00` and degrades at `0.30` or `0.45`, the failure is transport robustness

Run `0.30` and `0.45` only in AQ-1, after AQ-0 proves the noise-free ceiling.

TPU should save only compact artifacts:

- compressed signatures
- recovered state vectors
- layer-wise phase structure
- controller metadata
- hard-control outputs

Do not run heavy decoder comparisons on TPU.

## AQ-FFT-4: Complex Orientation Preservation Gate

AQ-FFT-3 established a useful but incomplete result: direct complex oscillator injection raised realized PSNR while leaving the imaginary and phase channels weak. The next gate keeps the same codec capacity and asks whether the remaining gap is caused by failure to preserve complex orientation.

Run:

```bash
--aq-fft-4
```

Fixed envelope:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `free`, `complex_pair_norm`, `complex_phase_locked`

Do not add:

- larger `K`
- noise
- `alpha065`
- video

Mathematical target:

```math
F_k = a_k + ib_k
```

should survive as a coupled quadrature pair, not as scalar magnitude and scalar phase targets that can drift independently.

AQ-FFT-4 adds:

- `support_weighted_angle_error`
- `energy_weighted_complex_mse`
- `quadrature_energy_recall`
- `imaginary_energy_recall`
- `complex_pair_norm_error`
- `phase_error_by_coefficient_slot`

It also prints a PSNR error budget:

- `oracle_psnr`
- `true_support_pred_coeffs`
- `pred_support_true_coeffs`
- `true_real_pred_imag`
- `pred_real_true_imag`
- `final`

Decision rule:

- if `complex_pair_norm` improves phase and imaginary cosine, the issue was complex-orientation preservation in the transported signature
- if `complex_phase_locked` improves PSNR, the decoder was violating pair-norm geometry
- if neither improves, the likely bottleneck is D-LinOSS signature extraction or a need for a genuinely complex-equivariant decoder

## AQ-FFT-5: Gauge-Aligned Complex Decoding

AQ-FFT-4 established `complex_pair_norm` as the current reference codec:

- `ifft_psnr = 23.1818 dB`
- `oracle_psnr = 25.0480 dB`
- motif/header recovery = `1.0000`
- phase and imaginary cosine remained near `0.51`

AQ-FFT-5 asks whether the weak phase/imaginary scores are caused by a recoverable complex gauge rotation.

Oracle diagnostic:

```math
\theta^*=\arg\left(\sum_k F_{\mathrm{true},k}\overline{F_{\mathrm{pred},k}}\right)
```

Aligned field:

```math
\hat{F}^{aligned}_k=e^{i\theta^*}\hat{F}_k
```

Mode:

```bash
--aq-fft-5
```

Controllers:

- `complex_pair_norm`
- `complex_pair_norm_gaugefix`

New metrics:

- `global_phase_aligned_imag_cos`
- `global_phase_aligned_phase_cos`
- `per_slot_phase_aligned_coeff_cos`
- `complex_gauge_error`
- `PSNR_before_alignment`
- `PSNR_after_alignment`

Success rule:

- if aligned phase/imaginary metrics jump, Bob needs gauge fixing rather than more transport capacity
- if aligned metrics stay low, the quadrature information is degraded and the next step is a complex-equivariant decoder or BP/charge decoder

Do not increase `K`, add noise, or run video until this gate is resolved.

## AQ-FFT-6: Hermitian-Constrained Complex Decoder

AQ-FFT-5 ruled out a global gauge rotation as the cause of weak imaginary and phase cosine. AQ-FFT-6 tests whether local real-image FFT structure is the missing constraint.

For real-valued patches:

```math
F[-k] = \overline{F[k]}
```

Mode:

```bash
--aq-fft-6
```

Controllers:

- `complex_pair_norm`
- `complex_pair_norm_hermitian`

The Hermitian decoder projects:

```math
\hat{F}_{sym}[k] = \frac{1}{2}\left(\hat{F}[k] + \overline{\hat{F}[-k]}\right)
```

and forces self-conjugate slots real.

New metrics:

- `hermitian_error_before`
- `hermitian_error_after`
- `self_conjugate_imag_energy`
- `pairwise_conjugate_consistency`
- `phase_cos_energy_weighted`

Decision rule:

- if Hermitian error drops and PSNR stays high or improves, keep the constrained decoder
- if raw phase stays low but energy-weighted phase is high, the weak phase metric is mostly low-energy slot noise
- if Hermitian constraints do not help, move to a complex-equivariant decoder or BP/charge decoder before increasing residual payload

## AQ-FFT-7: Residual Capacity Scaling

AQ-FFT-6 showed that Hermitian mismatch is not the active bottleneck and that energy-weighted phase is already excellent. AQ-FFT-7 therefore promotes reconstruction and energy-weighted metrics over raw phase/imaginary cosine.

Mode:

```bash
--aq-fft-7
```

Run shape:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=4`
- controller: `complex_pair_norm`

Reference:

- residual_K=2 realized PSNR near `23.18 dB`
- residual_K=2 oracle PSNR near `25.05 dB`

Primary metrics:

- `ifft_psnr`
- `oracle_psnr`
- `phase_cos_energy_weighted`
- `energy_weighted_complex_mse`
- `topK_energy_recall`
- motif/header recovery

Decision rule:

- if realized PSNR follows the oracle upward, proceed to noise with the larger residual body
- if oracle rises but realized PSNR does not, decoder capacity is the next bottleneck

### AQ-FFT-7 Outcome

AQ-FFT-7 completed successfully with residual_K=4.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT7_REMOTE_ANALYSIS_2026-06-02.md`

Observed:

| codec | residual K | realized PSNR | oracle PSNR | motif/header | energy-weighted phase |
| --- | ---: | ---: | ---: | ---: | ---: |
| `complex_pair_norm` | `2` | `~23.18 dB` | `~25.05 dB` | `1.0000` | `~0.9992` |
| `complex_pair_norm` | `4` | `26.3560 dB` | `31.1144 dB` | `1.0000` | `0.9992` |

This is the current strongest AQ-FFT result.

The experimentally defensible claim is:

> In this simulator, D-LinOSS complex oscillator transport can preserve a folded semantic/header state while carrying enough continuous k-space coefficient structure to reconstruct an unfolded `4 x 4` image patch field. Reconstruction improves when sparse residual k-space capacity is increased, while semantic/header recovery remains perfect.

Important boundary:

- This supports simulated folded-state recovery in a quantum-like complex oscillator state space.
- It does not establish physical quantum teleportation.
- It does not yet establish noisy robustness.
- It is one-seed clean-ceiling evidence and should be repeated before stronger claims.

Next gate:

- AQ-FFT-8 should hold residual_K=4 fixed and add `noise=0.30`
- compare `free`, `alpha065`, and `shape_charge_control`

### Visual Recovery Preview

AQ image/FFT modes can now emit compact PNG contact sheets with:

- row 1: original target patches
- row 2: recovered patches after decoding/inverse FFT
- row 3: amplified absolute error

This is enabled with:

```text
--save-preview-png
```

For the ready-node launcher, pass it through without changing the TRC-safe launch path:

```powershell
$env:PROGRAM_AP_EXTRA_ARGS='--save-preview-png'
```

The previews are generated on the TPU VM in the normal result directory and are recorded in `program_ap_summary.json` under `preview_pngs`. They are intentionally tiny diagnostic artifacts, not bulk result folders.

### AQ-YINYANG-0 Candidate

A useful next visual target is an 8-bit alternating yin-yang sequence:

- generate a small grayscale yin-yang frame, initially `16 x 16`
- alternate yin/yang polarity across frames
- tile each frame into `4 x 4` patches
- encode each patch through the AQ-FFT complex k-space codec
- transport/fold/recover the patch field
- emit target/recovered/error PNG previews

This should be a separate mode rather than an alias of the existing patch roster. The scientific question is whether the current folded semantic/k-space codec can recover a coherent recognizable visual symbol under temporal polarity changes, not merely whether it can reconstruct the existing synthetic patch catalog.

## Phase 2: Offline CPU Decoding

First offline sanity check:

- run the ridge decoder on the untransported encoded payload
- if this fails, AQ-0 transport results are not interpretable because the semantic layers are not recoverable from the encoded state itself

Run decoder comparisons locally/offline:

- ridge decoder
- MPS reconstruction decoder
- tunnel-eigen unfolded decoder

Metrics:

- layer-wise cosine fidelity
- semantic graph recovery
- inter-layer leakage
- phase preservation
- motif confusion matrix
- worst-control margin

Layer-wise fidelity is the main addition. Aggregate noisy cosine has been hiding where recovery actually fails.

## Success Criteria

Noise-free ceiling:

- `noise=0.00` should show high layer-wise fidelity across most semantic layers
- `L1` occupation should recover strongly
- `L2` transition graph should recover strongly
- `L3` FFT/phase should recover moderate-high
- `L4` recurrence/correlation should be recoverable, but may be the weakest layer

Moderate-noise transport:

- at `noise=0.30`, `alpha065` should preserve semantic graph recovery above `free` and hard controls

High-noise degradation:

- at `noise=0.45`, core/topological layers should survive while fine layers degrade

Success statement:

> Folded semantic information can be transported as a compressed recoverable subspace, with degradation occurring layer-wise rather than uniformly.

## MPS Compression Knob

MPS gives AQ a real compression axis:

- bond dimension `2`
- bond dimension `4`
- bond dimension `8`

Primary tradeoff:

```text
compression strength vs recovery fidelity vs noise robustness
```

Do not sweep all bond dimensions in the first TPU pass. Start AQ-0 with `bond_dim=4`; `bond_dim=2` is likely too compressed for the `8`-motif payload and could make the noise-free ceiling fail before transport is being tested.

## IBM Bridge

Keep IBM-lite separate until AQ has a working semantic payload.

Later IBM-facing constraints:

- small semantic subspace
- low bond dimension
- nearest-neighbor / heavy-hex-compatible graph
- local `Z`, `ZZ`, and parity observables only
- residual tunnel-eigen decoder

The niche is semantic compression plus noisy transport plus adversarial recovery testing.

## Recommended Run Order

1. `AP-U`: completed; key-lock retained
2. `AQ-0`: semantic payload, noise-free ceiling with `bond_dim=4`
3. `AQ-0b`: explicit `8`-motif layer diagnostic with `free`, `alpha055`, and `alpha065`
4. `AQ-1a-TF`: compact noisy motif transformation-field transport with `free`, `alpha055`, and `alpha065`
5. `AQ-1b`: compact operator-conditioned folded recovery with target leakage and swap controls
6. `AQ-1c`: noise-free operator identifiability gate
7. `AQ-offline`: compare ridge, MPS, and tunnel-eigen unfolded decoders

Only after these pass should we schedule a larger robustness or durability run.

## AQ-YINYANG-1 Visual Codec Gate

Added 2026-06-03: `AQ-YINYANG-1` is the one-frame follow-up to the two-frame tai-chi-tu visual recovery run. It keeps the successful `complex_pair_norm` k-space path but removes the alternating second frame so that the next result can distinguish a sparse-codec ceiling from a transport/unfolding failure.

Expected interpretation is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_YINYANG1_EXPECTATIONS_2026-06-03.md`

The key gate is whether the recovered PNG tracks the sparse oracle. If oracle PSNR remains low and realized PSNR stays within about 2 dB of it, the visible blur belongs to the image semanticization layer. If oracle PSNR is high but realized PSNR remains low, the problem is in D-LinOSS recovery or coefficient unfolding.

## AQ-YINYANG Findings

Added 2026-06-03:

- `AQ-YINYANG-1` showed that the sparse `10 / 16` coefficient codec could not represent the crisp symbolic image well enough; `oracle_psnr = 17.0387 dB`
- `AQ-YINYANG-2` retained all `16 / 16` FFT coefficients per patch and raised `oracle_psnr` to `80.0000 dB`
- `AQ-YINYANG-2` still reconstructed at `ifft_psnr = 21.4153 dB`, with perfect semantic identity and high coefficient cosine recovery
- `AQ-YINYANG-3` added direct pixel-space ridge decoding and showed essentially no lift: `pixel_ridge_psnr = 21.4213 dB`

Detailed run note:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_YINYANG_FINDINGS_2026-06-03.md`

Current conclusion:

> One-frame image encoding is not the blocker once all patch coefficients are retained. The remaining blocker is deeper than a simple coefficient-vs-pixel loss mismatch: the transported folded complex state is close in direction, but not yet numerically calibrated or locally consistent enough for pixel-faithful reconstruction.

## AQ-YINYANG-4 Raw-State Visual Gate

Added 2026-06-03:

`AQ-YINYANG-4` tested a simpler visual sanity path by bypassing the FFT body and transporting raw `4 x 4` pixel patch states from the user-provided `taichi_16x16_frame_1.png`.

Observed:

- motif recovery: `1.0000`
- pixel PSNR: `30.0046 dB`
- luma MAE: `0.0123`
- noisy cosine: `0.9978`

The first preview export revealed a visualization bug: target and recovered PNGs were assembled in randomized test-sample order rather than raster patch order. The preview writer now uses `patch_true_indices` to reorder patches before frame assembly.

Corrected local images:

- `tpu_previews/yinyang_ref16_frame1_ordered/aq_frame_depth3_noise0.0_seed11_complex_pair_norm_taichi_16x16_frame_1_target.png`
- `tpu_previews/yinyang_ref16_frame1_ordered/aq_frame_depth3_noise0.0_seed11_complex_pair_norm_taichi_16x16_frame_1_recovered.png`
- `tpu_previews/yinyang_ref16_frame1_ordered/aq_frame_depth3_noise0.0_seed11_complex_pair_norm_taichi_16x16_frame_1_compare.png`

Interpretation:

> At the one-frame `16` patch-state scale, D-LinOSS can recover a raw patchwise image state strongly enough for a recognizable visual reconstruction. This validates the raw transport/recovery path, but it deliberately bypasses the harder folded k-space codec.

Follow-up completed:

- `taichi_16x16_frame_2.png` also passed the one-frame raw-state control
- motif recovery: `1.0000`
- pixel PSNR: `29.8073 dB`
- luma MAE: `0.0133`
- noisy cosine: `0.9963`

Ordered two-frame follow-up:

- input frames: `taichi_16x16_frame_1.png` and `taichi_16x16_frame_2.png`
- recovery mode: ordered frame/raster patch order
- patch states: `32`
- motif recovery: `1.0000`
- pixel PSNR: `25.0450 dB`
- luma MAE: `0.0194`
- noisy cosine: `0.9843`
- control margin: `+0.0533`

This means the ordered visual-sequence gate passed in the narrow sense that:

- the frame-ordering bug is fixed
- both recovered frames correspond to the intended target polarity sequence
- the failure mode is no longer patch scramble or frame confusion

What remains open is joint quality under a larger shared roster. The two-frame run is visibly worse than the two one-frame controls, so the next problem is roster-size / joint-decoding degradation, not basic raw-state transport.

Implementation note:

- raw `AQ-YINYANG-4` now marks itself as a `fast_codec_probe`
- future sanity runs should keep the PNG outputs while skipping the extra adversarial-control signature burden that slowed the previous image checks
