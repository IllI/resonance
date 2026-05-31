# Program AQ Plan - Semantic Subspace Transport

## Core Claim

Program AQ should not claim "teleport arbitrary meaning." The experimentally defensible claim is narrower:

> Encode a structured semantic payload into a compressed tensor/MPS-like subspace, transport it through the tunnel-eigen residual channel, and test whether Bob can reconstruct the semantic layers above free and hard-control baselines.

This aligns with the quantum-autoencoder analogy:

```text
compress -> transport -> decompress
```

The target is a semantically sufficient subspace, not the full Hilbert space.

## Runtime Principle

AQ is a rapid-prototyping line, not a durability run. Keep TPU runs small enough to turn over quickly:

- run a noise-free ceiling first
- keep AP-U calibration fail-fast
- save compact transport artifacts and metadata only
- move decoder comparisons to offline CPU
- avoid IBM-lite and new adversaries until the semantic payload works

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
