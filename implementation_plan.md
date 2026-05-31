# Implementation Plan - After Program AP-U

## Current status

AP-U completed on `2026-05-29` and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AP_U_REMOTE_ANALYSIS_2026-05-29.md`

AP-T had exposed a comparator problem: APR and `alpha065` changed behavior when moved into the AP-T controller ordering. AP-U confirmed that the drift came from recovery key/fold-in construction, not from a genuine controller effect.

The key-lock course correction made `APR_reference_lock` and `APR_APT_path` match exactly on all `6/6` Stage 0 calibration cells.

Stage 1 did not promote an adaptive controller. Fixed `alpha=0.65` remains the current operating point for AQ-0.

## AP-U design

Stage 0 runs first:

- `APR_reference_lock`
- `APR_APT_path`

Both originally used the same controller parameters but different explicit recovery key slots:

- AP-S-Fidelity APR slot: `key_ctrl_idx=1`
- AP-T APR slot: `key_ctrl_idx=0`

AP-U result:

- the AP-T slot reproduced the drift
- key-locking `APR_APT_path` to the AP-S slot fixed the drift
- all Stage 0 calibration cells matched exactly after key-locking

Stage 0 compares every APR margin against the AP-S-Fidelity APR reference:

| noise | seed 11 | seed 29 | seed 47 |
| --- | ---: | ---: | ---: |
| `0.30` | `+0.0803` | `+0.0762` | `+0.1007` |
| `0.45` | `+0.0258` | `+0.0183` | `+0.0287` |

Tolerance:

- `abs(new_margin - reference_margin) <= 0.003`

If Stage 0 fails, AP-U skips Stage 1 and writes the calibration deltas. Stage 0 now fails fast: the first out-of-tolerance comparator cell stops calibration immediately, because that is already enough to prove the baseline shifted.

## Stage 1 variants

Stage 1 runs only after Stage 0 passes:

- `tunnel_eigen_residual_alpha065`
- `tunnel_eigen_residual_adaptive`
- `tunnel_eigen_residual_adaptive_v2_shifted`

Adaptive rules:

```python
adaptive_v1 = 0.50 + 0.30 * noise
adaptive_v2 = 0.56 + 0.22 * noise
```

This tests whether a shifted adaptive rule can keep the high-noise gains without undershooting the moderate-noise projection strength.

AP-U Stage 1 result:

| controller | positive cells | mean margin | min margin | decision |
| --- | ---: | ---: | ---: | --- |
| `alpha065` | `6/6` | `+0.0668` | `+0.0411` | keep as current operating point |
| `adaptive_v1` | `5/6` | `+0.0644` | `-0.0031` | not promoted |
| `adaptive_v2_shifted` | `6/6` | `+0.0627` | `+0.0473` | stable comparison arm, not promoted |

Promotion conclusion:

- `adaptive_v1` failed robustness because it had one negative high-noise cell.
- `adaptive_v2_shifted` stayed positive and beat APR on `4/6`, but beat `alpha065` on only `2/6`.
- `alpha065` has the best overall/high-noise mean margin and remains the safest default.

## Debug metadata

Every cell records:

- resolved projection alpha
- seed, noise, and recovery fold-in
- explicit recovery key slot
- adversary fold-in labels
- block size and within-block-random flag
- scoring weights
- hard-control list
- min-lift aggregation rule
- top-k mode and chosen adaptive mode count
- damping spread
- git commit
- mode

This makes calibration metrics separate from promotion metrics and prevents accidental promotion against a shifted baseline.

## Current implementation

`program_ap_tunnel_eigen_recovery.py` includes:

- `--ap-u`
- explicit `key_ctrl_idx` support
- Stage 0 pass/fail gate
- Stage 0 fail-fast behavior
- `adaptive_v2_shifted`
- AP-S APR reference margins
- per-cell debug metadata

`run_program_ap_ready_node.ps1` includes:

- `--ap-u` in the allowed modes
- local git commit stamping into the remote run

## Launch note

AP-U is complete. Delete the active TPU after documentation only when no immediate follow-up experiment is queued; if the next compact AQ step is ready, reuse the node instead of churning it.

## Program AQ: Semantic Subspace Transport

After AP-U calibration, the next experiment should be Program AQ. AQ should not claim "teleport arbitrary meaning." The narrower and testable claim is:

> Encode a structured semantic payload into a compressed tensor/MPS-like subspace, transport it through the tunnel-eigen residual channel, and measure whether Bob can reconstruct semantic layers above free and hard-control baselines.

AQ should be built for rapid TPU turnover. The goal is not a big durability proof yet; it is to establish a noise-free fidelity ceiling and identify where semantic recovery fails.

## AQ-0 fast proxy result

AQ-0 fast proxy completed on `2026-05-29` and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ0_REMOTE_ANALYSIS_2026-05-29.md`

TRC handling:

- used only the tiny final stdout table already inspected from the TPU VM
- did not download the remote result artifact, summary JSON, checkpoints, arrays, or directories

Result:

| controller | positive cells | mean margin | min margin | mean noisy cosine | min noisy cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `free` | `1/2` | `+0.0139` | `-0.0064` | `0.8704` | `0.8020` |
| `tunnel_eigen_residual_alpha065` | `2/2` | `+0.0453` | `+0.0045` | `0.8125` | `0.7335` |

Readout:

- `alpha065` improved margin over `free` on both seeds.
- `alpha065` made the weak seed `29` positive.
- The `alpha065` seed `11` noisy cosine fell to `0.7335`, below the desired `0.80` floor.
- Because that happened at `noise=0.00`, the fast proxy did not establish a clean AQ semantic ceiling.
- The proxy still used the legacy compact motif generator, not the locked `8`-motif AQ alphabet.
- The leading hypothesis is controller-induced semantic reshaping: `alpha065` may improve coarse folded topology/margin while damaging fine semantic-layer geometry.

Decision:

- do not launch AQ-1 noisy transport yet
- run AQ-0b first, still noise-free, with the explicit `8`-motif payload, `alpha055`/`alpha065` ablation, `L1`-`L4` layer metrics, and the untransported ridge sanity check

AQ-0b controller list:

- `free`
- `tunnel_eigen_residual_alpha055`
- `tunnel_eigen_residual_alpha065`

AQ-0b diagnostic question:

- does `alpha065` preserve `L1`/`L2` coarse semantic topology while damaging `L3`/`L4` fine phase and recurrence structure?

D-LinOSS tuning should wait until AQ-0b identifies the damaged layer. If `L3`/`L4` collapse while `L1`/`L2` survive, run a later tiny damping/projection micro-sweep. If `L1`/`L2` fail at `noise=0.00`, repair the encoder/decoder path before tuning the controller.

## AQ-0b result

AQ-0b completed on `2026-05-29` and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ0B_REMOTE_ANALYSIS_2026-05-29.md`

TRC handling:

- used only the final stdout tables already inspected from the TPU VM
- did not download the remote result artifact, summary JSON, checkpoints, arrays, or directories
- kept the active Europe TPU node alive for follow-up experiments

Result:

| controller | positive cells | mean margin | min margin | mean noisy cosine | mean L2 | mean L3 | mean L4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `2/2` | `+0.0189` | `+0.0106` | `0.8593` | `0.8593` | `0.9364` | `0.9514` |
| `tunnel_eigen_residual_alpha055` | `2/2` | `+0.0089` | `+0.0044` | `0.8868` | `0.8868` | `0.9014` | `0.9439` |
| `tunnel_eigen_residual_alpha065` | `1/2` | `+0.0097` | `-0.0016` | `0.8562` | `0.8562` | `0.9248` | `0.9486` |

What AQ-0b settled:

- the explicit `8`-motif AQ payload is recoverable at `noise=0.00`
- the earlier proxy fear of `L3`/`L4` collapse under `alpha065` was not confirmed
- the semantic ceiling is good enough to move on
- the next question is controller behavior under noise, not encoder/decoder repair

Updated decision:

- AQ-1 is now unblocked
- do not spend the next TPU cycle on damping micro-tuning
- run `AQ-1a-TF` first as a seed `11` gate with the three-controller comparison, but extend the payload from static motifs to motif transformation pairs

AQ-1a-TF controller list:

- `free`
- `tunnel_eigen_residual_alpha055`
- `tunnel_eigen_residual_alpha065`

AQ-1a-TF payload:

- `cyclic -> chirp`
- `palindrome -> braided`
- `sparse -> bifurcating`
- `fractal_nested -> alternating`

AQ-1a-TF adds `L5_transform`, a compact transformation code built from:

- `delta_L1`: occupation change
- `delta_L2`: transition-graph change
- `delta_L3`: FFT/phase change
- `delta_L4`: recurrence-topology change

AQ-1a-TF diagnostic question:

- once real transport noise is added, which controller best preserves semantic margin, the `L2` transition layer, and the new `L5` transformation field without sacrificing the other semantic layers?

Runtime threshold rule:

- first pass uses only seed `11` across noise `0.30` and `0.45`
- seed `29` is a follow-up run only if seed `11` finishes cleanly and the `L5` table is interpretable
- do not run the full two-seed AQ-1a-TF matrix as one job during constrained TPU windows

Primary transformation criterion:

```text
transform_exact_match =
  source motif correct
  AND target motif correct
  AND L5 transform class correct
```

If AQ-1a-TF shows a real noisy crossover, then it becomes worth exploring a later `projection_alpha` or `damping_spread` micro-sweep. If not, the best next move is not more controller tuning but stronger semantic payload or decoder differentiation.

## AQ-1a-TF gate result

AQ-1a-TF seed-11 gate completed on `2026-05-29` and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ1A_TF_GATE_REMOTE_ANALYSIS_2026-05-29.md`
- `emergent_quantum_geometries/docs/PROGRAM_AQ1A_TF_S29_REMOTE_ANALYSIS_2026-05-29.md`

Seed-11 result:

- `L5` transform recovery remained strong at both `noise=0.30` and `0.45`.
- `transform_exact_match` was `1.0000` for all controllers at `0.30`.
- At `0.45`, `free` and `alpha065` stayed at `1.0000`; `alpha055` dropped to `0.7500`.
- `alpha065` had the strongest high-noise transform profile.

Seed-29 follow-up result:

- `L5` remained measurable, but exact transform recovery did not generalize.
- Best seed-29 `transform_exact_match` was `0.5000`.
- Several controller/noise cells dropped to `0.0000`.
- `alpha065` preserved `L5 = 0.9148` at `noise=0.30`, then collapsed to `L5 = 0.5368` at `noise=0.45`.
- `free` was steadier at high noise but did not recover exact transforms reliably.

Revised decision:

- AQ-1a-TF showed that `L5` exists, but it did not prove stable operator transport.
- The likely failure mode is endpoint/label association rather than a generative operator.
- Do not widen AQ-1a-TF into a longer sequence-pair matrix.
- Move to AQ-1b.

## AQ-1b objective shift

The next run should not simply broaden sequence-pair recovery. It should move to operator-conditioned folded recovery.

AQ-1b payload:

- transport `source motif + compressed operator T`
- hold out direct target features from payload

AQ-1b recovery target:

- `source_hat`
- `operator_hat`
- `target_hat = operator_hat(source_hat)`

AQ-1b metrics:

- `operator_recovery`
- `derived_target_accuracy`
- `operator_application_error`
- `source_recovery`
- `target_direct_leakage_test`
- `operator_swap_consistency`
- `source_swap_consistency`
- `L1`-`L5` recovery

Primary KPI:

- `derived_target_accuracy`

Pre-launch AQ-1a-TF diagnostics:

- estimate `MI(L5; source)`
- estimate `MI(L5; target)`
- estimate `MI(L5; transform_class)`
- compute transform confusion matrices by seed, noise, and controller
- if `MI(L5; target) > MI(L5; transform_class)`, treat AQ-1a-TF as target leakage rather than operator transport

Critical controls:

- target leakage test: target prediction should stay near chance without the operator channel
- operator swap test: same source plus different operators should derive different targets
- source swap test: same operator plus different sources should derive targets according to the same rule

L5 operator encoding:

- replace pure delta summaries with a compact operator code over semantic layers
- represent each transform class as an action over `L1`, `L2`, `L3`, and `L4`
- use a low-rank factorization of the layer action rather than directly transporting target features

Success criteria:

- `operator_recovery` above chance
- `derived_target_accuracy` above target leakage
- high operator-swap consistency
- high source-swap consistency
- direct target leakage near chance

Strong first-pass target:

- `derived_target_accuracy >= 0.75` at `noise=0.30`
- derived target accuracy remains above chance at `noise=0.45`
- target leakage remains near chance

If AQ-1b works, it supports a stronger claim than endpoint-pair transport: the folded state carries a recoverable generative operator, not just two recoverable semantic labels.

Implementation status:

- `program_ap_tunnel_eigen_recovery.py` now includes `--aq-1b`.
- `run_program_ap_ready_node.ps1` allows `--aq-1b`.
- The compact AQ-1b gate uses `sequence_length = 8`, `noise = 0.30, 0.45`, seeds `11, 29`, and controllers `free`, `alpha055`, `alpha065`.
- AQ-1b completed and showed strong layer survival but weak compositional operator recovery.
- The exact source/operator tests now use one-hot recovery states for AQ-1b/AQ-1c, because mixture states make exact class recovery ambiguous.

## AQ-1c operator identifiability gate

AQ-1c should not add seeds, noise, hard controls, or alpha tuning. It should prove that the operator family is identifiable before returning to transport.

AQ-1c run shape:

- sources: `4`
- operators: `4`
- source/operator pairs: `16`
- noise: `0.00`
- depth: `3`
- controller: `free`
- seed: `11`
- hard controls: disabled

AQ-1c gates:

- source recovery must reach `0.75`
- operator recovery must reach `0.75`
- derived target accuracy must reach `0.75` only after source/operator recovery pass
- target leakage should remain low

Implementation status:

- `program_ap_tunnel_eigen_recovery.py` now includes `--aq-1c`.
- `run_program_ap_ready_node.ps1` allows `--aq-1c`.

AQ-1c completed on `2026-05-29` and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ1C_REMOTE_ANALYSIS_2026-05-29.md`

Result:

- source recovery: `0.5625`
- operator recovery: `0.4375`
- derived target accuracy: `0.2500`
- target leakage: `0.1875`
- `L5` recovery: `0.6242`

Decision:

- AQ-1c failed the noise-free identifiability gate.
- Do not run AQ-1d/noisy transport yet.
- Repair source/operator separability and add confusion tables before the next TPU cycle.

Course correction implemented:

- source recovery now uses an explicit source semantic code.
- AQ-1c source/operator payload positions are separated rather than blended.
- operator codes are more orthogonal.
- summaries now include `operator_invariance_score`, `source_confusion`, and `operator_confusion`.
- the next TPU run should rerun `--aq-1c`, not AQ-1d.

### Phase 0: Small AP-U Calibration

Before AQ, keep AP-U small:

- `APR_reference`
- `alpha065`
- `adaptive_v2`
- noise `0.30`, `0.45`
- seeds `11`, `29`, `47`

This protects comparability and should fail fast if the comparator baseline shifts. AP-U is a hard gate: if APR margins at `noise=0.30` differ from the references below by more than `0.020` on any seed, stop and do not run AQ.

Reference APR margins:

| seed | expected margin |
| --- | ---: |
| `11` | `+0.0803` |
| `29` | `+0.0762` |
| `47` | `+0.1007` |

### AQ Motif Alphabet

AQ uses an explicit `8`-motif semantic alphabet:

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

Each motif must have a distinct transition graph and recurrence signature, otherwise layer-wise recovery becomes uninterpretable.

Before coding, keep `fractal_nested` genuinely hierarchical rather than merely longer periodic, and keep `alternating` distinct from plain period-2 behavior. The two motifs should separate across occupation statistics, transition graph, FFT/phase signature, and recurrence topology.

### Layered Encoding

AQ-0 locks four semantic layers:

| layer | encodes | expected robustness |
| --- | --- | --- |
| `L1` | symbol occupation statistics | strongest |
| `L2` | transition graph | strong/moderate |
| `L3` | FFT/phase signature | moderate/fragile |
| `L4` | cross-motif correlation / recurrence topology | most fragile |

Expected degradation:

- `noise=0.00`: `L1`-`L4` reconstruct well
- `noise=0.30`: `L1`-`L3` mostly survive
- `noise=0.45`: `L1`-`L2` / core topology survive while `L3`-`L4` degrade

### Phase 1: TPU Payload Transport

First AQ TPU pass:

- payload: `8`-motif semantic alphabet
- encoding: layered MPS / folded semantic graph
- bond dimension: `4`
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- depth: `3`
- seeds: `11`, `29`, `47`
- noise: `0.00`
- hard controls: disabled

The `noise=0.00` ceiling is mandatory. If fidelity fails at `0.00`, the failure is encoder/decoder architecture. If it works at `0.00` but degrades at `0.30` or `0.45`, the failure is transport robustness.

Keep AQ-0 noise-free only. Add `0.30` and `0.45` in AQ-1 after the ceiling is established.

Optional AQ-0 comparison arm, only if runtime stays small:

- `adaptive_v2_shifted`

Do not include `adaptive_v1` in AQ-0.

Do not include hard controls in AQ-0. At the noise-free ceiling stage, the comparison is encoder/decoder ceiling versus free transport baseline. Hard controls belong in AQ-1.

TPU should save compact artifacts only:

- compressed signatures
- recovered state vectors
- layer-wise phase structure
- controller metadata
- hard-control outputs

Heavy decoder comparisons should stay off TPU.

### Phase 2: Offline Decoding

Offline CPU decoder comparisons:

- ridge decoder on the untransported encoded payload as the first sanity check
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

Layer-wise breakdown is required because aggregate noisy cosine can hide where recovery fails.

### Compression Knob

MPS gives AQ a real compression axis: bond dimension.

Planned sweep after the first ceiling probe:

- `bond_dim=2`
- `bond_dim=4`
- `bond_dim=8`

Do not sweep all bond dimensions in AQ-0. Start with one low bond dimension and expand only after the noise-free ceiling succeeds.

AQ-0 starts at `bond_dim=4`. Do not start at `bond_dim=2`; for an `8`-motif payload it is likely too compressed and could make the noise-free ceiling fail before transport is being tested.

### Success Shape

Strong AQ result:

- `noise=0.00`: high layer-wise fidelity across most semantic layers
- `L1` occupation: strong recovery
- `L2` transition graph: strong recovery
- `L3` FFT/phase: moderate-high recovery
- `L4` recurrence/correlation: recoverable, but allowed to be weakest
- `noise=0.30`: `alpha065` preserves semantic graph recovery above `free` and hard controls
- `noise=0.45`: core/topological layers survive while fine layers degrade

This would support:

> Folded semantic information can be transported as a compressed recoverable subspace, with degradation occurring layer-wise rather than uniformly.

### Run Order

1. `AP-U`: calibration / reference lock
2. `AQ-0`: semantic payload, noise-free ceiling with `bond_dim=4`
3. `AQ-0b`: explicit `8`-motif layer diagnostic with `free`, `alpha055`, and `alpha065`
4. `AQ-1a-TF`: compact noisy motif transformation-field transport with `free`, `alpha055`, and `alpha065`
5. `AQ-1b`: compact operator-conditioned folded recovery with target leakage and swap controls
6. `AQ-offline`: compare ridge, MPS, and tunnel-eigen unfolded decoders

IBM-lite stays out until AQ has a working semantic payload.

## AQ-IMG: Patch-Codec Bridge (Image-Domain Motifs)

AQ-IMG is the bridge from abstract sequence motifs to the smallest image-like object we can plausibly transport and reconstruct under the same folded semantic pipeline.

It is intentionally not "full image" or "video". It tests whether:

1. A tiny grid of image patches can be encoded as semantic patch states.
2. Bob can recover semantic patch identity and layered features.
3. A lightweight decoder can reconstruct coarse pixel values from the recovered semantic state.

Run records:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG0_REMOTE_ANALYSIS_2026-05-30.md`
- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG1_LITE_REMOTE_ANALYSIS_2026-05-31.md`

### AQ-IMG-0 (baseline, noise-free)

Key baseline outcome (noise `0.00`, controller `free`, seed `11`):

- motif recovery: `16/16` exact (`1.0000`)
- semantic layers preserved strongly (`L1`/`L2`/`L4` high; `L3` moderate-high)
- pixel reconstruction is the bottleneck: PSNR `16.8927 dB`

Interpretation:

- Transport is working at the semantic patch level.
- Reconstruction quality is limited primarily by the encoding (too few degrees of freedom per patch) and only secondarily by the decoder.

In AQ-IMG-0 the encoder collapses each patch into:

$$
p_i \\mapsto (m_i, r_i)
$$

where \(m_i\) is the nearest image-domain motif and \(r_i\) is a scalar residual. This is too coarse for within-class variation like phase, contrast, duty cycle, and local amplitude:

$$
p_i = \\mathrm{catalog}(m_i) + \\epsilon_i,
\\quad \\epsilon_i \\not\\approx \\text{(scalar)}.
$$

So we preserved \(I(\\tilde{z}; m_i)\) well, but not enough \(I(\\tilde{z}; p_i \\mid m_i)\) to yield high pixel fidelity.

### AQ-IMG-1-LITE (vector residual probe, regressed)

AQ-IMG-1-LITE was introduced to avoid long compile/runtime while probing a richer residual grammar. The corrected lite run did finish, but regressed badly:

- motif recovery: `0.3750`
- pixel PSNR: `9.7162 dB`

The per-patch readout showed a structured collapse (not pure randomness):

- several mid-energy directional motifs collapsed toward `v_edge`
- `flat` and `noise_burst` survived

Most likely causes (supported by the printed metrics and the design changes):

1. One exemplar per motif removed within-class structure (AQ-IMG-0 had repeated classes; the lite probe was effectively one patch per class).
2. The target feature width increased sharply without richer transport exposure (residual expanded from 1 scalar to a 5-vector, but the training geometry stayed tiny).
3. Runtime shortcuts neutralized control margins by construction (so `margin`/`block_lift`/`topology_lift` are not meaningful ranking signals for that run).

Conclusion:

- AQ-IMG-0 remains the stable baseline.
- AQ-IMG-1 is not promotable in its current form; the vector residual idea needs a fairer A/B test.

### Next Patch Step (controlled, runtime-bounded)

Do not expand to bigger grids or add noise yet. The next patch run should isolate only one variable.

Controlled gate:

- keep the same repeated small patch roster across both arms
- controller: `free` only
- keep the lite interval-node geometry/runtime guard

Compare:

1. scalar residual: \(p_i \\mapsto (m_i, r_i)\)
2. vector residual: \(p_i \\mapsto (m_i, r_i^{(0:4)})\)

Required reporting:

- motif accuracy
- pixel PSNR
- residual MAE
- per-class confusion and collapse pattern

If (2) improves PSNR without collapsing motif recovery, then it becomes worth scheduling:

- a full `16` patch run
- then a small noise sweep
- then a temporal sequencing run (AQ-SEQ) using patch states as the vocabulary
