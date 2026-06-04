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

## AQ-FFT: K-Space Semantic Codec Direction

The AQ-IMG result changes the encoding strategy.

What we learned:

- symbolic motif headers are recoverable
- scalar residuals are too coarse for high-quality pixel reconstruction
- the first vector-residual grammar widened the target without enough stable examples and collapsed motif separability

The next representation should align the recoverable payload with D-LinOSS dynamics. D-LinOSS already transports complex amplitude/phase oscillator structure, so image patches should be represented in frequency space rather than only through symbolic motif residuals.

For a patch \(p \in \mathbb{R}^{4 \times 4}\):

$$
F = \mathcal{F}_2(p)
$$

Keep the top-\(K\) coefficients by magnitude:

$$
\Omega_K = \operatorname{TopK}(|F_{uv}|),
\qquad
\tilde{F}_{uv} =
\begin{cases}
F_{uv}, & (u,v) \in \Omega_K \\
0, & \text{otherwise}
\end{cases}
$$

and reconstruct:

$$
\hat{p} = \operatorname{Re}\left(\mathcal{F}^{-1}_2(\hat{F})\right).
$$

This makes reconstruction objective instead of heuristic. It also gives clean diagnostics:

- coefficient cosine
- frequency-index recovery
- phase recovery
- IFFT PSNR
- oracle top-\(K\) PSNR

### Implemented AQ-FFT-0 Gate

`program_ap_tunnel_eigen_recovery.py` now includes:

- `--aq-fft-0`
- `--aq-fft-top-k` (default `6`)

AQ-FFT-0 keeps the image-domain motif header for interpretability, but moves the reconstructable body to sparse k-space:

$$
\text{payload} =
[\text{motif header}, \operatorname{Re}(\tilde{F}), \operatorname{Im}(\tilde{F}), \Omega_K].
$$

Runtime shape:

- patch grid: `4 x 4`
- patch size: `4 x 4`
- patches: `16`
- noise: `0.00`
- seed: `11`
- controllers: `free`, `tunnel_eigen_residual_alpha065`
- fast codec probe controls: enabled by construction to avoid expensive comparator branches

Success gate:

- K-space IFFT PSNR should beat AQ-IMG-1-LITE (`9.7162 dB`) immediately.
- With `K=6`, it should aim toward or above the practical `22-25 dB` band on the synthetic patches.
- Motif recovery should not collapse into the `v_edge` basin.
- Frequency-index recovery and phase recovery should be reported separately from motif recovery.

### AQ-FFT-0 Result and Codec Diagnosis

AQ-FFT-0 completed with the fixed low-frequency basis and phase encoded as `sin(phi), cos(phi)`.

Result summary:

- motif recovery: `1.0000`
- transform exact match: `1.0000`
- phase recovery: about `0.875`
- frequency overlap: about `0.917`
- top-K energy recall: `0.875`
- realized PSNR: about `11.1 dB`
- oracle PSNR: about `11.83 dB`

Interpretation:

- the semantic header channel works
- phase transport is not catastrophically broken
- fixed low-frequency `K=6` has a low reconstruction ceiling
- the immediate bottleneck is codec capacity, not only D-LinOSS recovery

This creates two useful regimes:

| codec | oracle ceiling | realized result | failure mode |
| --- | ---: | ---: | --- |
| dynamic top-K | high | low | index/phase/recovery burden |
| fixed low-frequency K=6 | low | near ceiling | insufficient payload capacity |

### AQ-FFT-1: Core + Residual K-Space Genome

The next codec should combine the stable fixed core with a small adaptive residual channel.

For each patch:

$$
F = \mathcal{F}_2(p)
$$

Define a fixed low-frequency core:

$$
\Omega_c = \{(0,0),(0,1),(1,0),(1,1),(0,2),(2,0)\}
$$

Then select a residual set from the remaining coefficients:

$$
\Omega_r = \operatorname{TopK}_{r}\left(|F_{uv}| \;:\; (u,v)\notin\Omega_c\right)
$$

The transported body is:

$$
\tilde{F}_{uv} =
\begin{cases}
F_{uv}, & (u,v)\in \Omega_c \cup \Omega_r\\
0, & \text{otherwise}
\end{cases}
$$

AQ-FFT-1 adds a DNA-like instruction grammar, used only as an engineering analogy:

| DNA analogy | codec field |
| --- | --- |
| base token | frequency slot |
| codon | coefficient instruction packet |
| gene | patch reconstruction program |
| regulatory marker | semantic header / checksum |
| expression | inverse FFT reconstruction |

Each instruction packet is:

$$
g_i = [\text{freq\_slot}_i,\text{residual\_slot}_i,\bar{m}_i,\cos\phi_i,\sin\phi_i,\text{checksum}]
$$

where:

$$
\bar{m}_i = \frac{|F_i|}{\sum_j |F_j| + \epsilon}
$$

This keeps the biology language disciplined: no consciousness claim is imported. The useful idea is modular instruction encoding with redundancy and checksum.

Implemented next-run mode:

- `--aq-fft-1`
- `--aq-fft-top-k 6`
- `--aq-fft-residual-k 2` for the first gate

Recommended next run:

- noise: `0.00`
- depth: `3`
- seed: `11`
- controllers: `free`, `alpha065`
- variants: run `residual_K=2` first, then `residual_K=4` only if `K=2` raises oracle PSNR without collapsing recovery

Pass criteria:

- oracle PSNR rises meaningfully above AQ-FFT-0's `11.83 dB`
- realized PSNR moves with oracle PSNR rather than staying near `11 dB`
- motif recovery remains `1.0000`
- phase recovery stays above `0.80`
- instruction/L5 recovery stays above `0.90`

### AQ-FFT-1 Result and Why It Produced These Values

AQ-FFT-1 completed as the intended codec-capacity gate and is documented in:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT1_REMOTE_ANALYSIS_2026-06-01.md`

Observed result:

| controller | motif | L5 grammar | phase cos | freq overlap | top1 freq | energy recall | realized PSNR | oracle PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `1.0000` | `0.9608` | `0.8930` | `0.7578` | `1.0000` | `0.8738` | `16.9995 dB` | `25.0480 dB` |
| `alpha065` | `1.0000` | `0.9599` | `0.8913` | `0.7578` | `1.0000` | `0.8738` | `16.6004 dB` | `25.0480 dB` |

Methodological explanation:

1. AQ-FFT-1 improved because the codec became more expressive.
   - AQ-FFT-0 transported only a fixed low-frequency `K=6` body.
   - AQ-FFT-1 kept that stable scaffold and added `2` adaptive residual coefficients.
   - That is why `oracle_psnr` jumped from about `11.83 dB` to `25.05 dB`: the codec itself could now represent much more of the patch.

2. Realized PSNR improved, but lagged far behind oracle, because the adaptive residual channel is harder to realize than the fixed scaffold.
   - `16.9995 dB` is a real gain over AQ-FFT-0.
   - But the remaining gap to `25.0480 dB` means the transport+decode stack is not fully recovering the new residual instructions.
   - In other words, capacity is no longer the only limit; residual expression has become the new bottleneck.

3. The semantic header remained stable.
   - motif recovery stayed `1.0000`
   - source/target/exact transform recovery stayed `1.0000`
   - L5 grammar recovery stayed about `0.96`
   This means the richer codec body did not break the semantic identity channel. That is important because it lets us improve image detail without losing the folded semantic state classification that AQ already transports well.

4. The frequency diagnostics tell us exactly where the recovery burden sits.
   - `top1_freq_acc = 1.0000`
   - `freq_idx_overlap = 0.7578`
   - `topK_energy_recall = 0.8738`

   This means the dominant spectral anchor is usually recovered correctly, but the full retained set is only partially recovered. So the problem is not that FFT transport failed wholesale. The narrower problem is that the extra residual slots are still only partly stable.

5. Phase is not the main thing breaking reconstruction anymore.
   - `phase_cos` stayed around `0.89`
   - `coeff_mag_cos` and `coeff_real_cos` stayed in the mid `0.85` range
   - `coeff_imag_cos` stayed weaker, around `0.63-0.65`

   That pattern suggests the `sin(phi), cos(phi)` change worked well enough to stop gross phase-wrap collapse. The remaining reconstruction gap is more about uneven complex-coefficient recovery, especially the more fragile imaginary component and residual-slot realization.

6. `free` slightly beating `alpha065` at `noise=0.00` is methodologically coherent.
   - `alpha065` helps when we need projection-driven robustness.
   - At the clean codec ceiling, that same projection likely smooths some reconstructive detail.
   - So `free` preserving slightly more patch detail here is not a contradiction; it just means this run was a codec diagnostic rather than a noisy controller-promotion test.

Bottom line:

- AQ-FFT-1 moved the experiment forward.
- AQ-FFT-0 said the semantic header was fine but the body was too small.
- AQ-FFT-1 says the body can be made much larger, and now the limiting factor is how well the residual instructions survive and decode.

That is the right failure to have next, because it means the representation improved enough to expose the next real bottleneck.

### AQ-FFT-2: Shape-Charge Controlled K-Space Transport

The pasted feedback points to a useful methodological correction from tensor-network simulation practice:

> respect the geometry and conserved quantities of the represented system before approximation.

For AQ-FFT, this means frequency coefficients should not be treated only as flat feature vectors. The next compact run should give D-LinOSS access to stabilizing shape observables during transport.

AQ-FFT-2 implements a first lightweight version of that idea.

Mode:

- `--aq-fft-2`
- `--aq-fft-top-k 6`
- `--aq-fft-residual-k 2`

Controller comparison:

- `free`
- `shape_charge_control`

The `shape_charge_control` arm keeps the same codec as AQ-FFT-1, but adds a conjugate-field-style correction based on six measured charges:

The triplet/codon side of the analogy is implemented as coefficient packet structure, not as biological or particle-physics literalism:

$$
\text{packet}_i =
(\text{slot/support}_i,\ |F_i|,\ [\cos\phi_i,\sin\phi_i]).
$$

The useful test is whether grouped support, magnitude, and phase behave like a coherent instruction unit. The resulting charge is measured globally through the shape-charge vector below.

| charge | observable |
| --- | --- |
| `Q_DC` | DC / luminance energy |
| `Q_lowfreq_energy` | energy in the fixed low-frequency basis |
| `Q_phase_coherence` | coherence of retained coefficient phases |
| `Q_residual_energy` | energy carried by adaptive residual coefficients |
| `Q_support_entropy` | entropy of retained coefficient support |
| `Q_motif_consistency` | semantic header consistency |

The charge vector is mapped onto deterministic node generators:

$$
G(n) = [1, c_n, \sin(2\pi c_n), \cos(2\pi c_n), s_n, \sin(4\pi c_n)]
$$

where \(c_n\) is the interval-node center and \(s_n\) is the interval span. The resulting conjugate field is:

$$
h_Q(n) =
\frac{(Q-\bar{Q})G(n)}
{\operatorname{std}((Q-\bar{Q})G(n))+\epsilon}
$$

and D-LinOSS receives the field as:

$$
\phi(n) \leftarrow \phi(n) + \eta h_Q(n)
$$

$$
\omega(n) \leftarrow \omega(n)\left(1 + 0.04\eta h_Q(n)\right)
$$

This is intentionally modest. It is not yet a full BP/MPS decoder or a high-bond tensor evolution. It is the smallest TPU-safe test of the key idea:

> conserved shape statistics should be active constraints during transport, not just passive metrics after transport.

New AQ-FFT-2 metrics:

- `fft_shape_charge_error`
- `fft_shape_charge_cosine`
- `fft_psnr_if_true_support`
- `fft_psnr_if_true_magnitudes`
- `fft_psnr_if_true_phases`

The error-budget PSNR values explain the remaining gap:

- true support with predicted coefficient values tests support/index error
- true magnitudes with predicted phase tests phase error
- predicted magnitudes with true phase tests magnitude error

Pass criteria:

- `shape_charge_control` lowers `fft_shape_charge_error` versus `free`
- realized PSNR improves above AQ-FFT-1's `17.00 dB`
- `coeff_imag_cos` and residual recovery improve
- motif/header recovery remains `1.0000`

If AQ-FFT-2 passes, try `residual_K=4`. If it fails, the next move should be a geometry-aware BP/message-passing decoder over the same recovered signatures rather than increasing payload size.

### AQ-FFT-2 Result: Why the Hypothesis Was Wrong

AQ-FFT-2 completed and did not support the shape-charge-control hypothesis.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT2_REMOTE_ANALYSIS_2026-06-01.md`

Observed result:

| controller | motif | phase cos | freq overlap | topK recall | shape error | realized PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `free` | `1.0000` | `0.8930` | `0.7578` | `0.8738` | `0.1221` | `16.9995 dB` |
| `shape_charge_control` | `1.0000` | `0.8708` | `0.7578` | `0.8738` | `0.1233` | `16.7852 dB` |

Conclusion:

- the added shape-charge field did not improve the coefficient-support bottleneck
- the added field slightly worsened phase-sensitive and imaginary-coefficient recovery
- the hypothesis that global shape-charge stabilization was the missing ingredient was wrong

Why this failed methodologically:

1. The controlled observables were too coarse.
   - `Q_DC`, `Q_lowfreq_energy`, `Q_phase_coherence`, `Q_residual_energy`, `Q_support_entropy`, and `Q_motif_consistency` are meaningful summaries.
   - But AQ-FFT-1's unresolved error lives in slotwise complex residual detail, not only in global summaries.

2. The control was not a true closed-loop feedback rule.
   - AQ-FFT-2 injected a charge-shaped perturbation into phase/frequency generation.
   - It did not run a transport-time correction loop based on current-vs-target charge drift.
   - In practice it behaved more like structured regularization than adaptive stabilization.

3. The diagnostic metrics show the real bottleneck did not move.
   - `freq_idx_overlap` stayed the same.
   - `topK_energy_recall` stayed the same.
   - `top1_freq_acc` had already saturated at `1.0000`.
   - So the control did not fix the support/index problem or the residual realization problem.

4. The shape-charge vector itself was already easy to recover.
   - `fft_shape_charge_cosine` stayed at `0.9956` for both arms.
   - That means the low-dimensional charge summaries were never the scarce object.
   - The scarce object is still the finer coefficient-level reconstruction.

So AQ-FFT-2 is a useful negative result. It tells us that global charge summaries are too weak a constraint to recover the missing residual structure.

## What Now Looks Solid

The stronger mathematical foundation in AQ currently sits in four places:

1. `AQ-0b`
   - explicit `8`-motif semantic states are recoverable at the noise-free ceiling
   - `L1`-`L4` survive with low leakage

2. `AQ-IMG-0`
   - semantic patch identity is perfectly recoverable
   - pixel fidelity is limited by encoder/decoder resolution, not by semantic transport collapse

3. `AQ-FFT-1`
   - the media-side problem is cleanly decomposed into codec capacity versus residual realization
   - this is a strong mathematical diagnostic frame even though the reconstruction is incomplete

4. `AQ-SEQ-0`
   - source semantic state recovery is exact
   - derived next-state accuracy is exact under a simple sequential gate
   - this is the cleanest support for semantic folded information recovery beyond static labels

The practical implication is that AQ should now lean on:

- source-state recovery
- explicit layer recovery
- sequential transition gates
- separate decoder/reconstruction layers

rather than making stronger claims about operator transport or global shape-charge stabilization.

## Next Run: AQ-DNA-0 Locality-Preserving Instruction Genome

The next run should use the failure of AQ-FFT-2 directly.

AQ-FFT-2 showed:

- global shape summaries are already easy to recover
- preserving those summaries does not recover slotwise residual instructions
- the bottleneck is local coefficient-packet realization, not global charge drift

So AQ-DNA-0 should move from continuous residual regression to discrete, local instruction tokens.

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_DNA0_RUN_SPEC_2026-06-01.md`

Launch target:

```bash
--aq-dna-0
```

Initial constraints:

- noise: `0.00`
- seed: `11`
- depth: `3`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- codec: AQ-FFT-1 fixed core `K=6` plus `residual_K=2`
- no alpha sweep
- no hard controls as full arms
- no noise sweep
- no residual_K expansion

Token grammar:

```text
packet = [
  patch_id,
  frequency_slot,
  magnitude_state,
  phase_state,
  residual_state,
  support_rank_state,
  motif_header,
  checksum
]
```

The token states should be mixture-like, not arbitrary hand thresholds:

- magnitude: low / medium / high
- phase: sector tokens
- residual: core / residual_low / residual_high
- support rank: dominant / core / enhancement

For the first TPU implementation, deterministic quantile states are acceptable as a GMM-lite approximation. A true GMM can come later if the tokenized pathway works.

Ordering:

- use Hilbert ordering over the `4 x 4` patch grid
- use fixed k-space zigzag/locality order within each patch
- place residual tokens next to their nearest fixed-core frequency neighbor

This follows the useful tensor-network lesson from the pasted feedback:

> preserve locality before folding so bond capacity is spent on semantic dependency, not on undoing a scrambled sequence.

Primary measurements:

- motif/header recovery
- token exact recovery
- magnitude-state recovery
- phase-state recovery
- residual-state recovery
- support-rank recovery
- realized PSNR
- oracle PSNR
- motif-only PSNR baseline

Dependency validation:

- `MI(header; support)`
- `MI(motif; residual_state)`
- `MI(phase_state; reconstruction_error)`
- `MI(slot; local_patch_class)`
- `MI(residual_state; pixel_error)`

Null controls:

- shuffle coefficient slots
- shuffle phase tokens
- shuffle semantic headers
- shuffle locality order
- block-permute local tokens

Report:

```math
\Delta MI = MI_{observed} - MI_{shuffle}
```

Success criteria:

- semantic header remains `1.0000`
- token recovery is above chance across all token classes
- observed token MI beats shuffled/null MI
- realized PSNR beats motif-only reconstruction
- ideal result: realized PSNR meets or exceeds AQ-FFT-1's `17.00 dB`

Interpretation:

- if MI survives above null but PSNR does not improve, AQ-DNA-0 still proves meaningful dependency recovery
- if MI and PSNR both improve, locality-preserving token genomes become the next media reconstruction path
- if both fail, repair token grammar or add a geometry-aware decoder before adding more TPU complexity

First AQ-DNA-0 true-gate result:

- remote analysis note: `emergent_quantum_geometries/docs/PROGRAM_AQ_DNA0_REMOTE_ANALYSIS_2026-06-01.md`
- the launch completed successfully on TPU
- the run used the real AQ-DNA-0 token branch rather than the earlier AQ-FFT-1 compatibility alias

Observed result:

- motif/header recovery `1.0000`
- `oracle_psnr = 25.0480 dB`
- realized `ifft_psnr = 16.9995 dB`
- `phase_cos = 0.8930`
- `freq_idx_overlap = 0.7578`
- `topK_energy_recall = 0.8738`
- packet exact match `0.9062`
- magnitude-state accuracy `0.9688`
- phase-state accuracy `0.9453`
- residual-state accuracy `0.9531`
- support-rank accuracy `0.9531`
- MI gaps over permutation nulls remained weak: `-0.0022`, `+0.0210`, `+0.0082`, `0.0000` bits

Interpretation:

- the token grammar is recoverable as a label system
- but the token branch did not improve realized reconstruction beyond the AQ-FFT-1 plateau
- and the weak MI-over-null signal means the token genome is not yet structurally informative enough to count as a locality/dependency win

Immediate planning implication:

- do not spend the next TPU cycle on controllers, noise, or larger payloads
- the next correction should make packet locality and packet dependencies materially affect the transport path rather than only the decoded label vocabulary

## Next Run: AQ-HYBRID-0 Continuous Geometry + Token Checksum

The latest implementation feedback suggests that hard tokenization may be masking the part of the signal D-LinOSS naturally preserves: continuous amplitude/phase geometry.

AQ-DNA-0 showed:

- discrete token labels recover well
- token dependency MI is weak
- realized PSNR does not move off the `~17 dB` plateau

So the next run should not be "more DNA tokens." It should be a hybrid branch:

- continuous k-space geometry remains the primary payload
- token states become checksums or local consistency constraints
- packet neighborhoods are measured directly

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_HYBRID0_RUN_SPEC_2026-06-01.md`

Planned mode:

```bash
--aq-hybrid-0
```

Runtime shape:

- noise: `0.00`
- seed: `11`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- no alpha sweep
- no noise sweep
- no residual expansion

Implementation direction:

- keep `sparse_fft_features` / continuous coefficient geometry in the primary `source_features`
- keep `sin(phi), cos(phi)` phase representation
- keep motif/header features
- add token checksum features as a side-channel, not as a replacement for continuous geometry
- add local packet-window metrics over adjacent k-space slots

Primary metrics:

- `ifft_psnr`
- `oracle_psnr`
- `coeff_mag_cos`
- `coeff_real_cos`
- `coeff_imag_cos`
- `phase_cos`
- `checksum_agreement`
- `window_token_consistency`
- `neighborhood_phase_consistency`
- continuous MI gap
- token MI gap

Decision rule:

- if PSNR exceeds `17.0 dB` and continuous MI beats token MI, promote the hybrid geometry-first representation
- if continuous MI beats token MI but PSNR does not improve, improve the decoder/window model before changing controllers
- if both continuous and token MI stay near null, locality still is not active enough in the transported representation

Status:

- AQ-HYBRID-0 remains a useful precursor design, but it is no longer the immediate next run.
- The next run should directly test whether the instruction packets need a hierarchical tensor geometry rather than another flat sequence or checksum side-channel.

## Program AS-0 Hierarchical Instruction Genome Transport

Program AS changes the representation while keeping the AQ-FFT-1 packet vocabulary.

Instead of feeding the instruction genome as a flat sequence, AS-0 arranges coefficient and residual packets as boundary leaves of a binary TTN:

```text
AQ-FFT-1 instruction packets
  -> binary TTN fold encoder
  -> root tensor + selected internal bonds + charge sectors
  -> D-LinOSS transport
  -> TTN unfold decoder
  -> coefficient field
  -> inverse FFT patch
```

The conceptual shift is:

```text
from: transport tokens, decode shape
to: fold tokens into shape, transport shape, unfold field
```

AS-0 does not start with full MERA. TTN is cheaper and enough to test whether multi-scale folding improves the residual instruction bottleneck.

Implemented mode:

```bash
--as-0
```

Runtime shape:

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

Payload:

- semantic header
- fixed k-space core
- sparse residual instructions
- phase-safe `sin(phi), cos(phi)` channels
- checksum

Each TTN internal bond carries:

| charge | meaning |
| --- | --- |
| `Q_DC` | luminance/DC charge |
| `Q_lowfreq_energy` | low-frequency energy |
| `Q_phase_coherence` | phase coherence |
| `Q_residual_support` | sparse residual support |
| `Q_motif_consistency` | semantic motif consistency |
| `Q_checksum_parity` | checksum parity |

The TTN fold tests:

```math
Q_{parent} \approx Q_{left} + Q_{right}
```

Primary metrics:

- `fft_patch_psnr`
- `fft_oracle_psnr`
- `as_root_tensor_cosine`
- `as_charge_conservation_error`
- `as_charge_cosine`
- `as_unfolded_coeff_psnr`
- `as_ttn_lift_over_flat_psnr`
- coefficient magnitude/real/imaginary cosine
- phase recovery
- top-K energy recall
- semantic header / motif recovery

Flat baseline:

- AQ-FFT-1 free realized PSNR: `~17.00 dB`
- AQ-FFT-1 oracle PSNR: `~25.05 dB`

Success condition:

- semantic header remains `1.0000`
- TTN realized PSNR exceeds the flat `~17.0 dB` baseline
- root tensor recovery is coherent
- charge recovery remains coherent
- coefficient recovery improves enough to explain the PSNR lift

Interpretation:

- if TTN improves PSNR or coefficient recovery, the missing ingredient was multi-scale shape encoding rather than controller tuning
- if root/charge recovery succeeds but PSNR stays flat, the unfold decoder is the likely bottleneck
- if root/charge recovery fails, the TTN representation is too compressed or not aligned with the D-LinOSS signature geometry

Only if AS-0 helps should the next run add MERA-style disentanglers. The MERA question is whether disentanglers can suppress local packet redundancy before coarse-graining, not whether larger payloads can brute-force reconstruction.

## AS-0 / AS-0b Outcome

AS-0 and AS-0b have now completed and give us a clear answer.

AS-0 result:

- `ifft_psnr = 16.9995 dB`
- `oracle_psnr = 25.0480 dB`
- `as_ttn_lift_over_flat_psnr ~= 0.0000 dB`
- `as_root_tensor_cosine = 0.9518`
- `as_charge_cosine = 0.9962`
- `as_charge_conservation_error = 0.1877`
- motif/header recovery stayed `1.0000`

Interpretation:

- the TTN root state survived transport well
- the charge summary survived transport very well
- the semantic/header channel stayed solved
- but reconstruction did not improve over the flat AQ-FFT-1 baseline

AS-0 therefore showed that hierarchical folding was recoverable as a global summary, but it did not show that the hierarchy improved coefficient-body realization.

AS-0b corrected the decoder path so the run actually reconstructed from the recovered TTN state rather than the old direct flat decoder:

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

Interpretation:

- the corrected TTN-unfold path worked operationally
- but it still did not improve reconstruction
- the hierarchy remains easier to recover than the slotwise residual coefficient detail that actually limits PSNR

Methodological conclusion:

- what worked:
  - semantic/header transport
  - TTN root recovery
  - TTN charge-summary recovery
- what did not work:
  - TTN folding did not lift realized PSNR
  - TTN unfolding did not improve support overlap, energy recall, or weak imaginary/residual coefficient recovery

So Program AS is useful as a negative result:

> hierarchical folded state transport is compatible with the channel, but it is not the main missing ingredient for reconstruction fidelity in the current AQ-FFT regime.

Updated decision:

- do not promote MERA as the automatic next step
- keep the TTN view as an analysis/representation tool
- put the next engineering focus back on coefficient-body representation and decoder design, especially the weak imaginary/residual channel

## Next Run: AQ-FFT-3 Direct Complex Oscillator Codec

The next correction targets the coefficient encoding itself.

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT3_RUN_SPEC_2026-06-02.md`

The repeated bottleneck is:

- `coeff_real_cos ~= 0.861`
- `coeff_imag_cos ~= 0.628`
- `phase_cos ~= 0.893`
- realized PSNR stuck near `17.0 dB`
- oracle PSNR near `25.05 dB`

The likely bug is that AQ-FFT-1 separated a complex coefficient into magnitude and phase channels, then reconstructed:

```math
\operatorname{Im}(\hat{F}_k)=|\hat{F}_k|\sin(\hat{\phi}_k)
```

Correlated magnitude and phase errors can make that product worse than either component separately.

AQ-FFT-3 keeps the AQ-FFT-1 codec capacity but injects the sparse complex coefficient directly into D-LinOSS:

```math
x_0[k] = F_k
```

The decoder then recovers:

```math
\hat{F}_k = \hat{x}[k]
```

Implemented mode:

```bash
--aq-fft-3
```

Runtime:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `free`, `tunnel_eigen_residual_alpha065`

Pass signal:

- `coeff_imag_cosine` rises above the `~0.628` plateau
- realized PSNR rises above `~17.0 dB`
- motif/header recovery stays solved

If AQ-FFT-3 does not lift the imaginary channel, the next likely suspect is D-LinOSS dynamics or signature extraction rather than TTN/charge/token representation.

## AQ-FFT-3 Outcome

AQ-FFT-3 completed and changed the picture in an important way.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT3_REMOTE_ANALYSIS_2026-06-02.md`

Observed result:

- `free`:
  `ifft_psnr = 20.6478 dB`,
  `oracle_psnr = 25.0480 dB`,
  `coeff_real_cos = 0.8713`,
  `coeff_imag_cos = 0.5086`,
  `phase_cos = 0.4980`
- `alpha065`:
  `ifft_psnr = 18.2753 dB`,
  `oracle_psnr = 25.0480 dB`,
  `coeff_real_cos = 0.8639`,
  `coeff_imag_cos = 0.5092`,
  `phase_cos = 0.4913`
- motif/header recovery stayed `1.0000`

What worked:

- direct complex injection lifted realized reconstruction strongly above AQ-FFT-1
- the semantic/header channel remained solved
- the codec ceiling stayed fixed, so the improvement was in realized transport and decode, not in representational capacity

What did not work:

- the imaginary channel remained weak
- phase fidelity became much worse than in AQ-FFT-1
- `alpha065` again underperformed `free` at the clean ceiling

Mathematical read:

PSNR is driven by total energy error in the reconstructed patch:

```math
\mathrm{MSE}(p,\hat{p}) \propto \sum_k |F_k-\hat{F}_k|^2
```

That means AQ-FFT-3 can improve PSNR by recovering the dominant energy-bearing coefficients better, especially DC and low-frequency real-dominant directions, even if it still fails to preserve lower-energy quadrature structure well.

So the result is not contradictory:

- better reconstruction of dominant coefficient energy
- weaker preservation of complex orientation

This is still a meaningful step toward reconstruction of unfolded semantic space.

Why:

- the folded semantic/header state remains transport-stable
- the coefficient body is now reconstructable enough to materially improve the image-domain field
- the program is beginning to demonstrate not just survival of folded semantic structure, but partial recovery of the unfolded field derived from it

Updated decision:

- keep direct complex encoding as the better reconstruction path than AQ-FFT-1
- do not interpret AQ-FFT-3 as a solved complex-field transport result
- focus the next correction on preserving complex orientation during transport/signature extraction, rather than going back to TTN, charge, or token-side redesigns first

## AQ-FFT-4 Complex Orientation Gate

Implemented follow-up:

```bash
--aq-fft-4
```

AQ-FFT-4 keeps the AQ-FFT-3 codec fixed:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=2`
- controllers: `free`, `complex_pair_norm`, `complex_phase_locked`

It does not include alpha tuning, larger `K`, noise, or video.

The new gate tests whether the remaining reconstruction gap is caused by broken complex-orientation preservation:

```math
F_k = \operatorname{Re}(F_k) + i\operatorname{Im}(F_k)
```

instead of missing k-space capacity.

`complex_pair_norm` injects unit complex coefficient directions into D-LinOSS slots and keeps magnitude as a coupled side signal. `complex_phase_locked` decodes a magnitude estimate and recombines it with the recovered complex direction so that:

```math
\operatorname{Re}(\hat{F}_k)^2 + \operatorname{Im}(\hat{F}_k)^2 \approx |\hat{F}_k|^2
```

New diagnostics:

- support-weighted angle error
- energy-weighted complex MSE
- quadrature and imaginary energy recall
- complex pair-norm error
- phase error by coefficient slot
- PSNR error budget separating support, real channel, and imaginary channel effects

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT4_RUN_SPEC_2026-06-02.md`

## AQ-FFT-5 Gauge-Aligned Complex Decoding

AQ-FFT-4 made `complex_pair_norm` the reference codec:

- `ifft_psnr = 23.1818 dB`
- `oracle_psnr = 25.0480 dB`
- motif/header recovery = `1.0000`
- imaginary and phase cosine stayed near `0.51`

AQ-FFT-5 tests whether the low imaginary/phase cosine is a complex gauge mismatch rather than true information loss.

For each patch, the diagnostic computes:

```math
\theta^*=\arg\left(\sum_k F_{\mathrm{true},k}\overline{F_{\mathrm{pred},k}}\right)
```

and evaluates the recovered field after:

```math
\hat{F}^{aligned}=e^{i\theta^*}\hat{F}
```

Implemented mode:

```bash
--aq-fft-5
```

Controllers:

- `complex_pair_norm`
- `complex_pair_norm_gaugefix`

`complex_pair_norm_gaugefix` uses an anchor-based decoder correction that rotates the recovered coefficient field so the DC coefficient is real-positive, with a strongest-coefficient fallback. This decoder does not use the true target coefficients.

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT5_RUN_SPEC_2026-06-02.md`

## AQ-FFT-6 Hermitian-Constrained Complex Decoder

AQ-FFT-5 ruled out a simple global gauge mismatch:

- aligned imaginary cosine stayed near `0.508`
- aligned phase cosine stayed near `0.508`
- gauge error was `0.0000`
- PSNR before/after alignment was unchanged

AQ-FFT-6 tests local FFT structure instead. For real-valued image patches:

```math
F[-k] = \overline{F[k]}
```

Implemented mode:

```bash
--aq-fft-6
```

Controllers:

- `complex_pair_norm`
- `complex_pair_norm_hermitian`

The Hermitian variant projects the recovered complex spectrum into the real-image FFT subspace before inverse FFT:

```math
\hat{F}_{sym}[k] = \frac{1}{2}\left(\hat{F}[k] + \overline{\hat{F}[-k]}\right)
```

Self-conjugate slots are forced real.

New diagnostics:

- Hermitian error before/after projection
- self-conjugate imaginary energy
- pairwise conjugate consistency
- energy-weighted phase cosine

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT6_RUN_SPEC_2026-06-02.md`

## AQ-FFT-7 Residual Capacity Scaling

AQ-FFT-6 showed:

- Hermitian error was already tiny
- Hermitian projection did not move PSNR
- energy-weighted phase cosine was excellent

So AQ-FFT-7 stops chasing raw phase/imaginary cosine and tests whether `complex_pair_norm` scales with more residual coefficients.

Implemented mode:

```bash
--aq-fft-7
```

Run shape:

- fixed core: `K=6`
- residual: `residual_K=4`
- controller: `complex_pair_norm`
- noise: `0.00`
- seed: `11`
- depth: `3`

Reference:

- residual_K=2 realized PSNR near `23.18 dB`
- residual_K=2 oracle PSNR near `25.05 dB`

Promoted metrics:

- realized PSNR
- oracle PSNR
- energy-weighted phase cosine
- energy-weighted complex MSE
- topK energy recall
- motif/header recovery

Run spec:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT7_RUN_SPEC_2026-06-02.md`

## AQ-FFT-7 Outcome

AQ-FFT-7 completed successfully and is now the strongest reconstruction result in Program AQ.

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_FFT7_REMOTE_ANALYSIS_2026-06-02.md`

Observed:

- residual_K=4
- controller: `complex_pair_norm`
- motif/header recovery: `1.0000`
- realized PSNR: `26.3560 dB`
- oracle PSNR: `31.1144 dB`
- energy-weighted phase cosine: `0.9992`
- top-1 frequency accuracy: `1.0000`

Comparison to residual_K=2:

- realized PSNR rose from about `23.18 dB` to `26.36 dB`
- oracle PSNR rose from about `25.05 dB` to `31.11 dB`

Scientific interpretation:

The raw phase and imaginary cosine weakness is no longer the primary reconstruction gate at this patch scale. AQ-FFT-5 ruled out global gauge mismatch; AQ-FFT-6 ruled out Hermitian mismatch as a meaningful blocker; AQ-FFT-7 then showed that increasing sparse residual k-space capacity raises both oracle and realized reconstruction.

Defensible claim:

> In this simulator, D-LinOSS complex oscillator transport can preserve a folded semantic/header state while carrying enough continuous k-space coefficient structure to reconstruct an unfolded `4 x 4` image patch field. The reconstruction improves when sparse residual k-space capacity is increased, while semantic/header recovery remains perfect.

This should be framed as simulated folded-state recovery in a quantum-like complex oscillator state space, not as physical quantum teleportation.

Next step:

- AQ-FFT-8 should hold residual_K=4 fixed and add noise `0.30`
- compare `free`, `alpha065`, and `shape_charge_control`

## AQ Visual Preview Export

The AQ image/FFT path now supports a lightweight visual readout:

```text
--save-preview-png
```

When enabled, the program writes compact grayscale PNG contact sheets from the patch tensors it already computes:

- original target patches
- recovered patches
- amplified absolute error

This keeps visualization compatible with the TRC discipline by default: previews are generated on the TPU VM, recorded in `program_ap_summary.json`, and do not require downloading result folders. A single small PNG can be fetched later if visual inspection is needed.

The ready-node launcher can pass the flag through with:

```powershell
$env:PROGRAM_AP_EXTRA_ARGS='--save-preview-png'
```

## AQ-YINYANG-0 Candidate

The next concrete visual corpus should be a small 8-bit yin-yang sequence:

- `16 x 16` grayscale frame
- two or more frames with yin/yang polarity alternating
- `4 x 4` patch tiling
- AQ-FFT complex k-space codec with `complex_pair_norm`
- target/recovered/error PNG preview export

This gives a recognizable symbolic target while staying close to the successful AQ-FFT-7 scale. It should be implemented as a distinct mode so it tests coherent visual reconstruction rather than reusing the existing synthetic patch roster.

## AQ-YINYANG-1 One-Frame Gate

AQ-YINYANG-0 produced recognizable but smeared two-frame reconstructions. The numbers suggested this was mostly a sparse-codec ceiling: realized PSNR was close to oracle PSNR, while coefficient recovery and energy-weighted phase were strong.

AQ-YINYANG-1 removes the second frame and recovers only the first tai-chi-tu frame. This isolates whether the blur comes from:

- the sparse `core_K=6 + residual_K=4` k-space representation,
- the two-frame alternating-polarity sequence,
- or D-LinOSS transport/unfolding itself.

Expectation note:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_YINYANG1_EXPECTATIONS_2026-06-03.md`

Decision rule:

If oracle PSNR is still low and realized PSNR stays close to oracle, the next work is image-domain encoding: denser support, edge-aware binary decoding, or a better visual motif vocabulary. If oracle PSNR is high but realized PSNR is low, the next work is transport signature extraction or coefficient unfolding.

## AQ-YINYANG Visual Findings

Run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_YINYANG_FINDINGS_2026-06-03.md`

Key result:

- `AQ-YINYANG-1` used `10 / 16` FFT coefficients per patch and showed a sparse-codec ceiling: `oracle_psnr = 17.0387 dB`, `ifft_psnr = 15.5356 dB`
- `AQ-YINYANG-2` used all `16 / 16` FFT coefficients per patch and removed the encoding ceiling: `oracle_psnr = 80.0000 dB`
- `AQ-YINYANG-2` still reconstructed at only `ifft_psnr = 21.4153 dB`, despite strong coefficient cosines and perfect semantic identity recovery
- `AQ-YINYANG-3` added direct pixel-space ridge decoding and barely moved reconstruction: `pixel_ridge_psnr = 21.4213 dB`

Conclusion:

The image can be represented losslessly in the all-coefficient semanticized k-space codec, and D-LinOSS recovers the folded coefficient state strongly. The remaining bottleneck is not just loss-function mismatch in the decoder. Small local coefficient-value and luminance errors still produce visible image distortion after unfolding, even when decoding directly in pixel space.

## AQ-YINYANG-4 Raw Image-State Gate

`AQ-YINYANG-4` now supports user-provided `taichi_*.png` reference inputs. The ready-node launcher uploads only those explicit PNG inputs plus the AP source file, preserving the TRC-safe workflow.

Latest completed gate:

- input: `tpu_previews/yinyang_reference_v2/taichi_16x16_frame_1.png`
- mode: `--aq-yinyang-4`
- payload: raw `4 x 4` pixel patches
- patch states: `16`
- controller: `complex_pair_norm`
- motif recovery: `1.0000`
- pixel PSNR: `30.0046 dB`
- luma MAE: `0.0123`

Important implementation correction:

- preview frame PNGs must be assembled by `patch_true_indices`, not the randomized recovery/test order
- corrected images are under `tpu_previews/yinyang_ref16_frame1_ordered/`

Second one-frame polarity control:

- input: `tpu_previews/yinyang_reference_v2/taichi_16x16_frame_2.png`
- mode: `--aq-yinyang-4 --yinyang-frame-start 2`
- payload: raw `4 x 4` pixel patches
- patch states: `16`
- controller: `complex_pair_norm`
- motif recovery: `1.0000`
- pixel PSNR: `29.8073 dB`
- luma MAE: `0.0133`
- noisy cosine: `0.9963`

Interpretation:

Both polarity frames are individually recoverable through the raw patch-state D-LinOSS path at roughly `30 dB` PSNR. That validates the image-state transport control, but it does not yet test ordered two-frame folding or sequential assembly.

Ordered two-frame recovery:

- inputs: `tpu_previews/yinyang_reference_v2/taichi_16x16_frame_1.png` and `tpu_previews/yinyang_reference_v2/taichi_16x16_frame_2.png`
- mode: `--aq-yinyang-4 --yinyang-frame-count 2 --yinyang-ordered-recovery`
- payload: raw `4 x 4` pixel patches
- patch states: `32`
- controller: `complex_pair_norm`
- motif recovery: `1.0000`
- pixel PSNR: `25.0450 dB`
- luma MAE: `0.0194`
- noisy cosine: `0.9843`
- control margin: `+0.0533`

Artifacts:

- `tpu_previews/yinyang_ordered2/aq_frame_depth3_noise0.0_seed11_complex_pair_norm_taichi_16x16_frame_1_compare.png`
- `tpu_previews/yinyang_ordered2/aq_frame_depth3_noise0.0_seed11_complex_pair_norm_taichi_16x16_frame_2_compare.png`
- `tpu_previews/yinyang_ordered2/program_ap_summary.json`

Interpretation:

The ordered recovery fix worked. The recovered images now correspond to the intended frame sequence, so the current limitation is no longer ordering or patch assembly. The meaningful drop from the one-frame controls to the two-frame roster points instead to joint recovery pressure from the larger shared state set.

Implementation note:

- raw `AQ-YINYANG-4` now sets `fast_codec_probe = True`
- this preserves the visual sanity outputs while trimming unnecessary adversarial-control work from future raw image-state checks

Next implementation step:

- compare ordered two-frame quality against two independently recovered one-frame outputs
- decide whether the next run should split frame identity and image body into separate channels before reintroducing temporal coupling

## AQ-STREAM-0 Entangled-Pair Streaming Gate

The roadmap now splits:

| branch | purpose | next action |
| --- | --- | --- |
| folded-state teleportation | revisit semantic folding with stronger encoding tools such as FFT/TTN/MERA | pause |
| entangled-pair streaming | test whether structured pair resources improve ordered frame recovery | run `AQ-STREAM-0` |

`AQ-STREAM-0` keeps the successful raw patch-state channel and asks whether ordered two-frame recovery improves when time, patch position, frame polarity, and pair correlations are explicit internal modes.

The three arms are:

- `flat_ordered_roster`
- `separable_mode_stream`
- `entangled_pair_stream`

Primary gate:

```text
entangled_pair_stream mean_psnr > 25.0450 dB
cross_frame_leakage decreases
patch_position_accuracy remains 1.0000
```

Strong gate:

```text
entangled_pair_stream mean_psnr >= 27 dB
```

Implementation status:

- added `--aq-stream-0`
- added stream-specific complex oscillator payload tables
- added time/patch/polarity/pair-resource metrics
- added run spec at `emergent_quantum_geometries/docs/PROGRAM_AQ_STREAM0_RUN_SPEC_2026-06-04.md`

Completed AQ-STREAM-0 readout:

| arm | mean PSNR | frame 1 | frame 2 | order | patch | leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat_ordered_roster` | `25.9232` | `25.7380` | `26.1167` | `1.0000` | `1.0000` | `0.0000` |
| `separable_mode_stream` | `28.4754` | `28.0911` | `28.8970` | `1.0000` | `1.0000` | `0.0000` |
| `entangled_pair_stream` | `17.5098` | `17.5350` | `17.4848` | `1.0000` | `1.0000` | `0.0000` |

Conclusion:

`AQ-STREAM-0` supports the streaming branch, but not the first entangled-pair body. The separable stream arm showed that explicit time/patch/polarity labels improve ordered two-frame recovery by about `+2.55 dB` over the flat baseline. The entangled arm failed despite perfect order and zero frame leakage, which points to destructive pair encoding rather than a failure of stream ordering.

Quick correction:

- keep the active TPU and same launch envelope
- replace `raw_patch + i * partner_patch` with a raw-preserving real pair-residual packet
- rerun `--aq-stream-0` to test whether pair information helps once it is auxiliary rather than occupying the source patch's imaginary quadrature

Follow-up result:

| arm | mean PSNR | frame 1 | frame 2 | order | patch | leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat_ordered_roster` | `25.9448` | `25.7323` | `26.1682` | `1.0000` | `1.0000` | `0.0000` |
| `separable_mode_stream` | `28.4640` | `28.0736` | `28.8930` | `1.0000` | `1.0000` | `0.0000` |
| `entangled_pair_stream` v2 | `29.2137` | `28.5450` | `30.0043` | `1.0000` | `1.0000` | `0.0000` |

The correction worked. The stream branch now has evidence that pair-resource information improves ordered visual recovery when it is encoded as an auxiliary real residual while preserving the source patch body. The first entangled result failed because the partner patch was quadrature-coupled into the source slot, not because pair resources are unusable.

Artifacts:

- `tpu_previews/aq_stream0_v2_2026-06-04/program_ap_summary.json`
- `tpu_previews/aq_stream0_v2_2026-06-04/aq_frame_depth3_noise0.0_seed11_entangled_pair_stream_taichi_16x16_frame_1_compare.png`
- `tpu_previews/aq_stream0_v2_2026-06-04/aq_frame_depth3_noise0.0_seed11_entangled_pair_stream_taichi_16x16_frame_2_compare.png`
