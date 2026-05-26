# Program AO Focused - Remote Analysis (2026-05-26)

## Scope and TRC compliance

- Analysis performed remotely via `gcloud ... ssh --command="cat ...json"` only.
- No SCP artifact download was used for this analysis pass.
- Source summary file on TPU VM:
  - `/home/cityz/program_ao/program_ao_focused_results/program_ao_summary.json`

## Run status

- Experiment: `folded_field_recovery_hard_controls`
- Mode: `--ao-focused` (semantic recovery only, depth fixed to 3)
- Backend: TPU (`jax` on `tpu`)
- Runtime: `2713.6 s` (~45.2 minutes)
- Records: `1` depth record (`depth=3`)
- Recovery sweep completed for noise: `0.18`, `0.30`, `0.45`

## Key outcomes

The success condition for AO was directional:

- Desired ordering: `folded_decoder > filament_stabilized > free`
- Desired control wins: folded decoder should beat hard controls with positive margins.

Observed summary:

1. `ibm_lite_folded` was strongest at low noise (`0.18`) and remained competitive.
2. `free` stayed unexpectedly strong at all noise levels, often matching or beating `folded_decoder` on score.
3. `folded_decoder` improved at high noise (`0.45`) and achieved positive minimum hard-control lift there.
4. `filament_stabilized` underperformed with negative minimum hard-control lift at all three noise points.
5. The hardest adversary was typically `block_permute` (minimum margin often set by block-permute gap).

## Detailed metrics (remote summary)

`noise=0.18`

- `free`: score `0.6099`, noisy cosine `0.9109`, min hard-control lift `+0.0000`
- `filament_stabilized`: score `0.5303`, noisy cosine `0.8151`, min lift `-0.0502`
- `folded_decoder`: score `0.5758`, noisy cosine `0.9000`, min lift `-0.0187`
- `ibm_lite_folded`: score `0.6210`, noisy cosine `0.9430`, min lift `+0.0158`

`noise=0.30`

- `free`: score `0.6397`, noisy cosine `0.9772`, min lift `+0.0000`
- `filament_stabilized`: score `0.5690`, noisy cosine `0.8989`, min lift `-0.0437`
- `folded_decoder`: score `0.5574`, noisy cosine `0.8835`, min lift `-0.0347`
- `ibm_lite_folded`: score `0.5849`, noisy cosine `0.8920`, min lift `-0.0153`

`noise=0.45`

- `free`: score `0.6053`, noisy cosine `0.8999`, min lift `+0.0030`
- `filament_stabilized`: score `0.5414`, noisy cosine `0.8652`, min lift `-0.0804`
- `folded_decoder`: score `0.5924`, noisy cosine `0.9225`, min lift `+0.0063`
- `ibm_lite_folded`: score `0.5704`, noisy cosine `0.8606`, min lift `-0.0212`

## Interpretation versus research objective

This focused AO run is a **mixed but useful negative/diagnostic result**:

- Evidence for robust structured-recovery superiority is not yet established because `free` remained very strong.
- `folded_decoder` shows promising behavior at high noise (`0.45`) where it recovered a positive minimum hard-control margin.
- `ibm_lite_folded` being strongest at `0.18` supports hardware-constrained feasibility, but consistency across noise is not yet there.
- `filament_stabilized` likely needs retuning (feedback/damping/tunnel coupling or gate-cost weight).

## Practical course-correction

1. Rebalance the folded score to penalize trivial/free wins:
   - increase hard-control margin weight (`beta`) and reduce raw noisy-cosine dominance (`alpha`).
2. Strengthen controls further in recovery:
   - keep `block_permute` as primary adversary (it is the hardest control in this run).
3. Add per-control confidence intervals across more seeds in focused mode:
   - current focused run used no trial sweep; add a lightweight 2-3 seed recovery-only repeat.
4. Keep `ibm_lite_folded` path:
   - it is already competitive and likely the most publishable bridge to constrained hardware claims.

## Experiment thread summary so far (AA -> AO focused)

- Earlier programs established that relational/geometry-aware control outperforms naive symbol-level decoding under several perturbations.
- AO-focused now tests hard controls directly in the recovery decoder.
- Current result: partial support for folded-topology recovery under noise, but not yet a decisive dominance result over free baseline.
- Next milestone: show repeated positive **minimum hard-control lift** for `folded_decoder` across seeds and at least one moderate/high noise regime while keeping IBM-lite constraints.

## Current decoding problem (why folded recovery is still unstable)

The central issue is that the current decoder can still win via shallow geometric/statistical shortcuts instead of true manifold unfolding. In practice:

1. `free` baseline remains too competitive, meaning the reconstruction target is still partially recoverable without a strong topology-preserving observer.
2. `filament_stabilized` often collapses on the hardest control (`block_permute`), indicating insufficient sensitivity to higher-order recurrence structure.
3. `folded_decoder` improves at higher noise but does not consistently dominate all hard controls, so the latent-to-semantic inverse map is not yet robust.
4. The current folded score can reward high noisy cosine even when one hard control margin is negative, which permits brittle decoding wins.

Interpretation: D-LinOSS currently behaves like a good geometric smoother, but not yet like a full semantic-topology observer/inverter for compressed folded manifolds.

## Postulate for next run: tunnel-eigen-guided unfolding in D-LinOSS

Hypothesis:

- The reconstruction should be guided by a small set of **quantum tunneling eigenmodes** (dominant eigenvectors/eigenvalues of the tunneling-weighted transition operator), which define the physically stable transport channels through the folded manifold.
- If D-LinOSS unfolds in that eigenbasis first, then decodes semantics, it should reduce shortcut recovery and improve robustness under hard controls.

Proposed D-LinOSS upgrade path:

1. Build a tunnel operator per sample:
   - Use the existing barrier/tunneling kernel to form `K_tunnel`.
   - Compute top-`k` eigenmodes `(lambda_i, v_i)` with largest persistent transport contribution.
2. Add an eigen-projected observer state:
   - Project latent signatures into `V_k` (tunnel eigenbasis).
   - Run D-LinOSS dynamics in this reduced basis with mode-wise damping/feedback tied to `lambda_i`.
3. Add unfolding head before semantic decode:
   - Head A reconstructs unfolded latent trajectory from compressed signature.
   - Head B decodes semantic graph from unfolded trajectory (not directly from compressed state).
4. Add reconstruction consistency constraints:
   - Cycle loss: folded -> unfolded -> refolded should preserve topology class.
   - Mode occupancy regularizer: semantic decode must depend on stable transport modes, not a single shallow statistic.
5. Tighten success metric:
   - Require positive margin against each hard control (`markov`, `block`, `phase`, `crofton`) per noise level.
   - Report worst-control margin as primary KPI; noisy cosine becomes secondary.

## Concrete next-run spec (fast, TPU-safe)

Use a recovery-only focused run with minimal compile churn:

- Depth: `3`
- Noise: `0.30`, `0.45`
- Controllers: `free`, `folded_decoder`, `ibm_lite_folded`, `tunnel_eigen_unfolded` (new)
- Seeds: `11`, `29`, `47` (recovery-only; skip full trial sweep)
- Hard controls in decoder path: `markov_shuffle`, `block_permute`, `phase_scramble`, `random_crofton`
- Primary endpoint: positive worst-control margin for `tunnel_eigen_unfolded` at `0.45` with reproducibility across seeds

If this holds, we get a stronger claim: recovery is being driven by stable transport geometry (tunnel eigenstructure), not symbol leakage or shallow recurrence artifacts.
