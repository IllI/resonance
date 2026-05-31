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
- Runtime: `1832.8 s` (~30.5 minutes)
- Records: `1` depth record (`depth=3`)
- Recovery sweep completed for noise: `0.30`, `0.45`

## Key outcomes

The success condition for AO was directional:

- Desired ordering: `folded_decoder > filament_stabilized > free`
- Desired control wins: folded decoder should beat hard controls with positive margins.

Observed summary:

1. `free` remained competitive, but the new scoring made its wins much smaller and easier to interpret.
2. `ibm_lite_folded` was strongest at `noise=0.30`, which is the cleanest hardware-path signal in the run.
3. `filament_stabilized` improved versus the earlier AO pass, suggesting the tunnel-affinity coupling change helped.
4. `folded_decoder` improved at `noise=0.45` and achieved a small positive minimum hard-control lift there.
5. `block_permute` remained the hardest adversary and is the control to keep sharpening.

## Detailed metrics (remote summary)

`noise=0.30`

- `free`: score `0.3599`, noisy cosine `0.9109`, min hard-control lift `+0.0616`
- `filament_stabilized`: score `0.2841`, noisy cosine `0.8230`, min lift `-0.0012`
- `folded_decoder`: score `0.2681`, noisy cosine `0.8759`, min lift `-0.0279`
- `ibm_lite_folded`: score `0.3475`, noisy cosine `0.9430`, min lift `+0.0585`

`noise=0.45`

- `free`: score `0.3450`, noisy cosine `0.9772`, min hard-control lift `+0.0281`
- `filament_stabilized`: score `0.3332`, noisy cosine `0.8964`, min lift `+0.0686`
- `folded_decoder`: score `0.2903`, noisy cosine `0.9169`, min lift `+0.0026`
- `ibm_lite_folded`: score `0.2759`, noisy cosine `0.8858`, min lift `-0.0402`

## Interpretation versus research objective

This focused AO run is a **mixed but useful negative/diagnostic result**:

- Evidence for robust structured-recovery superiority is not yet established because `free` still remains competitive under the tighter score.
- `ibm_lite_folded` is the clearest hardware-constrained signal at `noise=0.30`, so that path looks worth preserving.
- `filament_stabilized` improved relative to the prior AO pass, which supports the tunnel-affinity coupling fix.
- `folded_decoder` is still not dominant, but it now has a small positive margin at `noise=0.45`, so it is not dead; it just needs a stronger adversary and better mode guidance.

## Practical course-correction

1. Rebalance the folded score to penalize trivial/free wins:
   - keep `alpha` low and keep `beta` dominant.
2. Strengthen controls further in recovery:
   - keep `block_permute` as primary adversary and add a stricter `block_size=2` / within-block-randomization variant.
3. Add per-control confidence intervals across more seeds in focused mode:
   - current focused run used no trial sweep; add a lightweight 2-3 seed recovery-only repeat once the adversary is sharpened.
4. Keep `ibm_lite_folded` path:
   - it is already competitive and likely the most publishable bridge to constrained hardware claims.

## Experiment thread summary so far (AA -> AO focused)

- Earlier programs established that relational/geometry-aware control outperforms naive symbol-level decoding under several perturbations.
- AO-focused now tests hard controls directly in the recovery decoder.
- Current result: partial support for folded-topology recovery under noise, but not yet a decisive dominance result over free baseline.
- Next milestone: show repeated positive **minimum hard-control lift** for a tunnel-guided decoder across seeds and at least one moderate/high noise regime while keeping IBM-lite constraints.

## Current decoding problem (why folded recovery is still unstable)

The central issue is that the current decoder can still win via shallow geometric/statistical shortcuts instead of true manifold unfolding. In practice:

1. `free` baseline remains too competitive, meaning the reconstruction target is still partially recoverable without a strong topology-preserving observer.
2. `filament_stabilized` is better after the tunnel-affinity fix, but it still does not fully dominate the hardest control.
3. `folded_decoder` improves at higher noise but does not consistently dominate all hard controls, so the latent-to-semantic inverse map is not yet robust.
4. The current folded score is much harder to game than before, but it still needs an even sharper adversary to expose the weakest path.

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
   - Require positive margin against the strongest hard control, then report the full control table as a secondary diagnostic.
   - Report worst-control margin as primary KPI; noisy cosine becomes secondary.

## Concrete next-run spec (fast, TPU-safe)

Use a recovery-only focused run with minimal compile churn:

- Depth: `3`
- Noise: `0.30`, `0.45`
- Controllers: `free`, `folded_decoder`, `ibm_lite_folded`, `tunnel_eigen_unfolded` (new)
- Seeds: `11`, `29`, `47` for a lightweight recovery-only repeat
- Hard controls in decoder path: `block_permute` as the primary adversary, with the other controls kept for diagnostics
- Primary endpoint: positive worst-control margin for `tunnel_eigen_unfolded` at `0.45` with reproducibility across seeds

If this holds, we get a stronger claim: recovery is being driven by stable transport geometry (tunnel eigenstructure), not symbol leakage or shallow recurrence artifacts.
