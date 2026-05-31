# Program AO - Folded-Field Recovery + Hard Controls

Goal:
- Improve semantic folded-field recovery while proving recovery is not symbol-count, phase, shallow-transition, or coarse-Crofton leakage.

## Tracks

Hard falsification:
- `markov_shuffle`: preserves shallow transition statistics while disrupting higher-order recurrence.
- `block_permute`: preserves local motifs while disrupting global topology.
- `phase_scramble`: preserves amplitude/statistical carrier shape while disrupting coherent phase topology.
- `random_crofton`: preserves coarse manifold scale while breaking semantic structure.

Recovery strengthening:
- `folded_decoder`: stronger D-LinOSS folded-frame decoder.
- `ibm_lite_folded`: constrained nearest-neighbor / measurement-compatible approximation.

## Matrix

- Depths: `3, 4`
- Noise: `0.18, 0.30, 0.45`
- Controllers: `free`, `filament_stabilized`, `folded_decoder`, `ibm_lite_folded`
- Controls: `structured`, `markov_shuffle`, `block_permute`, `phase_scramble`, `random_crofton`

## Score

`folded_recovery_score = alpha * semantic_graph + beta * topology_margin + gamma * frame_consistency - delta * gate_cost`

Default weights:
- `alpha=0.45`
- `beta=0.30`
- `gamma=0.20`
- `delta=0.05`

## Success Condition

Strong result if:
- `folded_decoder > filament_stabilized > free` by folded recovery score.
- `folded_decoder` beats all hard controls under at least one noisy regime.
- `ibm_lite_folded` remains positive under at least one noisy regime, showing a plausible IBM-style path.

## TRC/Egress Discipline

- Run only in TRC-covered TPU zones/types.
- Prefer remote analysis in Europe zones.
- Do not download raw artifacts or full result folders.
- Download only tiny summaries/log tails when explicitly needed.
