# Program AN Remote Analysis (2026-05-26)

Run context:
- Zone: `europe-west4-a` (TRC-covered `v6e` spot)
- Node type: `v6e-4`
- Runtime: `v2-alpha-tpuv6e`
- Runtime duration: `1752.8 s` (~29.2 min)
- Analysis method: remote JSON read over SSH (`cat`), no artifact download/scp

## What completed

The run completed all three sections:
1. Seed/controller sweep (`depth=3`, `nodes=264`)
2. Held-out compressed topology transport
3. Semantic folded-field recovery

## Key results

### Held-out transport (selected)

Structured condition:
- `structured/tunneling_stabilized`: `heldout_graph_cosine=0.9517`, `lift_over_mean=+0.0832`, `lift_over_shuffled=+0.1010`
- `structured/filament_stabilized`: `heldout_graph_cosine=0.9298`, `lift_over_mean=+0.0968`, `lift_over_shuffled=+0.1105`

Negative controls:
- `random_crofton/free`: `heldout_graph_cosine=0.8019`, `lift_over_mean=-0.0112`, `lift_over_shuffled=-0.0317` (good negative behavior)
- `shuffle_topology/tunneling_stabilized`: `heldout_graph_cosine=0.9676`, `lift_over_mean=+0.1093`, `lift_over_shuffled=+0.1962` (undesired; this control remains too strong)

Interpretation:
- Crofton randomization is a meaningful negative test now.
- Topology shuffle is still not destructive enough for the current manifold encoding; it can still score as well as or better than structured.

### Semantic folded-field recovery

- `free`: `clean=0.9596`, `noisy=0.9596`, `noisy_lift_over_mean=+0.0820`, `noisy_lift_over_topology_shuffle=+0.0754`, `noisy_lift_over_phase_scramble=+0.0000`
- `tunneling_stabilized`: `clean=0.9277`, `noisy=0.8416`, `noisy_lift_over_mean=-0.0094`, `noisy_lift_over_topology_shuffle=-0.0366`, `noisy_lift_over_phase_scramble=+0.0160`
- `filament_stabilized`: `clean=0.9328`, `noisy=0.8840`, `noisy_lift_over_mean=+0.0343`, `noisy_lift_over_topology_shuffle=+0.1019`, `noisy_lift_over_phase_scramble=-0.0059`

Interpretation:
- Recovery is partially successful: `filament_stabilized` improves against topology-shuffle but not against phase-scramble.
- `tunneling_stabilized` degrades under noisy recovery in this configuration.
- The `free` controller appears robust in this setup, suggesting the perturbation model still under-penalizes trivial dynamics.

## Verdict against current goals

Progress:
- We now have a working remote-only TPU loop with complete summary generation.
- We now have at least one valid negative control (`random_crofton`) and a completed semantic recovery readout.

Not yet solved:
- The central objective ("topology carries semantic folded state better than controls") is not decisively met.
- `shuffle_topology` still passes too often and can outperform structured settings.
- Recovery metrics are mixed and controller-dependent.

## Recommended next run (short list)

1. Strengthen topology-destruction control:
   - Preserve symbol counts and first-order transitions, but break higher-order recurrence geometry (cycle-preserving Markov shuffle with block permutation).
2. Increase recovery perturbation severity:
   - Sweep `recovery_noise` (e.g., `0.18, 0.30, 0.45`) and track where each controller collapses.
3. Add a stricter win criterion:
   - Require structured controller to beat both `shuffle_topology` and `phase_scramble` on noisy recovery lift in the same run.
4. Keep remote-first operations:
   - Continue SSH-based analysis and summary-only artifacts to minimize egress charges in Europe zones.
