# Program AP - Tunnel-Eigen Guided Folded Recovery

## Core hypothesis

Folded semantic recovery improves when D-LinOSS unfolds the compressed manifold through stable tunneling eigenmodes before semantic graph decoding.

## Why AP follows AO

The latest AO result confirmed the diagnosis:

- `filament_stabilized` at `noise=0.45` moved from `-0.0804` to `+0.0686` on the primary adversarial margin after replacing Crofton-only feedback with tunnel-affinity coupling.
- That is a `+0.1490` swing on the hardest adversary margin.
- The improvement suggests the coupling corruption problem was real, and tunnel-affinity coupling is the right base for the next experiment.

## What AP keeps

- Tunnel-affinity coupling.
- Low-alpha / high-beta scoring.
- `block_permute` as primary adversary.
- `ibm_lite_folded` as hardware-constrained comparison.
- Remote-first summary analysis to avoid Europe-to-local egress.

## What AP adds

1. `tunnel_eigen_unfolded`
   - Uses the same tunnel-affinity structure as `filament_stabilized`.
   - Projects coupling through top-`k` eigenmodes of the node-level tunnel affinity.
   - Scales D-LinOSS damping by dominant eigenmode support.

2. Stronger block adversary
   - `block_permute_size2`
   - `block_permute_size2_within_random`

3. Primary KPI
   - Lead with `worst_control_margin`.
   - Success condition: `tunnel_eigen_unfolded worst_control_margin > 0` at `noise=0.45` across at least `2/3` seeds.

## Minimal TPU-safe run

- Depth: `3`
- Noise: `0.30`, `0.45`
- Controllers: `free`, `filament_stabilized`, `ibm_lite_folded`, `tunnel_eigen_unfolded`
- Seeds: `11`, `29`, `47`
- Primary control: `block_permute_size2_within_random`

## Expected interpretation

If AP succeeds, the claim strengthens from "filament stabilization helps" to:

> Folded semantic recovery is improved by unfolding through stable tunneling eigenmodes, suggesting recovery is driven by transport-mode geometry rather than symbol leakage or superficial recurrence statistics.
