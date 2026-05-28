# Program AP-S-Fidelity Remote Analysis (2026-05-27)

## Scope

- Source: remote SSH reads only
- No SCP or artifact download used
- Summary analyzed in place on Europe TPU via `cat`
- Run: `Program AP - Tunnel-Eigen Guided Folded Recovery`
- Node: `program-ap-eu-node-v1`
- Zone: `europe-west4-a`

## Run configuration

- Mode: `--ap-s-fidelity`
- Backend: TPU via JAX
- Runtime: `5543.5 s` (~92.4 minutes)
- Depth: `3`
- Noise: `0.30`, `0.45`
- Seeds: `11`, `29`, `47`
- Controllers:
  - `tunnel_eigen_residual_alpha045`
  - `tunnel_eigen_residual_APR`
  - `tunnel_eigen_residual_alpha065`
- Primary adversary:
  - `block_permute_size2_within_random`
- Primary KPI:
  - `worst_control_margin`

## Final AP-S-Fidelity results

### `noise=0.30`

`seed=11`

- `alpha045`: margin `+0.0517`, noisy cosine `0.9435`
- `alpha055/APR`: margin `+0.0803`, noisy cosine `0.8450`
- `alpha065`: margin `+0.0411`, noisy cosine `0.9448`

`seed=29`

- `alpha045`: margin `+0.0517`, noisy cosine `0.9095`
- `alpha055/APR`: margin `+0.0762`, noisy cosine `0.9407`
- `alpha065`: margin `+0.0729`, noisy cosine `0.8773`

`seed=47`

- `alpha045`: margin `+0.0289`, noisy cosine `0.9235`
- `alpha055/APR`: margin `+0.1007`, noisy cosine `0.8902`
- `alpha065`: margin `+0.0618`, noisy cosine `0.9023`

### `noise=0.45`

`seed=11`

- `alpha045`: margin `+0.0633`, noisy cosine `0.8780`
- `alpha055/APR`: margin `+0.0258`, noisy cosine `0.9432`
- `alpha065`: margin `+0.0469`, noisy cosine `0.9127`

`seed=29`

- `alpha045`: margin `+0.0957`, noisy cosine `0.9041`
- `alpha055/APR`: margin `+0.0183`, noisy cosine `0.9275`
- `alpha065`: margin `+0.0757`, noisy cosine `0.9352`

`seed=47`

- `alpha045`: margin `+0.0047`, noisy cosine `0.8801`
- `alpha055/APR`: margin `+0.0287`, noisy cosine `0.8675`
- `alpha065`: margin `+0.1025`, noisy cosine `0.9611`

## What the bracket proved

All three alpha variants stayed positive on all `6/6` seed-noise cells. The fidelity probe did not expose a hidden collapse mode in either direction.

The important result is a crossover, not a universal fixed-alpha winner:

| controller | noise=0.30 mean | noise=0.30 min | noise=0.45 mean | noise=0.45 min |
| --- | ---: | ---: | ---: | ---: |
| `alpha045` | `0.044` | `+0.029` | `0.054` | `+0.005` |
| `alpha055/APR` | `0.086` | `+0.076` | `0.024` | `+0.018` |
| `alpha065` | `0.059` | `+0.041` | `0.075` | `+0.047` |

At `noise=0.30`, `alpha055/APR` is the clear margin winner. At `noise=0.45`, `alpha065` is the clear margin winner and also keeps strong noisy cosine.

## Interpretation

AP-S-MECH established that the residual tunnel-eigen mechanism reproduces. AP-S-Fidelity now shows that the best projection weight depends on recovery noise.

Mechanistically:

- At lower noise, the transport geometry is still intact enough that `alpha=0.55` exploits eigen guidance while retaining useful residual semantic signal.
- At higher noise, the residual path is more contaminated, so the larger eigen projection weight in `alpha=0.65` improves the hard-control margin.

This supports a noise-adaptive rule rather than promoting `alpha=0.65` as a universal reference:

```python
projection_alpha = 0.50 + 0.30 * noise_level
```

That gives:

- `alpha=0.59` at `noise=0.30`
- `alpha=0.635` at `noise=0.45`

## What this means for the next run

The clean next move is Program AP-T:

- `tunnel_eigen_residual_APR`
- `tunnel_eigen_residual_alpha065`
- `tunnel_eigen_residual_adaptive`

AP-T should keep the same compact matrix:

- depth `3`
- noise `0.30`, `0.45`
- seeds `11`, `29`, `47`
- primary adversary `block_permute_size2_within_random`

Primary KPI:

- `worst_control_margin`

Success criteria:

1. Adaptive beats `alpha065` at `noise=0.30` on mean margin.
2. Adaptive matches or beats `alpha065` at `noise=0.45`.
3. Adaptive stays positive on all `6/6` cells.
4. Adaptive keeps noisy cosine above `0.80` on all cells.

## Bottom line

AP-S-Fidelity did not just tune alpha. It exposed a noise-dependent tradeoff between eigen-mode guidance and residual semantic preservation.

The next defensible claim to test is:

> The optimal residual tunnel-eigen projection weight scales with recovery noise, preserving moderate-noise residual signal while suppressing high-noise residual contamination.
