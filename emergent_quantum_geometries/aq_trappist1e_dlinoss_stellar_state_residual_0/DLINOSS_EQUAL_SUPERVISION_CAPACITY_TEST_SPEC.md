# AQ-DLINOSS-EQUAL-SUPERVISION-CAPACITY-TEST-0

## Status

Executed under the locked manifest on 8 TPU devices. The result is recorded in `results/dlinoss_equal_supervision_capacity_test_0_tpu.json` with 108 sweep rows and verdict `DO NOT PROMOTE`. The TRAPPIST/WASP-39b astronomy branch remains closed.

The test established high in-distribution synthetic capacity but did not clear the predeclared practical gates: improvement over the equal-supervision linear SSM was below 5%, mask-shuffle degradation was below 20%, and leave-window-out recovery failed. This result is independent of the Program 1331 transmission-spectrum extraction gap.

## Purpose

Determine whether physics-locked residual D-LinOSS can recover structured latent dynamics as well as or better than a linear SSM when both receive identical inputs, targets, train/test splits, and supervision.

## Dataset

Use a versioned masked synthetic generator with:

- full spectral input `[batch, time, 96]`
- known latent state `z`
- known full hidden residual
- ordered partial-observation masks
- an explicit timestamp grid and ground-truth transition recorded in the shard
- identical train, validation, and test seeds for both models

No JWST, MAST, WASP-39b, or TRAPPIST data are allowed.

### Locked units and pre-generation manifest

The authoritative pre-generation manifest is `configs/dlinoss_equal_supervision_capacity_test_0_manifest.json`. Its locked SHA-256 is recorded in `configs/dlinoss_equal_supervision_capacity_test_0_manifest.sha256`. It must be committed before shard generation. Once `shard_generated` becomes true, changing a locked field requires a new manifest version and a new run name.

Locked values:

- timestamp unit: `day`
- all timestamp arrays: `float64` days
- `tau_ref`: exactly `1.0 day` (`86400.0 seconds`)
- nominal visit separation: exactly `30.0 days`
- nominal transit duration: exactly `37.0 minutes` (`37/1440 days`)

Both frequency arms receive the same timestamp array and `dt` values in days. No arm may infer a time-unit conversion from generated values.

## Generator transition class

The generator must not be a pure linear SSM or a pure uncoupled oscillator. Use a damped oscillatory backbone with weak nonlinear cross-mode coupling and exogenous process noise:

```text
z_pred[t] = A_osc(dt[t]) z[t-1]
z[t] = z_pred[t] + eta * tanh(M z[t-1]) + B u[t] + q[t]
x_hidden[t] = D1 z[t] + beta * tanh(D2 z[t])
x_obs[t] = mask[t] * x_hidden[t] + smooth_nuisance[t] + epsilon[t]
```

Required constraints:

- `A_osc` is block diagonal before coupling, with at least three distinct damped rotation modes.
- `eta` and `beta` are nonzero but bounded so neither the linear nor oscillatory component dominates variance.
- Report variance fractions from the oscillatory, nonlinear, nuisance, and observation-noise components.
- Different latent components are visible through different wavelength windows.
- Mask order alternates single-window and multi-window observations.
- The same generated shard is used by every model and frequency arm.
- Generator frequencies and transition parameters are committed to the result manifest before training.

The linear SSM therefore has an advantage on the linear component, while D-LinOSS has an advantage only if fixed oscillatory structure and nonlinear state mixing are genuinely useful.

## Equal-supervision arms

Run two separate comparisons; do not mix their conclusions.

### Latent-supervised arm

Both models optimize the same weighted objective:

`latent MSE + residual-decoder MSE`

Both may use true training `z`; neither may use test `z` except for scoring.

### Residual-self-supervised arm

Both models optimize residual prediction only. Neither model may fit transition, observation, gain, or decoder parameters directly against true `z`. Latent recovery is measured afterward using the same held-out linear-probe protocol.

## Frequency-prior arms

Run both supervision comparisons under two D-LinOSS frequency conditions. The linear SSM result is shared and unchanged.

### Oracle-frequency D-LinOSS

Freeze imaginary eigenfrequencies to the generator's committed ground-truth oscillatory frequencies. This arm measures best-case architecture capacity and must be labeled `oracle_frequency`; it cannot establish prior robustness.

### Physical-prior D-LinOSS

Freeze imaginary eigenfrequencies without reading the generator transition or frequency manifest. Use the independently specified temporal prior:

- stellar rotation: `2*pi / 3.3 days`
- active-region evolution: `2*pi / 7 days`
- active-region evolution: `2*pi / 14 days`
- visit-scale drift: `2*pi / nominal visit separation`
- transit/chord mode: `2*pi / nominal transit duration`

This arm must use an explicit timestamp unit so frequencies are converted without inferring or fitting a scale from the generated data. Hyperparameter selection may not alter these frequencies.

For this run, compute `omega_fixed` in radians per day from the locked periods:

- `3.3 days`
- `7.0 days`
- `14.0 days`
- `30.0 days`
- `37/1440 days`

The resulting vector is fixed before shard generation and repeated deterministically to fill the hidden dimension.

Radiative `Gamma` values are not temporal imaginary eigenfrequencies. They may constrain real damping/line-width priors or wavelength weighting, but they must not be inserted into `omega_fixed`.

### Frequency information firewall

- The physical-prior arm may not inspect generator eigenvalues, latent autocorrelation spectra, or oracle-frequency validation results.
- Frequency-arm labels and hashes must be stored in every result.
- A frequency perturbation diagnostic of `0.5x`, `1.0x`, and `2.0x` may be reported, but only the predeclared physical prior participates in promotion.

## Fixed architecture contracts

- D-LinOSS receives `[T, 96]` residual spectra.
- Imaginary eigenfrequencies are fixed and absent from trainable parameters.
- Real damping terms are learned.
- The first 30% of timesteps update state but contribute no main loss.
- State assimilation uses the explicit stability parameterization below.
- Linear SSM uses the same observation masks and residual target.
- Both models use identical parameter-selection data and stopping rules.

## Stability parameterization

For D-LinOSS mode `k`, use a diagonal continuous-time pole converted exactly to discrete time:

```text
tau_ref = 1.0 day
gamma_k = softplus(rho_k) / tau_ref
lambda_k(dt) = exp((-gamma_k + i * omega_fixed_k) * dt)
```

Thus `abs(lambda_k) < 1` for positive `dt`; `rho_k` is trainable and `omega_fixed_k` is external and gradient-free.

Assimilation is incremental:

```text
h_pred = lambda(dt) * h_prev
innovation = mask * (x_obs - decode_obs(h_pred))
alpha_t = alpha_min + (alpha_max - alpha_min) * sigmoid(g(mask, h_pred))
h_next = h_pred + alpha_t * K(mask) * innovation
```

Use `alpha_min = 0.05` and `alpha_max = 0.45`. Constrain each mask-conditioned gain matrix to spectral norm <= 1.0. No code path may set `transport_scale = 1.0`, replace `h_pred` wholesale, or make `omega_fixed` trainable.

## Bounded sweep

Run the Cartesian product independently for each supervision and frequency-prior arm:

- `hidden_dim`: 20, 48, 96
- `training_steps`: 800, 2000, 4000
- seeds: 11, 23, 31

Select hyperparameters on validation data only. Report every configuration and seed; do not promote a best-seed result.

This is 27 configurations per model/arm, not 27 total. TPU execution may batch eight configurations at a time, but incomplete batches must be padded or run separately without dropping configurations.

## Required baselines and controls

- linear SSM
- feedforward/no-recurrence baseline
- smooth residual baseline
- persistence baseline
- time-order shuffle
- mask-order shuffle
- leave-one-window-out evaluation
- frozen/static operator
- label/target permutation false-positive control

## Promotion gates

The numerical gates below apply separately to the oracle-frequency and physical-prior arms, across the selected configuration and all held-out seeds:

- residual correlation >= 0.88
- latent correlation >= 0.87
- mean residual improvement over equal-supervision linear SSM >= 5%
- lower 95% confidence bound on improvement > 0
- mask-shuffle degradation >= 20%
- time-shuffle degradation >= 20%
- leave-window-out recovery >= 0.75
- false-positive rate <= 0.05

Promotion interpretation:

- Both frequency arms pass: `PROMOTE_ROBUST_PHYSICS_LOCKED_DLINOSS_CAPACITY`.
- Oracle-frequency passes and physical-prior fails: `PROMOTE_ORACLE_FREQUENCY_CAPACITY_ONLY`; architecture capacity is established, but prior robustness is not and astronomy remains blocked.
- Physical-prior passes while oracle-frequency fails: `STOP_SYNTHETIC`; this indicates an implementation, selection, or leakage error.
- Neither passes and linear SSM wins: `PROMOTE LINEAR_SSM_BASELINE_ONLY`.

## Allowed verdicts

- `PROMOTE_ROBUST_PHYSICS_LOCKED_DLINOSS_CAPACITY`
- `PROMOTE_ORACLE_FREQUENCY_CAPACITY_ONLY`
- `PROMOTE LINEAR_SSM_BASELINE_ONLY`
- `INCONCLUSIVE_OPTIMIZATION`
- `DO NOT PROMOTE`
- `STOP_SYNTHETIC`

## Verdict precedence and optimization criteria

Evaluate verdicts in this order:

1. `STOP_SYNTHETIC`
2. `INCONCLUSIVE_OPTIMIZATION`
3. promotion or `DO NOT PROMOTE`

`STOP_SYNTHETIC` is mandatory if any manifest/hash/unit check fails, physical-prior D-LinOSS passes while oracle-frequency D-LinOSS fails, target permutation exceeds the false-positive gate, or every configuration for a required model/arm produces non-finite loss or gradients.

A seed/configuration is optimization-valid only if:

- every requested step completed;
- training and validation losses and gradients remained finite;
- validation loss improved by less than 1% over the final 20% of steps, measured as `(loss_at_80pct - best_final_20pct) / abs(loss_at_80pct)`;
- its selected checkpoint predates test evaluation and was chosen from validation loss only.

`INCONCLUSIVE_OPTIMIZATION` is mandatory if no configuration has at least two optimization-valid seeds, if the selected configuration has across-seed residual-correlation standard deviation greater than `0.05`, or if its D-LinOSS-minus-linear-SSM improvement margin has across-seed standard deviation greater than `0.05`.

An isolated non-finite seed does not trigger `STOP_SYNTHETIC` when at least two seeds for that configuration are optimization-valid; it disqualifies that seed and may cause `INCONCLUSIVE_OPTIMIZATION` under the rules above. No failed seed may be silently restarted with a new seed.

## Interpretation boundary

Oracle-frequency passing establishes best-case synthetic representational capacity only. Astronomy remains blocked unless the independently specified physical-prior arm also passes every gate. Even robust synthetic passing does not authorize an atmosphere, stellar-contamination, molecule, time-geometry, WASP-39b, or TRAPPIST-1e claim; a separate approval is required before any astronomy run.
