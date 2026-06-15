# AQ-PAGE-WOOTTERS-DLINOSS-0 Results - 2026-06-15

## Purpose

This bridge experiment replaced the ridge-affine transition surrogate with a JAX-native implementation of the repository's complex diagonal D-LinOSS recurrence while holding the validated relational-clock evaluator fixed.

The experiment retained:

```text
calibration-only lag and clock selection
constrained monotone correspondence paths
one true-arm fit frozen before payload evaluation
controls that alter correspondence paths without refitting the model
heldout delta-transition cosine scoring
```

The tested D-LinOSS recurrence was:

```text
h_t = lambda * h_(t-1) + x_t B
y_t = Re(h_t C) + b
lambda_k = scale_k * exp(i omega_k)
```

This is the complex diagonal state-space core used by the repository's D-LinOSS implementation. It does not include the full Flax damping-matrix or MoQE classification head.

## TPU Configuration

```text
backend: TPU, 8 devices
time_steps: 128
calibration_steps: 64
latent_dim: 8
hidden_dim: 16
context_steps: 8
epochs: 800
seed: 11
noise: 0.00
```

Three calibration-selected generator variants were compared:

```text
dlinoss_diag_phase
dlinoss_hamiltonian
dlinoss_hamiltonian_plus_scale
```

`dlinoss_hamiltonian_plus_scale` won calibration selection in both modes.

## Results

### 0a Evaluator Sanity

```text
D-LinOSS relational score:     0.994504
ridge relational score:        0.999983
D-LinOSS relational gain:      1.189736
D-LinOSS minus ridge:          -0.005479
paired gain CI95:              [0.958223, 1.109748]
windows beating best null:     98.4%
```

All bridge gates passed. The D-LinOSS recurrence recovered the intentionally easy known-shift transition nearly as well as ridge.

### 0b State-Coupled Relational Clock

```text
D-LinOSS relational score:                 0.957300
ridge relational score:                    0.975130
D-LinOSS relational gain:                  0.669980
ridge relational gain:                     0.618502
D-LinOSS minus ridge:                      -0.017829
D-LinOSS relational-minus-external gain:   0.544691
paired gain CI95:                          [0.528529, 0.640926]
windows beating best null:                 98.4%
```

All bridge gates passed:

```text
Gate -1: evaluator sanity
Gate 0: stable calibration training
Gate 1: D-LinOSS relational path beats controls
Gate 2: D-LinOSS relational path beats external carrier
Gate 3: D-LinOSS remains within 0.05 of ridge
Payload gate: positive paired CI and >= 70% window wins
```

## Scientific Interpretation

The validated correspondence mechanism survives replacement of the ridge surrogate with a complex recurrent D-LinOSS transition model. D-LinOSS is slightly less accurate on the true relational path than ridge, but it produces a larger separation from the best null:

```text
D-LinOSS relational gain: 0.669980
ridge relational gain:    0.618502
```

This is evidence that the recurrent complex state-space model can learn and exploit the relationally aligned transition structure in this controlled synthetic history simulation.

The result does not establish physical Page-Wootters dynamics, emergent spacetime, or tensor-network history dynamics. The generated histories remain projected harmonic latent trajectories, and robustness has not yet been demonstrated across seeds, noise levels, sequence lengths, or latent dimensions.

## Next Gate

Run the preregistered robustness grid before promoting the mechanism:

```text
seeds:       11, 12, 13, 14, 15
time_steps: 128, 256, 512
noise:      0.00, 0.01, 0.03
latent_dim: 8, 16
```

Primary criteria:

```text
0a passes consistently
0b D-LinOSS relational gain > 0.10 in most runs
D-LinOSS relational-minus-external gain > 0.05 in most runs
paired bootstrap CI lower bound > 0 in most runs
D-LinOSS minus ridge >= -0.05
```

Only after robustness should the synthetic harmonic history generator be replaced with an explicit tensor-network history construction.

## Robustness Stages A-B

The staged robustness run held `time_steps=128` and `latent_dim=8` fixed while testing five seeds at each noise level.

### Stage A - Noise 0.00

All promotion criteria passed:

```text
0a all-gate pass rate:                    100%
0b relational gain > 0.10:               100%
relational-minus-external > 0.05:         80%
paired CI lower bound > 0:                100%
D-LinOSS minus ridge >= -0.05:            100%
positive curvature lift:                  100%
```

Aggregate medians and IQRs:

```text
relational score:             0.968738  [0.964697, 0.970576]
relational gain:              0.665881  [0.456199, 0.669980]
relational-minus-external:    0.392225  [0.233714, 0.544691]
D-LinOSS minus ridge:        -0.006342  [-0.010432, -0.004556]
curvature lift:               0.002894  [0.002892, 0.003657]
```

Model selection was not fully stable:

```text
dlinoss_diag_phase:                 4/5
dlinoss_hamiltonian_plus_scale:     1/5
```

The clean result is therefore robust across the five tested seeds, but it does not uniquely favor the phase-plus-scale parameterization.

### Stage B - Noise 0.01

The mechanism remained useful but became seed-sensitive:

```text
relational score:             0.879631  [0.837233, 0.936932]
relational gain:              0.591045  [0.293161, 0.792117]
relational-minus-external:    0.089337  [0.075753, 0.467812]
D-LinOSS minus ridge:        -0.024077  [-0.105254, -0.009644]
curvature lift:               0.000642  [-0.039345, 0.029571]
```

Relational alignment generally survived `noise=0.01`, but the ridge-competitiveness criterion failed for two seeds and curvature no longer had a stable sign.

### Stage B - Noise 0.03

The promoted mechanism failed:

```text
relational score:             0.168858  [0.166007, 0.462993]
relational gain:             -0.007546  [-0.215603, 0.004522]
relational-minus-external:    0.223098  [0.221862, 0.338120]
D-LinOSS minus ridge:        -0.156620  [-0.165310, -0.028168]
curvature lift:               0.226608  [0.204155, 0.406463]
```

The large apparent curvature lift at `noise=0.03` is not evidence of successful curvature-driven alignment. The full relational score and control separation collapsed while the phase-only arm collapsed more sharply. The lift is therefore relative damage reduction inside a failed regime.

Across the ten noisy configurations, the promotion rates were:

```text
0a all-gate pass rate:                    100%
0b relational gain > 0.10:                60%
relational-minus-external > 0.05:          90%
paired CI lower bound > 0:                 50%
D-LinOSS minus ridge >= -0.05:             50%
```

Stage B did not pass. Stages C and the 512-step extension were not run.

## Revised Standing Claim

The bridge is reproducible across five seeds at the noise-free ceiling and shows partial robustness at `noise=0.01`. It is not robust at `noise=0.03`. The primary validated mechanism remains relational phase/path alignment; the explicit curvature channel is small at the clean ceiling, unstable at low noise, and cannot yet support an independent curvature-driven claim.

The next experiment should address noise-conditioned clock matching or regularization before increasing history length or replacing the generator with tensor-network histories.

## AQ-PAGE-WOOTTERS-DLINOSS-NOISE-1

This follow-up was designed to answer a narrower question:

```text
Does noisy failure come mainly from correspondence-path recovery,
or from D-LinOSS transition learning even when the path is correct?
```

The first complete TPU run finished and produced a compact summary JSON:

[noise1_results.json](/C:/Users/cityz/IllI/newer_all/emergent_quantum_geometries/tmp_page_wootters_noise1_tpu/noise1_results.json)

However, that first implementation still mixed two effects:

```text
matcher-specific correspondence path
+ matcher-specific retraining target
```

So its reported `oracle_path_dlinoss_score` versus `learned_path_dlinoss_score` did not cleanly isolate path failure. It remained useful as a scouting run, but not as the final causal diagnosis.

### What the scouting run still told us

The scouting grid did show that noise sensitivity was real and that the simple matcher family was not the whole story. Some seeds degraded because the learned path drifted, but others suggested the training target itself had become unstable once the model was retrained on noisy correspondences.

That meant the next correction had to be structural:

```text
1. define an oracle path directly from the synthetic generator
2. fit one oracle-path D-LinOSS model on calibration
3. freeze that model
4. score learned matcher paths with the same frozen model
5. keep end-to-end retrained scores separate
```

### Corrected rerun status at pause

The corrected script was patched locally in:

[program_aq_page_wootters_dlinoss_noise1.py](/C:/Users/cityz/IllI/newer_all/emergent_quantum_geometries/program_aq_page_wootters_dlinoss_noise1.py)

It now uses:

```text
oracle path definition:
generator intrinsic-clock midpoint correspondence

separate score families:
frozen_operator_learned_path_score
oracle_path_dlinoss_score
end_to_end_learned_path_score
```

We launched a corrected TPU rerun, but intentionally stopped it when shutting down TPU resources for quota/token reasons. The partial corrected log is preserved here:

[noise1_corrected_partial.log](/C:/Users/cityz/IllI/newer_all/emergent_quantum_geometries/tmp_page_wootters_noise1_tpu/noise1_corrected_partial.log)

The partial corrected run already showed a much cleaner pattern on the completed `noise=0.01` and early `noise=0.02` seeds:

```text
oracle path score:         still high, roughly 0.70 to 0.82
frozen learned-path score: often sharply worse, sometimes negative
end-to-end retrained score: often recovers into the 0.85 to 0.96 range
path MAE:                  often large, around 7 to 8 on failing seeds
```

That is the strongest evidence so far that the first noisy breakdown is primarily a correspondence-path problem. D-LinOSS still appears able to model the transition when given a good path, but the learned relational path becomes unstable under noise and forces the end-to-end system to compensate by retraining on a degraded alignment.

This is not yet a finished claim, because the corrected grid was interrupted during `noise=0.03`:

```text
completed before stop:
noise 0.01 seeds 11-15
noise 0.02 seeds 11-15
noise 0.03 had not completed any seed

last completed line:
noise 0.02 seed 15

interrupted at:
noise 0.03 seed 11
```

### Where we left off

No TPUs remain active. The active rerun was stopped cleanly before node deletion, and the European TPU was deleted afterward.

Resume from:

[program_aq_page_wootters_dlinoss_noise1.py](/C:/Users/cityz/IllI/newer_all/emergent_quantum_geometries/program_aq_page_wootters_dlinoss_noise1.py)

Use the same TPU launch envelope as the interrupted corrected run:

```text
python3 -u program_aq_page_wootters_dlinoss_noise1.py
  --require-tpu
  --epochs 800
  --hidden-dim 16
  --context-steps 8
  --learning-rate 0.003
  --out-dir /home/cityz/aq_page_wootters_noise1/results_corrected
```

Local staging artifacts are here:

[tmp_page_wootters_noise1_tpu](/C:/Users/cityz/IllI/newer_all/emergent_quantum_geometries/tmp_page_wootters_noise1_tpu)

### Best current next step

Rerun the corrected `AQ-PAGE-WOOTTERS-DLINOSS-NOISE-1` grid to completion and only then promote the diagnosis. If the remaining `noise=0.03` seeds preserve high oracle-path scores while frozen learned-path scores collapse, we can say with much more confidence that noisy failure is dominated by path recovery rather than by the D-LinOSS transition model itself.
