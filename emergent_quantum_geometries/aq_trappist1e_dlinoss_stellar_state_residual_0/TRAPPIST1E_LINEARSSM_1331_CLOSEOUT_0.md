# AQ-TRAPPIST1E-LINEARSSM-1331-CLOSEOUT-0

## Final status

Controlled negative. The detector pipeline is validated, but the corrected public Program 1331 e-only tensor does not support an observable stellar-contamination sensitivity floor through 500 ppm.

## What the model was trained on

This run used the public Lewis/DREAMS Program 1331 TRAPPIST-1e NIRSpec/PRISM time-series products. After timebase repair and residual-first preprocessing, the effective model inputs were:

- integration-time spectra across the PRISM wavelength grid
- per-integration uncertainty arrays
- valid-data masks
- transit phase and visit ordering
- residualized spectral tensors rather than raw full-spectrum flux

The training target was not "detect methane" or "retrieve an atmosphere." The target was to recover injected or latent stellar-contamination structure in the residual tensor strongly enough that the correct time, phase, and wavelength organization mattered.

In astrophysical terms, the intended learnable structure was:

- visit-variable stellar contamination:
  spot/facula color contrast, chord-dependent contamination, activity-state drift, weak time-localized chromatic residuals
- not the static baseline:
  mean stellar continuum, fixed detector response, or visit-average transit shape

The relevant question was:

- does the public Program 1331 dataset contain enough observable time-resolved chromatic structure to identify stellar-state variation on top of the dominant static spectrum?

## Validated components

- Corrected Program 1331 integration-time tensor: `results/shards/program1331_integration_time_tensor.npz`.
- Residual-first preprocessing: `R_double_residual`.
- Strict phase, wavelength, shifted-template, and false-positive controls.
- WASP-39b linear-SSM positive control: `PROMOTE LINEAR_SSM_POSITIVE_CONTROL`, recovery correlation 0.990.
- Program 1331 paired-copy linear response: correlation 0.944 for nonzero injections.

## Blocking result

Program 1331 observable-only recovery remained 0.004--0.014 over 25--500 ppm. The strict wavelength-permutation degradation was 9.73%, below the 10% gate. Both the paired-copy and observable sensitivity floors are therefore null.

The paired-copy response is a method check, not a physical detection limit: subtracting outputs for identical base and injected copies cancels the real residual background in a linear model.

## What failed in the data

The negative result was not just "the model underperformed." The data did not expose the specific structure we needed at measurable strength.

What the model could see reliably:

- the dominant static visit spectrum
- smooth residual structure
- baseline detector-level or visit-level regularity

What it did not recover at promotable strength:

- a time-localized, wavelength-localized stellar contamination field that survived residualization and null controls
- a phase-dependent chromatic pattern strong enough to beat simple residual baselines
- an observable sensitivity floor for injected spot/facula-like contamination through 500 ppm on the real Program 1331 substrate

What that means astrophysically:

- the public four-visit Program 1331 release is sufficient to reproduce DREAMS-style per-visit GP behavior
- it is not sufficient, in this branch, to support a direct residual detector claim for stellar contamination or planetary transmission structure
- therefore the correct scientific statement is about contamination-model necessity, not atmospheric recovery

## Architecture decision

- D-LinOSS-v0 is retired from the exoplanet residual-detector role.
- The promoted path is residual-first preprocessing plus strict controls and a linear SSM/residual baseline detector.
- No further Program 1331 model variants are authorized from this branch.
- This closeout does not block the separate DREAMS paired-data branch. That branch subsequently reached three clean public b/e pairs, but both naive subtraction and `b-proxy + residual GP` failed promotion. Its current real-data status is `PROMOTE_GP_ONLY`, and its observing-design closeout is `PROMOTE_CLOSER_PAIRING_REQUIRED`.

## Scientific claim

The pipeline is validated on WASP-39b and on paired-copy TRAPPIST injections, but current public Program 1331 e-only data do not support an observable stellar-contamination sensitivity floor through 500 ppm. Future progress requires additional observations, verified paired b/e data, or a stronger external stellar-contamination prior.

No atmosphere, methane, CO2, real stellar-contamination, or time-geometry claim is supported.

## Optional future audit

`AQ-TRAPPIST1E-OBSERVABLE-NOISE-FLOOR-AUDIT-0` may estimate the injection amplitude required to exceed the residual background. It is a CPU-only measurement audit, not a promotion run.

## Authoritative results

- `results/wasp39b_linearssm_residual_positive_control_0_local.json`
- `results/trappist1e_linearssm_1331_residual_sensitivity_0.json`
