# AQ-TRAPPIST1E-LINEARSSM-1331-CLOSEOUT-0

## Final status

Methodological boundary result. The detector infrastructure is validated, but the Program 1331 experiments were performed on residualized integration flux rather than extracted transmission depths. They therefore do not measure the stellar-contamination or planetary-residual sensitivity of the DREAMS analysis observable.

## Pipeline-stage correction

The Lewis/DREAMS analysis does not apply its visit-level contamination GP directly to `x1dints` flux. It first fits the white-light transit and visit systematics, then fits each wavelength-channel light curve to extract a transmission spectrum for each visit:

`transit_depth[visit, wavelength] +/- uncertainty`

The per-visit GP operates on the scatter and residual structure among those four transmission spectra. This branch instead trained on tensors shaped approximately `[visit, integration, wavelength]`, even after residualization. That bypassed the transit-depth extraction stage and exposed the models to dominant integration-level temporal autocorrelation and instrumental/stellar flux structure that are largely divided out during spectrophotometric extraction.

Accordingly, the reported 25--500 ppm injection failures are not a limit on recoverability in transmission-depth space. They are a negative result for the tested raw-flux objective only.

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

The question actually answered was:

- can a residual detector recover injected chromatic structure directly from residualized integration flux?

The intended DREAMS question remains untested by this branch:

- after validated white-light and wavelength-channel transit fitting, can visit-variable contamination be separated across the four extracted transmission spectra?

## Validated components

- Corrected Program 1331 integration-time tensor: `results/shards/program1331_integration_time_tensor.npz`.
- Residual-first preprocessing: `R_double_residual`.
- Strict phase, wavelength, shifted-template, and false-positive controls.
- WASP-39b linear-SSM positive control: `PROMOTE LINEAR_SSM_POSITIVE_CONTROL`, recovery correlation 0.990.
- Program 1331 paired-copy linear response: correlation 0.944 for nonzero injections.

## Blocking result

Program 1331 observable-only recovery remained 0.004--0.014 over 25--500 ppm. The strict wavelength-permutation degradation was 9.73%, below the 10% gate. Both the paired-copy and observable sensitivity floors are therefore null.

The paired-copy response is a method check, not a physical detection limit: subtracting outputs for identical base and injected copies cancels the real residual background in a linear model.

## What failed in the learning setup

The models were optimized against the wrong stage of the reduction pipeline. Raw integration flux is dominated by the static stellar spectrum, smooth instrumental behavior, and adjacent-integration autocorrelation. A transmission spectrum is a fitted parameter product in which those components have already been modeled or divided out.

What the model could see reliably:

- the dominant static visit spectrum
- smooth residual structure
- baseline detector-level or visit-level regularity

What it did not recover at promotable strength:

- a time-localized, wavelength-localized stellar contamination field that survived residualization and null controls
- a phase-dependent chromatic pattern strong enough to beat simple residual baselines
- an observable sensitivity floor for injected spot/facula-like contamination through 500 ppm on the real Program 1331 substrate

What that means astrophysically:

- no sensitivity statement about the extracted Program 1331 transmission spectra follows from these raw-flux runs
- the four visits may still support DREAMS-style contamination inference after proper light-curve and transmission-spectrum extraction
- the correct result is a preprocessing and objective boundary, not evidence that the astrophysical signal is absent

## Architecture decision

- D-LinOSS-v0 is retired from the exoplanet residual-detector role.
- The promoted path is residual-first preprocessing plus strict controls and a linear SSM/residual baseline detector.
- No further Program 1331 model variants are authorized from this branch.
- This closeout does not block the separate DREAMS paired-data branch. That branch subsequently reached three clean public b/e pairs, but both naive subtraction and `b-proxy + residual GP` failed promotion. Its current real-data status is `PROMOTE_GP_ONLY`, and its observing-design closeout is `PROMOTE_CLOSER_PAIRING_REQUIRED`.

## Scientific claim

The tensor, residual, null-control, and TPU infrastructure is validated, but the Program 1331 sensitivity experiment was applied before transmission-spectrum extraction. The next real-data project must first reproduce the white-light fits and wavelength-channel transit-depth extraction, then compare contamination models in depth space. Additional observations or paired b/e data may still help, but they are not substitutes for the missing extraction stage.

No atmosphere, methane, CO2, real stellar-contamination, or time-geometry claim is supported.

## Optional future audit

No further raw-flux noise-floor audit is useful for the DREAMS question. A future sensitivity audit must inject a transit-depth signal into wavelength-channel light curves, rerun the same extraction, and score the recovered depth spectrum.

## Authoritative results

- `results/wasp39b_linearssm_residual_positive_control_0_local.json`
- `results/trappist1e_linearssm_1331_residual_sensitivity_0.json`
