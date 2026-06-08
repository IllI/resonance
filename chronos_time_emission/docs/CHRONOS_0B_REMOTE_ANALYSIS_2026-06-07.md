# CHRONOS-0b Remote Analysis (2026-06-07)

## Context

`CHRONOS-0b` was the compact correction run after the earlier `CHRONOS-0` negative/control-failure result.

The prior run had a significant severed control, which strongly suggested runtime-order or host-dispatch artifact rather than clean workload-coupling signal. `CHRONOS-0b` changed the experiment in three important ways:

- interleaved `active`, `severed`, `shuffled`, and `equal_flop` trials instead of block-running each condition;
- used a persistent Alice worker schedule rather than per-trial thread spawning;
- pinned Bob to a fixed TPU-side probe path instead of a replica-group configuration that had been unstable on the `v6e-8` device layout.

## TRC-safe retrieval

To stay conservative with the TPU TRC guidance, only the small text artifacts were fetched from the Europe TPU over SSH text output:

- remote `summary.md`
- remote `results.json`
- remote `ls -lh` for file sizes

The larger files were intentionally left on the EU node:

- `/home/cityz/chronos_time_emission/results0b/latencies.csv` (`136K`)
- `/home/cityz/chronos_time_emission/results0b/spectral_features.csv` (`2.7K`)

## Result

The corrected run completed successfully and wrote:

- `/home/cityz/chronos_time_emission/results0b/results.json`
- `/home/cityz/chronos_time_emission/results0b/latencies.csv`
- `/home/cityz/chronos_time_emission/results0b/spectral_features.csv`
- `/home/cityz/chronos_time_emission/results0b/summary.md`

Recovered metrics:

| condition | delta_mean_ns | MW_p | KS_p | mean_acc | sideband_acc | combined_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active | -1111 | 0.06607 | 0.08658 | 1.000 | 0.500 | 0.500 |
| severed | +492 | 0.3389 | 0.4266 | 0.250 | 0.500 | 0.250 |
| shuffled | -294 | 0.4949 | 0.5602 | 0.750 | 0.500 | 0.250 |
| equal_flop | +1193 | 0.4051 | 0.6350 | 0.500 | 0.500 | 0.500 |

## Interpretation

This is a cleaner result than `CHRONOS-0`, even though it is not a positive hit.

What improved:

- `severed` is no longer significant;
- `shuffled` is also null;
- the previous obvious control failure appears to have been removed.

What did not pass:

- `active` did not cross the intended gate of `MW_p < 0.01`;
- spectral and combined classifiers remained flat at chance-like levels;
- `equal_flop` also stayed null.

That means `CHRONOS-0b` behaved like a disciplined negative or near-null run:

- the correction repaired the control structure;
- the remaining active effect was only borderline (`MW_p ~= 0.066`);
- there is not enough evidence here to claim a robust TPU-side emission signature.

## Scientific read

The strongest conclusion from `CHRONOS-0b` is methodological:

the significant severed result from the earlier run was likely an artifact of sequencing and runtime structure, not a clean physical effect.

The corrected design removed that artifact, but it also removed the apparent signal. That is exactly the kind of outcome a good control correction is supposed to reveal.

At this point, `CHRONOS` looks like:

- a useful negative line if we stop here;
- or a candidate for exactly one more higher-powered active-only gate if we want to test whether the borderline active separation grows with more trials rather than with more analysis complexity.

## Local mirrors

Small mirrored copies of the fetched text artifacts were written locally at:

- `chronos_time_emission/results0b_remote/results.json`
- `chronos_time_emission/results0b_remote/summary.md`
