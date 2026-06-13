# CHRONOS-SCHUMANN Replication Spec - 2026-06-13

## Established 0b Observation

The same-host 0a run saturated every mode at `r=1.0000`, which identified common-mode host timing rather than a frequency-selective signal.

The two-host 0b run used independent TPU VMs:

- Alice: `us-east1-d`
- Bob: `europe-west4-a`
- Samples: `32768`
- Sampling rate: `128 Hz`
- Duration: about `256 s`
- Payload shared for analysis: compact normalized Q vectors only

Observed zero-lag result:

| Mode | r_zero |
| --- | ---: |
| full | -0.1199 |
| schumann | +0.2365 |
| anti | -0.2678 |
| grid | -0.3273 |

Primary separation:

```text
schumann_minus_anti = 0.5043
```

Permutation controls from `chronos_schumann_significance.py`:

| Control | p_schumann | p_delta |
| --- | ---: | ---: |
| shuffle | 0.0010 | 0.0010 |
| block | 0.0010 | 0.0010 |
| phase surrogate | 0.0040 | 0.0010 |

## Preregistered Replication Endpoint

Primary endpoint:

```text
zero_lag_schumann_minus_anti = r_schumann_zero_lag - r_anti_zero_lag
```

Lagged correlations are diagnostic only. A replication should not promote `r_best` or best-lag results to the primary claim.

Primary pass gate:

```text
zero_lag_schumann_minus_anti >= 0.10
phase_surrogate_p_delta <= 0.01
```

Secondary consistency checks:

```text
r_full < 0.90
r_schumann_zero_lag > 0
r_schumann_zero_lag > r_anti_zero_lag
```

## Lag Interpretation

For the current collector, lag units are raw samples. At `128 Hz`, one lag step is about `7.8125 ms`.

The first 0b significance scan found Schumann best lag near `-16` samples, about `-125 ms`, with only a small improvement over zero lag:

```text
r_best - r_zero = 0.0440
```

Both TPU VMs reported microsecond-scale NTP offsets, so this lag is not explained by a seconds-scale VM clock offset. It remains diagnostic until replicated with the zero-lag endpoint fixed in advance.

## Next Run

Run the same US/EU pair collection again, ideally with a new seed and the same sample count:

```powershell
$env:CHRONOS_INDEPENDENCE_SAMPLES = "32768"
$env:CHRONOS_INDEPENDENCE_SAMPLE_HZ = "128"
$env:CHRONOS_INDEPENDENCE_SIZE = "96"
.\chronos_time_emission\run_chronos_independence_pair.ps1
```

After collection, run:

```powershell
python .\chronos_time_emission\chronos_schumann_significance.py `
  --alice .\chronos_time_emission\results0b_independence_remote\alice_us_q.json `
  --bob .\chronos_time_emission\results0b_independence_remote\bob_eu_q.json `
  --permutations 1000 `
  --max-lag 32 `
  --block-size 256
```

Interpret only the preregistered primary endpoint as the replication result.

## Replication Result: `rep1`

The preregistered replication was run with:

- Alice: `us-east1-d`, label `alice_us_rep1`
- Bob: `europe-west4-a`, label `bob_eu_rep1`
- Seed: `12`
- Samples: `32768`
- Sampling rate: `128 Hz`
- Duration: about `256 s`
- Analysis file: `chronos_time_emission/results0b_independence_remote/significance_results_rep1.json`

Zero-lag replication result:

| Mode | r_zero |
| --- | ---: |
| full | +0.3646 |
| schumann | +0.1493 |
| anti | +0.4413 |
| grid | +0.1775 |

Primary endpoint:

```text
schumann_minus_anti = -0.2920
```

Permutation controls:

| Control | p_schumann | p_delta |
| --- | ---: | ---: |
| shuffle | 0.0010 | 1.0000 |
| block | 0.0010 | 1.0000 |
| phase surrogate | 0.0210 | 0.8222 |

Preregistered gate result:

```text
primary_pass = False
secondary_pass = False
diagnostic_lag_only = True
```

Interpretation:

The first two-host 0b run showed a strong positive zero-lag Schumann-minus-anti separation, but the `rep1` replication did not reproduce it. The replication instead showed stronger positive anti-Schumann correlation than Schumann correlation. This means the current evidence supports a frequency-structured cross-system timing effect, but not a stable Schumann-specific carrier claim.

The honest standing claim after `rep1` is:

```text
Two independent TPU hosts can show non-common-mode, frequency-structured timing correlations, but the Schumann-band advantage observed in the first 0b run did not replicate under the preregistered endpoint.
```

## Carrier Follow-Up Diagnostic

`chronos_carrier2_analysis.py` was added to test a more conservative carrier-lock question:

```text
Can a calibration half select a band/sign/lag carrier that improves payload-half alignment against shuffled, wrong-time, severed-phase, and random controls?
```

On the original 0b pair, the split diagnostic selected the grid band and passed the carrier gate:

```text
selected_band = grid
selected_lag = +25
payload_true_r = +0.5972
payload_true_gain = +1.5236
pass_gate = True
```

On `rep1`, the same procedure selected the anti band during calibration but failed on the payload half:

```text
selected_band = anti
selected_lag = -22
payload_true_r = +0.0605
pass_gate = False
```

This reinforces the replication conclusion. There may be structured cross-system timing features in the data, but the currently tested external-reference/carrier mechanism is not stable enough to support a Schumann-specific or persistent carrier claim.

The next experiment should not broaden the claim. It should diagnose stability:

- Repeat at least two more preregistered US/EU runs without changing the endpoint.
- Add time-of-day and NTP metadata to each compact summary.
- Add a same-zone/different-host or cross-provider arm when available to distinguish grid/infrastructure coupling from ambient Schumann-band coupling.
- Treat lagged peaks as diagnostics only unless a fixed lag hypothesis is preregistered.
