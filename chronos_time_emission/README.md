# CHRONOS Time-Emission Experiments

CHRONOS is a separate experimental line from the AQ-STREAM signal-recovery operator work.

Goal:

Determine whether tensor-network workload topology leaves reproducible timing or spectral fingerprints in co-located TPU collective latency, beyond trivial mean-latency differences.

This does not claim teleportation or hidden communication. The physical path under test is:

Alice workload topology -> ICI / collective contention -> Bob fixed psum probe latency

## Structure

- `run_chronos_tpu.py`: TPU experiment runner only.
- `workloads.py`: Alice and Bob TPU workloads.
- `telemetry.py`: randomized trial schedules and latency recording.
- `conditions.py`: active/control condition execution and CSV writing.
- `spectral.py`: Welch, sideband, and OKL feature extraction.
- `classifiers.py`: block train/test classifiers.
- `analyze_chronos.py`: local analysis from `latencies.csv`.

## CHRONOS-0 Gate

Run a compact validation first:

```powershell
python -m chronos_time_emission.run_chronos_tpu --blocks 4 --block-size 50 --d 256 --sparse-block 32 --layers 4 --condition all
```

TPU run target:

```bash
python3 -m chronos_time_emission.run_chronos_tpu \
  --require-tpu \
  --blocks 20 \
  --block-size 100 \
  --d 512 \
  --sparse-block 32 \
  --layers 6 \
  --condition all \
  --out chronos_time_emission/results/chronos0_tpu
```

Pass interpretation:

- Active Mann-Whitney `p < 0.01`.
- Severed and shuffled controls are not significant.
- Block-level sideband or combined classifier beats the mean-latency classifier.
- Equal-FLOP topology control separates states without relying only on dense versus sparse workload size.

## Outputs

Each run writes:

- `results.json`: compact metrics.
- `latencies.csv`: raw block/trial latency table.
- `spectral_features.csv`: block-by-label spectral features.
- `summary.md`: compact run summary table.

## Latest Findings

- `CHRONOS-0` completed but failed its control gate: the severed control was significant, which pointed to sequencing/runtime artifact rather than a clean workload-coupling signal.
- `CHRONOS-0b` corrected that with interleaved conditions and a persistent-worker schedule. The severed and shuffled controls became null, but the active condition only reached borderline separation (`MW_p ~= 0.066`) rather than the target gate (`p < 0.01`).
- Small `CHRONOS-0b` text artifacts were mirrored locally without pulling the larger EU TPU files:
  `chronos_time_emission/results0b_remote/summary.md`
  `chronos_time_emission/results0b_remote/results.json`
- Run note:
  `chronos_time_emission/docs/CHRONOS_0B_REMOTE_ANALYSIS_2026-06-07.md`
- `CHRONOS-SCHUMANN-0a` saturated all modes at `r=1.0000`, diagnosing same-host/common-mode timing rather than Schumann-specific structure.
- `CHRONOS-SCHUMANN-0b` added a two-host US/EU independence gate. The first run passed Gate -1 and produced a strong zero-lag Schumann-minus-anti separation (`+0.5043`, phase-surrogate `p_delta=0.0010`), but the preregistered `rep1` run did not replicate the primary endpoint (`schumann_minus_anti=-0.2920`, phase-surrogate `p_delta=0.8222`).
- Current Schumann status: frequency-structured cross-system timing correlations are present, but a stable Schumann-band carrier claim is not established. See:
  `chronos_time_emission/docs/CHRONOS_SCHUMANN_REPLICATION_SPEC_2026-06-13.md`
- Next planned pivot: `AQ-DLINOSS-CHRONO-0` tests learned cross-system temporal phase-transition geometry instead of promoting any single carrier band. See:
  `chronos_time_emission/docs/AQ_DLINOSS_CHRONO_0_RUN_SPEC_2026-06-14.md`
- `AQ-DLINOSS-CHRONO-0` completed as a clean null on the heldout transition endpoint. Gate `-1` passed, but the true heldout transition score (`0.0111`) fell below the strongest null control (`0.0893`), and both the external-band and relational-clock arms failed to clear the payload gate. The current evidence says the two-host temporal geometry is not stable enough for calibration-to-payload prediction under this hardware/window configuration.
- `CHRONOS-MARGINAL-DRIFT-1` then tested whether the broadband correlation from the live `AQ-DLINOSS-CHRONO-0` pair was same-time coupling or structural similarity. The one-hour asynchronous comparison kept `r_full` relatively high (`0.5744 -> 0.4957`), which points to hardware/extractor similarity rather than a usable cross-host timing key. Correlation remains diagnostic only; heldout transition gain remains the truth gate. See:
  `chronos_time_emission/docs/CHRONOS_MARGINAL_DRIFT_1_RUN_SPEC_2026-06-14.md`
