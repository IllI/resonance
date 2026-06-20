# TRAPPIST-1e D-LinOSS Stellar-State Residual 0

This branch learns visit-varying stellar contamination before testing for a stable planetary residual. It is not a methane detector, atmosphere claim, or literal time-distortion test.

## Branch closeout

This branch is closed as a methodological boundary result; see `TRAPPIST1E_LINEARSSM_1331_CLOSEOUT_0.md`. D-LinOSS-v0 is retired from the detector role, and no further Program 1331 raw-integration model variants should be launched from this branch.

The earlier statement that Program 1331 produced no observable sensitivity floor through 500 ppm is not an astrophysical sensitivity limit. Those experiments operated on residualized `x1dints` integration flux, before wavelength-by-wavelength transit fitting. DREAMS operates on extracted visit-level transmission spectra: transit depth as a function of wavelength. Because the tested tensor and the published analysis use different observables, the raw-flux result cannot establish that Program 1331 lacks a recoverable transmission-depth signal.

The only active real-data interpretation path that survived the branch is the separate DREAMS-style Program 1331 e-only GP reproduction. The paired b/e correction branch is also closed on the current public data: three clean pairs exist, but the learned transfer path remains frozen at `PROMOTE_GP_ONLY`, and the observing-design closeout is `PROMOTE_CLOSER_PAIRING_REQUIRED`.

## Current branch state

- D-LinOSS detector branch: closed and retired from JWST detector duty
- DREAMS Program 1331 reproduction: directionally passed
- Paired TRAPPIST-1b/e data: unlocked with three clean pairs
- Naive b-proxy subtraction: failed and worsened scatter
- Paired b-proxy plus residual GP: closed as `PROMOTE_GP_ONLY`; in-sample scatter improved, held-out prediction and null gates failed
- TPU pair-count power audit: no universal minimum pair count; state coherence and b/e time offset dominate identifiability
- TPU strong-prior audit: `PROMOTE_CLOSER_PAIRING_REQUIRED`; at the observed 4.26--6.01 hour offsets, only coarse scalar coupling was robust enough to identify
- Active science path: reproduce the white-light and spectrophotometric extraction, then apply the DREAMS-style per-visit GP in transmission-depth space

## Scientific target

The astrophysical target in this branch was not a generic classifier. It was a contamination-separation problem on JWST time-series transit spectroscopy:

- `Program 1331`: four TRAPPIST-1e NIRSpec/PRISM visits
- `Programs 9256/6456`: close TRAPPIST-1b/e pair visits intended to use `b` as a stellar-contamination proxy for `e`
- `WASP-39b`: high-SNR positive control for residual-detection infrastructure

The core scientific question was:

- Can visit-variable stellar contamination be separated from a stable planetary transmission residual in public TRAPPIST-1e data?

The detector-level observables actually supplied to the experimental models were:

- integration-time flux spectra as a function of wavelength
- per-integration uncertainties and data-quality masks
- transit phase and absolute integration time
- white-light normalized and continuum-removed residual tensors
- visit-to-visit variability across the four Program 1331 visits
- for paired runs, matched `b` and `e` spectra with measured b/e time offsets

These are not the final DREAMS observables. The published contamination analysis follows this sequence:

1. Fit each visit's white-light curve for transit timing, limb darkening, and visit baseline/systematics.
2. Fit each wavelength-channel light curve and extract `transit_depth[visit, wavelength]` with uncertainty.
3. Model visit-variable stellar contamination across the four extracted transmission spectra with a per-visit GP.
4. Compare the corrected/shared transmission spectrum with atmospheric hypotheses.

This branch attempted stage 3-like state modeling directly on stage 0 detector-level integration flux. It did not independently reproduce stages 1 and 2 first. The resulting `[visit, integration, wavelength]` tensors are useful for reduction and light-curve work, but they are not interchangeable with the `[visit, wavelength]` transmission-depth tables analyzed by DREAMS.

The main astrophysical signal classes we were trying to distinguish were:

- stellar contamination that varies between visits:
  spot/facula color contrast, activity-state drift, flare contamination, time-offset decorrelation
- stable or repeatable planetary residual structure:
  wavelength-localized transmission residuals that persist across visits or survive contamination correction
- trivial nonphysical structure that must not drive a claim:
  static continuum level, mask pattern, wavelength coverage, visit ID, pair ID, redshift-free bookkeeping artifacts

The branch did not attempt a defensible methane or atmosphere detection. Its raw-flux detector results now document a preprocessing boundary: a D-LinOSS, linear-SSM, or GP contamination comparison must be performed after a validated transmission-spectrum extraction, or be formulated as part of a joint transit-plus-contamination model.

## Synthetic capacity result

`AQ-DLINOSS-EQUAL-SUPERVISION-CAPACITY-TEST-0` already completed on eight TPU devices. It evaluated 108 training configurations: two supervision arms, two frequency-prior arms, three hidden dimensions, three training lengths, and three seeds. All arms achieved high in-distribution synthetic recovery, but the final verdict was `DO NOT PROMOTE` because the gain over the equal-supervision linear SSM was below 5%, mask-shuffle degradation stayed below 20%, and leave-window-out recovery failed. This result is independent of the JWST extraction error and does not authorize an astronomy claim.

## Canonical documents

Treat these as the authoritative branch record:

- `README_TRAPPIST1E_DLINOSS.md`: branch overview and operational constraints
- `trappist1e_dlinoss_failure_assessment.md`: full technical postmortem, including the equal-supervision rematch
- `TRAPPIST1E_LINEARSSM_1331_CLOSEOUT_0.md`: final science closeout for the TRAPPIST-1e Program 1331 detector path

Most of the other `.md` files in this folder are run specs or point-in-time memos. They are useful as historical scaffolding, but they should not be treated as the current branch verdict.

## DREAMS reproduction branch

The DREAMS reproduction is a separate CPU-first branch. Its directional GP result used released visit-level spectra; it did not independently reproduce the white-light and wavelength-channel extraction from `x1dints`. It therefore validates the qualitative GP behavior after extraction, not the full reduction pipeline. It does not reopen D-LinOSS-v0 or authorize a new detector claim.

Authoritative machine-readable state lives under `results/dreams_repro/`:

- `branch_freeze_manifest.json`: `PASS_BRANCH_FREEZE`
- `source_ledger.json`: `PASS_SOURCE_LEDGER` for Programs 1331, 6456, and 9256
- `timebase_validation.json`: `PASS_TIMEBASE` using native `INT_TIMES` rows and the four released fitted transit centers
- `white_light_repro_report.json`: `PASS_WHITE_LIGHT_REPRO` directionally against the released DREAMS light curves
- `gp_evidence_table.json`: `PASS_GP_REPRO_DIRECTIONAL`; the released four-visit spectra reproduce the direction of the DREAMS per-visit GP behavior
- `pair_program_manifest_9256_6456.json`: `PASS_PAIR_METADATA`; 14 public archived `x1dints` products, with 9256 observation 014 confirmed withdrawn
- `event_labels_9256_6456.json`: `PASS_PAIR_UNLOCK` with three clean, published/verified b/e pairs
- `be_pair_reduction_report.json`: `PASS_BE_PAIR_ABI` for `results/shards/trappist1_be_pairs_9256_6456.npz`
- `be_proxy_correction_report.json`: `PROMOTE_GP_ONLY`; naive b-proxy subtraction worsened pair scatter
- `gp_be_hybrid_closeout_0.json`: `PROMOTE_GP_ONLY`; freezes post-hoc paired-transfer tuning on the three-pair dataset
- `pair_count_power_audit_0.json`: preliminary CPU audit
- `pair_count_power_audit_0_tpu_summary.json`: `PROMOTE_PAIR_COUNT_REQUIREMENT` with a conditional, not global, requirement; 165,888 TPU trials show state stability and pair offset dominate pair count
- `remote_fetch/strong_prior_identifiability_0_tpu.json`: `PROMOTE_CLOSER_PAIRING_REQUIRED`; no physically structured transfer prior was robust at the three observed pair offsets
- `pair_cadence_requirement_0.json`: canonical observing-design interpretation of the pair-count and strong-prior TPU audits
- `gp_final_repro_1.json`: `PASS_GP_REPRO_DIRECTIONAL`; canonical active e-only GP result

The paired-data ABI gate is open using `9256-o008-be`, `9256-o009-be`, and the clean `6456-o003-be` pair anchored to the fitted transit centers published by Allen et al. (2025). Both raw subtraction and the learned b-proxy plus residual-GP model failed promotion. The learned model reduced in-sample scatter but failed leave-one-pair-out prediction and did not degrade under the required nulls. No atmospheric claim is authorized.

## Active path and paired-branch reopen gate

The active model is the DREAMS-style e-only GP recorded by `AQ-DREAMS-1331-GP-FINAL-REPRO-1`. The paired learned-transfer branch is frozen as `PROMOTE_GP_ONLY`. Its observing-design closeout is `PROMOTE_CLOSER_PAIRING_REQUIRED`.

The TPU power audit tested 3, 5, 8, 10, 15, and 20 clean pairs over 2--8 hour offsets, three state-stability levels, three flare levels, four amplitudes, and three prior strengths. It found no defensible universal pair-count minimum. High-stability regimes already promoted at 97.5% with three pairs, while low-stability regimes remained below 6% at twenty pairs. Shorter b/e offsets and independent activity-state information are therefore higher priorities than simply collecting more pairs. The implemented strong spot-facula prior was overconstrained and is not scientifically diagnostic.

The follow-up strong-prior audit tested scalar, smooth-wavelength, spot/facula, offset-decay, and rotation/activity transfer constraints at the actual pair offsets of 4.26, 4.42, and 6.01 hours. No physically structured constraint passed the robustness gate across the observed-offset regimes. Scalar coupling was identifiable in coherent regimes, but it does not support a wavelength-dependent stellar-contamination correction. The result is a cadence requirement, not evidence that the b/e proxy concept is invalid: future observations should prioritize materially shorter offsets, additional activity-state measurements, and then additional clean pairs. The existing audits show that 2-hour offsets perform best among the tested cadences, but they do not establish 2 hours as a hard maximum or a universal minimum pair count.

## Fixed tensor ABI

Every shard contains `x [V,T,L,C]`, `phase [V,T]`, `mask [V,T,L]`, `sigma [V,T,L,C]`, `wavelength_um [L]`, and visit metadata (`planet_id`, `epoch`, `visit_id`, `program_id`, `quality_flags`). Channels are normalized flux, continuum-removed flux, white-light-normalized flux, and an optional systematic proxy.

## Local fake smoke

```bash
bash scripts/run_local_fake_trappist.sh
```

The CPU fallback runs the eight null modes sequentially. TPU runs require eight visible devices and use one null mode per device.

## TPU reuse and compiled JAX cache

The wrapper sets `JAX_COMPILATION_CACHE_DIR` to a stable path under the TPU user's home directory. Re-running on the same active VM reuses compiled executables without copying them anywhere.

For preemption recovery, set `JAX_CACHE_GCS` to a bucket prefix in the same region as the TPU. Cache keys include TPU zone, `v6e-8`, JAX version, and experiment ABI. Never restore an EU cache into a US VM or reuse it after changing JAX/runtime/topology.

```bash
export TPU_ZONE=us-east1-d
export JAX_CACHE_GCS=gs://US_REGION_BUCKET/trappist-dlinoss/jax-cache
bash scripts/run_tpu_trappist_null_suite.sh INPUT_NPZ OUTPUT_JSON 800
```

Use `scripts/deploy_to_ready_tpu.ps1` to update an already active VM. It kills only the experiment process, uploads a small code payload, leaves the VM-local cache intact, and launches detached.

If no node is active, `scripts/run_tpu_trappist_shotgun.ps1` requests TRC-covered spot `v6e-8` resources in `us-east1-d` and `europe-west4-a`. It launches on the first ready node, then deletes the losing node and queued resource asynchronously.

## US versus Europe storage

- `us-east1-d`: workstation downloads are allowed under the project's TRC operating policy; results, checkpoints, and data may be retrieved when useful.
- `europe-west4-a`: fetch Zenodo/MAST directly on the VM and keep raw products, shards, checkpoints, and compilation cache on that VM or in a Europe-region bucket. Do not SCP those artifacts to the workstation.
- Keep separate `GCS_ROOT_US` and `GCS_ROOT_EU` prefixes. Never cross-sync raw data, checkpoints, or JAX caches between them.
- On Europe, download only a compact summary or short log tail. The US path does not apply this Europe-specific restriction.

GCS adds storage cost, so the default is VM-local persistence while a node remains active. A same-region GCS cache is opt-in and useful immediately before expected teardown or when preemption risk justifies storage.

## Data acquisition

Metadata-only discovery is cheap:

```bash
python data/fetch_trappist_zenodo.py --out-dir results/raw/zenodo --metadata-only
python data/fetch_mast_trappist_async.py --proposal-ids 1331 9256 --out-root results/raw/mast --query-only
```

For real downloads, run these commands on the winning VM and point `--out-dir`/`--out-root` at VM-local storage or a same-region GCS prefix. The MAST builder prefers `x1dints`/`x1d`, records FITS HDU layouts, and emits `STOP_DATA` with per-file failures instead of silently manufacturing a shard.

## Verdicts

`PROMOTE STELLAR STATE MODEL` requires held-out visit performance plus degradation under epoch, wavelength, phase, frozen-operator, and smooth-baseline nulls. A planetary residual is not promotable until paired TRAPPIST-1b/e data pass planet-label shuffle and cross-visit residual stability controls.

## Real reference-injection calibration

`AQ-TRAPPIST1E-DLINOSS-REAL-REFERENCE-INJECTION-0` starts from the four Lewis/DREAMS Program 1331 combined `x1dints` products (`o001`, `o002`, `o003`, `o104`). The companion release supplies the published white-light timestamps, uncertainties, and fitted transit centers. Build the compact fixed-shape shard with:

```bash
python data/build_lewis_real_tensor.py \
  --fits-root results/raw/mast_lewis/1331 \
  --bundle-root results/raw/zenodo/dreams_2509_05414/extracted/TRAPPIST-1e-GTO-2025-main \
  --out results/shards/lewis_1331_real_reference_tensor.npz
```

The TPU run injects 0--250 ppm flat, CH4-like, CO2-like, H2O-like, broad-slope, and localized residuals into copies of these real integrations. Program 1331 is e-only, so it can promote a stellar-contamination/sensitivity model but cannot pass the b-reference gate. That final gate is reserved for correctly phase-labeled Program 9256 pairs.

JAX caches compiled executables, not data. Keep the NPZ shard separately and preserve the shape-specific cache at `~/.cache/jax/europe-west4-a_v6e8_jax0.6.2_real_reference_injection_v0` between runs.

The Program 1331 e-only injection run failed its gates and is frozen. Its first shard also exposed a time-alignment defect: binned Lewis white-light timestamps had been aligned to unbinned spectral integrations. Do not reuse that shard or its absolute sensitivity metrics.

The paired-data branch is label-gated before any TPU training:

- `AQ-TRAPPIST1E-PAIR9256-LABEL-AUDIT-0`: `STOP_LABEL`; see `PAIR9256_LABEL_AUDIT_0.md`.
- `AQ-TRAPPIST1E-PAIR9256-EPHEMERIS-REPAIR-1`: `STOP_LABEL`; see `PAIR9256_EPHEMERIS_REPAIR_1.md`.

The repair stage writes `data/trappist1/visit_manifest_9256.json` and a smoke-only shard at `results/shards/pair9256_safe_008009_smoke.npz`. Only observations 008 and 009 are currently usable as clean b/e ABI smoke pairs. Observation 007 is explicitly treated as a c-e-d-b multi-transit window and excluded from b/e reference training. No paired TPU run is allowed until at least three clean b/e pairs pass TTV-aware planet labels, phase labels, pair offsets, compatible grids, mask gates, and unresolved-event closure.
