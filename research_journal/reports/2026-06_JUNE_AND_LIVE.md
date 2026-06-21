# June 2026: Committed History and Live Experimental Frontier

Evidence labels: `COMMITTED` means present in branch history; `LIVE` means modified or untracked at archive initialization. Metrics below are contemporaneous records, not independently reproduced by this archive pass.

## AQ FFT, YINYANG, and STREAM (June 1–5)

Commit `c016eb0` consolidated AQ FFT/DNA/HYBRID records and the ordered YINYANG recovery path. The work tested whether images semanticized into folded complex D-LinOSS states could be transported and decoded. It discovered two separate bottlenecks: a sparse codec whose own oracle ceiling was only 17.04 dB, and—after switching to a lossless full-coefficient codec—transport recovery of only 21.42 dB. Raw-patch controls reached roughly 30 dB per frame, while ordered two-frame recovery fell to 25.05 dB. A preview patch-order error was also corrected. These are controlled simulator diagnostics, not evidence of physical image teleportation.

Commits `d19d973`, `3c22f35`, and `05ed17b` moved into STREAM encodings, corrected pair construction, and added a recovery path. Later STREAM-4/5/6 documentation arrived in `e6a2158`. Numerous STREAM-2–8 and YINYANG visual artifacts remain `LIVE` under `tpu_previews/`.

Assessment: strong diagnostic/capability family; medium interpretation confidence; no physical-teleportation promotion.

## SO(3) kernel–Galerkin AQ (June 5–8)

Commits `dcf54b8` and `f5b8ba7` defined the Collins Corollary 6.6.8 numerical program, adding CPD kernels, auxiliary-space checks, known-solution verification, and 10x oversampling. `0912240` and `f5b9d65` recorded certified and confidence TPU runs; `deac28b`, `88ca6f0`, and `5de0613` reorganized the implementation and guidance under `emergent_quantum_geometries/kernel/`.

The pilot established TPU feasibility but had inconsistent convergence. Deterministic Haar quadrature reduced moment errors to approximately 1e-6 and exposed ill-conditioned dense Lagrange solves. A restricted confidence regime (`N=5`, 25–100 centers, five seeds) recorded all errors below 1e-2 and median slope -0.599, near the -2/3 target. Float32 condition numbers reached infinity at larger center counts, so the full asymptotic regime was not verified.

Assessment: `COMMITTED`, reproducible numerical-method candidate; not a numerical proof of the theorem. The restricted-slope interpretation needs mathematical review.

## Chronos external timing (June 7–15)

Commit `50f14f5` recorded CHRONOS-0b and a corrected negative result: an apparent severed-control signal disappeared after correcting sequencing/runtime structure. `ad5b914` added a two-host Schumann replication; the initial positive pair did not replicate, and `rep1` favored anti-Schumann structure. `bc154b2` added D-LinOSS temporal geometry and marginal drift diagnostics: held-out transition gain was null, and Alice/Bob broadband similarity persisted after a +1-hour shift, favoring hardware/extractor structure over synchronized coupling.

The `LIVE` CHRONO-MERA-STRAIN-0 continuation corrected an entropy-collapse implementation issue with `MIN_COARSE_SITES=8`; sanity checks then passed, but synchronized-spike and elasticity gates failed on all three pairings.

Assessment: high-value `control`/`closeout` family. The committed negative conclusions are comparatively strong; the MERA closeout still needs a reproducible commit.

## Temporal gauge (June 9 onward)

Only the initial specification is `COMMITTED` in `e6a2158`; implementation and result JSONs are `LIVE`. The family reused YINYANG/STREAM complex-pair encoding and Chronos timing features. A host-latency key passed a mechanism gate but failed strong synchronization. A first TPU-resident follow-up used a false severed control because a different seed retained the same smooth trajectory. A corrected block-permuted severing reportedly produced an all-gate pass with near-unity zero-lag correlation.

Assessment: `LIVE`, low-to-medium confidence. It may show that structured shared TPU telemetry can serve as a phase reference in one runtime. It does not establish ambient time as a universal or nonlocal key. Independent hosts, genuinely independent telemetry, seed audits, and reruns of the corrected severing logic are mandatory.

## Page–Wootters and D-LinOSS (June 15–17)

Commit `87c2501` recorded the initial synthetic bridge. Clean run 0b reported relational score 0.9573, gain 0.6700, and 98.4% of windows beating the best null. Five clean seeds reproduced the mechanism; noise 0.01 became seed-sensitive and noise 0.03 failed. The first noise diagnostic confounded path correspondence with matcher-specific retraining; the corrected design froze an oracle-path model and scored learned paths separately.

The extensive June 17 continuation is `LIVE`: matcher policy/regret, clock metrics, local/global observability, path bridges, event damping, and quantum/multiclass clock bounds. Current documents say local scalar-clock observability failed under noise; explicit causal paths were perfectly observable in the synthetic generator; an A10 event-damped bridge passed aggregate gates through noise 0.03; and per-window scores around 0.66–0.68 remained below the original 0.70 gate. Later bound diagnostics question the threshold, creating a post-hoc-selection risk that must be resolved by preregistration and independent audit.

Assessment: promising controlled synthetic relational alignment, not a physical Page–Wootters result. Generator leakage and path observability are critical audit targets.

## TRAPPIST/JWST D-LinOSS correction and closeout (June 20)

Commits `f5dddef`, `829d029`, and `0a3c89f` progressively narrowed the branch's claims. The decisive correction was that early models operated on residualized integration-level `x1dints`, while DREAMS analyzes extracted visit-level transmission depth. Therefore failed 25–500 ppm injections were not astrophysical sensitivity limits. Other documented failure modes include invalid binned/unbinned time alignment, unreliable labels/ephemerides, static-baseline domination, scalar spectral treatment, soft rather than physical damping priors, and overwritten state.

An equal-supervision synthetic rematch recovered the task, but D-LinOSS beat linear SSM by only about 1.25–1.36%, below the 5% gate; mask and leave-window controls failed.

The broader DREAMS/TRAPPIST/WASP-39b/Program-1331 audit corpus is `LIVE`. Current closeout records say D-LinOSS-v2 scored strongly on WASP-39b and a paired-copy task but only 0.0118 on observable-only Program 1331, with stronger baselines and immaterial damping ablation. A DREAMS e-only GP reportedly reproduced directionally; three b/e pairs recovered at 4.26, 4.42, and 6.01 hours; naive subtraction and learned wavelength transfer failed promotion.

Assessment: the pipeline-stage correction is a major epistemic pivot. Live result values need data checksums, extraction-equivalence verification, clean-pair label audit, and independence checks before citation.

## MPS capacity, GLIMPSE, and LRD variants (June 20–21)

Commits `df331df`, `8e27200`, `158bb4e`, and `88adba7` introduced the MPS spectral-capacity gate and explicit-mask follow-up. Committed run 0 improved D-LinOSS leave-window recovery from 0.4359 to 0.7462, just below 0.75, but mask-shuffle degradation was only 7.68%; verdict: do not promote.

A `LIVE` run 1 reportedly reached 0.7882 for MPS+linear-SSM versus 0.7525 for MPS+D-LinOSS, yielding `PROMOTE MPS_LINEARSSM_ONLY`. This is a useful architecture-selection result, but the JSON and modified README need a reproducible commit.

The `LIVE` GLIMPSE-17775 branch corrected the target provenance from imaging Program 3293 to DDT Program 9223. Public products provide a static spectrum, not a validated clock axis. A smoke comparison favored linear SSM (0.708) over MPS+linear-SSM (0.322), and the control library is explicitly incomplete. Related LRD folders explore damping, full spectra, inverse-BH-star, lightcone, censored/line-phase spectroscopy, and time residuals. They cannot currently establish intrinsic velocity widths, clocks, gravitational waves, or black-hole-star origins.

The June 21 grant closeout is itself `LIVE`. Its intended disposition is scientifically conservative: retire D-LinOSS from JWST detector duty; retain residual-first linear SSM/classical baselines, DREAMS e-only GP, MPS+linear-SSM synthetic architecture, and a control-blocked static GLIMPSE branch.

## June lineage

```text
AQ FFT/YINYANG codec diagnostics -> STREAM pairing -> temporal-gauge phase reference

Chronos timing nulls -> reject carrier/alignment claims
                     -> Page–Wootters internal relational simulations
                     -> path observability/event-damped bridge tests

TRAPPIST detector-level failure -> pipeline/timebase correction
                                -> equal-supervision rematch
                                -> MPS capacity gate
                                -> MPS + linear SSM preference
                                -> static GLIMPSE branch with controls blocked

DREAMS reproduction -> e-only GP -> b/e cadence proxies -> observing-design closeout
```

## Highest-priority validation queue

1. Temporal-gauge corrected 0c telemetry independence and all-gate pass.
2. June 17 Page–Wootters thresholds, leakage, and path observability.
3. MPS run-1 linear-SSM-only promotion.
4. D-LinOSS-v2 real-JWST nonpromotion and data/extraction provenance.
5. DREAMS GP reproduction independence and three-pair cadence assumptions.
6. Any GLIMPSE intrinsic-history, time, GW, or origin claim.
