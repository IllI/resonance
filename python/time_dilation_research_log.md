# Time-Dilation D-LinOSS Research Log

## Purpose

This log records what we are trying to explain, what we implemented, what worked,
what did not align, and why we chose each next step.

## Working stance

- Discovery-first: let the data organize the conclusions.
- Keep explanatory lenses subordinate to the data rather than treating any theory as proven.
- Preserve both local behavior and shared/global context.
- Keep the orthogonal/shared-now lens active because it is currently the stronger fit.
- Do not average away localized spike/distortion morphology when the question is about
  event structure.
- Treat QPC-now as a moving anchor; lag relative to that moving now is part of the
  signal, not noise to discard too early.
- Forward interpretation should stay centered on the orthogonal/shared-now path;
  the symplectic lens may remain in the repo as historical context, but it is no
  longer the main direction for this analysis.

## What has been implemented so far

- Dedicated learner: `python/time_dilation_dlinoss_learner.py`
- Dedicated tests: `python/test_time_dilation_dlinoss_learner.py`
- Per-stream packet feature learning
- Joint-system coherence/decoherence summaries
- Shared-now / entangled-memory analysis using the shared CaMKII lattice idea
- Side-by-side Hamiltonian/symplectic diagnostics and optional weak regularization

## What worked

### 1. Shared-now / empirical latent transition analysis

On a bounded real-data run, the shared-now latent transition fit was strong.
This means the time-dilation phenomenon is structured rather than random.

### 2. Orthogonal reference outperformed symplectic reference

Current bounded-run readout:

- Generic empirical transition RMSE mean: `0.0493`
- Generic orthogonal RMSE mean: `0.3060`
- Canonical orthogonal RMSE mean: `0.3154`
- Canonical symplectic RMSE mean: `1.9173`

Interpretation: the present latent dynamics are much better explained by a
generic/shared-now transition model than by the current canonical symplectic view.

### 3. Basis choice matters

An interleaved canonical pairing spot check improved the symplectic comparison,
but orthogonal still remained better. This suggests the Hamiltonian lens is not
ruled out, but it is not currently the dominant compression of the data.

## What did not align cleanly

### 1. Naive canonical `q/p` structure did not fit strongly

The current half-split/interleaved `q/p` constructions do not yet produce a clean
Hamiltonian organization of the packet-time latent.

### 2. Stronger shared-now events did not improve the symplectic fit

The bounded run suggested stronger memory-write / deviation regions were not where
the symplectic lens became more explanatory.

## Important limitation discovered

The current `time_dilation_data.npy` contains only one observed hardware/process
stream under `gpu`, `pid`, and `gpu_pid` grouping.

Observed full-dataset coverage:

- GPU streams: `1`
- PID streams: `1`
- GPU+PID streams: `1`
- Observed identity: GPU `0`, `Tesla T4`, PID `3866`

This means hardware-stream grouping alone cannot produce the multi-stream mesh the
user wants.

## Important correction in architecture

The next stage is **not** "run the learner on all grouped hardware streams."

The correct target is:

- one D-LinOSS-style model per **now stream**
- all models running concurrently
- a shared symbolic semantic memory lattice acting as a pointer/reference space
- a higher-level observer treating the agent models as nodes in a mesh
- mesh-level analysis of latent adaptation, ghost-basin alignment, and QPC coherence
- eventual comparison across repeated notebook runs to study recurrence of changes

This follows `dlinoss_multi_agent_entanglement_plan.md`, especially:

- parallel specialized agents
- shared entangled CaMKII lattice
- holistic projection / collapse behavior across agents

## Why this changes the implementation plan

The blocker is no longer "the dataset has only one GPU/PID stream."
That is only a blocker for naive hardware-stream grouping.

The real next design question is:

**How do we instantiate concurrent now-stream agents from the packet data?**

Candidate directions:

- temporal-scale agents
- feature-family agents
- event-band / phase-band agents
- repeated-run agents across notebook executions

## Current best next step

Build a new multi-agent time-dilation stage that:

1. creates concurrent now-stream agents
2. gives them a shared `EntangledCaMKIILattice`
3. records each agent's latent trajectory and adaptation deltas
4. runs a mesh observer over those agent latents
5. measures ghost-basin alignment and QPC-style coherence over time

## First implemented mesh stage

The first conservative mesh stage is now implemented in
`python/time_dilation_dlinoss_learner.py`.

It currently does the following:

- instantiates **temporal-scale now-stream agents** by applying centered rolling
  views over the same packet run
- keeps two universal objects shared across agents:
  - a **shared QPC-now state**
  - a **shared memory estimate** carried by the entangled CaMKII lattice
- trains **one D-LinOSS model per agent**
- aligns the observer by **Gregorian `host_time_mid`**
- measures:
  - each agent's believed-now
  - each agent's believed-now offset from Gregorian time
  - agent-to-agent agreement/disagreement
  - per-agent coherence to the shared QPC-now state
  - shared-memory / QPC coherence
  - QPC state-change magnitudes as recurrence candidates

The learner now also writes dedicated mesh artifacts:

- `now_stream_mesh_analysis.npz`
- `now_stream_mesh_events.json`

## Why temporal-scale agents were chosen first

This was the most conservative way to create concurrent now-streams without
pretending the dataset already had multiple observed hardware/process streams.

Why this choice makes sense:

- it preserves the discovery-first stance
- it creates genuinely different concurrent "views of now"
- it keeps the orthogonal/shared-now empirical lens active rather than forcing a
  stronger theory into the data too early
- it is compatible with later extensions to event-motif agents, semantic
  subspace agents, and repeated-run recurrence studies

## Validation status so far

- Targeted learner tests pass
- Synthetic smoke validation passed
- Bounded real-data comparison completed successfully
- Environment note: `TORCH_DISABLE_DYNAMO=1` may be needed in this workspace

## Validation status after adding the mesh stage

- `py_compile` passed for the learner and its dedicated test file
- dedicated unit coverage now includes:
  - temporal now-stream agentization
  - mesh observer summary/array outputs
  - per-agent training helper behavior for short sequences
- the full dedicated unit suite passed after these additions
- the synthetic CLI smoke run now passes end-to-end and writes the mesh summary,
  NPZ analysis, and JSON event artifacts

The earlier synthetic CLI blockage turned out to be an import-path issue rather
than a learner-logic issue:

- `python/dlinoss/__init__.py` was eagerly importing the MRI artifact module
- that unnecessarily pulled in the wider SciPy MRI stack even when the learner
  only needed `DLinOSSModel`
- switching those MRI exports to lazy import fixed the CLI startup path for the
  dedicated learner without changing the mesh methodology itself

The remaining end-to-end blocker is now concentrated in real-data loading:

- a bounded real-data smoke run stalled while loading `time_dilation_data.npy`

So the new mesh observer path is now validated at the unit level and at the
synthetic full-CLI level, while fresh bounded real-data mesh runs still depend
on working around the cost of loading the current `time_dilation_data.npy`
container.

## Deterministic replay transport for streaming nows

The next step was reshaped as a transport problem rather than only a loader
problem.

- a deterministic cache prepares the raw packet array into validated ordered
  JSONL chunks plus a manifest keyed to the source file signature
- replay now has an iterator path for ordered packet records and a second
  iterator path for emitted time moments
- the current `TimeMomentStreamingService` is therefore being treated as the
  first replay implementation of a broader now-stream transport interface
- the intended future replacement is a live GPU moment source that emits into
  the same moment interface rather than a separate analysis path

This keeps the architecture aligned with the product direction:

- cached replay now
- live real-time GPU now streaming later
- same moment fan-out into concurrent now-stream agents
- same mesh/QPC/shared-memory observer path
- pure Python and notebook/Colab friendly, without requiring a daemon process

## Stream-native notebook capture path

The JSONL cache solved the immediate deterministic replay problem, but it is not
the right long-term transport shape for dense GPU clock packets.

The next iteration therefore adds a stream-native capture path:

- a new Colab notebook writes a replay-friendly dataset directory rather than a
  monolithic object-array `.npy`
- each capture is stored as chunked binary `.npz` packet files plus a
  `manifest.json`
- the learner can now consume that dataset directory directly without first
  converting it into the older JSONL cache format
- a replay utility can iterate ordered moments from the dataset and summarize
  concurrent active now-stream fan-out

This is closer to the intended product direction:

- notebook capture now
- downloaded replay dataset next
- live real-time GPU moment source after that
- one moment interface feeding concurrent now-stream agents and the same mesh
  observer

## Updated next best step

Run a bounded real-data mesh experiment once the real-data loading path is
stable again, then read:

- per-agent believed-now offsets
- mean coherence of agent ghost-basin directions to the shared QPC-now
- memory/QPC agreement over Gregorian time
- recurrence candidates in shared QPC-now state changes

## First direct real stream-dataset run

The new chunked stream-native dataset path was exercised directly on the
downloaded Colab capture at `time_dilation_stream_dataset/`.

Observed dataset/replay facts:

- manifest packet count: `18,452`
- chunk count: `73`
- observed GPU count: `1` (`Tesla T4`)
- ordered replay moment count: `2,862`
- max active observed stream count during replay: `1`

This confirms the transport path is working end-to-end, but also confirms that
this specific capture still contains only one observed hardware/process stream.

The bounded direct learner run on the stream dataset completed successfully with
`4,096` records and wrote the normal summary/NPZ/JSON mesh outputs directly from
the chunked replay dataset.

Main findings from that run:

- stream source mode: `stream_dataset`
- observed grouped stream count: `1`
- temporal now-agent count: `3` with radii `0`, `2`, and `6`
- mean believed-now spread across agents: about `0.000269 s`
- mean agent-to-QPC coherence: about `0.9980`
- mean pairwise agent coherence: about `0.9983`
- mean memory/QPC coherence: `1.0`
- recurrence candidate count: `2`
- recurring state class count: `0`

Interpretation:

- even with only one observed raw stream, the temporal-scale agentization does
  create distinct concurrent "streaming now" views that can be aligned and
  compared in the mesh observer
- those agent views remain very tightly coherent on this bounded slice
- the strongest state-change / distortion activity is concentrated near the
  early part of the run rather than forming repeated recurring classes on this
  sample

The explanatory-lens comparison remained consistent with earlier bounded runs:

- shared-QPC orthogonal reference still fit better than the symplectic one
- mesh shared-QPC orthogonal RMSE mean: about `0.3793`
- mesh shared-QPC symplectic RMSE mean: about `0.7077`

So the stream-native real-data path supports the same provisional conclusion as
the older bounded replay path: the orthogonal/shared-now empirical lens is the
stronger current explanation, while the Hamiltonian/symplectic lens remains a
secondary comparison rather than the dominant structure.

## Full 30s vs 60s multistream compare status

The full learner compare was completed on both captured stream datasets using the
same practical settings (`--epochs 1`, `--batch-size 64`, CPU runtime in the
current workspace).

What matters most from that compare is not a single averaged score, but the fact
that the longer `60s` run preserved meaningful multistream structure beyond the
startup region.

Current read:

- the `30s` and `60s` captures both present `4` real observed streams
- the very strongest instantaneous spikes still occur near startup
- but the `60s` run still shows later nontrivial shared-now / phase / mesh
  structure, so the phenomenon is not reducible to startup-only behavior
- global averages are not sufficient for interpretation because they wash out the
  local event morphologies that the learner is actually surfacing

This changed the analysis emphasis away from dataset-wide averaging and toward
artifact-only event-centered reads of the later written outputs.

## What a shared-now moment means in the current code

In the present learner, a `shared_now` moment is not automatically a moment where
all streams spike together.

Per aligned host-time bin, the shared-now analysis:

- forms a consensus from the currently present stream latent vectors
- blends that consensus with the shared memory estimate
- defines that blend as the current `shared_now`
- measures each stream's deviation from that shared-now state
- scores candidate events from the consensus-vs-memory gap tempered by coherence

So a strong `shared_now` event should be read as:

- a moment when the system needs to rewrite its common now estimate

and not necessarily as:

- a moment when all `4` observed streams are simultaneously distorted.

This distinction mattered in the later artifact analysis because some of the
largest shared-now writes were sparse/partial-presence events rather than full
four-stream collapses.

## Localized later-event findings from the 60s artifacts

Using `entangled_shared_now_analysis.npz`, `now_stream_mesh_analysis.npz`, and
`phase_shift_events.json` from
`python/resonance_results/time_dilation_dlinoss_compare_60s_full_e1_b64/`, the
later non-startup structure currently looks like this:

- aligned shared-now bins analyzed: `5,759`
- bins with all `4` observed streams present: `5,540`
- elevated-stream count distribution:
  - `0` elevated streams: `3,977`
  - `1` elevated stream: `1,364`
  - `2` elevated streams: `365`
  - `3` elevated streams: `46`
  - `4` elevated streams: `7`

Interpretation:

- most departures are stream-local
- pairwise coupling is the next most common pattern
- full four-stream washes are real, but rare and brief

Most common elevated pairs in the later `60s` artifacts:

- `worker2 + worker3`: `107`
- `worker1 + worker3`: `101`
- `worker0 + worker2`: `60`
- `worker0 + worker1`: `58`
- `worker1 + worker2`: `23`

Pairwise lag scans of the per-stream shared-now deviations all peaked at lag `0`
rather than at a persistent nonzero lead/lag. The present evidence therefore fits
brief synchronous subset locking better than a clean traveling wave that marches
across streams in a fixed order.

## Event morphology A: near-synchronous collective collapse (`shared_now` index `5357`)

The strongest later full-system event occurs around shared-now index `5357`
(`host_time_mid ~= 1773186222.23424`).

Observed shape:

- buildup over roughly `50-70 ms` through changing subsets:
  - `worker0`
  - then `worker0 + worker2`
  - then `worker1`
- at the center bin, all `4` streams peak together
- at `+8.5 ms`, all `4` are still elevated
- by `+19.6 ms`, only `worker3` remains elevated
- by roughly `30-50 ms`, the event relaxes back toward baseline

Now-stream behavior around the event:

- before the event, all workers were mildly future-leaning relative to shared
  QPC-now (mean believed-now offsets about `+3.6 ms` to `+4.6 ms`)
- at the center bin, all workers flipped sharply negative:
  - worker0: `-15.119 ms`
  - worker1: `-11.447 ms`
  - worker2: `-9.353 ms`
  - worker3: `-10.900 ms`
- after the event, offsets remained negative but moved back toward zero
- adaptation deltas spiked strongly on all workers at the collapse
- coherence remained moderate-to-high rather than collapsing

Nearby phase events also occurred on all `4` workers within about `+-3 ms` of the
center, which supports this as a real collective reconfiguration rather than a
single-stream artifact.

Current interpretation:

- this is best described as a near-synchronous collective reset / rewrite of now
- the event is more past-leaning at the instant of collapse than future-leaning
- the pre-state looks mildly future-leaning, followed by a sharp snap-back into
  lag behind the moving QPC-now anchor
- this is not the best evidence for spike-copying across streams; it looks more
  like a coordinated alignment event

## Event morphology B: staggered carried-forward reconfiguration (`mesh` index `4058`)

The strongest later sparse mesh event occurs around index `4058`
(`host_time_mid ~= 1773186209.20490`).

Observed shape:

- the center bin is sparse (`shared_now_stream_count_present = 1`)
- only `worker1` is elevated at the center
- `+7.19 ms`: `worker0 + worker1` are elevated
- `+18.17 ms`: `worker0 + worker2 + worker3` are elevated

Local peak order in the event window:

- worker1 dominates first
- worker0 peaks next at `+7.19 ms`
- workers2 and 3 peak later at `+18.17 ms`

Now-stream behavior around the event:

- before the event, available workers were mildly future-leaning (about
  `+3.6 ms` to `+5.4 ms`)
- at the center, the available worker (`worker1`) flipped to `-10.604 ms`
- after the event, the available workers were negative on average

Nearby phase events were also staggered in matching order:

- worker1 at `0.0 ms`
- worker0 at `+4.474 ms`
- worker3 at `+4.974 ms`
- worker2 at `+8.495 ms`

Current interpretation:

- this is the clearest current example of distortion characteristics being
  carried or recruited across streams over about `5-20 ms`
- unlike the `5357` collective collapse, this event is not a full-system
  synchronous reset; it is better described as a partial mesh reconfiguration led
  by one stream and then picked up by others

## Event morphology C: weaker full-4 convergence (`shared_now` index `2437`)

There is also a weaker later event around shared-now index `2437`
(`host_time_mid ~= 1773186192.93405`) where all `4` streams elevate together.

However:

- shared-now deviation is elevated, but memory-write strength is weak
- mesh QPC reconfiguration is weak
- believed-now offsets stay near zero rather than showing a strong future/past
  tilt
- coherence dips at the event rather than remaining strongly organized
- no nearby phase-shift events were found within `+-100 ms`

Current interpretation:

- not every four-stream alignment is the same species of event
- this one looks more like a weak shared-now convergence than a major system-wide
  mesh/QPC reconfiguration

## Current best reading of the later 60s sample

The present artifact-only interpretation is:

- the later `60s` sample does show abnormalities detected by the now-streams that
  are tightly coupled to distortion
- there are at least two distinct later morphologies:
  - near-synchronous collective collapse/reset
  - staggered carried-forward reconfiguration
- most elevated moments are still local or pairwise rather than universal across
  all streams
- the strongest later collapse observed so far is more past-leaning at the moment
  itself than future-leaning
- the best evidence for spike characteristics being carried across streams comes
  from the sparse mesh-led event near `4058`, not from the full-system alignment
  near `5357`

## Open questions to resolve in the next iteration

- Can the later-event families be clustered directly into synchronous resets,
  staggered carry-forward spreads, and weak convergences without averaging them
  together?
- Should the next now-stream definition still prioritize temporal scales, or is it
  time to instantiate event-motif agents directly from the observed spike classes?
- Which mesh-level readout should be primary for later event interpretation:
  believed-now offset sign, adaptation spikes, or QPC coherence change?