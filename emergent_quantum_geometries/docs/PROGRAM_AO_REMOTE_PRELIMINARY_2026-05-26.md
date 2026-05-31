# Program AO Remote Preliminary Analysis (2026-05-26)

Current run context:
- Zone: `europe-west4-a`
- Node: `program-an-eu-node-v1`
- Accelerator: `v6e-4` spot
- Runtime: `v2-alpha-tpuv6e`
- Remote summary inspection only, no file download

## Status

The run is still active. The summary snapshot currently contains:
- `records=1`
- `runtime_s=201.4`
- `trial_keys=free, filament_stabilized, folded_decoder, ibm_lite_folded`

The snapshot has not yet populated:
- `heldout_transport`
- `semantic_recovery`

So AO has completed the opening controller sweep for the first depth slice, but the harder evaluation blocks have not been written into the summary yet.

## Early signal

Even in the first sweep, the new controllers are behaving differently from the old AN baseline:
- `folded_decoder` is ahead of `free` in the structured condition.
- `ibm_lite_folded` stays competitive while using the constrained coupling path.
- `markov_shuffle` and `block_permute` are producing more realistic negative-control behavior than the old simple shuffle did, though the run has not yet reached the later score tables.

## What this means

This run has not finished, so there is no final verdict yet. The main useful takeaway so far is that AO is alive and using the new controller set, but the current snapshot is too early to judge the full success criteria.

## Next checkpoint

Wait for the summary to fill in:
- `heldout_transport`
- `semantic_recovery`

Those are the parts that will decide whether `folded_decoder` really beats the hard controls and whether the IBM-lite path survives a noisy regime.
