# TRC Cloud TPU Rules for `time-emission`

Source: `[TRC] You have access to free Cloud TPUs.eml`

Full operational runbook: `emergent_quantum_geometries/docs/TPU_BIBLE.md`

## Goal: "Never Get Charged" (Practical Definition)

TRC can cover TPU *compute* in specific zones / quota combinations, but **it does not automatically make all Google Cloud usage free**.

To keep your billed cost at (or extremely close to) **$0**, you must:

1. Use only TRC-covered TPU types/zones, *and*
2. Avoid billable non-TPU services (especially **Network Internet Egress**), *and*
3. Tear down **all** resources immediately after a run (including queued resources).

Absolute “never charged” is only guaranteed if you **do not transfer meaningful data out to your laptop over the public internet**. The moment you SCP large artifacts from a Europe zone to a US laptop, you will likely incur internet egress charges.

## TRC-Covered TPU Quota Combinations

Use only the quota combinations below for TRC-covered TPU work:

- `europe-west4-b`: 64 spot Cloud TPU `v5e` chips
- `us-central2-b`: 32 on-demand Cloud TPU `v4` chips
- `us-central1-a`: 64 spot Cloud TPU `v5e` chips
- `us-east1-d`: 64 spot Cloud TPU `v6e` chips
- `us-central2-b`: 32 spot Cloud TPU `v4` chips
- `europe-west4-a`: 64 spot Cloud TPU `v6e` chips

## Non-Negotiables (Avoid Charges)

1. **Zone + Type must match TRC**:
   - Only create TPUs in the zones listed above.
   - Only use the allowed TPU generation for that zone.
   - If the TRC quota says spot/preemptible, you must pass `--spot` (or it will bill on-demand if it even succeeds).

2. **Do not leave anything running**:
   - Delete **TPU VMs** after the run.
   - Delete **queued resources** after the run.
   - Queued resources in nonterminal states can linger and are easy to forget.

3. **Avoid network egress (the #1 surprise bill)**:
   - **Do not SCP big folders** from TPU to your laptop.
   - Prefer: print key metrics to stdout, and keep only small logs/summaries locally.
   - If you must download anything, keep it tiny (single summary JSON + short log tail).

4. **Prefer US zones to avoid EU→US egress categories**:
   - If you need local downloads at all, prefer `us-east1-d` (TRC-covered) over Europe zones.

5. **Avoid other billable services by default**:
   - Don’t stream large logs to external services.
   - Don’t write big datasets to GCS unless you’re explicitly accepting storage + egress costs.

## “No-Charge” Run Pattern (Recommended)

This pattern is designed to keep billing at ~$0:

1. Create TPU (spot) in a TRC-covered zone.
2. Upload code (small).
3. Run experiment.
4. **Inspect results on the TPU**:
   - `tail` the log
   - print summary metrics
5. Optionally download ONLY:
   - `summary.json` (a few KB)
   - a small `.log` (or just a log tail)
6. Delete TPU VM and queued resource immediately.

## Deployment Bib (Read Before Every TPU Launch)

Before launching any `program_*` experiment, confirm all of the following:

- Target is a TRC-covered TPU config.
- Prefer `us-east1-d` + `v6e` + `--spot` for runs that need even tiny downloads.
- Do not use recursive SCP.
- Do not download raw trajectories, checkpoints, `.npz`, `.npy`, images, or full result folders by default.
- Remote experiment must write a compact `summary.json`.
- Remote wrapper may create a short `run_tail.log` with `tail -n 200`.
- Local download allow-list is only `summary.json`, `run_tail.log`, and tiny diagnostic JSON files.
- Any larger artifact requires an explicit user approval in the thread before transfer.
- Delete losing queued resources immediately after one zone becomes active.
- After the run, delete the TPU VM and any queued resource unless the user explicitly asks to keep it alive.

## Commands / Checklists

### Create (example)

Use a TRC-covered zone/type. Example v6e spot (adjust chips/size to your actual request):

```powershell
$p="time-emission"
$z="us-east1-d"
$n="program-node-v6e"
gcloud compute tpus tpu-vm create $n --project=$p --zone=$z --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e --spot
```

### Verify nothing is running (before/after)

```powershell
$p="time-emission"
gcloud compute tpus tpu-vm list --project=$p
gcloud compute tpus queued-resources list --project=$p
```

### Safe download rule

Only download tiny files. Never run recursive scp of result folders by default.

### Teardown (must run)

```powershell
$p="time-emission"
$z="us-east1-d"
$n="program-node-v6e"
gcloud compute tpus tpu-vm delete $n --project=$p --zone=$z --quiet
```

If you used queued-resources, delete those too:

```powershell
$p="time-emission"
$z="us-east1-d"
$qr="program-qr-v6e"
gcloud compute tpus queued-resources delete $qr --project=$p --zone=$z --force --quiet
```

Program AM safe default:

- TPU type: `v6e-8`
- Runtime: `v2-alpha-tpuv6e`
- Zones: `us-east1-d`, `europe-west4-a`
- Flag: `--spot`

## Billing Debug Rule (When You See Charges)

If Billing shows costs while you believe TRC should cover the TPU compute:

1. Check the SKU names. If you see:
   - `Network Internet Data Transfer Out ...`
   - `Network Inter Region Data Transfer Out ...`
   Then you are being charged for **egress**, not TPU compute.

2. Confirm you have zero TPUs/queued resources running:
   - `gcloud compute tpus tpu-vm list --project=time-emission`
   - `gcloud compute tpus queued-resources list --project=time-emission`
