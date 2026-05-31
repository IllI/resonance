# TPU Bible: TRC-Safe Experiment Operations

Last updated: 2026-05-29

Project: `time-emission`

Purpose: keep TPU experiments fast, reproducible, and as close to zero bill as possible under the TRC contract.

This document is stricter than a normal runbook because the recent billing evidence showed real non-TPU charges:

- `Network Internet Data Transfer Out from Netherlands to Americas`
- `Network Internet Data Transfer Out from Americas to Americas`

Those are network egress SKUs. TRC may cover eligible TPU compute, but it does not make public internet data transfer free.

## What The TRC Email Explicitly Says

The TRC email explicitly states:

- The project has access to listed Cloud TPU quota "free of charge for 30 days."
- The free trial is "only available for new Cloud TPUs you create in the zones listed above."
- "To avoid charges," Cloud TPUs must be created in the appropriate zone.
- "While your Cloud TPUs are free, you'll still be charged for the rest of the GCP services you use."
- Participants should delete unused Cloud TPUs and/or Queued Resources that may still be consuming quota.

The TRC email does not explicitly name network egress, SCP, downloads, or inter-region transfer. The egress rule in this runbook is an operational interpretation of the sentence about "the rest of the GCP services" combined with the actual Billing SKUs observed in this project.

## One Sentence Rule

Use TRC-covered TPU shapes, keep artifacts remote, download only tiny summaries, and delete the TPU immediately after the run.

## What The Billing Evidence Means

The Billing report grouped by SKU showed:

| SKU | Usage | Cost signal |
| --- | ---: | ---: |
| `Network Internet Data Transfer Out from Netherlands to Americas` | about `59 GiB` | about `$7` |
| `Network Internet Data Transfer Out from Americas to Americas` | about `51 GiB` | about `$6` |

Interpretation:

- This is not TPU compute.
- This is not caused by enabled APIs by themselves.
- This is not explained by a tiny `summary.json`.
- This is consistent with large downloads from TPU VMs or Compute-backed TPU workers to a local machine over public internet.

Likely causes:

- `gcloud compute tpus tpu-vm scp remote:local` on a result directory.
- Recursive SCP of `program_*_results`.
- Downloading `.npz`, `.npy`, checkpoints, images, raw trajectories, or full logs.
- Pulling large files from Europe zones to a U.S. laptop.
- Repeating the same broad download each day.

Less likely causes based on current inspection:

- Cloud Storage bucket storage. The observed bucket was only about `29 MB`.
- Idle Compute Engine VMs. None were listed during inspection.
- Persistent disks, snapshots, static IPs, GKE, or Pub/Sub. None were listed during inspection.

## Important Billing UI Trap

If `Show cumulative` is enabled in Google Cloud Billing, a day tooltip can show cumulative month-to-date cost at that date. That can make identical-looking bars or repeated amounts feel like a daily charge.

Still, if the SKU is network data transfer and the usage is tens of GiB, the charge is real egress. The fix is not to rely on TPU credits; the fix is to stop large data leaving Google Cloud.

## TRC-Covered TPU Configs

Only use the combinations below unless the TRC email is updated.

| Zone | TPU | Allocation |
| --- | --- | --- |
| `europe-west4-b` | `v5e` | spot |
| `us-central1-a` | `v5e` | spot |
| `us-central2-b` | `v4` | on-demand and spot |
| `us-east1-d` | `v6e` | spot |
| `europe-west4-a` | `v6e` | spot |

Current preferred zone:

- `us-east1-d`, `v6e-4`, `--spot`

Use Europe only when U.S. zones cannot get capacity, and then treat all analysis as remote-only.

## Absolute No-Download List

Never download these from TPU VMs by default:

- full result directories
- recursive `program_*_results`
- `.npz`
- `.npy`
- checkpoints
- raw vectors
- raw trajectories
- images or plots generated on TPU
- full logs
- environment directories
- cached packages
- any file over `1 MB` unless explicitly approved in the thread

## Allow-List

Local downloads are allowed only for tiny files:

- `program_ap_summary.json` or equivalent compact summary
- `run_tail.log` with at most a few hundred lines
- tiny diagnostic JSON files

Preferred limit:

- under `100 KB`

Hard review threshold:

- anything over `1 MB` requires an explicit approval message before transfer

Before downloading any file, check size remotely:

```powershell
$g='C:\Users\cityz\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
"y`n" | & $g compute tpus tpu-vm ssh NODE --project=time-emission --zone=ZONE --quiet --command="ls -lh /path/to/file; du -h /path/to/file"
```

## Zone-Specific Download Rules

### U.S. Zones

Tiny summary downloads are acceptable.

Allowed pattern:

```powershell
gcloud compute tpus tpu-vm scp NODE:/remote/path/summary.json .\summary.json --project=time-emission --zone=us-east1-d --quiet
```

Still prohibited:

- recursive SCP
- result folders
- large logs
- raw artifacts

### Europe Zones

Default: do not download anything.

Use remote reads:

```powershell
gcloud compute tpus tpu-vm ssh NODE --project=time-emission --zone=europe-west4-a --quiet --command="tail -n 120 /home/cityz/program_ap/prog_ap_run.log"
```

If a Europe result must be retained locally, first create a tiny text/JSON summary on the TPU and inspect its size. Only download after explicit approval.

## Safe Run Pattern

1. Create TPU in TRC-covered zone/type with the correct spot/on-demand mode.
2. Upload source only.
3. Verify JAX backend is `tpu`.
4. Run compact experiment.
5. Monitor with remote `tail`, not file download.
6. Write compact summary JSON.
7. Download only tiny summary if U.S. zone and needed.
8. Delete TPU VM immediately.
9. Delete queued resources immediately.
10. Check billing SKUs after large changes in workflow.

## Launch Checklist

Before launch, confirm:

- Zone and TPU type match TRC.
- `--spot` is present when using spot quota.
- The run writes compact summaries.
- Raw artifacts are either disabled or kept remote.
- No recursive SCP is in the script.
- No broad `gcloud storage cp -r` is in the script.
- No automatic full-log download is in the script.
- The command stamps git commit.
- The run can be stopped without losing the compact summary.

## Monitor Checklist

Safe monitoring commands:

```powershell
gcloud compute tpus tpu-vm ssh NODE --project=time-emission --zone=ZONE --quiet --command="pgrep -af '[p]rogram_ap_tunnel_eigen_recovery.py' || true; tail -n 120 /home/cityz/program_ap/prog_ap_run.log"
```

Do not stream logs continuously for hours. Poll occasionally.

If using a high-credit model in the assistant, avoid sleep-loop polling. Ask for one status pull when needed.

## Teardown Checklist

After every run:

```powershell
gcloud compute tpus tpu-vm delete NODE --project=time-emission --zone=ZONE --quiet
gcloud compute tpus queued-resources list --project=time-emission --zone=ZONE
```

Then check all TRC zones:

```powershell
gcloud compute tpus tpu-vm list --project=time-emission --zone=us-east1-d
gcloud compute tpus tpu-vm list --project=time-emission --zone=europe-west4-a
gcloud compute tpus tpu-vm list --project=time-emission --zone=us-central1-a
gcloud compute tpus tpu-vm list --project=time-emission --zone=us-central2-b
gcloud compute tpus tpu-vm list --project=time-emission --zone=europe-west4-b
```

Also check common non-TPU resources:

```powershell
gcloud compute instances list --project=time-emission
gcloud compute disks list --project=time-emission
gcloud compute addresses list --project=time-emission
gcloud compute snapshots list --project=time-emission
gcloud storage buckets list --project=time-emission
```

## Billing Debug Checklist

When costs appear:

1. Group Billing report by `SKU`.
2. Turn off `Show cumulative` to inspect per-day increments.
3. Look for:
   - `Cloud TPU`
   - `Network Internet Data Transfer Out ...`
   - `Network Inter Region Data Transfer Out ...`
   - `Cloud Storage`
   - `Cloud Logging`
4. If the SKU is `Network Internet Data Transfer Out`, find the direction:
   - `Netherlands to Americas` means Europe-to-U.S. public egress.
   - `Americas to Americas` can be U.S. TPU/Compute-to-local public egress.
5. Check whether any run script downloaded result folders.
6. Check local files for unexpectedly large summaries or artifacts.
7. Check active resources and delete anything not needed.

## Red Flags In Commands

Do not run commands containing these patterns without explicit review:

```text
scp -r
tpu-vm scp .../program_*_results
tpu-vm scp NODE:/home/... .
storage cp -r
gsutil -m cp -r
*.npz
*.npy
checkpoint
trajectory
results/
```

## Preferred Artifact Design

TPU jobs should output:

- metrics table in stdout
- compact `summary.json`
- optional `run_tail.log`

TPU jobs should not output by default:

- full recovered vectors
- full train/test states
- per-step trajectories
- plots
- tensor dumps

For AQ-style semantic work:

- Save compressed signatures only if they are tiny.
- Decode offline only from approved tiny artifacts.
- Keep raw recovered vectors remote unless explicitly approved.

## Why Last Month May Have Been Free

Plausible explanations for a new charge pattern under the same TRC contract:

- Last month used remote-only analysis and did not download large artifacts.
- Last month used U.S. zones or smaller downloads.
- This month included Europe-to-U.S. downloads.
- A script changed from `tail/cat summary` to recursive SCP or folder download.
- Billing grouped by Project hid the SKU source until grouped by SKU.
- Credits/savings may cover TPU compute but not internet egress.

The suspicious part is not that TPU compute billed daily; current evidence points away from compute. The suspicious part is that egress volume appears large and repeatable. That suggests a repeatable workflow or script transferred roughly the same artifact volume more than once.

## Operational Defaults From Now On

- U.S. TPU first when local summary download is needed.
- Europe TPU only with remote-only analysis.
- No recursive SCP ever by default.
- Tiny summary download only after remote size check.
- Delete TPU immediately after each run.
- Billing grouped by SKU after workflow changes.
- Treat egress as the main enemy, not TPU compute.
