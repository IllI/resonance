# Program P2: TPU Execution Instructions

These instructions formalize our deployment strategy for Cloud TPUs based on the successful run of Program P. It ensures strict compliance with Google's TPU Research Cloud (TRC) guidelines, preventing queue hangs, SSH timeouts, and quota exhaustion.

## 1. Approved Target Zones

According to your TRC quota, you are authorized for the following zones. **Always use the modern TPU VM Architecture and Queued Resource API.**

* `europe-west4-b` : v5litepod-4 (Spot)
* `us-central2-b`  : v4-8 (On-Demand and Spot)
* `us-central1-a`  : v5litepod-4 (Spot)
* `us-east1-d`     : v6e-8 (Spot)
* `europe-west4-a` : v6e-8 (Spot)

## 2. Best Practices for TRC Queuing

**Rule 1: Prefer On-Demand First**
If `us-central2-b` on-demand quota is available, prefer it over preemptible nodes.

**Rule 2: Proper Spot Flagging**
For all other zones, you must explicitly pass the `--spot` flag to avoid unexpected billing.

**Rule 3: Clean Up Suspended Nodes**
Spot nodes can transition to `SUSPENDED` if capacity is reclaimed. **Do not let them linger.** Run the deletion commands to clear them out, or they will count against your quota.

## 3. Step-by-Step Execution Pipeline

### Step 3.1: Queue the Resource
Select your target zone and resource. For example, `europe-west4-b`:
```powershell
gcloud compute tpus queued-resources create prog-p2-v5eb `
  --node-id=prog-p2-v5eb-node `
  --project=time-emission `
  --zone=europe-west4-b `
  --accelerator-type=v5litepod-4 `
  --runtime-version=v2-alpha-tpuv5-lite `
  --spot
```

### Step 3.2: Wait for ACTIVE state
Check the queue status until it says `ACTIVE`.
```powershell
gcloud compute tpus queued-resources list --project=time-emission --zone=europe-west4-b
```
*(Note: If you omit the `--zone` flag, gcloud may return 0 items depending on your default config.)*

### Step 3.3: Transfer Files
Once active, securely copy your JAX-native script to the node.
```powershell
gcloud compute tpus tpu-vm scp .\emergent_quantum_geometries\program_p2_tpu.py prog-p2-v5eb-node:/home/cityz/program_l/ --project=time-emission --zone=europe-west4-b
```

### Step 3.4: Execute Asynchronously
Run the program in the background using `nohup` so that local shell disconnects do not kill the simulation. Ensure the output is piped to a log file.
```powershell
gcloud compute tpus tpu-vm ssh prog-p2-v5eb-node --project=time-emission --zone=europe-west4-b `
  --command="nohup python3 -u /home/cityz/program_l/program_p2_tpu.py [ARGS...] > /home/cityz/program_l/prog_p2_run.log 2>&1 &"
```

### Step 3.5: Monitor and Download
Tail the log to verify the JAX JIT compilation completes and the fast vector loops begin.
```powershell
gcloud compute tpus tpu-vm ssh prog-p2-v5eb-node --project=time-emission --zone=europe-west4-b --command="tail -n 25 /home/cityz/program_l/prog_p2_run.log"
```
Once complete, download the results.
```powershell
gcloud compute tpus tpu-vm scp prog-p2-v5eb-node:/home/cityz/program_l/program_p2_results.json .\emergent_quantum_geometries\ --project=time-emission --zone=europe-west4-b
```

## 4. Teardown Compliance (CRITICAL)

Immediately after downloading the results, or if the node transitions to `SUSPENDED` or `PREEMPTED`, you must force-delete the resource to release the TRC quota.
```powershell
gcloud compute tpus queued-resources delete prog-p2-v5eb --project=time-emission --zone=europe-west4-b --force --quiet
```
Wait for the completion output to confirm the underlying `tpu-vm` node is fully destroyed.
