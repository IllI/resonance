# Chronos v4 TRC — Setup & Run Guide

## TRC Approved Zones (FREE)

| TPU | Zone | Chips | Type |
|---|---|---|---|
| **v4** | `us-central2-b` | 32 | **on-demand** ← Alice |
| **v6e** | `europe-west4-a` | 64 | spot ← Bob |
| v5e | `europe-west4-b` | 64 | spot |
| v5e | `us-central1-a` | 64 | spot |
| v6e | `us-east1-d` | 64 | spot |

> [!IMPORTANT]
> You MUST use the Queued Resource API (not legacy `tpu-vm create`) per TRC guidelines.
> Alice (v4 on-demand) = no preemption. Bob (v6e spot) = may be preempted, but geographic isolation guaranteed.

---

## Step 1: Open Cloud Shell

In the GCP Console, click the **`>_`** icon (top-right toolbar). When it opens, paste these commands.

---

## Step 2: Create Alice — v4 on-demand, us-central2-b

```bash
# On-demand v4 (TRC free, no --best-effort flag)
gcloud compute tpus queued-resources create chronos-alice \
  --node-id=chronos-alice-node \
  --project=time-emission \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --runtime-version=tpu-vm-v4-base \
  --quiet
```

Check status (wait for ACTIVE):
```bash
gcloud compute tpus queued-resources describe chronos-alice \
  --project=time-emission --zone=us-central2-b
```

---

## Step 3: Create Bob — v6e spot, europe-west4-a

```bash
# Spot v6e (TRC free, use --best-effort for spot quota)
gcloud compute tpus queued-resources create chronos-bob \
  --node-id=chronos-bob-node \
  --project=time-emission \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --runtime-version=v2-alpha-tpuv6e \
  --best-effort \
  --quiet
```

Check status:
```bash
gcloud compute tpus queued-resources describe chronos-bob \
  --project=time-emission --zone=europe-west4-a
```

---

## Step 4: SSH + Install Dependencies

Once both show ACTIVE:

**Alice:**
```bash
gcloud compute tpus tpu-vm ssh chronos-alice-node \
  --project=time-emission --zone=us-central2-b \
  --command="pip install -q jax[tpu] scipy && echo READY"
```

**Bob:**
```bash
gcloud compute tpus tpu-vm ssh chronos-bob-node \
  --project=time-emission --zone=europe-west4-a \
  --command="pip install -q jax[tpu] scipy && echo READY"
```

---

## Step 5: Upload & Run Scripts

Upload capture + learner script to each VM:
```bash
# Alice
gcloud compute tpus tpu-vm scp \
  chronos_v4_tpu_run.py chronos-alice-node: \
  --project=time-emission --zone=us-central2-b

# Bob  
gcloud compute tpus tpu-vm scp \
  chronos_v4_tpu_run.py chronos-bob-node: \
  --project=time-emission --zone=europe-west4-a
```

Run (in two Cloud Shell tabs):
```bash
# Tab 1 — Alice
gcloud compute tpus tpu-vm ssh chronos-alice-node \
  --project=time-emission --zone=us-central2-b \
  --command="python3 chronos_v4_tpu_run.py --role Alice_Scramble"

# Tab 2 — Bob
gcloud compute tpus tpu-vm ssh chronos-bob-node \
  --project=time-emission --zone=europe-west4-a \
  --command="python3 chronos_v4_tpu_run.py --role Bob_Passive"
```

---

## Step 6: Download Results

```bash
gcloud compute tpus tpu-vm scp \
  chronos-alice-node:chronos_v4_alice_scramble.zip . \
  --project=time-emission --zone=us-central2-b

gcloud compute tpus tpu-vm scp \
  chronos-bob-node:chronos_v4_bob_passive.zip . \
  --project=time-emission --zone=europe-west4-a
```

Then download from Cloud Shell: click **⋮ → Download file** and enter the zip paths.

---

## Step 7: ALWAYS DELETE AFTER USE

```bash
gcloud compute tpus queued-resources delete chronos-alice \
  --project=time-emission --zone=us-central2-b --quiet

gcloud compute tpus queued-resources delete chronos-bob \
  --project=time-emission --zone=europe-west4-a --quiet
```

> [!CAUTION]
> Do NOT skip Step 7. Even "SUSPENDED" queued resources may incur charges. Delete immediately after downloading your data.
