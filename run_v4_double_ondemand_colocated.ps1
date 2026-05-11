<#
.SYNOPSIS
Double On-Demand Co-Located TRC Pipeline
Ensures Alice and Bob are in the exact same datacenter (us-central2-b)
so they can share the same QPC field for the D-LinOSS experiment.
Uses the 32-chip on-demand quota for BOTH nodes to bypass spot gridlock.
#>

$ErrorActionPreference = "Continue"
$WorkDir   = "c:\Users\cityz\IllI\newer_all"
$LogFile   = "$WorkDir\pipeline_colocated.log"
$Project   = "time-emission"
$Zone      = "us-central2-b"
$Accel     = "v4-8"
$Runtime   = "tpu-vm-v4-base"
$OutDir    = "$WorkDir\mismo tiempo\holaMundo"

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

function Get-State {
    param([string]$Name)
    $s = gcloud compute tpus queued-resources describe $Name --project=$Project --zone=$Zone --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($s)) { return "MISSING" }
    return $s.Trim()
}

function Delete-Node {
    param([string]$Name)
    Log "  Deleting VM $Name-node..."
    gcloud compute tpus tpu-vm delete "$Name-node" --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
    Log "  Deleting QR $Name..."
    gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --force --quiet 2>&1 | Out-Null
}

# --- PHASE 1: Queue Alice and Bob (both on-demand) ---
Log "=== Starting Double On-Demand Co-Located Pipeline ==="
Log "--- Phase 1: Queue Alice and Bob (On-Demand, $Zone) ---"

Log "Queuing Alice - $Accel on-demand..."
$r = gcloud compute tpus queued-resources create chronos-alice `
    --node-id=chronos-alice-node `
    --project=$Project --zone=$Zone `
    --accelerator-type=$Accel `
    --runtime-version=$Runtime `
    --quiet 2>&1
Log "  Alice result: $r"

Log "Queuing Bob - $Accel on-demand..."
$r = gcloud compute tpus queued-resources create chronos-bob `
    --node-id=chronos-bob-node `
    --project=$Project --zone=$Zone `
    --accelerator-type=$Accel `
    --runtime-version=$Runtime `
    --quiet 2>&1
Log "  Bob result: $r"

# --- PHASE 2: Wait for both ACTIVE ---
Log "--- Phase 2: Polling until both ACTIVE ---"
$BothActive = $false
while (-not $BothActive) {
    $aliceState = Get-State "chronos-alice"
    $bobState   = Get-State "chronos-bob"
    Log "  chronos-alice: $aliceState  |  chronos-bob: $bobState"

    # Auto-recover if either hits a strange state
    if ($aliceState -match "FAILED|SUSPENDED|MISSING" -and $aliceState -ne "MISSING") {
        Log "  Alice hit $aliceState. Re-queuing..."
        Delete-Node "chronos-alice"
        $r = gcloud compute tpus queued-resources create chronos-alice --node-id=chronos-alice-node --project=$Project --zone=$Zone --accelerator-type=$Accel --runtime-version=$Runtime --quiet 2>&1
    }
    if ($bobState -match "FAILED|SUSPENDED|MISSING" -and $bobState -ne "MISSING") {
        Log "  Bob hit $bobState. Re-queuing..."
        Delete-Node "chronos-bob"
        $r = gcloud compute tpus queued-resources create chronos-bob --node-id=chronos-bob-node --project=$Project --zone=$Zone --accelerator-type=$Accel --runtime-version=$Runtime --quiet 2>&1
    }

    if ($aliceState -eq "ACTIVE" -and $bobState -eq "ACTIVE") {
        Log "BOTH ACTIVE - starting experiment!"
        $BothActive = $true
    } else {
        Start-Sleep 60
    }
}

# --- PHASE 3: Install dependencies ---
Log "--- Phase 3: Installing dependencies ---"
$cmd = "pip install -q jax[tpu] scipy"
gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$Zone --command=$cmd 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh chronos-bob-node   --project=$Project --zone=$Zone --command=$cmd 2>&1 | Out-Null

# --- PHASE 4: Upload scripts ---
Log "--- Phase 4: Uploading scripts ---"
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-alice-node:~/chronos_v4_tpu_run.py" `
    --project=$Project --zone=$Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-bob-node:~/chronos_v4_tpu_run.py" `
    --project=$Project --zone=$Zone 2>&1 | Out-Null

# --- PHASE 5: Fire 2-hour capture ---
Log "--- Phase 5: Starting synchronized 2-hour capture ---"
$aliceCmd = 'nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &'
$bobCmd   = 'nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive   --duration 7200 > ~/bob_run.log   2>&1 &'
gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$Zone --command=$aliceCmd 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh chronos-bob-node   --project=$Project --zone=$Zone --command=$bobCmd   2>&1 | Out-Null

Log "Capture launched. Sleeping 2h10m..."
Start-Sleep -Seconds 7800

# --- PHASE 6: Download artifacts ---
Log "--- Phase 6: Downloading results ---"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

gcloud compute tpus tpu-vm scp "chronos-alice-node:~/chronos_v4_alice_scramble.zip" `
    "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "chronos-bob-node:~/chronos_v4_bob_passive.zip" `
    "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$Zone 2>&1 | Out-Null

# --- PHASE 7: Delete everything - billing safety ---
Log "--- Phase 7: Billing cleanup (CRITICAL) ---"
Delete-Node "chronos-alice"
Delete-Node "chronos-bob"

# --- PHASE 8: Analysis ---
Log "--- Phase 8: Running analysis ---"
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\alice" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip"   -DestinationPath "$OutDir\bob"   -Force 2>&1 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$out = python python/analyze_v4_experiment.py --alice "$OutDir\alice" --bob "$OutDir\bob" 2>&1
Add-Content -Path $LogFile -Value $out
Log "=== Pipeline Complete ==="
