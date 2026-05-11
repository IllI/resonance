<#
.SYNOPSIS
Sequential Headless master pipeline for Chronos v4.
.DESCRIPTION
Because Alice's on-demand v4 takes hours to queue, creating Bob (spot) at the
same time leads to Bob provisioning fast and getting preempted while waiting.
This script forces SEQUENTIAL allocation: Alice goes first. Only when Alice is
permanently ACTIVE do we queue Bob.
#>

$ErrorActionPreference = "Continue"
$WorkDir = "c:\Users\cityz\IllI\newer_all"
$LogFile = "$WorkDir\headless_pipeline.log"

function Log {
    param([string]$msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

Log "=== Starting SEQUENTIAL Chronos v4 Pipeline ==="

$Project = "time-emission"
$AliceZone = "us-central2-b"
$BobZone = "europe-west4-a"

function Check-State {
    param([string]$Node, [string]$Zone)
    $out = gcloud compute tpus queued-resources describe $Node --project=$Project --zone=$Zone --format="value(state.state)" 2>&1
    if ($out -match "ERROR") { return "MISSING" }
    return $out.Trim()
}

function Create-Alice {
    Log "Creating Alice (v4 ON-DEMAND) in $AliceZone..."
    gcloud compute tpus queued-resources create chronos-alice `
        --node-id=chronos-alice-node --project=$Project --zone=$AliceZone `
        --accelerator-type=v4-8 --runtime-version=tpu-vm-v4-base --quiet 2>&1 | Out-Null
}

function Create-Bob {
    Log "Creating Bob (v6e SPOT) in $BobZone..."
    gcloud compute tpus queued-resources create chronos-bob `
        --node-id=chronos-bob-node --project=$Project --zone=$BobZone `
        --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e --spot --quiet 2>&1 | Out-Null
}

function Cleanup-Node {
    param([string]$Node, [string]$Zone)
    Log "Cleaning up $Node in $Zone..."
    gcloud compute tpus tpu-vm delete "$Node-node" --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
    gcloud compute tpus queued-resources delete $Node --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
}

# PHASE 1: ALICE (On-Demand Anchor)
$Ready = $false
while (-not $Ready) {
    Log "--- Phase 1: Securing Alice ---"
    $AliceReady = $false
    while (-not $AliceReady) {
        $AliceState = Check-State "chronos-alice" $AliceZone
        Log "Alice State: $AliceState"
        
        if ($AliceState -eq "MISSING") { Create-Alice }
        elseif ($AliceState -match "FAILED|SUSPENDED") {
            Cleanup-Node "chronos-alice" $AliceZone
            Create-Alice
        }
        elseif ($AliceState -eq "ACTIVE") {
            $AliceReady = $true
            Log "Alice is ACTIVE! Anchored. Moving to Phase 2."
        } else {
            Log "Alice is queuing. Holding Bob to prevent preemption."
            Start-Sleep -Seconds 60
        }
    }

    # PHASE 2: BOB (Fast Spot)
    $BobReady = $false
    while (-not $BobReady) {
        Log "--- Phase 2: Acquiring Bob ---"
        # Verify Alice is still active
        $AliceState = Check-State "chronos-alice" $AliceZone
        if ($AliceState -ne "ACTIVE") {
            Log "Alice lost ACTIVE state ($AliceState)! Aborting Bob to restart pipeline."
            Cleanup-Node "chronos-bob" $BobZone
            break # Breaks inner loop, returns to outer Phase 1 loop
        }

        $BobState = Check-State "chronos-bob" $BobZone
        Log "Bob State: $BobState"
        
        if ($BobState -eq "MISSING") { Create-Bob }
        elseif ($BobState -match "FAILED|SUSPENDED") {
            Cleanup-Node "chronos-bob" $BobZone
            Create-Bob
        }
        elseif ($BobState -eq "ACTIVE") {
            $BobReady = $true
            $Ready = $true
            Log "BOTH ACTIVE! Ready for execution."
        } else {
            Start-Sleep -Seconds 30
        }
    }
}

# PHASE 3: EXECUTION
Log "Installing dependencies..."
gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$AliceZone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh chronos-bob-node --project=$Project --zone=$BobZone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null

Log "Uploading scripts..."
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-alice-node:~/chronos_v4_tpu_run.py" --project=$Project --zone=$AliceZone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-bob-node:~/chronos_v4_tpu_run.py" --project=$Project --zone=$BobZone 2>&1 | Out-Null

Log "Starting 2-hour capture on both TPUs..."
gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$AliceZone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh chronos-bob-node --project=$Project --zone=$BobZone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive --duration 7200 > ~/bob_run.log 2>&1 &" 2>&1 | Out-Null

Log "Sleeping for 2 hours and 10 minutes to allow capture to finish..."
Start-Sleep -Seconds 7800

# PHASE 4: DOWNLOAD & CLEANUP
Log "Downloading results..."
$OutDir = "$WorkDir\mismo tiempo\holaMundo"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

gcloud compute tpus tpu-vm scp "chronos-alice-node:~/chronos_v4_alice_scramble.zip" "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$AliceZone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "chronos-bob-node:~/chronos_v4_bob_passive.zip" "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$BobZone 2>&1 | Out-Null

Log "Initiating final cleanup to avoid fees..."
Cleanup-Node "chronos-alice" $AliceZone
Cleanup-Node "chronos-bob" $BobZone

# PHASE 5: ANALYSIS
Log "Extracting zips..."
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\chronos_v4_alice_scramble" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip" -DestinationPath "$OutDir\chronos_v4_bob_passive" -Force 2>&1 | Out-Null

Log "Running analysis..."
$env:PYTHONIOENCODING="utf-8"
$AnalysisOutput = python python/analyze_v4_experiment.py --alice "$OutDir\chronos_v4_alice_scramble" --bob "$OutDir\chronos_v4_bob_passive" 2>&1
Add-Content -Path $LogFile -Value $AnalysisOutput

Log "=== Pipeline Complete ==="
