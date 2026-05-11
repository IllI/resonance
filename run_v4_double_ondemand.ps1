<#
.SYNOPSIS
Double On-Demand Pipeline for TRC TPU Synchronization
.DESCRIPTION
Queues exactly 2 ON-DEMAND TPUs in the same zone.
Because they are on-demand, they are immune to preemption.
The script waits until both hit ACTIVE, then runs the experiment.
#>

$ErrorActionPreference = "Continue"
$WorkDir = "c:\Users\cityz\IllI\newer_all"
$LogFile = "$WorkDir\double_ondemand.log"

function Log {
    param([string]$msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

Log "=== Starting Double On-Demand Pipeline ==="

$Project = "time-emission"
$Zone = "us-central2-b"
$Runtime = "tpu-vm-v4-base"
$Accel = "v4-8"

$Nodes = @("chronos-alice", "chronos-bob")

function Check-State {
    param([string]$Name)
    $state = gcloud compute tpus queued-resources describe $Name --project=$Project --zone=$Zone --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($state)) {
        return "MISSING"
    }
    return $state.Trim()
}

function Create-Node {
    param([string]$Name)
    Log "Queuing $Name ($Accel On-Demand) in $Zone..."
    $res = gcloud compute tpus queued-resources create $Name `
        --node-id="$Name-vm" --project=$Project --zone=$Zone `
        --accelerator-type=$Accel --runtime-version=$Runtime --quiet 2>&1
    Log "  Creation result: $res"
    Start-Sleep -Seconds 5
}

function Cleanup-Node {
    param([string]$Name)
    Log "Cleaning up $Name in $Zone..."
    gcloud compute tpus tpu-vm delete "$Name-vm" --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
    gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
}

# PHASE 1: QUEUE
Log "--- Phase 1: Launching On-Demand Queues ---"
foreach ($node in $Nodes) {
    Create-Node $node
}

$BothActive = $false

# PHASE 2: WAIT FOR RESOURCES
while (-not $BothActive) {
    $ActiveCount = 0
    
    Log "Polling queue status..."
    foreach ($node in $Nodes) {
        $state = Check-State $node
        Log "  $($node): $state"
        
        if ($state -eq "ACTIVE") {
            $ActiveCount++
        }
        elseif ($state -match "FAILED|SUSPENDED|MISSING") {
            Cleanup-Node $node
            Create-Node $node
        }
    }

    if ($ActiveCount -eq 2) {
        Log "JACKPOT! Both Alice and Bob are ACTIVE!"
        $BothActive = $true
    } else {
        Start-Sleep -Seconds 60
    }
}

# PHASE 3: EXECUTION
Log "--- Phase 3: Setup and Execution ---"
Log "Installing dependencies..."
gcloud compute tpus tpu-vm ssh "chronos-alice-vm" --project=$Project --zone=$Zone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh "chronos-bob-vm" --project=$Project --zone=$Zone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null

Log "Uploading scripts..."
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-alice-vm:~/chronos_v4_tpu_run.py" --project=$Project --zone=$Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "chronos-bob-vm:~/chronos_v4_tpu_run.py" --project=$Project --zone=$Zone 2>&1 | Out-Null

Log "Starting 2-hour capture..."
gcloud compute tpus tpu-vm ssh "chronos-alice-vm" --project=$Project --zone=$Zone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh "chronos-bob-vm" --project=$Project --zone=$Zone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive --duration 7200 > ~/bob_run.log 2>&1 &" 2>&1 | Out-Null

Log "Sleeping for 2 hours and 10 minutes..."
Start-Sleep -Seconds 7800

# PHASE 4: DOWNLOAD & CLEANUP
Log "--- Phase 4: Harvest & Wipe ---"
$OutDir = "$WorkDir\mismo tiempo\holaMundo"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Log "Downloading..."
gcloud compute tpus tpu-vm scp "chronos-alice-vm:~/chronos_v4_alice_scramble.zip" "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "chronos-bob-vm:~/chronos_v4_bob_passive.zip" "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$Zone 2>&1 | Out-Null

Log "Final cleanup..."
Cleanup-Node "chronos-alice"
Cleanup-Node "chronos-bob"

# PHASE 5: ANALYSIS
Log "--- Phase 5: Analysis ---"
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\chronos_v4_alice_scramble" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip" -DestinationPath "$OutDir\chronos_v4_bob_passive" -Force 2>&1 | Out-Null

$env:PYTHONIOENCODING="utf-8"
$AnalysisOutput = python python/analyze_v4_experiment.py --alice "$OutDir\chronos_v4_alice_scramble" --bob "$OutDir\chronos_v4_bob_passive" 2>&1
Add-Content -Path $LogFile -Value $AnalysisOutput

Log "=== Pipeline Complete ==="
