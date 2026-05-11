<#
.SYNOPSIS
Shotgun Headless Pipeline for TRC TPU Synchronization
.DESCRIPTION
Queues exactly 3 spot TPUs across 3 distinct zones concurrently.
The moment ANY TWO hit ACTIVE simultaneously, it kills the third queue
to prevent quota issues, and launches the experiment on the two winners.
This bypasses long zone-specific queues by racing them against each other.
#>

$ErrorActionPreference = "Continue"
$WorkDir = "c:\Users\cityz\IllI\newer_all"
$LogFile = "$WorkDir\shotgun_pipeline_v3.log"

function Log {
    param([string]$msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

Log "=== Starting SHOTGUN Synchronized Pipeline ==="

$Project = "time-emission"

# Define the 3 functional TRC spot zones
$Nodes = @(
    @{ Id = 1; Name = "chronos-node-1"; Zone = "us-central2-b"; Accel = "v4-8"; Runtime = "tpu-vm-v4-base"; Spot = $true },
    @{ Id = 2; Name = "chronos-node-2"; Zone = "us-east1-d"; Accel = "v6e-8"; Runtime = "v2-alpha-tpuv6e"; Spot = $true },
    @{ Id = 3; Name = "chronos-node-3"; Zone = "europe-west4-a"; Accel = "v6e-8"; Runtime = "v2-alpha-tpuv6e"; Spot = $true }
)

function Check-State {
    param([string]$Name, [string]$Zone)
    # Redirect stderr to null so we only get the raw status value
    $state = gcloud compute tpus queued-resources describe $Name --project=$Project --zone=$Zone --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($state)) {
        return "MISSING"
    }
    return $state.Trim()
}

function Create-Node {
    param($NodeObj)
    $nName = $NodeObj.Name
    $nZone = $NodeObj.Zone
    $nAccel = $NodeObj.Accel
    $nRuntime = $NodeObj.Runtime
    
    Log "Queuing $nName ($nAccel Spot) in $nZone..."
    $res = gcloud compute tpus queued-resources create $nName `
        --node-id="$nName-vm" --project=$Project --zone=$nZone `
        --accelerator-type=$nAccel --runtime-version=$nRuntime --spot --quiet 2>&1
    Log "  Creation result: $res"
    Start-Sleep -Seconds 5 # Give GCP a moment to propagate
}

function Cleanup-Node {
    param([string]$Name, [string]$Zone)
    Log "Cleaning up $Name in $Zone..."
    gcloud compute tpus tpu-vm delete "$Name-vm" --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
    gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
}

# PHASE 1: THE SHOTGUN RACE
Log "--- Phase 1: Launching Shotgun Queue ---"
$Nodes | ForEach-Object { Create-Node $_ }

$Winners = @()
$Loser = $null
$RaceOver = $false

while (-not $RaceOver) {
    $ActiveCount = 0
    $CurrentActive = @()
    
    Log "Polling race status..."
    foreach ($node in $Nodes) {
        $state = Check-State $node.Name $node.Zone
        Log "  $($node.Name) ($($node.Zone)): $state"
        
        if ($state -eq "ACTIVE") {
            $ActiveCount++
            $CurrentActive += $node
        }
        elseif ($state -match "FAILED|SUSPENDED|MISSING") {
            Cleanup-Node $node.Name $node.Zone
            Create-Node $node
        }
    }

    if ($ActiveCount -ge 2) {
        # We have our 2 nodes!
        $Winners = $CurrentActive[0..1]
        
        # Identify the loser to kill it immediately
        $Loser = $Nodes | Where-Object { $_.Name -ne $Winners[0].Name -and $_.Name -ne $Winners[1].Name } | Select-Object -First 1
        
        Log "RACE OVER! Winners: $($Winners[0].Name) and $($Winners[1].Name)"
        $RaceOver = $true
    } else {
        Start-Sleep -Seconds 30
    }
}

# PHASE 2: QUOTA PROTECTION
Log "--- Phase 2: Killing the Loser to Save Quota ---"
Cleanup-Node $Loser.Name $Loser.Zone

# PHASE 3: DESIGNATION
$Alice = $Winners[0]
$Bob = $Winners[1]
Log "Designating Alice -> $($Alice.Name) in $($Alice.Zone)"
Log "Designating Bob   -> $($Bob.Name) in $($Bob.Zone)"

# PHASE 4: EXECUTION
Log "--- Phase 3: Setup and Execution ---"
Log "Installing dependencies..."
gcloud compute tpus tpu-vm ssh "$($Alice.Name)-vm" --project=$Project --zone=$Alice.Zone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh "$($Bob.Name)-vm" --project=$Project --zone=$Bob.Zone --command="pip install -q 'jax[tpu]' scipy requests" 2>&1 | Out-Null

Log "Uploading scripts..."
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "$($Alice.Name)-vm:~/chronos_v4_tpu_run.py" --project=$Project --zone=$Alice.Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "$WorkDir\chronos_v4_tpu_run.py" "$($Bob.Name)-vm:~/chronos_v4_tpu_run.py" --project=$Project --zone=$Bob.Zone 2>&1 | Out-Null

Log "Starting 2-hour capture..."
gcloud compute tpus tpu-vm ssh "$($Alice.Name)-vm" --project=$Project --zone=$Alice.Zone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &" 2>&1 | Out-Null
gcloud compute tpus tpu-vm ssh "$($Bob.Name)-vm" --project=$Project --zone=$Bob.Zone --command="nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive --duration 7200 > ~/bob_run.log 2>&1 &" 2>&1 | Out-Null

Log "Sleeping for 2 hours and 10 minutes..."
Start-Sleep -Seconds 7800

# PHASE 5: DOWNLOAD & CLEANUP
Log "--- Phase 4: Harvest & Wipe ---"
$OutDir = "$WorkDir\mismo tiempo\holaMundo"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Log "Downloading..."
gcloud compute tpus tpu-vm scp "$($Alice.Name)-vm:~/chronos_v4_alice_scramble.zip" "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$Alice.Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "$($Bob.Name)-vm:~/chronos_v4_bob_passive.zip" "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$Bob.Zone 2>&1 | Out-Null

Log "Final cleanup of winners..."
Cleanup-Node $Alice.Name $Alice.Zone
Cleanup-Node $Bob.Name $Bob.Zone

# PHASE 6: ANALYSIS
Log "--- Phase 5: Analysis ---"
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\chronos_v4_alice_scramble" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip" -DestinationPath "$OutDir\chronos_v4_bob_passive" -Force 2>&1 | Out-Null

$env:PYTHONIOENCODING="utf-8"
$AnalysisOutput = python python/analyze_v4_experiment.py --alice "$OutDir\chronos_v4_alice_scramble" --bob "$OutDir\chronos_v4_bob_passive" 2>&1
Add-Content -Path $LogFile -Value $AnalysisOutput

Log "=== Pipeline Complete ==="
