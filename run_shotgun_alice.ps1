<#
.SYNOPSIS
Multi-Zone Shotgun Pipeline for Alice (Robust Version)
Bob is already ACTIVE in europe-west4-a.
This script blasts Alice requests to all available US datacenters.
It automatically recovers if Alice is preempted during setup.
#>

$ErrorActionPreference = "Continue"
$WorkDir   = "c:\Users\cityz\IllI\newer_all"
$LogFile   = "$WorkDir\shotgun_pipeline.log"
$Project   = "time-emission"
$OutDir    = "$WorkDir\mismo tiempo\holaMundo"

# Bob's details (already ACTIVE)
$ZoneBob  = "europe-west4-a"

# Alice Shotgun Candidates (US Only)
$AliceCandidates = @(
    @{ Name="us-east1-d"; Accel="v6e-8"; Run="v2-alpha-tpuv6e"; IsSpot=$true },
    @{ Name="us-central1-a"; Accel="v5litepod-8"; Run="v2-alpha-tpuv5-lite"; IsSpot=$true },
    @{ Name="us-central2-b"; Accel="v4-8"; Run="tpu-vm-v4-base"; IsSpot=$false }
)

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

function Get-State {
    param([string]$Name, [string]$Z)
    $s = gcloud compute tpus queued-resources describe $Name --project=$Project --zone=$Z --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($s)) { return "MISSING" }
    return $s.Trim()
}

function Delete-Node {
    param([string]$Name, [string]$Z)
    Log "  Deleting VM $Name-node in $Z..."
    gcloud compute tpus tpu-vm delete "$Name-node" --project=$Project --zone=$Z --quiet 2>&1 | Out-Null
    Log "  Deleting QR $Name in $Z..."
    gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Z --force --quiet 2>&1 | Out-Null
}

Log "=== Starting Robust US Shotgun Pipeline for Alice ==="

while ($true) {
    # --- PHASE 1: Blast Requests ---
    Log "--- Phase 1: Blasting requests across US zones ---"
    foreach ($c in $AliceCandidates) {
        $z = $c.Name
        if ((Get-State "chronos-alice-$z" $z) -ne "MISSING") { continue }
        
        $a = $c.Accel
        $r = $c.Run
        $typeStr = if ($c.IsSpot) { "spot" } else { "on-demand" }
        Log "Queuing Alice in $z ($a $typeStr)..."
        
        if ($c.IsSpot) {
            gcloud compute tpus queued-resources create chronos-alice-$z `
                --node-id=chronos-alice-node --project=$Project --zone=$z `
                --accelerator-type=$a --runtime-version=$r --spot --quiet 2>&1 | Out-Null
        } else {
            gcloud compute tpus queued-resources create chronos-alice-$z `
                --node-id=chronos-alice-node --project=$Project --zone=$z `
                --accelerator-type=$a --runtime-version=$r --quiet 2>&1 | Out-Null
        }
    }

    # --- PHASE 2: The Race ---
    Log "--- Phase 2: Polling until one US zone wins ---"
    $WinningZone = $null
    while ($null -eq $WinningZone) {
        $BobState = Get-State "chronos-bob" $ZoneBob
        if ($BobState -ne "ACTIVE") { Log "WARNING: Bob is $BobState in EU!" }
        
        $out = "  Bob(EU):$BobState | Alice:"
        foreach ($c in $AliceCandidates) {
            $z = $c.Name
            $state = Get-State "chronos-alice-$z" $z
            $out += " ($z):$state"
            if ($state -eq "ACTIVE") { $WinningZone = $c; break }
        }
        Log $out
        if ($null -eq $WinningZone) { Start-Sleep 60 }
    }

    Log "!!! WINNER DETECTED: Alice is ACTIVE in $($WinningZone.Name) !!!"

    # --- PHASE 3: Wipe Losers ---
    Log "--- Phase 3: Cleaning up loser zones ---"
    foreach ($c in $AliceCandidates) {
        if ($c.Name -ne $WinningZone.Name) {
            gcloud compute tpus queued-resources delete "chronos-alice-$($c.Name)" --project=$Project --zone=$c.Name --quiet 2>&1 | Out-Null
        }
    }

    # --- PHASE 4: Setup ---
    Log "--- Phase 4: Synchronized Setup ---"
    $ZoneAlice = $WinningZone.Name
    $setupCmd = "pip install -q 'jax[tpu]' scipy && python3 -c 'import jax; print(jax.devices())'"

    $jobAlice = Start-Job -ScriptBlock {
        param($Proj, $Z, $Cmd, $Work)
        # Note: -o ConnectTimeout is removed for Plink compatibility
        Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Proj --zone=$Z --command="$Cmd" 2>&1
        Write-Output "y" | gcloud compute tpus tpu-vm scp "$Work\chronos_v4_tpu_run.py" "chronos-alice-node:chronos_v4_tpu_run.py" --project=$Proj --zone=$Z 2>&1
    } -ArgumentList $Project, $ZoneAlice, $setupCmd, $WorkDir

    $jobBob = Start-Job -ScriptBlock {
        param($Proj, $Z, $Cmd, $Work)
        Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-bob-node --project=$Proj --zone=$Z --command="$Cmd" 2>&1
        Write-Output "y" | gcloud compute tpus tpu-vm scp "$Work\chronos_v4_tpu_run.py" "chronos-bob-node:chronos_v4_tpu_run.py" --project=$Proj --zone=$Z 2>&1
    } -ArgumentList $Project, $ZoneBob, $setupCmd, $WorkDir

    Log "  Waiting for setup (max 10m)..."
    Wait-Job $jobAlice, $jobBob -Timeout 600 | Out-Null
    $resA = Receive-Job $jobAlice
    $resB = Receive-Job $jobBob
    
    Log "  Alice ($ZoneAlice) Result: $resA"
    Log "  Bob ($ZoneBob) Result: $resB"

    if ($resA -match "PREEMPTED" -or $resA -match "failed" -or $resB -match "failed") {
        Log "ERROR: Setup failed or Alice was preempted. Re-starting race..."
        Delete-Node "chronos-alice-$ZoneAlice" $ZoneAlice
        continue
    }

    # --- PHASE 5: Launch Capture ---
    Log "--- Phase 5: Launching synchronized capture ---"
    $aliceCmd = 'nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &'
    $bobCmd   = 'nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive   --duration 7200 > ~/bob_run.log   2>&1 &'
    
    $launchA = Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$ZoneAlice --command=$aliceCmd 2>&1
    $launchB = Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-bob-node   --project=$Project --zone=$ZoneBob   --command=$bobCmd   2>&1
    
    if ($launchA -match "failed" -or $launchB -match "failed") {
        Log "ERROR: Launch failed. Re-starting race..."
        Delete-Node "chronos-alice-$ZoneAlice" $ZoneAlice
        continue
    }

    Log "Capture successfully fired on both continents. Entering 2h10m sleep..."
    break
}

# --- PHASE 6: Sleep & Harvest ---
Start-Sleep -Seconds 7800

Log "--- Phase 6: Downloading results ---"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Output "y" | gcloud compute tpus tpu-vm scp "chronos-alice-node:chronos_v4_alice_scramble.zip" `
    "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$ZoneAlice 2>&1 | Out-Null
Write-Output "y" | gcloud compute tpus tpu-vm scp "chronos-bob-node:chronos_v4_bob_passive.zip" `
    "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$ZoneBob 2>&1 | Out-Null

# --- PHASE 7: Teardown ---
Log "--- Phase 7: Cleanup ---"
Delete-Node "chronos-alice-$ZoneAlice" $ZoneAlice
Delete-Node "chronos-bob" $ZoneBob

# --- PHASE 8: Analysis ---
Log "--- Phase 8: Analysis ---"
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\alice" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip"   -DestinationPath "$OutDir\bob"   -Force 2>&1 | Out-Null
python python/analyze_v4_experiment.py --alice "$OutDir\alice" --bob "$OutDir\bob" | Add-Content -Path $LogFile
Log "=== Pipeline Complete ==="
