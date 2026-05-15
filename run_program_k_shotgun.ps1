<#
.SYNOPSIS
  Improved shotgun for Program K N=14 -- runs full analysis ON TPU before download.
.DESCRIPTION
  Key improvements over v1:
    1. SSH-verify winner before declaring -- catches immediate preemptions
    2. Retry loop: if winner preempted during install, re-race
    3. Full analysis runs on TPU VM (not local CPU)
    4. Downloads only final zip + log
    5. Full Bible delete ritual on all exit paths
    6. Uses `yes |` for any lingering confirmation prompts
#>

$ErrorActionPreference = "Continue"
$WorkDir   = "c:\Users\cityz\IllI\newer_all"
$ScriptDir = "$WorkDir\emergent_quantum_geometries"
$LogFile   = "$WorkDir\program_k_N14_shotgun.log"
$Project   = "time-emission"

function Log {
    param([string]$msg)
    $ts   = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

function Get-State {
    param([string]$Name, [string]$Zone)
    $s = gcloud compute tpus queued-resources describe $Name `
        --project=$Project --zone=$Zone `
        --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($s)) { return "MISSING" }
    return $s.Trim()
}

function Kill-Node {
    param([string]$Name, [string]$Zone)
    Log "  Kill $Name ($Zone)..."
    $r1 = gcloud compute tpus tpu-vm delete "$Name-vm" --project=$Project --zone=$Zone --quiet 2>&1
    $r2 = gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --quiet 2>&1
}

function Queue-Node {
    param($n)
    $nName=$n.Name; $nZone=$n.Zone; $nAccel=$n.Accel; $nRuntime=$n.Runtime
    $nSpot=$n.SpotFlag  # "--best-effort" for v6e spot, "" for v4 on-demand
    Log "  Queuing $nName ($nAccel) in $nZone (spot flag: '$nSpot')..."
    if ($nSpot) {
        $res = gcloud compute tpus queued-resources create $nName `
            --node-id="$nName-vm" --project=$Project --zone=$nZone `
            --accelerator-type=$nAccel --runtime-version=$nRuntime `
            $nSpot --quiet 2>&1
    } else {
        $res = gcloud compute tpus queued-resources create $nName `
            --node-id="$nName-vm" --project=$Project --zone=$nZone `
            --accelerator-type=$nAccel --runtime-version=$nRuntime `
            --quiet 2>&1
    }
    Log "  -> $res"
    Start-Sleep 3
}

function Test-SSH {
    # Returns true if SSH works on this node right now
    param([string]$NodeVm, [string]$Zone)
    $r = gcloud compute tpus tpu-vm ssh $NodeVm `
        --project=$Project --zone=$Zone `
        --command="echo SSH_OK" 2>&1
    return ($r -match "SSH_OK")
}

# TRC-approved spot zones (Bible-compliant flags)
# v4-8 us-central2-b: on-demand, NO spot flag
# v6e-8 us-east1-d:   spot, --best-effort
# v6e-8 europe-west4-a: spot, --best-effort
$Zones = @(
    @{ Name="kn14-1"; Zone="us-central2-b"; Accel="v4-8";  Runtime="tpu-vm-v4-base";  SpotFlag=""              },
    @{ Name="kn14-2"; Zone="us-east1-d";    Accel="v6e-8"; Runtime="v2-alpha-tpuv6e"; SpotFlag="--best-effort" },
    @{ Name="kn14-3"; Zone="europe-west4-a";Accel="v6e-8"; Runtime="v2-alpha-tpuv6e"; SpotFlag="--best-effort" }
)

# Program K v3 N=14 run command
# THREE JOBS: (1) causal hierarchy t_MI<t_dF~t_rec at large j
#             (2) disorder phase boundary W=1->8
#             (3) Clifford scramble stress test
$RemoteCmd = (
    "pip install -q 'jax[tpu]' scipy 2>/dev/null; " +
    "python3 ~/program_k_tpu.py " +
    "--N 14 --T-max 8.0 --n-coarse 40 --n-fine 100 --t-onset 3.5 " +
    "--K 200 --B 5 --j-focus 4 5 6 7 8 9 10 11 " +
    "--models XXZ Ising TiltedIsing MBL_XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 " +
    "--basis-scramble 5 --clifford-scramble 5 " +
    "--out ~/program_k_N14_results.json " +
    "> ~/program_k_N14.log 2>&1"
)

Log "=== Program K N=14 Shotgun (TPU-side execution) ==="
$RoundNo = 0

:RaceLoop while ($true) {
    $RoundNo++
    Log "--- Round $($RoundNo): Queuing 3 zones ---"
    foreach ($z in $Zones) { Queue-Node $z }

    # Race loop
    $Winner = $null
    while ($null -eq $Winner) {
        foreach ($z in $Zones) {
            $s = Get-State $z.Name $z.Zone
            Log "  $($z.Name): $s"
            if ($s -eq "ACTIVE" -and $null -eq $Winner) {
                Log "  SSH-verifying $($z.Name)..."
                if (Test-SSH "$($z.Name)-vm" $z.Zone) {
                    $Winner = $z
                    Log "*** WINNER (SSH verified): $($z.Name) in $($z.Zone) ***"
                } else {
                    Log "  SSH failed (likely PREEMPTED) -- skipping $($z.Name)"
                }
            }
            if ($s -match "PREEMPTED|FAILED|SUSPENDED") {
                Log "  $($z.Name) $s -- re-queuing..."
                Kill-Node $z.Name $z.Zone
                Start-Sleep 5
                Queue-Node $z
            }
        }
        if ($null -eq $Winner) { Start-Sleep 25 }
    }

    # Kill losers
    Log "--- Killing losers ---"
    foreach ($z in $Zones) {
        if ($z.Name -ne $Winner.Name) { Kill-Node $z.Name $z.Zone }
    }

    $WinVm   = "$($Winner.Name)-vm"
    $WinZone = $Winner.Zone

    # Upload script
    Log "--- Uploading script to $WinVm ---"
    $up = gcloud compute tpus tpu-vm scp `
        "$ScriptDir\program_k_tpu.py" "$WinVm`:~/program_k_tpu.py" `
        --project=$Project --zone=$WinZone 2>&1
    Log "  Upload: $up"

    # Verify upload succeeded; if not, winner may have gone PREEMPTED during upload
    $checkState = Get-State $Winner.Name $WinZone
    if ($checkState -ne "ACTIVE") {
        Log "  Winner went $checkState during upload -- restarting race..."
        Kill-Node $Winner.Name $WinZone
        continue RaceLoop
    }

    # Launch analysis on TPU (blocking, survives SSH timeout via nohup)
    Log "--- Launching N=14 analysis on $WinVm ($WinZone) ---"
    $launch = gcloud compute tpus tpu-vm ssh $WinVm `
        --project=$Project --zone=$WinZone `
        --command="nohup bash -c '$RemoteCmd' > /dev/null 2>&1 &" 2>&1
    Log "  Launch: $launch"

    # Poll until done (check every 60s, max 180 min)
    $Done = $false; $Elapsed = 0; $MaxMin = 180
    while (-not $Done -and $Elapsed -lt $MaxMin) {
        Start-Sleep 60; $Elapsed++
        $nodeState = Get-State $Winner.Name $WinZone
        if ($nodeState -ne "ACTIVE") {
            Log "  Node $nodeState after $Elapsed min -- aborting this round"
            Kill-Node $Winner.Name $WinZone
            continue RaceLoop
        }
        $tail = gcloud compute tpus tpu-vm ssh $WinVm `
            --project=$Project --zone=$WinZone `
            --command="tail -3 ~/program_k_N14.log 2>/dev/null; pgrep -f program_k_tpu || echo PROC_DONE" 2>&1
        Log "[$Elapsed min] $tail"
        if ($tail -match "PROC_DONE") { $Done = $true; Log "Analysis done." }
    }

    if (-not $Done) {
        Log "Timed out after $MaxMin min -- downloading whatever exists."
    }

    # Download results + log
    Log "--- Downloading results from $WinVm ---"
    gcloud compute tpus tpu-vm scp `
        "$WinVm`:~/program_k_N14_results.json" `
        "$ScriptDir\program_k_N14_results.json" `
        --project=$Project --zone=$WinZone 2>&1 | Out-Null
    gcloud compute tpus tpu-vm scp `
        "$WinVm`:~/program_k_N14_results.zip" `
        "$ScriptDir\program_k_N14_results.zip" `
        --project=$Project --zone=$WinZone 2>&1 | Out-Null
    gcloud compute tpus tpu-vm scp `
        "$WinVm`:~/program_k_N14.log" `
        "$ScriptDir\program_k_N14_tpu.log" `
        --project=$Project --zone=$WinZone 2>&1 | Out-Null
    Log "Download complete."
    break RaceLoop
}

# DELETE RITUAL (mandatory per Bible)
Log "=== DELETE RITUAL ==="
$allZones = @("us-central2-b","us-east1-d","europe-west4-a","us-central1-a","europe-west4-b")
foreach ($z in $allZones) {
    foreach ($n in @("kn14-1","kn14-2","kn14-3")) {
        gcloud compute tpus tpu-vm delete "$n-vm" --project=$Project --zone=$z --quiet 2>&1 | Out-Null
        gcloud compute tpus queued-resources delete $n --project=$Project --zone=$z --quiet 2>&1 | Out-Null
    }
}

# VERIFY CLEAN
Log "=== VERIFICATION ==="
foreach ($z in $allZones) {
    $qr = gcloud compute tpus queued-resources list --project=$Project --zone=$z 2>&1
    $vm = gcloud compute tpus tpu-vm list --project=$Project --zone=$z 2>&1
    Log "  $z  QR:$qr  VM:$vm"
}

Log "=== Done. Results in $ScriptDir ==="
