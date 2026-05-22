<#
.SYNOPSIS
  Shotgun pipeline for Program AH — Semantic Normal Modes
.DESCRIPTION
  Deploys exclusively to the TRC-approved Cloud TPU v6e-8 spot instances.
  Follows all TRC bible commandments (especially cleanup).
#>

$ErrorActionPreference = "Continue"
$Project = "time-emission"
$WorkDir = "e:\.git\resonance\emergent_quantum_geometries"
$LogFile = "$WorkDir\shotgun_ah.log"
$OutDir  = "$WorkDir\program_ah_results"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

function Log { param([string]$msg)
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue } catch {}
}

function Check-State { param([string]$Name, [string]$Zone)
    $s = gcloud compute tpus queued-resources describe $Name `
         --project=$Project --zone=$Zone --format="value(state.state)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($s)) { return "MISSING" }
    return $s.Trim()
}

function Cleanup-Node { param([string]$NodeId, [string]$Name, [string]$Zone)
    Log "  Cleaning up $Name ($Zone)..."
    gcloud compute tpus tpu-vm delete $NodeId --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
    Start-Sleep 5
    gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --quiet 2>&1 | Out-Null
}

$QR = "chronos-v6e-ah"
$NodeId = "chronos-v6e-ah-node"
$Zone = "us-east1-d"
$Accel = "v6e-8"
$Runtime = "v2-alpha-tpuv6e"

Log "=== Program AH Shotgun Pipeline ==="

Log "--- Phase 1: Launching TPU QR ---"
$cmd = "gcloud compute tpus queued-resources create $QR " +
       "--node-id=$NodeId --project=$Project --zone=$Zone " +
       "--accelerator-type=$Accel --runtime-version=$Runtime " +
       "--spot --quiet"
Invoke-Expression $cmd 2>&1 | Out-Null
Log "    Queued $QR in $Zone"

Log "--- Phase 2: Wait to become ACTIVE ---"
$RaceOver = $false
while (-not $RaceOver) {
    $s = Check-State $QR $Zone
    Log "  $QR ($Zone): $s"
    if ($s -eq "ACTIVE") { 
        $RaceOver = $true 
    } elseif ($s -match "FAILED|SUSPENDED") {
        Log "  $QR failed — retrying..."
        Cleanup-Node $NodeId $QR $Zone
        Invoke-Expression $cmd 2>&1 | Out-Null
    } else {
        Start-Sleep 30
    }
}

Log "--- Phase 3: Upload and launch ---"
$Scripts = Get-ChildItem "$WorkDir\program_*_tpu.py" | Select-Object -ExpandProperty FullName

foreach ($f in $Scripts) {
    Log "  Uploading $(Split-Path $f -Leaf)..."
    $out = gcloud compute tpus tpu-vm scp $f "${NodeId}:/home/cityz/" --project=$Project --zone=$Zone 2>&1
    if ($out -match "y/n") {
        $out = echo "y" | gcloud compute tpus tpu-vm scp $f "${NodeId}:/home/cityz/" --project=$Project --zone=$Zone 2>&1
    }
}

$launchCmd = "pip install -q 'jax[tpu]' flax optax && " +
             "nohup python3 -u ~/program_ah_tpu.py --sequence-lengths 16 32 --depth-sweep 2 3 " +
             "> ~/program_ah_tpu.log 2>&1 & echo PID:`$!"

$pid_out = gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command=$launchCmd 2>&1
Log "  Program AH launched: $pid_out"

Log "--- Phase 4: Wait for completion ---"
$WaitMinutes = 20
Log "  Sleeping $WaitMinutes minutes..."
Start-Sleep -Seconds ($WaitMinutes * 60)

Log "--- Phase 5: Harvesting results ---"
Log "  Checking tail log..."
$tail = gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="tail -20 ~/program_ah_tpu.log" 2>&1
Log $tail

Log "  Downloading results..."
gcloud compute tpus tpu-vm scp "${NodeId}:~/program_ah_results/program_ah_summary.json" "$OutDir\program_ah_summary.json" --project=$Project --zone=$Zone 2>&1 | Out-Null
gcloud compute tpus tpu-vm scp "${NodeId}:~/program_ah_tpu.log" "$OutDir\program_ah_tpu.log" --project=$Project --zone=$Zone 2>&1 | Out-Null

Log "--- Phase 6: Full cleanup (TRC Bible Commandments 4,5,6) ---"
Cleanup-Node $NodeId $QR $Zone

$listed = gcloud compute tpus queued-resources list --project=$Project --zone=$Zone 2>&1
Log "  Check $Zone : $listed"

Log "=== Shotgun AH Pipeline Complete ==="
Log "Results in: $OutDir"
