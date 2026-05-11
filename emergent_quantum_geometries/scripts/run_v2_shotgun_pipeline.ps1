<#
.SYNOPSIS
  Shotgun pipeline for OAT Teleport v2 (Schmidt rotation + tensor network RDM)
.DESCRIPTION
  Races 3 TRC-approved nodes. chronos-v5 is already provisioning in us-east1-d.
  Adds Alice (v4 on-demand us-central2-b) and Carol (v6e spot europe-west4-a).
  First 2+ ACTIVE nodes win and get split N-ranges to run in parallel.
  Full cleanup at end. Follows TRC bible commandments exactly.

  TRC-approved zones used:
    chronos-v5   : us-east1-d       v6e-8 spot   (already in queue)
    chronos-v2-alice : us-central2-b    v4-8 on-demand
    chronos-v2-carol : europe-west4-a   v6e-8 spot

  N-range split (so all nodes run simultaneously):
    Alice : N=[2,4,6,8]         chi_t_steps=32 K=5000
    Bob   : N=[10,12,14,16]     chi_t_steps=32 K=5000
    Carol : N=[2,4,6,8,10,12]   chi_t_steps=64 K=10000  (high-quality duplicate)
#>

$ErrorActionPreference = "Continue"
$Project = "time-emission"
$WorkDir = "c:\Users\cityz\IllI\newer_all\emergent_quantum_geometries"
$LogFile = "$WorkDir\shotgun_v2.log"
$OutDir  = "$WorkDir\shotgun_v2_results"
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

function SCP-Upload { param([string]$NodeId, [string]$Zone, [string[]]$Files)
    Log "  SCP upload to $NodeId ($Zone)..."
    foreach ($f in $Files) {
        $out = gcloud compute tpus tpu-vm scp $f "${NodeId}:/home/cityz/" `
               --project=$Project --zone=$Zone 2>&1
        if ($out -match "y/n") {
            $out = echo "y" | gcloud compute tpus tpu-vm scp $f "${NodeId}:/home/cityz/" `
                   --project=$Project --zone=$Zone 2>&1
        }
    }
}

# ── Define nodes ──────────────────────────────────────────────────────────────
$Nodes = @(
    @{ QR="chronos-v5";      NodeId="chronos-v5-node";    Zone="us-east1-d";    
       Accel="v6e-8"; Runtime="v2-alpha-tpuv6e"; Spot=$true; PreExisting=$true;
       Args="--N 10 12 14 16 --chi_t_steps 32 --K 5000";
       ResultFile="oat_teleport_v2_results_bob.json" },

    @{ QR="chronos-v2-alice"; NodeId="chronos-v2-alice-node"; Zone="us-central2-b";
       Accel="v4-8"; Runtime="tpu-vm-v4-base"; Spot=$false; PreExisting=$false;
       Args="--N 2 4 6 8 --chi_t_steps 32 --K 5000";
       ResultFile="oat_teleport_v2_results_alice.json" },

    @{ QR="chronos-v2-carol"; NodeId="chronos-v2-carol-node"; Zone="europe-west4-a";
       Accel="v6e-8"; Runtime="v2-alpha-tpuv6e"; Spot=$true; PreExisting=$false;
       Args="--N 2 4 6 8 10 12 --chi_t_steps 64 --K 10000";
       ResultFile="oat_teleport_v2_results_carol.json" }
)

Log "=== OAT Teleport v2 Shotgun Pipeline ==="
Log "N-range split: Alice=[2,4,6,8]  Bob=[10,12,14,16]  Carol=[2..12 high-K]"

# ── Phase 1: Launch non-preexisting nodes ─────────────────────────────────────
Log "--- Phase 1: Launching new nodes ---"
foreach ($n in $Nodes) {
    if ($n.PreExisting) {
        Log "  chronos-v5 already queued in $($n.Zone) — skipping create"
        continue
    }
    Log "  Creating $($n.QR) ($($n.Accel)) in $($n.Zone)..."
    $spotFlag = if ($n.Spot) { "--spot" } else { "" }
    $cmd = "gcloud compute tpus queued-resources create $($n.QR) " +
           "--node-id=$($n.NodeId) --project=$Project --zone=$($n.Zone) " +
           "--accelerator-type=$($n.Accel) --runtime-version=$($n.Runtime) " +
           "$spotFlag --quiet"
    Invoke-Expression $cmd 2>&1 | Out-Null
    Log "    Queued."
}

# ── Phase 2: Race to ACTIVE ───────────────────────────────────────────────────
Log "--- Phase 2: Racing to ACTIVE (need 2+) ---"
$Winners = @(); $RaceOver = $false

while (-not $RaceOver) {
    $active = @()
    foreach ($n in $Nodes) {
        $s = Check-State $n.QR $n.Zone
        Log "  $($n.QR) ($($n.Zone)): $s"
        if ($s -eq "ACTIVE") { $active += $n }
        elseif ($s -match "FAILED|SUSPENDED" -and -not $n.PreExisting) {
            Log "  $($n.QR) failed — retrying..."
            gcloud compute tpus queued-resources delete $n.QR `
                --project=$Project --zone=$n.Zone --quiet 2>&1 | Out-Null
            Start-Sleep 10
            $spotFlag = if ($n.Spot) { "--spot" } else { "" }
            $cmd = "gcloud compute tpus queued-resources create $($n.QR) " +
                   "--node-id=$($n.NodeId) --project=$Project --zone=$($n.Zone) " +
                   "--accelerator-type=$($n.Accel) --runtime-version=$($n.Runtime) " +
                   "$spotFlag --quiet"
            Invoke-Expression $cmd 2>&1 | Out-Null
        }
    }
    if ($active.Count -ge 2) {
        $Winners = $active; $RaceOver = $true
        Log "RACE DONE: $($Winners.Count) nodes ACTIVE"
    } else {
        Log "  $($active.Count)/2 active — waiting 30s..."
        Start-Sleep 30
    }
}

# Kill any non-winner nodes to protect quota
foreach ($n in $Nodes) {
    $isWinner = $Winners | Where-Object { $_.QR -eq $n.QR }
    if (-not $isWinner) {
        $s = Check-State $n.QR $n.Zone
        if ($s -match "WAITING|PROVISIONING|ACCEPTED") {
            Log "  Killing non-winner $($n.QR)..."
            gcloud compute tpus queued-resources delete $n.QR `
                --project=$Project --zone=$n.Zone --quiet 2>&1 | Out-Null
        }
    }
}

# ── Phase 3: Upload and launch (ALL winners in parallel) ─────────────────────
Log "--- Phase 3: Upload and launch on $($Winners.Count) nodes ---"

$Scripts = @(
    "$WorkDir\oat_teleport_v2_tpu.py",
    "$WorkDir\jila_oat_exact_tpu.py"
)

$Jobs = @()
foreach ($n in $Winners) {
    Log "  Setting up $($n.QR)..."
    SCP-Upload $n.NodeId $n.Zone $Scripts

    # Install deps + launch experiment
    $launchCmd = "pip install -q 'jax[tpu]' flax optax && " +
                 "nohup python3 -u ~/oat_teleport_v2_tpu.py $($n.Args) " +
                 "> ~/$($n.ResultFile -replace '.json','.log') 2>&1 & echo PID:`$!"

    $pid_out = gcloud compute tpus tpu-vm ssh $n.NodeId `
        --project=$Project --zone=$n.Zone `
        --command=$launchCmd 2>&1

    Log "  $($n.QR) launched: $pid_out"
}

# ── Phase 4: Wait for completion ─────────────────────────────────────────────
Log "--- Phase 4: Waiting ~20 min for all nodes to complete ---"
$WaitMinutes = 22
Log "  Sleeping $WaitMinutes minutes..."
Start-Sleep -Seconds ($WaitMinutes * 60)

# ── Phase 5: Harvest results ─────────────────────────────────────────────────
Log "--- Phase 5: Harvesting results ---"
foreach ($n in $Winners) {
    $logName = $n.ResultFile -replace '.json','.log'
    Log "  Checking $($n.QR) tail log..."
    $tail = gcloud compute tpus tpu-vm ssh $n.NodeId `
        --project=$Project --zone=$n.Zone `
        --command="tail -20 ~/$logName" 2>&1
    Log $tail

    Log "  Downloading $($n.ResultFile) from $($n.QR)..."
    gcloud compute tpus tpu-vm scp "${$n.NodeId}:~/oat_teleport_v2_results.json" `
        "$OutDir\$($n.ResultFile)" `
        --project=$Project --zone=$n.Zone 2>&1 | Out-Null
}

# ── Phase 6: Cleanup ALL — bible commandments 4,5,6 ─────────────────────────
Log "--- Phase 6: Full cleanup (TRC Bible Commandments 4,5,6) ---"
foreach ($n in $Winners) {
    Cleanup-Node $n.NodeId $n.QR $n.Zone
}

Log "--- Verifying ALL zones empty ---"
$Zones = @("us-central2-b","us-east1-d","europe-west4-a","us-central1-a","europe-west4-b")
foreach ($z in $Zones) {
    $listed = gcloud compute tpus queued-resources list --project=$Project --zone=$z 2>&1
    Log "  $z : $listed"
}

Log "=== Shotgun v2 Pipeline Complete ==="
Log "Results in: $OutDir"
