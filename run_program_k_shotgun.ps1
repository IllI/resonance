<#
.SYNOPSIS
  Shotgun single-winner TPU deployment for Program K (N=12 causal ordering).
.DESCRIPTION
  Races 3 spot TPUs across 3 TRC-approved zones simultaneously.
  Takes the FIRST one that goes ACTIVE, kills the other two immediately.
  Installs deps, uploads program_k_tpu.py, runs N=12 K=200, downloads results.
  Follows full delete ritual on every exit path.
#>

$ErrorActionPreference = "Continue"
$WorkDir  = "c:\Users\cityz\IllI\newer_all"
$ScriptDir = "$WorkDir\emergent_quantum_geometries"
$LogFile  = "$WorkDir\program_k_shotgun.log"
$Project  = "time-emission"
$OutDir   = "$WorkDir\emergent_quantum_geometries"

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "HH:mm:ss"
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
    Log "  Killing $Name ($Zone)..."
    $r1 = gcloud compute tpus tpu-vm delete "$Name-vm" --project=$Project --zone=$Zone --quiet 2>&1
    $r2 = gcloud compute tpus queued-resources delete $Name --project=$Project --zone=$Zone --quiet 2>&1
    Log "  Kill result: $r1 | $r2"
}

function Queue-Node {
    param($NodeObj)
    $nName    = $NodeObj.Name
    $nZone    = $NodeObj.Zone
    $nAccel   = $NodeObj.Accel
    $nRuntime = $NodeObj.Runtime
    Log "  Queuing $nName ($nAccel spot) in $nZone..."
    $res = gcloud compute tpus queued-resources create $nName `
        --node-id="$nName-vm" `
        --project=$Project --zone=$nZone `
        --accelerator-type=$nAccel `
        --runtime-version=$nRuntime `
        --spot --quiet 2>&1
    Log "  Queue result: $res"
    Start-Sleep 3
}

# Bible-approved TRC spot zones
$Zones = @(
    @{ Name="pk-node-1"; Zone="us-central2-b"; Accel="v4-8";  Runtime="tpu-vm-v4-base";    Spot=$true  },
    @{ Name="pk-node-2"; Zone="us-east1-d";    Accel="v6e-8"; Runtime="v2-alpha-tpuv6e";   Spot=$true  },
    @{ Name="pk-node-3"; Zone="europe-west4-a";Accel="v6e-8"; Runtime="v2-alpha-tpuv6e";   Spot=$true  }
)

Log "=== Program K Shotgun: N=12 Causal Ordering ==="
Log "Racing 3 spot zones: us-central2-b(v4) | us-east1-d(v6e) | europe-west4-a(v6e)"

# ---- PHASE 1: Launch all 3 concurrently -----------------------------------
Log "--- Phase 1: Shotgun queue ---"
foreach ($z in $Zones) { Queue-Node $z }

# ---- PHASE 2: Race -- first ACTIVE wins ------------------------------------
Log "--- Phase 2: Polling race (first ACTIVE wins) ---"
$Winner = $null

while ($null -eq $Winner) {
    foreach ($z in $Zones) {
        $s = Get-State $z.Name $z.Zone
        Log "  $($z.Name) ($($z.Zone)): $s"
        if ($s -eq "ACTIVE" -and $null -eq $Winner) {
            $Winner = $z
            Log "*** WINNER: $($z.Name) in $($z.Zone) ***"
        }
        if ($s -match "FAILED|SUSPENDED") {
            Log "  $($z.Name) FAILED -- re-queuing..."
            Kill-Node $z.Name $z.Zone
            Start-Sleep 2
            Queue-Node $z
        }
    }
    if ($null -eq $Winner) { Start-Sleep 25 }
}

# ---- PHASE 3: Kill losers immediately (quota protection) -------------------
Log "--- Phase 3: Killing losers ---"
foreach ($z in $Zones) {
    if ($z.Name -ne $Winner.Name) {
        Kill-Node $z.Name $z.Zone
    }
}

# ---- PHASE 4: Install deps + upload ----------------------------------------
$WinNode = "$($Winner.Name)-vm"
$WinZone = $Winner.Zone

Log "--- Phase 4: Install + upload to $WinNode ($WinZone) ---"
gcloud compute tpus tpu-vm ssh $WinNode `
    --project=$Project --zone=$WinZone `
    --command="pip install -q 'jax[tpu]' scipy && python3 -c 'import jax; print(jax.devices())'" 2>&1
Log "Dependencies installed."

gcloud compute tpus tpu-vm scp `
    "$ScriptDir\program_k_tpu.py" "$WinNode`:~/program_k_tpu.py" `
    --project=$Project --zone=$WinZone 2>&1 | Out-Null
Log "Script uploaded."

# ---- PHASE 5: Run N=12 K=200 in background ---------------------------------
Log "--- Phase 5: Launching N=12 K=200 ---"
$RunCmd = "nohup python3 ~/program_k_tpu.py " +
          "--N 12 --T-max 6.0 --n-coarse 30 --n-fine 80 --t-onset 3.0 " +
          "--K 200 --B 5 --j-focus 4 5 6 7 8 " +
          "--models XY XXZ Ising TiltedIsing " +
          "--out ~/program_k_N12_results.json " +
          "> ~/program_k_N12.log 2>&1 &"

gcloud compute tpus tpu-vm ssh $WinNode `
    --project=$Project --zone=$WinZone `
    --command=$RunCmd 2>&1 | Out-Null
Log "Job launched. Polling for completion..."

# ---- PHASE 6: Poll until done ----------------------------------------------
$Done = $false
$MaxWaitMin = 120
$Elapsed = 0
while (-not $Done -and $Elapsed -lt $MaxWaitMin) {
    Start-Sleep 60
    $Elapsed += 1
    $tail = gcloud compute tpus tpu-vm ssh $WinNode `
        --project=$Project --zone=$WinZone `
        --command="tail -5 ~/program_k_N12.log 2>/dev/null" 2>&1
    Log "[$Elapsed min] $tail"
    $running = gcloud compute tpus tpu-vm ssh $WinNode `
        --project=$Project --zone=$WinZone `
        --command="pgrep -f program_k_tpu || echo DONE" 2>&1
    if ($running -match "DONE") { $Done = $true; Log "Script finished." }
}

# ---- PHASE 7: Download results ---------------------------------------------
Log "--- Phase 7: Downloading results ---"
gcloud compute tpus tpu-vm scp `
    "$WinNode`:~/program_k_N12_results.json" `
    "$OutDir\program_k_N12_results.json" `
    --project=$Project --zone=$WinZone 2>&1
gcloud compute tpus tpu-vm scp `
    "$WinNode`:~/program_k_N12.log" `
    "$OutDir\program_k_N12_tpu.log" `
    --project=$Project --zone=$WinZone 2>&1
Log "Downloaded."

# ---- PHASE 8: THE DELETE RITUAL (mandatory) --------------------------------
Log "--- Phase 8: Delete ritual ---"
gcloud compute tpus tpu-vm delete $WinNode --project=$Project --zone=$WinZone --quiet 2>&1 | Out-Null
gcloud compute tpus queued-resources delete $Winner.Name --project=$Project --zone=$WinZone --quiet 2>&1 | Out-Null

# Verify clean
$remaining = gcloud compute tpus queued-resources list --project=$Project 2>&1
Log "Remaining QRs: $remaining"

# ---- DONE ------------------------------------------------------------------
Log "=== Program K Shotgun Complete ==="
Log "Results: $OutDir\program_k_N12_results.json"
