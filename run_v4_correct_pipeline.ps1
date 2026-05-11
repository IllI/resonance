<#
.SYNOPSIS
Dual-Continent TRC Pipeline per exact setup guide specifications.
Alice = v4 on-demand in us-central2-b 
Bob   = v6e spot in europe-west4-a
#>

$ErrorActionPreference = "Continue"
$WorkDir   = "c:\Users\cityz\IllI\newer_all"
$LogFile   = "$WorkDir\pipeline_correct.log"
$Project   = "time-emission"
$OutDir    = "$WorkDir\mismo tiempo\holaMundo"

$ZoneAlice = "us-east1-d"
$AccelAlice = "v6e-8"
$RunAlice = "v2-alpha-tpuv6e"

$ZoneBob = "europe-west4-a"
$AccelBob = "v6e-8"
$RunBob = "v2-alpha-tpuv6e"

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

# --- PHASE 1: Queue Alice (US) and Bob (EU) ---
Log "=== Starting Dual-Continent TRC Pipeline ==="
Log "--- Phase 1: Queue Alice (US-East, spot) and Bob (EU, spot) ---"

Log "Queuing Alice - $AccelAlice spot ($ZoneAlice)..."
$r = gcloud compute tpus queued-resources create chronos-alice `
    --node-id=chronos-alice-node `
    --project=$Project --zone=$ZoneAlice `
    --accelerator-type=$AccelAlice `
    --runtime-version=$RunAlice `
    --spot `
    --quiet 2>&1
Log "  Alice result: $r"

Log "Queuing Bob - $AccelBob spot ($ZoneBob)..."
$r = gcloud compute tpus queued-resources create chronos-bob `
    --node-id=chronos-bob-node `
    --project=$Project --zone=$ZoneBob `
    --accelerator-type=$AccelBob `
    --runtime-version=$RunBob `
    --spot `
    --quiet 2>&1
Log "  Bob result: $r"

# --- PHASE 2: Wait for both ACTIVE ---
Log "--- Phase 2: Polling until both ACTIVE ---"
$BothActive = $false
while (-not $BothActive) {
    $aliceState = Get-State "chronos-alice" $ZoneAlice
    $bobState   = Get-State "chronos-bob" $ZoneBob
    Log "  chronos-alice (US): $aliceState  |  chronos-bob (EU): $bobState"

    if ($aliceState -match "FAILED|SUSPENDED|MISSING") {
        Log "  Alice was preempted ($aliceState). Re-queuing in US-East..."
        Delete-Node "chronos-alice" $ZoneAlice
        Start-Sleep 5
        $r = gcloud compute tpus queued-resources create chronos-alice `
            --node-id=chronos-alice-node `
            --project=$Project --zone=$ZoneAlice `
            --accelerator-type=$AccelAlice `
            --runtime-version=$RunAlice `
            --spot `
            --quiet 2>&1
        Log "  Alice re-queue result: $r"
    }

    if ($bobState -match "FAILED|SUSPENDED|MISSING") {
        Log "  Bob was preempted ($bobState). Re-queuing in EU..."
        Delete-Node "chronos-bob" $ZoneBob
        Start-Sleep 5
        $r = gcloud compute tpus queued-resources create chronos-bob `
            --node-id=chronos-bob-node `
            --project=$Project --zone=$ZoneBob `
            --accelerator-type=$AccelBob `
            --runtime-version=$RunBob `
            --spot `
            --quiet 2>&1
        Log "  Bob re-queue result: $r"
    }

    if ($aliceState -eq "ACTIVE" -and $bobState -eq "ACTIVE") {
        Log "BOTH ACTIVE - starting experiment!"
        $BothActive = $true
    } else {
        Start-Sleep 60
    }
}

# --- PHASE 3 & 4: Setup (Concurrent) ---
Log "--- Phase 3 & 4: Installing dependencies & Uploading scripts ---"
$setupCmd = "pip install -q 'jax[tpu]' scipy && python3 -c 'import jax; print(jax.devices())'"

# Launch setup on Alice and Bob in parallel
$jobAlice = Start-Job -ScriptBlock {
    param($Proj, $Z, $Cmd, $Work)
    Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Proj --zone=$Z --command="$Cmd" -- -o ConnectTimeout=30 2>&1
    Write-Output "y" | gcloud compute tpus tpu-vm scp "$Work\chronos_v4_tpu_run.py" "chronos-alice-node:~/chronos_v4_tpu_run.py" --project=$Proj --zone=$Z 2>&1
} -ArgumentList $Project, $ZoneAlice, $setupCmd, $WorkDir

$jobBob = Start-Job -ScriptBlock {
    param($Proj, $Z, $Cmd, $Work)
    Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-bob-node --project=$Proj --zone=$Z --command="$Cmd" -- -o ConnectTimeout=30 2>&1
    Write-Output "y" | gcloud compute tpus tpu-vm scp "$Work\chronos_v4_tpu_run.py" "chronos-bob-node:~/chronos_v4_tpu_run.py" --project=$Proj --zone=$Z 2>&1
} -ArgumentList $Project, $ZoneBob, $setupCmd, $WorkDir

Log "  Waiting for setup to complete (max 15m)..."
Wait-Job $jobAlice, $jobBob -Timeout 900 | Out-Null
$resA = Receive-Job $jobAlice
$resB = Receive-Job $jobBob
Log "  Alice setup result: $resA"
Log "  Bob setup result: $resB"

# --- PHASE 5: Fire 2-hour capture ---
Log "--- Phase 5: Starting synchronized 2-hour capture ---"
$aliceCmd = 'nohup python3 ~/chronos_v4_tpu_run.py --role Alice_Scramble --duration 7200 > ~/alice_run.log 2>&1 &'
$bobCmd   = 'nohup python3 ~/chronos_v4_tpu_run.py --role Bob_Passive   --duration 7200 > ~/bob_run.log   2>&1 &'
Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-alice-node --project=$Project --zone=$ZoneAlice --command=$aliceCmd 2>&1 | Out-Null
Write-Output "y" | gcloud compute tpus tpu-vm ssh chronos-bob-node   --project=$Project --zone=$ZoneBob   --command=$bobCmd   2>&1 | Out-Null

Log "Capture launched. Sleeping 2h10m..."
Start-Sleep -Seconds 7800

# --- PHASE 6: Download artifacts ---
Log "--- Phase 6: Downloading results ---"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Output "y" | gcloud compute tpus tpu-vm scp "chronos-alice-node:~/chronos_v4_alice_scramble.zip" `
    "$OutDir\chronos_v4_alice_scramble.zip" --project=$Project --zone=$ZoneAlice 2>&1 | Out-Null
Write-Output "y" | gcloud compute tpus tpu-vm scp "chronos-bob-node:~/chronos_v4_bob_passive.zip" `
    "$OutDir\chronos_v4_bob_passive.zip" --project=$Project --zone=$ZoneBob 2>&1 | Out-Null

# --- PHASE 7: Delete everything - billing safety ---
Log "--- Phase 7: Billing cleanup (CRITICAL) ---"
Delete-Node "chronos-alice" $ZoneAlice
Delete-Node "chronos-bob" $ZoneBob

# --- PHASE 8: Analysis ---
Log "--- Phase 8: Running analysis ---"
Expand-Archive -Path "$OutDir\chronos_v4_alice_scramble.zip" -DestinationPath "$OutDir\alice" -Force 2>&1 | Out-Null
Expand-Archive -Path "$OutDir\chronos_v4_bob_passive.zip"   -DestinationPath "$OutDir\bob"   -Force 2>&1 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$out = python python/analyze_v4_experiment.py --alice "$OutDir\alice" --bob "$OutDir\bob" 2>&1
Add-Content -Path $LogFile -Value $out
Log "=== Pipeline Complete ==="
