<#
.SYNOPSIS
  run_program_l_parallel.ps1 -- Program L resilient TPU launcher.
  Queues all 6 TRC zones. Runs experiment on first ACTIVE winner.
  Keeps other zones queued as standbys. On preemption, falls back
  to next ACTIVE zone without re-queueing. Bible-compliant.
#>
param(
    [string]$N             = "10",
    [string]$Models        = "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing",
    [string]$Controllers   = "static haware agnostic",
    [string]$TMax          = "6.0",
    [string]$NSteps        = "30",
    [string]$B             = "3",
    [string]$Seed          = "42",
    [int]   $QueueMaxMin   = 35,
    [int]   $MaxMin        = 240,
    [string]$OutDir        = "program_l_results",
    [switch]$StoreObs,
    [string]$Adversarial   = "none",
    # Program O extensions
    [string]$Script        = "program_l_tpu.py",   # or program_o_tpu.py
    [string]$ProbeFamilies = "",                    # e.g. "neel x_basis equatorial"
    [switch]$RunChirp                               # Phase 1 chirp probe
)

$Project     = "time-emission"
$SrcDir      = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir   = "/home/cityz/program_l"
$LogFile     = "prog_l_run.log"

# TRC compliance: always clean up QRs on any exit (Ctrl+C, error, normal end)
trap {
    Write-Host "[TRAP] Unexpected exit -- running cleanup to free TRC quota..."
    if ($null -ne $Candidates) { Delete-All $Candidates }
    break
}
$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) { $env:Path = "$CloudSdkBin;$env:Path" }
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# Zone priority: v6e first (Python 3.10, jaxlib==0.4.13 compatible).
# v4 nodes use tpu-vm-v4-base runtime with Python 3.8 -- jaxlib 0.4.13 does NOT
# support Python 3.8, causing install failure. Keep v4 at the end as last-resort.
$Candidates = @(
    @{ QRName="prog-l-v6e1"; NodeId="prog-l-v6e1-node"; Zone="europe-west4-a"; Type="v6e-8"; Runtime="v2-alpha-tpuv6e"; Flag="--spot" },
    @{ QRName="prog-l-v6e2"; NodeId="prog-l-v6e2-node"; Zone="us-east1-d";     Type="v6e-8"; Runtime="v2-alpha-tpuv6e"; Flag="--spot" },
    @{ QRName="prog-l-v5eb"; NodeId="prog-l-v5eb-node"; Zone="europe-west4-b"; Type="v5e-8"; Runtime="v2-alpha-tpuv5e"; Flag="--spot" },
    @{ QRName="prog-l-v5ec"; NodeId="prog-l-v5ec-node"; Zone="us-central1-a";  Type="v5e-8"; Runtime="v2-alpha-tpuv5e"; Flag="--spot" },
    @{ QRName="prog-l-v4od"; NodeId="prog-l-v4od-node"; Zone="us-central2-b";  Type="v4-8";  Runtime="tpu-vm-v4-base";  Flag="" },
    @{ QRName="prog-l-v4s";  NodeId="prog-l-v4s-node";  Zone="us-central2-b";  Type="v4-8";  Runtime="tpu-vm-v4-base";  Flag="--spot" }
)
$AllZones = @("us-central2-b","europe-west4-a","us-east1-d","us-central1-a","europe-west4-b")

function Get-QRState([string]$Name, [string]$Zone) {
    $j = gcloud compute tpus queued-resources describe $Name --project=$Project --zone=$Zone --format=json 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($j)) { return "MISSING" }
    try {
        $d = $j | ConvertFrom-Json
        if ($d.state -and $d.state.state) { return [string]$d.state.state }
        if ($d.state) { return [string]$d.state }
    } catch {}
    return "UNKNOWN"
}

function Delete-All([array]$cands) {
    Write-Host "[RITUAL] Running Bible delete ritual..."
    foreach ($c in $cands) {
        gcloud compute tpus tpu-vm delete $c.NodeId --project=$Project --zone=$($c.Zone) --quiet 2>$null
        gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
    }
    Start-Sleep -Seconds 10
    Write-Host "[RITUAL] Verifying all resources deleted..."
    foreach ($z in $AllZones) {
        $qrs = gcloud compute tpus queued-resources list --project=$Project --zone=$z --format="value(name)" 2>$null
        if ($qrs) { Write-Host "[WARN] Zone ${z}: still has: $qrs" }
        else       { Write-Host "[OK]   Zone ${z}: clean." }
    }
    Write-Host "[RITUAL] Delete ritual complete."
}

function Launch-Experiment([hashtable]$c) {
    # Accept host key + wait for SSH
    Write-Host "[SSH] Connecting to $($c.Zone) ($($c.NodeId))..."
    $sshOk = $false
    $sshEnd = (Get-Date).AddMinutes(5)
    while (-not $sshOk -and (Get-Date) -lt $sshEnd) {
        $st = Get-QRState $c.QRName $c.Zone
        if ($st -ne "ACTIVE") { Write-Host "[ERROR] Node became $st before SSH."; return $false }
        "y" | gcloud compute tpus tpu-vm ssh $c.NodeId --project=$Project --zone=$($c.Zone) --command="echo SSH_READY" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $sshOk = $true } else { Start-Sleep -Seconds 10 }
    }
    if (-not $sshOk) { Write-Host "[ERROR] SSH timeout."; return $false }

    # Version gate: v4 TPUs run Python 3.8 which has no compatible TPU jaxlib wheel.
    # jaxlib==0.4.13 is not available for v4 TPUs at all (neither PyPI nor TPU index).
    # Skip immediately -- do NOT attempt install which causes infinite retry loop.
    Write-Host "[SETUP] Checking Python version..."
    $pyVer = "y" | gcloud compute tpus tpu-vm ssh $c.NodeId --project=$Project --zone=$($c.Zone) `
                --command="python3 -c 'import sys; print(sys.version_info.minor)'" 2>$null
    $pyMinor = if ($pyVer -is [array]) { [int]($pyVer[-1].Trim()) } else { [int]($pyVer.Trim()) }
    if ($pyMinor -lt 10) {
        Write-Host "[SKIP] Python 3.$pyMinor on $($c.Zone) -- no compatible TPU jaxlib wheel. Waiting for v6e zone."
        return $false
    }
    Write-Host "[OK] Python 3.$pyMinor -- v6e compatible."

    # Install deps (Python 3.10+, v6e only)
    Write-Host "[SETUP] Installing JAX/TPU deps..."
    $depCmd = "mkdir -p $RemoteDir && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax; print(jax.default_backend()); print(jax.devices())' && echo DEPS_OK"
    "y" | gcloud compute tpus tpu-vm ssh $c.NodeId --project=$Project --zone=$($c.Zone) --command=$depCmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Deps failed."; return $false }

    # Upload files (always include program_l_tpu.py as dependency; add script if different)
    Write-Host "[SCP] Uploading..."
    $uploadFiles = @("program_l_tpu.py", "analyze_program_l.py")
    if ($Script -ne "program_l_tpu.py" -and $Script -ne "") { $uploadFiles += $Script }
    foreach ($f in $uploadFiles) {
        $local  = Join-Path $SrcDir $f
        if (-not (Test-Path $local)) { Write-Host "[WARN] $f not found locally, skipping."; continue }
        $remote = $c.NodeId + ':' + $RemoteDir + '/' + $f
        gcloud compute tpus tpu-vm scp $local $remote --project=$Project --zone=$($c.Zone)
        if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] SCP failed: $f"; return $false }
    }
    Write-Host "[SCP] Upload complete."
    return $true
}

# ── 0. Project check ──────────────────────────────────────────────────────────
$cur = (gcloud config list project --format="value(core.project)" 2>$null).Trim()
if ($cur -ne $Project) { gcloud config set project $Project }
Write-Host "[SETUP] Project: $Project"

# ── 1. Pre-flight delete ritual ───────────────────────────────────────────────
Delete-All $Candidates

# ── 2. Queue all zones ────────────────────────────────────────────────────────
Write-Host "[SHOTGUN] Queuing all $($Candidates.Count) zones..."
foreach ($c in $Candidates) {
    $qa = @("compute","tpus","queued-resources","create",$c.QRName,
            "--node-id=$($c.NodeId)","--project=$Project","--zone=$($c.Zone)",
            "--accelerator-type=$($c.Type)","--runtime-version=$($c.Runtime)","--quiet")
    if ($c.Flag) { $qa += $c.Flag }
    Write-Host "  Queuing $($c.QRName) in $($c.Zone)..."
    & gcloud @qa 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -eq 0) { Write-Host "  Queued." } else { Write-Host "  [WARN] Failed." }
}

# ── 3. Build experiment command ───────────────────────────────────────────────
$obsFlag    = if ($StoreObs)  { " --store-obs" } else { "" }
$chirpFlag  = if ($RunChirp)  { " --run-chirp" } else { "" }
$probeFlag  = if ($ProbeFamilies -ne "") { " --probe-families $ProbeFamilies" } else { "" }

if ($Script -eq "program_o_tpu.py") {
    $RunCmd = "nohup python3 $RemoteDir/program_o_tpu.py" +
              " --N $N --T-max $TMax --n-steps $NSteps --B $B --seed $Seed" +
              " --models $Models --out-dir $RemoteDir/$OutDir" +
              " --backend jax --require-tpu --controllers $Controllers" +
              $probeFlag + $chirpFlag + $obsFlag +
              " > $RemoteDir/$LogFile 2>&1 &"
} else {
    $RunCmd = "nohup python3 $RemoteDir/program_l_tpu.py" +
              " --N $N --T-max $TMax --n-steps $NSteps --B $B --seed $Seed" +
              " --models $Models --out-dir $RemoteDir/$OutDir" +
              " --backend jax --require-tpu --controllers $Controllers" +
              " --adversarial $Adversarial" + $obsFlag +
              " > $RemoteDir/$LogFile 2>&1 &"
}

# ── 4. Wait for a winner; experiment loop with fallback ───────────────────────
Write-Host "[POLL] Waiting for first zone to go ACTIVE (shotgun -- all zones queued, stale PROVISIONING re-queued every 15min)..."
$Deadline       = (Get-Date).AddMinutes($QueueMaxMin)
$Winner         = $null
$Succeeded      = $false
$ProvStartTimes = @{}   # track when each QRName entered PROVISIONING
$ProvTimeoutMin = 15    # re-queue if stuck PROVISIONING longer than this

while (-not $Succeeded) {
    $active      = $null
    $anyProgress = $false
    foreach ($c in $Candidates) {
        $st = Get-QRState $c.QRName $c.Zone
        Write-Host "  $($c.Zone): $st"

        if ($st -eq "PROVISIONING") {
            $anyProgress = $true
            # Track when this zone first entered PROVISIONING
            if (-not $ProvStartTimes.ContainsKey($c.QRName)) {
                $ProvStartTimes[$c.QRName] = Get-Date
            }
            $provMin = ((Get-Date) - $ProvStartTimes[$c.QRName]).TotalMinutes
            if ($provMin -gt $ProvTimeoutMin) {
                Write-Host "  [REQUEUE] $($c.Zone) stuck PROVISIONING ${provMin}min -- deleting and re-queuing fresh..."
                gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
                Start-Sleep -Seconds 5
                $reqCmd = "gcloud compute tpus queued-resources create $($c.QRName) --project=$Project --zone=$($c.Zone) --accelerator-type=$($c.AccelType) --runtime-version=$($c.Runtime) --node-id=$($c.NodeId) --spot --quiet 2>&1"
                Invoke-Expression $reqCmd | Out-Null
                $ProvStartTimes.Remove($c.QRName)
                Write-Host "  [REQUEUE] $($c.Zone) re-queued."
            }
        } elseif ($st -eq "WAITING_FOR_RESOURCES") {
            # Reset PROVISIONING timer if it fell back
            if ($ProvStartTimes.ContainsKey($c.QRName)) { $ProvStartTimes.Remove($c.QRName) }
            $anyProgress = $true
        } elseif ($st -eq "ACTIVE" -and $c.QRName -ne ($Winner.QRName)) {
            $active = $c; break
        }
    }

    if ($null -eq $active) {
        # Extend deadline while zones are actively PROVISIONING (resources allocated)
        if ($anyProgress) {
            $extendedDeadline = (Get-Date).AddMinutes(8)
            if ($extendedDeadline -gt $Deadline) {
                $Deadline = $extendedDeadline
                Write-Host "  [EXTEND] Zones active -- deadline extended to $($Deadline.ToString('HH:mm:ss'))"
            }
        }
        if ((Get-Date) -gt $Deadline) {
            Write-Host "[FAIL] Deadline reached with no ACTIVE zone."
            Delete-All $Candidates; exit 1
        }
        Start-Sleep -Seconds 20; continue
    }

    # Cancel previous winner (preempted one) if any, and reset deadline for standby
    if ($null -ne $Winner) {
        Write-Host "[FALLBACK] Previous winner $($Winner.Zone) preempted. Trying $($active.Zone)..."
        gcloud compute tpus tpu-vm delete $Winner.NodeId --project=$Project --zone=$($Winner.Zone) --quiet 2>$null
        gcloud compute tpus queued-resources delete $Winner.QRName --project=$Project --zone=$($Winner.Zone) --quiet 2>$null
        # Give standby a fresh 12-minute window to go ACTIVE
        $Deadline = (Get-Date).AddMinutes(12)
        Write-Host "  [FALLBACK] Standby deadline reset to $($Deadline.ToString('HH:mm:ss'))"
    }

    $Winner = $active
    Write-Host "[WIN] $($Winner.Zone) is ACTIVE."

    # Try to set up and launch experiment
    $ready = Launch-Experiment $Winner
    if (-not $ready) {
        Write-Host "[RETRY] Setup failed on $($Winner.Zone). Checking standbys..."
        Start-Sleep -Seconds 15; continue
    }

    # Launch experiment
    Write-Host "[RUN] Launching Program L..."
    "y" | gcloud compute tpus tpu-vm ssh $Winner.NodeId --project=$Project --zone=$($Winner.Zone) --command=$RunCmd
    Write-Host "[RUN] Polling log..."

    # Poll for completion or preemption
    $runStart = Get-Date
    $runEnd   = $runStart.AddMinutes($MaxMin)
    $preempted = $false

    while ((Get-Date) -lt $runEnd) {
        Start-Sleep -Seconds 60
        $st = Get-QRState $Winner.QRName $Winner.Zone
        if ($st -ne "ACTIVE") {
            Write-Host "[PREEMPTED] $($Winner.Zone) is now $st during run. Checking standbys..."
            $preempted = $true; break
        }
        $tail = "y" | gcloud compute tpus tpu-vm ssh $Winner.NodeId --project=$Project --zone=$($Winner.Zone) `
                    --command="tail -3 $RemoteDir/$LogFile 2>/dev/null" 2>$null
        $elapsed = [math]::Round(((Get-Date)-$runStart).TotalMinutes,0)
        Write-Host "[$($elapsed)m] $tail"
        # gcloud may return multi-line array; take the last element (the actual count)
        $doneRaw = "y" | gcloud compute tpus tpu-vm ssh $Winner.NodeId --project=$Project --zone=$($Winner.Zone) `
                    --command="grep -c 'Saved ->' $RemoteDir/$LogFile 2>/dev/null || echo 0" 2>$null
        $doneVal = if ($doneRaw -is [array]) { $doneRaw[-1] } else { $doneRaw }
        if ([int]($doneVal.Trim()) -gt 0) {
            Write-Host "[DONE] Experiment complete on $($Winner.Zone)."
            $Succeeded = $true; break
        }
    }

    if (-not $Succeeded -and -not $preempted) {
        Write-Host "[FAIL] Experiment timed out."
        Delete-All $Candidates; exit 1
    }
    # If preempted, loop continues to find next standby
}

# ── 5. Download results ───────────────────────────────────────────────────────
$remoteResult = $Winner.NodeId + ':' + $RemoteDir + '/' + $OutDir + "/program_l_N${N}_results.json"
$remoteLog    = $Winner.NodeId + ':' + $RemoteDir + '/' + $LogFile
New-Item -ItemType Directory -Force -Path "$SrcDir\$OutDir" | Out-Null
gcloud compute tpus tpu-vm scp $remoteResult "$SrcDir\$OutDir\program_l_N${N}_results.json" --project=$Project --zone=$($Winner.Zone)
gcloud compute tpus tpu-vm scp $remoteLog    "$SrcDir\$OutDir\$LogFile"                      --project=$Project --zone=$($Winner.Zone) 2>$null

# ── 6. Analyze locally ────────────────────────────────────────────────────────
$rf = "$SrcDir\$OutDir\program_l_N${N}_results.json"
if (Test-Path $rf) {
    $env:PYTHONIOENCODING = "utf-8"
    python "$SrcDir\analyze_program_l.py" $rf
} else { Write-Host "[WARN] Results file not found." }

# ── 7. Delete ritual ──────────────────────────────────────────────────────────
Delete-All $Candidates
Write-Host "[COMPLETE] Done. Winner: $($Winner.Zone)"
