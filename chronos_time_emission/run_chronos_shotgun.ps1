$Project = "time-emission"
$Workspace = $PSScriptRoot
$LogFile = Join-Path $Workspace "chronos_shotgun.log"
$CloudSdk = "C:\Users\cityz\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$CloudSdkBin = "C:\Users\cityz\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
$ProgramExtraArgs = if ($env:CHRONOS_EXTRA_ARGS) { $env:CHRONOS_EXTRA_ARGS } else { "--blocks 20 --block-size 100 --d 512 --sparse-block 32 --layers 6 --condition all" }

$Candidates = @(
    @{ Zone = "us-east1-d";     QR = "chronos-time-emission-us-v1"; Node = "chronos-time-emission-us-node-v1"; Type = "v6e-8" },
    @{ Zone = "europe-west4-a"; QR = "chronos-time-emission-eu-v1"; Node = "chronos-time-emission-eu-node-v1"; Type = "v6e-8" }
)

$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

function Write-LaunchLog {
    param([string]$Message)
    Add-Content -Path $LogFile -Value ("{0} {1}" -f (Get-Date -Format o), $Message)
}

function Get-QueuedState {
    param([hashtable]$Candidate)
    $state = (& $CloudSdk compute tpus queued-resources describe $($Candidate["QR"]) --project=$Project --zone=$($Candidate["Zone"]) --format="value(state.state)" 2>$null)
    if ([string]::IsNullOrWhiteSpace($state)) { return "MISSING" }
    return $state.Trim()
}

function Get-NodeState {
    param([hashtable]$Candidate)
    $state = (& $CloudSdk compute tpus tpu-vm describe $($Candidate["Node"]) --project=$Project --zone=$($Candidate["Zone"]) --format="value(state)" 2>$null)
    if ([string]::IsNullOrWhiteSpace($state)) { return "MISSING" }
    return $state.Trim()
}

function Ensure-QueuedResource {
    param([hashtable]$Candidate)
    $state = Get-QueuedState $Candidate
    if ($state -match "ACCEPTED|WAITING_FOR_RESOURCES|PROVISIONING|CREATING_NODES|ACTIVE") { return }
    if ($state -ne "MISSING") {
        & $CloudSdk compute tpus queued-resources delete $($Candidate.QR) --project=$Project --zone=$($Candidate.Zone) --quiet 2>$null | Out-Null
        Start-Sleep -Seconds 3
    }
    Write-LaunchLog "queue_create zone=$($($Candidate.Zone)) qr=$($($Candidate.QR)) node=$($($Candidate.Node))"
    $createOutput = & $CloudSdk compute tpus queued-resources create $($Candidate.QR) `
        --node-id=$($Candidate.Node) `
        --project=$Project `
        --zone=$($Candidate.Zone) `
        --accelerator-type=$($Candidate.Type) `
        --runtime-version=v2-alpha-tpuv6e `
        --spot `
        --quiet 2>&1
    Write-LaunchLog "queue_create_exit zone=$($($Candidate.Zone)) code=$LASTEXITCODE output=$($createOutput -join ' | ')"
}

function Cleanup-Loser {
    param([hashtable]$Winner)
    foreach ($candidate in $Candidates) {
        if ($candidate.Zone -eq $Winner.Zone -and $candidate.Node -eq $Winner.Node) { continue }
        Write-LaunchLog "cleanup_loser zone=$($($candidate.Zone)) qr=$($($candidate.QR)) node=$($($candidate.Node))"
        & $CloudSdk compute tpus tpu-vm delete $($candidate.Node) --project=$Project --zone=$($candidate.Zone) --quiet 2>$null | Out-Null
        & $CloudSdk compute tpus queued-resources delete $($candidate.QR) --project=$Project --zone=$($candidate.Zone) --quiet 2>$null | Out-Null
    }
}

Write-LaunchLog "shotgun_start extra=$ProgramExtraArgs"
foreach ($candidate in $Candidates) {
    Ensure-QueuedResource $candidate
}

$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
    foreach ($candidate in $Candidates) {
        $qrState = Get-QueuedState $candidate
        $nodeState = Get-NodeState $candidate
        Write-LaunchLog "poll zone=$($($candidate.Zone)) qr=$qrState node=$nodeState"
        if ($qrState -eq "ACTIVE" -and $nodeState -eq "READY") {
            Cleanup-Loser $candidate
            Set-Location $Workspace
            $env:CHRONOS_NODE = $candidate.Node
            $env:CHRONOS_ZONE = $candidate.Zone
            $env:CHRONOS_EXTRA_ARGS = $ProgramExtraArgs
            Write-LaunchLog "launching zone=$($($candidate.Zone)) node=$($($candidate.Node))"
            & .\run_chronos_ready_node.ps1 *>> $LogFile
            Write-LaunchLog "launcher_exit zone=$($($candidate.Zone)) code=$LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    Start-Sleep -Seconds 60
}

Write-LaunchLog "timeout_waiting_for_ready_node"
exit 2
