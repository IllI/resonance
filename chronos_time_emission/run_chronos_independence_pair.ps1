$Project = "time-emission"
$Workspace = $PSScriptRoot
$LogFile = Join-Path $Workspace "chronos_independence_pair.log"
$CloudSdk = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
$Module = "chronos_time_emission.run_chronos_independence_tpu"
$Samples = if ($env:CHRONOS_INDEPENDENCE_SAMPLES) { $env:CHRONOS_INDEPENDENCE_SAMPLES } else { "32768" }
$SampleHz = if ($env:CHRONOS_INDEPENDENCE_SAMPLE_HZ) { $env:CHRONOS_INDEPENDENCE_SAMPLE_HZ } else { "128" }
$Size = if ($env:CHRONOS_INDEPENDENCE_SIZE) { $env:CHRONOS_INDEPENDENCE_SIZE } else { "96" }
$Seed = if ($env:CHRONOS_INDEPENDENCE_SEED) { $env:CHRONOS_INDEPENDENCE_SEED } else { "11" }
$RunId = if ($env:CHRONOS_INDEPENDENCE_RUN_ID) { $env:CHRONOS_INDEPENDENCE_RUN_ID } else { "" }

$Targets = @(
    @{ Zone = "us-east1-d";     Node = "chronos-time-emission-us-node-v1"; Label = "alice_us" },
    @{ Zone = "europe-west4-a"; Node = "chronos-time-emission-eu-node-v1"; Label = "bob_eu" }
)

$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

function Write-RunLog {
    param([string]$Message)
    Add-Content -Path $LogFile -Value ("{0} {1}" -f (Get-Date -Format o), $Message)
}

function Get-NodeState {
    param([hashtable]$Target)
    $state = (& $CloudSdk compute tpus tpu-vm describe $($Target.Node) --project=$Project --zone=$($Target.Zone) --format="value(state)" 2>$null)
    if ([string]::IsNullOrWhiteSpace($state)) { return "MISSING" }
    return $state.Trim()
}

function Launch-Collector {
    param([hashtable]$Target, [int64]$StartEpoch)
    $label = if ($RunId) { "$($Target.Label)_$RunId" } else { $Target.Label }
    $extra = "collect --node-label $label --samples $Samples --sample-hz $SampleHz --size $Size --seed $Seed --start-epoch $StartEpoch"
    Write-RunLog "launch label=$label zone=$($Target.Zone) node=$($Target.Node) start_epoch=$StartEpoch extra=$extra"
    $command = @"
`$env:CHRONOS_NODE='$($Target.Node)'
`$env:CHRONOS_ZONE='$($Target.Zone)'
`$env:CHRONOS_MODULE='$Module'
`$env:CHRONOS_EXTRA_ARGS='$extra'
Set-Location '$Workspace'
& .\run_chronos_ready_node.ps1
"@
    Start-Process -FilePath "C:\Program Files\PowerShell\7\pwsh.exe" `
        -ArgumentList @("-NoProfile", "-Command", $command) `
        -WindowStyle Hidden | Out-Null
}

Write-RunLog "pair_wait_start samples=$Samples sample_hz=$SampleHz size=$Size seed=$Seed run_id=$RunId"
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
    $states = @()
    foreach ($target in $Targets) {
        $state = Get-NodeState $target
        $states += "$($target.Label)=$state"
    }
    Write-RunLog ("poll " + ($states -join " "))
    if (($states -join " ") -notmatch "MISSING|CREATING|STARTING|PROVISIONING|STOPPING|DELETING" -and ($states -join " ") -match "alice_us=READY" -and ($states -join " ") -match "bob_eu=READY") {
        $startEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + 240
        foreach ($target in $Targets) {
            Launch-Collector $target $startEpoch
        }
        Write-RunLog "pair_launch_complete start_epoch=$startEpoch"
        exit 0
    }
    Start-Sleep -Seconds 60
}

Write-RunLog "timeout_waiting_for_pair"
exit 2
