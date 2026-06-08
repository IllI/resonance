$Project = "time-emission"
$Zone = if ($env:CHRONOS_ZONE) { $env:CHRONOS_ZONE } else { "us-east1-d" }
$NodeId = if ($env:CHRONOS_NODE) { $env:CHRONOS_NODE } else { "chronos-time-emission-node-v1" }
$ExtraArgs = if ($env:CHRONOS_EXTRA_ARGS) { $env:CHRONOS_EXTRA_ARGS } else { "--blocks 20 --block-size 100 --d 512 --sparse-block 32 --layers 6 --condition all" }
$RemoteDir = "/home/cityz/chronos_time_emission"
$RemoteOut = "$RemoteDir/results"
$LogFile = "chronos_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

$AllowedZones = @("us-east1-d", "europe-west4-a")
if ($Zone -notin $AllowedZones) {
    Write-Host "[ERROR] Zone $Zone is not in the CHRONOS v6e spot allow-list."
    exit 1
}
if ($ExtraArgs -match '[;&|`<>$]') {
    Write-Host "[ERROR] CHRONOS_EXTRA_ARGS contains shell control characters."
    exit 1
}

Write-Host "[CHECK] Verifying ready TPU VM: $NodeId in $Zone"
$NodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($NodeState)) {
    Write-Host "[ERROR] TPU VM not found. Queue a TRC v6e spot node first."
    exit 1
}
if ($NodeState.Trim() -ne "READY") {
    Write-Host "[ERROR] TPU VM state is $($NodeState.Trim()), expected READY."
    exit 1
}

Write-Host "[SSH] Preparing remote workspace..."
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="mkdir -p $RemoteDir $RemoteOut"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[SSH] Verifying TPU-backed JAX runtime..."
$RuntimeCheck = "python3 -c 'import jax; print(jax.__version__, jax.default_backend(), jax.devices()); assert jax.default_backend() == ""tpu""'"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RuntimeCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SETUP] Installing jax[tpu] and helpers..."
    "y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="python3 -m pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html scipy numpy"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    "y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RuntimeCheck
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Host "[SCP] Uploading CHRONOS source only..."
$SourceFiles = Get-ChildItem -Path $PSScriptRoot -Filter "*.py" -File
foreach ($sourceFile in $SourceFiles) {
    "y" | gcloud compute tpus tpu-vm scp $sourceFile.FullName "$NodeId`:$RemoteDir/" --project=$Project --zone=$Zone --quiet
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Host "[RUN] Launching CHRONOS on TPU..."
$RunCmd = "bash -lc 'cd /home/cityz; rm -f $RemoteDir/$LogFile; export PYTHONPATH=/home/cityz; nohup python3 -u -m chronos_time_emission.run_chronos_tpu --require-tpu --out $RemoteOut $ExtraArgs > $RemoteDir/$LogFile 2>&1 & echo CHRONOS_PID=`$!'"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RunCmd
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[MONITOR] Waiting for initial output..."
Start-Sleep -Seconds 45
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="tail -n 80 $RemoteDir/$LogFile 2>/dev/null || true"
Write-Host "[DONE] CHRONOS launched. No result files downloaded."
