# watch_program_am_status.ps1
# Lightweight local watcher to avoid chat/token polling.
#
# Monitors Program AM queued resource + VM state in TRC-approved v6e spot zones:
# - us-east1-d
# - europe-west4-a
#
# Usage (PowerShell):
#   .\\watch_program_am_status.ps1
#
# Optional env vars:
#   $env:PROJECT_ID   (default: time-emission)
#   $env:INTERVAL_SEC (default: 20)

$ErrorActionPreference = "SilentlyContinue"

$Project = if ($env:PROJECT_ID) { $env:PROJECT_ID } else { "time-emission" }
$IntervalSec = if ($env:INTERVAL_SEC) { [int]$env:INTERVAL_SEC } else { 20 }

$Zones = @("us-east1-d", "europe-west4-a")
$QRName = "program-am-qr-v6"
$NodeId = "program-am-node-v6"

$Gcloud = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $Gcloud)) {
  Write-Host "[ERROR] gcloud not found at: $Gcloud"
  exit 1
}

function Get-QrState([string]$zone) {
  & $Gcloud compute tpus queued-resources describe $QRName --project=$Project --zone=$zone --format="value(state.state)" 2>$null
}

function Get-VmState([string]$zone) {
  # state + health, tab-separated if both exist
  & $Gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$zone --format="value(state,health)" 2>$null
}

$last = @{}
Write-Host "Watching $Project ($QRName / $NodeId) every ${IntervalSec}s. Ctrl+C to stop."
while ($true) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  foreach ($z in $Zones) {
    $qr = (Get-QrState $z)
    $vm = (Get-VmState $z)
    if ([string]::IsNullOrWhiteSpace($qr)) { $qr = "-" }
    if ([string]::IsNullOrWhiteSpace($vm)) { $vm = "-" }
    $key = "$z"
    $cur = "$qr | $vm"
    if (-not $last.ContainsKey($key) -or $last[$key] -ne $cur) {
      $last[$key] = $cur
      Write-Host "[$ts] $z  QR=$qr  VM=$vm"
    }
  }
  Start-Sleep -Seconds $IntervalSec
}

