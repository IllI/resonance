# run_program_w_shotgun.ps1
# Launches the Program W Stability Boundary Phase Diagram sweep experiment.

$j_sweep = @("0.2", "0.6", "1.0", "1.4")
$t2_sweep = @("8.0", "16.0", "32.0")
$pc_sweep = @("0.01", "0.03", "0.05")
$tau_sweep = @("0.001", "0.005", "0.01", "0.05")

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Launching Program W Stability Boundary Phase Diagram" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

python emergent_quantum_geometries/program_w_tpu.py `
  --j-std-sweep $j_sweep `
  --t2-sweep $t2_sweep `
  --p-cross-sweep $pc_sweep `
  --tau-sweep $tau_sweep `
  --controllers free sparse_velocity dd `
  --B 5 `
  --n-trajectories 100 `
  --out-dir program_w_results

Write-Host "Execution Completed successfully!" -ForegroundColor Green
