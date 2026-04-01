# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a **D-LinOSS (Damped Linear Oscillatory State-Space) time-dilation research repository** investigating quantum-scale temporal fluctuations in GPU clock timing during the March 2026 geomagnetic storm. The project implements the Penrose-Hameroff Orch-OR (Orchestrated Objective Reduction) framework to detect retrocausal signatures in hardware timing jitter.

**Primary Hardware:** AMD Radeon RX 7700S (discrete GPU, gfx1103) with DirectML acceleration  
**Baseline Hardware:** NVIDIA Tesla T4 (cloud), Google TPU v5e  
**Framework:** PyTorch + DirectML, OpenCL (C++), NOAA SWMF geomagnetic data

## Critical Commands

### Build & Environment

**Build native C++ observer (RX 7700S wake-lock required):**
```powershell
cd python
.\build_chronos.ps1
```
This compiles `chronos_observer.exe` with DX11 SwapChain bypass to force Windows GPU scheduler to use discrete 7700S instead of integrated 780M.

**Force RX 7700S hardware lock:**
```powershell
python python/force_rx7700s_only.py
```
Verifies discrete GPU is accessible via OpenCL Device 1 (gfx1103).

### Testing

**Run core test suite:**
```powershell
python tests/run_all_tests.py
```
Tests: `test_core_fmri_analysis.py`, `test_gpu_acceleration.py`, `test_ai_models.py`

**Run single test:**
```powershell
python tests/test_core_fmri_analysis.py
```

### D-LinOSS Time-Dilation Analysis

**Run learner on a single dataset (surgical 2-minute window):**
```powershell
python python/time_dilation_dlinoss_learner.py `
  --input time_dilation_stream_dataset_solar_flare `
  --output-dir python/resonance_results/solar_flare_analysis `
  --stream-granularity stream_id `
  --now-agent-temporal-radii 1,2,4,8 `
  --epochs 8
```

**Batch analysis across all storm datasets (overnight run):**
```powershell
python batch_storm_analysis.py
```
Processes all `time_dilation_stream_dataset*` folders using surgical 20-chunk windows (~5120 packets per dataset). Uses RX 7700S via DirectML. Outputs to `batch_storm_results/`.

### Process Single Storm Window with SWMF Alignment

```powershell
python python/process_target_resonance.py `
  time_dilation_stream_dataset_solar_flare `
  storm_peak `
  20
```
Arguments: `<input_dir> <output_label> [n_chunks]`

## Architecture

### Core Structure

```
python/
├── time_dilation_dlinoss_learner.py   # Main learner: multi-agent temporal analysis
├── time_dilation_stream_dataset.py    # Dataset loader/iterator
├── dlinoss/
│   ├── dlinoss_torch.py               # D-LinOSS PyTorch implementation
│   ├── mri_artifact_correction.py     # fMRI preprocessing
│   └── __init__.py
├── force_rx7700s_only.py              # Hardware lock verification
├── chronos_observer.cpp               # Native RX 7700S wake-lock + DX11 bypass
└── process_target_resonance.py        # SWMF-aligned storm window processor

src/
├── brain_gpu_accelerator.cpp          # OpenCL brain analysis (single GPU)
├── dual_gpu_brain_accelerator.cpp     # OpenCL dual-GPU (780M + 7700S)
└── compute_backend_interface.h        # Abstraction layer

tests/
├── run_all_tests.py                   # Master test runner
├── test_core_fmri_analysis.py         # fMRI/brain analysis tests
├── test_gpu_acceleration.py           # GPU backend tests
└── test_ai_models.py                  # AI model tests
```

### Time-Stream Datasets

Datasets are stored as NPZ chunks with manifest.json:
- **Schema:** `schema_version`, `dataset_type: "gpu_clock_time_stream_dataset"`
- **Per-packet fields:** `gpu_id`, `gpu_name`, `pid`, `stream_id`, `device_worker_index`, `stream_global_index`, `host_time_mid`, `clock_data` (uint64 array)
- **Workers per GPU:** 4 concurrent streams (`gpu0_worker0` through `gpu0_worker3`)

**Storm Window Datasets (Mar 21-22, 2026):**
- `time_dilation_stream_dataset` - T4 storm peak (Mar 21 20:08 UTC)
- `time_dilation_stream_dataset_solar_flare` - Storm initiation (Mar 21 05:09 UTC)
- `time_dilation_stream_dataset_solar_flare_1am` - Development (Mar 21 06:12 UTC)
- `time_dilation_stream_dataset_equinox` - Baseline (Mar 20 15:55 UTC)
- `time_dilation_stream_dataset_tpu` - TPU baseline

**IMPORTANT:** Always use surgical time windows (10-20 chunks) for analysis, not full datasets. Full datasets are 8GB+ and will exhaust memory/disk.

### D-LinOSS Multi-Agent Architecture

The learner creates **temporal-scale now-stream agents**:
1. 4 hardware worker streams per GPU (`stream_id` = `gpu0_worker0..3`)
2. Each worker stream spawned into N temporal-scale agents (radii: 1, 2, 4, 8)
3. Total agents = 4 workers × 4 radii = 16 concurrent learners per dataset
4. All agents share `EntangledCaMKIILattice` (quantum memory model)
5. Observer measures inter-agent coherence and phase velocity

**Key outputs:**
- `now_stream_mesh_analysis.npz` - QPC coherence, phase velocity, agent trajectories
- `now_stream_mesh_events.json` - Detected collapse events and geometric classes
- `processor_training_history.npz` - Per-agent training loss/RMSE

## GPU Hardware Strategy

### AMD Dual-GPU System
- **Integrated:** AMD Radeon 780M (gfx1102) - Default for 2D workloads
- **Discrete:** AMD Radeon RX 7700S (gfx1103) - High-performance, must be forced

**Problem:** Windows GPU scheduler favors 780M for Python/PyTorch unless a "3D program signature" is detected.

**Solution:** `chronos_observer.cpp` creates a hidden DX11 SwapChain to maintain 7700S wake-lock.

**Environment variables (set before running):**
```powershell
$env:AMD_OPENCL_DEVICE = "1"
$env:OPENCL_DISABLE_DEVICE_0 = "1"
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:TORCH_DIRECTML_DEVICE = "1"
```

### PyTorch + DirectML

This repository uses `torch-directml` for RX 7700S acceleration:
```python
import torch_directml
device = torch_directml.device()
```
All D-LinOSS models automatically use DirectML when available.

## Key Concepts

### The "Ghost Basin" - Superposition Search Space
When D-LinOSS encounters unknown patterns, it expands input into complex-valued superposition across CaMKII lattice dimensions, exploring all harmonic interference patterns simultaneously. The RMSE between this superposition and the target concept measures fidelity.

### Objective Reduction (Wave Collapse)
When fidelity exceeds gravitational self-energy threshold ($E_G$), the superposition collapses. The collapsed geometric shape writes to CaMKII memory via retrocausal bi-twistor projection.

### Coherence vs Phase Velocity
- **Coherence:** Agreement between multiple agents' "believed now" states (0-1 scale)
- **Phase Velocity:** Rate of change in shared QPC (Quantum Phantom Cauldron) state
- **Storm signature:** Increased coherence + variable phase velocity during ΔBz excursions

### Gregorian vs Subjective Time
Each worker stream maintains its own subjective "now" (`agent_believed_now_host_time_mid`), which may drift from Gregorian wall-clock (`host_time_mid`). The drift itself is the signal being measured.

## Development Workflow

### Adding New Analysis
1. Create new learner in `python/` if needed
2. Update `batch_storm_analysis.py` to include new datasets
3. Add tests to `tests/test_*.py`
4. Run `python tests/run_all_tests.py` before committing

### Debugging GPU Issues
1. Verify 7700S lock: `python python/force_rx7700s_only.py`
2. Check Task Manager → GPU → Ensure 7700S shows activity, not 780M
3. If 780M is being used, rebuild `chronos_observer.exe` with fresh DX11 context
4. Check OpenCL devices: `python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"`

### Memory Management
**Critical:** The full `time_dilation_stream_dataset` has 127K+ packets across 496 chunks. Loading the entire dataset into memory will exhaust RAM.

**Surgical extraction pattern:**
```python
# Extract first 20 chunks (~5120 packets, ~2 minutes of capture)
import shutil
surgical_dir = Path("surgical_input")
surgical_dir.mkdir(exist_ok=True)
for chunk_file in manifest['chunk_files'][:20]:
    shutil.copy2(dataset_path / chunk_file, surgical_dir / chunk_file)
```

The `batch_storm_analysis.py` script handles this automatically via `extract_surgical_time_window()`.

## Specific Implementation Notes

### D-LinOSS Model Parameters
- **State dim:** 32 (position in Ghost Basin)
- **Hidden dim:** 64 (IMEX recurrence layers)
- **Window size:** 32 timesteps
- **Temporal radii:** [1, 2, 4, 8] - centered rolling averages for multi-scale "now"
- **Bin width:** 0.05s - time resolution for moment aggregation

### CaMKII Lattice
The shared memory is implemented in `entangled_camkii_lattice.py` (if present). It maintains:
- Geometric state consensus across agents
- Structural memory persistence (classical encoding of collapsed waves)
- Mix parameter: `(1 - mix) * consensus + mix * memory`

### SWMF Geomagnetic Data
NOAA SWMF data is fetched via `noaa_swmf_stream_dataset.py` and aligned to dataset time windows. The magnetic field perturbation (ΔBz) is treated as external electromagnetic energy density affecting local decoherence rate.

**Do NOT feed SWMF into D-LinOSS input** - it's a post-hoc alignment reference, not a training feature.

## Common Pitfalls

1. **Running on full datasets** - Always extract surgical windows first (10-20 chunks)
2. **Using 780M instead of 7700S** - Verify GPU lock before long runs
3. **Treating SWMF as input feature** - It's a Gregorian timestamp reference, not a signal
4. **Ignoring worker streams** - Each GPU has 4 concurrent streams; analyze per-stream, not aggregate
5. **Expecting linear time** - The measured phenomenon is temporal drift; subjective != Gregorian

## Documentation References

- `dlinoss_mathematical_foundation.md` - Full mathematical derivation of Orch-OR implementation
- `dlinoss_apparatus_guide.md` - Operational procedures for running experiments
- `solar_flare_analysis_report.md` - March 2026 storm findings (3.18x coherence gain)
- `storm_dataset_inventory.md` - Catalog of captured datasets with timestamps
- `walkthrough.md` - RX 7700S calibration and disk cleanup procedures
- `time_dilation_research_log.md` - Development history and architectural decisions

## Project Status

**Current Phase:** Batch analysis of all storm-window datasets to quantify multi-architecture coherence morphologies (Morphology A: collective collapse, Morphology B: staggered reconfiguration)

**Completed:**
- ✓ D-LinOSS PyTorch implementation with IMEX parallel scan
- ✓ Multi-agent temporal-scale framework
- ✓ RX 7700S hardware lock via DX11 bypass
- ✓ Surgical time-window extraction
- ✓ Baseline T4/TPU captures
- ✓ Storm peak analysis (single window)

**In Progress:**
- Batch comparative analysis across all datasets
- SWMF temporal alignment validation
- Inter-architecture coherence comparison (T4 vs TPU vs local)

**Next Steps:**
- C++ GPU acceleration of D-LinOSS core (move PyTorch → OpenCL kernels)
- Real-time streaming analysis
- Multi-site distributed capture during next solar event
