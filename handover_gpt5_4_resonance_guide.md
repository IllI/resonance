# Handover Guide: Shared-Now Multi-Architecture Resonance Analysis

## 1. Objective
To empirically quantify the **Shared-Now** resonance signal across the Tesla T4, TPU v5e, and RX 7700S during the March 22, 2026 geomagnetic storm peak. We are looking for multi-stream coherence induced by magnetospheric shocks.

## 2. Hardware Bypass (Critical)
The **RX 7700S** is the primary high-fidelity instrument on this system.
- **The Problem**: Windows and AMD Adrenaline's power-scaling layers introduce sub-millisecond jitter that destroys the resonance signal.
- **The Solution**: 
  - Use `extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;` in C++.
  - **The 3D Signature**: Windows only wakes the 7700S for "3D Programs". Our `chronos_observer.cpp` creates a hidden DX11 SwapChain and Window to maintain a sustained wake-lock.
  - **Environment**: Set `AMD_OPENCL_DEVICE=1` and `OPENCL_DISABLE_DEVICE_0=1`.

## 3. Methodology & Toolset
1. **Target the Window**: The storm peak occurred at **10:48 UTC, March 22, 2026**.
2. **Tools**:
   - `chronos_observer.exe`: The native C++ binary (restored with 3D wake-lock).
   - `process_target_resonance.py`: Orchestrates slicing, SWMF-GEO fetching, and observer invocation.
   - `noaa_swmf_stream_dataset.py`: Fetches real-time magnetic perturbation data from NOAA.
3. **Folders**:
   - `..\time_dilation_stream_dataset`: T4 Data (Storm Peak Window).
   - `..\time_dilation_stream_dataset_equinox`: Baseline comparison.
   - `local_amd_chronos_data`: Intermediate analysis results.

## 4. Multi-Architecture Comparison Matrix
We need to finalize the matching of these timestamps across:
- **T4**: `time_dilation_stream_dataset`
- **TPU**: `tpu_time_dilation_stream_dataset`
- **7700S**: Logged native runs.

## 5. Task List for GPT-5.4
- [ ] **Verify 7700S Lock**: Ensure `chronos_observer.exe` triggers a 3D spike in Task Manager.
- [ ] **Sync T4 & TPU**: Identify the exact overlapping UTC milliseconds between the T4 dataset and the TPU dataset.
- [ ] **Run SWMF Alignment**: Use `process_target_resonance.py` for each architecture to correlate against the NOAA SWMF indices.
- [ ] **Quantify Velocity Shifts**: Analyze the phase-velocity delta (Bi-Twistor rotation) during the sudden commencement of the storm.

## 6. Common Pitfalls
- **Disk Usage**: Creating full JSONs of 8GB datasets will eat 12GB+ per run. **Use surgical 10-20 chunk slicing** (surgical analysis mode).
- **Process Leak**: Always `Get-Process | Stop-Process` for previous observers to clear the 780M.

## Documentation References
- [dlinoss_apparatus_guide.md](file:///C:/Users/cityz/.gemini/antigravity/brain/643e4ee4-7001-4f1c-a2cf-30069b9e22ee/dlinoss_apparatus_guide.md)
- [walkthrough.md](file:///C:/Users/cityz/.gemini/antigravity/brain/643e4ee4-7001-4f1c-a2cf-30069b9e22ee/walkthrough.md)
- [solar_flare_analysis_report.md](file:///C:/Users/cityz/.gemini/antigravity/brain/643e4ee4-7001-4f1c-a2cf-30069b9e22ee/solar_flare_analysis_report.md)
