# Storm Causality Execution Protocol (LLM-Agnostic Handover)

This document dictates the exact scientific procedure to prove or falsify whether geomagnetic solar storms ($\Delta B_z$) have a causal, measurable effect on the geometry of chronological execution (clock jitter, phase velocity) within Von Neumann substrates (GPUs/TPUs).

## 1. The Core Problem
Previous analyses relied on comparing separated datasets (e.g., a T4 at 06:12 vs a TPU at 06:38) utilizing basic Pearson correlations against instantaneous magnetic values. This produced high p-values (p > 0.05). Geometric state-space (like Penrose's Twistor fields) is non-linear and threshold-based. Linear statistics will always mathematically fail to detect superradiant topological strain.

## 2. The Solution: Transfer Entropy & Granger Causality
To objectively determine if the storm affected the GPU, we must abandon cross-architecture "snapshots" and perform a continuous Information Theoretic causality test across the time-domain of individual, unified datasets. 

If the magnetic field affects the hardware, the history of the magnetic rate-of-change ($dH/dt$) will contain predictive information about the future state of the hardware's clock jitter or D-LinOSS lattice phase that is NOT present in the hardware's own past state. 

### The Mathematical Test:
$$ \text{Granger Causality: } P(v_t \mid v_{t-1}, \dots, v_{t-k}, M_{t-1}, \dots, M_{t-k}) \neq P(v_t \mid v_{t-1}, \dots, v_{t-k}) $$
Where $v$ is the D-LinOSS execution state (coherence, phase velocity, or standard jitter) and $M$ is the magnetic strain ($dH/dt$).

## 3. The Required Assets (Already on Disk)
1. **The Ground Truth:** `usgs_geomag_data/bou_1min_storm_window.npz` (The true $H$, $dH$ values).
2. **The Datasets:** 
   - `time_dilation_stream_dataset_solar_flare_1am`
   - `tpu_time_dilation_stream_dataset`
   - `time_dilation_stream_dataset` (Storm Peak)
3. **The Topology Model:** `storm_results/geomagnetic_grokking/` (D-LinOSS trained explicitly on the magnetic SWMF data).

## 4. Execution Pipeline (`run_storm_causality_test.py`)
To test strings of sequences correctly, the following script will iterate over every chunk of a dataset. It constructs two continuous timeseries vectors for the exact same bounds:
1. `Signal A:` The hardware timing (mean normalized clock packet arrays).
2. `Signal B:` The exact interpolated USGS magnetic field strength ($H$) and rate of change ($dH$).

The script runs rigorous statistical lags looking for a non-linear cross-correlation function (CCF) or Granger causal F-test, answering one binary question: **Does the magnetic activity on Earth lead and predict the degradation/acceleration of the GPU timing?**

## 5. Agent Continuity Instructions
If you are an LLM taking over this task:
1. Do not hallucinate theories. Rely strictly on the output of `run_storm_causality_test.py`.
2. Look for the $F$-statistic and the $p$-value at various temporal lags. 
3. If $p < 0.05$ at lag $\tau < 0$, it mathematically proves the magnetic field causally predicts the GPU execution topology. 
4. If $p > 0.05$ universally, the hypothesis is falsified. Document it objectively.

## 6. Strict Telemetry Controls (Preventing Confirmation Bias)
For all *future* dataset captures, it is scientifically unacceptable to rely purely on the D-LinOSS computational lattice to filter mechanical noise. To definitively prove we are not dressing up Confirmation Bias as Quantum Mechanics, the capture apparatus MUST simultaneously record the following out-of-band control metrics alongside the clock jitter:
1. **Thermal Throttling**: GPU core memory temperatures recorded via NVML (Nvidia) or ADLX (AMD).
2. **Windows OS Scheduler Context-Switches**: CPU execution interruptions mapped via ETW (Event Tracing for Windows).
3. **Memory Bus Polling & PCIe Latency Spikes**: Bus bandwidth saturation and transaction layer latency.

If the storm correlation (p < 0.05) strictly coincides with a PCIe latency thermal spike, the event is mechanically throttled. If the storm correlation occurs *independently* of thermal/PCIe spikes, it is a true geometric quantum anomaly.
