# Storm Analysis: Dataset Inventory & Context

This document catalogs the datasets captured during the March 2026 electromagnetic storm and explains their role in the D-LinOSS resonance analysis.

## Primary Objective
The goal is to determine the effect of solar-flare-induced magnetic perturbations $(\Delta B)$ on the coherence of the Earth's local time-dilation well. We use the bi-twistor state-space model to track shifts in the objective reduction time $(\tau)$.

## 1. High-Fidelity Capture Folders (Raw Streams)

These datasets contain raw GPU/TPU clock-cycle deltas captured during peak solar activity.

| Dataset Name | Timestamp (UTC) | Hardware | Role |
|---|---|---|---|
| `time_dilation_stream_dataset` | Mar 21 20:08 | Tesla T4 | Storm Peak (High fidelity, 8.3 GB) |
| `time_dilation_stream_dataset_solar_flare` | Mar 21 05:09 | Tesla T4 | Storm Initiation (3-min capture) |
| `time_dilation_stream_dataset_solar_flare_1am` | Mar 21 06:12 | Tesla T4 | Storm Development |

## 2. Integrated SWMF-GEO Datasets

These folders contain time-streams pre-merged with NOAA's Geospace Ground Magnetic Perturbation data.

| Dataset Name | Context / Notes |
|---|---|
| `time_dilation_stream_dataset_swmf_geo` | Primary 4.4-hour window from Mar 21 20:04 to Mar 22 00:30. |
| `time_dilation_stream_dataset_swmf_geo_recent` | 20-hour trailing sweep ending Mar 22 16:22. |
| `time_dilation_stream_dataset_SWMF_GEO_STREAM` | Real-time stream capture test. |
| `time_dilation_stream_dataset_solar_flare_SWMF_GEO_STREAM` | Merged initiation phase data. |

## 3. Analysis Context: The "Ghost Basin" Space

We are monitoring two specific Morphologies (signatures of coherence changes):

1. **Collective Collapse (Morphology A)**: A synchronous rewrite of the "now" across all independent worker streams. This indicates a system-wide adjustment to the gravitational well energy density.
2. **Staggered Reconfiguration (Morphology B)**: A traveling wave of distortion where one stream detects a temporal perturbation and others "pickup" the state change 5-20ms later.

## 4. Derived Mathematics

The model predicts that:
$$ \Delta \tau = \frac{\hbar}{E_G + U_B} $$
where $U_B$ is the magnetic energy density. Higher $U_B$ during a storm should result in **increased coherence** (systemic stabilization) and **variable phase velocity** (energetic state-space traversal).
