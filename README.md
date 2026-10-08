# Resonance: fMRI analysis and visualization research prototypes

Resonance contains software for exploring fMRI time series, brain-network
connectivity, anatomical labeling, and 3D visualization. It also includes
experimental ROI-detection, streaming, and OpenCL acceleration code.

**Status: research prototype.** Components have different levels of completeness.
The repository does not establish production readiness, clinical validity,
scanner interoperability, or a guaranteed processing latency. This README
describes the default branch; experiments on other branches may differ.

## What is in the repository

These links identify implementations, not certifications of accuracy or performance.

| Area | Source | Scope and limitations |
| --- | --- | --- |
| fMRI loading | [fmri_data_loader.py](python/fmri_data_loader.py) | NIfTI loading and dataset helpers; downloads require network access and dataset-specific checks. |
| Brain-network analysis | [neural_network_analyzer.py](python/neural_network_analyzer.py) | Preprocessing, correlation matrices, sliding-window connectivity, change detection, and network metrics. |
| ROI detection and network classification | [ai_roi_detector.py](python/ai_roi_detector.py) | Signal-processing features and classifier code; anatomical detection accuracy and pretrained-model performance are not established here. |
| Anatomical labeling and registration | [brain_atlas_manager.py](python/brain_atlas_manager.py), [spatiotemporal_brain_registration.py](python/spatiotemporal_brain_registration.py) | Atlas and registration prototypes. [Blue Brain integration](python/blue_brain_atlas_integrator.py) includes mock-data fallbacks; demos do not establish real atlas coverage or improved registration accuracy. |
| 3D visualization | [brain_3d_visualizer.py](python/brain_3d_visualizer.py), [brain_3d_model_generator.py](python/brain_3d_model_generator.py) | Brain-model and connectivity visualization; some paths use mock coordinates when atlas loading is unavailable. |
| Streaming experiments | [realtime_fmri_processor.py](python/realtime_fmri_processor.py), [realtime_ai_roi_system.py](python/realtime_ai_roi_system.py) | Queue-based processing, simulated scanner data, and conceptual interfaces. Some motion, ROI, and utilization values are mocks. |
| GPU experiments | [src/](src/), [gpu_accelerated_analyzer.py](python/gpu_accelerated_analyzer.py), [enhanced_dual_gpu_analyzer.py](python/enhanced_dual_gpu_analyzer.py) | C++/OpenCL and Python acceleration experiments; availability depends on drivers, build configuration, and hardware. |

## Try the time-series analysis

Use a source checkout. The previous README's unified `realtime_fmri_ai` API and
installation examples are not a verified end-to-end interface for this tree.
The example below imports an existing module directly and uses synthetic data.
It does not validate an fMRI experiment or require a scanner, atlas download,
trained classifier, or GPU.

```bash
git clone https://github.com/IllI/resonance.git
cd resonance
python -m venv .venv
# Activate the environment using your platform's activation command.
python -m pip install numpy scipy pandas matplotlib networkx
```

From the repository root:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("python").resolve()))
from neural_network_analyzer import BrainNetworkAnalyzer

# Shape: timepoints x regions. These are synthetic signals, not subject data.
rng = np.random.default_rng(42)
time_series = rng.normal(size=(256, 12))

analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
analyzer.load_time_series(time_series)
connectivity = analyzer.compute_connectivity_matrix(method="pearson")
print(connectivity.shape)  # (12, 12)
```

Other modules have additional dependencies, listed in [requirements.txt](requirements.txt)
and [setup.py](setup.py). The OpenCL experiments have a separate
[CMake configuration](CMakeLists.txt) and require suitable drivers and development libraries.

## Tests and validation status

- [Core analysis tests](tests/test_core_fmri_analysis.py) use synthetic time series and check correlation structure and analysis behavior.
- [AI/ROI tests](tests/test_ai_models.py) use synthetic volumes and signal-processing features.
- [GPU tests](tests/test_gpu_acceleration.py) exercise hardware-dependent paths.

With the required dependencies installed:

```bash
python tests/test_core_fmri_analysis.py
python tests/test_ai_models.py
```

Inspect failures and skipped tests. Missing imports can cause tests to skip;
an exit code alone does not show that the intended functionality was exercised.
Synthetic tests and demonstrations are not validation against independently
labeled subject data or published neuroscience benchmarks.

Historical summaries in [reports/](reports/) and the project progress documents
are development records. Assess them against the exact source, dataset, hardware,
and raw output that produced them. This README does not assert that all tests currently pass.

## Performance status

The earlier speedup table, sub-100ms claims, and atlas-registration improvement
claim have been removed because this README does not provide a reproducible
measurement supporting them. No replacement performance numbers are asserted.

A reproducible benchmark should record the source commit, inputs, hardware and
drivers, preprocessing, warm-up and synchronization method, repeated timings,
and agreement with the CPU reference. End-to-end latency includes input handling
and transfers, not just kernel time.

## Research direction

The focus is exploring how brain-network signals change during an experimental
session and making those signals easier to inspect visually. Further work includes
independently labeled datasets, reproducible validation, clear separation of real
and mock outputs, model-training records, and measured backend comparisons.
Scanner deployment and clinical use require separate validation.

## Citation

For this software, cite the repository and record the commit you used. This is a
software citation, not a claim of a peer-reviewed paper or archived release:

```bibtex
@misc{oneill_resonance,
  author = {O'Neill, Kevin},
  title = {Resonance: fMRI Analysis and Visualization Research Prototypes},
  howpublished = {Source code},
  url = {https://github.com/IllI/resonance}
}
```

The package metadata declares MIT licensing in [setup.py](setup.py).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the existing development guide. Some
older documents describe proposed APIs and planned capabilities. When reporting
a result, include its inputs, source version, method, raw output, and limitations.
