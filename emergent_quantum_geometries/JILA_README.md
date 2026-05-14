# OAT Channel PTM Geometry Pipeline
### JILA Integration — Rey Theory Group

This repository contains the analysis pipeline for characterizing
**OAT boundary channel geometry** from experimental or simulated data.

> **New to this tool? Start here:**  
> **[Step-by-Step User Guide (with screenshots) →](docs/JILA_PIPELINE_USER_GUIDE.md)**

---

## What This Does

Given experimental data (atom number N, interaction time χt, dephasing γ),
the pipeline computes:

- **Full 3×3 correlation tensor** T_ij = Tr[ρ₂ (σᵢ ⊗ σⱼ)]
- **PTM singular values** and nuclear norm
- **Theoretical teleportation fidelity** F_avg via Horodecki (1999)
- **Phase classification**: QUANTUM / CLASSICAL / SINGULAR
- **Concurrence** and PTM rank

**No raw experimental data is transmitted or stored.**
You run this locally on your own hardware.

---

## Quick Start

### Option A — Graphical Interface (recommended for most users)

```bash
python jila_gui.py
```

This opens the point-and-click dashboard. Load your data file, click **Run Local**, and the results window appears with plots and download buttons. See the **[User Guide](docs/JILA_PIPELINE_USER_GUIDE.md)** for a full walkthrough with screenshots.

### Option B — Command line

```bash
pip install numpy scipy h5py
```

**Single point:**
```bash
python jila_pipeline.py --N 4 --chi_t 1.456 --gamma_t 0.0
```

**From a CSV file** (columns: N, chi_t, gamma_t):
```bash
python jila_pipeline.py --input your_clock_data.csv --out results.json
```

**From HDF5** (datasets: N, chi_t, gamma_t):
```bash
python jila_pipeline.py --input your_data.h5 --out results.json
```

**From measured T_xx** (Ramsey spectroscopy output):
```python
from jila_pipeline import run_pipeline
result = run_pipeline(N=4, chi_t=1.456, T_xx_measured=0.557, T_yz_measured=-0.497)
print(result['phase_class'])   # 'QUANTUM'
print(result['F_avg_theory'])  # 0.758
```

---

## Input Formats

| Format | Extension | Arrays/Columns needed |
|---|---|---|
| **NumPy archive** | **`.npz`** | **`N`, `chi_t`, `gamma_t` (optional)** |
| CSV | `.csv` | `N`, `chi_t`, `gamma_t` (optional) |
| HDF5 | `.h5` | Datasets `N`, `chi_t`, `gamma_t` |
| JSON | `.json` | List of `{N, chi_t, gamma_t}` dicts |
| Direct (Ramsey) | Python | `T_xx_measured`, `T_yz_measured` |

**Sample files for testing:** see the [`sample_data/`](sample_data/) folder.
Start with `sample_ideal_sweep.npz` — it runs in under 5 seconds locally.

---

## Key Results From This Paper

At χt* = 1.456, N = 4 (optimal operating point):

```
T_xx = 0.557    (XX boundary correlator)
T_yz = -0.497   (YZ cross-correlator — source of teleportation advantage)
Nuclear norm = 1.551
F_avg = 0.758 > 2/3  ← above classical teleportation threshold
Phase: QUANTUM
```

At χt = π (exact singular null):
```
ρ₂ = I/4 exactly (machine precision)
F_avg = 0.500 (every input state → maximally mixed)
Phase: SINGULAR
```

---

## Pre-Registered Predictions for Sr/Yb Clock Experiments

| Observable | χt ≈ 0 | χt* = 1.456 | χt = π |
|---|---|---|---|
| T_xx | 0.500 ± 0.015 | 0.557 ± 0.020 | 0.000 ± 0.005 |
| T_yz | ~0 | -0.497 ± 0.025 | 0.000 ± 0.005 |
| PTM rank | 1 | 4 | 1 |
| Phase | CLASSICAL | QUANTUM | SINGULAR |

---

## Requirements

```
numpy >= 1.21
scipy >= 1.7
h5py >= 3.0  (only for HDF5 input)
```

---

## Citation

[Paper in preparation] — Rey Theory Group collaboration
*Contact author before citing.*
