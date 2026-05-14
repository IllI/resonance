# Sample Data Files

This folder contains ready-to-use sample files for testing the JILA Pipeline GUI.
Load any of them by clicking **"Browse / Drop File"** in the application.

> **Which format should I use?**  
> If your analysis pipeline is Python-based (which is typical for JILA/NIST labs running
> NumPy-heavy post-processing), use the **`.npz` files** — they load faster and preserve
> floating-point precision exactly. CSVs are provided as a universal fallback.

---

## Recommended: NumPy `.npz` Files

These are the native format for Python-based experimental pipelines.
A `.npz` file is a compressed archive of NumPy arrays — the same object you get from
`np.savez(...)` after a parameter sweep.

### `sample_ideal_sweep.npz` ← **Start here**

**20-point χt sweep, no decoherence (γ = 0), N = 4 atoms**

Mirrors the kind of output you would get from an ideal simulation or a very clean
experimental run. Equivalent Python code that generates this file:

```python
import numpy as np

chi_vals = np.linspace(0.05, np.pi, 20)
np.savez("my_oat_run.npz",
         N       = np.full(20, 4),
         chi_t   = chi_vals,
         gamma_t = np.zeros(20))
```

**What to expect when you load this:**
- Peak fidelity F ≈ 0.769 at χt ≈ 1.456
- 15 of 20 points classified **QUANTUM**
- The point at χt = π classifies as **SINGULAR** (maximally mixed state)

---

### `sample_with_decoherence.npz`

**Same sweep, γt = 0.05 (5% decoherence)**

Simulates realistic lab conditions — laser phase noise, trap heating, or finite
detection efficiency all appear as an effective decoherence in γt.

**What to expect:**
- Peak F ≈ 0.734 (lower than ideal)
- 12 of 20 points classified **QUANTUM** (narrower quantum window)
- Useful for comparing ideal vs. realistic conditions side-by-side

---

### `sample_phase_diagram_2d.npz`

**50-point 2D grid: 10 χt values × 5 γt values**

This is the most realistic sample — it mirrors a full 2D phase diagram sweep
where you step both the squeezing strength (χt) and the decoherence rate (γt).

```
chi_t:   10 values from 0.1 → 3.1
gamma_t:  5 values from 0.0 → 0.15
Total:   50 (chi_t, gamma_t) pairs
```

The pipeline flattens the grid and processes all 50 points. The results window
will show the full phase boundary across both dimensions.

**What to expect:**
- 24 of 50 points classified QUANTUM
- Peak F ≈ 0.770

---

## CSV Fallbacks

If you need to open or edit the data in Excel or a text editor, equivalent `.csv`
versions are provided:

| File | Description |
|------|-------------|
| `sample_ideal_sweep.csv` | Ideal sweep, same data as the `.npz` version |
| `sample_with_decoherence.csv` | Decoherence sweep, same data |
| `sample_nist_format.csv` | NIST Penning trap style columns (`Omega_chi`, `N_atoms`, etc.) — demonstrates custom column mapping |

---

## Using Your Own `.npz` File

The pipeline expects arrays named `N`, `chi_t`, and optionally `gamma_t`.
If your file uses different array names, use the column mapping dropdowns in the GUI.

**Minimal example:**
```python
import numpy as np

# Your experimental data
chi_values  = np.array([0.5, 1.0, 1.456, 2.0, 2.5])   # your squeezing parameter
atom_number = np.full(5, 100)                            # your atom count

np.savez("my_experiment.npz",
         N     = atom_number,
         chi_t = chi_values)
# gamma_t is optional — leave it out if you don't have a decoherence estimate
```

Then load `my_experiment.npz` in the GUI and click **Run Local**.

---

## Expected Results at a Glance

| File | Format | Points | Peak F | QUANTUM pts |
|------|--------|--------|--------|-------------|
| `sample_ideal_sweep.npz` | `.npz` | 20 | 0.769 | 15 |
| `sample_with_decoherence.npz` | `.npz` | 20 | 0.734 | 12 |
| `sample_phase_diagram_2d.npz` | `.npz` | 50 | 0.770 | 24 |
| `sample_ideal_sweep.csv` | `.csv` | 20 | 0.769 | 15 |
| `sample_nist_format.csv` | `.csv` | 20 | 0.769 | 15 |
