"""
jila_pipeline.py — OAT Channel Geometry Pipeline
JILA Rey Group Integration Adapter

Usage
-----
Feed your experimental or simulated (N, chi_t, gamma_t) data and get
back the full PTM geometry characterization without sharing raw data.

Quick start:
    from jila_pipeline import run_pipeline
    results = run_pipeline(N=4, chi_t=1.456, gamma_t=0.0)
    print(results)  # F_avg, T_matrix, phase_class, ...

Or from a data file:
    python jila_pipeline.py --input my_clock_data.csv --out results.json

Input formats accepted
----------------------
- CSV: columns [N, chi_t, gamma_t]  (gamma_t optional, defaults to 0)
- HDF5: datasets 'N', 'chi_t', 'gamma_t'
- JSON: list of {N, chi_t, gamma_t} dicts
- Direct: measured T_xx values (bypasses theoretical rho2 computation)
          useful if you have Ramsey spectroscopy data directly
"""

import numpy as np
import json
import argparse
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.linalg import eigvalsh

# ── Theoretical rho2 from Proposition 1 (OAT boundary state) ────────────────
def rho2_oat(chi_t: float, N: int) -> np.ndarray:
    """
    Boundary reduced density matrix of N-qubit OAT state at interaction
    time chi_t. Exact closed form from Proposition 1.

    Parameters
    ----------
    chi_t : float
        OAT interaction parameter (chi * t). Range [0, pi].
    N : int
        Total qubit number. Must be even. Boundary pair is qubits 0 and N-1.

    Returns
    -------
    rho : (4,4) complex ndarray
        Boundary 2-qubit density matrix in {|00>,|01>,|10>,|11>} basis.
    """
    m = N // 2 - 1
    B = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rho = np.zeros((4, 4), dtype=complex)
    for a, (iL, iR) in enumerate(B):
        for b, (jL, jR) in enumerate(B):
            phase = np.exp(1j * chi_t * ((iL-.5)*(iR-.5) - (jL-.5)*(jR-.5)))
            cR = np.cos((iR - jR) * chi_t / 2) ** m
            cL = np.cos((iL - jL) * chi_t / 2) ** m
            rho[a, b] = 0.25 * phase * cR * cL
    return rho


def rho2_from_Txx(T_xx: float, T_yz: float = 0.0) -> np.ndarray:
    """
    Reconstruct minimal rho2 from measured correlators.
    Use this if you have Ramsey spectroscopy data (T_xx) and optionally
    the YZ cross-correlator (T_yz).

    Note: This gives a valid 2-qubit state only if the measured values
    are consistent with a physical density matrix. The full PTM
    reconstruction may require additional correlator measurements.
    """
    rho = np.eye(4, dtype=complex) / 4
    # Add XX correlator
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    rho += T_xx / 4 * np.kron(sx, sx)
    rho += T_yz / 4 * np.kron(sy, sz)
    rho += T_yz / 4 * np.kron(sz, sy)
    return rho


def apply_dephasing(rho: np.ndarray, gamma_t: float) -> np.ndarray:
    """Apply boundary dephasing (e^{-2*gamma*|i-j|} damping)."""
    B = [(0,0),(0,1),(1,0),(1,1)]
    r = rho.copy()
    for a,(iL,iR) in enumerate(B):
        for b,(jL,jR) in enumerate(B):
            r[a,b] *= np.exp(-2*gamma_t*(int(iL!=jL) + int(iR!=jR)))
    return r


# ── Full T-matrix extraction ─────────────────────────────────────────────────
def t_matrix(rho: np.ndarray) -> np.ndarray:
    """
    Compute full 3x3 correlation tensor T_ij = Tr[rho (sigma_i ⊗ sigma_j)].

    Returns
    -------
    T : (3,3) real ndarray
        Row/col order: [X, Y, Z].
    """
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    paulis = [sx, sy, sz]
    T = np.zeros((3, 3))
    for i, si in enumerate(paulis):
        for j, sj in enumerate(paulis):
            T[i, j] = float(np.real(np.trace(np.kron(si, sj) @ rho)))
    return T


# ── Horodecki fidelity from nuclear norm ─────────────────────────────────────
def horodecki_fidelity(T: np.ndarray) -> float:
    """
    Maximum average teleportation fidelity via Horodecki (1999).
    F_max = (1 + nuclear_norm(T * M_{Phi+}) / 3) / 2
    where M_{Phi+} = diag(1, -1, 1).

    This is the theoretical upper bound; the explicit protocol achieves
    this when the optimizer finds the correct local SU(2)^2 rotation.
    """
    M = np.diag([1, -1, 1])
    TM = T @ M
    svs = np.linalg.svd(TM, compute_uv=False)
    f_max = (1 + np.sum(svs)) / 4
    return (2 * f_max + 1) / 3


def ptm_rank(T: np.ndarray, threshold: float = 0.01) -> int:
    """Effective rank of T-matrix above threshold."""
    svs = np.linalg.svd(T, compute_uv=False)
    return int(np.sum(svs > threshold))


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence of 2-qubit state."""
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy, sy)
    ev = np.sort(np.real(np.sqrt(np.maximum(
        eigvalsh(rho @ ss @ rho.conj() @ ss), 0))))[::-1]
    return max(0., float(ev[0] - ev[1] - ev[2] - ev[3]))


def phase_class(F_avg: float, T: np.ndarray) -> str:
    """
    Classify channel phase:
    - 'QUANTUM': F_avg > 2/3 (teleportation-capable)
    - 'SINGULAR': ρ = I/4 (completely depolarizing)
    - 'CLASSICAL': F_avg ≤ 2/3 but not singular
    """
    nuc = np.sum(np.linalg.svd(T, compute_uv=False))
    if nuc < 1e-10:
        return 'SINGULAR'
    elif F_avg > 2/3 + 0.001:
        return 'QUANTUM'
    else:
        return 'CLASSICAL'


# ── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(N: int, chi_t: float, gamma_t: float = 0.0,
                 T_xx_measured: float = None,
                 T_yz_measured: float = None) -> dict:
    """
    Run the full PTM geometry characterization pipeline.

    Parameters
    ----------
    N : int
        System size (number of qubits). Must be even, >= 4.
    chi_t : float
        OAT interaction parameter. Range [0, pi].
    gamma_t : float
        Dephasing rate * time. 0 = ideal.
    T_xx_measured : float, optional
        If provided, use measured T_xx instead of theoretical rho2.
        Useful for real experimental data from Ramsey spectroscopy.
    T_yz_measured : float, optional
        Measured YZ cross-correlator. Only used with T_xx_measured.

    Returns
    -------
    dict with keys:
        N, chi_t, gamma_t,
        T_matrix (3x3),
        T_xx, T_yz,
        nuclear_norm,
        F_avg_theory,
        ptm_rank,
        concurrence,
        phase_class
    """
    # Build resource state
    if T_xx_measured is not None:
        T_yz_val = T_yz_measured if T_yz_measured is not None else 0.0
        rho = rho2_from_Txx(T_xx_measured, T_yz_val)
    else:
        rho = rho2_oat(chi_t, N)
        if gamma_t > 0:
            rho = apply_dephasing(rho, gamma_t)

    # Characterize
    T = t_matrix(rho)
    svs = np.linalg.svd(T, compute_uv=False)
    nuc = float(np.sum(svs))
    F = horodecki_fidelity(T)
    C = concurrence(rho)
    rank = ptm_rank(T)
    pc = phase_class(F, T)

    return {
        "N": N,
        "chi_t": float(chi_t),
        "gamma_t": float(gamma_t),
        "T_matrix": T.tolist(),
        "T_xx": float(T[0, 0]),
        "T_yz": float(T[1, 2]),
        "T_singular_values": svs.tolist(),
        "nuclear_norm": nuc,
        "F_avg_theory": float(F),
        "ptm_rank": rank,
        "concurrence": float(C),
        "phase_class": pc,
        "above_classical": bool(F > 2/3),
    }


# ── Batch runner ─────────────────────────────────────────────────────────────
def run_batch(records: list) -> list:
    """
    Process a list of (N, chi_t, gamma_t) records.

    Parameters
    ----------
    records : list of dict
        Each dict must have keys 'N', 'chi_t'.
        Optional: 'gamma_t', 'T_xx_measured', 'T_yz_measured'.

    Returns
    -------
    list of result dicts (same length as input).
    """
    results = []
    for i, rec in enumerate(records):
        try:
            r = run_pipeline(
                N=int(rec['N']),
                chi_t=float(rec['chi_t']),
                gamma_t=float(rec.get('gamma_t', 0.0)),
                T_xx_measured=rec.get('T_xx_measured'),
                T_yz_measured=rec.get('T_yz_measured'),
            )
            results.append(r)
        except Exception as e:
            results.append({"error": str(e), "input": rec})
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(records)}", flush=True)
    return results


# ── File ingestion ────────────────────────────────────────────────────────────
def load_input(path: str) -> list:
    """Load CSV, JSON, or HDF5 input file."""
    if path.endswith('.csv'):
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        import h5py
        records = []
        with h5py.File(path, 'r') as f:
            N_arr = np.array(f['N'])
            chi_arr = np.array(f['chi_t'])
            gam_arr = np.array(f.get('gamma_t', np.zeros_like(chi_arr)))
            for N, chi, gam in zip(N_arr, chi_arr, gam_arr):
                records.append({'N': N, 'chi_t': chi, 'gamma_t': gam})
        return records
    else:
        raise ValueError(f"Unsupported format: {path}. Use .csv, .json, or .h5")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OAT Channel PTM Geometry Pipeline — JILA Integration')
    parser.add_argument('--input',  help='Input file (.csv, .json, .h5)')
    parser.add_argument('--out',    default='ptm_results.json',
                        help='Output JSON file')
    parser.add_argument('--N',      type=int,   default=4)
    parser.add_argument('--chi_t',  type=float, default=1.456)
    parser.add_argument('--gamma_t',type=float, default=0.0)
    args = parser.parse_args()

    if args.input:
        print(f"Loading {args.input}...", flush=True)
        records = load_input(args.input)
        print(f"Running pipeline on {len(records)} points...", flush=True)
        results = run_batch(records)
    else:
        print(f"Single point: N={args.N}, chi_t={args.chi_t}, gamma_t={args.gamma_t}")
        results = run_pipeline(args.N, args.chi_t, args.gamma_t)
        results = [results]

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out} ({len(results)} results)")

    # Print summary
    if len(results) > 0:
        r0 = results[0]
        if 'error' not in r0:
            print(f"\nSample result:")
            print(f"  T_xx = {r0['T_xx']:.4f}  T_yz = {r0['T_yz']:.4f}")
            print(f"  Nuclear norm = {r0['nuclear_norm']:.4f}")
            print(f"  F_avg (theory) = {r0['F_avg_theory']:.4f}")
            print(f"  PTM rank = {r0['ptm_rank']}")
            print(f"  Phase: {r0['phase_class']}")
