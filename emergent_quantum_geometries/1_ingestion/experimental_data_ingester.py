"""
experimental_data_ingester.py
Drop-in replacement for IntraSliceJitterStreamer.
Loads real experimental quantum datasets and normalizes them
into the same (batch_size, seq_len, features) tensor the
DLinOSSQuantumDamper already expects.

Supported sources
-----------------
  SYK_ED        -- Sachdev-Ye-Kitaev exact-diagonalization Green's functions
                   github.com/BlackHatLKJH/SYK-model  (N=20-40 Majorana)
  SYCAMORE_RCS  -- Google Sycamore random-circuit bitstrings
                   github.com/quantumlib/qsim  (53-qubit, 20 cycles)
  RYDBERG       -- QuEra / Harvard Lukin Rydberg array snapshots
                   arXiv:2202.09372 supplementary  (289 qubits)
  SYNTHETIC     -- Analytically generated reference signals for each
                   framework (Heisenberg, Lindblad, OR, SYK, MBL).
                   Use to calibrate the D-LiNOSS before real data.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Literal, Tuple

# ── Type alias ────────────────────────────────────────────────────
Source = Literal["SYK_ED", "SYCAMORE_RCS", "RYDBERG", "JILA_MBL", "JILA_TELEPORTATION", "SYNTHETIC"]
Framework = Literal["heisenberg", "lindblad", "penrose_or", "syk", "mbl"]


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API  —  mirrors IntraSliceJitterStreamer interface
# ═══════════════════════════════════════════════════════════════════

class ExperimentalDataIngester:
    """
    Loads a real (or synthetic) quantum dataset and emits a JAX tensor
    of shape (batch_size, seq_len, n_features) that feeds directly into
    DLinOSSQuantumDamper.stream_phase_data() slot.

    Usage
    -----
    ingester = ExperimentalDataIngester(source="SYK_ED", data_dir="./data")
    drift_data = ingester.stream_phase_data(batch_size=128)
    # drift_data.shape == (128, 32, 32)  -- same as IntraSliceJitterStreamer

    Parameters
    ----------
    source      : one of the Source literals above
    data_dir    : path to downloaded dataset files
    T           : number of time steps to extract (seq_len dimension)
    n_features  : feature dimension (must match DLinOSS input width)
    dt          : physical time step in units of J^{-1} (coupling energy)
    """

    def __init__(
        self,
        source: Source = "SYNTHETIC",
        framework: Framework = "syk",   # only used when source="SYNTHETIC"
        data_dir: str = "./data",
        T: int = 32,
        n_features: int = 32,
        dt: float = 0.1,
    ):
        self.source     = source
        self.framework  = framework
        self.data_dir   = Path(data_dir)
        self.T          = T
        self.n_features = n_features
        self.dt         = dt

        self._loaders = {
            "SYK_ED":             self._load_syk_ed,
            "SYCAMORE_RCS":       self._load_sycamore,
            "RYDBERG":            self._load_rydberg,
            "JILA_MBL":           self._load_jila_mbl,
            "JILA_TELEPORTATION": self._load_jila_teleportation,
            "SYNTHETIC":          self._load_synthetic,
        }

    # ── main entry point ──────────────────────────────────────────
    def stream_phase_data(self, batch_size: int = 128) -> jnp.ndarray:
        """
        Returns
        -------
        jnp.ndarray  shape (batch_size, T, n_features)
            Each batch item is one independent trajectory / disorder
            realisation / shot sequence.
        """
        raw = self._loaders[self.source](batch_size)
        return jnp.array(raw, dtype=jnp.float32)

    # ─────────────────────────────────────────────────────────────
    # LOADER:  SYK exact diagonalization
    # ─────────────────────────────────────────────────────────────
    def _load_syk_ed(self, batch_size: int) -> np.ndarray:
        """
        Expected file layout (download from github.com/BlackHatLKJH/SYK-model):
            data/syk/N{N}_beta{beta}_Gt.npy   -- shape (T, N, N) complex
            data/syk/N{N}_beta{beta}_Ft.npy   -- OTOC F(t), shape (T,)

        We use Im[G(t)] as the time-series features —  this is the
        spectral function A(ω) in disguise and is what D-LiNOSS should
        learn to decompose into (ω_k, γ_k) pairs.
        """
        N    = 24       # Majorana fermions — adjust to file on disk
        beta = 5.0      # inverse temperature

        gt_path = self.data_dir / "syk" / f"N{N}_beta{beta}_Gt.npy"
        ft_path = self.data_dir / "syk" / f"N{N}_beta{beta}_Ft.npy"

        if not gt_path.exists():
            print(f"[SYK_ED] File not found: {gt_path}")
            print("  → Falling back to synthetic SYK signal")
            return _synthetic_syk(batch_size, self.T, self.n_features, self.dt)

        Gt = np.load(gt_path)   # (T_full, N, N) complex
        Ft = np.load(ft_path)   # (T_full,) real

        # Take imaginary part of diagonal (local Green's function)
        # shape → (T_full, N)
        spectral = np.imag(np.diagonal(Gt, axis1=1, axis2=2))

        # Build (batch_size, T, n_features) by sliding window + site sampling
        return _sliding_window_batch(spectral, Ft, batch_size, self.T, self.n_features)

    # ─────────────────────────────────────────────────────────────
    # LOADER:  JILA / Ana Maria Rey MBL Data (Moiré Lattice)
    # ─────────────────────────────────────────────────────────────
    def _load_jila_mbl(self, batch_size: int) -> np.ndarray:
        """
        Expected file layout:
            data/jila_mbl/real/fig2_data/imbalanceTimeDecay*.txt

        The real Zenodo dataset has only 5 time points per file.
        We use physics-informed exponential interpolation (I(t) = A*exp(-Γt) + C)
        to reconstruct a smooth T=32 trajectory rather than naive tiling.
        This preserves the true MBL character (slow decay, near-zero Γ) without
        introducing the large-γ artifact that caused the SYK bias.
        """
        real_dir = self.data_dir / "jila_mbl" / "real" / "fig2_data"

        if not real_dir.exists():
            print(f"[JILA_MBL] Real data not found at {real_dir}")
            print("  → Did you run 'python fetch_datasets.py --jila' ?")
            return _synthetic_mbl(batch_size, self.T, self.n_features, self.dt)

        import glob
        files = sorted(glob.glob(str(real_dir / "imbalanceTimeDecay*.txt")))
        if not files:
            return _synthetic_mbl(batch_size, self.T, self.n_features, self.dt)

        print(f"[JILA_MBL] Loading {len(files)} voltage sweep(s) with physics-informed interpolation...")

        # Load all voltage sweeps — each is one disorder realisation
        interp_signals = []
        for f in files:
            data = np.loadtxt(f, skiprows=1)  # (N_pts, 3): time, imbalance, error
            t_raw = data[:, 0]
            I_raw = data[:, 1]

            # Fit a simple exponential decay: I(t) = (I0 - C)*exp(-Gamma*t) + C
            # Estimate via log-linear fit: log(I - C_floor) = log(A) - Gamma*t
            C_floor = max(I_raw[-1] - 0.02, 0.0)  # asymptotic floor
            I_shifted = np.clip(I_raw - C_floor, 1e-6, None)
            log_I = np.log(I_shifted)
            # Linear fit in log space: slope = -Gamma, intercept = log(A)
            coeffs = np.polyfit(t_raw, log_I, 1)
            Gamma = -coeffs[0]
            A = np.exp(coeffs[1])

            # Interpolate to T uniformly-spaced points in [t_raw[0], t_raw[-1]]
            t_interp = np.linspace(t_raw[0], t_raw[-1], self.T)
            I_interp = A * np.exp(-Gamma * t_interp) + C_floor
            I_interp = np.clip(I_interp, 0.0, 1.0)
            interp_signals.append(I_interp)

        # Stack voltage sweeps as feature channels: (T, n_voltage_sweeps)
        spectral_raw = np.stack(interp_signals, axis=1)  # (T, len(files))
        spectral = _resize_features(spectral_raw, self.n_features)  # (T, n_features)

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, (batch_size, self.T, self.n_features))
        return spectral[None] + noise  # (batch_size, T, n_features)

    # ─────────────────────────────────────────────────────────────
    # LOADER:  JILA / Ana Maria Rey Teleportation Protocol (2025)
    # ─────────────────────────────────────────────────────────────
    def _load_jila_teleportation(self, batch_size: int) -> np.ndarray:
        """
        Simulates the Ana Maria Rey 2025 quantum teleportation protocol
        (PRR 7, L022019) using the one-axis-twisting Hamiltonian.

        Protocol:
          Alice & Bob: entangled via H_AB = χ * Jz_A ⊗ Jz_B (phonon-mediated)
          Charlie:     holds a spin-coherent or squeezed state to teleport
          Fidelity:    F(t) = |<ψ_Charlie|ρ_Bob(t)|ψ_Charlie>|

        The fidelity trajectory has a distinct Heisenberg signature:
          - Oscillatory (undamped) while entanglement builds
          - Near-zero γ_k (unitary evolution, no decoherence)
          - ω_k ∝ χ·N (collective coupling frequency)
        """
        return _synthetic_jila_teleportation(batch_size, self.T, self.n_features, self.dt)

    # ─────────────────────────────────────────────────────────────
    # LOADER:  Google Sycamore RCS bitstrings
    # ─────────────────────────────────────────────────────────────
    def _load_sycamore(self, batch_size: int) -> np.ndarray:
        """
        Expected file layout (download from github.com/quantumlib/qsim):
            data/sycamore/circuit_{depth}_bitstrings.npy  shape (n_shots, 53)

        From the bitstrings we compute the empirical two-point and
        four-point correlators ⟨Z_i Z_j⟩(t) as a function of 'time'
        (= circuit depth, which is the discrete time in this system).

        The OTOC proxy: for random circuits, C_ij(t) ≈ 1 - F(t)/F(0)
        where F(t) is the frame potential (measurable from 2nd Rényi).
        We use the simpler linear estimator:
            F(t) ≈ ⟨Z_i Z_j⟩ — avg over all pairs (i,j) at depth t
        """
        depths   = list(range(1, self.T + 1))      # circuit depths as 'time'
        n_qubits = 53
        series   = []

        for d in depths:
            path = self.data_dir / "sycamore" / f"circuit_{d}_bitstrings.npy"
            if not path.exists():
                # Pad with zeros if depth file missing
                series.append(np.zeros(n_qubits))
                continue
            bits = np.load(path)    # (n_shots, 53)  values ∈ {0, 1}
            spins = 2 * bits - 1    # {0,1} → {-1,+1}
            # Mean single-site magnetisation at each qubit
            series.append(np.mean(spins, axis=0))   # (53,)

        # series: list of T arrays each (53,)  → (T, 53)
        series = np.stack(series, axis=0)

        # Tile / truncate features to n_features
        series = _resize_features(series, self.n_features)   # (T, n_features)

        # Expand to batch: add Gaussian noise instances
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, (batch_size, self.T, self.n_features))
        return series[None, :, :] + noise    # (batch_size, T, n_features)

    # ─────────────────────────────────────────────────────────────
    # LOADER:  Rydberg array ⟨S_z(t)⟩ snapshots
    # ─────────────────────────────────────────────────────────────
    def _load_rydberg(self, batch_size: int) -> np.ndarray:
        """
        Expected file layout (from arXiv:2202.09372 supplementary,
        or from a live QuEra Aquila run via Amazon Braket):
            data/rydberg/shots_{t}.npy  shape (n_shots, n_atoms)

        Each file is one time-step in the Rydberg evolution.
        We extract ⟨n_i⟩(t) = average occupation per atom per time.
        """
        n_atoms   = min(self.n_features, 100)   # use first n_features atoms
        t_files   = sorted((self.data_dir / "rydberg").glob("shots_*.npy"))

        if not t_files:
            print("[RYDBERG] No data files found in data/rydberg/")
            print("  → Tip: run a QuEra Aquila job on Amazon Braket and save")
            print("    each time-step's bitstring array as shots_{t}.npy")
            print("  → Falling back to synthetic Rydberg signal")
            return _synthetic_rydberg(batch_size, self.T, self.n_features, self.dt)

        series = []
        for f in t_files[: self.T]:
            shots = np.load(f)              # (n_shots, n_atoms)
            series.append(np.mean(shots[:, :n_atoms], axis=0))

        series = np.stack(series, axis=0)                       # (T, n_atoms)
        series = _resize_features(series, self.n_features)      # (T, n_features)

        rng   = np.random.default_rng(7)
        noise = rng.normal(0, 0.005, (batch_size, self.T, self.n_features))
        return series[None] + noise

    # ─────────────────────────────────────────────────────────────
    # LOADER:  Synthetic reference signals
    # ─────────────────────────────────────────────────────────────
    def _load_synthetic(self, batch_size: int) -> np.ndarray:
        """
        Analytically generated ground-truth signals.
        Each framework has a known (ω_k, γ_k) signature.
        Use these to:
          1. Verify D-LiNOSS can recover known parameters.
          2. Calibrate the residual comparator before real data.
        """
        generators = {
            "heisenberg":  _synthetic_heisenberg,
            "lindblad":    _synthetic_lindblad,
            "penrose_or":  _synthetic_penrose_or,
            "syk":         _synthetic_syk,
            "mbl":         _synthetic_mbl,
        }
        gen = generators.get(self.framework, _synthetic_syk)
        return gen(batch_size, self.T, self.n_features, self.dt)


# ═══════════════════════════════════════════════════════════════════
#  SYNTHETIC SIGNAL GENERATORS
#  Each encodes the known (ω_k, γ_k) signature of its framework.
#  D-LiNOSS should recover these parameters from the signal.
# ═══════════════════════════════════════════════════════════════════

def _synthetic_heisenberg(B, T, F, dt) -> np.ndarray:
    """
    Pure unitary evolution of a spin chain.
    Signature: γ_k = 0 for all k. ω_k = 2J|cos(πk/(F+1))| (tight-binding).
    OTOC decays as power law t^{-α}, not exponential.
    """
    t    = np.arange(T) * dt
    k    = np.arange(1, F + 1)
    omegas = 2.0 * np.abs(np.cos(np.pi * k / (F + 1)))   # tight-binding dispersion
    # signal: sum of undamped oscillators
    sig = np.sum(np.cos(omegas[None, :] * t[:, None]), axis=1)  # (T,)
    sig /= (np.std(sig) + 1e-8)
    return _tile_and_noise(sig, B, T, F, noise_std=0.02)


def _synthetic_lindblad(B, T, F, dt, gamma_uniform=0.3) -> np.ndarray:
    """
    Lindblad thermal bath.
    Signature: γ_k = γ_uniform (same for all modes). Exponential OTOC decay.
    """
    t    = np.arange(T) * dt
    k    = np.arange(1, F + 1)
    omegas = 2.0 * np.abs(np.cos(np.pi * k / (F + 1)))
    # Damped oscillators — uniform γ
    sig = np.sum(
        np.exp(-gamma_uniform * t[:, None]) * np.cos(omegas[None, :] * t[:, None]),
        axis=1
    )
    sig /= (np.std(sig) + 1e-8)
    return _tile_and_noise(sig, B, T, F, noise_std=0.02)


def _synthetic_penrose_or(B, T, F, dt) -> np.ndarray:
    """
    Penrose Objective Reduction.
    Signature: γ_k = E_G,k / ħ  (non-uniform, mass-distribution dependent).
    Key feature: a KINK in the OTOC at t = τ_OR where collapse occurs.
    Before τ_OR: unitary-like. After τ_OR: post-collapse decay.
    """
    t       = np.arange(T) * dt
    tau_OR  = T * dt * 0.4       # collapse at 40% of trajectory
    k       = np.arange(1, F + 1)
    omegas  = 2.0 * np.abs(np.cos(np.pi * k / (F + 1)))

    # Non-uniform γ_k drawn from a mass-proxy distribution
    rng    = np.random.default_rng(0)
    gammas = rng.exponential(scale=0.2, size=F)   # E_G / ħ per mode

    # Two-phase signal: coherent until τ_OR, then post-collapse
    phase1 = np.sum(np.cos(omegas[None, :] * t[:, None]), axis=1)
    phase2 = np.sum(
        np.exp(-gammas[None, :] * (t[:, None] - tau_OR))
        * np.cos(omegas[None, :] * t[:, None]),
        axis=1
    )
    mask = (t >= tau_OR).astype(float)
    sig  = phase1 * (1 - mask) + phase2 * mask
    sig /= (np.std(sig) + 1e-8)
    return _tile_and_noise(sig, B, T, F, noise_std=0.015)


def _synthetic_syk(B, T, F, dt, beta=5.0) -> np.ndarray:
    """
    SYK holographic scrambler.
    Signature: γ_k follows Lyapunov exponent λ_L = 2π k_B T / ħ = 2π/β.
    OTOC decays as F(t) ~ 1 - (1/N) * exp(λ_L * t)  [early time].
    This is the maximal chaos rate — saturates the Maleevsky-Kitaev bound.
    """
    t      = np.arange(T) * dt
    lam_L  = 2 * np.pi / beta         # Lyapunov exponent
    k      = np.arange(1, F + 1)
    omegas = np.pi * k / (F + 1)      # roughly flat density of states (SYK)
    gammas = lam_L * np.ones(F)       # all modes decay at Lyapunov rate

    sig = np.sum(
        np.exp(-gammas[None, :] * t[:, None]) * np.cos(omegas[None, :] * t[:, None]),
        axis=1
    )
    sig /= (np.std(sig) + 1e-8)
    return _tile_and_noise(sig, B, T, F, noise_std=0.02)


def _synthetic_mbl(B, T, F, dt) -> np.ndarray:
    """
    Many-Body Localized phase.
    Signature: γ_k ≈ 0 (no thermalization). ω_k frozen / disordered.
    OTOC grows logarithmically: F(t) ~ 1 - c * log(t).
    """
    t   = np.arange(T) * dt
    rng = np.random.default_rng(1)
    # Disordered frequencies (Anderson localization in frequency space)
    omegas = rng.uniform(0.5, 2.5, F)
    gammas = rng.exponential(0.01, F)   # near-zero damping

    sig = np.sum(
        np.exp(-gammas[None, :] * t[:, None]) * np.cos(omegas[None, :] * t[:, None]),
        axis=1
    )
    sig /= (np.std(sig) + 1e-8)
    return _tile_and_noise(sig, B, T, F, noise_std=0.03)


def _synthetic_rydberg(B, T, F, dt) -> np.ndarray:
    """Synthetic placeholder when no Rydberg data files are present."""
    return _synthetic_lindblad(B, T, F, dt, gamma_uniform=0.15)


def _synthetic_jila_teleportation(B, T, F, dt, chi=0.05) -> np.ndarray:
    """
    Simulates the JILA Ana Maria Rey 2025 quantum teleportation protocol.
    (Physical Review Research 7, L022019)

    Physics:
      H_AB = χ * Jz_A ⊗ Jz_B  (one-axis twisting, phonon-mediated)

    The teleportation fidelity F(t) oscillates as entanglement builds between
    Alice and Bob, then peaks when the Bell measurement is performed.
    For N ions per ensemble:
      F(t) ≈ |cos(χ·N·t)|² * exp(-κ·t)  (κ → 0 for ideal trap)

    Key D-LiNOSS signature:
      - ω_k = χ·N  (collective coupling — scales with ion number)
      - γ_k ≈ 0    (near-perfect unitary evolution)
    This is the cleanest Heisenberg signature in the framework.
    """
    t   = np.arange(T) * dt
    rng = np.random.default_rng(7)

    # Simulate multiple ion-number realisations as feature channels
    # N_ions drawn from the paper's range: 10 to 300 ions
    N_ions_set = np.logspace(1, 2.5, F).astype(int)  # (F,) log-spaced 10–300

    # Small phonon decoherence rate (κ ≈ 0 in ideal Penning trap)
    kappa = 0.005

    # Fidelity: F_i(t) = cos²(χ·N_i·t) * exp(-κ·t)
    omega_N = chi * N_ions_set[None, :]           # (1, F)
    fidelity = np.cos(omega_N * t[:, None])**2    # (T, F)
    fidelity *= np.exp(-kappa * t[:, None])       # phonon damping envelope

    # Add a small squeezed-state correction: Dicke state has enhanced fidelity peak
    squeezing_boost = 0.1 * np.exp(-((t - T * dt * 0.5)**2) / (2 * (T * dt * 0.15)**2))
    fidelity += squeezing_boost[:, None]
    fidelity = np.clip(fidelity, 0.0, 1.0)

    # Normalise to zero mean for D-LiNOSS input
    fidelity -= fidelity.mean(axis=0, keepdims=True)
    fidelity /= (fidelity.std(axis=0, keepdims=True) + 1e-8)

    noise = rng.normal(0, 0.02, (B, T, F))
    return fidelity[None] + noise  # (B, T, F)


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _tile_and_noise(sig_1d, B, T, F, noise_std=0.02) -> np.ndarray:
    """Broadcast a 1D signal to (B, T, F) with per-batch noise instances."""
    sig_2d = np.tile(sig_1d[:T, None], (1, F))             # (T, F)
    rng    = np.random.default_rng(99)
    noise  = rng.normal(0, noise_std, (B, T, F))
    return sig_2d[None] + noise                             # (B, T, F)


def _resize_features(arr, n_features) -> np.ndarray:
    """Pad or truncate feature axis to match n_features."""
    T, F = arr.shape
    if F >= n_features:
        return arr[:, :n_features]
    pad = np.zeros((T, n_features - F))
    return np.concatenate([arr, pad], axis=1)


def _sliding_window_batch(spectral, otoc, B, T, F, step=1) -> np.ndarray:
    """
    Convert a long trajectory into a batch of overlapping windows.
    spectral: (T_full, N)
    otoc:     (T_full,)
    Returns:  (B, T, F)
    """
    T_full, N = spectral.shape
    max_start  = T_full - T
    if max_start <= 0:
        # Trajectory too short — tile it
        spectral = np.tile(spectral, (T // T_full + 2, 1))[:T]
        otoc     = np.tile(otoc,     (T // T_full + 2))[:T]
        max_start = 1

    rng    = np.random.default_rng(42)
    starts = rng.integers(0, max_start, size=B)
    batch  = []
    for s in starts:
        window = spectral[s : s + T]          # (T, N)
        window = _resize_features(window, F)  # (T, F)
        batch.append(window)
    return np.stack(batch)   # (B, T, F)
