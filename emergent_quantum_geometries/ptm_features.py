"""
ptm_features.py — Canonical feature extractor for OAT boundary channels
========================================================================
Computes the full geometric feature vector from a density matrix or T-matrix:

  X = (T_ij, σ_k(T), ∂T_ij/∂χt, λ_k(PTM), V_Q, κ_Q, rank, f_max, F_avg)

Key insight from Session 2:
  f_max = 1/4 + (σ_1 + σ_2 + σ_3) / 4   ← uses ALL three singular values
  NOT the Horodecki one-SV formula (1 + σ_max)/4

This is consistent with F_avg=0.758 from TPU simulation and analytic T computation.

Usage:
  from ptm_features import PTMFeatures
  feat = PTMFeatures(N=4)
  X = feat.compute(chi_t, Gamma=0.0)
  corpus = feat.sweep(chi_t_grid, Gamma_grid)
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, asdict
from typing import Optional


# ── rho_2 from Proposition 1 (Exact closed form) ─────────────────────────────
def rho2_exact(chi_t: float, N: int, Gamma: float = 0.0) -> np.ndarray:
    """
    Exact boundary density matrix from Proposition 1.
    Dephasing applied analytically: rho_ij *= exp(-2*Gamma*d_H(i,j)*t)
    where t is absorbed into chi_t (chi_t = chi * t).
    For dephasing at fixed chi_t: Gamma_eff = Gamma / chi  (dimensionless).
    Convention: Gamma is the dephasing rate in units where chi=1.
    """
    m = N // 2 - 1
    B = [(0, 0), (0, 1), (1, 0), (1, 1)]
    r = np.zeros((4, 4), dtype=complex)
    for a, (iL, iR) in enumerate(B):
        for b, (jL, jR) in enumerate(B):
            phase = np.exp(-1j * chi_t * (
                (iL - 0.5) * (iR - 0.5) - (jL - 0.5) * (jR - 0.5)
            ))
            cosL = np.cos((iL - jL) * chi_t / 2) ** m
            cosR = np.cos((iR - jR) * chi_t / 2) ** m
            # Dephasing: Hamming distance between (iL,iR) and (jL,jR)
            d_H = int(iL != jL) + int(iR != jR)
            decay = np.exp(-2 * Gamma * d_H)
            r[a, b] = 0.25 * phase * cosL * cosR * decay
    return r


# ── T-matrix ──────────────────────────────────────────────────────────────────
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_sx, _sy, _sz]
_PAULI_NAMES = ["x", "y", "z"]


def compute_T_matrix(rho: np.ndarray) -> np.ndarray:
    """Full 3x3 PTM correlation tensor T_ij = Tr[rho (sigma_i x sigma_j)]."""
    T = np.zeros((3, 3))
    for i, si in enumerate(_PAULIS):
        for j, sj in enumerate(_PAULIS):
            T[i, j] = float(np.real(np.trace(np.kron(si, sj) @ rho)))
    return T


def compute_ptm_eigenvalues(rho: np.ndarray) -> np.ndarray:
    """
    4 eigenvalues of the Choi matrix / PTM structure.
    For a 2-qubit state rho, these are the eigenvalues of rho itself,
    which determine the channel rank and mixedness.
    """
    return np.sort(np.linalg.eigvalsh(rho))[::-1]  # descending


def f_max_analytic(T: np.ndarray) -> float:
    """
    Correct maximum singlet fraction using ALL three singular values of T.
    For |Psi+> = (|01>+|10>)/sqrt(2):
      f_max = 1/4 + (sigma_1 + sigma_2 + sigma_3) / 4
    This exploits the full T_yz off-diagonal structure via SO(3)xSO(3) rotation.
    """
    svs = np.linalg.svd(T, compute_uv=False)
    return 0.25 + float(np.sum(svs)) / 4.0


def T_matrix_rank(T: np.ndarray, threshold: float = 1e-3) -> int:
    """Numerical rank of T-matrix."""
    return int(np.sum(np.linalg.svd(T, compute_uv=False) > threshold))


# ── V_Q (Recovery Basin Volume) ───────────────────────────────────────────────
def _su2(p: np.ndarray) -> np.ndarray:
    """SU(2) from Euler angles (alpha, beta, gamma)."""
    a, b, g = p
    Rz1 = np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]])
    Ry = np.array([[np.cos(b / 2), -np.sin(b / 2)],
                   [np.sin(b / 2),  np.cos(b / 2)]])
    Rz2 = np.array([[np.exp(-1j * g / 2), 0], [0, np.exp(1j * g / 2)]])
    return Rz1 @ Ry @ Rz2


_PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)


def _singlet_frac(p: np.ndarray, rho: np.ndarray) -> float:
    U = np.kron(_su2(p[:3]), _su2(p[3:]))
    r = U @ rho @ U.conj().T
    return float(np.real(_PSI_P.conj() @ r @ _PSI_P))


def compute_V_Q(rho: np.ndarray, n_samples: int = 200,
                threshold: float = 2 / 3, seed: int = 0) -> float:
    """
    Fraction of Haar-random SU(2)^2 unitaries achieving F_avg > threshold.
    Each unitary parameterized by 6 Euler angles.
    """
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_samples):
        p = rng.uniform(-np.pi, np.pi, 6)
        f = _singlet_frac(p, rho)
        F = (2 * f + 1) / 3
        if F > threshold:
            count += 1
    return count / n_samples


def compute_kappa_Q(rho: np.ndarray, n_restarts: int = 8) -> float:
    """
    kappa_Q = -Tr(H) where H is Hessian of F_avg at the optimal unitary.
    Negative trace = sum of negative Hessian eigenvalues = curvature stiffness.
    Estimated numerically via finite differences around Powell optimum.
    """
    # Find optimum
    best_f = 0.0
    best_p = np.zeros(6)
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        p0 = rng.uniform(-np.pi, np.pi, 6)
        res = minimize(lambda p: -_singlet_frac(p, rho), p0,
                       method="Powell",
                       options={"ftol": 1e-12, "maxiter": 5000})
        if -res.fun > best_f:
            best_f = -res.fun
            best_p = res.x

    # Numerical Hessian via central differences
    eps = 1e-4
    n = 6
    H = np.zeros((n, n))
    f0 = _singlet_frac(best_p, rho)
    for i in range(n):
        for j in range(n):
            pp = best_p.copy(); pp[i] += eps; pp[j] += eps
            pm = best_p.copy(); pm[i] += eps; pm[j] -= eps
            mp = best_p.copy(); mp[i] -= eps; mp[j] += eps
            mm = best_p.copy(); mm[i] -= eps; mm[j] -= eps
            H[i, j] = (_singlet_frac(pp, rho) - _singlet_frac(pm, rho)
                       - _singlet_frac(mp, rho) + _singlet_frac(mm, rho)
                       ) / (4 * eps * eps)
    return float(-np.trace(H))


# ── T-matrix gradient (finite difference) ────────────────────────────────────
def compute_T_gradient(chi_t: float, N: int, Gamma: float = 0.0,
                       dchi: float = 0.02) -> np.ndarray:
    """∂T_ij/∂χt via central difference."""
    T_plus = compute_T_matrix(rho2_exact(chi_t + dchi, N, Gamma))
    T_minus = compute_T_matrix(rho2_exact(chi_t - dchi, N, Gamma))
    return (T_plus - T_minus) / (2 * dchi)


# ── Main feature dataclass ────────────────────────────────────────────────────
@dataclass
class PTMFeatureVector:
    # --- inputs ---
    chi_t: float
    N: int
    Gamma: float

    # --- T-matrix (9 floats) ---
    T_xx: float; T_xy: float; T_xz: float
    T_yx: float; T_yy: float; T_yz: float
    T_zx: float; T_zy: float; T_zz: float

    # --- T singular values (3 floats) ---
    sv_1: float; sv_2: float; sv_3: float

    # --- T gradient (9 floats) ---
    dT_xx: float; dT_xy: float; dT_xz: float
    dT_yx: float; dT_yy: float; dT_yz: float
    dT_zx: float; dT_zy: float; dT_zz: float

    # --- PTM / rho eigenvalues (4 floats, descending) ---
    lam_1: float; lam_2: float; lam_3: float; lam_4: float

    # --- Recoverability observables ---
    V_Q: float       # basin volume (Haar fraction > 2/3)
    kappa_Q: float   # curvature stiffness = -Tr(Hessian at optimum)
    T_rank: int      # numerical rank of T

    # --- Derived fidelity quantities ---
    f_max: float     # correct analytic formula: 1/4 + sum(svs)/4
    F_avg: float     # = (2*f_max + 1) / 3
    singularity_proximity: float  # = 1 - ||T||_F / ||T(chi=0)||_F

    def to_array(self) -> np.ndarray:
        """Float feature array (29 floats + 1 int rank)."""
        d = asdict(self)
        return np.array([
            d["chi_t"], d["N"], d["Gamma"],
            d["T_xx"], d["T_xy"], d["T_xz"],
            d["T_yx"], d["T_yy"], d["T_yz"],
            d["T_zx"], d["T_zy"], d["T_zz"],
            d["sv_1"], d["sv_2"], d["sv_3"],
            d["dT_xx"], d["dT_xy"], d["dT_xz"],
            d["dT_yx"], d["dT_yy"], d["dT_yz"],
            d["dT_zx"], d["dT_zy"], d["dT_zz"],
            d["lam_1"], d["lam_2"], d["lam_3"], d["lam_4"],
            d["V_Q"], d["kappa_Q"], float(d["T_rank"]),
            d["f_max"], d["F_avg"], d["singularity_proximity"],
        ], dtype=float)

    @staticmethod
    def feature_names() -> list:
        return [
            "chi_t", "N", "Gamma",
            "T_xx", "T_xy", "T_xz", "T_yx", "T_yy", "T_yz",
            "T_zx", "T_zy", "T_zz",
            "sv_1", "sv_2", "sv_3",
            "dT_xx", "dT_xy", "dT_xz", "dT_yx", "dT_yy", "dT_yz",
            "dT_zx", "dT_zy", "dT_zz",
            "lam_1", "lam_2", "lam_3", "lam_4",
            "V_Q", "kappa_Q", "T_rank",
            "f_max", "F_avg", "singularity_proximity",
        ]


# ── PTMFeatures: the main entry point ────────────────────────────────────────
class PTMFeatures:
    def __init__(self, N: int = 4, V_Q_samples: int = 200,
                 kQ_restarts: int = 8, fast: bool = False):
        """
        N: system size (even)
        V_Q_samples: Haar-random samples for basin volume
        kQ_restarts: Powell restarts for kappa_Q
        fast: if True, skip V_Q and kappa_Q (for large sweeps)
        """
        self.N = N
        self.V_Q_samples = V_Q_samples
        self.kQ_restarts = kQ_restarts
        self.fast = fast
        # Normalization reference: T at chi_t=0.001
        rho0 = rho2_exact(0.001, N, 0.0)
        T0 = compute_T_matrix(rho0)
        self._T0_frob = float(np.linalg.norm(T0, "fro")) + 1e-10

    def compute(self, chi_t: float, Gamma: float = 0.0,
                seed: int = 0) -> PTMFeatureVector:
        """Compute full feature vector for a single (chi_t, Gamma) point."""
        rho = rho2_exact(chi_t, self.N, Gamma)
        T = compute_T_matrix(rho)
        svs = np.linalg.svd(T, compute_uv=False)
        dT = compute_T_gradient(chi_t, self.N, Gamma)
        eigs = compute_ptm_eigenvalues(rho)
        fm = f_max_analytic(T)
        Favg = (2 * fm + 1) / 3
        T_frob = float(np.linalg.norm(T, "fro"))
        sing_prox = 1.0 - T_frob / self._T0_frob

        if self.fast:
            vq = 0.0
            kq = 0.0
        else:
            vq = compute_V_Q(rho, self.V_Q_samples, seed=seed)
            kq = compute_kappa_Q(rho, self.kQ_restarts)

        return PTMFeatureVector(
            chi_t=chi_t, N=self.N, Gamma=Gamma,
            T_xx=T[0,0], T_xy=T[0,1], T_xz=T[0,2],
            T_yx=T[1,0], T_yy=T[1,1], T_yz=T[1,2],
            T_zx=T[2,0], T_zy=T[2,1], T_zz=T[2,2],
            sv_1=svs[0], sv_2=svs[1], sv_3=svs[2],
            dT_xx=dT[0,0], dT_xy=dT[0,1], dT_xz=dT[0,2],
            dT_yx=dT[1,0], dT_yy=dT[1,1], dT_yz=dT[1,2],
            dT_zx=dT[2,0], dT_zy=dT[2,1], dT_zz=dT[2,2],
            lam_1=eigs[0], lam_2=eigs[1], lam_3=eigs[2], lam_4=eigs[3],
            V_Q=vq, kappa_Q=kq,
            T_rank=T_matrix_rank(T),
            f_max=fm, F_avg=Favg,
            singularity_proximity=sing_prox,
        )

    def sweep(self, chi_t_grid: np.ndarray, Gamma_grid: np.ndarray,
              verbose: bool = True) -> tuple:
        """
        Sweep over chi_t x Gamma grid. Returns (features_array, metadata_list).
        features_array: shape (n_chi * n_Gamma, n_features)
        """
        rows = []
        meta = []
        total = len(chi_t_grid) * len(Gamma_grid)
        k = 0
        for Gamma in Gamma_grid:
            for chi_t in chi_t_grid:
                fv = self.compute(chi_t, Gamma, seed=k)
                rows.append(fv.to_array())
                meta.append(asdict(fv))
                k += 1
                if verbose and k % 20 == 0:
                    print(f"  {k}/{total}  chi_t={chi_t:.3f}  Gamma={Gamma:.3f}"
                          f"  F_avg={fv.F_avg:.4f}  V_Q={fv.V_Q:.3f}")
        return np.array(rows), meta


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("PTMFeatures smoke test (N=4, fast mode)")
    print("=" * 60)
    feat = PTMFeatures(N=4, fast=True)

    test_points = [
        ("Product state",     0.001, 0.0),
        ("Optimal chi_t*",    1.189, 0.0),
        ("chi_t* full sim",   1.456, 0.0),
        ("Late phase",        2.0,   0.0),
        ("Singular chi_t=pi", np.pi, 0.0),
        ("Dephased optimal",  1.189, 0.1),
    ]

    print(f"\n{'Label':<25} {'F_avg':>7} {'sv_1':>7} {'sv_2':>7} {'T_yz':>7} {'rank':>5} {'sing_prox':>10}")
    print("-" * 70)
    for label, chi_t, Gamma in test_points:
        fv = feat.compute(chi_t, Gamma)
        print(f"{label:<25} {fv.F_avg:>7.4f} {fv.sv_1:>7.4f} {fv.sv_2:>7.4f}"
              f" {fv.T_yz:>7.4f} {fv.T_rank:>5d} {fv.singularity_proximity:>10.4f}")

    print("\nFeature vector length:", len(PTMFeatureVector.feature_names()))
    print("Feature names:", PTMFeatureVector.feature_names())

    # Verify known analytic values
    fv_pi = feat.compute(np.pi - 1e-6, 0.0)
    assert abs(fv_pi.F_avg - 0.5) < 0.01, f"Theorem 4 check FAILED: F={fv_pi.F_avg}"
    print("\nTheorem 4 check: F_avg(pi) =", round(fv_pi.F_avg, 4), "PASS" if abs(fv_pi.F_avg-0.5)<0.01 else "FAIL")

    fv_star = feat.compute(1.189, 0.0)
    assert fv_star.F_avg > 0.75, f"Peak fidelity check FAILED: F={fv_star.F_avg}"
    print("Peak fidelity check: F_avg(1.189) =", round(fv_star.F_avg, 4), "PASS")

    print("\nAll checks passed.")
