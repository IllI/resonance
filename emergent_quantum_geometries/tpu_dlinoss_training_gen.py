#!/usr/bin/env python3
"""
tpu_dlinoss_training_gen.py
----------------------------
Generates the D-LinOSS training library on TPU/CPU via JAX.

Systems generated:
  1. OAT boundary states  -- Lindblad calibration (analytic + MPS)
  2. SYK4                 -- scrambling calibration (ED, disorder average)
  3. Disordered XXZ       -- MBL calibration (ED, disorder average)
  4. Dicke model          -- HELD-OUT validation target (NOT used in training)

Each system produces witness-like time series y(tau).
D-LinOSS (matrix pencil) extracts (gamma_k, omega_k, A_k) for each.
Results saved as training_library.npz.

Usage:
  python tpu_dlinoss_training_gen.py --systems oat syk xxz dicke --out results/
"""

import argparse, os, time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# â”€â”€ JAX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1.  OAT WITNESS TIME SERIES  (analytic + exact MPS)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def oat_boundary_coherences(N: int, chi_t_vals):
    """
    Returns |z(chi_t)|, |w(chi_t)| for OAT boundary pair.
    Uses the exact MPS automaton formula.
    z = rho_{00,11},  w = rho_{01,10}
    """
    # Build all 2^N bitstrings; split into left/right halves
    half = N // 2
    idx = np.arange(2**N, dtype=np.int32)
    left  = (idx >> half) & ((1 << half) - 1)   # upper half bits
    right = idx & ((1 << half) - 1)              # lower half bits

    # Magnetization of each half  (count 1-bits minus count 0-bits) / 2
    def mag(x, n):
        bits = ((x[:, None] >> np.arange(n)[None, :]) & 1)
        return bits.sum(1) - (n - bits.sum(1))   # 2*(n_ones) - n

    mA = mag(left,  half).astype(np.float32) / 2
    mB = mag(right, half).astype(np.float32) / 2

    results = []
    for chi_t in chi_t_vals:
        phases = np.exp(-1j * chi_t * mA * mB) / (2**N)

        # z = rho_{00,11}: both boundary spins (s[0]=0,s[N-1]=0) vs (1,1)
        # boundary = (bit 0 of right, bit half-1 of left)
        left_msb  = (left  >> (half - 1)) & 1   # leftmost atom
        right_lsb = right & 1                    # rightmost atom

        mask_00 = (left_msb == 0) & (right_lsb == 0)
        mask_11 = (left_msb == 1) & (right_lsb == 1)
        mask_01 = (left_msb == 0) & (right_lsb == 1)
        mask_10 = (left_msb == 1) & (right_lsb == 0)

        phi_00 = phases[mask_00]
        phi_11 = phases[mask_11]
        phi_01 = phases[mask_01]
        phi_10 = phases[mask_10]

        z = np.sum(np.conj(phi_00)) * np.sum(phi_11) * 2**N
        w = np.sum(np.conj(phi_01)) * np.sum(phi_10) * 2**N

        results.append((abs(z), abs(w)))
    return np.array(results)   # shape (len(chi_t_vals), 2)


def oat_witness_series(N, chi_t_star, Gamma, tau_vals):
    """
    Lindblad-decayed witness at optimal OAT coupling time.
    y(tau) = Tr[W rho2(tau)] = (1/4 - r) * exp(-4*Gamma*tau)
    where r = max(|z|,|w|) at chi_t_star.
    """
    coh = oat_boundary_coherences(N, [chi_t_star])
    r = float(np.max(coh[0]))
    y0 = 0.25 - r                    # initial witness value (negative = entangled)
    return y0 * np.exp(-4 * Gamma * tau_vals)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2.  SYK4 RETARDED GREEN'S FUNCTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def syk4_hamiltonian_sparse(Nf, seed, J=1.0):
    """
    SYK4 Hamiltonian for Nf complex fermions in occupation-number basis.
    H = (J/Nf^{3/2}) sum_{i<j<k<l} J_{ijkl} c_i^dag c_j^dag c_k c_l + h.c.
    Returns sparse matrix of size 2^Nf x 2^Nf.
    """
    rng = np.random.default_rng(seed)
    dim = 2**Nf
    H = np.zeros((dim, dim), dtype=np.complex128)
    scale = J / Nf**1.5

    # Jordan-Wigner: c_i = (prod_{j<i} sigma_z^j) * sigma_-^i
    def c_op(i, dim, Nf):
        """Returns dense c_i operator."""
        op = np.eye(dim, dtype=complex)
        # Parity string
        for k in range(Nf):
            if k < i:
                sz = np.diag([(-1)**((np.arange(dim) >> (Nf-1-k)) & 1)])
                op = sz @ op
        # sigma_minus on site i
        sm = np.zeros((dim, dim), dtype=complex)
        for s in range(dim):
            bit = (s >> (Nf-1-i)) & 1
            if bit == 1:
                s2 = s ^ (1 << (Nf-1-i))
                sm[s2, s] = 1.0
        return sm @ op

    ops = [c_op(i, dim, Nf) for i in range(Nf)]

    for i in range(Nf):
        for j in range(i+1, Nf):
            for k in range(j+1, Nf):
                for l in range(k+1, Nf):
                    Jijkl = rng.standard_normal() * scale
                    term = Jijkl * (ops[i].conj().T @ ops[j].conj().T @ ops[k] @ ops[l])
                    H += term + term.conj().T

    return H


def syk4_green_function(Nf, beta, t_vals, seed=0, n_disorder=20):
    """
    Disorder-averaged retarded Green's function:
    G_R(t) = -i theta(t) <{c_0(t), c_0^dag(0)}>_beta
    averaged over n_disorder SYK realizations.
    """
    dim = 2**Nf
    GR_avg = np.zeros(len(t_vals), dtype=complex)

    for s in range(n_disorder):
        H = syk4_hamiltonian_sparse(Nf, seed + s)
        evals, evecs = la.eigh(H)
        evals -= evals.min()

        # Thermal state
        boltz = np.exp(-beta * evals)
        Z = boltz.sum()
        rho_th = evecs @ np.diag(boltz / Z) @ evecs.conj().T

        # c_0 operator
        def c0_elem(dim, Nf):
            sm = np.zeros((dim, dim), dtype=complex)
            for st in range(dim):
                bit = (st >> (Nf-1)) & 1
                if bit == 1:
                    s2 = st ^ (1 << (Nf-1))
                    sm[s2, st] = 1.0
            return sm

        c0 = c0_elem(dim, Nf)
        c0d = c0.conj().T

        # G_R(t) = -i Tr[rho_th {c0(t), c0d(0)}]
        for it, t in enumerate(t_vals):
            if t <= 0:
                GR_avg[it] += 0.0
                continue
            Ut  = evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T
            Utd = Ut.conj().T
            c0t = Ut @ c0 @ Utd
            anticomm = c0t @ c0d + c0d @ c0t
            GR_avg[it] += -1j * np.trace(rho_th @ anticomm)

    GR_avg /= n_disorder
    return GR_avg.real


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3.  DISORDERED XXZ â€” MBL IMBALANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def xxz_hamiltonian(L, W, J=1.0, Delta=1.0, seed=0):
    """
    H = J sum_i (XX + YY)/2 + Delta*ZZ  +  W sum_i h_i Z_i
    Returns dense matrix in the S_z=0 sector.
    """
    rng = np.random.default_rng(seed)
    disorder = rng.uniform(-W, W, L)

    dim = 2**L
    H = np.zeros((dim, dim))

    for s in range(dim):
        # Disorder diagonal
        for i in range(L):
            bit = (s >> i) & 1
            H[s, s] += disorder[i] * (1 - 2*bit)

        # Hopping and ZZ
        for i in range(L-1):
            bi  = (s >> i) & 1
            bi1 = (s >> (i+1)) & 1
            # ZZ
            H[s, s] += J * Delta * (1 - 2*bi) * (1 - 2*bi1)
            # XX + YY: flip-flop if opposite spins
            if bi != bi1:
                s2 = s ^ (1 << i) ^ (1 << (i+1))
                H[s, s2] += J

    return H


def xxz_imbalance(L, W, t_vals, seed=0, n_disorder=30):
    """
    Charge imbalance I(t) = <sum_i (-1)^i n_i(t)> / <sum_i (-1)^i n_i(0)>
    starting from Neel state |01010...>.
    """
    neel = sum(1 << i for i in range(0, L, 2))   # even sites occupied
    dim = 2**L

    I_avg = np.zeros(len(t_vals))

    for s_idx in range(n_disorder):
        H = xxz_hamiltonian(L, W, seed=seed + s_idx)
        evals, evecs = la.eigh(H)
        psi0 = evecs[:, 0] * 0  # start in Neel state
        psi0 = np.zeros(dim); psi0[neel] = 1.0

        # Project to eigenbasis
        coeffs = evecs.conj().T @ psi0

        for it, t in enumerate(t_vals):
            phases = np.exp(-1j * evals * t)
            psi_t = evecs @ (phases * coeffs)

            imb = 0.0
            norm = 0.0
            for i in range(L):
                n_i = np.abs(psi_t)**2  # probabilities
                occ = np.array([(st >> i) & 1 for st in range(dim)], dtype=float)
                sign = (-1)**i
                imb  += sign * np.dot(n_i, occ)
                norm += abs(sign) * np.dot(np.abs(psi0)**2, occ)

            I_avg[it] += imb / max(norm, 1e-12)

    return I_avg / n_disorder


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4.  DICKE MODEL  (held-out validation)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def dicke_hamiltonian(N_spins, omega0, omegam, g, n_phonons=20):
    """
    H = omega0 Jz + omegam a^dag a + (g/sqrt(N)) (a + a^dag) Jx
    In total-spin sector J = N/2 (symmetric subspace).
    Basis: |m, n> where m in {-J,...,J}, n in {0,...,n_phonons-1}.
    """
    J = N_spins / 2.0
    m_vals = np.arange(-J, J+1)   # 2J+1 values
    n_vals = np.arange(n_phonons)
    dim_spin  = len(m_vals)
    dim_phonon = n_phonons
    dim = dim_spin * dim_phonon

    def idx(im, n): return im * dim_phonon + n

    H = np.zeros((dim, dim))
    g_eff = g / np.sqrt(N_spins)

    for im, m in enumerate(m_vals):
        for n in n_vals:
            i = idx(im, n)
            # omega0 Jz
            H[i, i] += omega0 * m
            # omegam a^dag a
            H[i, i] += omegam * n
            # g_eff (a + a^dag) Jx -- Jx = (J+ + J-)/2
            # a: |n> -> sqrt(n)|n-1>; a^dag: |n> -> sqrt(n+1)|n+1>
            # J+: |m> -> sqrt(J(J+1)-m(m+1))|m+1>
            # J-: |m> -> sqrt(J(J+1)-m(m-1))|m-1>
            Jx_factor_p = np.sqrt(max(J*(J+1) - m*(m+1), 0)) / 2 if im+1 < dim_spin else 0
            Jx_factor_m = np.sqrt(max(J*(J+1) - m*(m-1), 0)) / 2 if im-1 >= 0 else 0

            if n+1 < dim_phonon:
                a_fac = np.sqrt(n+1)
                if im+1 < dim_spin:
                    H[idx(im+1, n+1), i] += g_eff * Jx_factor_p * a_fac
                    H[i, idx(im+1, n+1)] += g_eff * Jx_factor_p * a_fac
                if im-1 >= 0:
                    H[idx(im-1, n+1), i] += g_eff * Jx_factor_m * a_fac
                    H[i, idx(im-1, n+1)] += g_eff * Jx_factor_m * a_fac
            if n-1 >= 0:
                a_fac = np.sqrt(n)
                if im+1 < dim_spin:
                    H[idx(im+1, n-1), i] += g_eff * Jx_factor_p * a_fac
                    H[i, idx(im+1, n-1)] += g_eff * Jx_factor_p * a_fac
                if im-1 >= 0:
                    H[idx(im-1, n-1), i] += g_eff * Jx_factor_m * a_fac
                    H[i, idx(im-1, n-1)] += g_eff * Jx_factor_m * a_fac

    return H


def dicke_jz_series(N_spins, omega0, omegam, g, t_vals, n_phonons=20):
    """
    <Jz(t)> starting from ground state.
    """
    H = dicke_hamiltonian(N_spins, omega0, omegam, g, n_phonons)
    evals, evecs = la.eigh(H)
    psi0 = evecs[:, 0]

    J = N_spins / 2.0
    m_vals = np.arange(-J, J+1)
    dim_phonon = n_phonons

    Jz_op = np.zeros(len(psi0))
    for im, m in enumerate(m_vals):
        for n in range(dim_phonon):
            Jz_op[im * dim_phonon + n] = m

    coeffs = evecs.conj().T @ psi0
    y = np.zeros(len(t_vals))
    for it, t in enumerate(t_vals):
        psi_t = evecs @ (np.exp(-1j * evals * t) * coeffs)
        y[it] = np.real(np.dot(psi_t.conj(), Jz_op * psi_t))

    return y


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5.  D-LinOSS: MATRIX PENCIL DECOMPOSITION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def dlinoss_decompose(y, t, K_max=6, noise_threshold=1e-3):
    """
    Matrix pencil method to extract damped sinusoid modes from y(t).
    Returns dict with gamma_k, omega_k, A_k, K_eff.
    Reference: Hua & Sarkar (1990).
    """
    M = len(y)
    L_pencil = M // 3   # pencil parameter

    # Build Hankel matrix
    Y = np.array([[y[i+j] for j in range(L_pencil+1)] for i in range(M - L_pencil)])

    # Split Hankel into Y1 (drop last col) and Y2 (drop first col)
    Y1 = Y[:, :-1]
    Y2 = Y[:, 1:]

    # SVD of Y1 for K_eff detection and pseudoinverse
    U1, s1, Vh1 = la.svd(Y1, full_matrices=False)

    # K_eff: number of singular values above threshold
    s_norm = s1 / (s1[0] + 1e-30)
    K_eff = max(1, int(np.sum(s_norm > noise_threshold)))
    K_eff = min(K_eff, K_max, len(s1))

    # Truncate to K_eff modes
    U1_k  = U1[:, :K_eff]       # (rows, K)
    s1_k  = s1[:K_eff]          # (K,)
    Vh1_k = Vh1[:K_eff, :]      # (K, cols)

    # Compact pencil matrix (K x K): Z = diag(1/s) @ U^H @ Y2 @ V^H
    Z_mat = np.diag(1.0/s1_k) @ (U1_k.conj().T @ Y2) @ Vh1_k.conj().T
    poles, _ = la.eig(Z_mat)

    dt = t[1] - t[0] if len(t) > 1 else 1.0
    gamma_k  = -np.log(np.abs(poles) + 1e-30) / dt
    omega_k  =  np.angle(poles) / dt

    # Amplitudes via least squares: Vandermonde system
    # Z_cols shape: (M, K_eff)
    # Vandermonde matrix: shape (M, K_eff)
    Z_cols_c = np.array([poles**n for n in range(M)])  # (M, K)
    A_complex, _, _, _ = np.linalg.lstsq(Z_cols_c, y.astype(complex), rcond=None)
    A_k = np.abs(A_complex)

    return {
        "K_eff":   K_eff,
        "gamma_k": gamma_k.real,
        "omega_k": omega_k.real,
        "A_k":     A_k,
        "s_vals":  s1,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6.  MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+",
                        default=["oat", "syk", "xxz", "dicke"],
                        choices=["oat", "syk", "xxz", "dicke"])
    parser.add_argument("--out", default="dlinoss_training/")
    parser.add_argument("--n_tau", type=int, default=40)
    parser.add_argument("--tau_max", type=float, default=500.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tau_vals = np.linspace(0.01, args.tau_max, args.n_tau)
    results  = {}

    # â”€â”€ OAT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "oat" in args.systems:
        print("\n=== Generating OAT training data ===")
        oat_records = []
        for N in [2, 4, 6, 8, 12, 16, 24, 32]:
            # Find optimal chi_t
            chi_t_grid = np.linspace(0.1, 6.0, 60)
            coh = oat_boundary_coherences(N, chi_t_grid)
            r_vals = np.max(coh, axis=1)
            chi_t_star = chi_t_grid[np.argmax(r_vals)]
            C_peak = 2 * max(0, float(np.max(r_vals)) - 0.25)

            for Gamma in [0.001, 0.005, 0.01, 0.05]:
                y = oat_witness_series(N, chi_t_star, Gamma, tau_vals)
                modal = dlinoss_decompose(y, tau_vals)
                record = {
                    "system": "OAT", "N": N, "chi_t_star": chi_t_star,
                    "Gamma": Gamma, "C_peak": C_peak,
                    "y": y, **{k: modal[k] for k in ["K_eff","gamma_k","omega_k","A_k"]}
                }
                oat_records.append(record)
                print(f"  N={N:2d}  chi_t*={chi_t_star:.3f}  C={C_peak:.4f}"
                      f"  Gamma={Gamma:.3f}  K_eff={modal['K_eff']}"
                      f"  gamma_dom={modal['gamma_k'][0]:.4f}  (expect {4*Gamma:.4f})")
        results["OAT"] = oat_records

    # â”€â”€ SYK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "syk" in args.systems:
        print("\n=== Generating SYK4 training data ===")
        syk_records = []
        tau_syk = np.linspace(0.01, 20.0, args.n_tau)
        for Nf in [6, 8, 10, 12]:
            for beta in [1.0, 5.0, 10.0]:
                print(f"  Nf={Nf}  beta={beta}  (ED dim={2**Nf})")
                y = syk4_green_function(Nf, beta, tau_syk, seed=42, n_disorder=10)
                modal = dlinoss_decompose(y, tau_syk)
                syk_records.append({
                    "system": "SYK4", "Nf": Nf, "beta": beta,
                    "y": y, **{k: modal[k] for k in ["K_eff","gamma_k","omega_k","A_k"]}
                })
                print(f"    K_eff={modal['K_eff']}  gamma_dom={modal['gamma_k'][0]:.4f}")
        results["SYK4"] = syk_records

    # â”€â”€ XXZ (MBL) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "xxz" in args.systems:
        print("\n=== Generating XXZ/MBL training data ===")
        xxz_records = []
        tau_xxz = np.linspace(0.01, 100.0, args.n_tau)
        for L in [8, 10, 12]:
            for W in [0.5, 2.0, 5.0, 10.0]:
                print(f"  L={L}  W={W}  (dim={2**L})")
                y = xxz_imbalance(L, W, tau_xxz, seed=0, n_disorder=15)
                modal = dlinoss_decompose(y, tau_xxz)
                xxz_records.append({
                    "system": "XXZ", "L": L, "W": W,
                    "y": y, **{k: modal[k] for k in ["K_eff","gamma_k","omega_k","A_k"]}
                })
                print(f"    K_eff={modal['K_eff']}  I_inf={float(y[-1]):.4f}")
        results["XXZ"] = xxz_records

    # â”€â”€ DICKE  (held-out) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "dicke" in args.systems:
        print("\n=== Generating Dicke model data (HELD-OUT) ===")
        dicke_records = []
        tau_dicke = np.linspace(0.01, 50.0, args.n_tau)
        omega0 = 1.0; omegam = 1.0
        g_crit = np.sqrt(omega0 * omegam) / 2   # superradiant QPT
        for N_spins in [2, 4, 6]:
            for g_ratio in [0.2, 0.6, 0.9, 1.1, 1.5, 2.0]:
                g = g_ratio * g_crit
                print(f"  N={N_spins}  g/g_c={g_ratio}  g={g:.4f}")
                y = dicke_jz_series(N_spins, omega0, omegam, g, tau_dicke)
                modal = dlinoss_decompose(y, tau_dicke)
                dicke_records.append({
                    "system": "Dicke", "N_spins": N_spins,
                    "g_ratio": g_ratio, "g": g,
                    "y": y, **{k: modal[k] for k in ["K_eff","gamma_k","omega_k","A_k"]}
                })
                print(f"    K_eff={modal['K_eff']}")
        results["Dicke"] = dicke_records

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out_path = os.path.join(args.out, "training_library.npz")
    np.savez(out_path, **{
        k: np.array([
            {kk: (vv.tolist() if hasattr(vv,'tolist') else vv)
             for kk, vv in rec.items()} for rec in v
        ], dtype=object) for k, v in results.items()
    })
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\nâ”€â”€ D-LinOSS Signature Summary â”€â”€")
    print(f"{'System':<10} {'Params':<25} {'K_eff':<6} {'gamma_dom':<12}")
    print("-" * 55)
    for sys_name, recs in results.items():
        for r in recs:
            if sys_name == "OAT":
                params = f"N={r['N']} G={r['Gamma']:.3f}"
            elif sys_name == "SYK4":
                params = f"Nf={r['Nf']} b={r['beta']}"
            elif sys_name == "XXZ":
                params = f"L={r['L']} W={r['W']}"
            else:
                params = f"N={r['N_spins']} g/gc={r['g_ratio']}"
            gdom = float(np.max(r['gamma_k'])) if len(r['gamma_k']) else 0
            print(f"{sys_name:<10} {params:<25} {r['K_eff']:<6} {gdom:<12.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
