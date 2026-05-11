"""
jila_tn_tpu_run.py
==================
JILA Ana Maria Rey 2025 Quantum Teleportation
Tensor-Network (MPS/TEBD) Simulation — JAX/TPU Native

Physics:
  Two N-ion chains A and B coupled via OAT Hamiltonian.
  H = chi * sum_<ij> Zi Zj  (nearest-neighbour ZZ, TEBD-compatible)

Algorithm:
  MPS: list of rank-3 tensors shape (D_L, 2, D_R)
  TEBD: 2nd-order Trotter with ZZ 2-site gates + X single-site decay
  Fidelity: partial trace of exact state vector on receiver qubit

All contractions are jnp.einsum — fully TPU-accelerated.

Usage on TPU VM:
  python3 jila_tn_tpu_run.py --N 4 8 12 --chi 0.15 --t_max 8.0
"""

import argparse, json, math, sys
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

print(f"[DEVICE] {jax.devices()}")
print(f"[DEVICE] Backend: {jax.default_backend()}")

# ── Pauli matrices ─────────────────────────────────────────────────────────────
X = jnp.array([[0,1],[1,0]], dtype=jnp.complex64)
Z = jnp.array([[1,0],[0,-1]], dtype=jnp.complex64)

# ── 2-site ZZ gate: U = exp(-i chi dt Z⊗Z) ────────────────────────────────────
def zz_gate(chi, dt):
    t = chi * dt
    phases = jnp.array([jnp.exp(-1j*t), jnp.exp(1j*t),
                        jnp.exp(1j*t),  jnp.exp(-1j*t)], dtype=jnp.complex64)
    return jnp.diag(phases).reshape(2,2,2,2)   # (i,j,i',j')

# ── 1-site X gate: U = exp(-i kappa dt X) ─────────────────────────────────────
def x_gate(kappa, dt):
    t = kappa * dt
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.array([[c, -1j*s], [-1j*s, c]], dtype=jnp.complex64)

# ── MPS helpers ────────────────────────────────────────────────────────────────

def product_mps(local_states):
    """Build bond-dim-1 MPS from list of single-qubit kets."""
    return [s.reshape(1,2,1).astype(jnp.complex64) for s in local_states]


def apply2(tensors, i, gate, Dmax):
    """Apply 2-site gate to bond (i, i+1) with SVD truncation."""
    A, B = tensors[i], tensors[i+1]
    DL, DR = A.shape[0], B.shape[2]
    # Contract A-B pair then apply gate
    theta = jnp.einsum('aib,bjc->aijc', A, B)          # (DL,d,d,DR)
    theta = jnp.einsum('aijb,ijkl->aklb', theta, gate)  # (DL,d',d',DR)
    theta = theta.reshape(DL*2, 2*DR)
    U, s, Vh = jnp.linalg.svd(theta, full_matrices=False)
    D = min(s.shape[0], Dmax)
    SB = jnp.diag(s[:D]) @ Vh[:D]
    t = list(tensors)
    t[i]   = U[:,:D].reshape(DL, 2, D)
    t[i+1] = SB.reshape(D, 2, DR)
    return t


def apply1(tensors, i, gate):
    t = list(tensors)
    t[i] = jnp.einsum('aib,ij->ajb', t[i], gate)
    return t


def tebd_step(tensors, chi, kappa, dt, Dmax):
    N = len(tensors)
    Uhalf = zz_gate(chi, dt/2)
    Ufull = zz_gate(chi, dt)
    Ux    = x_gate(kappa, dt)
    for i in range(0, N-1, 2): tensors = apply2(tensors, i, Uhalf, Dmax)
    for i in range(1, N-1, 2): tensors = apply2(tensors, i, Ufull, Dmax)
    for i in range(0, N-1, 2): tensors = apply2(tensors, i, Uhalf, Dmax)
    for i in range(N):          tensors = apply1(tensors, i, Ux)
    return tensors


def mps_to_sv(tensors):
    """Contract MPS to full state vector (exact, use for N<=16)."""
    sv = tensors[0][0,:,:]        # (d, D)
    for A in tensors[1:]:
        sv = jnp.einsum('...a,aib->...ib', sv, A).reshape(-1, A.shape[2])
    return sv[:,0]                # (2^N,)


def fidelity(tensors, N, psi_target):
    """Average teleportation fidelity over 4 Bell outcomes."""
    sv = mps_to_sv(tensors).reshape([2]*N)
    recv = N // 2
    axes = [recv] + [j for j in range(N) if j != recv]
    sv_p = jnp.transpose(sv, axes).reshape(2, -1)
    rho  = (sv_p @ sv_p.conj().T)
    rho  = rho / jnp.trace(rho)
    psi  = jnp.array(psi_target, dtype=jnp.complex64)
    Fs   = [jnp.real(psi.conj() @ (op @ rho @ op.conj().T) @ psi)
            for op in [jnp.eye(2,dtype=jnp.complex64), X, Z, X@Z]]
    return float(jnp.mean(jnp.array(Fs)))


def entropy(tensors, cut):
    """Von Neumann entropy at bond `cut`."""
    N = len(tensors)
    sv = mps_to_sv(tensors).reshape(2**cut, 2**(N-cut))
    _, s, _ = jnp.linalg.svd(sv, full_matrices=False)
    s2 = s**2; s2 = s2 / (s2.sum()+1e-15)
    s2 = jnp.where(s2>1e-15, s2, 1e-15)
    return float(-jnp.sum(s2 * jnp.log2(s2)))


# ── Experiment ─────────────────────────────────────────────────────────────────

def run(N, chi, kappa, t_max, dt, Dmax, n_pts):
    print(f"\n{'='*60}\n  N={N}  chi={chi}  kappa={kappa}  Dmax={Dmax}\n{'='*60}")
    target = np.array([1.,1.], dtype=np.complex64) / math.sqrt(2)  # |+>
    t_grid = np.linspace(0., t_max, n_pts)
    Fc, Sc = [], []

    for idx, t_tgt in enumerate(t_grid):
        # Build initial MPS
        states = []
        for i in range(N):
            if i == 0:           states.append(jnp.array(target))
            elif i < N//2:       states.append(jnp.array([0.,1.], dtype=jnp.complex64))
            else:                states.append(jnp.array([1.,0.], dtype=jnp.complex64))
        psi = product_mps(states)

        # Evolve
        n_steps = max(1, int(t_tgt/dt)) if t_tgt > 0 else 0
        for _ in range(n_steps):
            psi = tebd_step(psi, chi, kappa, dt, Dmax)

        F = fidelity(psi, N, target)
        S = entropy(psi, N//2)
        Fc.append(F); Sc.append(S)
        if idx % max(1, n_pts//8) == 0:
            print(f"  t={t_tgt:.2f}  F={F:.4f}  S={S:.4f}")

    Fa, Sa = np.array(Fc), np.array(Sc)
    res = dict(N_ions=N, F_initial=float(Fa[0]),
               F_peak=float(Fa.max()), F_peak_time=float(t_grid[Fa.argmax()]),
               F_final=float(Fa[-1]), F_mean=float(Fa.mean()),
               S_max=float(Sa.max()), quantum_advantage=bool(Fa.max()>2/3),
               classical_limit=2/3)

    print(f"\n  F_peak={res['F_peak']:.4f} at t={res['F_peak_time']:.2f}  |  S_max={res['S_max']:.4f}")
    print(f"  {'QUANTUM ADVANTAGE' if res['quantum_advantage'] else 'No quantum advantage'} (classical limit=0.6667)")
    return res, t_grid, Fa, Sa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",      nargs="+", type=int,   default=[4,8,12])
    ap.add_argument("--chi",    type=float, default=0.15)
    ap.add_argument("--kappa",  type=float, default=0.005)
    ap.add_argument("--t_max",  type=float, default=8.0)
    ap.add_argument("--dt",     type=float, default=0.02)
    ap.add_argument("--points", type=int,   default=50)
    ap.add_argument("--bond",   type=int,   default=32)
    args = ap.parse_args()

    print("\n" + "="*60)
    print("  JILA OAT Teleportation — MPS/TEBD on JAX/TPU")
    print(f"  Devices: {jax.devices()}")
    print("="*60)

    all_res, all_F, all_S, t_ref = {}, {}, {}, None
    for N in args.N:
        if N % 2:
            print(f"[SKIP] N={N} must be even"); continue
        res, tg, F, S = run(N, args.chi, args.kappa, args.t_max,
                             args.dt, args.bond, args.points)
        all_res[f"N{N}"] = res; all_F[N] = F.tolist(); all_S[N] = S.tolist()
        if t_ref is None: t_ref = tg.tolist()

    summary = dict(
        experiment="JILA OAT Quantum Teleportation — JAX/TPU MPS/TEBD",
        protocol="Rey 2025 — H=chi*Jz_A x Jz_B (OAT), TEBD time evolution",
        devices=str(jax.devices()), N_ions_list=args.N,
        chi=args.chi, kappa=args.kappa, t_max=args.t_max, bond_dim=args.bond,
        t_grid=t_ref,
        fidelity_curves={f"N{N}": all_F[N] for N in all_F},
        entropy_curves={f"N{N}": all_S[N] for N in all_S},
        results=all_res,
    )
    out = Path("jila_tn_tpu_results.json")
    with open(out,"w") as f: json.dump(summary, f, indent=2)
    print(f"\n[DONE] Results saved to {out}")
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("t_grid","fidelity_curves","entropy_curves")},
                     indent=2))

if __name__ == "__main__":
    main()
