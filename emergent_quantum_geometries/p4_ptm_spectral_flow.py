"""
p4_ptm_spectral_flow.py — Priority 4: PTM Eigenvalue / Singular Value Flow
============================================================================
Tracks the full PTM spectrum as a function of chi_t.

Hypothesis (SS104): chi_t=pi is a RANK-COLLAPSE SINGULARITY.
At chi_t=pi: T_xx=T_yy=T_zz=0 -> PTM has rank 1 -> all singular values
except one collapse to zero simultaneously.

The PTM maps Pauli operators {I,X,Y,Z} on Alice's input to Bob's output
before classical correction. For a 2-qubit resource state rho2, the channel is:
  E(rho_in) = Tr_A[(rho_in^T ⊗ I) rho2]   (teleportation map)

PTM element: T_ij = Tr[sigma_i E(sigma_j/2)] (i,j in {0,1,2,3} = {I,X,Y,Z})
"""

import numpy as np

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src, 'x', 'exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

# Pauli matrices
I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sY = np.array([[0,-1j],[1j,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, sX, sY, sZ]
LABELS = ['I','X','Y','Z']

def teleportation_channel(rho2, rho_in):
    """E(rho_in) = Tr_A[(rho_in^T ⊗ I_B) * rho2].
    Partial trace over Alice's 2D subspace."""
    rho_in_T = rho_in.T
    op = np.kron(rho_in_T, I2) @ rho2
    # Partial trace over Alice (first qubit)
    out = np.zeros((2,2), dtype=complex)
    for k in range(2):
        bra = np.zeros(2); bra[k] = 1.0
        out += np.kron(bra[np.newaxis,:], I2) @ op @ np.kron(bra[:,np.newaxis], I2)
    return out

def compute_ptm(rho2):
    """Compute full 4x4 PTM for the teleportation channel."""
    P = np.zeros((4,4))
    for j in range(4):
        rho_in = PAULIS[j] / 2.0  # input (may not be normalized — OK for PTM)
        out = teleportation_channel(rho2, rho_in)
        for i in range(4):
            P[i,j] = np.real(np.trace(PAULIS[i] @ out))
    return P

print("="*72)
print("PRIORITY 4: PTM SPECTRAL FLOW")
print("="*72)
N = 4

chi_ts = np.concatenate([
    np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
    np.linspace(0.35, 0.65, 7),
    np.array([0.70, 0.75, 0.80, 0.85, 0.90, 0.95]),
    np.array([0.97, 0.98, 0.99, 0.995, 1.000])
]) * np.pi

print(f"\nComputing PTM at {len(chi_ts)} chi_t points for N={N}...")
print()
print(f"  {'chi_t/pi':>9}  {'T_xx':>8}  {'T_yy':>8}  {'T_zz':>8}  "
      f"{'rank':>5}  {'s1':>8}  {'s2':>8}  {'s3':>8}  {'cond':>10}")
print("  "+"-"*85)

ptm_results = []
for chi_t in chi_ts:
    rho2 = oat_rho2_exact(N, chi_t)
    P    = compute_ptm(rho2)
    sv   = np.linalg.svd(P, compute_uv=False)   # descending order
    # Rank: number of sv above threshold 1e-6
    rnk  = int(np.sum(sv > 1e-6))
    cond = sv[0] / max(sv[1], 1e-12)
    # PTM diagonal elements (Bloch vector components)
    Txx, Tyy, Tzz = P[1,1], P[2,2], P[3,3]
    ptm_results.append({
        'chi_t': chi_t, 'P': P.copy(), 'sv': sv.copy(),
        'rank': rnk, 'cond': cond, 'Txx': Txx, 'Tyy': Tyy, 'Tzz': Tzz
    })
    print(f"  {chi_t/np.pi:9.4f}  {Txx:8.5f}  {Tyy:8.5f}  {Tzz:8.5f}  "
          f"{rnk:5d}  {sv[0]:8.5f}  {sv[1]:8.5f}  {sv[2]:8.5f}  {cond:10.2f}")

# PTM at chi_t=pi
r_pi = ptm_results[-1]
print(f"\n  Full PTM at chi_t=pi:")
for i,row in enumerate(r_pi['P']):
    print(f"    [{', '.join(f'{v:8.5f}' for v in row)}]  ({LABELS[i]})")
print(f"\n  Singular values at chi_t=pi: {r_pi['sv']}")
print(f"  Rank at chi_t=pi: {r_pi['rank']}  (expected: 1)")

# ── Rank-collapse summary ─────────────────────────────────────────────────
print()
print("="*72)
print("SPECTRAL FLOW ANALYSIS")
print("="*72)

# Find where rank changes
print("\n  Rank transitions:")
prev_rank = None
for r in ptm_results:
    if r['rank'] != prev_rank:
        print(f"    chi_t/pi={r['chi_t']/np.pi:.4f}: rank {prev_rank} -> {r['rank']}")
        prev_rank = r['rank']

# Singular value flow near pi
print("\n  Singular value decay near chi_t=pi:")
print(f"  {'chi_t/pi':>9}  {'sv[0]':>10}  {'sv[1]':>10}  {'sv[2]':>10}  {'sv[3]':>10}")
for r in ptm_results[-8:]:
    sv = r['sv']
    print(f"  {r['chi_t']/np.pi:9.5f}  {sv[0]:10.7f}  {sv[1]:10.7f}  {sv[2]:10.7f}  {sv[3]:10.7f}")

# Verify rank-collapse conjecture
pi_rank = ptm_results[-1]['rank']
max_rank = max(r['rank'] for r in ptm_results)
rank_collapses = pi_rank == 1

print()
print("="*72)
print("RANK-COLLAPSE CONJECTURE TEST")
print("="*72)
print(f"\n  Maximum PTM rank observed: {max_rank}")
print(f"  PTM rank at chi_t=pi:      {pi_rank}")
print(f"  Rank-collapse at pi:        {rank_collapses}")
if rank_collapses:
    print(f"\n  CONJECTURE CONFIRMED: chi_t=pi is a PTM rank-collapse singularity.")
    print(f"  At chi_t=pi, the teleportation channel becomes a rank-1 (completely")
    print(f"  depolarizing) map in Bloch-vector space.")
    print(f"\n  PAPER CLAIM:")
    print(f"  'The teleportation-capable phase terminates at a rank-deficient")
    print(f"  channel singularity (PTM rank 1 at chi_t=pi), providing a rigorous")
    print(f"  operator-theoretic characterization of the geometric collapse.'")
else:
    print(f"\n  Rank at pi = {pi_rank} (expected 1). Check threshold or off-diagonal terms.")

# Analytic verification: T_xx = cos^{N-2}(chi_t/2)
print()
print("  Analytic vs numeric T_xx check:")
print(f"  {'chi_t/pi':>9}  {'T_xx_numeric':>14}  {'T_xx_analytic':>15}  {'error':>10}")
for r in ptm_results[::4]:
    Txx_an = np.cos(r['chi_t']/2)**(N-2)
    err    = abs(r['Txx'] - Txx_an)
    print(f"  {r['chi_t']/np.pi:9.4f}  {r['Txx']:14.8f}  {Txx_an:15.8f}  {err:10.2e}")
