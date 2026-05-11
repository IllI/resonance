"""Debug extract_boundary_rho vs exact for N=6."""
import math, numpy as np, sys
sys.path.insert(0, '.')
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import jz_table, plus_state, oat_evolve, boundary_rdm, concurrence
import jax.numpy as jnp

N = 6; n = 3; chi_t = math.pi / 3
mA, mB = jz_table(N)
psi_t = np.array(oat_evolve(jnp.array(plus_state(N)), jnp.array(mA), jnp.array(mB), chi_t))
rho_exact = boundary_rdm(psi_t, N)

tensors = oat_mps(N, chi_t)
print("Bond dims:", [T.shape for T in tensors])

# Manually contract full wavefunction from MPS and compare
N_sites = len(tensors)
# Contract MPS into full state vector
sv = tensors[0]  # (1, 2, chi_R)
for k in range(1, N_sites):
    T = tensors[k]  # (chi_L, 2, chi_R)
    # sv: (..., chi), T: (chi, 2, chi_R) -> (..., 2, chi_R)
    sv = np.tensordot(sv, T, axes=([[-1], [0]]))  # contract last of sv with first of T

# sv shape: (1, 2, 2, 2, 2, 2, 2, 1) -- flatten
sv = sv.reshape(-1)
print(f"MPS norm: {np.linalg.norm(sv):.6f}")
print(f"Exact norm: {np.linalg.norm(psi_t):.6f}")

# Compare elementwise
diff_sv = np.max(np.abs(sv.ravel() - psi_t.ravel()))
# Phase freedom: check up to global phase
phase = np.dot(psi_t.conj(), sv) / (np.linalg.norm(sv)**2)
sv_phased = sv * phase / abs(phase) if abs(phase) > 1e-10 else sv
diff_phased = np.max(np.abs(sv_phased - psi_t))
print(f"Wavefunction max diff (raw): {diff_sv:.2e}")
print(f"Wavefunction max diff (phase-aligned): {diff_phased:.2e}")

# Now check TN rho2
rho_tn = extract_boundary_rho(tensors, n)
print(f"\nExact C: {concurrence(rho_exact):.6f}")
print(f"TN     C: {concurrence(rho_tn):.6f}")
print(f"rho diff:  {np.max(np.abs(rho_exact - rho_tn)):.2e}")
print("\nExact rho2 diagonal:", np.diag(rho_exact).real)
print("TN     rho2 diagonal:", np.diag(rho_tn).real)
