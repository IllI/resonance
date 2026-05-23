# _l_02_operators.py  —  Pauli helpers, PTM, entropies, F_rec

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [_sx, _sy, _sz]


def kron_site(op, site, N):
    """Embed 2x2 `op` at qubit index `site` in N-qubit Hilbert space."""
    ops = [np.eye(2, dtype=complex)] * N
    ops[site] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def partial_trace_boundary(rho, N):
    """Trace out all but sites 0 and N-1; return 4x4 boundary density matrix."""
    dim = 2 ** N
    keep = [0, N - 1]
    env_sites = [s for s in range(N) if s not in keep]
    n_env = len(env_sites)
    rho4 = np.zeros((4, 4), dtype=complex)
    for ia in range(2):
        for ib in range(2):
            for ja in range(2):
                for jb in range(2):
                    val = 0.0 + 0j
                    for e in range(2 ** n_env):
                        # build full basis index for bra and ket
                        def full_idx(a_bit, b_bit, env_bits):
                            idx = 0
                            ep = 0
                            for q in range(N):
                                if q == 0:
                                    b = a_bit
                                elif q == N - 1:
                                    b = b_bit
                                else:
                                    b = (env_bits >> (n_env - 1 - ep)) & 1
                                    ep += 1
                                idx = idx * 2 + b
                            return idx
                        bra = full_idx(ia, ib, e)
                        ket = full_idx(ja, jb, e)
                        val += rho[bra, ket]
                    rho4[ia * 2 + ib, ja * 2 + jb] = val
    return rho4


def von_neumann(rho):
    """Von Neumann entropy in bits."""
    ev = np.real(np.linalg.eigvalsh(rho))
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def mutual_info(rho4):
    """I(A:B) for 4x4 two-qubit state."""
    rA = np.array([[rho4[0, 0] + rho4[1, 1], rho4[0, 2] + rho4[1, 3]],
                   [rho4[2, 0] + rho4[3, 1], rho4[2, 2] + rho4[3, 3]]])
    rB = np.array([[rho4[0, 0] + rho4[2, 2], rho4[0, 1] + rho4[2, 3]],
                   [rho4[1, 0] + rho4[3, 2], rho4[1, 1] + rho4[3, 3]]])
    return max(0.0, von_neumann(rA) + von_neumann(rB) - von_neumann(rho4))


def compute_T_matrix(rho4):
    """3x3 PTM correlation matrix T_ij = Tr[(si x sj) rho4]."""
    T = np.zeros((3, 3))
    for i, si in enumerate(PAULIS):
        for j, sj in enumerate(PAULIS):
            T[i, j] = np.real(np.trace(np.kron(si, sj) @ rho4))
    return T


def f_rec_from_T(T):
    """Max recoverable fidelity F_rec = (1 + sv_max(T)) / 2."""
    sv = np.linalg.svd(T, compute_uv=False)
    return float((1.0 + sv[0]) / 2.0)


def spectral_entropy_T(T):
    """Normalised spectral entropy of |T| singular values."""
    sv = np.linalg.svd(T, compute_uv=False)
    sv = sv / (sv.sum() + 1e-15)
    sv = sv[sv > 1e-12]
    return float(-np.sum(sv * np.log(sv + 1e-15)))


def operator_spreading(rho, N, site=0):
    """OTOC proxy: <Z_0(t) Z_{N//2}> as scrambling measure."""
    mid = N // 2
    Z0  = kron_site(_sz, site, N)
    Zm  = kron_site(_sz, mid, N)
    val = np.real(np.trace(Z0 @ Zm @ rho))
    return float(val)
