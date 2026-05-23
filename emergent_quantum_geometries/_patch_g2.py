src = open('teleport_program_g2.py').read()

# Fix 1: su2_from_so3 sign convention (was computing U-dagger)
src = src.replace(
    '    return np.array([[ qw+1j*qz,  qy+1j*qx],\n'
    '                     [-qy+1j*qx,  qw-1j*qz]], dtype=complex)',
    '    return np.array([[ qw-1j*qz, -qy-1j*qx],\n'
    '                     [ qy-1j*qx,  qw+1j*qz]], dtype=complex)'
)

# Fix 2: teleport_aligned should pre-rotate BOTH Alice and Bob resource qubits,
# then use STANDARD Pauli corrections (no extra Ub in corrections)
src = src.replace(
    '''def teleport_aligned(rho_in, rho_resource, Ua, Ub_corrections):
    """
    Teleportation with:
      Ua: Alice's pre-rotation (SU(2)) applied to her resource qubit
      Ub_corrections: list of 4 Bob corrections (SU(2)) for each Bell outcome
    """
    # Pre-rotate Alice's half of resource
    Ua_full = np.kron(Ua, I2)  # acts on Alice qubit (first qubit of resource)
    rho_res_rot = Ua_full @ rho_resource @ Ua_full.conj().T

    rho3 = np.kron(rho_in, rho_res_rot)
    out = np.zeros((2,2), dtype=complex)
    for bk, Uk in zip(BELLS, Ub_corrections):
        Pb = np.outer(bk, bk.conj())
        Pk = np.kron(Pb, I2)
        rp = Pk @ rho3 @ Pk
        rpr = rp.reshape(2,2,2,2,2,2)
        rB = np.einsum('ijkijl->kl', rpr)
        out += Uk @ rB @ Uk.conj().T
    return out''',
    '''def teleport_aligned(rho_in, rho_resource, Ua, Ub):
    """
    Teleportation with optimal pre-rotations on both Alice and Bob resource qubits.
    Ua: Alice SU(2) pre-rotation on her resource qubit
    Ub: Bob SU(2) pre-rotation on his resource qubit
    Uses STANDARD Pauli corrections after Bell measurement.
    Achieves f_max = (1 + Tr(R_A^T T R_B))/4 by Horodecki theorem.
    """
    Ua_full = np.kron(Ua, I2)   # Alice resource qubit (qubit 0 of resource)
    Ub_full = np.kron(I2, Ub)   # Bob resource qubit (qubit 1 of resource)
    rho_res_rot = Ub_full @ (Ua_full @ rho_resource @ Ua_full.conj().T) @ Ub_full.conj().T
    rho3 = np.kron(rho_in, rho_res_rot)
    out = np.zeros((2,2), dtype=complex)
    for bk, Uk in zip(BELLS, CORR_STD):
        Pb = np.outer(bk, bk.conj())
        Pk = np.kron(Pb, I2)
        rp = Pk @ rho3 @ Pk
        rpr = rp.reshape(2,2,2,2,2,2)
        rB = np.einsum('ijkijl->kl', rpr)
        out += Uk @ rB @ Uk.conj().T
    return out'''
)

# Fix 3: ptm_aligned_protocol returns (Ua, Ub) not (Ua, Ub_corrections, svs)
src = src.replace(
    '''    # Alice prerotation (SU(2) from SO(3) = U^T)
    Ua = su2_from_so3(U.T)

    # Bob corrections: standard Pauli corrections composed with V
    Ub_V = su2_from_so3(V)
    Ub_corr = [Uk @ Ub_V for Uk in CORR_STD]

    return Ua, Ub_corr, s''',
    '''    # Ua: Alice pre-rotates her resource to align T to diagonal basis
    # Ub: Bob pre-rotates his resource to align Bob's frame
    Ua = su2_from_so3(U)
    Ub = su2_from_so3(V)
    return Ua, Ub, s'''
)

# Fix 4: haar_mean_fidelity signature already accepts Ua, Ub_corr ->
# update callers inside sweep and paper_comparison_table
src = src.replace(
    'def haar_mean_fidelity(rho_resource, Ua, Ub_corr, n=300, seed=0):\n'
    '    rng = np.random.default_rng(seed)\n'
    '    fids = []\n'
    '    for _ in range(n):\n'
    '        psi = rng.standard_normal(2) + 1j*rng.standard_normal(2)\n'
    '        psi /= np.linalg.norm(psi)\n'
    '        ri = np.outer(psi, psi.conj())\n'
    '        ro = teleport_aligned(ri, rho_resource, Ua, Ub_corr)\n'
    '        fids.append(float(np.real(np.trace(ri @ ro))))\n'
    '    return float(np.mean(fids)), float(np.std(fids))',
    'def haar_mean_fidelity(rho_resource, Ua, Ub, n=300, seed=0):\n'
    '    rng = np.random.default_rng(seed)\n'
    '    fids = []\n'
    '    for _ in range(n):\n'
    '        psi = rng.standard_normal(2) + 1j*rng.standard_normal(2)\n'
    '        psi /= np.linalg.norm(psi)\n'
    '        ri = np.outer(psi, psi.conj())\n'
    '        ro = teleport_aligned(ri, rho_resource, Ua, Ub)\n'
    '        fids.append(float(np.real(np.trace(ri @ ro))))\n'
    '    return float(np.mean(fids)), float(np.std(fids))'
)

# Fix 5: Standard protocol caller (pass I2, I2 instead of I2, CORR_STD)
src = src.replace(
    'def teleport_standard(rho_in, rho_resource):\n'
    '    return teleport_aligned(rho_in, rho_resource, I2, CORR_STD)',
    'def teleport_standard(rho_in, rho_resource):\n'
    '    return teleport_aligned(rho_in, rho_resource, I2, I2)'
)

open('teleport_program_g2.py', 'w').write(src)
print('patched ok')
