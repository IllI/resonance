"""Test if oat_rho2_exact uses cross-half-only ZZ vs full OAT."""
import numpy as np
from scipy.linalg import expm

src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}; exec(compile(src,'x','exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

N=4; sX=np.array([[0,1],[1,0]],dtype=complex)
sZ=np.array([[1,0],[0,-1]],dtype=complex); I2=np.eye(2,dtype=complex)

def kron4(*ops):
    r=ops[0]
    for o in ops[1:]: r=np.kron(r,o)
    return r

def txx_from_psi_ptrace03(psi):
    """Partial trace over qubits 1,2; keep qubits 0,3. Return T_xx."""
    rho2=np.zeros((4,4),dtype=complex)
    for a in range(2):
        for b in range(2):
            for q0 in range(2):
                for q3 in range(2):
                    for q0p in range(2):
                        for q3p in range(2):
                            i=q0*8+a*4+b*2+q3
                            j=q0p*8+a*4+b*2+q3p
                            ri=q0*2+q3; ci=q0p*2+q3p
                            rho2[ri,ci]+=psi[i]*psi[j].conj()
    return np.real(np.trace(np.kron(sX,sX)@rho2))/2

# Initial state
plus=np.array([1,1],dtype=complex)/np.sqrt(2)
psi0=plus
for _ in range(N-1): psi0=np.kron(psi0,plus)

# ZZ pairs
def ZZ(i,j,n=4):
    ops=[I2]*n; ops[i]=sZ; ops[j]=sZ
    return kron4(*ops)

pairs_all   = [(i,j) for i in range(4) for j in range(i+1,4)]          # all 6
pairs_cross = [(0,2),(0,3),(1,2),(1,3)]                                  # cross-half only
pairs_intra = [(0,1),(2,3)]                                              # intra-half only

print("  Model comparison at multiple chi_t:")
print(f"  {'chi_t/pi':>9}  {'oat_formula':>12}  {'all_ZZ':>8}  {'cross_ZZ':>10}  {'chi_t*L*R':>10}")
for chi_t in [0.01*np.pi, 0.355*np.pi, 0.5*np.pi, np.pi]:
    T_oat = np.real(np.trace(np.kron(sX,sX)@oat_rho2_exact(N,chi_t)))/2

    # Model A: full OAT - all ZZ pairs, angle chi_t/2
    H_all = sum(ZZ(i,j) for i,j in pairs_all) * (chi_t/2)
    psi_all = expm(-1j*H_all)@psi0
    T_all = txx_from_psi_ptrace03(psi_all)

    # Model B: cross-half ZZ only, angle chi_t/4 (from H = chi_t*JzL*JzR)
    H_cross = sum(ZZ(i,j) for i,j in pairs_cross) * (chi_t/4)
    psi_cross = expm(-1j*H_cross)@psi0
    T_cross = txx_from_psi_ptrace03(psi_cross)

    # Model C: H = chi_t * JzL * JzR, trace out L1=q0, R1=q3, keep q1,q2
    # JzL = (sz0+sz1)/2, JzR = (sz2+sz3)/2
    JzL = (kron4(sZ,I2,I2,I2) + kron4(I2,sZ,I2,I2))/2
    JzR = (kron4(I2,I2,sZ,I2) + kron4(I2,I2,I2,sZ))/2
    H_lr = chi_t * JzL @ JzR
    psi_lr = expm(-1j*H_lr)@psi0
    T_lr = txx_from_psi_ptrace03(psi_lr)

    print(f"  {chi_t/np.pi:9.4f}  {T_oat:12.5f}  {T_all:8.5f}  {T_cross:10.5f}  {T_lr:10.5f}")

print()
# Also try partial trace over different qubit pairs for cross model
print("  Cross-half model: trace over q0,q3 -> keep q1,q2")
chi_t = 0.355*np.pi
H_cross2 = sum(ZZ(i,j) for i,j in pairs_cross) * (chi_t/4)
psi_c2 = expm(-1j*H_cross2)@psi0
rho2_12 = np.zeros((4,4),dtype=complex)
for a in range(2):      # trace qubit 0
    for b in range(2):  # trace qubit 3
        for q1 in range(2):
            for q2 in range(2):
                for q1p in range(2):
                    for q2p in range(2):
                        i=a*8+q1*4+q2*2+b
                        j=a*8+q1p*4+q2p*2+b
                        ri=q1*2+q2; ci=q1p*2+q2p
                        rho2_12[ri,ci]+=psi_c2[i]*psi_c2[j].conj()
T12 = np.real(np.trace(np.kron(sX,sX)@rho2_12))/2
T_oat = np.real(np.trace(np.kron(sX,sX)@oat_rho2_exact(N,chi_t)))/2
print(f"  T_xx (keep q1,q2): {T12:.5f}  vs oat_rho2: {T_oat:.5f}")
