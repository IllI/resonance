import numpy as np
import scipy.linalg as la

# Load original function with UTF-8
src = open('tpu_dlinoss_training_gen.py', encoding='utf-8-sig').read().split('if __name__')[0]
ns = {}
exec(compile(src, 'tpu_dlinoss_training_gen', 'exec'), ns)
oat_rho2_exact = ns['oat_rho2_exact']

def C_wootters(rho):
    sy = np.array([[0,-1j],[1j,0]])
    ss = np.kron(sy,sy)
    rt = ss @ rho.conj() @ ss
    R = la.sqrtm(rho @ rt)
    e = sorted(np.sqrt(np.maximum(np.real(la.eigvals(R)), 0)), reverse=True)
    return max(0., e[0]-e[1]-e[2]-e[3])

grid = np.linspace(0.05, 6.0, 120)
print(f"{'N':>4}  {'C_peak':>8}  {'chi_t*':>7}  {'rho[0,3]':>12}")
print("-"*45)
for N in [2, 4, 6, 8, 12, 16, 32]:
    Cs = [C_wootters(oat_rho2_exact(N, t)) for t in grid]
    idx = np.argmax(Cs)
    rho = oat_rho2_exact(N, grid[idx])
    print(f"  {N:2d}  {Cs[idx]:8.5f}  {grid[idx]:7.4f}  |rho[0,3]|={abs(rho[0,3]):.5f}")

# Also check: what basis element corresponds to singlet?
print()
print("Checking rho structure at N=4, chi_t=0.355:")
rho4 = oat_rho2_exact(4, 0.355)
print("Full matrix:")
for i in range(4):
    row = [f"{rho4[i,j]:+.4f}" for j in range(4)]
    print(f"  {row}")
print(f"Trace={np.trace(rho4).real:.6f}")
print(f"Is Hermitian: {np.allclose(rho4, rho4.conj().T)}")
