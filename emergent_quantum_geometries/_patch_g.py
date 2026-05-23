src = open('teleport_program_g.py').read()

# Fix ibm_comparison_table to show both standard and optimal fidelity
old = (
    "    print(f\"{'Region':<22} {'chi_t':>7} {'PTM_rank':>9} {'F_avg':>7} \"\n"
    "          f\"{'F_e':>7} {'I(R:B)':>7} {'F_haar':>8} {'F_min':>7} {'stratum':>12}\")\n"
    "    print(\"-\" * 92)"
)
new = (
    "    print(f\"{'Region':<22} {'chi_t':>7} {'rank':>5} {'F_opt':>6} \"\n"
    "          f\"{'F_std':>6} {'F_haar':>7} {'F_min':>6} {'F_e':>6} {'I(R:B)':>7} {'stratum':>12}\")\n"
    "    print(\"-\" * 92)"
)
src = src.replace(old, new)

# Fix the print inside the loop
old2 = (
    "        print(f\"{label:<22} {chi_t:>7.4f} {rank:>9d} {F_avg:>7.4f} \"\n"
    "              f\"{F_e:>7.4f} {I_RB:>7.4f} {hf['mean']:>8.4f} \"\n"
    "              f\"{hf['min']:>7.4f} {st:>12}\")"
)
new2 = (
    "        # F_std: standard protocol (singlet fraction)\n"
    "        rho_r = rho2_exact(chi_t, N, 0.0)\n"
    "        phi_p = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)\n"
    "        F_std_theory = float(np.real(np.dot(phi_p.conj(), rho_r @ phi_p)))\n"
    "        F_std_theory = (1 + 2*F_std_theory)/3\n"
    "        print(f\"{label:<22} {chi_t:>7.4f} {rank:>5d} {F_avg:>6.4f} \"\n"
    "              f\"{F_std_theory:>6.4f} {hf['mean']:>7.4f} {hf['min']:>6.4f} \"\n"
    "              f\"{F_e:>6.4f} {I_RB:>7.4f} {st:>12}\")"
)
src = src.replace(old2, new2)

# Update the legend
old3 = "    print(\"  Classical teleportation bound: F_avg=2/3, F_e=1/2, I(R:B)=0\")"
new3 = (
    "    print(\"  F_opt = optimal protocol (f_max formula, uses best local pre-rotation)\")\n"
    "    print(\"  F_std = standard Bell meas (no pre-rotation, what IBM hardware runs)\")\n"
    "    print(\"  F_haar = simulation mean over 300 Haar states (should match F_std)\")\n"
    "    print(\"  Classical bound: F_std=F_opt=2/3, F_e=1/2, I(R:B)=0\")"
)
src = src.replace(old3, new3)

open('teleport_program_g.py','w').write(src)
print('patched ok')
