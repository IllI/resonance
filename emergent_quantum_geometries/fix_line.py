lines = open('recovery_phase_diagram.py', encoding='utf-8').readlines()
for i, l in enumerate(lines):
    if 'Gam' in l and 'print' in l and i > 100:
        lines[i] = '    print("  chi_t/Gam", end="")\n'
        print("Fixed line", i+1, repr(lines[i]))
        break
open('recovery_phase_diagram.py', 'w', encoding='utf-8').writelines(lines)
print("Done")
