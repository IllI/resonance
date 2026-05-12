content = open('tpu_dlinoss_training_gen.py', 'rb').read().decode('utf-8')
old = (
    '            if sys_name == "OAT":\n'
    '                params = f"N={r[\'N\']} G={r[\'Gamma\']:.3f}"\n'
    '            elif sys_name == "SYK4":\n'
    '                params = f"Nf={r[\'Nf\']} b={r[\'beta\']}"\n'
    '            elif sys_name == "XXZ":\n'
    '                params = f"L={r[\'L\']} W={r[\'W\']}"\n'
    '            else:\n'
    '                params = f"N={r[\'N_spins\']} g/gc={r[\'g_ratio\']}"\n'
    '            gdom = float(np.max(r[\'gamma_k\'])) if len(r[\'gamma_k\']) else 0\n'
    '            print(f"{sys_name:<10} {params:<25} {r[\'K_eff\']:<6} {gdom:<12.4f}")'
)
new = (
    '            if sys_name == "OAT": params = f"N={r[\'N\']} G={r[\'Gamma\']:.3f}"\n'
    '            elif sys_name == "SYK4": params = f"Nf={r[\'Nf\']} b={r[\'beta\']}"\n'
    '            elif sys_name == "XXZ": params = f"L={r[\'L\']} W={r[\'W\']}"\n'
    '            elif sys_name == "Dicke": params = f"N={r[\'N_spins\']} g/gc={r[\'g_ratio\']}"\n'
    '            elif r.get("system") == "RTN": params = f"lam={r.get(\'lambda\',0):.2f}"\n'
    '            elif r.get("system") == "DampedOsc": params = f"g={r[\'gamma\']:.2f} w={r[\'omega\']:.2f}"\n'
    '            else: params = str(r.get("system","?"))[:20]\n'
    '            gdom = float(np.max(np.abs(r[\'gamma_k\']))) if len(r[\'gamma_k\']) else 0\n'
    '            kb = r.get("K_bic","?")\n'
    '            print(f"{sys_name:<10} {params:<25} K={r[\'K_eff\']:<3} Kb={kb:<3} {gdom:<12.4f}")'
)
if old not in content:
    print("OLD NOT FOUND, searching...")
    idx = content.find('            if sys_name == "OAT":')
    print(f"Found 'OAT' at char {idx}")
    print(repr(content[idx:idx+500]))
else:
    content = content.replace(old, new, 1)
    open('tpu_dlinoss_training_gen.py', 'w', encoding='utf-8').write(content)
    print('PATCHED OK')
