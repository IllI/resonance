import numpy as np

data = np.load('dlinoss_training/training_library.npz', allow_pickle=True)
print('=== FULL RESULTS SUMMARY ===')
for sys_name in data.files:
    recs = data[sys_name]
    print(f'\n--- {sys_name} ({len(recs)} records) ---')
    for r in recs:
        r = r.item() if hasattr(r, 'item') else r
        gk = r.get('gamma_k', [])
        gdom = float(max(gk)) if len(gk) else 0
        ke = r.get('K_eff', 0)
        if sys_name == 'OAT':
            print(f"  N={r['N']:2d} G={r['Gamma']:.3f}  K={ke}  gdom={gdom:.4f}")
        elif sys_name == 'SYK4':
            print(f"  Nf={r['Nf']}  b={r['beta']}  K={ke}  gdom={gdom:.4f}")
        elif sys_name == 'XXZ':
            yvals = r['y'] if isinstance(r['y'], list) else list(r['y'])
            print(f"  L={r['L']}  W={r['W']}  K={ke}  I_inf={float(yvals[-1]):.4f}  gdom={gdom:.4f}")
        elif sys_name == 'Dicke':
            print(f"  N={r['N_spins']}  g/gc={r['g_ratio']}  K={ke}  gdom={gdom:.4f}")
