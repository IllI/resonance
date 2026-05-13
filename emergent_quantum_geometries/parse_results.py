import json
with open('teleport_results.json') as f:
    r = json.load(f)

b = r['B']
above = [(x['chi_t'], x['gamma_t'], x['F_teleport']) for x in b if x['F_teleport'] > 2/3]
below = [(x['chi_t'], x['gamma_t'], x['F_teleport']) for x in b if x['F_teleport'] <= 2/3]
print(f"Above classical (F>2/3): {len(above)} points")
print(f"Below classical: {len(below)} points")

gam0 = sorted([x for x in b if x['gamma_t']==0], key=lambda x: x['chi_t'])
print("\ngam=0 slice:")
for x in gam0:
    m = '> 2/3 QUANTUM' if x['F_teleport'] > 2/3 else '<= 2/3 classical'
    print(f"  chi={x['chi_t']:.3f}  F={x['F_teleport']:.4f}  {m}")

print("\nPart C V_Q vs F (full sweep):")
for c in r['C']:
    print(f"  chi={c['chi_t']:.3f}  V_Q={c['V_Q']:.4f}  F={c['F_teleport']:.4f}  theory={c['F_theory']:.4f}")
