import json, csv, os, sys

os.makedirs("tests/fixtures", exist_ok=True)

with open("teleport_results.json") as f:
    r = json.load(f)

# Part B gamma=0 slice
rows = [x for x in r["B"] if x["gamma_t"] == 0.0]
with open("tests/fixtures/oat_phase_diagram_gam0.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N","chi_t","gamma_t","F_sim","F_theory"])
    w.writeheader()
    for x in rows:
        w.writerow({"N":4,"chi_t":round(x["chi_t"],6),"gamma_t":0.0,
                    "F_sim":round(x["F_teleport"],6),"F_theory":round(x["F_theory"],6)})

# Part E noise sweep
with open("tests/fixtures/oat_noise_robustness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N","chi_t","gamma_t","p_dep","F_sim","F_theory"])
    w.writeheader()
    for x in r["E"]:
        w.writerow({"N":4,"chi_t":1.45626,"gamma_t":0.0,
                    "p_dep":round(x["p_dep"],3),"F_sim":round(x["F_star"],6),
                    "F_theory":round(x["F_theory"],6)})

# Part A anchor points
with open("tests/fixtures/oat_anchor_points.json", "w") as f:
    json.dump({"chi_t_star": r["A"]["chi_t_star"],
               "chi_t_pi":   r["A"]["chi_t_pi"]}, f, indent=2)

print(f"oat_phase_diagram_gam0.csv  ({len(rows)} rows)")
print(f"oat_noise_robustness.csv    ({len(r['E'])} rows)")
print("oat_anchor_points.json")
