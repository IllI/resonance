import json
import numpy as np
import pandas as pd

# Load the N12 results JSON
filepath = r"c:\Users\cityz\IllI\newer_all\emergent_quantum_geometries\program_v2_results\program_v2_N12_results.json"
with open(filepath, 'r') as f:
    data = json.load(f)

print("Arguments used in this run:")
for k, v in data['args'].items():
    print(f"  {k}: {v}")

# Parse records into a pandas DataFrame
records = data['records']
df = pd.DataFrame(records)

print(f"\nTotal records: {len(df)}")
print("Unique J_std:", df['J_std'].unique())
print("Unique T2:", df['T2'].unique())
print("Unique distance classes:", df['dist_class'].unique())
print("Unique controllers:", df['controller'].unique())

# Group the data and calculate averages
grouped = df.groupby(['J_std', 'T2', 'dist_class', 'controller', 'tau']).mean(numeric_only=True).reset_index()

# Write a nice report of findings
print("\n" + "="*80)
print("1. SUMMARY OF CONTROLLERS BY PERFORMANCE (overall averages)")
print("="*80)
overall_perf = df.groupby(['controller']).mean(numeric_only=True)[['F_ent', 'mutual_information', 'bloch_similarity', 'n_interventions']]
print(overall_perf.to_markdown())

print("\n" + "="*80)
print("2. COMPARISON OF BASES vs ADAPTIVE (TAU SWEEP)")
print("="*80)
# Group adaptive by tau to see how threshold affects it
adaptive_tau = df[df['controller'] == 'vector_adaptive'].groupby('tau').mean(numeric_only=True)[['F_ent', 'mutual_information', 'bloch_similarity', 'n_interventions']]
print("Vector Adaptive by tau threshold:")
print(adaptive_tau.to_markdown())

print("\n" + "="*80)
print("3. REGIME TRANSITIONS: REST RAINT-OPTIMAL VS INTERVENTION-BENEFICIAL")
print("="*80)
# For each setting (J_std, T2, dist_class), find which controller is best and if adaptive beats free/static/dd
for (j, t2, dist), group in df.groupby(['J_std', 'T2', 'dist_class']):
    print(f"\n--- Regime: J_std={j:.2f}, T2={t2:.1f}, Dist={dist} ---")
    
    # Get mean performance of standard baselines
    free_perf = group[group['controller'] == 'free']['F_ent'].mean()
    static_perf = group[group['controller'] == 'static']['F_ent'].mean()
    dd_perf = group[group['controller'] == 'dd']['F_ent'].mean()
    
    # Get best adaptive tau
    adaptive_group = group[group['controller'] == 'vector_adaptive']
    if not adaptive_group.empty:
        best_adaptive_idx = adaptive_group.groupby('tau')['F_ent'].mean().idxmax()
        best_adaptive_fent = adaptive_group.groupby('tau')['F_ent'].mean().max()
        best_adaptive_gates = adaptive_group[adaptive_group['tau'] == best_adaptive_idx]['n_interventions'].mean()
    else:
        best_adaptive_idx = None
        best_adaptive_fent = 0
        best_adaptive_gates = 0
        
    print(f"  F_ent: Free={free_perf:.4f} | Static={static_perf:.4f} | DD={dd_perf:.4f} | Best Adaptive (tau={best_adaptive_idx})={best_adaptive_fent:.4f} (gates={best_adaptive_gates:.1f})")
    
    # Determine the regime type
    # If Free is the absolute best or within 1% of the best, it's restraint-optimal (or passive coherent transport dominant)
    best_fent = max(free_perf, static_perf, dd_perf, best_adaptive_fent)
    if free_perf >= best_fent - 0.01:
        regime = "RESTRAINT-OPTIMAL (Passive coherence is sufficient/best)"
    elif best_adaptive_fent > free_perf and best_adaptive_fent > static_perf and best_adaptive_fent > dd_perf:
        regime = f"INTERVENTION-BENEFICIAL: ADAPTIVE (beats passive by {best_adaptive_fent - free_perf:.4f})"
    elif static_perf >= best_fent - 0.01:
        regime = "INTERVENTION-BENEFICIAL: STATIC (Static control is best)"
    else:
        regime = "INTERVENTION-BENEFICIAL: DD (Dynamical Decoupling is best)"
    print(f"  Verdict: {regime}")

print("\n" + "="*80)
print("4. DETAILED ANALYSIS OF DISORDER J_std ON PERFORMANCE")
print("="*80)
j_effects = df.groupby(['J_std', 'controller']).mean(numeric_only=True)[['F_ent', 'n_interventions']]
print(j_effects.to_markdown())

print("\n" + "="*80)
print("5. DETAILED ANALYSIS OF T2 (DEPHASING) ON PERFORMANCE")
print("="*80)
t2_effects = df.groupby(['T2', 'controller']).mean(numeric_only=True)[['F_ent', 'n_interventions']]
print(t2_effects.to_markdown())
