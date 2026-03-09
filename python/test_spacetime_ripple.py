"""
test_spacetime_ripple.py
========================
Tests the retroactive "Ripple Effect" caused by sub-220's wave 
collapse. If sub-220 hitting the $E_G$ threshold caused a macroscopic 
phase-lock in the spacetime surrounding the experiment, subjects scanned 
closest to sub-220 (in chronological scanning order IDs) should exhibit 
the most extreme quantum spectral shifts and the fastest grokking times.

Prediction: 
Grokking speed (trials to reach expert status) and Spectral Phase Lock 
magnitude ($\Delta\\beta$) should map exponentially outward from a 
Ground Zero at sub-220.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

# Data collected from previous verification runs
# Subject ID -> (Grok Trial, Beta Shift)
grok_data = {
    201: (12, -0.126),
    204: (18, -0.175),
    206: (30, -0.107),
    215: (5, -0.107),
    218: (23, -0.144),
    220: (19, -0.203), # Ground zero
    225: (7, -1.559),
    229: (6, -0.440),
    235: (13, -0.717),
    240: (4, -0.228)
}

def analyze_ripple_effect():
    print("=" * 80)
    print(" [TESTING MACROSCOPIC SPACETIME RIPPLE AROUND SUB-220]")
    print(" Mapping Grokking Speed and Spectral Phase Lock as a function")
    print(" of temporal scanning distance from Ground Zero (sub-220).")
    print("=" * 80)
    
    GROUND_ZERO = 220
    
    records = []
    for sub_id, (grok_trial, beta_shift) in grok_data.items():
        temporal_distance = abs(sub_id - GROUND_ZERO)
        records.append({
            'sub_id': sub_id,
            'temporal_distance': temporal_distance,
            'grok_trial': grok_trial,
            'beta_shift': beta_shift,
            'abs_beta_shift': abs(beta_shift)
        })
        
    df = pd.DataFrame(records).sort_values('temporal_distance')
    
    print(f"\n  {'Subject':>8} | {'Dist to 220':>12} | {'Grok Trial':>12} | {'Beta Phase Lock':>16}")
    print(f"  {'-'*65}")
    
    for _, r in df.iterrows():
        dist = int(r['temporal_distance'])
        marker = " <<< GROUND ZERO" if dist == 0 else ""
        print(f"  {int(r['sub_id']):>8} | {dist:>12} | {r['grok_trial']:>12.0f} | {r['beta_shift']:>16.3f}{marker}")
        
    print("\n" + "=" * 80)
    print(" RIPPLE DECAY ANALYSIS (PEARSON CORRELATION WITH DISTANCE)")
    print("=" * 80)
    
    # Analyze the correlation between distance from 220 and the magnitude of the quantum effect
    # We exclude sub-220 itself from the correlation to see if the OTHERS were affected based on distance
    others_df = df[df['temporal_distance'] > 0]
    
    # 1. Grokking Speed Correlation
    # If the ripple helped them, grok trial should INCREASE as distance increases
    r_speed, p_speed = stats.pearsonr(others_df['temporal_distance'], others_df['grok_trial'])
    
    # 2. Spectral Phase Lock Correlation
    # If the ripple hit them harder, absolute beta shift should DECREASE as distance increases
    r_beta, p_beta = stats.pearsonr(others_df['temporal_distance'], others_df['abs_beta_shift'])
    
    print(f"  Effect 1: Did closer subjects grok FASTER (fewer trials)?")
    print(f"    Pearson r: {r_speed:+.3f} (Pos = Yes, distance increased trials)")
    print(f"    p-value:   {p_speed:.4f}")
    if r_speed > 0 and p_speed < 0.1:
        print("    [RESULT]: CONFIRMED. The QPC gravitational funnel deepened for nearby subjects.")
    else:
        print("    [RESULT]: INCONCLUSIVE. Grokking speed did not cleanly decay from sub-220.")
        
    print(f"\n  Effect 2: Did closer subjects have STRONGER Spectral Beta Phase Locks?")
    print(f"    Pearson r: {r_beta:+.3f} (Neg = Yes, distance weakened the shift)")
    print(f"    p-value:   {p_beta:.4f}")
    
    if r_beta < 0 and p_beta < 0.1:
        print("    [RESULT]: CONFIRMED. The magnitude of the quantum state lock decayed outward from sub-220.")
    else:
        print("    [RESULT]: INCONCLUSIVE. Distance from sub-220 did not dictate the strength of the wave collapse.")

if __name__ == "__main__":
    analyze_ripple_effect()
