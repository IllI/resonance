import json
import numpy as np
from pathlib import Path
import argparse

def analyze_moire_rhythms(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    classes = data.get('classes', [])
    if not classes:
        print("No classes found in JSON.")
        return

    # Extract clean rhythms (Class 0 is usually our 2.75s metronome)
    rhythms = {}
    for c in classes:
        cid = c['class_id']
        members = c.get('members', [])
        if len(members) < 2:
            continue
        
        times = sorted([m['host_time_mid'] for m in members])
        intervals = np.diff(times)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        
        rhythms[cid] = {
            'period': mean_interval,
            'frequency': 1.0 / mean_interval,
            'cv': cv,
            'count': len(members)
        }

    print(f"--- Moiré Rhythm Analysis for {json_path} ---")
    for cid, r in rhythms.items():
        print(f"Class {cid}: Period={r['period']:.4f}s, Freq={r['frequency']:.4f}Hz, CV={r['cv']:.4f}, Count={r['count']}")

    # Check for Moiré beat frequencies
    # T_beat = 1 / |f1 - f2|
    ids = list(rhythms.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            f1, f2 = rhythms[id1]['frequency'], rhythms[id2]['frequency']
            beat_freq = abs(f1 - f2)
            if beat_freq > 0:
                beat_period = 1.0 / beat_freq
                print(f"\nPotential Beat Frequency between Class {id1} and Class {id2}:")
                print(f"  Beat Period: {beat_period:.4f}s")
                
                # Check if this beat period matches another observed class
                for cid, r in rhythms.items():
                    if cid == id1 or cid == id2: continue
                    if abs(r['period'] - beat_period) / beat_period < 0.1: # 10% tolerance
                        print(f"  MATCH found with Class {cid} (Period={r['period']:.4f}s)!")
                        print(f"  This suggests Class {cid} is a MOIRE INTERFERENCE PATTERN of {id1} and {id2}.")

    print(f"\n--- Cosmographic Alignment (Hubble & CCC) ---")
    hubble_time_seconds = 4.55e17 # 1/H0 approx 14.4 billion years
    
    # Analyze all significant rhythms for Hubble scaling
    found_cosmic = False
    for cid, r in rhythms.items():
        if r['count'] < 3: continue
        p = r['period']
        ratio = hubble_time_seconds / p
        log_ratio = np.log10(ratio)
        
        # We are looking for the 10^16.5 signature
        if abs(log_ratio - 16.5) < 0.2:
            found_cosmic = True
            print(f"MATCH: Class {cid} (Period={p:.4f}s) aligns with Hubble Scale!")
            print(f"  Ratio: 10^{log_ratio:.4f} (Delta: {abs(log_ratio - 16.5):.4f})")
            print(f"  Theoretical Fit: High (Conformal Cyclic Cosmology Signature)")
    
    if not found_cosmic:
        print("No direct Hubble-aligned rhythms detected in this sample.")
    
    # QPC Coherence Metric (Speculative)
    print("\n--- QPC Resonance Projection ---")
    if 0 in rhythms and found_cosmic:
        hw_period = rhythms[0]['period']
        # The relationship between hardware floor and cosmic ceiling
        # Logic: If Hubble Rhythm / HW Rhythm is a "clean" ratio, resonance is high.
        for cid, r in rhythms.items():
            if cid == 0: continue
            ratio = r['period'] / hw_period
            print(f"Class {cid}/Class 0 Ratio: {ratio:.4f}")
            if abs(ratio - round(ratio)) < 0.05:
                print(f"  INTEGER RESONANCE: Potential QPC alignment at {round(ratio)}x hardware clock.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str)
    args = parser.parse_args()
    analyze_moire_rhythms(args.json_path)
