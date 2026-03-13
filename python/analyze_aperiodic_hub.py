import json
import sys
import numpy as np
from pathlib import Path

def analyze_anomalies(json_path):
    print(f"--- Aperiodic Anomaly Analysis ---\n")
    with open(json_path, 'r') as f:
        data = json.load(f)
        events = data.get('events', data) # Handle both list and dict-with-events formats
    
    # Extract classes and their event timestamps
    classes = {}
    for event in events:
        c = event['class_id']
        t = event['host_time_mid']
        if c not in classes:
            classes[c] = []
        classes[c].append(t)
    
    # Calculate stats per class
    class_stats = {}
    total_events = len(events)
    print(f"Total Phase Shift Events: {total_events}")
    
    # We are looking for the 'Hubble' classes identified previously: 2, 3, 14
    hubble_classes = [2, 3, 14]
    
    for c, times in classes.items():
        times = sorted(times)
        if len(times) > 1:
            diffs = np.diff(times)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            cv = std_diff / mean_diff if mean_diff > 0 else float('inf')
            
            # Identify 'aperiodic' via negative recurrence or extremely high CV
            has_negative = any(d < 0 for d in diffs) # Should be sorted, so diffs are positive, but just in case
            
            class_stats[c] = {
                'mean': mean_diff,
                'std': std_diff,
                'cv': cv,
                'count': len(times)
            }
            
    # Sort classes by Coefficient of Variation (CV) to find the most "aperiodic"
    sorted_by_cv = sorted(class_stats.items(), key=lambda x: x[1]['cv'], reverse=True)
    
    print("\n--- Top 3 Most Aperiodic Rhythms (Highest Variance) ---")
    aperiodic_classes = []
    for c, stats in sorted_by_cv[:3]:
        print(f"Class {c}: Mean Period={stats['mean']:.4f}s, CV={stats['cv']:.4f}, Count={stats['count']}")
        # 14 is one of our Hubble classes! If it's highly aperiodic, that's incredibly interesting.
        aperiodic_classes.append((c, stats['mean']))
        
    print("\n--- Hubble Signature Classes ---")
    for hc in hubble_classes:
        if hc in class_stats:
            stats = class_stats[hc]
            print(f"Class {hc}: Mean Period={stats['mean']:.4f}s, CV={stats['cv']:.4f}, Count={stats['count']}")

    print("\n--- Interference Analysis (Aperiodic vs Hubble) ---")
    # Check beat frequencies between the most aperiodic classes and the Hubble classes
    f_hubble = [1.0/class_stats[hc]['mean'] for hc in hubble_classes if hc in class_stats]
    
    for ac, amean in aperiodic_classes:
        if ac in hubble_classes:
            print(f"* Note: Class {ac} is BOTH a Hubble Signature AND Highly Aperiodic.")
            continue
            
        f_a = 1.0 / amean
        for hc in hubble_classes:
            if hc in class_stats:
                f_h = 1.0 / class_stats[hc]['mean']
                # Beat frequency
                f_beat = abs(f_a - f_h)
                if f_beat > 0:
                    t_beat = 1.0 / f_beat
                    print(f"Beat Frequency between Aperiodic Class {ac} and Hubble Class {hc}: {t_beat:.4f}s")
                    
                    # Does this beat match any other observed class?
                    for oc, ostats in class_stats.items():
                        if abs(ostats['mean'] - t_beat) < 0.5: # Half second tolerance
                            print(f"  -> MATCHES Class {oc} (Mean: {ostats['mean']:.4f}s)!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_aperiodic_hub.py <json_path>")
        sys.exit(1)
    analyze_anomalies(sys.argv[1])
