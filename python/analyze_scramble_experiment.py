"""
analyze_scramble_experiment.py
==============================
Post-hoc analysis for scrambling-triggered injection experiment.
Loads alice_scramble_events.json and bob_passive_events.json,
computes cross-correlation of event timestamps, and runs a
permutation test for event clustering.

Usage:
    python python/analyze_scramble_experiment.py \
        --alice /path/to/alice_scramble_events.json \
        --bob   /path/to/bob_passive_events.json
"""

import argparse, json, pathlib
import numpy as np
from scipy import stats

def load(path):
    return json.loads(pathlib.Path(path).read_text())

def event_cross_correlation(alice_events, bob_events, total_duration,
                             lag_range=(-60, 60), lag_step=1.0):
    """
    For each lag τ, count how many Alice events have a Bob event within ±Δt=5s.
    Returns (lags, counts, chance_level).
    """
    alice_t = np.array([e["elapsed"] for e in alice_events])
    bob_t   = np.array([e["elapsed"] for e in bob_events])
    
    lags = np.arange(lag_range[0], lag_range[1] + lag_step, lag_step)
    counts = np.zeros(len(lags))
    delta_t = 5.0  # window half-width

    for i, lag in enumerate(lags):
        shifted_alice = alice_t + lag
        matched = 0
        for at in shifted_alice:
            if np.any(np.abs(bob_t - at) <= delta_t):
                matched += 1
        counts[i] = matched

    # Chance level: expected matches if events were uniformly distributed
    n_A = len(alice_t)
    n_B = len(bob_t)
    p_match_chance = 1.0 - (1.0 - 2 * delta_t / total_duration) ** n_B
    chance_level = n_A * p_match_chance

    return lags, counts, chance_level

def permutation_test(alice_events, bob_events, total_duration,
                     best_lag, n_perm=10000, delta_t=5.0):
    """
    Test if observed matching at best_lag exceeds chance via permutation.
    Randomly shifts Alice's events within [0, total_duration].
    """
    alice_t = np.array([e["elapsed"] for e in alice_events])
    bob_t   = np.array([e["elapsed"] for e in bob_events])
    n_A = len(alice_t)

    def count_matches(a_times):
        matched = sum(1 for at in a_times if np.any(np.abs(bob_t - at) <= delta_t))
        return matched

    observed = count_matches(alice_t + best_lag)

    rng = np.random.default_rng(42)
    perm_counts = np.zeros(n_perm)
    for i in range(n_perm):
        # Random shift: jitter each Alice event uniformly in [0, total_duration]
        shuffled = rng.uniform(0, total_duration, n_A)
        perm_counts[i] = count_matches(shuffled)

    p_val = (perm_counts >= observed).mean()
    return observed, perm_counts, p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alice", default="alice_scramble_events.json")
    parser.add_argument("--bob",   default="bob_passive_events.json")
    args = parser.parse_args()

    alice = load(args.alice)
    bob   = load(args.bob)

    print("=" * 60)
    print("SCRAMBLING-TRIGGERED INJECTION ANALYSIS")
    print("=" * 60)
    print(f"Alice: {len(alice['events'])} scrambling events over {alice['duration']:.0f}s")
    print(f"Bob:   {len(bob['events'])} phase-shift events over {bob['duration']:.0f}s")
    print(f"Wall-clock offset Alice-Bob: {alice['t0'] - bob['t0']:.2f}s")

    if not alice["events"] or not bob["events"]:
        print("ERROR: No events in one or both datasets.")
        return

    total_dur = min(alice["duration"], bob["duration"])

    # 1. Cross-correlation scan
    print("\nComputing cross-correlation (lag -60s to +60s)...")
    lags, counts, chance = event_cross_correlation(
        alice["events"], bob["events"], total_dur)

    best_lag_idx = np.argmax(counts)
    best_lag = lags[best_lag_idx]
    best_count = counts[best_lag_idx]

    print(f"\nCross-correlation peak:")
    print(f"  Best lag: {best_lag:+.1f}s")
    print(f"  Matches at best lag: {best_count:.0f} / {len(alice['events'])}")
    print(f"  Chance level: {chance:.1f}")
    print(f"  Observed / Chance: {best_count / chance:.2f}x")

    # Near-zero lag (simultaneous / causal)
    zero_lag_count = counts[np.argmin(np.abs(lags))]
    print(f"\nAt lag=0s: {zero_lag_count:.0f} matches (chance={chance:.1f})")

    # 2. Permutation test at best lag
    print(f"\nRunning permutation test at lag={best_lag:+.1f}s (N=10,000)...")
    obs, perm_dist, p_val = permutation_test(
        alice["events"], bob["events"], total_dur, best_lag)

    print(f"\nPERMUTATION TEST RESULT:")
    print(f"  Observed matches: {obs}")
    print(f"  Permutation mean: {perm_dist.mean():.2f} ± {perm_dist.std():.2f}")
    print(f"  p-value (one-tailed): {p_val:.4f}")
    print(f"  {'SIGNIFICANT *** (p<0.05)' if p_val < 0.05 else 'not significant'}")

    # 3. Inter-event interval comparison
    alice_gaps = np.diff([e["elapsed"] for e in alice["events"]])
    bob_gaps   = np.diff([e["elapsed"] for e in bob["events"]])
    print(f"\nInter-event intervals:")
    print(f"  Alice: mean={alice_gaps.mean():.1f}s  std={alice_gaps.std():.1f}s")
    print(f"  Bob:   mean={bob_gaps.mean():.1f}s  std={bob_gaps.std():.1f}s")

    # 4. Save results
    results = {
        "alice_events": len(alice["events"]),
        "bob_events": len(bob["events"]),
        "best_lag_s": float(best_lag),
        "best_match_count": int(best_count),
        "chance_level": float(chance),
        "ratio_obs_chance": float(best_count / chance),
        "zero_lag_count": int(zero_lag_count),
        "permutation_p": float(p_val),
        "permutation_mean": float(perm_dist.mean()),
        "significant": p_val < 0.05,
        "cross_correlation": {"lags": lags.tolist(), "counts": counts.tolist(),
                              "chance": float(chance)},
    }
    out = pathlib.Path("resonance_results/scramble_experiment_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
