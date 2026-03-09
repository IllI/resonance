"""
Validates bitwistor_orbit.py mathematical consistency.
Tests that:
1. Minkowski metric satisfies the hyperboloid constraint
2. OIS is bounded [0, 1]
3. Lorentz distance is symmetric and non-negative
4. Bi-Twistor grok signal approaches 0 at convergence
"""

import numpy as np
from bitwistor_orbit import (
    MinkowskiHyperbolicMap,
    calculate_qpc_centroid,
    calculate_orbit_radius,
    orbit_incidence_score,
    calc_twistor_coherence,
    bitwistor_grok_signal,
    extract_trial_features
)

def test_hyperboloid_constraint():
    """Every projected point must satisfy -x0^2 + x1^2 + x2^2 + x3^2 = -R^2."""
    m = MinkowskiHyperbolicMap()
    for _ in range(1000):
        h = np.random.uniform(0.01, 1.0)
        b = np.random.uniform(0.01, 3.0)
        k = np.random.uniform(0.01, 10.0)
        R = np.random.uniform(0.1, 50.0)
        v = m.project_ghost_basin(h, b, k, R)
        
        lhs = -v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2
        rhs = -(max(0.001, R))**2
        
        assert abs(lhs - rhs) < 1e-8, f"Hyperboloid constraint violated: {lhs} != {rhs}"
    print("PASS: Hyperboloid constraint holds for 1000 random vectors")

def test_lorentz_symmetry():
    """Lorentz distance must be symmetric and non-negative."""
    m = MinkowskiHyperbolicMap()
    for _ in range(500):
        v1 = m.project_ghost_basin(*np.random.uniform(0.1, 5.0, 4))
        v2 = m.project_ghost_basin(*np.random.uniform(0.1, 5.0, 4))
        
        d12 = m.lorentz_distance(v1, v2)
        d21 = m.lorentz_distance(v2, v1)
        
        assert d12 >= 0, f"Negative distance: {d12}"
        assert abs(d12 - d21) < 1e-10, f"Asymmetric: {d12} != {d21}"
    print("PASS: Lorentz distance is symmetric and non-negative for 500 pairs")

def test_ois_bounds():
    """OIS must be in [0, 1]."""
    for theta in np.linspace(0, 10, 100):
        for r in np.linspace(0.1, 5, 50):
            ois = orbit_incidence_score(theta, r)
            assert 0.0 <= ois <= 1.0, f"OIS out of bounds: {ois} (theta={theta}, r={r})"
    print("PASS: OIS bounded [0, 1] for 5000 parameter combinations")

def test_ois_monotonic():
    """OIS must decrease monotonically as distance increases."""
    r = 2.0
    prev_ois = 1.0
    for theta in np.linspace(0, 5, 100):
        ois = orbit_incidence_score(theta, r)
        assert ois <= prev_ois + 1e-10, f"OIS not monotonically decreasing"
        prev_ois = ois
    print("PASS: OIS is monotonically decreasing with distance")

def test_twistor_coherence_bounds():
    """TCS components must be non-negative."""
    for _ in range(1000):
        theta = np.random.uniform(0, 5)
        rt = np.random.uniform(0.5, 5.0)
        rt_med = np.random.uniform(1.0, 4.0)
        k = np.random.uniform(-5, 5)
        n = np.random.randint(100, 50000)
        s = np.random.uniform(0.5, 3.0)
        
        tcs, peep, phase = calc_twistor_coherence(theta, rt, rt_med, k, n, s)
        
        assert tcs >= 0, f"Negative TCS: {tcs}"
        assert 0 <= peep <= 1, f"Peephole out of range: {peep}"
        assert 0 <= phase <= 1, f"Phase out of range: {phase}"
    print("PASS: TCS, Peephole, Phase all non-negative for 1000 random inputs")

def test_grok_signal_convergence():
    """Grok signal must approach 0 when all conditions are met."""
    signal = bitwistor_grok_signal(
        loss_derivative=0.0,
        rt_derivative=0.0,
        ois=1.0
    )
    assert abs(signal) < 1e-10, f"Grok signal not zero at convergence: {signal}"
    
    signal_exploring = bitwistor_grok_signal(
        loss_derivative=0.5,
        rt_derivative=0.3,
        ois=0.2
    )
    assert signal_exploring > 0, f"Grok signal should be positive during exploration: {signal_exploring}"
    print("PASS: Grok signal correctly converges to 0 and stays positive during exploration")

def test_qpc_centroid_stability():
    """QPC centroid must be stable under small perturbations."""
    m = MinkowskiHyperbolicMap()
    base_vectors = [m.project_ghost_basin(0.5, 1.0, 2.0, 8.0 + i*0.01) for i in range(20)]
    centroid1 = calculate_qpc_centroid(base_vectors, m)
    
    perturbed_vectors = [m.project_ghost_basin(0.5, 1.0, 2.0, 8.0 + i*0.01 + np.random.normal(0, 0.001)) for i in range(20)]
    centroid2 = calculate_qpc_centroid(perturbed_vectors, m)
    
    drift = m.lorentz_distance(centroid1, centroid2)
    assert drift < 0.01, f"QPC centroid unstable under small perturbation: drift={drift}"
    print(f"PASS: QPC centroid stable (drift={drift:.6f} under 0.001 noise)")

def test_extract_trial_features():
    """Trial feature extraction must handle edge cases."""
    # Normal case
    block = np.random.randn(100, 3)
    features = extract_trial_features(block, 100)
    assert features is not None
    assert features['Energy'] > 0
    assert features['S_trial'] > 0
    assert features['N_Active'] == 100
    
    # Empty case
    empty_block = np.array([])
    features = extract_trial_features(empty_block, 0)
    assert features is None
    print("PASS: Trial feature extraction handles normal and edge cases")

if __name__ == "__main__":
    print("=" * 60)
    print(" MATHEMATICAL VALIDATION: bitwistor_orbit.py")
    print("=" * 60)
    print()
    
    test_hyperboloid_constraint()
    test_lorentz_symmetry()
    test_ois_bounds()
    test_ois_monotonic()
    test_twistor_coherence_bounds()
    test_grok_signal_convergence()
    test_qpc_centroid_stability()
    test_extract_trial_features()
    
    print()
    print("=" * 60)
    print(" ALL TESTS PASSED: Math is internally consistent.")
    print(" No assumptions about data correctness were made.")
    print("=" * 60)
