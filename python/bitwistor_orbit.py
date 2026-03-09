"""
bitwistor_orbit.py
==================
Core module implementing the Bi-Twistor Orbit Mechanism for mapping
the QPC gravity well onto physical Minkowski space.

This module provides the mathematical functions for:
1. Minkowski projection of Ghost Basin vectors
2. Dynamic QPC centroid tracking with orbit radius
3. Orbit Incidence Score (OIS) calculation
4. Bi-Twistor convergence signal for D-LinOSS pruning

All functions are pure mathematical transformations.
No data loading, no side effects, no assumptions about correctness labels.
The data tells us what it tells us.
"""

import numpy as np
from scipy.stats import kurtosis


# ====================================================================== #
#  MINKOWSKI METRIC (Reusable, tested in analyze_moving_qpc.py)
# ====================================================================== #

class MinkowskiHyperbolicMap:
    """Projects electromagnetic brain states onto a Minkowski hyperboloid.
    
    The Minkowski metric eta = diag(-1, 1, 1, 1) enforces that all points
    exist strictly outside the Wuji singularity (R=0), stabilized by
    their own metabolic energy expenditure.
    """

    def project_ghost_basin(self, hurst_h, psd_beta, spatial_kurtosis, energy_scalar):
        """Project a trial's electromagnetic state onto the Minkowski hyperboloid.
        
        Args:
            hurst_h: Fractal memory (Hurst exponent), range [0, 1]
            psd_beta: Power spectral density slope, range [0, 2+]
            spatial_kurtosis: Spatial distribution shape
            energy_scalar: Total metabolic energy (radius from singularity)
        
        Returns:
            4-vector [x0, x1, x2, x3] on the hyperboloid satisfying
            -x0^2 + x1^2 + x2^2 + x3^2 = -R^2
        """
        h = max(0.001, hurst_h)
        b = max(0.001, psd_beta)
        k = max(0.001, spatial_kurtosis)
        R = max(0.001, energy_scalar)
        x1 = (h / 1.0) * R
        x2 = (b / 2.0) * R
        x3 = (k / 1.0) * R
        x0 = np.sqrt(R**2 + x1**2 + x2**2 + x3**2)
        return np.array([x0, x1, x2, x3])

    def minkowski_dot(self, u, v):
        """Minkowski inner product with signature (-,+,+,+)."""
        return -u[0]*v[0] + u[1]*v[1] + u[2]*v[2] + u[3]*v[3]

    def lorentz_distance(self, u, v):
        """Geodesic distance on the hyperboloid (Lorentz model).
        
        Returns the arccosh of the normalized Minkowski dot product,
        giving the proper distance between two points on the hyperboloid.
        """
        Ru = np.sqrt(max(1e-10, -self.minkowski_dot(u, u)))
        Rv = np.sqrt(max(1e-10, -self.minkowski_dot(v, v)))
        dot = self.minkowski_dot(u, v)
        val = max(1.0, -dot / (Ru * Rv))
        return np.arccosh(val)


# ====================================================================== #
#  QPC CENTROID AND ORBIT RANGE
# ====================================================================== #

def calculate_qpc_centroid(vectors, minkowski):
    """Calculate the Minkowski center-of-mass of a set of Ghost Basin vectors.
    
    This is the dynamic QPC position, updated as more Expert observations
    accumulate. Returns None if no vectors are provided.
    """
    if not vectors:
        return None
    mean_vec = np.mean(vectors, axis=0)
    mean_R = np.mean([
        np.sqrt(max(1e-10, -minkowski.minkowski_dot(v, v)))
        for v in vectors
    ])
    qpc_x1, qpc_x2, qpc_x3 = mean_vec[1], mean_vec[2], mean_vec[3]
    qpc_x0 = np.sqrt(mean_R**2 + qpc_x1**2 + qpc_x2**2 + qpc_x3**2)
    return np.array([qpc_x0, qpc_x1, qpc_x2, qpc_x3])


def calculate_orbit_radius(vectors, centroid, minkowski):
    """Calculate the 95% confidence orbit radius of the QPC gravity well.
    
    The orbit radius defines the boundary of the QPC's projective shadow
    in physical Minkowski space. Ghost Basins inside this radius are
    considered "captured" by the gravity well.
    
    r_orbit = mean(d) + 2 * std(d)
    
    This is the Bi-Twistor "flipped light cone" base.
    """
    if not vectors or centroid is None:
        return 0.1  # minimum nonzero radius
    
    distances = [minkowski.lorentz_distance(v, centroid) for v in vectors]
    
    if len(distances) < 2:
        return max(0.1, distances[0] if distances else 0.1)
    
    return max(0.1, np.mean(distances) + 2.0 * np.std(distances))


# ====================================================================== #
#  ORBIT INCIDENCE SCORE (OIS)
# ====================================================================== #

def orbit_incidence_score(theta_minkowski, orbit_radius):
    """Calculate the Orbit Incidence Score (OIS).
    
    OIS = max(0, 1 - theta_Minkowski / r_orbit)
    
    Args:
        theta_minkowski: Lorentz distance from Ghost Basin to QPC centroid
        orbit_radius: The QPC's projective orbit range
    
    Returns:
        OIS in [0, 1]. 
        OIS > 0: Ghost Basin is inside the QPC gravity well (passive entanglement)
        OIS = 0: Ghost Basin is outside (active scanning required)
    """
    if orbit_radius <= 0:
        return 0.0
    return max(0.0, 1.0 - (theta_minkowski / orbit_radius))


# ====================================================================== #
#  TWISTOR COHERENCE (Forward Cone - kept for comparison)
# ====================================================================== #

def calc_twistor_coherence(theta_minkowski, rt, rt_median, spatial_kurtosis, n_active, s_trial):
    """Calculate the forward Twistor Coherence Score (TCS).
    
    This is the classical (non-flipped) twistor projection:
    TCS = Power * Peephole_Alignment * Phase_Factor
    
    where:
        Power = (N_active / 10000) * S_trial
        Peephole_Alignment = 1 / cosh(theta_Mink * R_riemann)
        Phase_Factor = cos(theta_dyn)
    """
    # Riemann Sphere breadth (Penrose cone width)
    r_riemann = 1.0 + abs(spatial_kurtosis)
    
    # Penrose cone mismatch
    delta_twistor = theta_minkowski * r_riemann
    peephole_alignment = 1.0 / np.cosh(delta_twistor)
    
    # Dynamic phase from Goedel ray causality (RT-based)
    if np.isnan(rt) or rt_median <= 0:
        theta_dyn = 0.0
    else:
        theta_dyn = (np.pi / 2.0) * min(1.0, abs(1.0 - (rt / rt_median)))
    phase_factor = max(0.0, np.cos(theta_dyn))
    
    # Structural power
    n_norm = n_active / 10000.0
    power = n_norm * s_trial
    
    tcs = power * peephole_alignment * phase_factor
    
    return tcs, peephole_alignment, phase_factor


# ====================================================================== #
#  BI-TWISTOR CONVERGENCE SIGNAL (for D-LinOSS pruning)
# ====================================================================== #

def bitwistor_grok_signal(loss_derivative, rt_derivative, ois, alpha=1.0, beta=1.0, gamma=1.0):
    """Calculate the Bi-Twistor convergence signal for pruning.
    
    Delta_Grok = alpha * dL/dt + beta * dRT/dt + gamma * (1 - OIS)
    
    When this approaches zero, the model has:
    - Converged in loss (alpha term)
    - Stabilized reaction time (beta term) 
    - Entered the QPC gravity well (gamma term)
    
    All three conditions must be satisfied simultaneously.
    """
    return alpha * loss_derivative + beta * rt_derivative + gamma * (1.0 - ois)


# ====================================================================== #
#  TRIAL-LEVEL FEATURE EXTRACTION
# ====================================================================== #

def extract_trial_features(block_data, n_active):
    """Extract trial-level biological antenna features from a BOLD block.
    
    Args:
        block_data: 2D array [n_voxels, n_timepoints] of z-scored BOLD signal
        n_active: Number of active voxels in the mask
    
    Returns:
        dict with Energy, Kurtosis, S_trial (burst sharpness)
    """
    if block_data.size == 0:
        return None
    
    trial_energy = np.mean(np.abs(block_data)) * 10.0
    
    # Find peak timepoint
    peak_tr = np.argmax(np.max(block_data, axis=0))
    voxels_at_peak = block_data[:, peak_tr]
    trial_kurtosis = kurtosis(voxels_at_peak)
    
    # Burst sharpness: ratio of peak amplitude to mean amplitude
    mean_amp = np.mean(np.abs(block_data))
    peak_amp = np.mean(np.abs(voxels_at_peak))
    s_trial = peak_amp / (mean_amp + 1e-5)
    
    return {
        'Energy': trial_energy,
        'Kurtosis': trial_kurtosis,
        'S_trial': s_trial,
        'N_Active': n_active
    }
