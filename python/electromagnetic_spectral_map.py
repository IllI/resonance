"""
electromagnetic_spectral_map.py
──────────────────────────────────────────────────────────────
A mathematical utility for the D-LinOSS framework that models the 
physical realities of biological brain resonance.

It utilizes look-up tables for MRI scanner magnet strengths and specific
cytoarchitectural (neuron type) physics to reconstruct the missing variables
in an fMRI experiment. 

Given an observed BOLD spike, it calculates:
1. The estimated absolute ATP metabolic energy.
2. The number of specific localized neurons required to fire in harmony 
   to cross the ambient measurement threshold.
3. The absolute physical volume that those non-contiguous neurons occupy.

This acts as a "detective's tool" to piece together the underlying scale 
of the physiological network without relying on human labels.
"""

import math

class ElectromagneticSpectralMap:
    def __init__(self):
        # ────────────────────────────────────────────────────────
        # 1. MRI Scanner Thresholds (The Measurement Tool)
        # ────────────────────────────────────────────────────────
        # Determines the baseline detection sensitivity. Higher Tesla
        # magnets have higher Signal-to-Noise Ratios (SNR), reducing 
        # the minimum biological energy needed to register a 'bright voxel'.
        # baseline_detection_pW: Baseline threshold in picowatts (pW) of 
        # local metabolic power required to induce a 1% BOLD signal change.
        self.MAGNET_THRESHOLDS = {
            "1.5T": {"snr_multiplier": 0.5, "baseline_detection_pW": 450_000_000.0},
            "3T":   {"snr_multiplier": 1.0, "baseline_detection_pW": 220_000_000.0}, # Standard clinical/research
            "7T":   {"snr_multiplier": 2.8, "baseline_detection_pW": 80_000_000.0},  # UHR (e.g., De Martino Auditory)
            "11.7T":{"snr_multiplier": 5.5, "baseline_detection_pW": 30_000_000.0}   # Cutting-edge
        }

        # ────────────────────────────────────────────────────────
        # 2. Cytoarchitectural Dictionary (The Biological Strings)
        # ────────────────────────────────────────────────────────
        # Defines the microscopic physical properties of neurons depending
        # on their anatomical region.
        # energy_per_ap_pJ: Energy per Action Potential in picoJoules.
        # volume_per_cell_um3: Physical size of soma + local dendritic tree in cubic micrometers.
        # base_frequency_hz: The resonant frequency band the network naturally forms.
        self.NEURON_DICTIONARY = {
            "motor_cortex": {
                "cell_type": "Betz Pyramidal (Layer V)",
                "energy_per_ap_pJ": 25.0,     # Massive axons require huge ATP expenditure
                "volume_per_cell_um3": 65000, 
                "base_frequency_hz": "Beta (13-30Hz)"
            },
            "occipital_cortex": {
                "cell_type": "Small Pyramidal & Interneurons",
                "energy_per_ap_pJ": 8.5,      # Smaller visual pathway cells
                "volume_per_cell_um3": 12000,
                "base_frequency_hz": "Gamma (30-80Hz)"
            },
            "auditory_subcortical_mgb": {
                "cell_type": "Relay Thalamocortical",
                "energy_per_ap_pJ": 12.0,     # Dense subcortical clusters
                "volume_per_cell_um3": 15000,
                "base_frequency_hz": "Alpha (8-13Hz)" # Spanning the early primary broadcast
            },
            "prefrontal_cortex": {
                "cell_type": "Spindle / Medium Pyramidal",
                "energy_per_ap_pJ": 14.0,     
                "volume_per_cell_um3": 25000,
                "base_frequency_hz": "Theta (4-8Hz)"
            }
        }

    def evaluate_resonance_spike(self, region_key, magnet_strength, bold_integral_z, active_duration_sec, coil_channels=32):
        """
        Calculates the physical reality of a neural spike detected by D-LinOSS.
        Applies Physics Understanding Correction (PUC) for the Observer Effect.
        """
        if region_key not in self.NEURON_DICTIONARY:
            raise KeyError(f"Region '{region_key}' not found in Cytoarchitectural Dictionary.")
        if magnet_strength not in self.MAGNET_THRESHOLDS:
            raise KeyError(f"Magnet '{magnet_strength}' not found. Choose from {list(self.MAGNET_THRESHOLDS.keys())}")

        magnet = self.MAGNET_THRESHOLDS[magnet_strength]
        anatomy = self.NEURON_DICTIONARY[region_key]

        # --- PHYSICS UNDERSTANDING CORRECTION (PUC) ---
        # The Observer Effect: Measurement apparatus (RF coils) create 
        # electromagnetic interference (Faraday shielding effect).
        # A 32-channel coil has more copper loops than a 12-channel coil,
        # proportionally dampening the endogenous low-Hz biological signal detected.
        # We mathematically recover the TRUE underlying field energy.
        interference_factor = 1.0 + (coil_channels * 0.005) # e.g. 1.16 for 32ch
        true_bold_integral_z = bold_integral_z * interference_factor

        # 1. Macroscopic Energy Output (Thermodynamics)
        # Calculates the total estimated power (in picowatts) bridging the detection threshold
        # Total Power = Magnet Threshold * True BOLD Z-score Integral
        estimated_power_pW = magnet["baseline_detection_pW"] * true_bold_integral_z
        total_energy_pJ = estimated_power_pW * active_duration_sec

        # 2. Microscopic Neuronal Count
        # How many total action potentials were fired in the window?
        total_aps = total_energy_pJ / anatomy["energy_per_ap_pJ"]
        
        # For a synchronized "Ghost" response, assume an average burst rate of 25 spikes/sec per neuron
        assumed_spikes_per_neuron = 25 * active_duration_sec
        active_neurons = total_aps / assumed_spikes_per_neuron

        # 3. Distributed Spatial Footprint (Network Volume)
        # Since Neural Field theory states this is a non-contiguous resonant field, this volume
        # represents the absolute cubic millimeters of *pure cellular mass* utilized, which may be 
        # scattered like a constellation across the broader spatial area rather than a solid blob.
        volume_um3 = active_neurons * anatomy["volume_per_cell_um3"]
        volume_mm3 = volume_um3 / 1_000_000_000 # Convert um^3 to mm^3

        return {
            "region": region_key,
            "magnet": magnet_strength,
            "detected_energy_pJ": round(total_energy_pJ, 2),
            "estimated_neurons": int(active_neurons),
            "distributed_network_volume_mm3": round(volume_mm3, 6),
            "cell_type": anatomy["cell_type"],
            "harmonic_band": anatomy["base_frequency_hz"]
        }


if __name__ == "__main__":
    mapper = ElectromagneticSpectralMap()

    print("=========================================================")
    print("  ELECTROMAGNETIC SPECTRAL TOPOLOGY CALIBRATION")
    print("=========================================================")

    # Test 1: Simulating the 7T Auditory MGB response
    # We detected a strong early phase shift (~15.6s processing window, BOLD integral ~4.5z)
    mgb_7t = mapper.evaluate_resonance_spike(
        region_key="auditory_subcortical_mgb",
        magnet_strength="7T",
        bold_integral_z=4.5, 
        active_duration_sec=15.6
    )

    print("\n[Case 1: De Martino Auditory Experiment - MGB Spark]")
    print(f"Scanner     : {mgb_7t['magnet']}")
    print(f"Cell Type   : {mgb_7t['cell_type']}")
    print(f"Frequency   : {mgb_7t['harmonic_band']}")
    print(f"Energy (ATP): {mgb_7t['detected_energy_pJ']:,.2f} pJ")
    print(f"Est. Neurons: {mgb_7t['estimated_neurons']:,} synchronizing nodes")
    print(f"Net. Volume : {mgb_7t['distributed_network_volume_mm3']} mm^3 scattered field footprint")

    # Test 2: Simulating the Haxby 3T Occipital response
    # Older, weaker 3T scanner, processing a 24-second window, BOLD integral ~12.0z
    vis_3t = mapper.evaluate_resonance_spike(
        region_key="occipital_cortex",
        magnet_strength="3T",
        bold_integral_z=12.0,
        active_duration_sec=24.0
    )

    print("\n[Case 2: Haxby Visual Experiment - Occipital Activation]")
    print(f"Scanner     : {vis_3t['magnet']}")
    print(f"Cell Type   : {vis_3t['cell_type']}")
    print(f"Frequency   : {vis_3t['harmonic_band']}")
    print(f"Energy (ATP): {vis_3t['detected_energy_pJ']:,.2f} pJ")
    print(f"Est. Neurons: {vis_3t['estimated_neurons']:,} synchronizing nodes")
    print(f"Net. Volume : {vis_3t['distributed_network_volume_mm3']} mm^3 scattered field footprint")

    print("\n=> UTILITY READY. Can be used to retroactively trace historical datasets and define structural requirements for learning in future simulations.")
