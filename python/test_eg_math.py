import torch
import math

# Constants
G = 6.67430e-11 # Gravitational constant (m^3 kg^-1 s^-2)
hbar = 1.054571817e-34 # Reduced Planck constant (J s)
m_trp_da = 204.23 # Mass of Tryptophan in Daltons
kg_per_da = 1.66053906660e-27 # Da to kg
m_trp = m_trp_da * kg_per_da # ~3.39e-25 kg
delta_x = 2.5e-15 # Fermi separation of atomic nuclei in superposition (m)

def calculate_E_G(N_active_trp):
    # Total superposed mass
    M_superposed = N_active_trp * m_trp
    
    # E_G = G * M^2 / delta_x
    E_G_joules = G * (M_superposed ** 2) / delta_x
    
    # Decoherence time tau = hbar / E_G
    if E_G_joules > 0:
        tau_seconds = hbar / E_G_joules
    else:
        tau_seconds = float('inf')
        
    return E_G_joules, tau_seconds

# Let's see what values we get for N_active_trp
for N in [1e6, 1e9, 1e12, 1e15]:
    e_g, tau = calculate_E_G(N)
    print(f"Wattage (N_trp): {N:.0e} | Mass: {N*m_trp:.2e} kg | E_G: {e_g:.2e} Joules | Tau: {tau:.2e} s")
