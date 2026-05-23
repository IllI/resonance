#!/usr/bin/env python3
"""
program_l_tpu.py  —  Program L: Representation-Agnostic Adaptive Recovery
=========================================================================
Tests whether a D-LinOSS controller using only transport observables can
stabilize recoverable quantum transport under noise without knowing the
Hamiltonian or coordinate system.

Three controllers compared on identical noise trajectories:
  Static   — fixed pre-calibrated pulses (lower bound)
  HAware   — full H + noise model access   (upper bound)
  Agnostic — observable-only D-LinOSS      (test)

Pre-registered thresholds FROZEN before first run.
"""
import argparse
import json
import os
import time
import numpy as np
import scipy.linalg as la

# ── PRE-REGISTERED THRESHOLDS (FROZEN) ─────────────────────────────────────
TH_F_REC      = 2 / 3   # classical communication limit (Programs H–K)
TH_SV1_MIN    = 0.30    # PTM leading singular value floor
TH_SPEC_H_MIN = 0.40    # spectral entropy floor
TH_EE_MIN     = 0.10    # entanglement entropy floor
TH_MI_MIN     = 0.03    # mutual information floor (Program K pre-reg)
TH_BI_MAX     = 0.25    # basis-invariant residual ceiling
P_DROPOUT     = 0.30    # representation dropout probability
CTRL_INTERVAL = 5       # timesteps between control decisions

# ── OBSERVABLE VECTOR (12-dim, locked) ─────────────────────────────────────
OBS_NAMES = [
    "ptm_sv1", "ptm_sv2", "ptm_sv3", "ptm_H",
    "EE", "MI", "OS", "H_T",
    "D_eff", "front_v", "noise_slope", "basis_invar",
]
N_OBS = len(OBS_NAMES)  # 12

# ── CLIFFORD LIBRARY (6 key single-qubit gates) ─────────────────────────────
_I      = np.eye(2, dtype=complex)
_X      = np.array([[0, 1], [1, 0]], dtype=complex)
_Y      = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z      = np.array([[1, 0], [0, -1]], dtype=complex)
_Hg     = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S      = np.array([[1, 0], [0, 1j]], dtype=complex)
CLIFFORDS      = [_I, _X, _Y, _Z, _Hg, _S]
CLIFFORD_NAMES = ["I", "X", "Y", "Z", "H", "S"]
N_CLIFF = len(CLIFFORDS)
