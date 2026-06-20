#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
SOURCE="$ROOT/../aq_trappist1e_dlinoss_stellar_state_residual_0"
INPUT=${1:-$SOURCE/results/shards/dlinoss_equal_supervision_capacity_test_0.npz}
OUTPUT=${2:-$ROOT/results/mera_spectral_capacity_test_0_tpu.json}
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax/mera_spectral_capacity_0}
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$(dirname "$OUTPUT")"
python3 "$ROOT/run_capacity.py" --npz "$INPUT" --out "$OUTPUT" --require-eight-devices
