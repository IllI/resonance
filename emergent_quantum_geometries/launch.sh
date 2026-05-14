#!/bin/bash
# JILA Pipeline GUI Launcher (macOS / Linux)
# Run: bash launch.sh   (or chmod +x launch.sh && ./launch.sh)
#
# Requirements: Python 3.9+ must be installed.
#   macOS:  brew install python3   OR   download from python.org
#   Linux:  sudo apt install python3 python3-venv  (Ubuntu/Debian)
#           sudo dnf install python3               (Fedora/RHEL)
#
# Everything else (numpy, scipy, matplotlib, etc.) is installed
# automatically into a local .venv/ folder — nothing goes global.

cd "$(dirname "$0")"
python3 launch.py
