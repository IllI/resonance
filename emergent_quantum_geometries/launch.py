"""
launch.py — Self-bootstrapping launcher for the JILA Pipeline GUI.

What this does, in order:
  1. Checks that Python 3.9+ is available.
  2. Creates a local virtual environment (.venv/) inside this folder if it
     doesn't already exist — nothing is installed globally.
  3. Installs all required packages into that venv from requirements_gui.txt.
  4. Checks whether 'gcloud' is available (needed only for TPU runs).
  5. Launches jila_gui.py from inside the venv.

Usage (the only command a user ever needs to type):
  python launch.py

After the first run, subsequent launches skip steps 2-3 (already done).
To force a clean reinstall, delete the .venv/ folder and run again.
"""

import sys
import os
import subprocess
import platform

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
VENV    = os.path.join(HERE, ".venv")
IS_WIN  = platform.system() == "Windows"

# Python executable inside the venv
if IS_WIN:
    VENV_PY  = os.path.join(VENV, "Scripts", "python.exe")
    VENV_PIP = os.path.join(VENV, "Scripts", "pip.exe")
else:
    VENV_PY  = os.path.join(VENV, "bin", "python")
    VENV_PIP = os.path.join(VENV, "bin", "pip")

REQS    = os.path.join(HERE, "requirements_gui.txt")
GUI     = os.path.join(HERE, "jila_gui.py")
MARKER  = os.path.join(VENV, ".jila_installed")   # written after successful install


def banner(msg):
    print(f"\n{'-'*60}\n  {msg}\n{'-'*60}")


def run(cmd, **kw):
    """Run a subprocess and raise on failure."""
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result


# ── Step 1: Check Python version ──────────────────────────────────────────────
if sys.version_info < (3, 9):
    print(f"ERROR: Python 3.9 or newer is required (you have {sys.version}).")
    print("Download Python from https://www.python.org/downloads/")
    sys.exit(1)


# ── Step 2: Create venv if needed ─────────────────────────────────────────────
if not os.path.isdir(VENV):
    banner("First run: creating isolated environment in .venv/")
    print("This takes about 10–30 seconds and only happens once.")
    run([sys.executable, "-m", "venv", VENV])
    print("  Virtual environment created.")


# ── Step 3: Install dependencies if needed ────────────────────────────────────
if not os.path.isfile(MARKER):
    banner("Installing dependencies into .venv/ (one-time setup)...")
    print("Packages: numpy, scipy, matplotlib, pillow, h5py\n")

    # Upgrade pip silently first
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # Install required packages, skip lines starting with # or blank
    pkgs = []
    with open(REQS) as f:
        for line in f:
            line = line.split("#")[0].strip()   # strip inline comments
            if line:
                pkgs.append(line)

    run([VENV_PIP, "install"] + pkgs)

    # Write marker so we don't reinstall on every launch
    with open(MARKER, "w") as f:
        f.write("installed\n")

    print("\n  All dependencies installed successfully.")
else:
    # Already installed — quick silent check to catch any corruption
    result = subprocess.run(
        [VENV_PY, "-c", "import numpy, scipy, matplotlib"],
        capture_output=True
    )
    if result.returncode != 0:
        # Something broken — force reinstall
        print("Dependency check failed — reinstalling...")
        os.remove(MARKER)
        run([VENV_PIP, "install", "-r", REQS])
        with open(MARKER, "w") as f:
            f.write("installed\n")


# ── Step 4: Check for gcloud (TPU runs only — local runs work without it) ─────
import shutil
gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
if gcloud is None:
    print(
        "\n  NOTE: 'gcloud' (Google Cloud SDK) was not found on this system.\n"
        "  The GUI will open and local CPU runs will work normally.\n"
        "  To use TPU deployment, install the Google Cloud SDK:\n"
        "    https://cloud.google.com/sdk/docs/install\n"
        "  You do NOT need to install it globally if you only run locally.\n"
    )
else:
    print(f"  gcloud found: {gcloud}")


# ── Step 5: Launch the GUI ─────────────────────────────────────────────────────
banner("Launching JILA Pipeline GUI...")
os.chdir(HERE)   # ensure relative imports (jila_pipeline, jila_results) work

# Use execv so the venv Python replaces this process — no zombie parent
if IS_WIN:
    # Windows doesn't support os.execv cleanly; use subprocess instead
    result = subprocess.run([VENV_PY, GUI])
    sys.exit(result.returncode)
else:
    os.execv(VENV_PY, [VENV_PY, GUI])
