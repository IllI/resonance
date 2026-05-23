"""
scripts/capture_gui_screenshots.py

Launches jila_gui.py, drives each step programmatically, and saves
real window screenshots to docs/images/ for the user guide.

Run from the repo root:
    python scripts/capture_gui_screenshots.py
"""
import sys, os, time, json, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Windows console fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from PIL import ImageGrab
from unittest.mock import patch, MagicMock
import numpy as np

# ── output dir ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

FIXTURE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "fixtures", "oat_phase_diagram_gam0.csv")


def grab_window(widget, name, pad=6):
    """Capture a real screenshot of a Tk widget's bounding box."""
    widget.update_idletasks()
    widget.update()
    time.sleep(0.4)   # allow rendering to settle
    x = widget.winfo_rootx() - pad
    y = widget.winfo_rooty() - pad
    w = widget.winfo_width()  + pad * 2
    h = widget.winfo_height() + pad * 2
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    path = os.path.join(OUT_DIR, f"{name}.png")
    img.save(path)
    print(f"  ✓ saved {path}")
    return path


def run_capture():
    from jila_gui import JILAGui
    from jila_results import ResultsWindow

    print("Launching GUI…")
    app = JILAGui()
    app.deiconify()           # make visible
    app.lift()
    app.focus_force()
    app.update()
    time.sleep(0.6)

    # ── Step 1: Initial state ──────────────────────────────────────────────────
    print("Step 1: Initial launch")
    grab_window(app, "01_initial_launch")

    # ── Step 2: Load data file ─────────────────────────────────────────────────
    print("Step 2: Loading data file")
    app._load_file(FIXTURE_CSV)
    app.update()
    time.sleep(0.3)
    grab_window(app, "02_data_loaded")

    # ── Step 3: Column mapping panel ──────────────────────────────────────────
    print("Step 3: Column mapping visible")
    grab_window(app, "03_column_mapping")

    # ── Step 4: Running locally ────────────────────────────────────────────────
    print("Step 4: Local run in progress")
    # Patch show_results so the results window doesn't open automatically
    with patch("jila_gui.show_results"):
        app._run_local()
    app.update()
    time.sleep(0.3)
    grab_window(app, "04_local_run_complete")

    # ── Step 5: Results window ─────────────────────────────────────────────────
    print("Step 5: Results window")
    results_win = ResultsWindow(app, app._last_results, FIXTURE_CSV)
    results_win.deiconify()
    results_win.lift()
    results_win.focus_force()
    results_win.update()
    time.sleep(0.6)
    grab_window(results_win, "05_results_window")

    # ── Step 6: Plots tab ──────────────────────────────────────────────────────
    print("Step 6: Plots tab")
    try:
        nb = results_win._notebook
        nb.select(0)    # Plots tab
        results_win.update()
        time.sleep(0.4)
        grab_window(results_win, "06_plots_tab")
    except Exception as e:
        print(f"  (skipped plots tab: {e})")

    # ── Step 7: Data table tab ────────────────────────────────────────────────
    print("Step 7: Data table tab")
    try:
        nb.select(1)    # Table tab
        results_win.update()
        time.sleep(0.3)
        grab_window(results_win, "07_data_table")
    except Exception as e:
        print(f"  (skipped table tab: {e})")

    # ── Step 8: Export buttons ────────────────────────────────────────────────
    print("Step 8: Export panel")
    try:
        nb.select(0)
        results_win.update()
        time.sleep(0.2)
        grab_window(results_win, "08_export_panel")
    except Exception as e:
        print(f"  (skipped export panel: {e})")

    # ── Step 9: TPU tab ───────────────────────────────────────────────────────
    print("Step 9: TPU deployment tab")
    try:
        # Switch to main app TPU tab
        app.lift(); app.focus_force()
        for tab_id in app._notebook.tabs():
            if "TPU" in app._notebook.tab(tab_id, "text"):
                app._notebook.select(tab_id)
                break
        app.update(); time.sleep(0.3)
        grab_window(app, "09_tpu_tab")
    except Exception as e:
        print(f"  (skipped TPU tab: {e})")

    # ── Step 10: Download export (show file dialog mock screenshot) ───────────
    print("Step 10: Results with export buttons highlighted")
    results_win.lift(); results_win.focus_force()
    results_win.update(); time.sleep(0.2)
    grab_window(results_win, "10_export_ready")

    print("\nAll screenshots saved to:", OUT_DIR)
    results_win.destroy()
    app.destroy()


if __name__ == "__main__":
    run_capture()
