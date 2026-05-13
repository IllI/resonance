"""
jila_gui.py — OAT Channel PTM Geometry Pipeline
JILA Integration GUI (Tkinter)

Run: python jila_gui.py
Deps: pip install numpy scipy h5py  (tkinter is built-in)
Optional: pip install nptdms (for LabVIEW .tdms files)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, subprocess, json, os, sys, shutil
import numpy as np

# ── colour palette ───────────────────────────────────────────────────────────
BG      = "#1a1a2e"
PANEL   = "#16213e"
ACCENT  = "#0f3460"
BLUE    = "#4cc9f0"
GREEN   = "#4ade80"
YELLOW  = "#fbbf24"
RED     = "#f87171"
WHITE   = "#f0f0f0"
GRAY    = "#6b7280"

TRC_ZONES = {
    "v6e-8 · europe-west4-a (spot)": ("europe-west4-a","v6e-8","tpu-vm-v6e-base"),
    "v6e-8 · us-east1-d (spot)":     ("us-east1-d",    "v6e-8","tpu-vm-v6e-base"),
    "v5e-8 · us-central1-a (spot)":  ("us-central1-a", "v5e-8","tpu-vm-v5-lite-pjrt"),
    "v4-8  · us-central2-b (on-demand)":("us-central2-b","v4-8","tpu-vm-v4-base"),
}

# ── file loaders ─────────────────────────────────────────────────────────────
def detect_and_load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        return _load_h5(path)
    elif ext == ".csv":
        return _load_csv(path)
    elif ext in (".npz", ".npy"):
        return _load_npz(path)
    elif ext == ".mat":
        return _load_mat(path)
    elif ext == ".tdms":
        return _load_tdms(path)
    elif ext == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def _load_csv(path):
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, list(rows[0].keys()) if rows else []

def _load_h5(path):
    import h5py
    rows, keys = [], []
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        n = len(np.array(f[keys[0]]))
        for i in range(n):
            rows.append({k: float(np.array(f[k])[i]) for k in keys})
    return rows, keys

def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    n = len(d[keys[0]])
    rows = [{k: float(d[k][i]) for k in keys} for i in range(n)]
    return rows, keys

def _load_mat(path):
    from scipy.io import loadmat
    d = loadmat(path)
    keys = [k for k in d if not k.startswith("_")]
    n = len(np.array(d[keys[0]]).flatten())
    rows = [{k: float(np.array(d[k]).flatten()[i]) for k in keys} for i in range(n)]
    return rows, keys

def _load_tdms(path):
    from nptdms import TdmsFile
    f = TdmsFile.read(path)
    rows, keys = [], []
    for group in f.groups():
        for ch in group.channels():
            keys.append(ch.name)
    n = min(len(f[g.name][k.name].data) for g in f.groups() for k in g.channels())
    for i in range(n):
        row = {}
        for g in f.groups():
            for ch in g.channels():
                row[ch.name] = float(ch.data[i])
        rows.append(row)
    return rows, keys

def _load_json(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return data, list(data[0].keys())
    raise ValueError("JSON must be a list of dicts")

# ── main app ─────────────────────────────────────────────────────────────────
class JILAGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OAT Channel PTM Geometry — JILA Pipeline")
        self.configure(bg=BG)
        self.geometry("920x700")
        self.resizable(True, True)
        self._rows = []
        self._col_vars = {}
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=ACCENT, pady=10)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="OAT Channel PTM Geometry Pipeline",
                 font=("Helvetica", 16, "bold"), bg=ACCENT, fg=WHITE).pack()
        tk.Label(hdr, text="JILA Rey Group · TRC-Compliant TPU Deployment",
                 font=("Helvetica", 10), bg=ACCENT, fg=BLUE).pack()

        # ── main paned ───────────────────────────────────────────────────────
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        left  = tk.Frame(pw, bg=PANEL)
        right = tk.Frame(pw, bg=PANEL)
        pw.add(left,  weight=1)
        pw.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

        # ── status bar ───────────────────────────────────────────────────────
        sb = tk.Frame(self, bg=ACCENT, pady=3)
        sb.grid(row=2, column=0, sticky="ew")
        self.status = tk.Label(sb, text="Ready — drop a data file to begin",
                               bg=ACCENT, fg=GREEN, font=("Helvetica", 9))
        self.status.pack(side=tk.LEFT, padx=8)

    def _lbl(self, parent, text, row, col=0, fg=WHITE, bold=False):
        f = ("Helvetica", 9, "bold") if bold else ("Helvetica", 9)
        tk.Label(parent, text=text, bg=PANEL, fg=fg, font=f)\
          .grid(row=row, column=col, sticky="w", padx=6, pady=2)

    def _entry(self, parent, default, row, col=1, width=14):
        v = tk.StringVar(value=default)
        e = tk.Entry(parent, textvariable=v, bg=ACCENT, fg=WHITE,
                     insertbackground=WHITE, width=width,
                     relief=tk.FLAT, font=("Helvetica", 9))
        e.grid(row=row, column=col, sticky="ew", padx=6, pady=2)
        return v

    def _build_left(self, p):
        p.columnconfigure(1, weight=1)

        # File section
        self._lbl(p, "── DATA FILE ──", 0, bold=True, fg=BLUE)
        tk.Button(p, text="Browse / Select File", command=self._browse,
                  bg=BLUE, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=4)\
          .grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.file_lbl = tk.Label(p, text="No file selected",
                                 bg=PANEL, fg=GRAY, wraplength=320,
                                 font=("Helvetica", 8))
        self.file_lbl.grid(row=2, column=0, columnspan=2, padx=6)

        # Column mapping
        self._lbl(p, "── COLUMN MAPPING ──", 3, bold=True, fg=BLUE)
        self._lbl(p, "Map file columns → pipeline parameters", 4, fg=GRAY)
        for i, (name, default) in enumerate([
            ("N  (atom number / qubits)", "N"),
            ("chi_t  (interaction param)", "chi_t"),
            ("gamma_t  (dephasing, opt.)", "gamma_t"),
            ("T_xx  (measured, opt.)", "T_xx_measured"),
            ("T_yz  (measured, opt.)", "T_yz_measured"),
        ]):
            row = 5 + i
            self._lbl(p, name, row)
            v = tk.StringVar(value=default)
            e = tk.Entry(p, textvariable=v, bg=ACCENT, fg=WHITE,
                         insertbackground=WHITE, width=18,
                         relief=tk.FLAT, font=("Helvetica", 9))
            e.grid(row=row, column=1, sticky="ew", padx=6, pady=1)
            self._col_vars[name.split()[0].rstrip("(")] = v

        # Params section
        self._lbl(p, "── PARAMETERS ──", 10, bold=True, fg=BLUE)
        self._lbl(p, "GCP Project", 11)
        self.v_project = self._entry(p, "time-emission", 11)
        self._lbl(p, "Node ID", 12)
        self.v_node    = self._entry(p, "jila-run-node",  12)
        self._lbl(p, "QR ID", 13)
        self.v_qr      = self._entry(p, "jila-run-qr",    13)
        self._lbl(p, "TPU Zone / Type", 14)
        self.v_zone = tk.StringVar(value=list(TRC_ZONES.keys())[0])
        ttk.Combobox(p, textvariable=self.v_zone,
                     values=list(TRC_ZONES.keys()),
                     state="readonly", width=30)\
          .grid(row=14, column=1, sticky="ew", padx=6, pady=2)

        # Buttons
        tk.Button(p, text="▶  Run Locally (CPU)",
                  command=lambda: threading.Thread(target=self._run_local).start(),
                  bg=GREEN, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=6)\
          .grid(row=16, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        tk.Button(p, text="🚀  Deploy to TPU (TRC)",
                  command=lambda: threading.Thread(target=self._run_tpu).start(),
                  bg=YELLOW, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=6)\
          .grid(row=17, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
        tk.Button(p, text="🗑  Delete TPU Resources (Cleanup)",
                  command=lambda: threading.Thread(target=self._cleanup).start(),
                  bg=RED, fg=WHITE, font=("Helvetica", 9),
                  relief=tk.FLAT, padx=8, pady=4)\
          .grid(row=18, column=0, columnspan=2, sticky="ew", padx=6, pady=2)

    def _build_right(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)
        tk.Label(p, text="Output / Log", bg=PANEL, fg=BLUE,
                 font=("Helvetica", 10, "bold"))\
          .grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.log = scrolledtext.ScrolledText(
            p, bg="#0d1117", fg=GREEN, font=("Courier", 8),
            relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED)
        self.log.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _log(self, msg, color=None):
        self.log.config(state=tk.NORMAL)
        tag = color or "normal"
        self.log.insert(tk.END, msg + "\n", tag)
        self.log.tag_config("ok",   foreground=GREEN)
        self.log.tag_config("warn", foreground=YELLOW)
        self.log.tag_config("err",  foreground=RED)
        self.log.tag_config("info", foreground=BLUE)
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _set_status(self, msg, color=GREEN):
        self.status.config(text=msg, fg=color)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("All supported",
                        "*.csv *.h5 *.hdf5 *.npz *.npy *.mat *.tdms *.json"),
                       ("CSV", "*.csv"), ("HDF5", "*.h5 *.hdf5"),
                       ("NumPy", "*.npz *.npy"), ("MATLAB", "*.mat"),
                       ("TDMS", "*.tdms"), ("JSON", "*.json")])
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path):
        self._log(f"Loading: {os.path.basename(path)}", "info")
        try:
            rows, cols = detect_and_load(path)
            self._rows = rows
            self._path = path
            self.file_lbl.config(
                text=f"{os.path.basename(path)}  ({len(rows)} rows)\n"
                     f"Columns: {', '.join(cols)}", fg=GREEN)
            self._log(f"✓ {len(rows)} rows  |  Columns: {cols}", "ok")
            # Auto-map columns
            for key, var in self._col_vars.items():
                for c in cols:
                    if key.lower() in c.lower():
                        var.set(c)
                        break
            self._set_status(f"Loaded {len(rows)} rows from {os.path.basename(path)}")
        except Exception as e:
            self._log(f"✗ {e}", "err")
            self._set_status(str(e), RED)

    def _build_records(self):
        mapping = {k: v.get() for k, v in self._col_vars.items()}
        records = []
        for row in self._rows:
            rec = {}
            for param, col in mapping.items():
                if col and col in row:
                    rec[param] = row[col]
            records.append(rec)
        return records

    # ── local run ─────────────────────────────────────────────────────────────
    def _run_local(self):
        if not self._rows:
            messagebox.showerror("No data", "Load a data file first.")
            return
        self._set_status("Running locally…", YELLOW)
        self._log("── Local CPU run ──", "info")
        try:
            from jila_pipeline import run_batch
            records = self._build_records()
            results = run_batch(records)
            out = os.path.join(os.path.dirname(self._path), "ptm_results.json")
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            self._log(f"✓ {len(results)} results → {out}", "ok")
            for r in results[:5]:
                if "error" not in r:
                    self._log(
                        f"  chi={r['chi_t']:.3f}  F={r['F_avg_theory']:.4f}"
                        f"  rank={r['ptm_rank']}  [{r['phase_class']}]", "ok")
            self._set_status(f"Done — {out}", GREEN)
        except Exception as e:
            self._log(f"✗ {e}", "err")
            self._set_status(str(e), RED)

    # ── TPU deployment ────────────────────────────────────────────────────────
    def _run_tpu(self):
        if not self._rows:
            messagebox.showerror("No data", "Load a data file first.")
            return

        zone_key  = self.v_zone.get()
        zone, acc, runtime = TRC_ZONES[zone_key]
        project   = self.v_project.get()
        node_id   = self.v_node.get()
        qr_id     = self.v_qr.get()
        spot_flag = ["--best-effort"] if "spot" in zone_key else []

        self._log("── TRC-Compliant TPU Deployment ──", "info")
        self._log(f"Zone: {zone}  Acc: {acc}  Project: {project}", "info")

        # Save data to temp JSON
        tmp_data = os.path.join(os.path.dirname(self._path), "_jila_tpu_input.json")
        with open(tmp_data, "w") as f:
            json.dump(self._build_records(), f)
        self._log(f"Saved input: {tmp_data}", "info")

        def run(cmd, desc):
            self._log(f"$ {' '.join(cmd)}", "warn")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.stdout: self._log(r.stdout.strip(), "ok")
            if r.stderr: self._log(r.stderr.strip(), "warn")
            if r.returncode != 0:
                raise RuntimeError(f"{desc} failed (rc={r.returncode})")

        try:
            # 1. Create QR
            self._set_status("Creating TPU queued resource…", YELLOW)
            run(["gcloud","compute","tpus","queued-resources","create", qr_id,
                 f"--node-id={node_id}", f"--project={project}",
                 f"--zone={zone}", f"--accelerator-type={acc}",
                 f"--runtime-version={runtime}","--quiet"] + spot_flag,
                "QR create")

            # 2. Wait for READY
            self._log("Waiting for node READY…", "info")
            import time
            for _ in range(30):
                r = subprocess.run(
                    ["gcloud","compute","tpus","tpu-vm","list",
                     f"--project={project}", f"--zone={zone}",
                     "--filter",f"name={node_id}", "--format=value(state)"],
                    capture_output=True, text=True)
                if "READY" in r.stdout:
                    self._log("✓ Node READY", "ok"); break
                time.sleep(10)
            else:
                raise RuntimeError("Node did not become READY in 5 min")

            # 3. Upload files
            self._set_status("Uploading files…", YELLOW)
            for src in [tmp_data, "jila_pipeline.py"]:
                run(["gcloud","compute","tpus","tpu-vm","scp",
                     src, f"{node_id}:{os.path.basename(src)}",
                     f"--project={project}", f"--zone={zone}"], "SCP")

            # 4. Install deps + run
            self._set_status("Running pipeline on TPU…", YELLOW)
            cmd_str = (
                "pip install numpy scipy -q && "
                "python3 jila_pipeline.py "
                f"--input _jila_tpu_input.json --out ptm_results.json"
            )
            run(["gcloud","compute","tpus","tpu-vm","ssh", node_id,
                 f"--project={project}", f"--zone={zone}",
                 f"--command={cmd_str}"], "Pipeline run")

            # 5. Download results
            self._set_status("Downloading results…", YELLOW)
            out = os.path.join(os.path.dirname(self._path), "ptm_results.json")
            run(["gcloud","compute","tpus","tpu-vm","scp",
                 f"{node_id}:ptm_results.json", out,
                 f"--project={project}", f"--zone={zone}"], "SCP results")

            with open(out) as f:
                results = json.load(f)
            self._log(f"✓ {len(results)} results → {out}", "ok")
            for r in results[:5]:
                if "error" not in r:
                    self._log(
                        f"  chi={r['chi_t']:.3f}  F={r['F_avg_theory']:.4f}"
                        f"  rank={r['ptm_rank']}  [{r['phase_class']}]", "ok")
            self._set_status(f"✓ Done — {out}", GREEN)
            self._log("⚠ CLEANUP REQUIRED — click Delete TPU Resources!", "warn")

        except Exception as e:
            self._log(f"✗ {e}", "err")
            self._set_status(str(e), RED)
            self._log("⚠ Manual cleanup may be needed!", "warn")

    # ── cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self):
        project = self.v_project.get()
        node_id = self.v_node.get()
        qr_id   = self.v_qr.get()
        zone, _, _ = TRC_ZONES[self.v_zone.get()]

        if not messagebox.askyesno("Confirm Cleanup",
            f"Delete:\n  Node: {node_id}\n  QR: {qr_id}\n  Zone: {zone}\n\n"
            "This is required by TRC rules. Proceed?"):
            return

        self._log("── TRC Cleanup ──", "warn")
        for cmd, desc in [
            (["gcloud","compute","tpus","tpu-vm","delete", node_id,
              f"--project={project}", f"--zone={zone}", "--quiet"], "Delete node"),
            (["gcloud","compute","tpus","queued-resources","delete", qr_id,
              f"--project={project}", f"--zone={zone}", "--quiet"], "Delete QR"),
        ]:
            self._log(f"$ {' '.join(cmd)}", "warn")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.stdout: self._log(r.stdout.strip(), "ok")
            if r.returncode == 0:
                self._log(f"✓ {desc}", "ok")
            else:
                self._log(f"✗ {desc}: {r.stderr.strip()}", "err")

        # Verify zero resources
        r = subprocess.run(
            ["gcloud","compute","tpus","queued-resources","list",
             f"--project={project}", f"--zone={zone}"],
            capture_output=True, text=True)
        self._log("Remaining QRs:\n" + (r.stdout.strip() or "(none)"), "ok")
        self._set_status("Cleanup complete — verify billing page", GREEN)


if __name__ == "__main__":
    app = JILAGui()
    app.mainloop()
