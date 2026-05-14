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
import time as _t
import numpy as np

try:
    from jila_results import show_results
except ImportError:
    def show_results(parent, results, path):
        pass

try:
    from jila_status_panel import TRCQueue, StatusPanel
except ImportError:
    class TRCQueue:
        def __init__(self, *a, **kw): pass
        def acquire(self): return True
        def release(self): pass
    StatusPanel = None

# ── colour palette ────────────────────────────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
BLUE   = "#4cc9f0"
GREEN  = "#4ade80"
YELLOW = "#fbbf24"
RED    = "#f87171"
WHITE  = "#f0f0f0"
GRAY   = "#6b7280"

# ── TRC-locked configuration (TRC Bible, Bob slot) ───────────────────────────
TRC_ZONE    = "europe-west4-a"
TRC_ACC     = "v6e-8"
TRC_RUNTIME = "v2-alpha-tpuv6e"
TRC_ALL_ZONES = [
    "europe-west4-a", "europe-west4-b",
    "us-east1-d", "us-central1-a", "us-central2-b",
]
MAX_WAIT_MIN = 15   # fail-fast: bail after 15 min
POLL_SEC     = 20   # probe QR state every 20 s

# ── file loaders ─────────────────────────────────────────────────────────────
def detect_and_load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):   return _load_h5(path)
    elif ext == ".csv":           return _load_csv(path)
    elif ext in (".npz", ".npy"): return _load_npz(path)
    elif ext == ".mat":           return _load_mat(path)
    elif ext == ".tdms":          return _load_tdms(path)
    elif ext == ".json":          return _load_json(path)
    else: raise ValueError(f"Unsupported format: {ext}")

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

# ── main app ──────────────────────────────────────────────────────────────────
class JILAGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OAT Channel PTM Geometry — JILA Pipeline")
        self.configure(bg=BG)
        self.geometry("1100x700")
        self.resizable(True, True)
        # data state
        self._rows        = []
        self._col_vars    = {}
        self._path        = ""
        self._last_results = []
        # TPU state
        self._tpu_active  = False
        self._tpu_proc    = None
        self._tpu_gcloud  = shutil.which("gcloud.CMD") or shutil.which("gcloud")
        self._cancel_tpu  = False
        self._tpu_node    = ""
        self._tpu_qr      = ""
        self._sp          = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── window close ─────────────────────────────────────────────────────────
    def _on_close(self):
        """Hides window immediately; runs TRC zone sweep in background thread."""
        if self._tpu_active:
            ans = messagebox.askyesno(
                "TPU Job Running",
                "A TPU job is currently active.\n\n"
                "Closing will delete all TPU resources (TRC Commandments 4+5).\n\n"
                "Close and clean up?")
            if not ans:
                return
        self.withdraw()
        def _cleanup_then_destroy():
            self._emergency_tpu_cleanup()
            self.after(0, self.destroy)
        threading.Thread(target=_cleanup_then_destroy, daemon=True).start()

    def _emergency_tpu_cleanup(self):
        """Full all-zone sweep. Runs in background thread. Uses --format=json."""
        import json as _ej
        project = self.v_project.get() if hasattr(self, "v_project") else "time-emission"
        gcloud  = self._tpu_gcloud or "gcloud"
        self._tpu_active = False
        for z in TRC_ALL_ZONES:
            try:
                rq = subprocess.run(
                    [gcloud, "compute", "tpus", "queued-resources", "list",
                     f"--project={project}", f"--zone={z}", "--format=json"],
                    capture_output=True, text=True, timeout=20,
                    shell=(sys.platform == "win32"))
                for item in _ej.loads(rq.stdout or "[]"):
                    name = item.get("name", "").split("/")[-1]
                    if name:
                        subprocess.run(
                            [gcloud, "compute", "tpus", "queued-resources", "delete",
                             name, f"--project={project}", f"--zone={z}", "--quiet"],
                            capture_output=True, timeout=120,
                            shell=(sys.platform == "win32"))
                rv = subprocess.run(
                    [gcloud, "compute", "tpus", "tpu-vm", "list",
                     f"--project={project}", f"--zone={z}", "--format=json"],
                    capture_output=True, text=True, timeout=20,
                    shell=(sys.platform == "win32"))
                for item in _ej.loads(rv.stdout or "[]"):
                    name = item.get("name", "").split("/")[-1]
                    if name:
                        subprocess.run(
                            [gcloud, "compute", "tpus", "tpu-vm", "delete",
                             name, f"--project={project}", f"--zone={z}", "--quiet"],
                            capture_output=True, timeout=120,
                            shell=(sys.platform == "win32"))
            except Exception:
                pass
        try:
            TRCQueue(os.getpid(), self._tpu_node, self._tpu_qr).release()
        except Exception:
            pass

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        hdr = tk.Frame(self, bg=ACCENT, pady=10)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="OAT Channel PTM Geometry Pipeline",
                 font=("Helvetica", 16, "bold"), bg=ACCENT, fg=WHITE).pack()
        tk.Label(hdr, text="JILA Rey Group · TRC-Compliant TPU Deployment",
                 font=("Helvetica", 10), bg=ACCENT, fg=BLUE).pack()
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        left  = tk.Frame(pw, bg=PANEL)
        right = tk.Frame(pw, bg=PANEL)
        pw.add(left, weight=1)
        pw.add(right, weight=2)
        self._build_left(left)
        self._build_right(right)
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
        self._lbl(p, "── DATA FILE ──", 0, bold=True, fg=BLUE)
        tk.Button(p, text="Browse / Select File", command=self._browse,
                  bg=BLUE, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=4)\
          .grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.file_lbl = tk.Label(p, text="No file selected",
                                 bg=PANEL, fg=GRAY, wraplength=320,
                                 font=("Helvetica", 8))
        self.file_lbl.grid(row=2, column=0, columnspan=2, padx=6)
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
            tk.Entry(p, textvariable=v, bg=ACCENT, fg=WHITE,
                     insertbackground=WHITE, width=18,
                     relief=tk.FLAT, font=("Helvetica", 9))\
              .grid(row=row, column=1, sticky="ew", padx=6, pady=1)
            self._col_vars[name.split()[0].rstrip("(")] = v
        self._lbl(p, "── PARAMETERS ──", 10, bold=True, fg=BLUE)
        self._lbl(p, "GCP Project", 11)
        self.v_project = self._entry(p, "time-emission", 11)
        self._lbl(p, "Node ID", 12)
        self.v_node    = self._entry(p, "jila-run-node", 12)
        self._lbl(p, "QR ID", 13)
        self.v_qr      = self._entry(p, "jila-run-qr",   13)
        self._lbl(p, "TPU Zone / Type", 14)
        tk.Label(p, text=f"{TRC_ACC}  ·  {TRC_ZONE} (spot) — TRC locked",
                 bg=PANEL, fg=YELLOW, font=("Helvetica", 9, "italic"))\
          .grid(row=14, column=1, sticky="w", padx=6, pady=2)
        tk.Button(p, text="▶  Run Locally (CPU)",
                  command=lambda: threading.Thread(target=self._run_local, daemon=True).start(),
                  bg=GREEN, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=6)\
          .grid(row=16, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        tk.Button(p, text="🚀  Deploy to TPU (TRC)",
                  command=lambda: threading.Thread(target=self._run_tpu, daemon=True).start(),
                  bg=YELLOW, fg=BG, font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, padx=8, pady=6)\
          .grid(row=17, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
        tk.Button(p, text="🗑  Delete TPU Resources (Cleanup)",
                  command=self._cleanup_start,
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
        self.log.tag_config("ok",   foreground=GREEN)
        self.log.tag_config("warn", foreground=YELLOW)
        self.log.tag_config("err",  foreground=RED)
        self.log.tag_config("info", foreground=BLUE)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _log(self, msg, color=None):
        tag = color or "info"
        try:
            self.log.config(state=tk.NORMAL)
            self.log.insert(tk.END, msg + "\n", tag)
            self.log.see(tk.END)
            self.log.config(state=tk.DISABLED)
        except Exception:
            pass

    def _set_status(self, msg, color=GREEN):
        try:
            self.status.config(text=msg, fg=color)
        except Exception:
            pass

    def _sp_stage(self, stage):
        if self._sp:
            try: self._sp.set_stage(stage)
            except Exception: pass

    def _sp_status(self, msg, color=None):
        if self._sp:
            try: self._sp.set_status(msg)
            except Exception: pass

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("All supported",
                        "*.csv *.h5 *.hdf5 *.npz *.npy *.mat *.tdms *.json"),
                       ("CSV","*.csv"),("HDF5","*.h5 *.hdf5"),
                       ("NumPy","*.npz *.npy"),("MATLAB","*.mat"),
                       ("TDMS","*.tdms"),("JSON","*.json")])
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
            self._last_results = results
            self._log(f"✓ {len(results)} results → {out}", "ok")
            for r in results[:5]:
                if "error" not in r:
                    self._log(
                        f"  chi={r['chi_t']:.3f}  F={r['F_avg_theory']:.4f}"
                        f"  rank={r['ptm_rank']}  [{r['phase_class']}]", "ok")
            self._set_status(f"Done — {out}", GREEN)
            self.after(0, lambda: show_results(self, results, self._path))
        except Exception as e:
            self._log(f"✗ {e}", "err")
            self._set_status(str(e), RED)

    # ── TPU deployment ────────────────────────────────────────────────────────
    def _run_tpu(self):
        if not self._rows:
            messagebox.showerror("No data", "Load a data file first.")
            return

        project    = self.v_project.get()
        node_id    = self.v_node.get()
        qr_id      = self.v_qr.get()
        zone       = TRC_ZONE
        gcloud_exe = self._tpu_gcloud or "gcloud"

        self._tpu_node     = node_id
        self._tpu_qr       = qr_id
        self._cancel_tpu   = False
        self._tpu_active   = True
        self._last_results = []

        self._log("── TRC-Compliant TPU Deployment ──", "info")
        self._log(f"Zone: {zone}  Acc: {TRC_ACC}  Project: {project}", "info")

        tmp_data = os.path.join(os.path.dirname(self._path), "_jila_tpu_input.json")
        with open(tmp_data, "w") as f:
            json.dump(self._build_records(), f)

        trc_q         = TRCQueue(os.getpid(), node_id, qr_id)
        qr_created    = False
        error_occurred = False
        results       = []

        def run(cmd, desc, timeout=120):
            self._log(f"$ {' '.join(cmd)}", "warn")
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, shell=(sys.platform == "win32"))
            if r.stdout: self._log(r.stdout.strip(), "ok")
            if r.stderr: self._log(r.stderr.strip(), "warn")
            if r.returncode != 0:
                raise RuntimeError(f"{desc} failed (rc={r.returncode}): {r.stderr.strip()[:200]}")

        def _trc_delete():
            for cmd, desc in [
                ([gcloud_exe, "compute", "tpus", "tpu-vm", "delete", node_id,
                  f"--project={project}", f"--zone={zone}", "--quiet"], "Delete node"),
                ([gcloud_exe, "compute", "tpus", "queued-resources", "delete", qr_id,
                  f"--project={project}", f"--zone={zone}", "--quiet"], "Delete QR"),
            ]:
                try:
                    self._log(f"$ {' '.join(cmd)}", "warn")
                    subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=180, shell=(sys.platform == "win32"))
                except Exception:
                    pass

        try:
            # ── Pre-flight: sweep all zones for orphans ──────────────────────
            import json as _pj
            self._log("Pre-flight: scanning all TRC zones for orphans...", "info")
            for z in TRC_ALL_ZONES:
                try:
                    rq = subprocess.run(
                        [gcloud_exe, "compute", "tpus", "queued-resources", "list",
                         f"--project={project}", f"--zone={z}", "--format=json"],
                        capture_output=True, text=True, timeout=20,
                        shell=(sys.platform == "win32"))
                    for item in _pj.loads(rq.stdout or "[]"):
                        name = item.get("name", "").split("/")[-1]
                        if name:
                            self._log(f"  Deleting orphan QR {name} in {z}...", "warn")
                            subprocess.run(
                                [gcloud_exe, "compute", "tpus", "queued-resources",
                                 "delete", name, f"--project={project}",
                                 f"--zone={z}", "--quiet"],
                                capture_output=True, timeout=120,
                                shell=(sys.platform == "win32"))
                    rv = subprocess.run(
                        [gcloud_exe, "compute", "tpus", "tpu-vm", "list",
                         f"--project={project}", f"--zone={z}", "--format=json"],
                        capture_output=True, text=True, timeout=20,
                        shell=(sys.platform == "win32"))
                    for item in _pj.loads(rv.stdout or "[]"):
                        name = item.get("name", "").split("/")[-1]
                        if name:
                            self._log(f"  Deleting orphan VM {name} in {z}...", "warn")
                            subprocess.run(
                                [gcloud_exe, "compute", "tpus", "tpu-vm",
                                 "delete", name, f"--project={project}",
                                 f"--zone={z}", "--quiet"],
                                capture_output=True, timeout=120,
                                shell=(sys.platform == "win32"))
                except Exception:
                    pass
            self._log("Pre-flight complete.", "ok")

            # ── 1. Create QR ─────────────────────────────────────────────────
            self._set_status("Creating TPU queued resource...", YELLOW)
            self._log("Step 1: QR create", "info")
            run([gcloud_exe, "compute", "tpus", "queued-resources", "create", qr_id,
                 f"--node-id={node_id}", f"--project={project}",
                 f"--zone={zone}", f"--accelerator-type={TRC_ACC}",
                 f"--runtime-version={TRC_RUNTIME}",
                 "--spot", "--quiet"], "QR create")
            qr_created = True

            # ── 2. Poll for ACTIVE — dual-phase timeout ───────────────────────
            # Phase 1 (WAITING_FOR_RESOURCES): 15 min — bail if no capacity
            # Phase 2 (PROVISIONING):          20 min — hardware found, wait for boot
            PROV_MAX_MIN   = 20
            MAX_TOTAL_POLLS = (MAX_WAIT_MIN + PROV_MAX_MIN) * 60 // POLL_SEC
            self._log("Step 2: waiting for QR ACTIVE...", "info")
            last_state   = ""
            prov_attempt = None   # attempt# when PROVISIONING first seen

            for attempt in range(MAX_TOTAL_POLLS):
                try:
                    qr_r = subprocess.run(
                        [gcloud_exe, "compute", "tpus", "queued-resources",
                         "describe", qr_id,
                         f"--project={project}", f"--zone={zone}",
                         "--format=json"],
                        capture_output=True, text=True,
                        shell=(sys.platform == "win32"), timeout=25)
                    if qr_r.returncode != 0:
                        err = (qr_r.stderr or qr_r.stdout or "").strip()
                        if "not_found" in err.lower() or "not found" in err.lower():
                            raise RuntimeError(
                                "QR auto-cancelled by GCP — no v6e-8 spot capacity "
                                "in europe-west4-a.\nWait 5-10 min and retry.")
                        qr_state = f"error (rc={qr_r.returncode})"
                        self._log(f"  describe stderr: {err[:200]}", "warn")
                    else:
                        data = _pj.loads(qr_r.stdout)
                        qr_state = (data.get("state", {}).get("state") or "").strip() or "querying"
                except subprocess.TimeoutExpired:
                    qr_state = "querying (api timeout)"
                except RuntimeError:
                    raise
                except Exception as pe:
                    qr_state = f"querying (parse: {type(pe).__name__})"

                # Detect transition into PROVISIONING
                if qr_state == "PROVISIONING" and prov_attempt is None:
                    prov_attempt = attempt
                    self._log(
                        f"  Hardware found! Node is booting "
                        f"(up to {PROV_MAX_MIN} min). Hang tight!", "ok")

                # Phase-aware elapsed / progress
                if prov_attempt is not None:
                    phase_sec    = (attempt - prov_attempt) * POLL_SEC
                    elapsed_str  = f"{phase_sec//60}m {phase_sec%60:02d}s provisioning"
                    prov_polls   = PROV_MAX_MIN * 60 // POLL_SEC
                    pct          = min(99, int((attempt - prov_attempt) / prov_polls * 100))
                    window_label = f"{PROV_MAX_MIN}min boot window"
                else:
                    phase_sec    = attempt * POLL_SEC
                    elapsed_str  = f"{phase_sec//60}m {phase_sec%60:02d}s"
                    wait_polls   = MAX_WAIT_MIN * 60 // POLL_SEC
                    pct          = min(99, int(attempt / max(wait_polls, 1) * 100))
                    window_label = f"{MAX_WAIT_MIN}min capacity window"

                if qr_state != last_state:
                    self._log(f"QR state: {qr_state}  ({elapsed_str} elapsed)", "info")
                    last_state = qr_state
                else:
                    self._log(f"  . still {qr_state}  ({elapsed_str})", "info")

                self._set_status(
                    f"Waiting for TPU node...  {qr_state}  ({elapsed_str})  "
                    f"[{pct}% of {window_label}]", YELLOW)

                if qr_state == "ACTIVE":
                    self._log("QR ACTIVE - node is ready.", "ok")
                    break
                if qr_state in ("FAILED", "SUSPENDED", "DELETING"):
                    raise RuntimeError(
                        f"QR entered terminal state: {qr_state}. "
                        "Spot preempted - retry later.")

                # Phase-specific timeout enforcement
                if prov_attempt is None and phase_sec >= MAX_WAIT_MIN * 60:
                    raise RuntimeError(
                        f"No spot capacity after {MAX_WAIT_MIN} min - "
                        "europe-west4-a is congested. Wait and retry.")
                if prov_attempt is not None and phase_sec >= PROV_MAX_MIN * 60:
                    raise RuntimeError(
                        f"Stuck in PROVISIONING for {PROV_MAX_MIN} min - "
                        "spot node may have been preempted during boot.")

                _t.sleep(POLL_SEC)
            else:
                raise RuntimeError(
                    f"Timeout after {MAX_WAIT_MIN + PROV_MAX_MIN} min total.")

            # ── 3. Wait for SSH ──────────────────────────────────────────────
            self._log("Step 3: waiting for SSH daemon...", "info")
            self._set_status("Node ACTIVE - waiting for SSH...", YELLOW)
            SSH_MAX   = 12
            ssh_ready = False
            for probe in range(SSH_MAX):
                if probe > 0 and probe % 3 == 0:
                    try:
                        qc = subprocess.run(
                            [gcloud_exe, "compute", "tpus", "queued-resources",
                             "describe", qr_id,
                             f"--project={project}", f"--zone={zone}", "--format=json"],
                            capture_output=True, text=True,
                            shell=(sys.platform == "win32"), timeout=30)
                        d2 = _pj.loads(qc.stdout)
                        ls = (d2.get("state", {}).get("state") or "").strip()
                        if ls in ("SUSPENDED", "PREEMPTED", "FAILED", "DELETING"):
                            raise RuntimeError(
                                f"Spot node preempted (QR={ls}) after ACTIVE. "
                                "Normal for spot - retry.")
                    except (RuntimeError, subprocess.TimeoutExpired):
                        raise
                    except Exception:
                        pass
                self._set_status(
                    f"Waiting for SSH daemon... (probe {probe+1}/{SSH_MAX})", YELLOW)
                try:
                    pr = subprocess.run(
                        [gcloud_exe, "compute", "tpus", "tpu-vm", "ssh", node_id,
                         f"--project={project}", f"--zone={zone}",
                         "--command=echo ssh_ok"],
                        capture_output=True, text=True,
                        shell=(sys.platform == "win32"), timeout=90)
                    if pr.returncode == 0 and "ssh_ok" in pr.stdout:
                        self._log("SSH ready", "ok")
                        ssh_ready = True
                        break
                except subprocess.TimeoutExpired:
                    self._log(f"  SSH probe {probe+1} timed out (90s)", "warn")
                except Exception as e:
                    self._log(f"  SSH probe {probe+1}: {e}", "warn")
                _t.sleep(20)
            if not ssh_ready:
                raise RuntimeError("SSH daemon did not start after 4 min")

            # ── 4. Upload ────────────────────────────────────────────────────
            self._set_status("Uploading files...", YELLOW)
            self._log("Step 4: uploading data + pipeline", "info")
            for src in [tmp_data, "jila_pipeline.py"]:
                run([gcloud_exe, "compute", "tpus", "tpu-vm", "scp",
                     src, f"{node_id}:{os.path.basename(src)}",
                     f"--project={project}", f"--zone={zone}"], "SCP upload")

            # ── 5. Run pipeline ──────────────────────────────────────────────
            self._set_status("Running pipeline on TPU...", YELLOW)
            self._log("Step 5: running pipeline", "info")
            cmd_str = (
                "pip install numpy scipy -q && "
                f"python3 jila_pipeline.py "
                f"--input {os.path.basename(tmp_data)} --out ptm_results.json"
            )
            run([gcloud_exe, "compute", "tpus", "tpu-vm", "ssh", node_id,
                 f"--project={project}", f"--zone={zone}",
                 f"--command={cmd_str}"], "Pipeline run", timeout=600)

            # ── 6. Download ──────────────────────────────────────────────────
            self._set_status("Downloading results...", YELLOW)
            self._log("Step 6: downloading results", "info")
            out = os.path.join(os.path.dirname(self._path), "ptm_results.json")
            run([gcloud_exe, "compute", "tpus", "tpu-vm", "scp",
                 f"{node_id}:ptm_results.json", out,
                 f"--project={project}", f"--zone={zone}"], "SCP results")

            with open(out) as f:
                results = json.load(f)
            self._last_results = results
            self._log(f"{len(results)} results -> {out}", "ok")
            for r in results[:5]:
                if "error" not in r:
                    self._log(
                        f"  chi={r['chi_t']:.3f}  F={r['F_avg_theory']:.4f}"
                        f"  rank={r['ptm_rank']}  [{r['phase_class']}]", "ok")
            self._set_status("Done - click Delete to complete TRC cleanup", GREEN)
            self._log("CLEANUP REQUIRED - click Delete TPU Resources!", "warn")
            if results:
                self.after(0, lambda: show_results(self, results, self._path))

        except Exception as e:
            error_occurred = True
            self._log(f"Error: {e}", "err")
            self._set_status(str(e)[:120], RED)

        finally:
            if qr_created and error_occurred:
                self._log("Error - auto TRC cleanup running...", "warn")
                _trc_delete()
                self._set_status("Run failed - TPU auto-cleaned.", RED)
            self._tpu_active = False
            try:
                trc_q.release()
            except Exception:
                pass

    # ── cleanup ───────────────────────────────────────────────────────────────
    def _cleanup_start(self):
        """Main thread: show dialog, then hand off to background thread."""
        project = self.v_project.get()
        node_id = self.v_node.get()
        qr_id   = self.v_qr.get()
        zone    = TRC_ZONE
        if not messagebox.askyesno("Confirm Cleanup",
            f"Delete:\n  Node: {node_id}\n  QR: {qr_id}\n  Zone: {zone}\n\n"
            "This is required by TRC rules. Proceed?"):
            return
        self._cancel_tpu = True
        if self._tpu_proc is not None:
            try:
                self._tpu_proc.kill()
                self._log("Killed hanging subprocess.", "warn")
            except Exception:
                pass
            self._tpu_proc = None
        threading.Thread(
            target=self._cleanup_work,
            args=(project, node_id, qr_id, zone),
            daemon=True
        ).start()

    def _cleanup_work(self, project, node_id, qr_id, zone):
        """Background: TRC Bible delete ritual. NOT_FOUND = already gone = OK."""
        gcloud_exe = self._tpu_gcloud or "gcloud"
        self._log("-- TRC Cleanup --", "warn")
        # TRC Bible: Step 1 delete node, Step 2 delete QR
        for cmd, desc in [
            ([gcloud_exe, "compute", "tpus", "tpu-vm", "delete", node_id,
              f"--project={project}", f"--zone={zone}", "--quiet"], "Delete node"),
            ([gcloud_exe, "compute", "tpus", "queued-resources", "delete", qr_id,
              f"--project={project}", f"--zone={zone}", "--quiet"], "Delete QR"),
        ]:
            self._log(f"$ {' '.join(cmd)}", "warn")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=180, shell=(sys.platform == "win32"))
                if r.returncode == 0:
                    self._log(f"OK {desc}", "ok")
                else:
                    err = (r.stderr or r.stdout or "").strip()
                    if "not_found" in err.lower() or "not found" in err.lower():
                        self._log(f"OK {desc}: already gone", "ok")
                    else:
                        self._log(f"FAIL {desc}: {err}", "err")
            except subprocess.TimeoutExpired:
                self._log(f"FAIL {desc}: timed out (180s)", "err")
        # TRC Bible Step 3: verify zero resources
        try:
            import json as _cj
            r = subprocess.run(
                [gcloud_exe, "compute", "tpus", "queued-resources", "list",
                 f"--project={project}", f"--zone={zone}", "--format=json"],
                capture_output=True, text=True, timeout=30,
                shell=(sys.platform == "win32"))
            remaining = _cj.loads(r.stdout or "[]")
            if remaining:
                self._log(
                    f"WARNING - {len(remaining)} resource(s) still exist! Check billing.",
                    "err")
            else:
                self._log("TRC verified: zero resources remain in zone.", "ok")
        except Exception as e:
            self._log(f"Verify check failed: {e}", "err")
        self._cancel_tpu  = False
        self._tpu_active  = False
        self._set_status("Cleanup complete - verify billing page", GREEN)
        try:
            TRCQueue(os.getpid(), self._tpu_node, self._tpu_qr).release()
        except Exception:
            pass

    def _cleanup(self):
        """Legacy entry point - calls _cleanup_start."""
        self._cleanup_start()


if __name__ == "__main__":
    app = JILAGui()
    app.mainloop()
