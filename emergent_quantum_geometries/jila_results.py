"""
jila_results.py — Results viewer and export panel for the JILA Pipeline GUI.
Called automatically after a successful local or TPU run.

Deps: matplotlib (pip install matplotlib)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, os, csv
import numpy as np

# ── colour palette (matches jila_gui.py) ─────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
BLUE   = "#4cc9f0"
GREEN  = "#4ade80"
YELLOW = "#fbbf24"
RED    = "#f87171"
WHITE  = "#f0f0f0"
GRAY   = "#6b7280"

CLASSICAL_LIMIT = 2 / 3


def show_results(parent, results: list, source_path: str = ""):
    """
    Open the results viewer window.

    Parameters
    ----------
    parent      : tk root window
    results     : list of dicts from run_batch / run_pipeline
    source_path : original input file path (used to suggest export directory)
    """
    win = ResultsWindow(parent, results, source_path)
    win.grab_set()


class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, results, source_path):
        super().__init__(parent)
        self.title("Results — OAT PTM Geometry")
        self.configure(bg=BG)
        self.geometry("1100x720")
        self.resizable(True, True)
        self._results = [r for r in results if "error" not in r]
        self._source_path = source_path
        self._fig = None
        self._build()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Header
        hdr = tk.Frame(self, bg=ACCENT, pady=8)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="PTM Geometry Results",
                 font=("Helvetica", 14, "bold"), bg=ACCENT, fg=WHITE).pack(side=tk.LEFT, padx=12)
        n_q = sum(1 for r in self._results if r.get("phase_class") == "QUANTUM")
        n_s = sum(1 for r in self._results if r.get("phase_class") == "SINGULAR")
        tk.Label(hdr,
                 text=f"{len(self._results)} points  |  {n_q} QUANTUM  |  {n_s} SINGULAR",
                 font=("Helvetica", 9), bg=ACCENT, fg=BLUE).pack(side=tk.LEFT, padx=8)

        # Tab pane
        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        self._tab_plots  = tk.Frame(nb, bg=BG)
        self._tab_table  = tk.Frame(nb, bg=BG)
        self._tab_export = tk.Frame(nb, bg=BG)
        nb.add(self._tab_plots,  text="  Figures  ")
        nb.add(self._tab_table,  text="  Data Table  ")
        nb.add(self._tab_export, text="  Export  ")

        self._build_plots_tab()
        self._build_table_tab()
        self._build_export_tab()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — Figures
    # ─────────────────────────────────────────────────────────────────────────
    def _build_plots_tab(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except ImportError:
            tk.Label(self._tab_plots,
                     text="matplotlib not installed.\npip install matplotlib",
                     bg=BG, fg=RED, font=("Helvetica", 12)).pack(expand=True)
            return

        self._tab_plots.columnconfigure(0, weight=1)
        self._tab_plots.rowconfigure(0, weight=1)

        rs = self._results
        if not rs:
            tk.Label(self._tab_plots, text="No results to display.",
                     bg=BG, fg=GRAY, font=("Helvetica", 11)).pack(expand=True)
            return

        chi  = np.array([r["chi_t"]        for r in rs])
        F    = np.array([r["F_avg_theory"]  for r in rs])
        Txx  = np.array([r["T_xx"]          for r in rs])
        Tyz  = np.array([r["T_yz"]          for r in rs])
        nuc  = np.array([r["nuclear_norm"]  for r in rs])
        conc = np.array([r["concurrence"]   for r in rs])
        rank = np.array([r["ptm_rank"]      for r in rs])
        pc   = [r["phase_class"] for r in rs]

        # Colour per phase class
        col_map = {"QUANTUM": "#4cc9f0", "CLASSICAL": GRAY, "SINGULAR": RED}
        colors = [col_map.get(p, GRAY) for p in pc]

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes.flat:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors=WHITE, labelsize=7)
            ax.xaxis.label.set_color(WHITE)
            ax.yaxis.label.set_color(WHITE)
            ax.title.set_color(WHITE)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRAY)

        idx = np.argsort(chi)
        chi_s, F_s, Txx_s, Tyz_s, nuc_s, conc_s = (
            chi[idx], F[idx], Txx[idx], Tyz[idx], nuc[idx], conc[idx])

        # ── Panel 1: F_avg vs chi_t ──────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(chi_s, F_s, color=BLUE, lw=2, marker="o", ms=4, label="F_avg (theory)")
        ax.axhline(CLASSICAL_LIMIT, color=RED, ls="--", lw=1.2, label="Classical limit (2/3)")
        ax.axhline(0.5, color=GRAY, ls=":", lw=0.8, label="Depolarising (1/2)")
        # shade quantum region
        ax.fill_between(chi_s, CLASSICAL_LIMIT, F_s,
                        where=(F_s > CLASSICAL_LIMIT), alpha=0.15, color=BLUE)
        ax.set_xlabel("χt (interaction parameter)")
        ax.set_ylabel("F_avg")
        ax.set_title("Teleportation Fidelity vs χt")
        ax.legend(fontsize=6, labelcolor=WHITE, facecolor="#161b22", edgecolor=GRAY)

        # ── Panel 2: T_xx and T_yz vs chi_t ─────────────────────────────────
        ax = axes[0, 1]
        ax.plot(chi_s, Txx_s, color=GREEN,  lw=2, marker="o", ms=3, label="T_xx")
        ax.plot(chi_s, Tyz_s, color=YELLOW, lw=2, marker="s", ms=3, label="T_yz")
        ax.axhline(0, color=GRAY, ls="--", lw=0.7)
        ax.set_xlabel("χt")
        ax.set_ylabel("Correlator value")
        ax.set_title("T-matrix Correlators vs χt")
        ax.legend(fontsize=6, labelcolor=WHITE, facecolor="#161b22", edgecolor=GRAY)

        # ── Panel 3: Nuclear norm + phase boundary ───────────────────────────
        ax = axes[0, 2]
        ax.plot(chi_s, nuc_s, color="#c084fc", lw=2, marker="D", ms=3,
                label="‖T‖₍nuclear₎")
        # Phase boundary as vertical line at F=2/3 crossing
        crossings = [chi_s[i] for i in range(len(F_s)-1)
                     if (F_s[i]-CLASSICAL_LIMIT)*(F_s[i+1]-CLASSICAL_LIMIT) < 0]
        for cx in crossings:
            ax.axvline(cx, color=RED, ls="--", lw=1.2, alpha=0.7, label=f"Phase boundary χt≈{cx:.2f}")
        ax.set_xlabel("χt")
        ax.set_ylabel("Nuclear norm ‖T‖_*")
        ax.set_title("PTM Nuclear Norm vs χt")
        ax.legend(fontsize=6, labelcolor=WHITE, facecolor="#161b22", edgecolor=GRAY)

        # ── Panel 4: Concurrence vs chi_t ────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(chi_s, conc_s, color="#fb923c", lw=2, marker="^", ms=3, label="Concurrence")
        ax.set_xlabel("χt")
        ax.set_ylabel("Concurrence C")
        ax.set_title("Entanglement (Concurrence) vs χt")
        ax.legend(fontsize=6, labelcolor=WHITE, facecolor="#161b22", edgecolor=GRAY)

        # ── Panel 5: PTM rank vs chi_t ───────────────────────────────────────
        ax = axes[1, 1]
        rank_s = rank[idx]
        ax.step(chi_s, rank_s, color=BLUE, lw=2, where="mid", label="PTM rank")
        ax.set_xlabel("χt")
        ax.set_ylabel("Effective rank")
        ax.set_ylim(-0.2, max(rank_s)+0.5 if len(rank_s) else 3)
        ax.set_title("PTM Rank vs χt")
        ax.legend(fontsize=6, labelcolor=WHITE, facecolor="#161b22", edgecolor=GRAY)

        # ── Panel 6: T-matrix heatmap at chi_t* ─────────────────────────────
        ax = axes[1, 2]
        # Find chi_t closest to peak fidelity
        peak_idx = np.argmax(F_s)
        T_at_peak = np.array(rs[idx[peak_idx]]["T_matrix"])
        im = ax.imshow(T_at_peak, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["X","Y","Z"], color=WHITE, fontsize=8)
        ax.set_yticks([0,1,2]); ax.set_yticklabels(["X","Y","Z"], color=WHITE, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{T_at_peak[i,j]:.3f}", ha="center", va="center",
                        color=WHITE if abs(T_at_peak[i,j]) < 0.5 else BG, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE)
        ax.set_title(f"T-matrix at χt*={chi_s[peak_idx]:.3f}")

        fig.suptitle("OAT Channel PTM Geometry — JILA Pipeline Results",
                     color=WHITE, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        self._fig = fig

        # Embed in tab
        canvas = FigureCanvasTkAgg(fig, master=self._tab_plots)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = tk.Frame(self._tab_plots, bg=BG)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas, toolbar_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 — Data table
    # ─────────────────────────────────────────────────────────────────────────
    def _build_table_tab(self):
        self._tab_table.columnconfigure(0, weight=1)
        self._tab_table.rowconfigure(0, weight=1)

        cols = ("chi_t","F_avg","T_xx","T_yz","nuc_norm","rank","concurrence","phase")
        hdrs = ("χt","F_avg","T_xx","T_yz","‖T‖_*","rank","concurrence","phase")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background="#161b22", foreground=WHITE,
                        fieldbackground="#161b22", rowheight=20,
                        font=("Courier", 8))
        style.configure("Dark.Treeview.Heading",
                        background=ACCENT, foreground=BLUE,
                        font=("Helvetica", 8, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT)])

        tv = ttk.Treeview(self._tab_table, columns=cols, show="headings",
                          style="Dark.Treeview")
        for c, h in zip(cols, hdrs):
            tv.heading(c, text=h)
            tv.column(c, width=95, anchor="center")

        phase_tags = {"QUANTUM": GREEN, "CLASSICAL": WHITE, "SINGULAR": RED}
        for r in self._results:
            pc = r.get("phase_class", "—")
            tv.insert("", tk.END, values=(
                f"{r['chi_t']:.4f}",
                f"{r['F_avg_theory']:.5f}",
                f"{r['T_xx']:.4f}",
                f"{r['T_yz']:.4f}",
                f"{r['nuclear_norm']:.4f}",
                r['ptm_rank'],
                f"{r['concurrence']:.4f}",
                pc,
            ), tags=(pc,))
            tv.tag_configure(pc, foreground=phase_tags.get(pc, WHITE))

        sb = ttk.Scrollbar(self._tab_table, orient=tk.VERTICAL, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — Export
    # ─────────────────────────────────────────────────────────────────────────
    def _build_export_tab(self):
        p = self._tab_export
        p.columnconfigure(0, weight=1)

        tk.Label(p, text="Export Results", font=("Helvetica", 13, "bold"),
                 bg=BG, fg=WHITE).grid(row=0, column=0, pady=12)
        tk.Label(p, text="Choose format and destination:",
                 bg=BG, fg=GRAY, font=("Helvetica", 9)).grid(row=1, column=0)

        btns = [
            ("📄  CSV  (Excel / Origin compatible)",    "#22c55e", self._export_csv),
            ("📋  JSON  (machine-readable)",             "#3b82f6", self._export_json),
            ("🔢  NumPy NPZ  (array format)",           "#a855f7", self._export_npz),
            ("📦  HDF5  (h5py / QuTiP compatible)",     "#f97316", self._export_h5),
            ("🖼  PNG  (figure image, 300 dpi)",        "#ec4899", self._export_png),
            ("📊  PDF  (publication-quality figure)",   "#eab308", self._export_pdf),
        ]
        for i, (label, color, cmd) in enumerate(btns):
            tk.Button(p, text=label, command=cmd,
                      bg=color, fg=BG, font=("Helvetica", 10, "bold"),
                      relief=tk.FLAT, padx=12, pady=8)\
              .grid(row=2+i, column=0, sticky="ew", padx=60, pady=4)

        self._export_status = tk.Label(p, text="", bg=BG, fg=GREEN,
                                       font=("Helvetica", 9))
        self._export_status.grid(row=2+len(btns)+1, column=0, pady=8)

    def _suggest_dir(self):
        return os.path.dirname(self._source_path) if self._source_path else os.getcwd()

    def _ok(self, msg):
        self._export_status.config(text=f"✓  {msg}", fg=GREEN)

    def _err(self, e):
        self._export_status.config(text=f"✗  {e}", fg=RED)

    def _flat(self):
        """Flatten results to list of flat dicts for tabular export."""
        out = []
        for r in self._results:
            T = r.get("T_matrix", [[0]*3]*3)
            flat = {
                "N": r.get("N"), "chi_t": r.get("chi_t"),
                "gamma_t": r.get("gamma_t"),
                "F_avg_theory": r.get("F_avg_theory"),
                "T_xx": r.get("T_xx"), "T_yz": r.get("T_yz"),
                "T_yy": T[1][1], "T_zz": T[2][2],
                "nuclear_norm": r.get("nuclear_norm"),
                "ptm_rank": r.get("ptm_rank"),
                "concurrence": r.get("concurrence"),
                "phase_class": r.get("phase_class"),
                "above_classical": r.get("above_classical"),
            }
            out.append(flat)
        return out

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")],
            initialdir=self._suggest_dir(), initialfile="ptm_results.csv")
        if not path: return
        try:
            self._export_csv_to(path)
            self._ok(f"Saved {os.path.basename(path)}")
        except Exception as e: self._err(e)

    def _export_csv_to(self, path):
        rows = self._flat()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    def _export_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON","*.json")],
            initialdir=self._suggest_dir(), initialfile="ptm_results.json")
        if not path: return
        try:
            self._export_json_to(path)
            self._ok(f"Saved {os.path.basename(path)}")
        except Exception as e: self._err(e)

    def _export_json_to(self, path):
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2)

    def _export_npz(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("NumPy NPZ","*.npz")],
            initialdir=self._suggest_dir(), initialfile="ptm_results.npz")
        if not path: return
        try:
            self._export_npz_to(path)
            self._ok(f"Saved {os.path.basename(path)}")
        except Exception as e: self._err(e)

    def _export_npz_to(self, path):
        rows = self._flat()
        arrays = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        np.savez(path, **arrays)

    def _export_h5(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".h5", filetypes=[("HDF5","*.h5 *.hdf5")],
            initialdir=self._suggest_dir(), initialfile="ptm_results.h5")
        if not path: return
        try:
            self._export_h5_to(path)
            self._ok(f"Saved {os.path.basename(path)}")
        except ImportError:
            self._err("h5py not installed. pip install h5py")
        except Exception as e: self._err(e)

    def _export_h5_to(self, path):
        import h5py
        rows = self._flat()
        with h5py.File(path, "w") as f:
            for k in rows[0]:
                vals = [r[k] for r in rows]
                try:
                    f.create_dataset(k, data=np.array(vals, dtype=float))
                except Exception:
                    f.create_dataset(k, data=np.array([str(v) for v in vals]))

    def _export_png(self):
        if self._fig is None:
            self._err("No figure to export (matplotlib not available)"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG image","*.png")],
            initialdir=self._suggest_dir(), initialfile="ptm_geometry.png")
        if not path: return
        try:
            self._export_png_to(path)
            self._ok(f"Saved {os.path.basename(path)} (300 dpi)")
        except Exception as e: self._err(e)

    def _export_png_to(self, path):
        self._fig.savefig(path, dpi=300, bbox_inches="tight",
                          facecolor=self._fig.get_facecolor())

    def _export_pdf(self):
        if self._fig is None:
            self._err("No figure to export (matplotlib not available)"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF","*.pdf")],
            initialdir=self._suggest_dir(), initialfile="ptm_geometry.pdf")
        if not path: return
        try:
            self._export_pdf_to(path)
            self._ok(f"Saved {os.path.basename(path)}")
        except Exception as e: self._err(e)

    def _export_pdf_to(self, path):
        self._fig.savefig(path, bbox_inches="tight",
                          facecolor=self._fig.get_facecolor())
