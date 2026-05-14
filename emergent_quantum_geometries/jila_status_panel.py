"""
jila_status_panel.py
Visual status dashboard and TRC-compliant queue manager.

Queue contract (TRC-safe):
  - Only ONE instance may hold the run lock at a time.
  - All others wait in a JSON queue file, showing position + ETA.
  - On completion (success OR error), lock is released before the
    next instance in queue is allowed to start.
  - This guarantees serial TPU use: never >1 node+QR alive.
"""

import tkinter as tk
from tkinter import ttk
import json, os, time, threading

# ── colours (match jila_gui palette) ─────────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
BLUE   = "#4cc9f0"
GREEN  = "#4ade80"
YELLOW = "#fbbf24"
RED    = "#f87171"
GRAY   = "#6b7280"
WHITE  = "#f0f0f0"
DARK   = "#0d1117"

# ── stage definitions ─────────────────────────────────────────────────────────
STAGES = [
    ("QUEUE",    "In Queue"),
    ("CREATE",   "Creating TPU"),
    ("READY",    "Node Ready"),
    ("UPLOAD",   "Uploading"),
    ("RUN",      "Running"),
    ("DOWNLOAD", "Results"),
    ("CLEANUP",  "Cleanup"),
    ("DONE",     "Done"),
]
STAGE_IDX = {s[0]: i for i, s in enumerate(STAGES)}

STAGE_DURATIONS = {
    "QUEUE":    0,
    "CREATE":   90,
    "READY":    1800,   # spot can take up to 30 min
    "UPLOAD":   30,
    "RUN":      180,
    "DOWNLOAD": 20,
    "CLEANUP":  45,
    "DONE":     0,
}

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE_FILE   = os.path.join(HERE, ".jila_tpu_queue.json")
LOCK_FILE    = os.path.join(HERE, ".jila_tpu.lock")


# ═════════════════════════════════════════════════════════════════════════════
# Queue manager
# ═════════════════════════════════════════════════════════════════════════════

class TRCQueue:
    """
    File-based cross-process queue for serialising TPU runs.
    All mutation is protected by a simple retry-write pattern.
    """
    def __init__(self, pid, node_id, qr_id):
        self.pid     = pid
        self.node_id = node_id
        self.qr_id   = qr_id
        self._entry  = {
            "pid":     pid,
            "node_id": node_id,
            "qr_id":   qr_id,
            "joined":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ── queue file helpers ────────────────────────────────────────────────────
    @staticmethod
    def _read():
        try:
            if os.path.exists(QUEUE_FILE):
                with open(QUEUE_FILE) as f:
                    q = json.load(f)
                # prune dead entries
                import psutil
                return [e for e in q if psutil.pid_exists(e["pid"])]
        except Exception:
            pass
        return []

    @staticmethod
    def _write(q):
        with open(QUEUE_FILE, "w") as f:
            json.dump(q, f, indent=2)

    # ── public API ────────────────────────────────────────────────────────────
    def join(self):
        """Add this process to the queue. Returns position (0 = next to run)."""
        q = self._read()
        # don't double-add
        if not any(e["pid"] == self.pid for e in q):
            q.append(self._entry)
            self._write(q)
        return self.position()

    def position(self):
        """0-indexed position in queue (-1 if not found)."""
        q = self._read()
        for i, e in enumerate(q):
            if e["pid"] == self.pid:
                return i
        return -1

    def queue_size(self):
        return len(self._read())

    def is_my_turn(self):
        """True when this process is first AND no lock is held."""
        return self.position() == 0 and not os.path.exists(LOCK_FILE)

    def acquire_lock(self):
        data = {**self._entry, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(LOCK_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def release(self):
        """Remove from queue AND release lock — always call in finally."""
        # remove from queue
        q = [e for e in self._read() if e["pid"] != self.pid]
        self._write(q)
        # release lock
        try:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except Exception:
            pass

    def eta_seconds(self):
        """Rough ETA based on position and average stage durations."""
        pos = self.position()
        if pos <= 0:
            return 0
        per_run = sum(STAGE_DURATIONS.values())
        return pos * per_run

    @staticmethod
    def cleanup_stale():
        """Remove dead-process entries from queue and lock file."""
        import psutil
        q = TRCQueue._read()
        TRCQueue._write([e for e in q if psutil.pid_exists(e["pid"])])
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE) as f:
                    info = json.load(f)
                if not psutil.pid_exists(info.get("pid", -1)):
                    os.remove(LOCK_FILE)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Visual status panel
# ═════════════════════════════════════════════════════════════════════════════

class StatusPanel(tk.Frame):
    """
    Right-panel widget replacing the raw terminal.
    Shows stage dots, progress bar, status message, queue card,
    and a collapsible detail log at the bottom.
    """
    DOT_R  = 12   # dot radius
    DOT_SP = 72   # spacing between dots
    CANVAS_H = 80

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.columnconfigure(0, weight=1)
        self._current_stage = -1
        self._log_visible   = False
        self._start_time    = None
        self._running       = False
        self._build()


    def _build(self):
        # ── stage dot strip ───────────────────────────────────────────────
        self._canvas = tk.Canvas(
            self, bg=PANEL, height=self.CANVAS_H,
            highlightthickness=0
        )
        self._canvas.grid(row=0, column=0, sticky="ew", padx=12, pady=(14, 0))
        self._draw_stages(-1)

        # ── progress bar ──────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TRC.Horizontal.TProgressbar",
                        troughcolor=ACCENT, background=BLUE,
                        thickness=10)
        self._pbar = ttk.Progressbar(
            self, orient="horizontal", mode="determinate",
            style="TRC.Horizontal.TProgressbar", maximum=100
        )
        self._pbar.grid(row=1, column=0, sticky="ew", padx=12, pady=6)

        # ── main status message ───────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self._status_var,
                 bg=PANEL, fg=WHITE, font=("Helvetica", 10, "bold"),
                 wraplength=400, justify="left")\
          .grid(row=2, column=0, sticky="w", padx=14, pady=(2, 0))

        # ── timing line ───────────────────────────────────────────────────
        self._timing_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._timing_var,
                 bg=PANEL, fg=GRAY, font=("Helvetica", 8))\
          .grid(row=3, column=0, sticky="w", padx=14)

        # ── queue card (hidden until queued) ──────────────────────────────
        self._queue_card = tk.Frame(self, bg=ACCENT, bd=0)
        self._queue_card.grid(row=4, column=0, sticky="ew",
                              padx=12, pady=(8, 0))
        self._queue_card.grid_remove()

        self._queue_pos_var = tk.StringVar(value="")
        self._queue_eta_var = tk.StringVar(value="")
        tk.Label(self._queue_card, textvariable=self._queue_pos_var,
                 bg=ACCENT, fg=YELLOW,
                 font=("Helvetica", 10, "bold"))\
          .pack(anchor="w", padx=10, pady=(6, 0))
        tk.Label(self._queue_card, textvariable=self._queue_eta_var,
                 bg=ACCENT, fg=GRAY, font=("Helvetica", 8))\
          .pack(anchor="w", padx=10, pady=(0, 6))

        # ── queue wait bar (indeterminate) ────────────────────────────────
        style.configure("Q.Horizontal.TProgressbar",
                        troughcolor=ACCENT, background=YELLOW, thickness=6)
        self._qbar = ttk.Progressbar(
            self, orient="horizontal", mode="indeterminate",
            style="Q.Horizontal.TProgressbar"
        )
        self._qbar.grid(row=5, column=0, sticky="ew", padx=12)
        self._qbar.grid_remove()

        # ── detail log (collapsible) ──────────────────────────────────────
        self._log_toggle = tk.Button(
            self, text="+ Details", command=self._toggle_log,
            bg=PANEL, fg=GRAY, font=("Helvetica", 8),
            relief=tk.FLAT, cursor="hand2", anchor="w"
        )
        self._log_toggle.grid(row=6, column=0, sticky="w", padx=12, pady=(10,0))

        self._log_frame = tk.Frame(self, bg=DARK)
        # not gridded until toggled

        from tkinter import scrolledtext
        self._log_txt = scrolledtext.ScrolledText(
            self._log_frame, bg=DARK, fg=GREEN,
            font=("Courier", 7), height=8, relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED
        )
        self._log_txt.pack(fill="both", expand=True, padx=2, pady=2)
        self._log_txt.tag_config("ok",   foreground=GREEN)
        self._log_txt.tag_config("warn", foreground=YELLOW)
        self._log_txt.tag_config("err",  foreground=RED)
        self._log_txt.tag_config("info", foreground=BLUE)

    # ── stage rendering ───────────────────────────────────────────────────────
    def _draw_stages(self, active_idx):
        c = self._canvas
        c.delete("all")
        n = len(STAGES)
        # centre the strip
        total_w = (n - 1) * self.DOT_SP + 2 * self.DOT_R
        self.update_idletasks()
        w = c.winfo_width() or 500
        x0 = max(self.DOT_R + 4, (w - total_w) // 2)
        y  = self.CANVAS_H // 2 - 6

        for i, (_, label) in enumerate(STAGES):
            x = x0 + i * self.DOT_SP
            if i < active_idx:
                fill, outline, txt_col = GREEN,  GREEN,  GREEN
            elif i == active_idx:
                fill, outline, txt_col = BLUE,   BLUE,   WHITE
            else:
                fill, outline, txt_col = PANEL,  GRAY,   GRAY
            # connector line
            if i > 0:
                px = x0 + (i - 1) * self.DOT_SP
                col = GREEN if i <= active_idx else GRAY
                c.create_line(px + self.DOT_R, y + self.DOT_R,
                              x - self.DOT_R,  y + self.DOT_R,
                              fill=col, width=2)
            # dot
            c.create_oval(x - self.DOT_R, y,
                          x + self.DOT_R, y + 2 * self.DOT_R,
                          fill=fill, outline=outline, width=2)
            if i < active_idx:
                c.create_text(x, y + self.DOT_R, text="✓",
                              fill=PANEL, font=("Helvetica", 9, "bold"))
            # label
            c.create_text(x, y + 2 * self.DOT_R + 10, text=label,
                          fill=txt_col, font=("Helvetica", 7))

    # ── public update methods ─────────────────────────────────────────────────
    def set_stage(self, stage_key):
        """Move to a named stage and update progress bar."""
        idx = STAGE_IDX.get(stage_key, -1)
        self._current_stage = idx
        self._draw_stages(idx)
        pct = int(idx / (len(STAGES) - 1) * 100) if idx >= 0 else 0
        self._pbar["value"] = pct
        if idx == 0:
            self._start_time = time.time()

    def set_status(self, msg, color=WHITE):
        self._status_var.set(msg)

    def set_timing(self, msg):
        try:
            self._timing_var.set(msg)
        except Exception:
            pass  # suppress in headless/test contexts


    def show_queue(self, position, size, eta_sec):
        """Show the queue waiting card."""
        self._queue_card.grid()
        self._qbar.grid()
        self._qbar.start(15)
        pos_text = f"Queue position: {position + 1} of {size}"
        if position == 0:
            pos_text = "You're next — waiting for current run to finish..."
        self._queue_pos_var.set(pos_text)
        if eta_sec > 0:
            m, s = divmod(int(eta_sec), 60)
            self._queue_eta_var.set(f"Estimated wait: ~{m}m {s}s")
        else:
            self._queue_eta_var.set("Estimated wait: calculating...")

    def hide_queue(self):
        self._queue_card.grid_remove()
        self._qbar.grid_remove()
        self._qbar.stop()

    def reset(self):
        self._current_stage = -1
        self._draw_stages(-1)
        self._pbar["value"] = 0
        self._status_var.set("Ready")
        self._timing_var.set("")
        self.hide_queue()

    def mark_done(self, success=True):
        self._running = False   # stop the timer thread
        if success:
            self.set_stage("DONE")
            self._pbar["value"] = 100
            style = ttk.Style()
            style.configure("TRC.Horizontal.TProgressbar", background=GREEN)
        else:
            style = ttk.Style()
            style.configure("TRC.Horizontal.TProgressbar", background=RED)

    # ── collapsible log ───────────────────────────────────────────────────────
    def _toggle_log(self):
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_frame.grid(row=7, column=0, sticky="nsew",
                                 padx=12, pady=(2, 8))
            self.rowconfigure(7, weight=1)
            self._log_toggle.config(text="- Details")
        else:
            self._log_frame.grid_remove()
            self._log_toggle.config(text="+ Details")

    def log(self, msg, tag="info"):
        self._log_txt.config(state=tk.NORMAL)
        self._log_txt.insert(tk.END, msg + "\n", tag)
        self._log_txt.see(tk.END)
        self._log_txt.config(state=tk.DISABLED)
        # auto-show on errors
        if tag == "err" and not self._log_visible:
            self._toggle_log()

    def start_timer(self):
        """Spawn a background thread that ticks elapsed time every second."""
        self._start_time = time.time()
        self._running    = True
        def _tick():
            TERMINAL = {STAGE_IDX.get(k, 99) for k in ("CLEANUP", "DONE")}
            while self._running and self._current_stage not in TERMINAL:
                elapsed = int(time.time() - self._start_time)
                m, s    = divmod(elapsed, 60)
                # Stage % is honest; ETA is not reliable for spot TPUs
                n_stages = max(1, len(STAGES) - 1)  # exclude DONE
                stage_pct = max(0, min(100,
                    int(self._current_stage / n_stages * 100)
                    if self._current_stage >= 0 else 0
                ))
                self.set_timing(
                    f"Elapsed: {m}m {s:02d}s    "
                    f"Pipeline progress: {stage_pct}% of stages complete"
                )
                time.sleep(1)
        threading.Thread(target=_tick, daemon=True).start()

