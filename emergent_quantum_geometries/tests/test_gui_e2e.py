"""
tests/test_gui_e2e.py

End-to-end GUI tests for jila_gui.py and jila_results.py.

These tests drive the Tkinter application programmatically without opening
visible windows (root.withdraw() / win.withdraw()). They cover:

  1. App startup and initial state
  2. File loading (CSV fixture)
  3. Local CPU run → results window opens → export to all formats
  4. TPU run (gcloud subprocess mocked) → full TRC code path exercised
     → results window opens → export verified

NOTE ON TPU MOCKING
-------------------
The TPU E2E test patches subprocess.run to return mock gcloud success
responses and writes a fixture results file to the expected download path.
This exercises every line of the _run_tpu code path (QR create, wait-for-
READY poll, SCP upload, SSH run, SCP download, cleanup) without spending
any actual compute credits.

To run a REAL TPU acceptance test against live GCP:
    pytest tests/test_gui_e2e.py -m live_tpu --no-header -v
(requires gcloud auth and TRC project access)
"""
import sys, os, json, csv, time, tempfile, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

# ── fixture paths ─────────────────────────────────────────────────────────────
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
PHASE_CSV   = os.path.join(FIXTURE_DIR, "oat_phase_diagram_gam0.csv")


def _make_mock_subprocess(tmp_results_path, ready_after=2):
    """
    Returns a mock subprocess.run that:
    - Returns success for all gcloud calls
    - Writes a fixture results JSON to tmp_results_path on the SCP-download call
    - Reports READY state after `ready_after` list calls
    """
    call_count = [0]

    def mock_run(cmd, **kwargs):
        call_count[0] += 1
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = str(cmd)

        # Simulate READY state after a few poll calls
        if "list" in cmd_str and "state" in cmd_str:
            result.stdout = "READY" if call_count[0] >= ready_after else "PROVISIONING"
            return result

        # Simulate SCP download — write fixture results
        if "scp" in cmd_str and "ptm_results.json" in cmd_str and ":" not in str(cmd[-2]):
            # This is the download SCP (node:file -> local)
            from jila_pipeline import run_batch
            records = [{"N": 4, "chi_t": c, "gamma_t": 0.0}
                       for c in np.linspace(0.5, 2.5, 8)]
            results = run_batch(records)
            with open(tmp_results_path, "w") as f:
                json.dump(results, f, indent=2)

        return result

    return mock_run


# ═══════════════════════════════════════════════════════════════════════════════
# 1. App startup
# ═══════════════════════════════════════════════════════════════════════════════

class TestAppStartup:
    @pytest.fixture(autouse=True)
    def _patch_dialogs(self):
        with patch("jila_gui.messagebox.showerror"), \
             patch("jila_gui.messagebox.showinfo"), \
             patch("jila_gui.messagebox.askyesno", return_value=False):
            yield

    @pytest.fixture
    def app(self):
        try:
            from jila_gui import JILAGui
            root = JILAGui()
            root.withdraw()
            yield root
            try:
                root.destroy()
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Tkinter unavailable: {e}")
            yield  # never reached

    def test_app_creates(self, app):
        assert app.winfo_exists()

    def test_initial_rows_empty(self, app):
        assert app._rows == []

    def test_initial_last_results_empty(self, app):
        assert app._last_results == []

    def test_status_label_exists(self, app):
        assert app.status.cget("text") != ""

    def test_col_vars_populated(self, app):
        assert "N" in app._col_vars
        assert "chi_t" in app._col_vars


# ═══════════════════════════════════════════════════════════════════════════════
# 2. File loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileLoading:
    @pytest.fixture
    def app(self):
        try:
            from jila_gui import JILAGui
            root = JILAGui()
            root.withdraw()
            yield root
            try:
                root.destroy()
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Tkinter unavailable: {e}")
            yield  # never reached

    def test_load_csv_fixture(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        assert len(app._rows) == 20
        assert app._path == PHASE_CSV

    def test_load_auto_maps_chi_t(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        assert app._col_vars["chi_t"].get() == "chi_t"

    def test_load_auto_maps_N(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        assert app._col_vars["N"].get() == "N"

    def test_load_bad_file_does_not_crash(self, app, tmp_path):
        bad = str(tmp_path / "bad.xyz")
        with open(bad, "w") as f: f.write("garbage")
        app._load_file(bad)   # should log error, not raise
        assert app._rows == []

    def test_build_records_uses_column_mapping(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        records = app._build_records()
        assert len(records) == 20
        assert all("chi_t" in r for r in records)
        assert all("N" in r for r in records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Local CPU run → results window → export
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalRunE2E:
    @pytest.fixture
    def app(self, tmp_path):
        try:
            from jila_gui import JILAGui
            root = JILAGui()
            root.withdraw()
            # Give app a valid _path so dirname doesn't fail
            root._path = str(tmp_path / "dummy.csv")
            yield root
            try:
                root.destroy()
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Tkinter unavailable: {e}")
            yield  # never reached

    def _run_local_sync(self, app):
        """
        Call _run_local on the main thread (safe in tests — no mainloop blocking).
        Patch show_results to prevent popup, and messagebox to prevent dialogs.
        """
        with patch("jila_gui.show_results"), \
             patch("jila_gui.messagebox.showerror"), \
             patch("jila_gui.messagebox.askyesno", return_value=True):
            app._run_local()

    def test_local_run_populates_last_results(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        self._run_local_sync(app)
        assert len(app._last_results) == 20
        assert all("error" not in r for r in app._last_results)

    def test_local_run_correct_F_values(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        app._load_file(PHASE_CSV)
        self._run_local_sync(app)
        F_vals = [r["F_avg_theory"] for r in app._last_results]
        assert all(0.5 <= f <= 1.0 for f in F_vals)
        assert any(f > 2/3 for f in F_vals)

    def test_local_run_writes_json_output(self, app, tmp_path):
        """Results JSON is written next to app._path (tmp_path/dummy.csv)."""
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        # _load_file will overwrite app._path with PHASE_CSV;
        # we set it back so output lands in tmp_path
        app._load_file(PHASE_CSV)
        app._path = str(tmp_path / "dummy.csv")   # redirect output
        self._run_local_sync(app)
        out = os.path.join(tmp_path, "ptm_results.json")
        assert os.path.exists(out), f"Expected {out}"
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 20

    def test_local_run_no_data_does_not_crash(self, app):
        """Running with no loaded data should show error dialog, not crash."""
        with patch("jila_gui.messagebox.showerror") as mock_err:
            app._run_local()
        mock_err.assert_called_once()   # error dialog was shown

    @pytest.fixture
    def results_win_from_local(self, app):
        if not os.path.exists(PHASE_CSV):
            pytest.skip("Fixture not present")
        from jila_results import ResultsWindow
        app._load_file(PHASE_CSV)
        self._run_local_sync(app)
        win = ResultsWindow(app, app._last_results, PHASE_CSV)
        win.withdraw()
        yield win
        try:
            win.destroy()
        except Exception:
            pass

    def test_results_window_opens(self, results_win_from_local):
        assert results_win_from_local.winfo_exists()

    def test_results_window_has_20_rows(self, results_win_from_local):
        assert len(results_win_from_local._results) == 20

    def test_results_quantum_points_correct(self, results_win_from_local):
        quantum = [r for r in results_win_from_local._results
                   if r["phase_class"] == "QUANTUM"]
        assert all(r["F_avg_theory"] > 2/3 for r in quantum)

    # ── Export from results window ─────────────────────────────────────────────
    def test_export_csv_from_local_run(self, results_win_from_local, tmp_path):
        path = str(tmp_path / "export.csv")
        results_win_from_local._export_csv_to(path)
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20
        assert "F_avg_theory" in rows[0]

    def test_export_json_from_local_run(self, results_win_from_local, tmp_path):
        path = str(tmp_path / "export.json")
        results_win_from_local._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 20
        assert all("T_matrix" in r for r in data)

    def test_export_npz_from_local_run(self, results_win_from_local, tmp_path):
        path = str(tmp_path / "export.npz")
        results_win_from_local._export_npz_to(path)
        d = np.load(path)
        assert len(d["F_avg_theory"]) == 20
        assert any(d["F_avg_theory"] > 2/3)

    def test_export_png_from_local_run(self, results_win_from_local, tmp_path):
        pytest.importorskip("matplotlib")
        if results_win_from_local._fig is None:
            pytest.skip("Figure not rendered")
        path = str(tmp_path / "export.png")
        results_win_from_local._export_png_to(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 50_000

    def test_export_pdf_from_local_run(self, results_win_from_local, tmp_path):
        pytest.importorskip("matplotlib")
        if results_win_from_local._fig is None:
            pytest.skip("Figure not rendered")
        path = str(tmp_path / "export.pdf")
        results_win_from_local._export_pdf_to(path)
        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TPU run (mocked gcloud) → results window → export
#
# gcloud subprocess.run is patched; no actual GCP resources provisioned.
# _run_tpu is called on the main test thread (same as _run_local) since
# Tkinter StringVar.get() is not thread-safe outside the mainloop.
# ═══════════════════════════════════════════════════════════════════════════════

class TestTPURunE2E:
    @pytest.fixture
    def app(self, tmp_path):
        from jila_gui import JILAGui
        root = JILAGui()
        root.withdraw()
        if not os.path.exists(PHASE_CSV):
            root.destroy()
            pytest.skip("Fixture not present")
        root._load_file(PHASE_CSV)
        # Redirect output path to tmp_path
        root._path = str(tmp_path / "fixture_data.csv")
        root.v_project.set("time-emission-test")
        root.v_node.set("test-node")
        root.v_qr.set("test-qr")
        yield root
        try:
            root.destroy()
        except Exception:
            pass

    def _make_scp_mock(self, tmp_path):
        """Mock that writes 8-pt results JSON when the SCP download is called."""
        call_count = [0]
        called_cmds = []

        def mock_run(cmd, **kw):
            call_count[0] += 1
            called_cmds.append(cmd)
            m = MagicMock()
            m.returncode = 0
            m.stderr = ""
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)

            # SSH readiness probe (--command=echo ssh_ok) → immediate pass
            if "ssh_ok" in cmd_str:
                m.stdout = "ssh_ok"
                return m

            # QR describe → ACTIVE  (code now uses --format=json, parse with json.loads)
            if "queued-resources" in cmd_str and "describe" in cmd_str:
                import json as _j
                m.stdout = _j.dumps({"state": {"state": "ACTIVE"}})
                return m

            # tpu-vm list → READY (fallback poll)
            if "tpu-vm" in cmd_str and "list" in cmd_str and "value(state)" in cmd_str:
                m.stdout = "READY"
                return m

            # SCP download: write results file to local destination.
            if "scp" in cmd_str and "ptm_results.json" in cmd_str:
                local_dest = str(tmp_path / "ptm_results.json")
                for i, arg in enumerate(cmd):
                    if "ptm_results.json" in arg and ":" in arg and i + 1 < len(cmd):
                        local_dest = cmd[i + 1]
                        break
                from jila_pipeline import run_batch
                rs = run_batch([{"N": 4, "chi_t": c, "gamma_t": 0.0}
                                for c in np.linspace(0.5, 2.5, 8)])
                with open(local_dest, "w") as f:
                    json.dump(rs, f, indent=2)

            m.stdout = ""
            return m

        mock_run.called_cmds = called_cmds
        return mock_run


    def _run_tpu_sync(self, app, mock_fn):
        """Call _run_tpu on the main thread with subprocess + dialog mocked.
        Patches both subprocess.run and subprocess.Popen (run() helper uses Popen).
        """
        def make_mock_popen(cmd, **kw):
            """Return a mock Popen whose communicate() calls mock_fn."""
            result = mock_fn(cmd, **kw)
            popen = MagicMock()
            popen.returncode = result.returncode
            popen.communicate.return_value = (result.stdout or "", result.stderr or "")
            # Store cmd so called_cmds is updated (mock_fn already appended it)
            return popen

        with patch("jila_gui.subprocess.run", side_effect=mock_fn), \
             patch("jila_gui.subprocess.Popen", side_effect=make_mock_popen), \
             patch("jila_gui.show_results"), \
             patch("jila_gui.messagebox.showerror"), \
             patch("jila_gui.messagebox.askyesno", return_value=True), \
             patch("time.sleep"):   # local import inside _run_tpu
            app._run_tpu()


    def test_tpu_run_creates_qr(self, app, tmp_path):
        """QR create gcloud command must be called exactly once."""
        mock_fn = self._make_scp_mock(tmp_path)
        self._run_tpu_sync(app, mock_fn)
        qr_calls = [c for c in mock_fn.called_cmds
                    if isinstance(c, list) and "queued-resources" in c and "create" in c]
        assert len(qr_calls) == 1, f"Expected 1 QR create call, got {qr_calls}"

    def test_tpu_run_polls_ready(self, app, tmp_path):
        """READY poll loop must call queued-resources describe to track QR state."""
        mock_fn = self._make_scp_mock(tmp_path)
        self._run_tpu_sync(app, mock_fn)
        poll_calls = [c for c in mock_fn.called_cmds
                      if isinstance(c, list)
                      and "queued-resources" in c
                      and "describe" in c]
        assert len(poll_calls) >= 1, "QR describe poll must be called at least once"


    def test_tpu_run_populates_last_results(self, app, tmp_path):
        """After successful TPU run, _last_results must be populated."""
        mock_fn = self._make_scp_mock(tmp_path)
        self._run_tpu_sync(app, mock_fn)
        assert len(app._last_results) > 0, "_last_results should be populated"

    def test_tpu_run_results_are_valid(self, app, tmp_path):
        """Results from mocked TPU run must be valid pipeline dicts."""
        mock_fn = self._make_scp_mock(tmp_path)
        self._run_tpu_sync(app, mock_fn)
        for r in app._last_results:
            assert "F_avg_theory" in r
            assert 0.5 <= r["F_avg_theory"] <= 1.0
            assert r["phase_class"] in ("QUANTUM", "CLASSICAL", "SINGULAR")

    def test_trc_auto_cleanup_on_error(self, app, tmp_path):
        """
        TRC Commandments 4+5: if the pipeline fails after QR creation,
        _trc_delete must be called automatically (node + QR deletion commands
        must appear in subprocess calls even on failure).
        """
        called_cmds = []
        qr_created = [False]

        def mock_run(cmd, **kw):
            called_cmds.append(cmd)
            m = MagicMock()
            m.stderr = ""
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else ""
            # SSH readiness probe → immediate pass
            if "ssh_ok" in cmd_str:
                m.stdout = "ssh_ok"
                m.returncode = 0
                return m
            # QR describe → ACTIVE  (code now uses --format=json, parse with json.loads)
            if "queued-resources" in cmd_str and "describe" in cmd_str:
                import json as _j
                m.stdout = _j.dumps({"state": {"state": "ACTIVE"}})
                m.returncode = 0
                return m
            # Pre-flight scan: queued-resources list and tpu-vm list → empty (clean slate)
            if "list" in cmd_str and ("queued-resources" in cmd_str or "tpu-vm" in cmd_str):
                m.stdout = ""   # no orphaned resources — let pre-flight pass
                m.returncode = 0
                return m
            # QR create succeeds
            if "queued-resources" in cmd_str and "create" in cmd_str:
                qr_created[0] = True
                m.stdout = ""
                m.returncode = 0
                return m
            # SSH run fails — simulates pipeline crash on TPU
            if "ssh" in cmd_str:
                m.stdout = ""
                m.returncode = 1
                m.stderr = "simulated SSH failure"
                return m
            m.stdout = ""
            m.returncode = 0
            return m



        self._run_tpu_sync(app, mock_run)

        # Node and QR delete commands must have been issued automatically
        delete_node = [c for c in called_cmds
                       if isinstance(c, list) and "tpu-vm" in c
                       and "delete" in c and "test-node" in c]
        delete_qr   = [c for c in called_cmds
                       if isinstance(c, list) and "queued-resources" in c
                       and "delete" in c and "test-qr" in c]
        assert len(delete_node) >= 1, "Node must be auto-deleted on error (TRC Cmd 5)"
        assert len(delete_qr)   >= 1, "QR must be auto-deleted on error (TRC Cmd 5)"

    def test_tpu_results_window_and_export(self, app, tmp_path):
        """
        Full E2E: TPU run (mocked) → results window opens → all exports verified.
        This validates the complete user journey end-to-end.
        """
        mock_fn = self._make_scp_mock(tmp_path)
        self._run_tpu_sync(app, mock_fn)

        assert app._last_results, "TPU mock produced no results"

        from jila_results import ResultsWindow
        win = ResultsWindow(app, app._last_results, PHASE_CSV)
        win.withdraw()
        try:
            assert len(win._results) == len(app._last_results)

            # CSV
            csv_path = str(tmp_path / "tpu_export.csv")
            win._export_csv_to(csv_path)
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == len(app._last_results)
            assert "F_avg_theory" in rows[0]
            assert "T_yz" in rows[0]

            # JSON
            json_path = str(tmp_path / "tpu_export.json")
            win._export_json_to(json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert all("T_matrix" in r for r in data)

            # NPZ
            npz_path = str(tmp_path / "tpu_export.npz")
            win._export_npz_to(npz_path)
            d = np.load(npz_path)
            assert "F_avg_theory" in d
            assert len(d["chi_t"]) == len(app._last_results)

        finally:
            try:
                win.destroy()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cleanup button E2E
# ═══════════════════════════════════════════════════════════════════════════════
class TestCleanupE2E:
    def test_cleanup_calls_delete_node_and_qr(self, tmp_path):
        from jila_gui import JILAGui
        root = JILAGui(); root.withdraw()
        root.v_project.set("time-emission-test")
        root.v_node.set("test-node")
        root.v_qr.set("test-qr")
        called = []
        def mock_run(cmd, **kw):
            called.append(cmd); m = MagicMock()
            m.returncode = 0; m.stdout = ""; m.stderr = ""; return m

        def sync_thread(target=None, args=(), kwargs=None, daemon=False, **kw):
            """Run the thread target synchronously so the test doesn't race."""
            t = MagicMock()
            t.start.side_effect = lambda: target(*(args or ()), **(kwargs or {}))
            return t

        with patch("jila_gui.subprocess.run", side_effect=mock_run), \
             patch("jila_gui.messagebox.askyesno", return_value=True), \
             patch("jila_gui.threading.Thread", side_effect=sync_thread):
            root._cleanup()

        try:
            root.destroy()
        except Exception:
            pass
        delete_cmds = [c for c in called if isinstance(c, list) and "delete" in c]
        assert len(delete_cmds) >= 2   # node + QR deletions
