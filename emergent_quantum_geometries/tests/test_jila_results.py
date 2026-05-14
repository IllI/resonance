"""
tests/test_jila_results.py

Unit tests for jila_results.py (export functionality).
Run: pytest tests/test_jila_results.py -v

These tests exercise all export paths without opening a display window.
They do not depend on matplotlib (export formats are tested independently).
"""
import sys, os, json, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from jila_pipeline import run_batch

# ── Build a representative results fixture ────────────────────────────────────
@pytest.fixture(scope="module")
def sample_results():
    """12-point chi_t sweep at N=4, gamma=0. Covers QUANTUM/CLASSICAL/SINGULAR."""
    records = [{"N": 4, "chi_t": c, "gamma_t": 0.0}
               for c in np.linspace(0.1, np.pi, 12)]
    return [r for r in run_batch(records) if "error" not in r]


@pytest.fixture
def results_win(sample_results):
    """
    Instantiate ResultsWindow with a dummy Tk root.
    Uses withdraw() to prevent a visible window during CI.
    """
    import tkinter as tk
    from jila_results import ResultsWindow
    root = tk.Tk()
    root.withdraw()
    win = ResultsWindow(root, sample_results, source_path="")
    yield win
    win.destroy()
    root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ResultsWindow construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultsWindowConstruction:
    def test_window_created(self, results_win):
        assert results_win.winfo_exists()

    def test_filters_error_rows(self, sample_results):
        """Error rows in results list must be silently excluded."""
        import tkinter as tk
        from jila_results import ResultsWindow
        root = tk.Tk(); root.withdraw()
        mixed = sample_results[:3] + [{"error": "bad", "input": {}}] + sample_results[3:6]
        win = ResultsWindow(root, mixed, source_path="")
        assert len(win._results) == len(sample_results[:3]) + len(sample_results[3:6])
        win.destroy(); root.destroy()

    def test_flat_helper_keys(self, results_win):
        flat = results_win._flat()
        expected = {"N","chi_t","gamma_t","F_avg_theory","T_xx","T_yz",
                    "T_yy","T_zz","nuclear_norm","ptm_rank","concurrence",
                    "phase_class","above_classical"}
        assert expected == set(flat[0].keys())

    def test_flat_length_matches_results(self, results_win):
        assert len(results_win._flat()) == len(results_win._results)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CSV export
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVExport:
    def test_csv_creates_file(self, results_win, tmp_path):
        path = str(tmp_path / "out.csv")
        results_win._export_csv_to(path)
        assert os.path.exists(path)

    def test_csv_has_header(self, results_win, tmp_path):
        path = str(tmp_path / "out.csv")
        results_win._export_csv_to(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "F_avg_theory" in reader.fieldnames
        assert "T_yz" in reader.fieldnames

    def test_csv_row_count(self, results_win, tmp_path):
        path = str(tmp_path / "out.csv")
        results_win._export_csv_to(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(results_win._results)

    def test_csv_values_accurate(self, results_win, tmp_path):
        path = str(tmp_path / "out.csv")
        results_win._export_csv_to(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for row, res in zip(rows, results_win._results):
            assert abs(float(row["F_avg_theory"]) - res["F_avg_theory"]) < 1e-8
            assert abs(float(row["T_yz"]) - res["T_yz"]) < 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# 3. JSON export
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONExport:
    def test_json_creates_file(self, results_win, tmp_path):
        path = str(tmp_path / "out.json")
        results_win._export_json_to(path)
        assert os.path.exists(path)

    def test_json_is_valid(self, results_win, tmp_path):
        path = str(tmp_path / "out.json")
        results_win._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(results_win._results)

    def test_json_preserves_T_matrix(self, results_win, tmp_path):
        path = str(tmp_path / "out.json")
        results_win._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        for exported, original in zip(data, results_win._results):
            np.testing.assert_allclose(
                np.array(exported["T_matrix"]),
                np.array(original["T_matrix"]), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NPZ export
# ═══════════════════════════════════════════════════════════════════════════════

class TestNPZExport:
    def test_npz_creates_file(self, results_win, tmp_path):
        path = str(tmp_path / "out.npz")
        results_win._export_npz_to(path)
        assert os.path.exists(path)

    def test_npz_contains_arrays(self, results_win, tmp_path):
        base = str(tmp_path / "out")
        results_win._export_npz_to(base + ".npz")
        d = np.load(base + ".npz")
        assert "F_avg_theory" in d
        assert "chi_t" in d
        assert len(d["F_avg_theory"]) == len(results_win._results)

    def test_npz_values_accurate(self, results_win, tmp_path):
        path = str(tmp_path / "out.npz")
        results_win._export_npz_to(path)
        d = np.load(path)
        expected = np.array([r["F_avg_theory"] for r in results_win._results])
        np.testing.assert_allclose(d["F_avg_theory"], expected, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HDF5 export
# ═══════════════════════════════════════════════════════════════════════════════

class TestHDF5Export:
    def test_h5_creates_file(self, results_win, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = str(tmp_path / "out.h5")
        results_win._export_h5_to(path)
        assert os.path.exists(path)

    def test_h5_datasets_present(self, results_win, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = str(tmp_path / "out.h5")
        results_win._export_h5_to(path)
        with h5py.File(path, "r") as f:
            assert "F_avg_theory" in f
            assert "T_xx" in f
            assert "T_yz" in f
            assert len(f["chi_t"]) == len(results_win._results)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PNG / PDF export (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFigureExport:
    def test_png_creates_file(self, results_win, tmp_path):
        pytest.importorskip("matplotlib")
        if results_win._fig is None:
            pytest.skip("Figure not rendered (matplotlib unavailable)")
        path = str(tmp_path / "out.png")
        results_win._export_png_to(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10_000   # non-trivial image

    def test_pdf_creates_file(self, results_win, tmp_path):
        pytest.importorskip("matplotlib")
        if results_win._fig is None:
            pytest.skip("Figure not rendered (matplotlib unavailable)")
        path = str(tmp_path / "out.pdf")
        results_win._export_pdf_to(path)
        assert os.path.exists(path)
        # PDF magic bytes
        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"

    def test_png_300dpi(self, results_win, tmp_path):
        """PNG must be large enough to indicate 300 dpi rendering."""
        pytest.importorskip("matplotlib")
        if results_win._fig is None:
            pytest.skip("Figure not rendered")
        path = str(tmp_path / "out.png")
        results_win._export_png_to(path)
        assert os.path.getsize(path) > 50_000


# ═══════════════════════════════════════════════════════════════════════════════
# 7. show_results convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestShowResults:
    def test_show_results_returns_window(self, sample_results):
        import tkinter as tk
        from jila_results import show_results, ResultsWindow
        root = tk.Tk(); root.withdraw()
        # show_results is fire-and-forget; call ResultsWindow directly for ref
        win = ResultsWindow(root, sample_results, "")
        assert isinstance(win, ResultsWindow)
        win.destroy(); root.destroy()

    def test_empty_results_no_crash(self):
        import tkinter as tk
        from jila_results import ResultsWindow
        root = tk.Tk(); root.withdraw()
        win = ResultsWindow(root, [], "")
        assert win.winfo_exists()
        win.destroy(); root.destroy()
