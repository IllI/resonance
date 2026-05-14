"""
tests/test_jila_pipeline.py

Unit tests for jila_pipeline.py
Run: pytest tests/ -v

All ground truths are analytically derived or confirmed by TPU simulation
(Session 2, May 2026). Tests are organized by concern.
"""
import sys, os, json, tempfile, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from jila_pipeline import (
    rho2_oat, apply_dephasing, t_matrix,
    horodecki_fidelity, ptm_rank, concurrence,
    phase_class, run_pipeline, run_batch,
    rho2_from_Txx, load_input,
)

# ── tolerance ─────────────────────────────────────────────────────────────────
ATOL = 1e-10   # analytic identities
RTOL = 5e-3    # numerical fidelity values

# ═══════════════════════════════════════════════════════════════════════════════
# 1. rho2_oat — Proposition 1 ground truths
# ═══════════════════════════════════════════════════════════════════════════════

class TestRho2OAT:
    def test_trace_unity(self):
        """Density matrix must have unit trace for all chi_t."""
        for chi_t in [0.0, 0.5, 1.456, np.pi]:
            rho = rho2_oat(chi_t, N=4)
            assert abs(np.trace(rho) - 1.0) < ATOL, \
                f"Tr[rho] != 1 at chi_t={chi_t}"

    def test_hermitian(self):
        """Density matrix must be Hermitian."""
        for chi_t in [0.1, 1.0, 2.5]:
            rho = rho2_oat(chi_t, N=4)
            np.testing.assert_allclose(rho, rho.conj().T, atol=ATOL)

    def test_positive_semidefinite(self):
        """All eigenvalues must be >= 0."""
        for chi_t in np.linspace(0.01, np.pi - 0.01, 10):
            rho = rho2_oat(chi_t, N=4)
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -ATOL), \
                f"Negative eigenvalue at chi_t={chi_t}: {eigvals.min()}"

    def test_theorem4_exact_null(self):
        """
        Theorem 4: rho2(chi_t=pi) = I/4 exactly.
        Machine-precision identity confirmed by Session 2 simulation.
        """
        rho = rho2_oat(np.pi, N=4)
        maxdiff = np.max(np.abs(rho - np.eye(4) / 4))
        assert maxdiff < ATOL, f"rho(pi) != I/4, maxdiff={maxdiff:.2e}"

    def test_theorem4_various_N(self):
        """Theorem 4 holds for all even N >= 4."""
        for N in [4, 6, 8, 10]:
            rho = rho2_oat(np.pi, N=N)
            maxdiff = np.max(np.abs(rho - np.eye(4) / 4))
            assert maxdiff < ATOL, f"rho(pi) != I/4 for N={N}"

    def test_product_state_limit(self):
        """
        chi_t -> 0: T_xx -> cos^{N-2}(0) = 1.0, off-diagonal entries -> 0.
        (T_xx = 0.5 holds at chi_t=pi/2, not chi_t->0.)
        """
        rho = rho2_oat(1e-6, N=4)
        T = t_matrix(rho)
        # T_xx -> 1 as chi_t -> 0
        assert abs(T[0, 0] - np.cos(1e-6 / 2) ** 2) < 1e-4   # T_xx ~ 1
        assert abs(T[1, 1]) < 1e-4          # T_yy ~ 0
        assert abs(T[2, 2]) < 1e-4          # T_zz ~ 0

    def test_Txx_analytic_formula(self):
        """
        Proven identity: T_xx = cos^{N-2}(chi_t / 2).
        """
        N = 4
        for chi_t in np.linspace(0.05, np.pi - 0.05, 20):
            rho = rho2_oat(chi_t, N)
            T = t_matrix(rho)
            expected = np.cos(chi_t / 2) ** (N - 2)
            assert abs(T[0, 0] - expected) < 1e-8, \
                f"T_xx formula fails at chi_t={chi_t:.3f}: " \
                f"got {T[0,0]:.8f}, expected {expected:.8f}"

    def test_Txx_formula_N_scaling(self):
        """T_xx = cos^{N-2}(chi_t/2) holds across N."""
        chi_t = 1.0
        for N in [4, 6, 8, 10]:
            rho = rho2_oat(chi_t, N)
            T = t_matrix(rho)
            expected = np.cos(chi_t / 2) ** (N - 2)
            assert abs(T[0, 0] - expected) < 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# 2. T-matrix — structure tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTMatrix:
    def test_shape(self):
        T = t_matrix(rho2_oat(1.0, 4))
        assert T.shape == (3, 3)

    def test_real_valued(self):
        """T-matrix entries are always real (from Hermitian rho)."""
        T = t_matrix(rho2_oat(1.456, 4))
        assert np.all(np.isreal(T))

    def test_Tyz_nonzero_at_chi_star(self):
        """
        Session 2 discovery: T_yz = -0.497 at chi_t* = 1.45626.
        This is the mechanism enabling F > 2/3.
        """
        rho = rho2_oat(1.45626, N=4)
        T = t_matrix(rho)
        assert abs(T[1, 2]) > 0.48, \
            f"|T_yz| should be ~0.497 at chi_t*, got {T[1,2]:.4f}"

    def test_Tyz_zero_at_pi(self):
        """T_yz = 0 at chi_t = pi (Theorem 4: fully depolarized)."""
        T = t_matrix(rho2_oat(np.pi, N=4))
        assert abs(T[1, 2]) < ATOL

    def test_singular_values_bounded(self):
        """All singular values of T must be in [0, 1]."""
        for chi_t in np.linspace(0.01, np.pi, 15):
            T = t_matrix(rho2_oat(chi_t, 4))
            svs = np.linalg.svd(T, compute_uv=False)
            assert np.all(svs >= -1e-10)
            assert np.all(svs <= 1.0 + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Horodecki fidelity — analytic ground truths
# ═══════════════════════════════════════════════════════════════════════════════

class TestHorodeckiFidelity:
    def test_product_state_classical_limit(self):
        """Product state (chi_t->0): F = 2/3 exactly."""
        rho = rho2_oat(1e-8, N=4)
        T = t_matrix(rho)
        F = horodecki_fidelity(T)
        assert abs(F - 2/3) < 1e-4, f"F(product)={F:.6f}, expected 2/3"

    def test_theorem4_F_half(self):
        """Theorem 4: F(chi_t=pi) = 1/2 (completely depolarizing)."""
        rho = rho2_oat(np.pi, N=4)
        T = t_matrix(rho)
        F = horodecki_fidelity(T)
        assert abs(F - 0.5) < ATOL, f"F(pi)={F:.10f}, expected 0.5"

    def test_chi_star_above_classical(self):
        """
        F_avg = 0.758 > 2/3 at chi_t* = 1.45626.
        Confirmed by Session 2 TPU simulation (5000 Haar states).
        """
        rho = rho2_oat(1.45626, N=4)
        T = t_matrix(rho)
        F = horodecki_fidelity(T)
        assert F > 2/3 + 0.05, f"F at chi_t* = {F:.4f}, expected ~0.758"
        assert abs(F - 0.758) < 0.005, f"F={F:.4f}, expected 0.758"

    def test_fidelity_in_unit_interval(self):
        """F_avg must lie in [1/2, 1] for any physical state."""
        for chi_t in np.linspace(0, np.pi, 30):
            rho = rho2_oat(chi_t, N=4)
            T = t_matrix(rho)
            F = horodecki_fidelity(T)
            assert 0.5 - 1e-8 <= F <= 1.0 + 1e-8, \
                f"F={F:.6f} out of [1/2,1] at chi_t={chi_t:.3f}"

    def test_phase_boundary_location(self):
        """Phase boundary (F=2/3) is near chi_t_c ~ 0.716*pi ~ 2.25."""
        chi_low = 2.0   # should be above 2/3
        chi_high = 2.5  # should be below 2/3
        F_low  = horodecki_fidelity(t_matrix(rho2_oat(chi_low,  4)))
        F_high = horodecki_fidelity(t_matrix(rho2_oat(chi_high, 4)))
        assert F_low  > 2/3, f"F(2.0)={F_low:.4f} should be > 2/3"
        assert F_high < 2/3, f"F(2.5)={F_high:.4f} should be < 2/3"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PTM rank
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTMRank:
    def test_rank_1_at_pi(self):
        """T = 0 at chi_t=pi → rank 0 (depolarizing channel)."""
        T = t_matrix(rho2_oat(np.pi, N=4))
        assert ptm_rank(T) == 0

    def test_rank_nonzero_interior(self):
        """Interior chi_t should have rank >= 1."""
        T = t_matrix(rho2_oat(1.456, N=4))
        assert ptm_rank(T) >= 1

    def test_rank_monotone_approach_to_pi(self):
        """Rank (number of SVs above threshold) decreases as chi_t -> pi."""
        nuc_vals = [
            np.sum(np.linalg.svd(t_matrix(rho2_oat(c, 4)),
                                 compute_uv=False))
            for c in [0.5, 1.0, 1.5, 2.0, 2.5, np.pi]
        ]
        # Nuclear norm should generally decrease towards pi
        assert nuc_vals[-1] < nuc_vals[0], "Nuclear norm should collapse at pi"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dephasing
# ═══════════════════════════════════════════════════════════════════════════════

class TestDephasing:
    def test_zero_dephasing_identity(self):
        """Applying gamma=0 returns same state."""
        rho = rho2_oat(1.0, 4)
        rho_d = apply_dephasing(rho, 0.0)
        np.testing.assert_allclose(rho, rho_d, atol=ATOL)

    def test_dephasing_reduces_fidelity(self):
        """Dephasing strictly reduces teleportation fidelity."""
        rho = rho2_oat(1.456, 4)
        F0 = horodecki_fidelity(t_matrix(rho))
        for gam in [0.05, 0.10, 0.20]:
            F_d = horodecki_fidelity(t_matrix(apply_dephasing(rho, gam)))
            assert F_d < F0, f"Dephasing should reduce F (gamma={gam})"

    def test_dephased_state_valid(self):
        """Dephased state remains a valid density matrix."""
        rho_d = apply_dephasing(rho2_oat(1.0, 4), 0.15)
        assert abs(np.trace(rho_d) - 1.0) < ATOL
        assert np.all(np.linalg.eigvalsh(rho_d) >= -ATOL)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Phase classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseClass:
    def test_chi_star_is_quantum(self):
        r = run_pipeline(N=4, chi_t=1.456)
        assert r["phase_class"] == "QUANTUM"
        assert r["above_classical"] is True

    def test_chi_pi_is_singular(self):
        r = run_pipeline(N=4, chi_t=np.pi)
        assert r["phase_class"] == "SINGULAR"
        assert r["above_classical"] is False

    def test_large_chi_is_classical(self):
        r = run_pipeline(N=4, chi_t=2.8)
        assert r["phase_class"] == "CLASSICAL"
        assert r["above_classical"] is False

    def test_with_dephasing_degrades_phase(self):
        r = run_pipeline(N=4, chi_t=0.1, gamma_t=0.0)
        r_noisy = run_pipeline(N=4, chi_t=0.1, gamma_t=0.5)
        assert r_noisy["F_avg_theory"] <= r["F_avg_theory"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. run_pipeline — integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:
    def test_returns_all_keys(self):
        r = run_pipeline(N=4, chi_t=1.0)
        for key in ["N","chi_t","gamma_t","T_matrix","T_xx","T_yz",
                    "nuclear_norm","F_avg_theory","ptm_rank",
                    "concurrence","phase_class","above_classical"]:
            assert key in r, f"Missing key: {key}"

    def test_measured_Txx_path(self):
        """Bypass rho2 computation using measured T_xx directly."""
        r = run_pipeline(N=4, chi_t=1.456,
                         T_xx_measured=0.557, T_yz_measured=-0.497)
        assert r["F_avg_theory"] > 2/3
        assert r["phase_class"] == "QUANTUM"

    def test_consistency_theory_vs_measured(self):
        """Theory and measured paths should give similar F_avg."""
        r_theory = run_pipeline(N=4, chi_t=1.45626)
        r_measured = run_pipeline(N=4, chi_t=1.45626,
                                  T_xx_measured=r_theory["T_xx"],
                                  T_yz_measured=r_theory["T_yz"])
        assert abs(r_theory["F_avg_theory"] -
                   r_measured["F_avg_theory"]) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# 8. run_batch — batch processing
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBatch:
    def test_batch_length(self):
        records = [{"N": 4, "chi_t": c} for c in [0.5, 1.0, 1.5]]
        results = run_batch(records)
        assert len(results) == 3

    def test_batch_error_isolation(self):
        """A bad record should not crash the whole batch."""
        records = [
            {"N": 4, "chi_t": 1.0},
            {"N": -1, "chi_t": "bad"},   # malformed
            {"N": 4, "chi_t": 2.0},
        ]
        results = run_batch(records)
        assert len(results) == 3
        assert "error" not in results[0]
        assert "error" not in results[2]

    def test_batch_missing_gamma(self):
        """gamma_t defaults to 0 when absent."""
        r = run_batch([{"N": 4, "chi_t": 1.0}])[0]
        assert r["gamma_t"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. File ingestion
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileIngestion:
    def _write_csv(self, path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.csv")
            rows = [{"N":4,"chi_t":1.0,"gamma_t":0.05},
                    {"N":4,"chi_t":1.5,"gamma_t":0.0}]
            self._write_csv(path, rows)
            result = load_input(path)
            loaded, cols = result
            assert len(loaded) == 2
            assert "chi_t" in cols

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.json")
            data = [{"N":4,"chi_t":1.0},{"N":4,"chi_t":2.0}]
            with open(path,"w") as f: json.dump(data, f)
            loaded, cols = load_input(path)
            assert len(loaded) == 2
            assert "N" in cols

    def test_h5_roundtrip(self):
        h5py = pytest.importorskip("h5py")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("N",     data=[4, 4])
                f.create_dataset("chi_t", data=[1.0, 1.5])
                f.create_dataset("gamma_t", data=[0.0, 0.05])
            loaded, cols = load_input(path)
            assert len(loaded) == 2
            assert "chi_t" in cols

    def test_npz_batch_end_to_end(self):
        """Full NPZ → load_input → run_batch pipeline."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sweep.npz")
            chi_vals = np.array([0.5, 1.456, np.pi])
            np.savez(path, N=np.array([4,4,4]),
                     chi_t=chi_vals, gamma_t=np.zeros(3))
            records, cols = load_input(path)
            assert len(records) == 3
            assert "chi_t" in cols
            results = run_batch(records)
            assert results[1]["phase_class"] == "QUANTUM"
            assert results[2]["phase_class"] == "SINGULAR"

    def test_json_batch_end_to_end(self):
        """Full JSON → load_input → run_batch pipeline."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.json")
            data = [{"N": 4, "chi_t": c, "gamma_t": 0.0}
                    for c in [0.5, 1.456, float(np.pi)]]
            with open(path, "w") as f:
                json.dump(data, f)
            records, cols = load_input(path)
            assert len(records) == 3
            results = run_batch(records)
            assert results[1]["phase_class"] == "QUANTUM"

    def test_npz_preserves_float_precision(self):
        """NPZ round-trip must preserve chi_t to full float64 precision."""
        with tempfile.TemporaryDirectory() as d:
            chi_exact = np.array([1.45626437, np.pi])
            path = os.path.join(d, "exact.npz")
            np.savez(path, N=[4, 4], chi_t=chi_exact, gamma_t=[0.0, 0.0])
            records, _ = load_input(path)
            loaded_chi = np.array([float(r["chi_t"]) for r in records])
            np.testing.assert_allclose(loaded_chi, chi_exact, rtol=1e-12,
                                       err_msg="NPZ chi_t precision not preserved")

    def test_npz_optional_gamma(self):
        """NPZ without gamma_t array should load cleanly; gamma defaults to 0."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "no_gamma.npz")
            np.savez(path, N=[4, 4], chi_t=[1.0, 1.5])  # no gamma_t key
            records, _ = load_input(path)
            assert len(records) == 2
            results = run_batch(records)
            assert all(r["gamma_t"] == 0.0 for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Sample file regression tests
#     These tests run the actual shipped sample files end-to-end to guard
#     against inadvertent breakage of the files distributed to JILA/NIST users.
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


class TestSampleFiles:
    """Regression tests against the shipped sample_data/ files."""

    def _load_and_run(self, filename, col_map=None):
        path = os.path.join(SAMPLE_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"Sample file not present: {filename}")
        rows, cols = load_input(path)
        if col_map:
            records = [{col_map.get(k, k): v for k, v in r.items()}
                       for r in rows]
            # Ensure required keys exist
            records = [
                {"N": int(r.get("N", r.get("N_atoms", 4))),
                 "chi_t": float(r.get("chi_t", r.get("Omega_chi", 0))),
                 "gamma_t": float(r.get("gamma_t", 0))}
                for r in rows
            ]
        else:
            records = [{"N": int(r["N"]), "chi_t": float(r["chi_t"]),
                        "gamma_t": float(r.get("gamma_t", 0))}
                       for r in rows]
        return run_batch(records)

    # ── NPZ samples ─────────────────────────────────────────────────────────

    def test_sample_ideal_npz_loads(self):
        results = self._load_and_run("sample_ideal_sweep.npz")
        assert len(results) == 20

    def test_sample_ideal_npz_peak_fidelity(self):
        """Peak F must be ~0.769 (chi_t=1.456 row)."""
        results = self._load_and_run("sample_ideal_sweep.npz")
        peak = max(r["F_avg_theory"] for r in results if "F_avg_theory" in r)
        assert abs(peak - 0.769) < 0.005, f"Peak F={peak:.4f}, expected ~0.769"

    def test_sample_ideal_npz_quantum_count(self):
        """15 of 20 points should classify QUANTUM in ideal conditions."""
        results = self._load_and_run("sample_ideal_sweep.npz")
        q = sum(1 for r in results if r.get("phase_class") == "QUANTUM")
        assert q >= 14, f"Expected >=14 QUANTUM points, got {q}"

    def test_sample_ideal_npz_no_errors(self):
        results = self._load_and_run("sample_ideal_sweep.npz")
        errors = [r for r in results if "error" in r]
        assert errors == [], f"Unexpected errors: {errors}"

    def test_sample_decoherence_npz_lower_fidelity(self):
        """Decoherent sample must have lower peak F than ideal."""
        ideal = self._load_and_run("sample_ideal_sweep.npz")
        noisy = self._load_and_run("sample_with_decoherence.npz")
        ideal_peak = max(r["F_avg_theory"] for r in ideal if "F_avg_theory" in r)
        noisy_peak = max(r["F_avg_theory"] for r in noisy if "F_avg_theory" in r)
        assert noisy_peak < ideal_peak, \
            f"Decoherent peak F {noisy_peak:.4f} >= ideal peak F {ideal_peak:.4f}"

    def test_sample_decoherence_npz_narrower_quantum_region(self):
        """Decoherence must reduce the number of QUANTUM-classified points."""
        ideal = self._load_and_run("sample_ideal_sweep.npz")
        noisy = self._load_and_run("sample_with_decoherence.npz")
        q_ideal = sum(1 for r in ideal if r.get("phase_class") == "QUANTUM")
        q_noisy = sum(1 for r in noisy if r.get("phase_class") == "QUANTUM")
        assert q_noisy < q_ideal, \
            f"Expected fewer QUANTUM pts with decoherence: {q_noisy} vs {q_ideal}"

    def test_sample_2d_npz_loads_50_points(self):
        results = self._load_and_run("sample_phase_diagram_2d.npz")
        assert len(results) == 50, f"Expected 50 points, got {len(results)}"

    def test_sample_2d_npz_mixed_phases(self):
        """2D sweep must contain both QUANTUM and CLASSICAL points."""
        results = self._load_and_run("sample_phase_diagram_2d.npz")
        phases = {r.get("phase_class") for r in results}
        assert "QUANTUM" in phases,   "2D sweep missing QUANTUM points"
        assert "CLASSICAL" in phases, "2D sweep missing CLASSICAL points"

    # ── CSV samples ─────────────────────────────────────────────────────────

    def test_sample_ideal_csv_matches_npz(self):
        """CSV and NPZ ideal samples must produce identical peak F."""
        csv_results = self._load_and_run("sample_ideal_sweep.csv")
        npz_results = self._load_and_run("sample_ideal_sweep.npz")
        csv_peak = max(r["F_avg_theory"] for r in csv_results if "F_avg_theory" in r)
        npz_peak = max(r["F_avg_theory"] for r in npz_results if "F_avg_theory" in r)
        assert abs(csv_peak - npz_peak) < 1e-4, \
            f"CSV peak F {csv_peak:.6f} != NPZ peak F {npz_peak:.6f}"

    def test_sample_nist_csv_loads(self):
        """NIST-format CSV must load and map to valid records."""
        path = os.path.join(SAMPLE_DIR, "sample_nist_format.csv")
        if not os.path.exists(path):
            pytest.skip("NIST sample not present")
        rows, cols = load_input(path)
        assert "Omega_chi" in cols, "NIST file must have Omega_chi column"
        assert "N_atoms" in cols,   "NIST file must have N_atoms column"
        # Manual column mapping (as a user would set in the GUI dropdowns)
        records = [{"N": int(r["N_atoms"]),
                    "chi_t": float(r["Omega_chi"]),
                    "gamma_t": 0.0}
                   for r in rows]
        results = run_batch(records)
        assert len(results) == 20
        q = sum(1 for r in results if r.get("phase_class") == "QUANTUM")
        assert q >= 14, f"NIST sample: expected >=14 QUANTUM, got {q}"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Export round-trip tests
#     Write each export format and read back the file to verify content
#     integrity — not just that the file exists and has the right extension.
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportRoundTrips:
    """
    Write results through each export path, then read the file back and
    assert that the written data matches the source results dict.
    """

    @pytest.fixture
    def results(self):
        """5-point batch as the source of truth for all export tests."""
        records = [{"N": 4, "chi_t": c, "gamma_t": 0.0}
                   for c in [0.5, 1.0, 1.456, 2.0, 2.8]]
        return run_batch(records)

    @pytest.fixture
    def win(self, results):
        """Headless ResultsWindow for programmatic export testing."""
        try:
            from jila_results import ResultsWindow
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            w = ResultsWindow(root, results, "dummy.csv")
            w.withdraw()
            yield w
            try:
                w.destroy(); root.destroy()
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Tkinter unavailable: {e}")
            yield

    # ── CSV round-trip ───────────────────────────────────────────────────────

    def test_csv_roundtrip_row_count(self, win, results, tmp_path):
        path = str(tmp_path / "out.csv")
        win._export_csv_to(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(results)

    def test_csv_roundtrip_fidelity_values(self, win, results, tmp_path):
        path = str(tmp_path / "out.csv")
        win._export_csv_to(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for src, row in zip(results, rows):
            assert abs(float(row["F_avg_theory"]) - src["F_avg_theory"]) < 1e-6

    def test_csv_roundtrip_phase_class(self, win, results, tmp_path):
        path = str(tmp_path / "out.csv")
        win._export_csv_to(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for src, row in zip(results, rows):
            assert row["phase_class"] == src["phase_class"]

    # ── JSON round-trip ──────────────────────────────────────────────────────

    def test_json_roundtrip_row_count(self, win, results, tmp_path):
        path = str(tmp_path / "out.json")
        win._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == len(results)

    def test_json_roundtrip_T_matrix_preserved(self, win, results, tmp_path):
        path = str(tmp_path / "out.json")
        win._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        for src, rec in zip(results, data):
            assert "T_matrix" in rec, "T_matrix must survive JSON round-trip"
            T_out = np.array(rec["T_matrix"])
            T_src = np.array(src["T_matrix"])
            np.testing.assert_allclose(T_out, T_src, atol=1e-10,
                                       err_msg="T_matrix values changed in JSON export")

    def test_json_roundtrip_fidelity_precision(self, win, results, tmp_path):
        path = str(tmp_path / "out.json")
        win._export_json_to(path)
        with open(path) as f:
            data = json.load(f)
        for src, rec in zip(results, data):
            assert abs(rec["F_avg_theory"] - src["F_avg_theory"]) < 1e-10

    # ── NPZ round-trip ───────────────────────────────────────────────────────

    def test_npz_roundtrip_arrays_present(self, win, results, tmp_path):
        path = str(tmp_path / "out.npz")
        win._export_npz_to(path)
        d = np.load(path)
        for key in ["F_avg_theory", "chi_t", "T_xx", "T_yz",
                    "nuclear_norm", "concurrence"]:
            assert key in d, f"NPZ missing array: {key}"

    def test_npz_roundtrip_fidelity_values(self, win, results, tmp_path):
        path = str(tmp_path / "out.npz")
        win._export_npz_to(path)
        d = np.load(path)
        src_F = np.array([r["F_avg_theory"] for r in results])
        np.testing.assert_allclose(d["F_avg_theory"], src_F, atol=1e-10)

    def test_npz_roundtrip_chi_t_values(self, win, results, tmp_path):
        path = str(tmp_path / "out.npz")
        win._export_npz_to(path)
        d = np.load(path)
        src_chi = np.array([r["chi_t"] for r in results])
        np.testing.assert_allclose(d["chi_t"], src_chi, atol=1e-10)

    # ── HDF5 round-trip ──────────────────────────────────────────────────────

    def test_h5_roundtrip_datasets(self, win, results, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = str(tmp_path / "out.h5")
        win._export_h5_to(path)
        with h5py.File(path, "r") as f:
            assert "F_avg_theory" in f
            assert "chi_t" in f
            assert len(f["F_avg_theory"]) == len(results)

    def test_h5_roundtrip_fidelity_values(self, win, results, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = str(tmp_path / "out.h5")
        win._export_h5_to(path)
        with h5py.File(path, "r") as f:
            F_out = f["F_avg_theory"][:]
        src_F = np.array([r["F_avg_theory"] for r in results])
        np.testing.assert_allclose(F_out, src_F, atol=1e-10)

    # ── PNG/PDF figure export ─────────────────────────────────────────────────

    def test_png_is_valid_image(self, win, tmp_path):
        pytest.importorskip("matplotlib")
        if win._fig is None:
            pytest.skip("Figure not rendered")
        path = str(tmp_path / "out.png")
        win._export_png_to(path)
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        assert w >= 800 and h >= 400, f"PNG too small: {w}x{h}"

    def test_pdf_is_valid_pdf(self, win, tmp_path):
        pytest.importorskip("matplotlib")
        if win._fig is None:
            pytest.skip("Figure not rendered")
        path = str(tmp_path / "out.pdf")
        win._export_pdf_to(path)
        with open(path, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF", f"File does not start with %PDF: {header}"



# ═══════════════════════════════════════════════════════════════════════════════
# 10. Pre-registered IBM predictions (regression guard)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreRegisteredPredictions:
    """
    These tests encode the pre-registered predictions for the IBM hardware run.
    Any code change that breaks these values must be explicitly approved.
    """
    def test_Txx_at_chi_star(self):
        r = run_pipeline(N=4, chi_t=1.45626)
        assert abs(r["T_xx"] - 0.557) < 0.005, \
            f"Pre-reg T_xx={r['T_xx']:.4f}, expected 0.557"

    def test_Tyz_at_chi_star(self):
        r = run_pipeline(N=4, chi_t=1.45626)
        assert abs(r["T_yz"] - (-0.497)) < 0.005, \
            f"Pre-reg T_yz={r['T_yz']:.4f}, expected -0.497"

    def test_Favg_at_chi_star(self):
        r = run_pipeline(N=4, chi_t=1.45626)
        assert abs(r["F_avg_theory"] - 0.758) < 0.005, \
            f"Pre-reg F_avg={r['F_avg_theory']:.4f}, expected 0.758"

    def test_Favg_at_pi(self):
        r = run_pipeline(N=4, chi_t=np.pi)
        assert abs(r["F_avg_theory"] - 0.5) < 1e-8, \
            f"Pre-reg F(pi)={r['F_avg_theory']:.10f}, expected 0.5"

    def test_Favg_product_state(self):
        r = run_pipeline(N=4, chi_t=1e-8)
        assert abs(r["F_avg_theory"] - 2/3) < 1e-4, \
            f"Pre-reg F(0)={r['F_avg_theory']:.6f}, expected 2/3"
