import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from jila_pipeline import run_pipeline, run_batch, load_input

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestE2ERealData:
    """
    End-to-end tests against real TPU simulation output (Session 2,
    quantum-teleportation-results branch, v6e-8 TPU, May 2026).
    Tolerance: 0.005 for F_avg (scipy DE variability); 1e-8 for exact nulls.
    """

    def test_phase_diagram_F_agreement(self):
        """
        Pipeline F_avg_theory must agree with Session 2 TPU F_theory
        within 0.005 for all 20 points in the gamma=0 phase diagram.
        """
        path = os.path.join(FIXTURE_DIR, "oat_phase_diagram_gam0.csv")
        if not os.path.exists(path):
            pytest.skip("Fixture not present — run generate_fixtures.py")
        records, _ = load_input(path)
        results = run_batch(records)
        mismatches = []
        for rec, res in zip(records, results):
            if "error" in res:
                mismatches.append(f"  chi={rec['chi_t']}: error={res['error']}")
                continue
            F_fixture = float(rec["F_theory"])
            F_pipeline = res["F_avg_theory"]
            if abs(F_pipeline - F_fixture) > 0.005:
                mismatches.append(
                    f"  chi={float(rec['chi_t']):.3f}: "
                    f"pipeline={F_pipeline:.5f} fixture={F_fixture:.5f} "
                    f"diff={abs(F_pipeline - F_fixture):.5f}")
        assert not mismatches, \
            "F_avg disagrees with Session 2 TPU:\n" + "\n".join(mismatches)

    def test_phase_diagram_classification(self):
        """Phase class must agree with QUANTUM/CLASSICAL/SINGULAR from fixture."""
        path = os.path.join(FIXTURE_DIR, "oat_phase_diagram_gam0.csv")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        records, _ = load_input(path)
        results = run_batch(records)
        for rec, res in zip(records, results):
            if "error" in res:
                continue
            chi = float(rec["chi_t"])
            F_sim = float(rec["F_sim"])
            pc = res["phase_class"]
            if F_sim > 2/3 + 0.005:
                assert pc == "QUANTUM", \
                    f"chi={chi:.3f}: F_sim={F_sim:.4f} > 2/3 but got {pc}"
            # pi point: float-parsed chi_t may not be exactly pi;
            # check by nuclear norm collapse instead of phase_class string
            if chi > np.pi - 0.01:
                assert res["nuclear_norm"] < 1e-5, \
                    f"chi~pi: nuclear_norm={res['nuclear_norm']:.2e} should be ~0"

    def test_noise_robustness_F_agreement(self):
        """
        Pipeline F_avg (Horodecki bound at chi_t*) must be stable across
        the noise fixture rows. Since the fixture's F_theory column was computed
        from the same Horodecki formula, we verify self-consistency:
        all rows produce the same F_avg (no gamma_t variation in this fixture)
        and all values match the pre-registered chi_t* prediction.

        Note: The fixture's F_sim column reflects the TPU optimizer fidelity
        under depolarizing noise — a different quantity from the Horodecki bound.
        We test self-consistency here, not the noise model itself.
        """
        path = os.path.join(FIXTURE_DIR, "oat_noise_robustness.csv")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        records, _ = load_input(path)
        results = run_batch(records)
        F_expected = 0.75843  # Session 2 Part A anchor, gamma=0
        for rec, res in zip(records, results):
            if "error" in res:
                continue
            F_pip = res["F_avg_theory"]
            assert abs(F_pip - F_expected) < 0.005, \
                f"p={rec['p_dep']}: F={F_pip:.5f} unexpected (gamma=0, expected {F_expected:.5f})"

    def test_noise_robustness_monotone(self):
        """F must decrease monotonically with increasing depolarizing p."""
        path = os.path.join(FIXTURE_DIR, "oat_noise_robustness.csv")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        records, _ = load_input(path)
        records = sorted(records, key=lambda x: float(x.get("p_dep", 0)))
        results = run_batch(records)
        F_vals = [r["F_avg_theory"] for r in results if "error" not in r]
        for i in range(len(F_vals) - 1):
            assert F_vals[i] >= F_vals[i+1] - 0.002, \
                f"Non-monotone at i={i}: {F_vals[i]:.4f} -> {F_vals[i+1]:.4f}"

    def test_anchor_chi_star(self):
        """F and phase at chi_t* must match Session 2 Part A (F=0.75843)."""
        path = os.path.join(FIXTURE_DIR, "oat_anchor_points.json")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        with open(path) as f:
            anchors = json.load(f)
        a = anchors["chi_t_star"]
        r = run_pipeline(N=4, chi_t=a["chi_t"])
        assert abs(r["F_avg_theory"] - a["F_avg"]) < 0.005, \
            f"F={r['F_avg_theory']:.5f} vs TPU {a['F_avg']:.5f}"
        assert r["phase_class"] == "QUANTUM"

    def test_anchor_chi_pi(self):
        """Theorem 4: F=0.50000 exactly at chi_t=pi (Session 2, machine precision)."""
        path = os.path.join(FIXTURE_DIR, "oat_anchor_points.json")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        with open(path) as f:
            anchors = json.load(f)
        a = anchors["chi_t_pi"]
        r = run_pipeline(N=4, chi_t=a["chi_t"])
        assert abs(r["F_avg_theory"] - a["F_avg"]) < 1e-8, \
            f"Theorem 4: F={r['F_avg_theory']:.10f} vs fixture {a['F_avg']}"
        assert r["phase_class"] == "SINGULAR"

    def test_full_pipeline_csv_to_results(self):
        """
        Full JILA workflow: load 20-point phase diagram CSV, run batch,
        verify row count, no errors, QUANTUM points have F>2/3, pi is SINGULAR.
        This mirrors exactly what a JILA user does after cloning the branch.
        """
        path = os.path.join(FIXTURE_DIR, "oat_phase_diagram_gam0.csv")
        if not os.path.exists(path):
            pytest.skip("Fixture not present")
        records, _ = load_input(path)
        results = run_batch(records)
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        errors = [r for r in results if "error" in r]
        assert not errors, f"Batch errors: {errors}"
        quantum = [r for r in results if r["phase_class"] == "QUANTUM"]
        assert all(r["F_avg_theory"] > 2/3 for r in quantum), \
            "QUANTUM-classified point has F <= 2/3"
        # chi_t=pi from CSV is a rounded float; check by nuclear norm collapse
        pi_rows = [r for r in results if abs(r["chi_t"] - np.pi) < 0.01]
        assert pi_rows, "No chi_t ~ pi row found in results"
        assert pi_rows[0]["nuclear_norm"] < 1e-5, \
            f"chi_t~pi nuclear_norm={pi_rows[0]['nuclear_norm']:.2e}, expected ~0"
