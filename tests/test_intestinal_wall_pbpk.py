"""Tests for intestinal wall PBPK model (Phase 447)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.intestinal_wall_pbpk import (
    IntestinalWallPKResult,
    gut_wall_extraction_sensitivity,
    simulate_intestinal_wall_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sim(**kwargs) -> IntestinalWallPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        solubility_mg_mL=1.0,
        peff_cm_s=1e-4,
        cl_gut_L_per_h=1.0,
        vd_gut_L=1.0,
        cl_portal_L_per_h=5.0,
        vd_portal_L=3.0,
        t_end_h=8.0,
        dt_h=0.01,
    )
    defaults.update(kwargs)
    return simulate_intestinal_wall_pk(**defaults)


# ---------------------------------------------------------------------------
# Type and structure tests
# ---------------------------------------------------------------------------


class TestSimulateIntestinalWallPKStructure:
    def test_returns_correct_type(self):
        result = _default_sim()
        assert isinstance(result, IntestinalWallPKResult)

    def test_result_has_all_fields(self):
        result = _default_sim()
        assert hasattr(result, "drug_name")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "peff_cm_s")
        assert hasattr(result, "cl_gut_L_per_h")
        assert hasattr(result, "times_h")
        assert hasattr(result, "a_lumen_mg")
        assert hasattr(result, "c_gut_mg_L")
        assert hasattr(result, "c_portal_mg_L")
        assert hasattr(result, "auc_portal")
        assert hasattr(result, "fg_estimated")
        assert hasattr(result, "notes")

    def test_list_fields_are_lists(self):
        result = _default_sim()
        assert isinstance(result.times_h, list)
        assert isinstance(result.a_lumen_mg, list)
        assert isinstance(result.c_gut_mg_L, list)
        assert isinstance(result.c_portal_mg_L, list)

    def test_time_array_length_consistent(self):
        result = _default_sim(t_end_h=4.0, dt_h=0.1)
        n = len(result.times_h)
        assert len(result.a_lumen_mg) == n
        assert len(result.c_gut_mg_L) == n
        assert len(result.c_portal_mg_L) == n

    def test_drug_name_preserved(self):
        result = _default_sim(drug_name="Midazolam")
        assert result.drug_name == "Midazolam"

    def test_dose_mg_preserved(self):
        result = _default_sim(dose_mg=250.0)
        assert result.dose_mg == 250.0

    def test_peff_preserved(self):
        result = _default_sim(peff_cm_s=5e-4)
        assert result.peff_cm_s == 5e-4

    def test_cl_gut_preserved(self):
        result = _default_sim(cl_gut_L_per_h=2.5)
        assert result.cl_gut_L_per_h == 2.5


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_initial_lumen_amount_equals_dose(self):
        result = _default_sim(dose_mg=200.0)
        assert abs(result.a_lumen_mg[0] - 200.0) < 1e-9

    def test_initial_gut_concentration_zero(self):
        result = _default_sim()
        assert result.c_gut_mg_L[0] == 0.0

    def test_initial_portal_concentration_zero(self):
        result = _default_sim()
        assert result.c_portal_mg_L[0] == 0.0

    def test_first_time_point_is_zero(self):
        result = _default_sim()
        assert result.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Dynamics / physics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_lumen_amount_monotonically_decreasing(self):
        result = _default_sim(peff_cm_s=1e-3)
        lumen = result.a_lumen_mg
        for i in range(1, len(lumen)):
            assert lumen[i] <= lumen[i - 1] + 1e-9

    def test_lumen_amount_nonnegative(self):
        result = _default_sim()
        assert all(a >= 0.0 for a in result.a_lumen_mg)

    def test_gut_concentration_nonnegative(self):
        result = _default_sim()
        assert all(c >= 0.0 for c in result.c_gut_mg_L)

    def test_portal_concentration_nonnegative(self):
        result = _default_sim()
        assert all(c >= 0.0 for c in result.c_portal_mg_L)

    def test_portal_concentration_rises_then_falls(self):
        result = _default_sim(peff_cm_s=5e-4, t_end_h=10.0, dt_h=0.05)
        cmax_idx = result.c_portal_mg_L.index(max(result.c_portal_mg_L))
        # Peak should not be at very beginning or very end
        assert 0 < cmax_idx < len(result.c_portal_mg_L) - 1

    def test_higher_peff_gives_higher_portal_cmax(self):
        low = _default_sim(peff_cm_s=1e-5)
        high = _default_sim(peff_cm_s=1e-3)
        assert max(high.c_portal_mg_L) > max(low.c_portal_mg_L)

    def test_zero_cl_gut_gives_higher_fg(self):
        """No gut wall metabolism -> more drug reaches portal."""
        zero_cl = _default_sim(cl_gut_L_per_h=0.0)
        high_cl = _default_sim(cl_gut_L_per_h=10.0)
        assert zero_cl.fg_estimated >= high_cl.fg_estimated


# ---------------------------------------------------------------------------
# AUC and fg
# ---------------------------------------------------------------------------


class TestAUCAndFg:
    def test_auc_portal_positive(self):
        result = _default_sim()
        assert result.auc_portal > 0.0

    def test_fg_between_0_and_1(self):
        result = _default_sim()
        assert 0.0 <= result.fg_estimated <= 1.0

    def test_high_cl_gut_reduces_fg(self):
        low = _default_sim(cl_gut_L_per_h=0.1)
        high = _default_sim(cl_gut_L_per_h=50.0)
        assert low.fg_estimated > high.fg_estimated

    def test_auc_scales_with_dose(self):
        r1 = _default_sim(dose_mg=100.0)
        r2 = _default_sim(dose_mg=200.0)
        # AUC should approximately double
        ratio = r2.auc_portal / r1.auc_portal
        assert 1.8 < ratio < 2.2

    def test_notes_contains_ka(self):
        result = _default_sim()
        assert "ka=" in result.notes


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default_sim(drug_name="")

    def test_whitespace_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default_sim(drug_name="   ")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=-10.0)

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            _default_sim(peff_cm_s=0.0)

    def test_negative_cl_gut_raises(self):
        with pytest.raises(ValueError, match="cl_gut_L_per_h"):
            _default_sim(cl_gut_L_per_h=-1.0)

    def test_zero_vd_gut_raises(self):
        with pytest.raises(ValueError, match="vd_gut_L"):
            _default_sim(vd_gut_L=0.0)

    def test_zero_vd_portal_raises(self):
        with pytest.raises(ValueError, match="vd_portal_L"):
            _default_sim(vd_portal_L=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _default_sim(t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _default_sim(dt_h=0.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            _default_sim(solubility_mg_mL=0.0)


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------


class TestGutWallExtractionSensitivity:
    def _run_sweep(self, cl_values=None):
        if cl_values is None:
            cl_values = [0.0, 1.0, 5.0, 10.0, 50.0]
        return gut_wall_extraction_sensitivity(
            drug_name="TestDrug",
            dose_mg=100.0,
            peff_cm_s=1e-4,
            cl_gut_values=cl_values,
            vd_gut=1.0,
            cl_portal=5.0,
            vd_portal=3.0,
        )

    def test_returns_list(self):
        results = self._run_sweep()
        assert isinstance(results, list)

    def test_result_length_matches_input(self):
        cl_values = [0.0, 1.0, 5.0]
        results = self._run_sweep(cl_values=cl_values)
        assert len(results) == 3

    def test_each_result_has_required_keys(self):
        results = self._run_sweep()
        for r in results:
            assert "cl_gut_L_per_h" in r
            assert "fg" in r
            assert "auc_portal" in r

    def test_fg_monotonically_decreasing_with_cl_gut(self):
        cl_values = [0.0, 1.0, 5.0, 20.0, 100.0]
        results = self._run_sweep(cl_values=cl_values)
        fgs = [r["fg"] for r in results]
        for i in range(1, len(fgs)):
            assert fgs[i] <= fgs[i - 1] + 1e-6

    def test_fg_between_0_and_1_for_all_entries(self):
        results = self._run_sweep()
        for r in results:
            assert 0.0 <= r["fg"] <= 1.0

    def test_empty_cl_gut_values_raises(self):
        with pytest.raises(ValueError, match="cl_gut_values"):
            gut_wall_extraction_sensitivity(
                drug_name="D",
                dose_mg=100.0,
                peff_cm_s=1e-4,
                cl_gut_values=[],
                vd_gut=1.0,
                cl_portal=5.0,
                vd_portal=3.0,
            )

    def test_negative_cl_gut_value_raises(self):
        with pytest.raises(ValueError):
            gut_wall_extraction_sensitivity(
                drug_name="D",
                dose_mg=100.0,
                peff_cm_s=1e-4,
                cl_gut_values=[-1.0, 1.0],
                vd_gut=1.0,
                cl_portal=5.0,
                vd_portal=3.0,
            )

    def test_cl_gut_values_preserved_in_output(self):
        cl_values = [0.5, 2.5, 7.0]
        results = self._run_sweep(cl_values=cl_values)
        out_cl = [r["cl_gut_L_per_h"] for r in results]
        assert out_cl == cl_values
