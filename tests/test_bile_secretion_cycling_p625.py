"""Tests for Phase 625 — Drug Bile Secretion and Enterohepatic Cycling."""

import math

import pytest

from omega_pbpk.core.bile_secretion_cycling_p625 import (
    BileSecretionCyclingResult,
    simulate_bile_secretion_cycling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    """Run simulation with sensible defaults, overriding via kwargs."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_biliary_L_per_h=2.0,
        cl_sys_L_per_h=10.0,
        vd_L=50.0,
        f_reabsorb=0.6,
        bile_flow_mL_per_h=30.0,
        meal_times_h=None,
        t_end_h=48.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_bile_secretion_cycling(**params)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        res = _default_result()
        assert isinstance(res, BileSecretionCyclingResult)

    def test_drug_name_preserved(self):
        res = _default_result(drug_name="Morphine")
        assert res.drug_name == "Morphine"

    def test_dose_preserved(self):
        res = _default_result(dose_mg=200.0)
        assert res.dose_mg == 200.0

    def test_list_fields_are_lists(self):
        res = _default_result()
        for field in [
            res.times_h,
            res.c_plasma_mg_L,
            res.c_bile_mg_mL,
            res.c_duodenum_mg_mL,
            res.cumulative_secreted_mg,
            res.cumulative_reabsorbed_mg,
            res.secondary_peak_times_h,
        ]:
            assert isinstance(field, list)

    def test_list_lengths_consistent(self):
        res = _default_result()
        n = len(res.times_h)
        assert len(res.c_plasma_mg_L) == n
        assert len(res.c_bile_mg_mL) == n
        assert len(res.c_duodenum_mg_mL) == n
        assert len(res.cumulative_secreted_mg) == n
        assert len(res.cumulative_reabsorbed_mg) == n

    def test_times_start_at_zero(self):
        res = _default_result()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_times_end_near_t_end(self):
        res = _default_result(t_end_h=24.0, dt_h=0.1)
        assert res.times_h[-1] == pytest.approx(24.0, abs=0.15)

    def test_notes_is_string(self):
        res = _default_result()
        assert isinstance(res.notes, str)


# ---------------------------------------------------------------------------
# Plasma PK correctness
# ---------------------------------------------------------------------------


class TestPlasmaPK:
    def test_plasma_starts_at_zero(self):
        res = _default_result()
        assert res.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-6)

    def test_plasma_rises_then_falls(self):
        res = _default_result(meal_times_h=[6.0, 12.0, 18.0, 24.0])
        cmax = max(res.c_plasma_mg_L)
        assert cmax > 0.0
        # Early concentration should be less than cmax
        assert res.c_plasma_mg_L[0] < cmax

    def test_plasma_non_negative(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_plasma_mg_L)

    def test_auc_positive(self):
        res = _default_result()
        assert res.auc_mg_h_per_L > 0.0

    def test_higher_dose_higher_auc(self):
        res1 = _default_result(dose_mg=100.0)
        res2 = _default_result(dose_mg=200.0)
        assert res2.auc_mg_h_per_L > res1.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Bile secretion
# ---------------------------------------------------------------------------


class TestBileSecretion:
    def test_bile_starts_at_zero(self):
        res = _default_result()
        assert res.c_bile_mg_mL[0] == pytest.approx(0.0, abs=1e-9)

    def test_bile_non_negative(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_bile_mg_mL)

    def test_cumulative_secreted_increases(self):
        res = _default_result()
        # Cumulative should be monotonically non-decreasing
        for i in range(1, len(res.cumulative_secreted_mg)):
            assert res.cumulative_secreted_mg[i] >= res.cumulative_secreted_mg[i - 1] - 1e-9

    def test_zero_biliary_cl_no_secretion(self):
        res = _default_result(cl_biliary_L_per_h=0.0)
        # No biliary secretion → bile stays zero
        assert res.cumulative_secreted_mg[-1] == pytest.approx(0.0, abs=1e-9)

    def test_cumulative_secreted_bounded_by_dose(self):
        res = _default_result()
        assert res.cumulative_secreted_mg[-1] <= res.dose_mg * 1.05


# ---------------------------------------------------------------------------
# Enterohepatic cycling
# ---------------------------------------------------------------------------


class TestEHC:
    def test_ehc_fraction_in_range(self):
        res = _default_result()
        assert 0.0 <= res.ehc_fraction <= 1.0

    def test_higher_biliary_cl_higher_ehc(self):
        res_lo = _default_result(cl_biliary_L_per_h=1.0, f_reabsorb=0.8)
        res_hi = _default_result(cl_biliary_L_per_h=5.0, f_reabsorb=0.8)
        assert res_hi.ehc_fraction >= res_lo.ehc_fraction

    def test_zero_reabsorb_no_ehc_fraction(self):
        res = _default_result(f_reabsorb=0.0)
        assert res.ehc_fraction == pytest.approx(0.0, abs=1e-6)

    def test_secondary_peaks_list(self):
        res = _default_result()
        assert isinstance(res.secondary_peak_times_h, list)

    def test_cycling_events_count_matches_list(self):
        res = _default_result()
        assert res.n_cycling_events == len(res.secondary_peak_times_h)

    def test_cycling_events_detected_with_high_biliary_cl(self):
        """High biliary CL + meals should produce at least one secondary peak."""
        res = simulate_bile_secretion_cycling(
            drug_name="HighEHC",
            dose_mg=100.0,
            cl_biliary_L_per_h=8.0,
            cl_sys_L_per_h=5.0,
            vd_L=50.0,
            f_reabsorb=0.9,
            meal_times_h=[4.0, 10.0, 16.0, 22.0, 28.0],
            t_end_h=48.0,
            dt_h=0.05,
        )
        # At least one secondary cycling event expected
        assert res.n_cycling_events >= 1

    def test_no_cycling_without_biliary_cl(self):
        res = simulate_bile_secretion_cycling(
            drug_name="NoBile",
            dose_mg=100.0,
            cl_biliary_L_per_h=0.0,
            cl_sys_L_per_h=10.0,
            vd_L=50.0,
            f_reabsorb=0.6,
            t_end_h=48.0,
            dt_h=0.1,
        )
        assert res.n_cycling_events == 0

    def test_cumulative_reabsorbed_increases(self):
        res = _default_result()
        for i in range(1, len(res.cumulative_reabsorbed_mg)):
            assert res.cumulative_reabsorbed_mg[i] >= res.cumulative_reabsorbed_mg[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# Effective half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_positive(self):
        res = _default_result()
        assert res.t_half_effective_h > 0.0

    def test_t_half_increases_with_reabsorption(self):
        """Higher f_reabsorb → less net elimination → longer effective t½."""
        res_lo = _default_result(f_reabsorb=0.0, cl_biliary_L_per_h=3.0)
        res_hi = _default_result(f_reabsorb=0.9, cl_biliary_L_per_h=3.0)
        assert res_hi.t_half_effective_h > res_lo.t_half_effective_h

    def test_t_half_analytical_formula(self):
        """Verify analytical formula: ln(2)*vd / (cl_sys + cl_biliary*(1-f))."""
        vd = 50.0
        cl_sys = 10.0
        cl_bil = 3.0
        f = 0.6
        expected = math.log(2.0) * vd / (cl_sys + cl_bil * (1.0 - f))
        res = _default_result(
            cl_sys_L_per_h=cl_sys, cl_biliary_L_per_h=cl_bil, vd_L=vd, f_reabsorb=f
        )
        assert res.t_half_effective_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Meal time effects
# ---------------------------------------------------------------------------


class TestMealTime:
    def test_custom_meal_times(self):
        """Custom meal times should run without error."""
        res = simulate_bile_secretion_cycling(
            drug_name="Rifampin",
            dose_mg=300.0,
            cl_biliary_L_per_h=4.0,
            cl_sys_L_per_h=8.0,
            vd_L=80.0,
            meal_times_h=[2.0, 8.0, 14.0],
            t_end_h=24.0,
            dt_h=0.1,
        )
        assert isinstance(res, BileSecretionCyclingResult)

    def test_duodenum_concentration_non_negative(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_duodenum_mg_mL)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=0.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-10.0)

    def test_negative_biliary_cl(self):
        with pytest.raises(ValueError, match="cl_biliary"):
            _default_result(cl_biliary_L_per_h=-1.0)

    def test_zero_sys_cl(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _default_result(cl_sys_L_per_h=0.0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_result(vd_L=0.0)

    def test_f_reabsorb_above_one(self):
        with pytest.raises(ValueError, match="f_reabsorb"):
            _default_result(f_reabsorb=1.5)

    def test_f_reabsorb_negative(self):
        with pytest.raises(ValueError, match="f_reabsorb"):
            _default_result(f_reabsorb=-0.1)

    def test_empty_meal_times(self):
        with pytest.raises(ValueError, match="meal_times_h"):
            _default_result(meal_times_h=[])
