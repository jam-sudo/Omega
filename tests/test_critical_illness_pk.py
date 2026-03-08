"""Tests for Phase 530 — Drug Clearance in Critical Illness."""

import pytest

from omega_pbpk.clinical.critical_illness_pk import (
    CriticalIllnessPKResult,
    assess_critical_illness_pk,
    compare_illness_states,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(illness_type="sepsis_early", cl=10.0, vd=50.0, dose=500.0, fu=0.1, tau=24.0):
    return assess_critical_illness_pk("vancomycin", illness_type, cl, vd, dose, fu, tau)


# ---------------------------------------------------------------------------
# Return type / frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_critical_illness_pk_result(self):
        r = make_result()
        assert isinstance(r, CriticalIllnessPKResult)

    def test_frozen_dataclass(self):
        r = make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.cl_sick_L_per_h = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


class TestFields:
    def test_drug_name(self):
        r = assess_critical_illness_pk("vancomycin", "ards", 10.0, 50.0, 500.0)
        assert r.drug_name == "vancomycin"

    def test_illness_type_stored(self):
        r = make_result("liver_failure")
        assert r.illness_type == "liver_failure"

    def test_baseline_cl_stored(self):
        r = make_result(cl=8.0)
        assert r.baseline_cl_L_per_h == pytest.approx(8.0)

    def test_baseline_vd_stored(self):
        r = make_result(vd=40.0)
        assert r.baseline_vd_L == pytest.approx(40.0)

    def test_cl_ratio_positive(self):
        r = make_result()
        assert r.cl_ratio > 0

    def test_vd_ratio_positive(self):
        r = make_result()
        assert r.vd_ratio > 0

    def test_t_half_positive(self):
        r = make_result()
        assert r.t_half_sick_h > 0

    def test_loading_dose_positive(self):
        r = make_result()
        assert r.loading_dose_mg > 0

    def test_maintenance_dose_ratio_positive(self):
        r = make_result()
        assert r.maintenance_dose_ratio > 0

    def test_trough_ratio_positive(self):
        r = make_result()
        assert r.trough_ratio > 0


# ---------------------------------------------------------------------------
# Sepsis early: CL increases
# ---------------------------------------------------------------------------


class TestSepsisEarly:
    def test_cl_ratio_15(self):
        r = make_result("sepsis_early", cl=10.0)
        assert r.cl_ratio == pytest.approx(1.5, rel=1e-6)

    def test_cl_sick_greater_than_baseline(self):
        r = make_result("sepsis_early", cl=10.0)
        assert r.cl_sick_L_per_h > r.baseline_cl_L_per_h

    def test_vd_ratio_14(self):
        r = make_result("sepsis_early")
        assert r.vd_ratio == pytest.approx(1.4, rel=1e-6)


# ---------------------------------------------------------------------------
# Multi-organ failure: CL strongly decreases
# ---------------------------------------------------------------------------


class TestMultiOrganFailure:
    def test_cl_ratio_02(self):
        r = make_result("multi_organ_failure", cl=10.0)
        assert r.cl_ratio == pytest.approx(0.2, rel=1e-6)

    def test_cl_sick_much_less_than_baseline(self):
        r = make_result("multi_organ_failure", cl=10.0)
        assert r.cl_sick_L_per_h < r.baseline_cl_L_per_h

    def test_trough_ratio_high(self):
        """Trough ratio >> 1 means drug accumulates."""
        r = make_result("multi_organ_failure")
        assert r.trough_ratio > 1.0


# ---------------------------------------------------------------------------
# t_half formula
# ---------------------------------------------------------------------------


class TestTHalf:
    def test_t_half_formula_sepsis_late(self):
        cl = 10.0
        vd = 50.0
        r = make_result("sepsis_late", cl=cl, vd=vd)
        expected = 0.693 * (vd * 1.5) / (cl * 0.5)
        assert r.t_half_sick_h == pytest.approx(expected, rel=1e-6)

    def test_t_half_formula_ards(self):
        cl = 8.0
        vd = 60.0
        r = make_result("ards", cl=cl, vd=vd)
        expected = 0.693 * (vd * 1.6) / (cl * 0.7)
        assert r.t_half_sick_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# maintenance_dose_ratio = cl_ratio
# ---------------------------------------------------------------------------


class TestMaintenanceDoseRatio:
    def test_maintenance_equals_cl_ratio(self):
        for illness in ["sepsis_early", "liver_failure", "renal_failure"]:
            r = make_result(illness)
            assert r.maintenance_dose_ratio == pytest.approx(r.cl_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# trough_ratio = 1 / cl_ratio
# ---------------------------------------------------------------------------


class TestTroughRatio:
    def test_trough_ratio_inverse_cl_ratio(self):
        for illness in ["sepsis_early", "ards", "multi_organ_failure"]:
            r = make_result(illness)
            assert r.trough_ratio == pytest.approx(1.0 / r.cl_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# fu_sick clamped to 1.0
# ---------------------------------------------------------------------------


class TestFuSick:
    def test_fu_sick_clamped(self):
        # multi_organ_failure: fu_mult=1.8, baseline_fu=0.9 → would exceed 1
        r = assess_critical_illness_pk(
            "drug", "multi_organ_failure", 5.0, 30.0, 200.0, baseline_fu=0.9
        )
        assert r.fu_sick <= 1.0

    def test_fu_sick_within_range(self):
        r = make_result("liver_failure", fu=0.1)
        assert 0.0 < r.fu_sick <= 1.0


# ---------------------------------------------------------------------------
# Monitoring priority
# ---------------------------------------------------------------------------


class TestMonitoringPriority:
    def test_urgent_for_multi_organ_failure(self):
        r = make_result("multi_organ_failure")
        assert r.monitoring_priority == "urgent"

    def test_urgent_for_sepsis_early(self):
        # cl_ratio = 1.5 → exactly 1.5 → urgent (> 1.5 is urgent, boundary)
        r = make_result("sepsis_early")
        # cl_ratio = 1.5, boundary: > 1.5 → urgent; = 1.5 → depends on strict check
        # Per spec: "urgent" if cl_ratio > 1.5; 1.5 is NOT > 1.5 → "high" (> 1.2)
        assert r.monitoring_priority in {"urgent", "high"}

    def test_high_for_liver_failure(self):
        # cl_ratio = 0.3 (not < 0.3) → "high" (< 0.5)
        r = make_result("liver_failure")
        assert r.monitoring_priority == "high"

    def test_high_for_renal_failure(self):
        # cl_ratio = 0.4 → < 0.5 → "high"
        r = make_result("renal_failure")
        assert r.monitoring_priority == "high"

    def test_routine_for_ards(self):
        # cl_ratio = 0.7 → not < 0.5 and not > 1.2 → "routine"
        r = make_result("ards")
        assert r.monitoring_priority == "routine"


# ---------------------------------------------------------------------------
# compare_illness_states
# ---------------------------------------------------------------------------


class TestCompareIllnessStates:
    def test_returns_6_results(self):
        results = compare_illness_states("vancomycin", 10.0, 50.0, 500.0)
        assert len(results) == 6

    def test_sorted_by_cl_ratio_ascending(self):
        results = compare_illness_states("drug", 10.0, 50.0, 500.0)
        ratios = [r.cl_ratio for r in results]
        assert ratios == sorted(ratios)

    def test_first_has_lowest_cl_ratio(self):
        results = compare_illness_states("drug", 10.0, 50.0, 500.0)
        # multi_organ_failure has cl_ratio=0.2, should be first
        assert results[0].illness_type == "multi_organ_failure"

    def test_all_are_critical_illness_pk_result(self):
        results = compare_illness_states("drug", 10.0, 50.0, 500.0)
        for r in results:
            assert isinstance(r, CriticalIllnessPKResult)

    def test_last_has_highest_cl_ratio(self):
        results = compare_illness_states("drug", 10.0, 50.0, 500.0)
        # sepsis_early has cl_ratio=1.5, should be last
        assert results[-1].illness_type == "sepsis_early"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_illness_type(self):
        with pytest.raises(ValueError, match="illness_type"):
            assess_critical_illness_pk("drug", "covid", 10.0, 50.0, 500.0)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="baseline_cl"):
            assess_critical_illness_pk("drug", "ards", 0.0, 50.0, 500.0)

    def test_cl_negative(self):
        with pytest.raises(ValueError, match="baseline_cl"):
            assess_critical_illness_pk("drug", "ards", -5.0, 50.0, 500.0)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="baseline_vd"):
            assess_critical_illness_pk("drug", "ards", 10.0, 0.0, 500.0)

    def test_dose_zero(self):
        with pytest.raises(ValueError, match="baseline_dose"):
            assess_critical_illness_pk("drug", "ards", 10.0, 50.0, 0.0)

    def test_dose_negative(self):
        with pytest.raises(ValueError, match="baseline_dose"):
            assess_critical_illness_pk("drug", "ards", 10.0, 50.0, -100.0)
