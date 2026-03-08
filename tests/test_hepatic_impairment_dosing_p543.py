"""Tests for Phase 543: Hepatic Impairment Drug Dosing."""

import math

import pytest

from omega_pbpk.clinical.hepatic_impairment_dosing_p543 import (
    HepaticImpairmentDosingResult,
    adjust_dose_hepatic_impairment,
    compare_severity_levels,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

DRUG = "TestDrug"
DOSE = 100.0  # mg
CL = 20.0  # L/h
VD = 200.0  # L
FU = 0.1


def _call(severity, **kw):
    defaults = dict(
        drug_name=DRUG,
        severity=severity,
        baseline_dose_mg=DOSE,
        baseline_cl_L_per_h=CL,
        baseline_vd_L=VD,
        baseline_fu=FU,
    )
    defaults.update(kw)
    return adjust_dose_hepatic_impairment(**defaults)


# ── Return type & fields ──────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_dataclass(self):
        r = _call("child_pugh_a")
        assert isinstance(r, HepaticImpairmentDosingResult)

    def test_has_drug_name(self):
        r = _call("child_pugh_a")
        assert r.drug_name == DRUG

    def test_has_severity(self):
        r = _call("child_pugh_b")
        assert r.severity == "child_pugh_b"

    def test_has_all_fields(self):
        r = _call("child_pugh_c")
        for field in [
            "drug_name",
            "severity",
            "baseline_dose_mg",
            "baseline_cl_L_per_h",
            "baseline_fu",
            "cl_impaired_L_per_h",
            "fu_impaired",
            "vd_impaired_L",
            "cl_ratio",
            "fu_ratio",
            "adjusted_dose_mg",
            "dose_adjustment_pct",
            "css_avg_baseline_mg_L",
            "css_avg_impaired_mg_L",
            "css_ratio",
            "monitoring_required",
            "contraindicated",
            "notes",
        ]:
            assert hasattr(r, field)

    def test_is_frozen(self):
        r = _call("child_pugh_a")
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore


# ── Child-Pugh A ──────────────────────────────────────────────────────────────


class TestChildPughA:
    def test_cl_ratio(self):
        r = _call("child_pugh_a")
        assert math.isclose(r.cl_ratio, 0.75, rel_tol=1e-6)

    def test_cl_impaired(self):
        r = _call("child_pugh_a")
        assert math.isclose(r.cl_impaired_L_per_h, CL * 0.75, rel_tol=1e-6)

    def test_fu_ratio(self):
        r = _call("child_pugh_a")
        assert math.isclose(r.fu_ratio, 1.2, rel_tol=1e-6)

    def test_vd_impaired(self):
        r = _call("child_pugh_a")
        assert math.isclose(r.vd_impaired_L, VD * 1.1, rel_tol=1e-6)

    def test_monitoring_not_required(self):
        # cl_ratio = 0.75 >= 0.5 → monitoring not required
        r = _call("child_pugh_a")
        assert r.monitoring_required is False


# ── Child-Pugh B ──────────────────────────────────────────────────────────────


class TestChildPughB:
    def test_cl_ratio(self):
        r = _call("child_pugh_b")
        assert math.isclose(r.cl_ratio, 0.5, rel_tol=1e-6)

    def test_monitoring_required(self):
        # cl_ratio = 0.5 is NOT < 0.5 strictly, so monitoring_required = False at exactly 0.5
        # But per spec: cl_ratio < 0.5 → monitoring required
        # For B cl_ratio == 0.5, monitoring_required = False (boundary)
        r = _call("child_pugh_b")
        assert r.monitoring_required is False  # 0.5 is NOT strictly < 0.5

    def test_adjusted_dose_less_than_baseline(self):
        r = _call("child_pugh_b")
        assert r.adjusted_dose_mg < DOSE


# ── Child-Pugh C ──────────────────────────────────────────────────────────────


class TestChildPughC:
    def test_cl_ratio(self):
        r = _call("child_pugh_c")
        assert math.isclose(r.cl_ratio, 0.25, rel_tol=1e-6)

    def test_cl_impaired(self):
        r = _call("child_pugh_c")
        assert math.isclose(r.cl_impaired_L_per_h, CL * 0.25, rel_tol=1e-6)

    def test_monitoring_required(self):
        # cl_ratio = 0.25 < 0.5
        r = _call("child_pugh_c")
        assert r.monitoring_required is True

    def test_adjusted_dose_reduced(self):
        r = _call("child_pugh_c")
        assert r.adjusted_dose_mg < DOSE

    def test_adjusted_dose_clamped_floor(self):
        # Even at clamp floor it's still >= 0.1 * baseline
        r = _call("child_pugh_c")
        assert r.adjusted_dose_mg >= 0.1 * DOSE

    def test_adjusted_dose_clamped_ceiling(self):
        r = _call("child_pugh_c")
        assert r.adjusted_dose_mg <= 2.0 * DOSE

    def test_fu_multiplier(self):
        r = _call("child_pugh_c")
        assert math.isclose(r.fu_ratio, 1.7, rel_tol=1e-6)

    def test_vd_multiplier(self):
        r = _call("child_pugh_c")
        assert math.isclose(r.vd_impaired_L, VD * 1.6, rel_tol=1e-6)


# ── CSS ratio ─────────────────────────────────────────────────────────────────


class TestCssRatio:
    def test_css_ratio_near_one(self):
        """Adjusted dose should restore Css_avg ≈ baseline (within clamping limits)."""
        r = _call("child_pugh_a")
        # Should be close to 1.0 when not clamped
        assert 0.5 < r.css_ratio < 2.0

    def test_css_avg_baseline_positive(self):
        r = _call("child_pugh_b")
        assert r.css_avg_baseline_mg_L > 0.0

    def test_css_avg_impaired_positive(self):
        r = _call("child_pugh_b")
        assert r.css_avg_impaired_mg_L > 0.0


# ── Contraindicated ───────────────────────────────────────────────────────────


class TestContraindicated:
    def test_contraindicated_high_cl_severe(self):
        # cl_ratio < 0.3 (Child-Pugh C) AND cl > 10 → contraindicated
        r = adjust_dose_hepatic_impairment(
            drug_name="HighCLDrug",
            severity="child_pugh_c",
            baseline_dose_mg=100.0,
            baseline_cl_L_per_h=15.0,  # > 10
            baseline_vd_L=100.0,
        )
        assert r.contraindicated is True

    def test_not_contraindicated_low_cl(self):
        # cl_ratio < 0.3 (Child-Pugh C) but cl <= 10 → not contraindicated
        r = adjust_dose_hepatic_impairment(
            drug_name="LowCLDrug",
            severity="child_pugh_c",
            baseline_dose_mg=100.0,
            baseline_cl_L_per_h=5.0,  # <= 10
            baseline_vd_L=50.0,
        )
        assert r.contraindicated is False

    def test_not_contraindicated_child_pugh_a(self):
        r = _call("child_pugh_a")
        assert r.contraindicated is False

    def test_not_contraindicated_child_pugh_b(self):
        r = _call("child_pugh_b")
        assert r.contraindicated is False


# ── Dose adjustment percentage ────────────────────────────────────────────────


class TestDoseAdjustmentPct:
    def test_dose_adjustment_pct_negative_for_c(self):
        r = _call("child_pugh_c")
        assert r.dose_adjustment_pct < 0.0

    def test_dose_adjustment_pct_formula(self):
        r = _call("child_pugh_a")
        expected_pct = (r.adjusted_dose_mg / DOSE - 1.0) * 100.0
        assert math.isclose(r.dose_adjustment_pct, expected_pct, rel_tol=1e-6)


# ── compare_severity_levels ───────────────────────────────────────────────────


class TestCompareSeverityLevels:
    def test_returns_list_of_3(self):
        results = compare_severity_levels(DRUG, DOSE, CL, VD)
        assert len(results) == 3

    def test_all_are_result_objects(self):
        results = compare_severity_levels(DRUG, DOSE, CL, VD)
        for r in results:
            assert isinstance(r, HepaticImpairmentDosingResult)

    def test_sorted_by_cl_ratio_ascending(self):
        results = compare_severity_levels(DRUG, DOSE, CL, VD)
        ratios = [r.cl_ratio for r in results]
        assert ratios == sorted(ratios)

    def test_first_is_most_severe(self):
        results = compare_severity_levels(DRUG, DOSE, CL, VD)
        assert results[0].severity == "child_pugh_c"

    def test_last_is_mildest(self):
        results = compare_severity_levels(DRUG, DOSE, CL, VD)
        assert results[-1].severity == "child_pugh_a"


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_severity(self):
        with pytest.raises(ValueError, match="severity"):
            _call("child_pugh_d")

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="baseline_dose_mg"):
            _call("child_pugh_a", baseline_dose_mg=0.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="baseline_dose_mg"):
            _call("child_pugh_a", baseline_dose_mg=-10.0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="baseline_cl_L_per_h"):
            _call("child_pugh_a", baseline_cl_L_per_h=0.0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="baseline_vd_L"):
            _call("child_pugh_a", baseline_vd_L=0.0)

    def test_zero_fu(self):
        with pytest.raises(ValueError, match="baseline_fu"):
            _call("child_pugh_a", baseline_fu=0.0)

    def test_fu_greater_than_one(self):
        with pytest.raises(ValueError, match="baseline_fu"):
            _call("child_pugh_a", baseline_fu=1.1)
