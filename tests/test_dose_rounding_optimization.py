"""Tests for Phase 953 — Dose Rounding Optimization."""

import math
import pytest

from omega_pbpk.clinical.dose_rounding_optimization import (
    DoseRoundingResult,
    optimize_dose_rounding,
    compare_dosing_regimens,
)

# Default test parameters
STRENGTHS = [25.0, 50.0, 100.0, 200.0]
DRUG = "TestDrug"


def default_result(**kwargs):
    params = dict(
        drug_name=DRUG,
        ideal_dose_mg=100.0,
        available_strengths_mg=STRENGTHS,
        cl_L_per_h=10.0,
        vd_L=50.0,
        dosing_interval_h=12.0,
        mec_mg_L=0.5,
        mtc_mg_L=5.0,
    )
    params.update(kwargs)
    return optimize_dose_rounding(**params)


class TestReturnType:
    def test_returns_dose_rounding_result(self):
        r = default_result()
        assert isinstance(r, DoseRoundingResult)

    def test_is_frozen(self):
        r = default_result()
        with pytest.raises((AttributeError, TypeError)):
            r.recommended_dose_mg = 999.0  # type: ignore[misc]


class TestBasicFields:
    def test_recommended_dose_positive(self):
        r = default_result()
        assert r.recommended_dose_mg > 0

    def test_recommended_strength_in_available(self):
        r = default_result()
        assert r.recommended_strength_mg in STRENGTHS

    def test_n_units_positive(self):
        r = default_result()
        assert r.n_units > 0

    def test_css_max_positive(self):
        r = default_result()
        assert r.css_max_mg_L > 0

    def test_css_min_positive(self):
        r = default_result()
        assert r.css_min_mg_L > 0

    def test_within_therapeutic_window_is_bool(self):
        r = default_result()
        assert isinstance(r.within_therapeutic_window, bool)

    def test_deviation_non_negative(self):
        r = default_result()
        assert r.deviation_from_ideal_pct >= 0

    def test_alternative_doses_is_tuple(self):
        r = default_result()
        assert isinstance(r.alternative_doses, tuple)

    def test_notes_non_empty(self):
        r = default_result()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_drug_name_preserved(self):
        r = default_result(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"


class TestSpecificBehavior:
    def test_exact_strength_zero_deviation(self):
        """If ideal_dose exactly matches a strength, deviation should be 0."""
        r = optimize_dose_rounding(
            drug_name=DRUG,
            ideal_dose_mg=100.0,
            available_strengths_mg=[100.0],
            cl_L_per_h=10.0,
            vd_L=50.0,
            dosing_interval_h=12.0,
            mec_mg_L=0.001,  # very low MEC to ensure in window
            mtc_mg_L=1000.0,  # very high MTC
        )
        assert abs(r.deviation_from_ideal_pct) < 1e-9

    def test_recommended_dose_equals_strength_times_units(self):
        r = default_result()
        expected = r.recommended_strength_mg * r.n_units
        assert abs(r.recommended_dose_mg - expected) < 1e-6

    def test_css_min_less_than_css_max(self):
        r = default_result()
        assert r.css_min_mg_L < r.css_max_mg_L

    def test_css_min_less_than_css_max_various(self):
        """css_min < css_max for different parameters."""
        for interval in [6.0, 8.0, 12.0, 24.0]:
            r = default_result(dosing_interval_h=interval)
            assert r.css_min_mg_L < r.css_max_mg_L, (
                f"css_min should be < css_max at interval={interval}h"
            )


class TestCompareDosing:
    def test_compare_returns_three_results_default(self):
        results = compare_dosing_regimens(
            drug_name=DRUG,
            ideal_dose_mg=100.0,
            available_strengths_mg=STRENGTHS,
        )
        assert len(results) == 3

    def test_compare_all_dose_rounding_result_instances(self):
        results = compare_dosing_regimens(
            drug_name=DRUG,
            ideal_dose_mg=100.0,
            available_strengths_mg=STRENGTHS,
        )
        for r in results:
            assert isinstance(r, DoseRoundingResult)

    def test_compare_custom_intervals(self):
        results = compare_dosing_regimens(
            drug_name=DRUG,
            ideal_dose_mg=100.0,
            available_strengths_mg=STRENGTHS,
            intervals_h=[6.0, 12.0, 24.0, 48.0],
        )
        assert len(results) == 4

    def test_compare_sorted_in_window_first(self):
        results = compare_dosing_regimens(
            drug_name=DRUG,
            ideal_dose_mg=100.0,
            available_strengths_mg=STRENGTHS,
            mec_mg_L=0.01,
            mtc_mg_L=1000.0,
        )
        # All in window when MEC/MTC are extreme → deviation sorted ascending
        for r in results:
            assert r.within_therapeutic_window is True


class TestValidation:
    def test_ideal_dose_zero_raises(self):
        with pytest.raises(ValueError, match="ideal_dose_mg"):
            optimize_dose_rounding(DRUG, 0.0, STRENGTHS)

    def test_ideal_dose_negative_raises(self):
        with pytest.raises(ValueError, match="ideal_dose_mg"):
            optimize_dose_rounding(DRUG, -50.0, STRENGTHS)

    def test_empty_strengths_raises(self):
        with pytest.raises(ValueError, match="available_strengths_mg"):
            optimize_dose_rounding(DRUG, 100.0, [])

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            optimize_dose_rounding(DRUG, 100.0, [0.0, 50.0])

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, cl_L_per_h=0.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, cl_L_per_h=-1.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, vd_L=0.0)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, vd_L=-10.0)

    def test_mec_gte_mtc_raises(self):
        with pytest.raises(ValueError, match="mec_mg_L"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, mec_mg_L=5.0, mtc_mg_L=5.0)

    def test_mec_gt_mtc_raises(self):
        with pytest.raises(ValueError, match="mec_mg_L"):
            optimize_dose_rounding(DRUG, 100.0, STRENGTHS, mec_mg_L=6.0, mtc_mg_L=5.0)
