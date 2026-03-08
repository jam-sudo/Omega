"""Tests for Phase 493 — Dialysis Drug Removal Kinetics."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.dialysis_removal_kinetics import (
    DialysisRemovalResult,
    compare_dialyzers,
    predict_dialysis_removal,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dialysis_removal_result(self):
        result = predict_dialysis_removal("aspirin", "high_flux", 180.0, 0.5)
        assert isinstance(result, DialysisRemovalResult)

    def test_result_is_frozen(self):
        result = predict_dialysis_removal("aspirin", "high_flux", 180.0, 0.5)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "changed"  # type: ignore[misc]

    def test_fields_present(self):
        result = predict_dialysis_removal("test", "low_flux", 300.0, 0.8)
        assert hasattr(result, "drug_name")
        assert hasattr(result, "sieving_coefficient")
        assert hasattr(result, "cl_dialysis_L_per_h")
        assert hasattr(result, "fraction_removed_per_session")
        assert hasattr(result, "post_dialysis_rebound_pct")
        assert hasattr(result, "supplemental_dose_pct")
        assert hasattr(result, "dialysis_class")
        assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Sieving coefficient formula
# ---------------------------------------------------------------------------


class TestSievingCoefficient:
    def test_sc_formula(self):
        """SC = fu * exp(-MW / MW_cutoff) for high_flux (cutoff=50000)."""
        mw = 300.0
        fu = 0.5
        mw_cutoff = 50_000.0
        expected_sc = fu * math.exp(-mw / mw_cutoff)
        result = predict_dialysis_removal("drug", "high_flux", mw, fu)
        assert abs(result.sieving_coefficient - expected_sc) < 1e-8

    def test_sc_in_range(self):
        result = predict_dialysis_removal("drug", "high_flux", 180.0, 0.9)
        assert 0.0 <= result.sieving_coefficient <= 1.0

    def test_higher_fu_higher_sc(self):
        r_low = predict_dialysis_removal("drug", "high_flux", 300.0, 0.1)
        r_high = predict_dialysis_removal("drug", "high_flux", 300.0, 0.9)
        assert r_high.sieving_coefficient > r_low.sieving_coefficient

    def test_higher_mw_lower_sc(self):
        r_small = predict_dialysis_removal("drug", "high_flux", 200.0, 0.5)
        r_large = predict_dialysis_removal("drug", "high_flux", 5000.0, 0.5)
        assert r_large.sieving_coefficient < r_small.sieving_coefficient

    def test_very_large_mw_near_zero_sc(self):
        result = predict_dialysis_removal("protein", "low_flux", 150_000.0, 1.0)
        assert result.sieving_coefficient < 0.01


# ---------------------------------------------------------------------------
# Dialyzer comparison: high_flux > low_flux
# ---------------------------------------------------------------------------


class TestDialyzerComparison:
    def test_high_flux_removes_more_than_low_flux(self):
        """For same drug, high_flux should remove more due to larger MW_cutoff."""
        r_high = predict_dialysis_removal("drug", "high_flux", 300.0, 0.5)
        r_low = predict_dialysis_removal("drug", "low_flux", 300.0, 0.5)
        assert r_high.fraction_removed_per_session >= r_low.fraction_removed_per_session

    def test_peritoneal_lowest_cl(self):
        """Peritoneal has very low Qd (20 mL/min), so lowest CL for typical MW."""
        r_peritoneal = predict_dialysis_removal("drug", "peritoneal", 200.0, 0.8)
        r_hf = predict_dialysis_removal("drug", "high_flux", 200.0, 0.8)
        assert r_hf.cl_dialysis_L_per_h > r_peritoneal.cl_dialysis_L_per_h


# ---------------------------------------------------------------------------
# Fraction removed in [0, 1]
# ---------------------------------------------------------------------------


class TestFractionRemoved:
    def test_fraction_in_range(self):
        for dt in ("high_flux", "low_flux", "high_efficiency", "peritoneal"):
            r = predict_dialysis_removal("drug", dt, 300.0, 0.5)
            assert 0.0 <= r.fraction_removed_per_session <= 1.0

    def test_supplemental_dose_pct_equals_fraction_times_100(self):
        result = predict_dialysis_removal("drug", "high_flux", 180.0, 0.7)
        assert (
            abs(result.supplemental_dose_pct - result.fraction_removed_per_session * 100.0) < 1e-9
        )

    def test_longer_session_removes_more(self):
        r_short = predict_dialysis_removal("drug", "high_flux", 300.0, 0.8, session_h=2.0)
        r_long = predict_dialysis_removal("drug", "high_flux", 300.0, 0.8, session_h=6.0)
        assert r_long.fraction_removed_per_session > r_short.fraction_removed_per_session


# ---------------------------------------------------------------------------
# Dialysis classification
# ---------------------------------------------------------------------------


class TestDialysisClassification:
    def test_not_dialyzable_low_fu_high_mw(self):
        """Low fu + high MW → SC < 0.1 → not_dialyzable."""
        result = predict_dialysis_removal("drug", "low_flux", 20_000.0, 0.05)
        assert result.dialysis_class == "not_dialyzable"

    def test_well_dialyzable_high_fu_small_mw(self):
        """High fu + small MW → SC > 0.5 → well_dialyzable."""
        result = predict_dialysis_removal("drug", "high_flux", 100.0, 0.99)
        assert result.dialysis_class == "well_dialyzable"

    def test_partially_dialyzable(self):
        """Mid-range SC → partially."""
        result = predict_dialysis_removal("drug", "high_flux", 500.0, 0.6)
        sc = result.sieving_coefficient
        if 0.1 <= sc <= 0.5:
            assert result.dialysis_class == "partially"


# ---------------------------------------------------------------------------
# Rebound
# ---------------------------------------------------------------------------


class TestRebound:
    def test_rebound_20pct_large_vd(self):
        result = predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, vd_L=100.0)
        assert result.post_dialysis_rebound_pct == 20.0

    def test_no_rebound_small_vd(self):
        result = predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, vd_L=30.0)
        assert result.post_dialysis_rebound_pct == 0.0

    def test_vd_exactly_50_no_rebound(self):
        """vd_L == 50.0 should NOT trigger rebound (strictly > 50)."""
        result = predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, vd_L=50.0)
        assert result.post_dialysis_rebound_pct == 0.0


# ---------------------------------------------------------------------------
# compare_dialyzers
# ---------------------------------------------------------------------------


class TestCompareDialyzers:
    def test_returns_list_of_4(self):
        results = compare_dialyzers("drug", 300.0, 0.5)
        assert len(results) == 4

    def test_sorted_descending(self):
        results = compare_dialyzers("drug", 300.0, 0.5)
        fracs = [r.fraction_removed_per_session for r in results]
        assert fracs == sorted(fracs, reverse=True)

    def test_all_are_dialysis_removal_result(self):
        results = compare_dialyzers("drug", 300.0, 0.5)
        for r in results:
            assert isinstance(r, DialysisRemovalResult)

    def test_same_drug_name_in_all(self):
        results = compare_dialyzers("testdrug", 500.0, 0.7)
        assert all(r.drug_name == "testdrug" for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dialyzer_type(self):
        with pytest.raises(ValueError, match="dialyzer_type"):
            predict_dialysis_removal("drug", "super_flux", 300.0, 0.5)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_dialysis_removal("drug", "high_flux", 0.0, 0.5)

    def test_negative_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_dialysis_removal("drug", "high_flux", -100.0, 0.5)

    def test_zero_fu(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_dialysis_removal("drug", "high_flux", 300.0, 0.0)

    def test_fu_greater_than_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_dialysis_removal("drug", "high_flux", 300.0, 1.5)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, vd_L=0.0)

    def test_zero_session_h(self):
        with pytest.raises(ValueError, match="session_h"):
            predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, session_h=0.0)

    def test_negative_session_h(self):
        with pytest.raises(ValueError, match="session_h"):
            predict_dialysis_removal("drug", "high_flux", 300.0, 0.5, session_h=-1.0)
