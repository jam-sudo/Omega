"""Tests for Phase 253: prediction/bioequivalence_predictor_p253.py"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.bioequivalence_predictor_p253 import (
    BioequivalenceResult,
    predict_bioequivalence,
    screen_formulations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> BioequivalenceResult:
    params = dict(
        drug_name="TestDrug",
        test_formulation="Generic_A",
        reference_formulation="Innovator",
        auc_test=100.0,
        auc_reference=100.0,
        cmax_test=10.0,
        cmax_reference=10.0,
    )
    params.update(kwargs)
    return predict_bioequivalence(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_bioequivalence_result(self):
        r = _default()
        assert isinstance(r, BioequivalenceResult)

    def test_has_drug_name(self):
        r = _default()
        assert r.drug_name == "TestDrug"

    def test_has_test_formulation(self):
        r = _default()
        assert r.test_formulation == "Generic_A"

    def test_has_reference_formulation(self):
        r = _default()
        assert r.reference_formulation == "Innovator"

    def test_has_be_conclusion_str(self):
        r = _default()
        assert isinstance(r.be_conclusion, str)

    def test_has_notes_str(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_has_bool_fields(self):
        r = _default()
        assert isinstance(r.auc_within_limits, bool)
        assert isinstance(r.cmax_within_limits, bool)


# ---------------------------------------------------------------------------
# Ratio calculation
# ---------------------------------------------------------------------------


class TestRatioCalculation:
    def test_auc_ratio_exact(self):
        r = _default(auc_test=90.0, auc_reference=100.0)
        assert math.isclose(r.auc_ratio, 0.90, rel_tol=1e-9)

    def test_cmax_ratio_exact(self):
        r = _default(cmax_test=12.0, cmax_reference=10.0)
        assert math.isclose(r.cmax_ratio, 1.20, rel_tol=1e-9)

    def test_auc_ratio_unity_for_equal_values(self):
        r = _default(auc_test=50.0, auc_reference=50.0)
        assert math.isclose(r.auc_ratio, 1.0, rel_tol=1e-9)

    def test_cmax_ratio_unity_for_equal_values(self):
        r = _default(cmax_test=5.0, cmax_reference=5.0)
        assert math.isclose(r.cmax_ratio, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# BE conclusion when ratios are near 1.0
# ---------------------------------------------------------------------------


class TestBEConclusion:
    def test_be_when_ratios_near_one(self):
        # Both AUC and Cmax ratio = 1.0 with default CV=15%, n=12 should be BE
        r = _default()
        assert r.be_conclusion == "BE"
        assert r.auc_within_limits is True
        assert r.cmax_within_limits is True

    def test_not_be_when_auc_ratio_0_70(self):
        # AUC ratio = 0.70 is well outside [0.80, 1.25]
        r = _default(auc_test=70.0, auc_reference=100.0)
        assert r.be_conclusion == "Not_BE"
        assert r.auc_within_limits is False

    def test_not_be_when_cmax_ratio_outside_limits(self):
        r = _default(cmax_test=14.0, cmax_reference=10.0)
        assert r.be_conclusion == "Not_BE"
        assert r.cmax_within_limits is False

    def test_be_conclusion_is_be_or_not_be(self):
        r = _default()
        assert r.be_conclusion in ("BE", "Not_BE")

    def test_not_be_when_both_out_of_range(self):
        r = _default(auc_test=60.0, auc_reference=100.0, cmax_test=60.0, cmax_reference=100.0)
        assert r.be_conclusion == "Not_BE"
        assert r.auc_within_limits is False
        assert r.cmax_within_limits is False


# ---------------------------------------------------------------------------
# CI bounds ordering
# ---------------------------------------------------------------------------


class TestCIBounds:
    def test_auc_ci_lower_less_than_upper(self):
        r = _default()
        assert r.auc_ci_lower < r.auc_ci_upper

    def test_cmax_ci_lower_less_than_upper(self):
        r = _default()
        assert r.cmax_ci_lower < r.cmax_ci_upper

    def test_auc_ratio_within_ci(self):
        r = _default(auc_test=95.0, auc_reference=100.0)
        assert r.auc_ci_lower <= r.auc_ratio <= r.auc_ci_upper

    def test_cmax_ratio_within_ci(self):
        r = _default(cmax_test=105.0, cmax_reference=100.0)
        assert r.cmax_ci_lower <= r.cmax_ratio <= r.cmax_ci_upper

    def test_ci_bounds_positive(self):
        r = _default()
        assert r.auc_ci_lower > 0
        assert r.auc_ci_upper > 0
        assert r.cmax_ci_lower > 0
        assert r.cmax_ci_upper > 0

    def test_ci_width_matches_formula(self):
        # For ratio=1.0, cv=15%, n=12:
        # ci_lower = exp(-1.796 * 0.15 / sqrt(12))
        # ci_upper = exp(+1.796 * 0.15 / sqrt(12))
        r = _default(cv_intra_pct=15.0, n_subjects=12)
        expected_lower = math.exp(-1.796 * 0.15 / math.sqrt(12))
        expected_upper = math.exp(+1.796 * 0.15 / math.sqrt(12))
        assert math.isclose(r.auc_ci_lower, expected_lower, rel_tol=1e-6)
        assert math.isclose(r.auc_ci_upper, expected_upper, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# BE limits [0.80, 1.25]
# ---------------------------------------------------------------------------


class TestBELimits:
    def test_boundary_0_80_exactly(self):
        # auc_ratio = 0.80: ci_lower must be >= 0.80 for BE
        # With small CV and large n, ci bounds close to ratio
        r = predict_bioequivalence(
            "Drug",
            "Test",
            "Ref",
            auc_test=80.0,
            auc_reference=100.0,
            cmax_test=100.0,
            cmax_reference=100.0,
            cv_intra_pct=1.0,
            n_subjects=100,
        )
        # ratio=0.80, very narrow CI => ci_lower < 0.80 => not BE
        assert r.auc_within_limits is False

    def test_ratio_1_0_low_cv_large_n_passes(self):
        r = predict_bioequivalence(
            "Drug",
            "Test",
            "Ref",
            auc_test=100.0,
            auc_reference=100.0,
            cmax_test=100.0,
            cmax_reference=100.0,
            cv_intra_pct=5.0,
            n_subjects=100,
        )
        assert r.auc_within_limits is True
        assert r.cmax_within_limits is True


# ---------------------------------------------------------------------------
# Screen formulations
# ---------------------------------------------------------------------------


class TestScreenFormulations:
    def _scenarios(self):
        return [
            {"formulation": "Generic_C", "auc": 70.0, "cmax": 7.0},  # Not_BE
            {"formulation": "Generic_A", "auc": 100.0, "cmax": 10.0},  # BE
            {"formulation": "Generic_B", "auc": 95.0, "cmax": 9.5},  # BE
        ]

    def test_returns_list(self):
        results = screen_formulations("Drug", 100.0, 10.0, self._scenarios())
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        results = screen_formulations("Drug", 100.0, 10.0, self._scenarios())
        assert len(results) == 3

    def test_be_first_in_sorted_order(self):
        results = screen_formulations("Drug", 100.0, 10.0, self._scenarios())
        assert results[0].be_conclusion == "BE"

    def test_not_be_last_in_sorted_order(self):
        results = screen_formulations("Drug", 100.0, 10.0, self._scenarios())
        # Last result should be Not_BE (ratio=0.70)
        assert results[-1].be_conclusion == "Not_BE"

    def test_be_results_sorted_by_auc_ratio_closeness(self):
        results = screen_formulations("Drug", 100.0, 10.0, self._scenarios())
        be_results = [r for r in results if r.be_conclusion == "BE"]
        # Generic_A has auc_ratio=1.0 (closest), Generic_B has 0.95
        assert be_results[0].auc_ratio == 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_auc_test_zero_raises(self):
        with pytest.raises(ValueError, match="auc_test"):
            _default(auc_test=0.0)

    def test_auc_reference_negative_raises(self):
        with pytest.raises(ValueError, match="auc_reference"):
            _default(auc_reference=-10.0)

    def test_cmax_test_zero_raises(self):
        with pytest.raises(ValueError, match="cmax_test"):
            _default(cmax_test=0.0)

    def test_cmax_reference_negative_raises(self):
        with pytest.raises(ValueError, match="cmax_reference"):
            _default(cmax_reference=-5.0)

    def test_cv_zero_raises(self):
        with pytest.raises(ValueError, match="cv_intra_pct"):
            _default(cv_intra_pct=0.0)

    def test_cv_100_raises(self):
        with pytest.raises(ValueError, match="cv_intra_pct"):
            _default(cv_intra_pct=100.0)

    def test_cv_negative_raises(self):
        with pytest.raises(ValueError, match="cv_intra_pct"):
            _default(cv_intra_pct=-5.0)

    def test_n_subjects_below_6_raises(self):
        with pytest.raises(ValueError, match="n_subjects"):
            _default(n_subjects=5)
