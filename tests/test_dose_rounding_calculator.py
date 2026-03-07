"""Tests for Phase 206 — prediction/dose_rounding_calculator.py."""

import pytest

from omega_pbpk.prediction.dose_rounding_calculator import (
    DoseRoundingResult,
    round_to_available_dose,
    screen_dose_adjustments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRENGTHS_50_100 = (50.0, 100.0)
STRENGTHS_25_50_100 = (25.0, 50.0, 100.0)


def _round(dose: float, strengths=STRENGTHS_50_100, **kw) -> DoseRoundingResult:
    return round_to_available_dose(
        drug_name="TestDrug",
        calculated_dose_mg=dose,
        available_strengths_mg=strengths,
        **kw,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dose_rounding_result(self):
        r = _round(100.0)
        assert isinstance(r, DoseRoundingResult)

    def test_drug_name_preserved(self):
        r = round_to_available_dose("Metformin", 500.0, STRENGTHS_25_50_100)
        assert r.drug_name == "Metformin"

    def test_calculated_dose_preserved(self):
        r = _round(150.0)
        assert r.calculated_dose_mg == pytest.approx(150.0)

    def test_available_strengths_preserved(self):
        r = _round(100.0, strengths=STRENGTHS_25_50_100)
        assert set(r.available_strengths_mg) == {25.0, 50.0, 100.0}

    def test_notes_is_string(self):
        r = _round(100.0)
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Rounded dose equals sum of tablet_combination
# ---------------------------------------------------------------------------


class TestRoundedDoseConsistency:
    def test_rounded_dose_equals_sum_of_combination(self):
        r = _round(150.0)
        assert r.rounded_dose_mg == pytest.approx(sum(r.tablet_combination), rel=1e-9)

    def test_n_tablets_equals_len_combination(self):
        r = _round(150.0)
        assert r.n_tablets == len(r.tablet_combination)

    def test_exact_match_zero_error(self):
        r = _round(100.0)
        assert r.dose_error_pct == pytest.approx(0.0, abs=1e-9)
        assert r.rounded_dose_mg == pytest.approx(100.0)

    def test_exact_match_one_tablet(self):
        r = _round(50.0, strengths=STRENGTHS_50_100)
        assert r.n_tablets == 1
        assert r.rounded_dose_mg == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Acceptable flag
# ---------------------------------------------------------------------------


class TestAcceptableFlag:
    def test_zero_error_is_acceptable(self):
        r = _round(100.0)
        assert r.acceptable is True

    def test_error_within_20pct_acceptable(self):
        # 100 mg calculated, nearest is 100 mg → 0% error
        r = _round(100.0)
        assert r.acceptable is True

    def test_large_error_not_acceptable(self):
        # Only 100 mg available, calculated = 1.0 mg → huge error
        r = round_to_available_dose("Drug", 1.0, (100.0,), max_tablets=1, max_error_pct=20.0)
        assert r.acceptable is False

    def test_error_at_exactly_20pct_acceptable(self):
        # 100 mg calculated, only 120 mg tablet available → error = 20%
        r = round_to_available_dose("Drug", 100.0, (120.0,), max_tablets=1, max_error_pct=20.0)
        assert r.rounded_dose_mg == pytest.approx(120.0)
        assert r.dose_error_pct == pytest.approx(20.0, rel=1e-6)
        assert r.acceptable is True

    def test_error_just_above_20pct_not_acceptable(self):
        # 100 mg calc, only 130 mg reachable
        r = round_to_available_dose("Drug", 100.0, (130.0,), max_tablets=1, max_error_pct=20.0)
        assert r.acceptable is False


# ---------------------------------------------------------------------------
# Tablet preference (fewer is better)
# ---------------------------------------------------------------------------


class TestTabletPreference:
    def test_single_tablet_preferred_over_two(self):
        # 100 mg is available as single tablet; also reachable as 50+50
        r = _round(100.0, strengths=STRENGTHS_50_100)
        assert r.n_tablets == 1

    def test_fewest_tablets_selected(self):
        # 150 mg with strengths (50, 100): fewest tablets = 1 (either 50 or 100)
        # Among 1-tablet options: 100 mg has 33% error, 50 mg has 67% error
        # Algorithm picks 1 tablet (fewest), lowest |error%| → 100 mg
        r = _round(150.0, strengths=STRENGTHS_50_100, max_tablets=4)
        assert r.n_tablets == 1  # fewest tablets is priority
        assert r.rounded_dose_mg == pytest.approx(100.0)  # closest among 1-tablet


# ---------------------------------------------------------------------------
# screen_dose_adjustments
# ---------------------------------------------------------------------------


class TestScreenDoseAdjustments:
    def test_returns_correct_count(self):
        weights = [40.0, 50.0, 60.0, 70.0, 80.0]
        results = screen_dose_adjustments("Drug", 70.0, 2.0, STRENGTHS_25_50_100, weights)
        assert len(results) == len(weights)

    def test_each_result_is_dose_rounding_result(self):
        weights = [50.0, 70.0]
        results = screen_dose_adjustments("Drug", 70.0, 5.0, STRENGTHS_25_50_100, weights)
        for r in results:
            assert isinstance(r, DoseRoundingResult)

    def test_doses_match_weight_times_mg_per_kg(self):
        weights = [60.0, 80.0]
        mg_per_kg = 3.0
        results = screen_dose_adjustments("Drug", 70.0, mg_per_kg, STRENGTHS_25_50_100, weights)
        for r, w in zip(results, weights):
            assert r.calculated_dose_mg == pytest.approx(w * mg_per_kg)

    def test_empty_weight_range_returns_empty_list(self):
        results = screen_dose_adjustments("Drug", 70.0, 5.0, STRENGTHS_25_50_100, [])
        assert results == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_calculated_dose_zero_raises(self):
        with pytest.raises(ValueError, match="calculated_dose_mg"):
            round_to_available_dose("Drug", 0.0, (100.0,))

    def test_calculated_dose_negative_raises(self):
        with pytest.raises(ValueError, match="calculated_dose_mg"):
            round_to_available_dose("Drug", -50.0, (100.0,))

    def test_empty_available_strengths_raises(self):
        with pytest.raises(ValueError, match="available_strengths_mg"):
            round_to_available_dose("Drug", 100.0, ())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_strength_available(self):
        r = round_to_available_dose("Drug", 100.0, (100.0,))
        assert r.rounded_dose_mg == pytest.approx(100.0)
        assert r.dose_error_pct == pytest.approx(0.0, abs=1e-9)

    def test_combination_tuple_is_sorted_or_consistent(self):
        r = _round(150.0)
        # tablet_combination should sum to rounded_dose
        assert sum(r.tablet_combination) == pytest.approx(r.rounded_dose_mg, rel=1e-9)

    def test_three_strengths_combination(self):
        # 175 mg with (25, 50, 100): fewest tablets preferred → 1 tablet of 100 mg
        # (closer to 175 than 25 or 50); if we allow more tablets, still fewest first
        strengths = (25.0, 50.0, 100.0)
        r = round_to_available_dose("Drug", 175.0, strengths, max_tablets=4)
        # 1-tablet option closest: 100 mg (error -42.8%), 50 mg (error -71.4%), 25 mg (-85.7%)
        # Algorithm returns fewest tablets with lowest |error%| = 100 mg
        assert r.n_tablets == 1
        assert r.rounded_dose_mg == pytest.approx(100.0)
