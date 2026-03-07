"""Tests for Phase 181 — dose_rounding module."""

import math as _math

import pytest

from omega_pbpk.clinical.dose_rounding import (
    DoseRoundingResult,
    bsa_based_dose_rounding,
    dose_rounding_report,
    pediatric_dose_by_age,
    round_dose,
    round_to_available_tablet,
    select_optimal_regimen,
    weight_based_dose_rounding,
)

# ---------------------------------------------------------------------------
# round_dose — exact match
# ---------------------------------------------------------------------------


def test_exact_match_single_tablet():
    result = round_dose("DrugA", 100.0, [100.0])
    assert result.n_tablets == 1
    assert result.tablet_strength_mg == 100.0
    assert result.rounded_dose_mg == pytest.approx(100.0)
    assert result.dose_error_pct == pytest.approx(0.0)
    assert result.within_10pct is True


def test_exact_match_multiple_strengths():
    result = round_dose("DrugB", 50.0, [25.0, 50.0, 100.0])
    assert result.rounded_dose_mg == pytest.approx(50.0)
    assert result.dose_error_pct == pytest.approx(0.0)
    assert result.n_tablets == 1


# ---------------------------------------------------------------------------
# round_dose — rounding behaviour
# ---------------------------------------------------------------------------


def test_rounds_to_nearest_above():
    # target 75 mg, only 50 mg tablet, max_tablets=2 → 1x50=50 and 2x50=100
    # both are equidistant (33% error); prefer_fewer_tablets=True picks 1x50
    result = round_dose("DrugC", 75.0, [50.0], max_tablets=2)
    # With prefer_fewer_tablets the algorithm picks the fewer-tablet option
    assert result.tablet_strength_mg == pytest.approx(50.0)
    assert result.n_tablets in (1, 2)  # either is valid; algorithm picks 1
    assert result.n_tablets == 1


def test_rounds_to_nearest_below():
    # target 75 mg, only 50 mg tablet, max_tablets=1 → 1x50=50
    result = round_dose("DrugD", 75.0, [50.0], max_tablets=1)
    assert result.rounded_dose_mg == pytest.approx(50.0)
    assert result.n_tablets == 1


def test_chooses_closest_strength():
    # target 90 mg, strengths [50, 100] → 1x100 is closer (10% err vs 44%)
    result = round_dose("DrugE", 90.0, [50.0, 100.0])
    assert result.rounded_dose_mg == pytest.approx(100.0)
    assert result.tablet_strength_mg == pytest.approx(100.0)
    assert result.n_tablets == 1


def test_multi_tablet_combination():
    # target 150 mg, strengths [50] → 3x50
    result = round_dose("DrugF", 150.0, [50.0], max_tablets=4)
    assert result.n_tablets == 3
    assert result.rounded_dose_mg == pytest.approx(150.0)
    assert result.dose_error_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# within_10pct flag
# ---------------------------------------------------------------------------


def test_within_10pct_true():
    result = round_dose("DrugG", 100.0, [105.0])
    assert result.within_10pct is True


def test_within_10pct_false():
    result = round_dose("DrugH", 100.0, [50.0], max_tablets=1)
    # 1x50 → -50% error
    assert result.within_10pct is False


def test_within_10pct_boundary_exactly_10():
    # target 100, strength 110 → 10% error (should be within)
    result = round_dose("DrugI", 100.0, [110.0], max_tablets=1)
    assert result.dose_error_pct == pytest.approx(10.0)
    assert result.within_10pct is True


# ---------------------------------------------------------------------------
# regimen_description
# ---------------------------------------------------------------------------


def test_regimen_description_singular():
    result = round_dose("DrugJ", 50.0, [50.0])
    assert "1 x 50 mg tablet" in result.regimen_description
    # singular: no trailing 's' after 'tablet'
    assert result.regimen_description == "1 x 50 mg tablet"


def test_regimen_description_plural():
    result = round_dose("DrugK", 100.0, [50.0], max_tablets=4)
    assert result.regimen_description == "2 x 50 mg tablets"


# ---------------------------------------------------------------------------
# prefer_fewer_tablets
# ---------------------------------------------------------------------------


def test_prefer_fewer_tablets():
    # target 100 mg, strengths [50, 100] — both give 0% error but 1x100 uses
    # fewer tablets than 2x50
    result = round_dose("DrugL", 100.0, [50.0, 100.0], prefer_fewer_tablets=True)
    assert result.n_tablets == 1
    assert result.tablet_strength_mg == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_target_zero():
    with pytest.raises(ValueError, match="target_dose_mg"):
        round_dose("DrugM", 0.0, [50.0])


def test_invalid_target_negative():
    with pytest.raises(ValueError, match="target_dose_mg"):
        round_dose("DrugN", -10.0, [50.0])


def test_empty_strengths():
    with pytest.raises(ValueError, match="available_strengths_mg"):
        round_dose("DrugO", 100.0, [])


def test_negative_strength():
    with pytest.raises(ValueError, match="positive"):
        round_dose("DrugP", 100.0, [-50.0])


# ---------------------------------------------------------------------------
# select_optimal_regimen
# ---------------------------------------------------------------------------


def test_select_optimal_regimen_returns_list():
    results = select_optimal_regimen("DrugQ", 200.0, [50.0, 100.0])
    assert isinstance(results, list)
    assert len(results) == 4  # default [1, 2, 3, 4]
    for r in results:
        assert isinstance(r, DoseRoundingResult)


def test_select_optimal_regimen_sorted_ascending():
    results = select_optimal_regimen("DrugR", 200.0, [50.0, 100.0])
    errors = [abs(r.dose_error_pct) for r in results]
    assert errors == sorted(errors)


def test_select_optimal_regimen_custom_frequencies():
    results = select_optimal_regimen("DrugS", 300.0, [100.0], frequencies=[2, 3])
    assert len(results) == 2
    # freq=3 → 100 mg/dose → exact match
    exact = [r for r in results if r.dose_error_pct == pytest.approx(0.0)]
    assert len(exact) >= 1


def test_select_optimal_regimen_higher_freq_lower_per_dose():
    # 200 mg/day, strengths [50]: freq=4 → 50 mg/dose (exact)
    results = select_optimal_regimen("DrugT", 200.0, [50.0], frequencies=[1, 4])
    # The best (lowest error) should be the 4x/day option
    assert results[0].dose_error_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Phase 558 — round_to_available_tablet
# ---------------------------------------------------------------------------


class TestRoundToAvailableTablet:
    def test_exact_match_returns_zero_deviation(self):
        result = round_to_available_tablet(100.0, [50.0, 100.0, 200.0])
        assert result["recommended_dose_mg"] == pytest.approx(100.0, rel=1e-9)
        assert result["tablet_strength_mg"] == 100.0
        assert result["deviation_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_closest_tablet_selected(self):
        """For dose 130 mg, closest tablet among 50/100/200 is 100."""
        result = round_to_available_tablet(130.0, [50.0, 100.0, 200.0])
        assert result["tablet_strength_mg"] == 100.0

    def test_n_tablets_is_positive(self):
        result = round_to_available_tablet(75.0, [50.0])
        assert result["n_tablets"] >= 0.5

    def test_deviation_pct_nonnegative(self):
        result = round_to_available_tablet(73.0, [50.0, 100.0])
        assert result["deviation_pct"] >= 0.0

    def test_n_tablets_rounded_to_half(self):
        """180 mg / 100 mg tablet → 2.0 tablets (rounds 1.8 to nearest 0.5)."""
        result = round_to_available_tablet(180.0, [100.0])
        assert result["n_tablets"] == 2.0

    def test_single_tablet_option(self):
        result = round_to_available_tablet(50.0, [25.0])
        assert result["n_tablets"] == pytest.approx(2.0, rel=1e-9)
        assert result["recommended_dose_mg"] == pytest.approx(50.0, rel=1e-9)

    def test_returns_required_keys(self):
        result = round_to_available_tablet(100.0, [100.0])
        for key in ["recommended_dose_mg", "tablet_strength_mg", "n_tablets", "deviation_pct"]:
            assert key in result

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            round_to_available_tablet(0.0, [50.0])

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            round_to_available_tablet(-10.0, [50.0])

    def test_invalid_empty_tablets(self):
        with pytest.raises(ValueError):
            round_to_available_tablet(100.0, [])

    def test_invalid_tablet_strength_zero(self):
        with pytest.raises(ValueError):
            round_to_available_tablet(100.0, [0.0, 50.0])


# ---------------------------------------------------------------------------
# Phase 558 — weight_based_dose_rounding
# ---------------------------------------------------------------------------


class TestWeightBasedDoseRounding:
    def test_target_dose_computed_correctly(self):
        result = weight_based_dose_rounding(70.0, 2.0, [50.0, 100.0, 150.0])
        assert result["target_dose_mg"] == pytest.approx(140.0, rel=1e-9)

    def test_returns_weight_and_dose_per_kg(self):
        result = weight_based_dose_rounding(60.0, 1.5, [50.0])
        assert result["weight_kg"] == 60.0
        assert result["dose_mg_per_kg"] == 1.5

    def test_max_dose_cap_applied(self):
        """70 kg × 5 mg/kg = 350 mg, capped at 200 mg."""
        result = weight_based_dose_rounding(70.0, 5.0, [100.0, 200.0], max_dose_mg=200.0)
        assert result["target_dose_mg"] == pytest.approx(200.0, rel=1e-9)

    def test_no_cap_without_max_dose(self):
        result = weight_based_dose_rounding(70.0, 5.0, [100.0, 200.0])
        assert result["target_dose_mg"] == pytest.approx(350.0, rel=1e-9)

    def test_n_tablets_positive(self):
        result = weight_based_dose_rounding(50.0, 2.0, [50.0])
        assert result["n_tablets"] > 0

    def test_invalid_weight_zero(self):
        with pytest.raises(ValueError):
            weight_based_dose_rounding(0.0, 2.0, [50.0])

    def test_invalid_dose_per_kg_zero(self):
        with pytest.raises(ValueError):
            weight_based_dose_rounding(70.0, 0.0, [50.0])

    def test_invalid_max_dose_zero(self):
        with pytest.raises(ValueError):
            weight_based_dose_rounding(70.0, 2.0, [50.0], max_dose_mg=0.0)


# ---------------------------------------------------------------------------
# Phase 558 — bsa_based_dose_rounding
# ---------------------------------------------------------------------------


class TestBsaBasedDoseRounding:
    def test_mosteller_bsa_formula(self):
        """BSA = sqrt(170 * 70 / 3600)."""
        height, weight = 170.0, 70.0
        expected_bsa = _math.sqrt(height * weight / 3600.0)
        result = bsa_based_dose_rounding(height, weight, 50.0, [50.0, 100.0])
        assert result["bsa_m2"] == pytest.approx(expected_bsa, rel=1e-6)

    def test_target_dose_proportional_to_bsa(self):
        height, weight = 170.0, 70.0
        dose_per_m2 = 100.0
        bsa = _math.sqrt(height * weight / 3600.0)
        expected_target = bsa * dose_per_m2
        result = bsa_based_dose_rounding(height, weight, dose_per_m2, [50.0, 100.0, 150.0])
        assert result["target_dose_mg"] == pytest.approx(expected_target, rel=1e-6)

    def test_returns_required_keys(self):
        result = bsa_based_dose_rounding(170.0, 70.0, 100.0, [100.0])
        for key in ["bsa_m2", "target_dose_mg", "recommended_dose_mg", "deviation_pct"]:
            assert key in result

    def test_invalid_height_zero(self):
        with pytest.raises(ValueError):
            bsa_based_dose_rounding(0.0, 70.0, 100.0, [50.0])

    def test_invalid_weight_zero(self):
        with pytest.raises(ValueError):
            bsa_based_dose_rounding(170.0, 0.0, 100.0, [50.0])

    def test_invalid_dose_per_m2_zero(self):
        with pytest.raises(ValueError):
            bsa_based_dose_rounding(170.0, 70.0, 0.0, [50.0])


# ---------------------------------------------------------------------------
# Phase 558 — pediatric_dose_by_age
# ---------------------------------------------------------------------------


class TestPediatricDoseByAge:
    def test_clark_age10_formula(self):
        """Clark: weight = 2.5*10+8 = 33 kg; dose = 33/68 * adult."""
        adult = 100.0
        weight = 2.5 * 10 + 8  # 33 kg
        expected = (weight / 68.0) * adult
        result = pediatric_dose_by_age(10.0, adult, method="clark")
        assert result == pytest.approx(expected, rel=1e-9)

    def test_young_method_ratio(self):
        """Young: dose = age / (age + 12) * adult."""
        age, adult = 6.0, 200.0
        expected = age / (age + 12.0) * adult
        result = pediatric_dose_by_age(age, adult, method="young")
        assert result == pytest.approx(expected, rel=1e-9)

    def test_diling_below_12(self):
        """Diling for age < 12 uses Young's rule."""
        age, adult = 8.0, 100.0
        expected = age / (age + 12.0) * adult
        result = pediatric_dose_by_age(age, adult, method="diling")
        assert result == pytest.approx(expected, rel=1e-9)

    def test_diling_at_or_above_12(self):
        """Diling for age >= 12: dose = age/20 * adult."""
        age, adult = 15.0, 200.0
        expected = age / 20.0 * adult
        result = pediatric_dose_by_age(age, adult, method="diling")
        assert result == pytest.approx(expected, rel=1e-9)

    def test_all_methods_return_positive_dose(self):
        for method in ["clark", "young", "diling"]:
            result = pediatric_dose_by_age(8.0, 100.0, method=method)
            assert result > 0.0

    def test_default_method_is_clark(self):
        expected = pediatric_dose_by_age(5.0, 100.0, method="clark")
        assert pediatric_dose_by_age(5.0, 100.0) == pytest.approx(expected, rel=1e-9)

    def test_age_zero_young_returns_zero(self):
        result = pediatric_dose_by_age(0.0, 100.0, method="young")
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_invalid_negative_age(self):
        with pytest.raises(ValueError):
            pediatric_dose_by_age(-1.0, 100.0)

    def test_invalid_adult_dose_zero(self):
        with pytest.raises(ValueError):
            pediatric_dose_by_age(5.0, 0.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            pediatric_dose_by_age(5.0, 100.0, method="unknown")


# ---------------------------------------------------------------------------
# Phase 558 — dose_rounding_report
# ---------------------------------------------------------------------------


class TestDoseRoundingReport:
    def test_within_tolerance_true_when_close(self):
        report = dose_rounding_report("TestDrug", 100.0, [100.0], tolerance_pct=20.0)
        assert report["within_tolerance"] is True

    def test_within_tolerance_false_when_far(self):
        """Dose 60 mg, only tablet 100 mg → 0.5*100=50 mg → 16.7% or 1*100=100 mg → 67% deviation.
        Closest tablet to 60 is 100 mg; round(60/100 * 2)/2 = round(1.2)/2 = 0.5, so 0.5*100=50 → dev=16.7%.
        Use tolerance_pct=10 to ensure it's outside tolerance."""
        report = dose_rounding_report("TestDrug", 60.0, [100.0], tolerance_pct=10.0)
        assert report["within_tolerance"] is False

    def test_returns_drug_name(self):
        report = dose_rounding_report("Metformin", 500.0, [250.0, 500.0])
        assert report["drug_name"] == "Metformin"

    def test_returns_target_dose_mg(self):
        report = dose_rounding_report("X", 75.0, [25.0, 50.0, 100.0])
        assert report["target_dose_mg"] == pytest.approx(75.0, rel=1e-9)

    def test_notes_not_empty(self):
        report = dose_rounding_report("X", 100.0, [100.0])
        assert len(report["notes"]) > 0

    def test_required_keys_present(self):
        report = dose_rounding_report("X", 100.0, [50.0, 100.0])
        for key in [
            "drug_name",
            "target_dose_mg",
            "recommended_dose_mg",
            "tablet_strength_mg",
            "n_tablets",
            "deviation_pct",
            "within_tolerance",
            "notes",
        ]:
            assert key in report

    def test_invalid_negative_tolerance(self):
        with pytest.raises(ValueError):
            dose_rounding_report("X", 100.0, [100.0], tolerance_pct=-1.0)
