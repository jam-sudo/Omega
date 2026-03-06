"""Tests for Phase 181 — dose_rounding module."""
import pytest

from omega_pbpk.clinical.dose_rounding import DoseRoundingResult, round_dose, select_optimal_regimen

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
    results = select_optimal_regimen(
        "DrugS", 300.0, [100.0], frequencies=[2, 3]
    )
    assert len(results) == 2
    # freq=3 → 100 mg/dose → exact match
    exact = [r for r in results if r.dose_error_pct == pytest.approx(0.0)]
    assert len(exact) >= 1


def test_select_optimal_regimen_higher_freq_lower_per_dose():
    # 200 mg/day, strengths [50]: freq=4 → 50 mg/dose (exact)
    results = select_optimal_regimen("DrugT", 200.0, [50.0], frequencies=[1, 4])
    # The best (lowest error) should be the 4x/day option
    assert results[0].dose_error_pct == pytest.approx(0.0)
