"""Tests for Phase 1072 — Stability Forced Degradation Kinetics."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.stability_forced_degradation import (
    VALID_CONDITIONS,
    StabilityDegradationResult,
    predict_stress_degradation,
    screen_stress_conditions,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_stress_degradation("Drug", "acid")
    assert isinstance(result, StabilityDegradationResult)


def test_result_is_frozen():
    result = predict_stress_degradation("Drug", "acid")
    with pytest.raises((AttributeError, TypeError)):
        result.k_degradation_per_h = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Purity ordering: purity_at_1h >= purity_at_24h >= purity_at_48h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_purity_ordering(condition):
    r = predict_stress_degradation("Drug", condition)
    assert r.purity_at_1h <= 100.0
    assert r.purity_at_1h > 0.0
    assert r.purity_at_24h <= r.purity_at_1h
    assert r.purity_at_48h <= r.purity_at_24h


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_purity_at_1h_in_range(condition):
    r = predict_stress_degradation("Drug", condition)
    assert 0.0 < r.purity_at_1h <= 100.0


# ---------------------------------------------------------------------------
# t50 and t90 ordering: t90 < t50 (10% loss is faster than 50% loss)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_t90_less_than_t50(condition):
    r = predict_stress_degradation("Drug", condition)
    assert r.t90_h < r.t50_h


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_t50_positive(condition):
    r = predict_stress_degradation("Drug", condition)
    assert r.t50_h > 0.0


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_t90_positive(condition):
    r = predict_stress_degradation("Drug", condition)
    assert r.t90_h > 0.0


# ---------------------------------------------------------------------------
# Lability effect on degradation rate
# ---------------------------------------------------------------------------


def test_higher_acid_lability_lower_purity_24h():
    r_low = predict_stress_degradation("Drug", "acid", acid_lability=0.0)
    r_high = predict_stress_degradation("Drug", "acid", acid_lability=8.0)
    assert r_high.purity_at_24h < r_low.purity_at_24h


def test_higher_base_lability_lower_purity_24h():
    r_low = predict_stress_degradation("Drug", "base", base_lability=0.0)
    r_high = predict_stress_degradation("Drug", "base", base_lability=8.0)
    assert r_high.purity_at_24h < r_low.purity_at_24h


def test_higher_oxidation_sensitivity_lower_purity_24h():
    r_low = predict_stress_degradation("Drug", "oxidation", oxidation_sensitivity=0.0)
    r_high = predict_stress_degradation("Drug", "oxidation", oxidation_sensitivity=8.0)
    assert r_high.purity_at_24h < r_low.purity_at_24h


def test_higher_photo_sensitivity_lower_purity_24h():
    r_low = predict_stress_degradation("Drug", "photolysis", photo_sensitivity=0.0)
    r_high = predict_stress_degradation("Drug", "photolysis", photo_sensitivity=8.0)
    assert r_high.purity_at_24h < r_low.purity_at_24h


def test_higher_temp_faster_thermal_degradation():
    r_cool = predict_stress_degradation("Drug", "thermal", temperature_C=25.0)
    r_hot = predict_stress_degradation("Drug", "thermal", temperature_C=70.0)
    assert r_hot.k_degradation_per_h > r_cool.k_degradation_per_h
    assert r_hot.purity_at_24h < r_cool.purity_at_24h


# ---------------------------------------------------------------------------
# Stability rating
# ---------------------------------------------------------------------------


def test_stability_rating_valid_values():
    for cond in VALID_CONDITIONS:
        r = predict_stress_degradation("Drug", cond)
        assert r.stability_rating in {"stable", "moderate", "labile"}


def test_labile_rating_when_high_lability():
    # acid_lability=10 → k = 0.001*(1+5) = 0.006 /h → t90 = ln(100/90)/0.006 ≈ 17.6 h < 24 h
    r = predict_stress_degradation("Drug", "acid", acid_lability=10.0)
    assert r.stability_rating == "labile"


def test_stable_rating_when_no_lability():
    # acid default k=0.001 /h → t90 = -ln(0.9)/0.001 ≈ 105 h → "moderate" (24 < 105 <= 168)
    # For "stable" we need t90 > 168 h → k < -ln(0.9)/168 ≈ 0.000627
    # Use thermal at 25°C with default Ea=80 kJ/mol → k_thermal=0.0002 → t90≈ 527 h → "stable"
    r = predict_stress_degradation("Drug", "thermal", temperature_C=25.0)
    assert r.stability_rating == "stable"


def test_moderate_rating_intermediate():
    # photo_sensitivity=5 → k=0.003*(1+1)=0.006 → t90≈17.6h → labile
    # photo_sensitivity=0 → k=0.003 → t90≈35h → moderate
    r = predict_stress_degradation("Drug", "photolysis", photo_sensitivity=0.0)
    assert r.stability_rating == "moderate"


# ---------------------------------------------------------------------------
# All 5 conditions individually
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["acid", "base", "oxidation", "photolysis", "thermal"])
def test_all_conditions_return_valid_result(condition):
    r = predict_stress_degradation("Drug", condition)
    assert isinstance(r, StabilityDegradationResult)
    assert r.stress_condition == condition
    assert r.k_degradation_per_h > 0.0


# ---------------------------------------------------------------------------
# screen_stress_conditions
# ---------------------------------------------------------------------------


def test_screen_returns_5_results():
    results = screen_stress_conditions("Drug")
    assert len(results) == 5


def test_screen_sorted_by_k_descending():
    results = screen_stress_conditions("Drug")
    ks = [r.k_degradation_per_h for r in results]
    assert ks == sorted(ks, reverse=True)


def test_screen_contains_all_conditions():
    results = screen_stress_conditions("Drug")
    conditions_found = {r.stress_condition for r in results}
    assert conditions_found == VALID_CONDITIONS


def test_screen_with_labilities():
    results = screen_stress_conditions(
        "Drug",
        acid_lability=5.0,
        base_lability=3.0,
        oxidation_sensitivity=7.0,
        photo_sensitivity=2.0,
        temperature_C=40.0,
    )
    assert len(results) == 5
    ks = [r.k_degradation_per_h for r in results]
    assert ks == sorted(ks, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_stress_condition_raises():
    with pytest.raises(ValueError, match="stress_condition"):
        predict_stress_degradation("Drug", "uv_light")


def test_acid_lability_out_of_range_raises():
    with pytest.raises(ValueError, match="acid_lability"):
        predict_stress_degradation("Drug", "acid", acid_lability=11.0)


def test_acid_lability_negative_raises():
    with pytest.raises(ValueError, match="acid_lability"):
        predict_stress_degradation("Drug", "acid", acid_lability=-1.0)


def test_base_lability_out_of_range_raises():
    with pytest.raises(ValueError, match="base_lability"):
        predict_stress_degradation("Drug", "base", base_lability=15.0)


def test_oxidation_sensitivity_out_of_range_raises():
    with pytest.raises(ValueError, match="oxidation_sensitivity"):
        predict_stress_degradation("Drug", "oxidation", oxidation_sensitivity=-0.1)


def test_photo_sensitivity_out_of_range_raises():
    with pytest.raises(ValueError, match="photo_sensitivity"):
        predict_stress_degradation("Drug", "photolysis", photo_sensitivity=10.5)


def test_temperature_out_of_range_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_stress_degradation("Drug", "thermal", temperature_C=110.0)


def test_temperature_negative_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_stress_degradation("Drug", "thermal", temperature_C=-5.0)


# ---------------------------------------------------------------------------
# Analytical consistency checks
# ---------------------------------------------------------------------------


def test_t90_analytical_acid():
    # k_acid = 0.001 * (1 + 0.5*0) = 0.001
    # t90 = -ln(0.9) / 0.001 = ln(1/0.9) / 0.001
    r = predict_stress_degradation("Drug", "acid", acid_lability=0.0)
    expected_t90 = -math.log(0.9) / 0.001
    assert r.t90_h == pytest.approx(expected_t90, rel=1e-6)


def test_t50_analytical_acid():
    # t50 = ln(2) / 0.001
    r = predict_stress_degradation("Drug", "acid", acid_lability=0.0)
    expected_t50 = math.log(2) / 0.001
    assert r.t50_h == pytest.approx(expected_t50, rel=1e-6)


def test_purity_at_1h_analytical_acid():
    r = predict_stress_degradation("Drug", "acid", acid_lability=0.0)
    expected = 100.0 * math.exp(-0.001 * 1.0)
    assert r.purity_at_1h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# notes field
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    r = predict_stress_degradation("Drug", "acid")
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0
