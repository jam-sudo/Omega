"""Tests for Phase 269: prediction/protein_stability_predictor.py"""

import pytest

from omega_pbpk.prediction.protein_stability_predictor import (
    ProteinStabilityResult,
    predict_protein_stability,
    screen_storage_conditions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(condition="refrigerated", ph=7.0, mw=150.0, pi=7.0):
    return predict_protein_stability(
        protein_name="TestmAb",
        mw_kDa=mw,
        pi=pi,
        stress_condition=condition,
        storage_ph=ph,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_result()
    assert isinstance(result, ProteinStabilityResult)


def test_result_is_frozen():
    result = _make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.mw_kDa = 200.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Monomer percentage range
# ---------------------------------------------------------------------------


def test_monomer_pct_30days_in_range():
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition)
        assert 0.0 <= r.predicted_monomer_pct_30days <= 100.0, (
            f"{condition}: 30d={r.predicted_monomer_pct_30days}"
        )


def test_monomer_pct_12months_in_range():
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition)
        assert 0.0 <= r.predicted_monomer_pct_12months <= 100.0, (
            f"{condition}: 12m={r.predicted_monomer_pct_12months}"
        )


def test_monomer_pct_30days_ge_12months():
    """30 day monomer fraction must be >= 12 month fraction (monotonic decay)."""
    r = _make_result("room_temp_open")
    assert r.predicted_monomer_pct_30days >= r.predicted_monomer_pct_12months


# ---------------------------------------------------------------------------
# Frozen vs room temperature comparison
# ---------------------------------------------------------------------------


def test_frozen_more_stable_than_room_temp_30d():
    r_room = _make_result("room_temp_open", pi=8.0)
    r_frozen = _make_result("frozen", pi=8.0)
    assert r_frozen.predicted_monomer_pct_30days >= r_room.predicted_monomer_pct_30days


def test_frozen_more_stable_than_room_temp_12m():
    r_room = _make_result("room_temp_open")
    r_frozen = _make_result("frozen")
    assert r_frozen.predicted_monomer_pct_12months > r_room.predicted_monomer_pct_12months


# ---------------------------------------------------------------------------
# Shelf life
# ---------------------------------------------------------------------------


def test_shelf_life_positive():
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition, pi=8.0)
        assert r.shelf_life_days > 0.0, f"{condition}: shelf_life={r.shelf_life_days}"


def test_shelf_life_capped_at_3650():
    # Frozen + far from pI => very low k_total => shelf life should be capped at 3650
    r = _make_result("frozen", ph=5.0, pi=7.0)
    assert r.shelf_life_days <= 3650.0


def test_frozen_shelf_life_longer_than_room_temp():
    r_room = _make_result("room_temp_open", pi=8.0)
    r_frozen = _make_result("frozen", pi=8.0)
    assert r_frozen.shelf_life_days > r_room.shelf_life_days


# ---------------------------------------------------------------------------
# Stability grade
# ---------------------------------------------------------------------------


def test_stability_grade_valid_values():
    valid_grades = {"A", "B", "C"}
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition)
        assert r.stability_grade in valid_grades, f"{condition}: grade={r.stability_grade}"


def test_frozen_gets_grade_A():
    r = _make_result("frozen")
    assert r.stability_grade == "A"


# ---------------------------------------------------------------------------
# Dominant degradation pathway
# ---------------------------------------------------------------------------


def test_dominant_pathway_valid_values():
    valid_pathways = {"aggregation", "oxidation", "hydrolysis"}
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition)
        assert r.dominant_degradation_pathway in valid_pathways, (
            f"{condition}: dominant={r.dominant_degradation_pathway}"
        )


def test_light_exposed_dominant_oxidation_or_aggregation():
    # light_exposed has oxidation_risk=1.0 so k_ox=0.0005; dominant might be oxidation
    r = _make_result("light_exposed", pi=8.0)
    assert r.dominant_degradation_pathway in {"oxidation", "aggregation", "hydrolysis"}


# ---------------------------------------------------------------------------
# Screen storage conditions
# ---------------------------------------------------------------------------


def test_screen_returns_5_results():
    results = screen_storage_conditions("TestmAb", mw_kDa=150.0, pi=7.0)
    assert len(results) == 5


def test_screen_sorted_descending():
    results = screen_storage_conditions("TestmAb", mw_kDa=150.0, pi=7.0)
    pcts = [r.predicted_monomer_pct_12months for r in results]
    assert pcts == sorted(pcts, reverse=True), f"Not sorted descending: {pcts}"


def test_screen_first_is_most_stable():
    results = screen_storage_conditions("TestmAb", mw_kDa=150.0, pi=7.0)
    # First result should be frozen (most stable)
    assert results[0].stress_condition == "frozen"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        predict_protein_stability("X", mw_kDa=-1.0, pi=7.0, stress_condition="frozen")


def test_invalid_pi_too_high_raises():
    with pytest.raises(ValueError, match="pi"):
        predict_protein_stability("X", mw_kDa=150.0, pi=15.0, stress_condition="frozen")


def test_invalid_pi_negative_raises():
    with pytest.raises(ValueError, match="pi"):
        predict_protein_stability("X", mw_kDa=150.0, pi=-1.0, stress_condition="frozen")


def test_invalid_storage_ph_raises():
    with pytest.raises(ValueError, match="storage_ph"):
        predict_protein_stability(
            "X", mw_kDa=150.0, pi=7.0, stress_condition="frozen", storage_ph=3.0
        )


def test_invalid_storage_ph_too_high_raises():
    with pytest.raises(ValueError, match="storage_ph"):
        predict_protein_stability(
            "X", mw_kDa=150.0, pi=7.0, stress_condition="frozen", storage_ph=11.0
        )


def test_invalid_stress_condition_raises():
    with pytest.raises(ValueError, match="stress_condition"):
        predict_protein_stability("X", mw_kDa=150.0, pi=7.0, stress_condition="boiling_water")


# ---------------------------------------------------------------------------
# Rate ordering
# ---------------------------------------------------------------------------


def test_rates_non_negative():
    for condition in ["room_temp_open", "refrigerated", "frozen", "freeze_thaw", "light_exposed"]:
        r = _make_result(condition)
        assert r.aggregation_rate_per_day >= 0.0
        assert r.oxidation_rate_per_day >= 0.0
        assert r.hydrolysis_rate_per_day >= 0.0


def test_aggregation_rate_clamped_max():
    r = _make_result("room_temp_open", pi=7.0, ph=7.0, mw=150.0)
    assert r.aggregation_rate_per_day <= 0.1
