"""Tests for drug stability and degradation — Phase 166."""

import math

import pytest

from omega_pbpk.prediction.drug_stability import (
    AcceleratedStabilityResult,
    StabilityResult,
    accelerated_stability_test,
    compare_storage_conditions,
    predict_stability,
)


# ── Validation ──────────────────────────────────────────────────────────

def test_negative_purity_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_stability("DrugA", initial_purity_pct=-1)


def test_zero_purity_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_stability("DrugA", initial_purity_pct=0)


def test_purity_above_100_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_stability("DrugA", initial_purity_pct=101)


def test_zero_k_deg_raises():
    with pytest.raises(ValueError, match="k_deg_per_month"):
        predict_stability("DrugA", k_deg_per_month=0)


def test_negative_k_deg_raises():
    with pytest.raises(ValueError, match="k_deg_per_month"):
        predict_stability("DrugA", k_deg_per_month=-0.01)


def test_zero_ea_raises():
    with pytest.raises(ValueError, match="activation_energy"):
        predict_stability("DrugA", activation_energy_kJ_per_mol=0)


def test_zero_t_months_raises():
    with pytest.raises(ValueError, match="t_months"):
        predict_stability("DrugA", t_months=0)


# ── Basic result structure ──────────────────────────────────────────────

def _default_result() -> StabilityResult:
    return predict_stability("TestDrug", t_months=24.0)


def test_result_type():
    r = _default_result()
    assert isinstance(r, StabilityResult)


def test_drug_name_stored():
    r = _default_result()
    assert r.drug_name == "TestDrug"


def test_temperature_stored():
    r = _default_result()
    assert r.temperature_C == 25.0


def test_time_array_starts_at_zero():
    r = _default_result()
    assert r.t_months[0] == pytest.approx(0.0)


def test_purity_starts_at_initial():
    r = _default_result()
    assert r.purity_pct[0] == pytest.approx(100.0)


def test_purity_arrays_same_length():
    r = _default_result()
    assert len(r.purity_pct) == len(r.t_months)


# ── Temperature effects ────────────────────────────────────────────────

def test_higher_temp_faster_degradation():
    r25 = predict_stability("D", temperature_C=25.0, t_months=24.0)
    r40 = predict_stability("D", temperature_C=40.0, t_months=24.0)
    assert r40.purity_at_end_pct < r25.purity_at_end_pct


def test_higher_temp_higher_rate():
    r25 = predict_stability("D", temperature_C=25.0)
    r40 = predict_stability("D", temperature_C=40.0)
    assert r40.degradation_rate_per_month > r25.degradation_rate_per_month


def test_cold_storage_slower_degradation():
    r5 = predict_stability("D", temperature_C=5.0, t_months=24.0)
    r25 = predict_stability("D", temperature_C=25.0, t_months=24.0)
    assert r5.purity_at_end_pct > r25.purity_at_end_pct


def test_shelf_life_decreases_with_temperature():
    r25 = predict_stability("D", temperature_C=25.0, k_deg_per_month=0.01, t_months=60.0)
    r40 = predict_stability("D", temperature_C=40.0, k_deg_per_month=0.01, t_months=60.0)
    # Both should have shelf life values; 40°C should be shorter
    assert r25.shelf_life_months is not None
    assert r40.shelf_life_months is not None
    assert r40.shelf_life_months < r25.shelf_life_months


# ── Stability categories ───────────────────────────────────────────────

def test_excellent_category():
    r = predict_stability("D", k_deg_per_month=0.0001, t_months=24.0)
    assert r.stability_category == "excellent"


def test_good_category():
    # Need purity between 97% and 99% at end
    # purity = 100 * exp(-k*24) => for 98%: k = -ln(0.98)/24 ≈ 0.000842
    r = predict_stability("D", k_deg_per_month=0.000842, t_months=24.0)
    assert r.stability_category == "good"


def test_poor_category():
    r = predict_stability("D", k_deg_per_month=0.05, t_months=24.0)
    assert r.stability_category == "poor"


def test_marginal_category():
    # purity between 95% and 97% at end
    # For 96%: k = -ln(0.96)/24 ≈ 0.001701
    r = predict_stability("D", k_deg_per_month=0.001701, t_months=24.0)
    assert r.stability_category == "marginal"


# ── Shelf life ──────────────────────────────────────────────────────────

def test_stable_drug_no_shelf_life():
    """Very slow degradation — purity never drops below limit."""
    r = predict_stability("D", k_deg_per_month=0.0001, t_months=24.0)
    assert r.shelf_life_months is None


def test_shelf_life_positive_when_present():
    r = predict_stability("D", k_deg_per_month=0.05, t_months=24.0)
    assert r.shelf_life_months is not None
    assert r.shelf_life_months > 0


# ── compare_storage_conditions ──────────────────────────────────────────

def test_compare_returns_three_results():
    results = compare_storage_conditions("DrugA")
    assert len(results) == 3


def test_compare_temperatures():
    results = compare_storage_conditions("DrugA")
    temps = [r.temperature_C for r in results]
    assert temps == [5.0, 25.0, 40.0]


def test_compare_ordering():
    results = compare_storage_conditions("DrugA", t_months=24.0)
    # 5°C should have highest purity, 40°C lowest
    assert results[0].purity_at_end_pct > results[1].purity_at_end_pct
    assert results[1].purity_at_end_pct > results[2].purity_at_end_pct


# ── accelerated_stability_test ──────────────────────────────────────────

def test_accelerated_result_type():
    r = accelerated_stability_test("DrugA")
    assert isinstance(r, AcceleratedStabilityResult)


def test_accelerated_shelf_life_positive():
    r = accelerated_stability_test("DrugA")
    assert r.predicted_shelf_life_25C_months > 0


def test_accelerated_default_conditions():
    r = accelerated_stability_test("DrugA")
    assert len(r.conditions) == 3


def test_accelerated_arrhenius_slope_positive():
    r = accelerated_stability_test("DrugA")
    assert r.arrhenius_slope > 0


def test_accelerated_validation_k_deg():
    with pytest.raises(ValueError, match="k_deg_per_month"):
        accelerated_stability_test("DrugA", k_deg_per_month=0)


def test_accelerated_validation_ea():
    with pytest.raises(ValueError, match="activation_energy"):
        accelerated_stability_test("DrugA", activation_energy_kJ_per_mol=0)


# ── Notes ───────────────────────────────────────────────────────────────

def test_notes_is_list():
    r = _default_result()
    assert isinstance(r.notes, list)


def test_high_humidity_note():
    r = predict_stability("D", humidity_pct=80.0)
    assert any("humidity" in n.lower() for n in r.notes)
