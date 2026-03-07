"""Tests for Phase 279 — Stress Stability Predictor."""

import math

import pytest

from omega_pbpk.prediction.stress_stability_predictor import (
    StressStabilityResult,
    compare_conditions,
    predict_stress_stability,
)

_ALL_CONDITIONS = [
    "thermal",
    "humidity",
    "photolytic",
    "acid_hydrolysis",
    "base_hydrolysis",
    "oxidative",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _default_result(**kwargs) -> StressStabilityResult:
    defaults = dict(
        drug_name="TestDrug",
        condition="thermal",
        temperature_C=40.0,
        humidity_pct=75.0,
        initial_purity_pct=99.5,
        k_deg_ref_per_day=1e-4,
        ea_kJ_per_mol=80.0,
    )
    defaults.update(kwargs)
    return predict_stress_stability(**defaults)


# ── Return type ───────────────────────────────────────────────────────────────


def test_return_type():
    result = _default_result()
    assert isinstance(result, StressStabilityResult)


def test_frozen_dataclass():
    result = _default_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ── Field preservation ────────────────────────────────────────────────────────


def test_drug_name_preserved():
    result = _default_result(drug_name="Aspirin")
    assert result.drug_name == "Aspirin"


def test_condition_preserved():
    result = _default_result(condition="oxidative")
    assert result.condition == "oxidative"


def test_temperature_preserved():
    result = _default_result(temperature_C=60.0)
    assert result.temperature_C == pytest.approx(60.0)


def test_humidity_preserved():
    result = _default_result(humidity_pct=50.0)
    assert result.humidity_pct == pytest.approx(50.0)


def test_initial_purity_preserved():
    result = _default_result(initial_purity_pct=98.0)
    assert result.initial_purity_pct == pytest.approx(98.0)


# ── Purity ordering ───────────────────────────────────────────────────────────


def test_purity_decreases_over_time():
    result = _default_result(k_deg_ref_per_day=1e-3)
    assert result.purity_at_30d >= result.purity_at_90d
    assert result.purity_at_90d >= result.purity_at_180d
    assert result.purity_at_180d >= result.purity_at_365d


def test_purity_at_30d_greater_than_purity_at_180d():
    result = _default_result(k_deg_ref_per_day=1e-3)
    assert result.purity_at_30d > result.purity_at_180d


# ── Purity bounds ─────────────────────────────────────────────────────────────


def test_all_purity_values_in_range():
    result = _default_result()
    for val in [
        result.purity_at_30d,
        result.purity_at_90d,
        result.purity_at_180d,
        result.purity_at_365d,
    ]:
        assert 0.0 <= val <= 100.0


def test_purity_clamp_very_high_k_deg():
    # Very fast degradation — purity should clamp to 0
    result = predict_stress_stability(
        drug_name="Unstable",
        condition="thermal",
        temperature_C=40.0,
        humidity_pct=50.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=100.0,
    )
    assert result.purity_at_365d == pytest.approx(0.0, abs=1e-6)


# ── Temperature effect ────────────────────────────────────────────────────────


def test_higher_temperature_faster_degradation():
    low_temp = predict_stress_stability(
        drug_name="Drug",
        condition="thermal",
        temperature_C=25.0,
        humidity_pct=0.0,
        initial_purity_pct=99.5,
        k_deg_ref_per_day=1e-3,
    )
    high_temp = predict_stress_stability(
        drug_name="Drug",
        condition="thermal",
        temperature_C=60.0,
        humidity_pct=0.0,
        initial_purity_pct=99.5,
        k_deg_ref_per_day=1e-3,
    )
    assert high_temp.purity_at_30d < low_temp.purity_at_30d


# ── Shelf life ────────────────────────────────────────────────────────────────


def test_shelf_life_positive():
    result = _default_result()
    assert result.shelf_life_days > 0.0


def test_shelf_life_formula():
    k_ref = 1e-3
    result = predict_stress_stability(
        drug_name="Drug",
        condition="thermal",
        temperature_C=25.0,  # at ref temp → k_eff ≈ k_ref
        humidity_pct=0.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=k_ref,
    )
    # at 25 C Arrhenius correction = 1; thermal multiplier = 1
    expected_shelf = -math.log(0.9) / k_ref
    assert result.shelf_life_days == pytest.approx(expected_shelf, rel=0.01)


# ── Stability class ───────────────────────────────────────────────────────────

_VALID_STABILITY_CLASSES = {"stable", "moderately_stable", "unstable"}


def test_stability_class_valid():
    result = _default_result()
    assert result.stability_class in _VALID_STABILITY_CLASSES


def test_stability_class_stable():
    result = predict_stress_stability(
        drug_name="Drug",
        condition="thermal",
        temperature_C=5.0,
        humidity_pct=0.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=1e-6,
    )
    assert result.stability_class == "stable"


def test_stability_class_unstable():
    result = predict_stress_stability(
        drug_name="Drug",
        condition="thermal",
        temperature_C=80.0,
        humidity_pct=0.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=0.1,
    )
    assert result.stability_class == "unstable"


# ── All conditions run ────────────────────────────────────────────────────────


@pytest.mark.parametrize("condition", _ALL_CONDITIONS)
def test_all_conditions_run(condition):
    result = predict_stress_stability(
        drug_name="Drug",
        condition=condition,
        temperature_C=40.0,
        humidity_pct=75.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=1e-3,
    )
    assert result.condition == condition
    assert result.k_deg_eff_per_day > 0.0


# ── compare_conditions ────────────────────────────────────────────────────────


def test_compare_conditions_sorted_ascending_purity_30d():
    results = compare_conditions(
        drug_name="Drug",
        conditions_list=_ALL_CONDITIONS,
        temperature_C=40.0,
        humidity_pct=75.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=1e-3,
    )
    purities = [r.purity_at_30d for r in results]
    assert purities == sorted(purities)


def test_compare_conditions_length():
    conditions = ["thermal", "photolytic", "oxidative"]
    results = compare_conditions(
        drug_name="Drug",
        conditions_list=conditions,
        temperature_C=40.0,
        humidity_pct=60.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_day=5e-4,
    )
    assert len(results) == 3


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_condition_raises():
    with pytest.raises(ValueError, match="condition"):
        predict_stress_stability(
            drug_name="Drug",
            condition="radioactive",
            temperature_C=40.0,
            humidity_pct=50.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=1e-3,
        )


def test_temperature_too_high_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=250.0,
            humidity_pct=50.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=1e-3,
        )


def test_temperature_too_low_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=-100.0,
            humidity_pct=50.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=1e-3,
        )


def test_humidity_above_100_raises():
    with pytest.raises(ValueError, match="humidity_pct"):
        predict_stress_stability(
            drug_name="Drug",
            condition="humidity",
            temperature_C=40.0,
            humidity_pct=110.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=1e-3,
        )


def test_initial_purity_zero_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=40.0,
            humidity_pct=50.0,
            initial_purity_pct=0.0,
            k_deg_ref_per_day=1e-3,
        )


def test_initial_purity_above_100_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=40.0,
            humidity_pct=50.0,
            initial_purity_pct=101.0,
            k_deg_ref_per_day=1e-3,
        )


def test_k_deg_zero_raises():
    with pytest.raises(ValueError, match="k_deg_ref_per_day"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=40.0,
            humidity_pct=50.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=0.0,
        )


def test_ea_zero_raises():
    with pytest.raises(ValueError, match="ea_kJ_per_mol"):
        predict_stress_stability(
            drug_name="Drug",
            condition="thermal",
            temperature_C=40.0,
            humidity_pct=50.0,
            initial_purity_pct=99.0,
            k_deg_ref_per_day=1e-3,
            ea_kJ_per_mol=0.0,
        )
