"""Tests for Phase 376 — Biologic Drug Aggregation Kinetics."""

import math

import pytest

from omega_pbpk.core.biologic_aggregation_kinetics import (
    _STRESS_MULTIPLIERS,
    BiologicAggregationResult,
    screen_stress_conditions,
    simulate_aggregation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(
    stress_condition="normal",
    initial_monomer_mg_mL=1.0,
    base_rate=0.001,
    t_end_h=720.0,
    dt_h=1.0,
):
    return simulate_aggregation(
        drug_name="TestMAb",
        stress_condition=stress_condition,
        initial_monomer_mg_mL=initial_monomer_mg_mL,
        base_aggregation_rate_per_h=base_rate,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _base()
    assert isinstance(result, BiologicAggregationResult)


def test_result_has_list_fields():
    result = _base()
    assert isinstance(result.times_h, list)
    assert isinstance(result.monomer_fraction, list)
    assert isinstance(result.aggregate_fraction, list)


# ---------------------------------------------------------------------------
# 2. Trajectory correctness
# ---------------------------------------------------------------------------


def test_monomer_starts_at_one():
    result = _base()
    assert abs(result.monomer_fraction[0] - 1.0) < 1e-9


def test_aggregate_starts_at_zero():
    result = _base()
    assert abs(result.aggregate_fraction[0] - 0.0) < 1e-9


def test_monomer_decreases_over_time():
    result = _base()
    fracs = result.monomer_fraction
    assert fracs[-1] < fracs[0]


def test_aggregate_increases_over_time():
    result = _base()
    agg = result.aggregate_fraction
    assert agg[-1] > agg[0]


def test_monomer_plus_aggregate_equals_one():
    result = _base()
    for mf, af in zip(result.monomer_fraction, result.aggregate_fraction):
        assert abs(mf + af - 1.0) < 1e-9


def test_trajectory_length_consistent():
    result = _base(t_end_h=100.0, dt_h=10.0)
    assert len(result.times_h) == len(result.monomer_fraction) == len(result.aggregate_fraction)


# ---------------------------------------------------------------------------
# 3. t50 / t10 / shelf life
# ---------------------------------------------------------------------------


def test_t50_positive():
    result = _base()
    assert result.t50_h > 0.0


def test_t10_positive():
    result = _base()
    assert result.t10_h > 0.0


def test_t10_less_than_t50():
    result = _base()
    assert result.t10_h < result.t50_h


def test_shelf_life_equals_t10():
    result = _base()
    assert abs(result.shelf_life_h - result.t10_h) < 1e-9


def test_t50_formula_correct():
    result = _base(base_rate=0.01, stress_condition="normal")
    expected_t50 = math.log(2.0) / (0.01 * 1.0)
    assert abs(result.t50_h - expected_t50) < 1e-6


def test_t10_formula_correct():
    result = _base(base_rate=0.01, stress_condition="normal")
    expected_t10 = -math.log(0.9) / (0.01 * 1.0)
    assert abs(result.t10_h - expected_t10) < 1e-6


# ---------------------------------------------------------------------------
# 4. Stress conditions effect
# ---------------------------------------------------------------------------


def test_freeze_thaw_shorter_shelf_life_than_normal():
    normal = _base(stress_condition="normal")
    ft = _base(stress_condition="freeze_thaw")
    assert ft.shelf_life_h < normal.shelf_life_h


def test_heat_60C_very_fast_aggregation():
    result = _base(stress_condition="heat_60C")
    # At 200x multiplier and base_rate=0.001, effective_rate=0.2/h
    # t10 = -ln(0.9)/0.2 ~ 0.527 h
    assert result.shelf_life_h < 5.0


def test_higher_stress_shorter_shelf_life():
    agit = _base(stress_condition="agitation")  # multiplier 8
    heat = _base(stress_condition="heat_40C")  # multiplier 20
    assert heat.shelf_life_h < agit.shelf_life_h


def test_stress_multiplier_stored_correctly():
    result = _base(stress_condition="freeze_thaw")
    assert abs(result.stress_multiplier - 5.0) < 1e-9


def test_effective_rate_equals_base_times_multiplier():
    base_rate = 0.005
    result = simulate_aggregation(
        drug_name="X",
        stress_condition="agitation",
        initial_monomer_mg_mL=1.0,
        base_aggregation_rate_per_h=base_rate,
    )
    expected_rate = base_rate * _STRESS_MULTIPLIERS["agitation"]
    assert abs(result.aggregation_rate_per_h - expected_rate) < 1e-9


# ---------------------------------------------------------------------------
# 5. screen_stress_conditions
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_stress_conditions("TestMAb", 1.0)
    assert isinstance(results, list)


def test_screen_returns_seven_results():
    results = screen_stress_conditions("TestMAb", 1.0)
    assert len(results) == 7


def test_screen_sorted_shelf_life_descending():
    results = screen_stress_conditions("TestMAb", 1.0)
    shelf_lives = [r.shelf_life_h for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_screen_most_stable_is_normal():
    results = screen_stress_conditions("TestMAb", 1.0)
    # Normal (multiplier=1) should be most stable (longest shelf life)
    assert results[0].stress_condition == "normal"


# ---------------------------------------------------------------------------
# 6. Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_initial_monomer():
    with pytest.raises(ValueError, match="initial_monomer_mg_mL"):
        simulate_aggregation(
            drug_name="X",
            stress_condition="normal",
            initial_monomer_mg_mL=-1.0,
        )


def test_validation_zero_base_rate():
    with pytest.raises(ValueError, match="base_aggregation_rate_per_h"):
        simulate_aggregation(
            drug_name="X",
            stress_condition="normal",
            initial_monomer_mg_mL=1.0,
            base_aggregation_rate_per_h=0.0,
        )


def test_validation_unknown_stress_condition():
    with pytest.raises(ValueError, match="stress_condition"):
        simulate_aggregation(
            drug_name="X",
            stress_condition="unknown_condition",
            initial_monomer_mg_mL=1.0,
        )
