"""Tests for Phase 259: drug_crystal_growth_predictor."""

import pytest

from omega_pbpk.prediction.drug_crystal_growth_predictor import (
    CRYSTAL_HABITS,
    CrystalGrowthResult,
    compare_storage_conditions,
    predict_crystal_growth,
)

# ---------------------------------------------------------------------------
# Basic return type and structural tests
# ---------------------------------------------------------------------------


def test_returns_crystal_growth_result():
    result = predict_crystal_growth("DrugA", 50.0, "cubic")
    assert isinstance(result, CrystalGrowthResult)


def test_result_is_frozen():
    result = predict_crystal_growth("DrugA", 50.0, "cubic")
    with pytest.raises((AttributeError, TypeError)):
        result.final_size_um = 999.0  # type: ignore[misc]


def test_drug_name_preserved():
    result = predict_crystal_growth("TestDrug", 50.0, "cubic")
    assert result.drug_name == "TestDrug"


def test_crystal_habit_preserved():
    result = predict_crystal_growth("DrugA", 50.0, "needle")
    assert result.crystal_habit == "needle"


# ---------------------------------------------------------------------------
# Size constraints
# ---------------------------------------------------------------------------


def test_final_size_ge_initial_size_cubic():
    result = predict_crystal_growth("DrugA", 50.0, "cubic")
    assert result.final_size_um >= result.initial_size_um


def test_final_size_ge_initial_size_needle():
    result = predict_crystal_growth("DrugA", 100.0, "needle", 40.0, 75.0)
    assert result.final_size_um >= result.initial_size_um


def test_size_increase_pct_nonnegative():
    for habit in CRYSTAL_HABITS:
        result = predict_crystal_growth("DrugA", 50.0, habit)
        assert result.size_increase_pct >= 0.0, f"failed for habit={habit}"


def test_initial_size_preserved():
    result = predict_crystal_growth("DrugA", 75.0, "plate")
    assert result.initial_size_um == 75.0


# ---------------------------------------------------------------------------
# Temperature effects
# ---------------------------------------------------------------------------


def test_higher_temperature_more_growth():
    r_hot = predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=40.0, storage_rh_pct=60.0)
    r_cool = predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=5.0, storage_rh_pct=60.0)
    assert r_hot.size_increase_pct > r_cool.size_increase_pct


def test_refrigerated_less_growth_than_ambient():
    r_ambient = predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=25.0)
    r_fridge = predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=5.0)
    assert r_fridge.size_increase_pct < r_ambient.size_increase_pct


def test_freezer_least_growth():
    r_ambient = predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=25.0)
    r_freezer = predict_crystal_growth(
        "DrugA", 50.0, "cubic", storage_temp_C=-20.0, storage_rh_pct=10.0
    )
    assert r_freezer.size_increase_pct < r_ambient.size_increase_pct


# ---------------------------------------------------------------------------
# Ostwald ripening risk categories
# ---------------------------------------------------------------------------


def test_ostwald_ripening_risk_valid_values():
    valid = {"low", "moderate", "high"}
    for habit in CRYSTAL_HABITS:
        result = predict_crystal_growth("DrugA", 50.0, habit, 40.0, 75.0)
        assert result.ostwald_ripening_risk in valid, (
            f"habit={habit} gave risk={result.ostwald_ripening_risk}"
        )


def test_low_risk_no_stabilizer():
    # Cold, dry, slow-growing spherulite should be low risk
    result = predict_crystal_growth(
        "DrugA", 50.0, "spherulite", storage_temp_C=-20.0, storage_rh_pct=10.0
    )
    assert result.ostwald_ripening_risk == "low"
    assert result.recommended_stabilizer == "none"


def test_high_risk_hpc_stabilizer():
    # Small initial size (1 um) + hot, humid, fast-growing dendrite = high risk
    # size_increase_pct >> 50% for these conditions
    result = predict_crystal_growth(
        "DrugA", 1.0, "dendrite", storage_temp_C=55.0, storage_rh_pct=90.0
    )
    assert result.ostwald_ripening_risk == "high"
    assert result.recommended_stabilizer == "HPC"


def test_recommended_stabilizer_is_string():
    for habit in CRYSTAL_HABITS:
        result = predict_crystal_growth("DrugA", 50.0, habit)
        assert isinstance(result.recommended_stabilizer, str)


# ---------------------------------------------------------------------------
# compare_storage_conditions
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_storage_conditions("DrugB", 50.0, "cubic")
    assert len(results) == 4


def test_compare_sorted_ascending_by_size_increase():
    results = compare_storage_conditions("DrugB", 50.0, "cubic")
    increases = [r.size_increase_pct for r in results]
    assert increases == sorted(increases)


def test_compare_best_is_freezer():
    results = compare_storage_conditions("DrugB", 50.0, "cubic")
    # -20C/10RH should have least growth
    best = results[0]
    assert best.storage_temp_C == -20.0


def test_compare_worst_is_stress_condition():
    results = compare_storage_conditions("DrugB", 50.0, "dendrite")
    worst = results[-1]
    # 40C/75RH should have most growth
    assert worst.storage_temp_C == 40.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_initial_size_zero():
    with pytest.raises(ValueError, match="initial_size_um"):
        predict_crystal_growth("DrugA", 0.0, "cubic")


def test_invalid_initial_size_negative():
    with pytest.raises(ValueError, match="initial_size_um"):
        predict_crystal_growth("DrugA", -5.0, "cubic")


def test_invalid_temp_too_high():
    with pytest.raises(ValueError, match="storage_temp_C"):
        predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=65.0)


def test_invalid_temp_too_low():
    with pytest.raises(ValueError, match="storage_temp_C"):
        predict_crystal_growth("DrugA", 50.0, "cubic", storage_temp_C=-90.0)


def test_invalid_crystal_habit():
    with pytest.raises(ValueError, match="crystal_habit"):
        predict_crystal_growth("DrugA", 50.0, "hexagonal")
