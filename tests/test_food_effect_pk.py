"""Tests for Phase 468 — Food Effect on Drug PK."""

import pytest

from omega_pbpk.clinical.food_effect_pk import (
    FoodEffectResult,
    compare_meal_types,
    simulate_food_effect_pk,
)


def default_result(meal_type: str = "fasted", bcs_class: int = 1) -> FoodEffectResult:
    return simulate_food_effect_pk(
        drug_name="DrugA",
        dose_mg=100.0,
        bcs_class=bcs_class,
        meal_type=meal_type,
    )


# --- Return type ---


def test_return_type():
    result = default_result()
    assert isinstance(result, FoodEffectResult)


def test_result_attributes_present():
    result = default_result()
    assert result.drug_name == "DrugA"
    assert result.dose_mg == pytest.approx(100.0)
    assert result.bcs_class == 1
    assert result.meal_type == "fasted"


# --- Fasted meal type: no adjustment ---


def test_fasted_f_adjusted_equals_f_fasted():
    """For fasted, fa_mult=1.0, so f_adjusted = f_fasted."""
    result = simulate_food_effect_pk("X", 100.0, 1, "fasted", f_fasted=0.7)
    assert result.f_adjusted == pytest.approx(0.7)


def test_fasted_ka_adjusted_equals_ka_fasted():
    """For fasted, ka_mult=1.0, so ka_adjusted = ka_fasted."""
    result = simulate_food_effect_pk("X", 100.0, 1, "fasted", ka_fasted_per_h=1.5)
    assert result.ka_adjusted_per_h == pytest.approx(1.5)


def test_fasted_food_effect_on_auc_pct_zero():
    """Comparing fasted to fasted reference: 0% change."""
    result = default_result("fasted")
    assert result.food_effect_on_auc_pct == pytest.approx(0.0, abs=1e-6)


def test_fasted_food_effect_on_cmax_pct_zero():
    result = default_result("fasted")
    assert result.food_effect_on_cmax_pct == pytest.approx(0.0, abs=1e-6)


# --- High fat BCS I/II: food increases absorption ---


def test_high_fat_bcs1_f_adjusted_greater_than_fasted():
    """BCS I: fa_mult=1.3, so f_adjusted > f_fasted."""
    r_fasted = simulate_food_effect_pk("X", 100.0, 1, "fasted", f_fasted=0.6)
    r_hf = simulate_food_effect_pk("X", 100.0, 1, "high_fat", f_fasted=0.6)
    assert r_hf.f_adjusted > r_fasted.f_adjusted


def test_high_fat_bcs2_f_adjusted_greater_than_fasted():
    r_fasted = simulate_food_effect_pk("X", 100.0, 2, "fasted", f_fasted=0.6)
    r_hf = simulate_food_effect_pk("X", 100.0, 2, "high_fat", f_fasted=0.6)
    assert r_hf.f_adjusted > r_fasted.f_adjusted


# --- High fat BCS III/IV: food decreases absorption ---


def test_high_fat_bcs3_f_adjusted_less_than_fasted():
    """BCS III: fa_mult=0.7, so f_adjusted < f_fasted."""
    r_fasted = simulate_food_effect_pk("X", 100.0, 3, "fasted", f_fasted=0.7)
    r_hf = simulate_food_effect_pk("X", 100.0, 3, "high_fat", f_fasted=0.7)
    assert r_hf.f_adjusted < r_fasted.f_adjusted


def test_high_fat_bcs4_f_adjusted_less_than_fasted():
    r_fasted = simulate_food_effect_pk("X", 100.0, 4, "fasted", f_fasted=0.7)
    r_hf = simulate_food_effect_pk("X", 100.0, 4, "high_fat", f_fasted=0.7)
    assert r_hf.f_adjusted < r_fasted.f_adjusted


# --- AUC and Cmax ---


def test_cmax_positive():
    result = default_result()
    assert result.cmax_mg_L > 0.0


def test_tmax_non_negative():
    result = default_result()
    assert result.tmax_h >= 0.0


def test_auc_positive():
    result = default_result()
    assert result.auc_mg_h_per_L > 0.0


def test_times_and_concentrations_same_length():
    result = default_result()
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_fasted_cmax_stored():
    result = default_result("high_fat")
    assert result.fasted_cmax_mg_L > 0.0


def test_fasted_auc_stored():
    result = default_result("high_fat")
    assert result.fasted_auc_mg_h_per_L > 0.0


# --- Clinical significance ---


def test_clinical_significance_valid_string():
    result = default_result()
    assert result.clinical_significance in {"none", "minor", "moderate", "major"}


def test_fasted_clinical_significance_none():
    """fasted vs fasted: 0% change → 'none'."""
    result = default_result("fasted")
    assert result.clinical_significance == "none"


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- compare_meal_types ---


def test_compare_meal_types_returns_4():
    results = compare_meal_types("DrugA", 100.0, 1)
    assert len(results) == 4


def test_compare_meal_types_sorted_by_auc_descending():
    results = compare_meal_types("DrugA", 100.0, 1)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_meal_types_all_meal_types_represented():
    results = compare_meal_types("DrugA", 100.0, 1)
    meal_types = {r.meal_type for r in results}
    assert meal_types == {"fasted", "high_fat", "low_fat", "standard"}


# --- Validation errors ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_food_effect_pk("X", 0.0, 1, "fasted")


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_food_effect_pk("X", -50.0, 1, "fasted")


def test_validation_invalid_bcs_class():
    with pytest.raises(ValueError, match="bcs_class"):
        simulate_food_effect_pk("X", 100.0, 5, "fasted")


def test_validation_invalid_meal_type():
    with pytest.raises(ValueError, match="meal_type"):
        simulate_food_effect_pk("X", 100.0, 1, "breakfast")


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_food_effect_pk("X", 100.0, 1, "fasted", cl_sys_L_per_h=0.0)


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_food_effect_pk("X", 100.0, 1, "fasted", vd_sys_L=0.0)


def test_validation_f_fasted_zero():
    with pytest.raises(ValueError, match="f_fasted"):
        simulate_food_effect_pk("X", 100.0, 1, "fasted", f_fasted=0.0)


def test_validation_f_fasted_greater_than_one():
    with pytest.raises(ValueError, match="f_fasted"):
        simulate_food_effect_pk("X", 100.0, 1, "fasted", f_fasted=1.5)
