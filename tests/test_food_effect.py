"""Tests for Phase 257: food_effect.py"""

import pytest

from omega_pbpk.clinical.food_effect import (
    FoodEffectResult,
    compare_fed_states,
    simulate_food_effect,
)

# --- Helpers ----------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_base_per_h=1.0,
        f_oral_fasted=0.5,
        bcs_class="II",
        fed_state="fasted",
    )
    defaults.update(kwargs)
    return simulate_food_effect(**defaults)


# --- Input validation -------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _default(dose_mg=0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _default(dose_mg=-10)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _default(cl_L_per_h=0)


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _default(cl_L_per_h=-1)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _default(vd_L=0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _default(vd_L=-5)


def test_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_base_per_h"):
        _default(ka_base_per_h=0)


def test_ka_negative_raises():
    with pytest.raises(ValueError, match="ka_base_per_h"):
        _default(ka_base_per_h=-1)


def test_f_oral_zero_raises():
    with pytest.raises(ValueError, match="f_oral_fasted"):
        _default(f_oral_fasted=0.0)


def test_f_oral_above_one_raises():
    with pytest.raises(ValueError, match="f_oral_fasted"):
        _default(f_oral_fasted=1.1)


def test_invalid_fed_state_raises():
    with pytest.raises(ValueError, match="fed_state"):
        _default(fed_state="lunch")


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        _default(bcs_class="V")


# --- Basic output checks ----------------------------------------------------


def test_cmax_positive():
    result = _default()
    assert result.cmax > 0


def test_auc_positive():
    result = _default()
    assert result.auc > 0


def test_times_h_nonempty():
    result = _default()
    assert len(result.times_h) > 0


def test_c_plasma_same_length_as_times():
    result = _default()
    assert len(result.c_plasma_mg_L) == len(result.times_h)


# --- Fasted reference -------------------------------------------------------


def test_fasted_f_relative_is_one():
    result = _default(fed_state="fasted")
    assert result.f_relative == pytest.approx(1.0, abs=1e-9)


def test_fasted_cmax_ratio_is_one():
    result = _default(fed_state="fasted")
    assert result.cmax_ratio == pytest.approx(1.0, abs=1e-9)


# --- Food effect on BCS II drug ---------------------------------------------


def test_high_fat_f_relative_greater_than_one_bcs2():
    """High-fat meal should increase AUC for BCS II drug."""
    result = _default(fed_state="high_fat", bcs_class="II")
    assert result.f_relative > 1.0


def test_high_fat_longer_tmax_than_fasted():
    """Lag time in fed state delays tmax."""
    fasted = _default(fed_state="fasted")
    high_fat = _default(fed_state="high_fat")
    assert high_fat.tmax_h > fasted.tmax_h


# --- compare_fed_states -----------------------------------------------------


def test_compare_fed_states_returns_three():
    results = compare_fed_states("Drug", 100.0, 5.0, 50.0)
    assert len(results) == 3


def test_compare_fed_states_order():
    results = compare_fed_states("Drug", 100.0, 5.0, 50.0)
    assert [r.fed_state for r in results] == ["fasted", "low_fat", "high_fat"]


# --- BCS class sensitivity --------------------------------------------------


def test_bcs1_smaller_food_effect_than_bcs2():
    """BCS I drugs are less affected by food than BCS II."""
    bcs1 = _default(fed_state="high_fat", bcs_class="I")
    bcs2 = _default(fed_state="high_fat", bcs_class="II")
    # BCS I food effect (deviation from 1.0) should be smaller
    assert abs(bcs1.f_relative - 1.0) < abs(bcs2.f_relative - 1.0)


# --- Category and recommendation --------------------------------------------


def test_food_effect_category_valid():
    for state in ("fasted", "low_fat", "high_fat"):
        result = _default(fed_state=state)
        assert result.food_effect_category in {"none", "moderate", "significant"}


def test_recommendation_nonempty():
    for state in ("fasted", "low_fat", "high_fat"):
        result = _default(fed_state=state)
        assert len(result.recommendation) > 0


# --- Dose linearity ---------------------------------------------------------


def test_dose_doubling_doubles_auc():
    r1 = _default(dose_mg=100.0)
    r2 = _default(dose_mg=200.0)
    assert r2.auc == pytest.approx(2 * r1.auc, rel=0.05)


# --- Result is FoodEffectResult ---------------------------------------------


def test_result_type():
    result = _default()
    assert isinstance(result, FoodEffectResult)


def test_drug_name_stored():
    result = simulate_food_effect("Ibuprofen", 400.0, 5.0, 50.0)
    assert result.drug_name == "Ibuprofen"


def test_fed_state_stored():
    result = _default(fed_state="low_fat")
    assert result.fed_state == "low_fat"


def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0
