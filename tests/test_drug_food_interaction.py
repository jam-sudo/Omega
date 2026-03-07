"""Tests for clinical/drug_food_interaction.py — Phase 188 and Phase 743."""

import pytest

from omega_pbpk.clinical.drug_food_interaction import (
    FoodEffectPKResult,
    FoodEffectResult,
    assess_food_effect,
    assess_food_effect_pk,
    compare_food_conditions,
    high_fat_meal_impact,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = assess_food_effect("TestDrug", bcs_class="I", logP=1.0)
    assert isinstance(r, FoodEffectResult)


def test_fields_populated():
    r = assess_food_effect("TestDrug", bcs_class="I", logP=1.0)
    assert r.drug_name == "TestDrug"
    assert r.bcs_class == "I"


# ---------------------------------------------------------------------------
# BCS I: neutral food effect
# ---------------------------------------------------------------------------


def test_bcs_I_neutral_food_effect():
    r = assess_food_effect("Drug", bcs_class="I", logP=0.0, food_condition="fed_high_fat")
    # logP=0 → auc_ratio = 1.0, food_effect_category should be neutral
    assert r.food_effect_category == "neutral"


def test_bcs_I_auc_ratio_close_to_one():
    r = assess_food_effect("Drug", bcs_class="I", logP=1.0, food_condition="fed_high_fat")
    assert 0.9 <= r.auc_ratio_fed_fasted <= 1.2


# ---------------------------------------------------------------------------
# BCS II + high logP + high_fat: positive food effect
# ---------------------------------------------------------------------------


def test_bcs_II_high_logP_positive_food_effect():
    r = assess_food_effect("Drug", bcs_class="II", logP=5.0, food_condition="fed_high_fat")
    assert r.food_effect_category == "positive"


def test_bcs_II_high_fat_greater_auc_than_low_fat():
    r_hf = assess_food_effect("Drug", bcs_class="II", logP=4.0, food_condition="fed_high_fat")
    r_lf = assess_food_effect("Drug", bcs_class="II", logP=4.0, food_condition="fed_low_fat")
    assert r_hf.auc_ratio_fed_fasted > r_lf.auc_ratio_fed_fasted


# ---------------------------------------------------------------------------
# BCS III: negative food effect
# ---------------------------------------------------------------------------


def test_bcs_III_negative_food_effect():
    r = assess_food_effect("Drug", bcs_class="III", logP=1.0, food_condition="fed_high_fat")
    assert r.food_effect_category == "negative"


def test_bcs_III_auc_ratio_less_than_one():
    r = assess_food_effect("Drug", bcs_class="III", logP=1.0, food_condition="fed_high_fat")
    assert r.auc_ratio_fed_fasted < 1.0


# ---------------------------------------------------------------------------
# Tmax delay
# ---------------------------------------------------------------------------


def test_high_fat_tmax_delay_greater_than_low_fat():
    r_hf = assess_food_effect("Drug", bcs_class="I", logP=2.0, food_condition="fed_high_fat")
    r_lf = assess_food_effect("Drug", bcs_class="I", logP=2.0, food_condition="fed_low_fat")
    assert r_hf.tmax_delay_h > r_lf.tmax_delay_h


def test_fasted_no_tmax_delay():
    r = assess_food_effect("Drug", bcs_class="I", logP=2.0, food_condition="fasted")
    assert r.tmax_delay_h == 0.0


# ---------------------------------------------------------------------------
# Fasted condition: ratios = 1.0
# ---------------------------------------------------------------------------


def test_fasted_auc_ratio_is_one():
    r = assess_food_effect("Drug", bcs_class="II", logP=3.0, food_condition="fasted")
    assert r.auc_ratio_fed_fasted == 1.0


def test_fasted_cmax_ratio_is_one():
    r = assess_food_effect("Drug", bcs_class="II", logP=3.0, food_condition="fasted")
    assert r.cmax_ratio_fed_fasted == 1.0


# ---------------------------------------------------------------------------
# Clinical relevance and recommendation
# ---------------------------------------------------------------------------


def test_neutral_recommendation():
    r = assess_food_effect("Drug", bcs_class="I", logP=0.0, food_condition="fed_high_fat")
    assert r.recommendation == "May be taken with or without food"


def test_positive_high_relevance_recommendation():
    # BCS II + high logP should give "Take with food"
    r = assess_food_effect("Drug", bcs_class="II", logP=5.0, food_condition="fed_high_fat")
    if r.clinical_relevance in ("moderate", "high"):
        assert r.recommendation == "Take with food"


def test_clinical_relevance_field_valid():
    r = assess_food_effect("Drug", bcs_class="II", logP=3.0, food_condition="fed_high_fat")
    assert r.clinical_relevance in ("none", "low", "moderate", "high")


# ---------------------------------------------------------------------------
# high_fat_meal_impact
# ---------------------------------------------------------------------------


def test_high_fat_meal_impact_returns_3():
    results = high_fat_meal_impact("Drug", bcs_class="II", logP=3.0)
    assert len(results) == 3


def test_high_fat_meal_impact_conditions():
    results = high_fat_meal_impact("Drug", bcs_class="II", logP=3.0)
    conditions = {r.food_condition for r in results}
    assert conditions == {"fasted", "fed_low_fat", "fed_high_fat"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError, match="Invalid BCS class"):
        assess_food_effect("Drug", bcs_class="V", logP=2.0)


def test_invalid_food_condition_raises():
    with pytest.raises(ValueError, match="Invalid food_condition"):
        assess_food_effect("Drug", bcs_class="I", logP=2.0, food_condition="breakfast")


def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="Invalid formulation"):
        assess_food_effect("Drug", bcs_class="I", logP=2.0, formulation="tablet")


def test_bcs_IV_returns_result():
    r = assess_food_effect("Drug", bcs_class="IV", logP=1.0, food_condition="fed_high_fat")
    assert isinstance(r, FoodEffectResult)


# ===========================================================================
# Phase 743 — assess_food_effect_pk tests
# ===========================================================================


def test_pk_returns_dataclass():
    r = assess_food_effect_pk("TestDrug", dose_mg=100.0)
    assert isinstance(r, FoodEffectPKResult)


def test_pk_fasted_cmax_ratio_approx_one():
    """When food_type='fasted', fed==fasted so ratios should be ~1.0."""
    r = assess_food_effect_pk("Drug", dose_mg=100.0, food_type="fasted")
    assert abs(r.cmax_ratio - 1.0) < 0.01
    assert abs(r.auc_ratio - 1.0) < 0.01


def test_pk_high_fat_delays_tmax():
    """Fed high_fat (kge=0.5) should delay tmax vs fasted (kge=2.0)."""
    r = assess_food_effect_pk(
        "Drug", dose_mg=100.0, food_type="high_fat", ka_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert r.tmax_fed_h >= r.tmax_fasted_h


def test_pk_high_logp_high_fat_auc_ratio_gt_1():
    """High logP drug with high_fat → lymphatic enhancement → higher AUC."""
    r = assess_food_effect_pk(
        "Lipophilic", dose_mg=100.0, logp=6.0, f_oral=0.5, food_type="high_fat"
    )
    assert r.auc_ratio > 1.0


def test_pk_grapefruit_high_cyp3a4_increases_auc_ratio():
    """Grapefruit + high fm_cyp3a4 → CYP3A4 inhibition → higher fed_f → higher AUC ratio."""
    r_no_gfj = assess_food_effect_pk(
        "Drug", dose_mg=100.0, fm_cyp3a4=0.9, grapefruit=False, food_type="high_fat", f_oral=0.3
    )
    r_gfj = assess_food_effect_pk(
        "Drug", dose_mg=100.0, fm_cyp3a4=0.9, grapefruit=True, food_type="high_fat", f_oral=0.3
    )
    assert r_gfj.auc_ratio > r_no_gfj.auc_ratio


def test_pk_cmax_ratio_positive():
    r = assess_food_effect_pk("Drug", dose_mg=50.0, food_type="high_fat")
    assert r.cmax_ratio > 0
    assert r.auc_ratio > 0


def test_pk_food_effect_magnitude_valid():
    for ft in ("fasted", "low_fat", "high_fat"):
        r = assess_food_effect_pk("Drug", dose_mg=100.0, food_type=ft)
        assert r.food_effect_magnitude in ("none", "moderate", "significant")


def test_pk_fda_recommendation_valid():
    for ft in ("fasted", "low_fat", "high_fat"):
        r = assess_food_effect_pk("Drug", dose_mg=100.0, food_type=ft)
        assert r.fda_recommendation in ("take with food", "take fasted", "no restriction")


def test_compare_food_conditions_returns_3():
    results = compare_food_conditions("Drug", dose_mg=100.0)
    assert len(results) == 3


def test_compare_food_conditions_all_types():
    results = compare_food_conditions("Drug", dose_mg=100.0)
    food_types = {r.food_type for r in results}
    assert food_types == {"fasted", "low_fat", "high_fat"}


def test_pk_notes_nonempty():
    r = assess_food_effect_pk("Drug", dose_mg=100.0)
    assert len(r.notes) > 0


def test_pk_dose_mg_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        assess_food_effect_pk("Drug", dose_mg=0.0)


def test_pk_dose_mg_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        assess_food_effect_pk("Drug", dose_mg=-50.0)


def test_pk_linear_dose_scaling_cmax():
    """2x dose → 2x Cmax (linear 1-cpt model)."""
    r1 = assess_food_effect_pk("Drug", dose_mg=100.0, food_type="fasted")
    r2 = assess_food_effect_pk("Drug", dose_mg=200.0, food_type="fasted")
    assert abs(r2.cmax_fasted / r1.cmax_fasted - 2.0) < 0.01


def test_pk_bile_enhancement_increases_auc():
    """Bile salt enhancement should increase AUC ratio vs no enhancement."""
    r_no_bile = assess_food_effect_pk(
        "Drug", dose_mg=100.0, food_type="high_fat", has_bile_enhancement=False, f_oral=0.5
    )
    r_bile = assess_food_effect_pk(
        "Drug", dose_mg=100.0, food_type="high_fat", has_bile_enhancement=True, f_oral=0.5
    )
    assert r_bile.auc_ratio >= r_no_bile.auc_ratio


def test_pk_f_oral_gt_1_raises():
    with pytest.raises(ValueError, match="f_oral"):
        assess_food_effect_pk("Drug", dose_mg=100.0, f_oral=1.5)


def test_pk_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        assess_food_effect_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_pk_fasted_auc_fasted_equals_auc_fed():
    """For fasted food_type, the fasted and fed simulations use same parameters."""
    r = assess_food_effect_pk(
        "Drug", dose_mg=100.0, food_type="fasted", ka_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert abs(r.auc_fasted - r.auc_fed) < 1e-6
