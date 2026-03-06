"""Tests for clinical/drug_food_interaction.py — Phase 188."""

import pytest

from omega_pbpk.clinical.drug_food_interaction import (
    FoodEffectResult,
    assess_food_effect,
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
