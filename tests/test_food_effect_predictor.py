"""Tests for Phase 402: food_effect_predictor.py"""

import pytest

from omega_pbpk.prediction.food_effect_predictor import (
    FoodEffectResult,
    compare_food_states,
    predict_food_effect,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bcs2_high_fat(**kwargs):
    """BCS II drug + high fat meal — canonical food effect scenario."""
    defaults = dict(
        drug_name="GriseofulvinAnalog",
        logp=2.5,
        mw_da=352.8,
        dose_mg=500.0,
        solubility_fasted_mg_mL=0.01,  # low solubility → BCS II
        pka=7.0,
        drug_type="neutral",
        food_state="high_fat_meal",
    )
    defaults.update(kwargs)
    return predict_food_effect(**defaults)


def _bcs3_high_fat(**kwargs):
    """BCS III drug + high fat meal — hydrophilic, permeability-limited."""
    defaults = dict(
        drug_name="Atenolol",
        logp=-0.5,
        mw_da=266.3,
        dose_mg=100.0,
        solubility_fasted_mg_mL=10.0,  # high solubility, low logP → BCS III
        pka=9.6,
        drug_type="base",
        food_state="high_fat_meal",
    )
    defaults.update(kwargs)
    return predict_food_effect(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_food_effect_result():
    result = _bcs2_high_fat()
    assert isinstance(result, FoodEffectResult)


def test_result_frozen():
    result = _bcs2_high_fat()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Modified"  # type: ignore[misc]


def test_result_has_all_fields():
    result = _bcs2_high_fat()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "food_state")
    assert hasattr(result, "logp")
    assert hasattr(result, "bcs_class")
    assert hasattr(result, "fasted_cmax_mg_L")
    assert hasattr(result, "fed_cmax_mg_L")
    assert hasattr(result, "cmax_ratio_fed_fasted")
    assert hasattr(result, "fasted_auc_mg_h_per_L")
    assert hasattr(result, "fed_auc_mg_h_per_L")
    assert hasattr(result, "auc_ratio_fed_fasted")
    assert hasattr(result, "fasted_tmax_h")
    assert hasattr(result, "fed_tmax_h")
    assert hasattr(result, "gastric_emptying_effect")
    assert hasattr(result, "dissolution_effect")
    assert hasattr(result, "bile_salt_effect")
    assert hasattr(result, "clinical_food_recommendation")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# BCS classification
# ---------------------------------------------------------------------------


def test_bcs_class_ii_low_solubility():
    result = _bcs2_high_fat()
    assert result.bcs_class == "II"


def test_bcs_class_iii_high_solubility_low_logp():
    result = _bcs3_high_fat()
    assert result.bcs_class == "III"


def test_bcs_class_i_high_sol_high_perm():
    result = predict_food_effect(
        drug_name="MetforminAnalog",
        logp=1.0,
        mw_da=165.6,
        dose_mg=100.0,
        solubility_fasted_mg_mL=5.0,  # high sol → Do < 1
        pka=12.4,
        drug_type="base",
        food_state="high_fat_meal",
    )
    assert result.bcs_class == "I"


def test_bcs_class_override():
    result = predict_food_effect(
        drug_name="TestDrug",
        logp=2.0,
        mw_da=300.0,
        dose_mg=100.0,
        solubility_fasted_mg_mL=5.0,
        pka=7.0,
        drug_type="neutral",
        bcs_class="IV",
        food_state="high_fat_meal",
    )
    assert result.bcs_class == "IV"


# ---------------------------------------------------------------------------
# BCS II + high fat increases AUC
# ---------------------------------------------------------------------------


def test_bcs2_high_fat_increases_auc():
    result = _bcs2_high_fat()
    assert result.auc_ratio_fed_fasted > 1.0
    assert result.fed_auc_mg_h_per_L > result.fasted_auc_mg_h_per_L


def test_bcs2_high_fat_auc_ratio_correct():
    result = _bcs2_high_fat()
    # Expected ratio from lookup table
    assert abs(result.auc_ratio_fed_fasted - 2.0) < 1e-9


def test_bcs2_high_fat_cmax_ratio_correct():
    result = _bcs2_high_fat()
    assert abs(result.cmax_ratio_fed_fasted - 1.8) < 1e-9


# ---------------------------------------------------------------------------
# BCS III + high fat reduces AUC
# ---------------------------------------------------------------------------


def test_bcs3_high_fat_reduces_auc():
    result = _bcs3_high_fat()
    assert result.auc_ratio_fed_fasted < 1.0
    assert result.fed_auc_mg_h_per_L < result.fasted_auc_mg_h_per_L


def test_bcs3_high_fat_auc_ratio_correct():
    result = _bcs3_high_fat()
    assert abs(result.auc_ratio_fed_fasted - 0.85) < 1e-9


def test_bcs3_high_fat_cmax_ratio_correct():
    result = _bcs3_high_fat()
    assert abs(result.cmax_ratio_fed_fasted - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Tmax delayed for fat meals
# ---------------------------------------------------------------------------


def test_tmax_delayed_high_fat():
    result = _bcs2_high_fat(food_state="high_fat_meal")
    assert result.fed_tmax_h > result.fasted_tmax_h
    assert abs(result.fed_tmax_h / result.fasted_tmax_h - 1.5) < 1e-9


def test_tmax_delayed_low_fat():
    result = _bcs2_high_fat(food_state="low_fat_meal")
    assert result.fed_tmax_h > result.fasted_tmax_h
    assert abs(result.fed_tmax_h / result.fasted_tmax_h - 1.2) < 1e-9


def test_tmax_unchanged_grapefruit():
    result = _bcs2_high_fat(food_state="grapefruit")
    assert abs(result.fed_tmax_h - result.fasted_tmax_h) < 1e-9


# ---------------------------------------------------------------------------
# Grapefruit: no AUC effect for BCS I
# ---------------------------------------------------------------------------


def test_bcs1_grapefruit_auc_ratio():
    result = predict_food_effect(
        drug_name="BCS1Drug",
        logp=1.5,
        mw_da=200.0,
        dose_mg=50.0,
        solubility_fasted_mg_mL=5.0,
        pka=7.0,
        drug_type="neutral",
        bcs_class="I",
        food_state="grapefruit",
    )
    assert abs(result.auc_ratio_fed_fasted - 1.5) < 1e-9  # from lookup table


# ---------------------------------------------------------------------------
# Gastric emptying, dissolution, and bile salt effects
# ---------------------------------------------------------------------------


def test_gastric_emptying_delayed_high_fat():
    result = _bcs2_high_fat(food_state="high_fat_meal")
    assert result.gastric_emptying_effect == "delayed"


def test_gastric_emptying_unchanged_grapefruit():
    result = _bcs2_high_fat(food_state="grapefruit")
    assert result.gastric_emptying_effect == "unchanged"


def test_dissolution_enhanced_bcs2_high_fat():
    result = _bcs2_high_fat(food_state="high_fat_meal")
    assert result.dissolution_effect == "enhanced"


def test_dissolution_unchanged_bcs1_high_fat():
    result = predict_food_effect(
        drug_name="BCS1Drug",
        logp=1.5,
        mw_da=200.0,
        dose_mg=50.0,
        solubility_fasted_mg_mL=5.0,
        pka=7.0,
        drug_type="neutral",
        bcs_class="I",
        food_state="high_fat_meal",
    )
    assert result.dissolution_effect == "unchanged"


def test_bile_salt_enhanced_high_fat():
    result = _bcs2_high_fat(food_state="high_fat_meal")
    assert result.bile_salt_effect == "enhanced"


def test_bile_salt_unchanged_low_fat():
    result = _bcs2_high_fat(food_state="low_fat_meal")
    assert result.bile_salt_effect == "unchanged"


# ---------------------------------------------------------------------------
# Clinical food recommendation
# ---------------------------------------------------------------------------


def test_bcs2_high_fat_with_food_recommendation():
    result = _bcs2_high_fat(food_state="high_fat_meal")
    # BCS II + AUC ratio 2.0 > 1.5 → with_food
    assert result.clinical_food_recommendation == "with_food"


def test_bcs3_without_food_recommendation():
    result = _bcs3_high_fat(food_state="high_fat_meal")
    # AUC ratio 0.85 → not < 0.8; but check recommendation
    # 0.85 >= 0.8, cmax_ratio 0.7 < 2.0, AUC < 1.5 for BCS III → either
    assert result.clinical_food_recommendation in ("without_food", "either")


def test_valid_recommendation_values():
    valid_recs = {"with_food", "without_food", "either", "avoid_fat"}
    for bcs in ("I", "II", "III", "IV"):
        for fs in ("high_fat_meal", "low_fat_meal", "grapefruit"):
            result = predict_food_effect(
                drug_name="Drug",
                logp=2.0,
                mw_da=300.0,
                dose_mg=100.0,
                solubility_fasted_mg_mL=0.01,
                pka=7.0,
                drug_type="neutral",
                bcs_class=bcs,
                food_state=fs,
            )
            assert result.clinical_food_recommendation in valid_recs


# ---------------------------------------------------------------------------
# compare_food_states
# ---------------------------------------------------------------------------


def test_compare_returns_list_of_three():
    results = compare_food_states(
        drug_name="GriseofulvinAnalog",
        logp=2.5,
        mw_da=352.8,
        dose_mg=500.0,
        solubility_fasted_mg_mL=0.01,
        pka=7.0,
        drug_type="neutral",
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_auc_ratio_descending():
    results = compare_food_states(
        drug_name="GriseofulvinAnalog",
        logp=2.5,
        mw_da=352.8,
        dose_mg=500.0,
        solubility_fasted_mg_mL=0.01,
        pka=7.0,
        drug_type="neutral",
    )
    ratios = [r.auc_ratio_fed_fasted for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_all_food_states_represented():
    results = compare_food_states(
        drug_name="TestDrug",
        logp=1.0,
        mw_da=300.0,
        dose_mg=100.0,
        solubility_fasted_mg_mL=5.0,
        pka=7.0,
        drug_type="neutral",
    )
    food_states = {r.food_state for r in results}
    assert food_states == {"high_fat_meal", "low_fat_meal", "grapefruit"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_logp_out_of_range_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_food_effect(
            drug_name="Drug", logp=15.0, mw_da=300.0, dose_mg=100.0,
            solubility_fasted_mg_mL=1.0, pka=7.0, drug_type="neutral",
            food_state="high_fat_meal",
        )


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_da"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=0.0, dose_mg=100.0,
            solubility_fasted_mg_mL=1.0, pka=7.0, drug_type="neutral",
            food_state="high_fat_meal",
        )


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=300.0, dose_mg=-1.0,
            solubility_fasted_mg_mL=1.0, pka=7.0, drug_type="neutral",
            food_state="high_fat_meal",
        )


def test_solubility_zero_raises():
    with pytest.raises(ValueError, match="solubility"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=300.0, dose_mg=100.0,
            solubility_fasted_mg_mL=0.0, pka=7.0, drug_type="neutral",
            food_state="high_fat_meal",
        )


def test_invalid_drug_type_raises():
    with pytest.raises(ValueError, match="drug_type"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=300.0, dose_mg=100.0,
            solubility_fasted_mg_mL=1.0, pka=7.0, drug_type="zwitterion",
            food_state="high_fat_meal",
        )


def test_invalid_food_state_raises():
    with pytest.raises(ValueError, match="food_state"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=300.0, dose_mg=100.0,
            solubility_fasted_mg_mL=1.0, pka=7.0, drug_type="neutral",
            food_state="fasted",
        )


def test_pka_out_of_range_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_food_effect(
            drug_name="Drug", logp=2.0, mw_da=300.0, dose_mg=100.0,
            solubility_fasted_mg_mL=1.0, pka=20.0, drug_type="neutral",
            food_state="high_fat_meal",
        )
