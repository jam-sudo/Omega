"""Tests for Phase 182 — meal_effect_pk."""

import pytest

from omega_pbpk.clinical.meal_effect_pk import (
    MealEffectResult,
    assess_meal_effect,
    screen_food_timing,
)

_VALID_FOOD_EFFECT_CLASSES = {"none", "minor", "moderate", "significant"}


# ---------------------------------------------------------------------------
# Basic return-type and field-preservation tests
# ---------------------------------------------------------------------------


def test_assess_meal_effect_returns_dataclass():
    result = assess_meal_effect("DrugA", bcs_class=1, logP=1.0)
    assert isinstance(result, MealEffectResult)


def test_drug_name_preserved():
    result = assess_meal_effect("Ibuprofen", bcs_class=2, logP=3.5)
    assert result.drug_name == "Ibuprofen"


def test_bcs_class_preserved():
    result = assess_meal_effect("D", bcs_class=3, logP=0.0)
    assert result.bcs_class == 3


def test_logp_preserved():
    result = assess_meal_effect("D", bcs_class=1, logP=2.5)
    assert result.logP == pytest.approx(2.5)


def test_fed_state_default():
    result = assess_meal_effect("D", bcs_class=1, logP=1.0)
    assert result.fed_state == "fed"


def test_fed_state_custom():
    result = assess_meal_effect("D", bcs_class=1, logP=1.0, fed_state="high-fat meal")
    assert result.fed_state == "high-fat meal"


# ---------------------------------------------------------------------------
# BCS-class-specific behaviour
# ---------------------------------------------------------------------------


def test_bcs1_food_effect_none_or_minor():
    result = assess_meal_effect("D", bcs_class=1, logP=1.0)
    assert result.food_effect_class in {"none", "minor"}


def test_bcs1_auc_ratio_near_one():
    result = assess_meal_effect("D", bcs_class=1, logP=1.0)
    assert result.auc_ratio_fed_fasted == pytest.approx(1.0, abs=0.2)


def test_bcs2_lipophilic_auc_ratio_above_one():
    """Lipophilic BCS II drugs (logP > 2) → food increases AUC."""
    result = assess_meal_effect("D", bcs_class=2, logP=4.0)
    assert result.auc_ratio_fed_fasted > 1.0


def test_bcs2_high_logp_cmax_ratio_above_one():
    result = assess_meal_effect("D", bcs_class=2, logP=5.0)
    assert result.cmax_ratio_fed_fasted > 1.0


def test_bcs2_low_logp_food_effect():
    """BCS II with logP <= 2 still has some food effect but smaller."""
    result = assess_meal_effect("D", bcs_class=2, logP=1.0)
    assert result.auc_ratio_fed_fasted >= 1.0


def test_bcs3_auc_ratio_at_or_below_one():
    """BCS III: food generally reduces absorption."""
    result = assess_meal_effect("D", bcs_class=3, logP=0.5)
    assert result.auc_ratio_fed_fasted <= 1.0


def test_bcs4_positive_logp_benefit():
    """BCS IV with logP > 0: food can improve absorption."""
    result = assess_meal_effect("D", bcs_class=4, logP=3.0)
    assert result.auc_ratio_fed_fasted >= 1.0


# ---------------------------------------------------------------------------
# General property tests
# ---------------------------------------------------------------------------


def test_auc_ratio_positive():
    for bcs in [1, 2, 3, 4]:
        result = assess_meal_effect("D", bcs_class=bcs, logP=1.5)
        assert result.auc_ratio_fed_fasted > 0.0


def test_cmax_ratio_positive():
    for bcs in [1, 2, 3, 4]:
        result = assess_meal_effect("D", bcs_class=bcs, logP=1.5)
        assert result.cmax_ratio_fed_fasted > 0.0


def test_food_effect_class_valid_values():
    for bcs in [1, 2, 3, 4]:
        for logp in [-1.0, 0.0, 2.0, 5.0]:
            result = assess_meal_effect("D", bcs_class=bcs, logP=logp)
            assert result.food_effect_class in _VALID_FOOD_EFFECT_CLASSES


def test_administration_recommendation_not_empty():
    result = assess_meal_effect("D", bcs_class=2, logP=4.0)
    assert len(result.administration_recommendation) > 0


def test_notes_not_empty():
    result = assess_meal_effect("D", bcs_class=1, logP=1.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_food_timing tests
# ---------------------------------------------------------------------------


def test_screen_food_timing_returns_correct_count():
    timings = [0.0, 1.0, 2.0, 4.0]
    results = screen_food_timing("D", bcs_class=2, logP=4.0, meal_timings_h=timings)
    assert len(results) == len(timings)


def test_screen_food_timing_all_dataclass():
    results = screen_food_timing("D", bcs_class=2, logP=3.0, meal_timings_h=[0.0, 2.0])
    for r in results:
        assert isinstance(r, MealEffectResult)


def test_screen_food_timing_with_food_label():
    results = screen_food_timing("D", bcs_class=2, logP=3.0, meal_timings_h=[0.0])
    assert results[0].fed_state == "with food"


def test_screen_food_timing_food_effect_decreases_with_time():
    """For BCS II drug: food effect on AUC should decrease as meal is further from dose."""
    results = screen_food_timing("D", bcs_class=2, logP=5.0, meal_timings_h=[0.0, 4.0])
    ratio_0h = results[0].auc_ratio_fed_fasted - 1.0
    ratio_4h = results[1].auc_ratio_fed_fasted - 1.0
    assert ratio_0h >= ratio_4h


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


def test_invalid_bcs_class_zero():
    with pytest.raises(ValueError, match="bcs_class"):
        assess_meal_effect("D", bcs_class=0, logP=1.0)


def test_invalid_bcs_class_five():
    with pytest.raises(ValueError, match="bcs_class"):
        assess_meal_effect("D", bcs_class=5, logP=1.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logP"):
        assess_meal_effect("D", bcs_class=1, logP=-15.0)


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logP"):
        assess_meal_effect("D", bcs_class=1, logP=20.0)
