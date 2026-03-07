"""Tests for omega_pbpk.clinical.food_drug_interaction — Phase 509."""

import pytest

from omega_pbpk.clinical.food_drug_interaction import (
    FoodEffectResult,
    FoodInteractionSimResult,
    fat_meal_enhancement,
    gastric_ph_effect,
    predict_food_effect,
    simulate_fasted_vs_fed,
)

VALID_CATEGORIES = {"no_effect", "moderate", "high"}


# ---------------------------------------------------------------------------
# fat_meal_enhancement
# ---------------------------------------------------------------------------


def test_fat_meal_enhancement_high_logp_gt_low_logp():
    """High logP → bigger fat-meal fold change than low logP."""
    fold_high = fat_meal_enhancement(5.0, 0.1)
    fold_low = fat_meal_enhancement(0.5, 0.1)
    assert fold_high > fold_low


def test_fat_meal_enhancement_low_logp_returns_one():
    """logP < 1 → fold change == 1.0 (no fat-meal benefit)."""
    fold = fat_meal_enhancement(0.0, 1.0)
    assert fold == pytest.approx(1.0)


def test_fat_meal_enhancement_high_logp_above_one():
    """High logP (5) → fold > 1.0."""
    fold = fat_meal_enhancement(5.0, 0.5)
    assert fold > 1.0


def test_fat_meal_enhancement_invalid_solubility_raises():
    with pytest.raises(ValueError):
        fat_meal_enhancement(3.0, 0.0)


def test_fat_meal_enhancement_negative_solubility_raises():
    with pytest.raises(ValueError):
        fat_meal_enhancement(3.0, -0.1)


# ---------------------------------------------------------------------------
# gastric_ph_effect
# ---------------------------------------------------------------------------


def test_gastric_ph_effect_basic_drug_decreases_absorption():
    """Basic drug pKa=8 → fed pH reduces absorption (solubility_ratio < 1)."""
    result = gastric_ph_effect(pka=8.0, drug_type="base")
    assert result["absorption_impact"] in ("decreased", "unchanged")
    # For basic drug: less ionized at higher pH → lower solubility ratio
    assert result["solubility_ratio_fed_fasted"] <= 1.0


def test_gastric_ph_effect_acid_drug_increases_absorption():
    """Acid drug pKa=4 → fed pH increases ionisation → higher solubility ratio."""
    result = gastric_ph_effect(pka=4.0, drug_type="acid")
    assert result["solubility_ratio_fed_fasted"] >= 1.0


def test_gastric_ph_effect_neutral_unchanged():
    """Neutral drug → absorption impact unchanged, ratio ~ 1."""
    result = gastric_ph_effect(pka=7.0, drug_type="neutral")
    assert result["absorption_impact"] == "unchanged"
    assert result["solubility_ratio_fed_fasted"] == pytest.approx(1.0)


def test_gastric_ph_effect_returns_required_keys():
    """Check all expected keys are present."""
    result = gastric_ph_effect(pka=5.0, drug_type="acid")
    for key in (
        "fasted_ionized_fraction",
        "fed_ionized_fraction",
        "solubility_ratio_fed_fasted",
        "absorption_impact",
    ):
        assert key in result


def test_gastric_ph_effect_ionized_fractions_in_range():
    result = gastric_ph_effect(pka=8.0, drug_type="base")
    assert 0.0 <= result["fasted_ionized_fraction"] <= 1.0
    assert 0.0 <= result["fed_ionized_fraction"] <= 1.0


def test_gastric_ph_effect_invalid_drug_type_raises():
    with pytest.raises(ValueError):
        gastric_ph_effect(pka=7.0, drug_type="zwitterion")


def test_gastric_ph_effect_negative_ph_raises():
    with pytest.raises(ValueError):
        gastric_ph_effect(pka=7.0, drug_type="neutral", fasted_ph=-1.0)


# ---------------------------------------------------------------------------
# predict_food_effect
# ---------------------------------------------------------------------------


def test_predict_food_effect_returns_food_effect_result():
    result = predict_food_effect("ibuprofen", logP=3.5, mw=206.0)
    assert isinstance(result, FoodEffectResult)


def test_predict_food_effect_high_logp_auc_ratio_gt_one():
    """High logP (>3) → food enhances absorption (AUC ratio > 1)."""
    result = predict_food_effect("lipophilic", logP=5.0, mw=350.0)
    assert result.auc_ratio_fed_fasted > 1.0


def test_predict_food_effect_low_logp_neutral_close_to_one():
    """Low logP, neutral → AUC ratio close to 1.0 (no fat enhancement)."""
    result = predict_food_effect("metformin", logP=-1.0, mw=165.0, drug_type="neutral")
    assert abs(result.auc_ratio_fed_fasted - 1.0) < 0.15


def test_predict_food_effect_basic_drug_auc_ratio_below_one():
    """Basic drug (pKa 8) with low logP → fed pH reduces absorption → AUC ratio < 1.

    logP=0.5 ensures fat_fold=1.0 (no fat enhancement), so pH effect dominates.
    """
    result = predict_food_effect("basic_drug", logP=0.5, mw=300.0, pka=8.0, drug_type="base")
    assert result.auc_ratio_fed_fasted < 1.0


def test_predict_food_effect_high_logp_category_high():
    """High logP (5) + fat enhancement → category 'high'."""
    result = predict_food_effect("fat_absorbed", logP=5.0, mw=400.0)
    assert result.food_effect_category == "high"


def test_predict_food_effect_category_in_valid_set():
    result = predict_food_effect("drug_x", logP=2.5, mw=250.0)
    assert result.food_effect_category in VALID_CATEGORIES


def test_predict_food_effect_tmax_difference_positive():
    """Fed Tmax is always delayed vs fasted → tmax_difference_h > 0."""
    result = predict_food_effect("compound", logP=3.0, mw=300.0)
    assert result.tmax_difference_h > 0.0


def test_predict_food_effect_drug_name_preserved():
    result = predict_food_effect("aspirin", logP=1.2, mw=180.0)
    assert result.drug_name == "aspirin"


def test_predict_food_effect_logp_preserved():
    result = predict_food_effect("x", logP=4.2, mw=300.0)
    assert result.logP == pytest.approx(4.2)


def test_predict_food_effect_cmax_ratio_less_than_auc_ratio():
    """Cmax ratio should be <= AUC ratio (delayed absorption dampens Cmax)."""
    result = predict_food_effect("drug_y", logP=4.0, mw=350.0)
    assert result.cmax_ratio_fed_fasted <= result.auc_ratio_fed_fasted + 1e-9


def test_predict_food_effect_recommendation_is_string():
    result = predict_food_effect("drug_z", logP=3.0, mw=300.0)
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_predict_food_effect_invalid_mw_raises():
    with pytest.raises(ValueError):
        predict_food_effect("x", logP=3.0, mw=0.0)


def test_predict_food_effect_invalid_solubility_raises():
    with pytest.raises(ValueError):
        predict_food_effect("x", logP=3.0, mw=300.0, solubility_mg_mL=0.0)


def test_predict_food_effect_invalid_cl_raises():
    with pytest.raises(ValueError):
        predict_food_effect("x", logP=3.0, mw=300.0, cl_L_per_h=0.0)


def test_predict_food_effect_invalid_vd_raises():
    with pytest.raises(ValueError):
        predict_food_effect("x", logP=3.0, mw=300.0, vd_L=0.0)


def test_predict_food_effect_invalid_dose_raises():
    with pytest.raises(ValueError):
        predict_food_effect("x", logP=3.0, mw=300.0, dose_mg=-5.0)


# ---------------------------------------------------------------------------
# simulate_fasted_vs_fed
# ---------------------------------------------------------------------------


def test_simulate_fasted_vs_fed_returns_correct_type():
    result = simulate_fasted_vs_fed(
        "ibuprofen",
        logP=3.5,
        mw=206.0,
        solubility_mg_mL=0.1,
        cl_L_per_h=5.0,
        vd_L=10.0,
        dose_mg=400.0,
    )
    assert isinstance(result, FoodInteractionSimResult)


def test_simulate_fasted_vs_fed_times_start_at_zero():
    result = simulate_fasted_vs_fed(
        "drug",
        logP=3.0,
        mw=300.0,
        solubility_mg_mL=1.0,
        cl_L_per_h=10.0,
        vd_L=70.0,
        dose_mg=100.0,
    )
    assert result.times_h[0] == pytest.approx(0.0)


def test_simulate_fasted_vs_fed_high_logp_fed_auc_gt_fasted():
    """High logP → fed AUC > fasted AUC."""
    result = simulate_fasted_vs_fed(
        "lipophilic",
        logP=5.0,
        mw=400.0,
        solubility_mg_mL=0.05,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dose_mg=100.0,
    )
    assert result.auc_fed > result.auc_fasted


def test_simulate_fasted_vs_fed_fed_cmax_occurs_later():
    """Fed Tmax should be later than fasted Tmax (delayed absorption)."""
    result = simulate_fasted_vs_fed(
        "ibuprofen",
        logP=3.5,
        mw=206.0,
        solubility_mg_mL=0.1,
        cl_L_per_h=5.0,
        vd_L=10.0,
        dose_mg=400.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    tmax_fasted = result.times_h[result.c_fasted_mg_L.index(result.cmax_fasted)]
    tmax_fed = result.times_h[result.c_fed_mg_L.index(result.cmax_fed)]
    assert tmax_fed >= tmax_fasted


def test_simulate_fasted_vs_fed_concentrations_nonnegative():
    result = simulate_fasted_vs_fed(
        "drug",
        logP=2.0,
        mw=250.0,
        solubility_mg_mL=1.0,
        cl_L_per_h=10.0,
        vd_L=70.0,
        dose_mg=100.0,
    )
    assert all(c >= -1e-9 for c in result.c_fasted_mg_L)
    assert all(c >= -1e-9 for c in result.c_fed_mg_L)


def test_simulate_fasted_vs_fed_aucs_positive():
    result = simulate_fasted_vs_fed(
        "drug",
        logP=2.0,
        mw=250.0,
        solubility_mg_mL=1.0,
        cl_L_per_h=10.0,
        vd_L=70.0,
        dose_mg=100.0,
    )
    assert result.auc_fasted > 0.0
    assert result.auc_fed > 0.0


def test_simulate_fasted_vs_fed_cmaxes_positive():
    result = simulate_fasted_vs_fed(
        "drug",
        logP=2.0,
        mw=250.0,
        solubility_mg_mL=1.0,
        cl_L_per_h=10.0,
        vd_L=70.0,
        dose_mg=100.0,
    )
    assert result.cmax_fasted > 0.0
    assert result.cmax_fed > 0.0


def test_simulate_fasted_vs_fed_invalid_mw_raises():
    with pytest.raises(ValueError):
        simulate_fasted_vs_fed(
            "x",
            logP=3.0,
            mw=0.0,
            solubility_mg_mL=1.0,
            cl_L_per_h=10.0,
            vd_L=70.0,
            dose_mg=100.0,
        )


def test_simulate_fasted_vs_fed_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_fasted_vs_fed(
            "x",
            logP=3.0,
            mw=300.0,
            solubility_mg_mL=1.0,
            cl_L_per_h=-1.0,
            vd_L=70.0,
            dose_mg=100.0,
        )
