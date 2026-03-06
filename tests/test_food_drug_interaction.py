"""Tests for omega_pbpk.clinical.food_drug_interaction module."""

import pytest

from omega_pbpk.clinical.food_drug_interaction import (
    FoodEffectResult,
    food_effect_bcs_matrix,
    predict_food_effect,
)

VALID_CATEGORIES = {"positive", "negative", "neutral"}


# --- Basic result type and structure ---


def test_predict_food_effect_returns_food_effect_result():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert isinstance(result, FoodEffectResult)


def test_food_state_is_fed():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.food_state == "fed"


def test_drug_name_preserved():
    result = predict_food_effect("aspirin", logP=1.2)
    assert result.drug_name == "aspirin"


def test_drug_name_preserved_custom():
    result = predict_food_effect("griseofulvin", logP=2.18)
    assert result.drug_name == "griseofulvin"


# --- Category values ---


def test_food_effect_category_in_valid_set():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.food_effect_category in VALID_CATEGORIES


def test_high_logP_gives_positive_food_effect():
    """High logP (>4) lipophilic drugs should have positive food effect (AUC_ratio > 1.25)."""
    result = predict_food_effect("compound_x", logP=5.0)
    assert result.predicted_AUC_ratio > 1.25
    assert result.food_effect_category == "positive"


def test_low_logP_category_is_valid():
    result = predict_food_effect("metformin", logP=-1.5)
    assert result.food_effect_category in VALID_CATEGORIES


def test_moderate_logP_category_is_valid():
    result = predict_food_effect("moderate_drug", logP=2.0)
    assert result.food_effect_category in VALID_CATEGORIES


# --- Numeric field constraints ---


def test_cmax_ratio_positive():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.predicted_Cmax_ratio > 0


def test_bile_factor_at_least_one():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.bile_factor >= 1.0


def test_bile_factor_increases_with_logP():
    low = predict_food_effect("drug_a", logP=0.0)
    high = predict_food_effect("drug_b", logP=5.0)
    assert high.bile_factor > low.bile_factor


def test_motility_factor_between_zero_and_one():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert 0 < result.motility_factor < 1


def test_gastric_pH_positive():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.gastric_pH > 0


def test_solubility_mg_mL_positive():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert result.solubility_mg_mL > 0


def test_logP_preserved_in_result():
    result = predict_food_effect("drug_x", logP=4.2)
    assert result.logP == pytest.approx(4.2)


def test_pka_preserved_in_result():
    result = predict_food_effect("drug_x", logP=3.0, pka=5.5)
    assert result.pka == pytest.approx(5.5)


# --- ValueError for invalid inputs ---


def test_raises_value_error_for_zero_dose():
    with pytest.raises(ValueError):
        predict_food_effect("drug_x", logP=3.0, dose_mg=0)


def test_raises_value_error_for_negative_dose():
    with pytest.raises(ValueError):
        predict_food_effect("drug_x", logP=3.0, dose_mg=-10.0)


def test_raises_value_error_for_zero_solubility():
    with pytest.raises(ValueError):
        predict_food_effect("drug_x", logP=3.0, solubility_fasted_mg_mL=0)


def test_raises_value_error_for_negative_solubility():
    with pytest.raises(ValueError):
        predict_food_effect("drug_x", logP=3.0, solubility_fasted_mg_mL=-0.01)


# --- BCS matrix ---


def test_bcs_matrix_returns_dict_with_four_keys():
    matrix = food_effect_bcs_matrix("ibuprofen", logP=3.5)
    assert set(matrix.keys()) == {"I", "II", "III", "IV"}


def test_bcs_matrix_values_are_food_effect_results():
    matrix = food_effect_bcs_matrix("ibuprofen", logP=3.5)
    for key in ("I", "II", "III", "IV"):
        assert isinstance(matrix[key], FoodEffectResult)


def test_bcs_matrix_all_fed_state():
    matrix = food_effect_bcs_matrix("ibuprofen", logP=3.5)
    for key in ("I", "II", "III", "IV"):
        assert matrix[key].food_state == "fed"


def test_bcs_matrix_drug_name_preserved():
    matrix = food_effect_bcs_matrix("ketoconazole", logP=4.3)
    for key in ("I", "II", "III", "IV"):
        assert matrix[key].drug_name == "ketoconazole"


def test_bcs_matrix_categories_valid():
    matrix = food_effect_bcs_matrix("ibuprofen", logP=3.5)
    for key in ("I", "II", "III", "IV"):
        assert matrix[key].food_effect_category in VALID_CATEGORIES


def test_bcs_matrix_bcs_class_II_high_logP_positive():
    """BCS class II entry for high logP drug should show positive food effect."""
    matrix = food_effect_bcs_matrix("lipophilic_drug", logP=5.0)
    assert matrix["II"].predicted_AUC_ratio > 1.25
    assert matrix["II"].food_effect_category == "positive"


# --- Default parameters ---


def test_default_bcs_class_is_II():
    """Default bcs_class='II' should match explicit class II result for same inputs."""
    default_result = predict_food_effect("drug_x", logP=3.5)
    matrix = food_effect_bcs_matrix("drug_x", logP=3.5)
    assert default_result.predicted_AUC_ratio == pytest.approx(
        matrix["II"].predicted_AUC_ratio, rel=1e-5
    )


def test_mechanism_is_string():
    result = predict_food_effect("ibuprofen", logP=3.5)
    assert isinstance(result.mechanism, str)
    assert len(result.mechanism) > 0
