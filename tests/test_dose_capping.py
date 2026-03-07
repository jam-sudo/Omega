"""Tests for Phase 261: dose_capping.py"""

import pytest

from omega_pbpk.clinical.dose_capping import (
    compute_body_weights,
    recommend_dose_obese,
)

# ---------------------------------------------------------------------------
# compute_body_weights — validation
# ---------------------------------------------------------------------------


def test_tbw_zero_raises():
    with pytest.raises(ValueError, match="total_body_weight_kg must be > 0"):
        compute_body_weights(0.0, 175.0, "male")


def test_tbw_negative_raises():
    with pytest.raises(ValueError, match="total_body_weight_kg must be > 0"):
        compute_body_weights(-10.0, 175.0, "male")


def test_height_zero_raises():
    with pytest.raises(ValueError, match="height_cm must be > 0"):
        compute_body_weights(70.0, 0.0, "male")


def test_height_negative_raises():
    with pytest.raises(ValueError, match="height_cm must be > 0"):
        compute_body_weights(70.0, -5.0, "female")


def test_invalid_sex_raises():
    with pytest.raises(ValueError, match="sex must be 'male' or 'female'"):
        compute_body_weights(70.0, 175.0, "other")


# ---------------------------------------------------------------------------
# compute_body_weights — normal BMI (70 kg, 175 cm, male)
# ---------------------------------------------------------------------------


@pytest.fixture()
def normal_male():
    return compute_body_weights(70.0, 175.0, "male")


def test_normal_male_obesity_class(normal_male):
    assert normal_male.obesity_class == "normal"


def test_normal_male_bmi(normal_male):
    # 70 / 1.75^2 ≈ 22.9
    assert 22.0 < normal_male.bmi < 24.0


def test_normal_male_ibw_reasonable(normal_male):
    # Devine male: 50 + 2.3*(175/2.54 - 60) ≈ 68.9 kg
    assert 65.0 < normal_male.ibw_kg < 73.0


def test_normal_male_ibw_positive(normal_male):
    assert normal_male.ibw_kg > 0


def test_normal_male_abw_close_to_ibw(normal_male):
    # When TBW ≈ IBW, ABW ≈ IBW
    assert abs(normal_male.abw_kg - normal_male.ibw_kg) < 5.0


def test_normal_male_lbw_positive(normal_male):
    assert normal_male.lbw_kg > 0


# ---------------------------------------------------------------------------
# compute_body_weights — obese patient (120 kg, 170 cm, female)
# ---------------------------------------------------------------------------


@pytest.fixture()
def obese_female():
    return compute_body_weights(120.0, 170.0, "female")


def test_obese_female_obesity_class(obese_female):
    # BMI = 120 / 1.7^2 ≈ 41.5 → obese_III
    assert obese_female.obesity_class == "obese_III"


def test_obese_female_abw_less_than_tbw(obese_female):
    assert obese_female.abw_kg < obese_female.total_body_weight_kg


def test_obese_female_abw_greater_than_ibw(obese_female):
    assert obese_female.abw_kg >= obese_female.ibw_kg


def test_obese_female_lbw_positive(obese_female):
    assert obese_female.lbw_kg > 0


def test_obese_female_lbw_less_than_tbw(obese_female):
    assert obese_female.lbw_kg < obese_female.total_body_weight_kg


# ---------------------------------------------------------------------------
# Obesity class coverage
# ---------------------------------------------------------------------------


def test_overweight_class():
    # BMI ~27 → overweight
    bw = compute_body_weights(80.0, 172.0, "male")
    assert bw.obesity_class == "overweight"


def test_obese_i_class():
    # BMI ~32
    bw = compute_body_weights(95.0, 172.0, "male")
    assert bw.obesity_class == "obese_I"


def test_obese_ii_class():
    # BMI ~37
    bw = compute_body_weights(110.0, 172.0, "female")
    assert bw.obesity_class == "obese_II"


# ---------------------------------------------------------------------------
# recommend_dose_obese — validation
# ---------------------------------------------------------------------------


def test_mg_per_kg_zero_raises():
    with pytest.raises(ValueError, match="mg_per_kg must be > 0"):
        recommend_dose_obese("DrugX", 100.0, 170.0, "male", 0.0)


def test_mg_per_kg_negative_raises():
    with pytest.raises(ValueError, match="mg_per_kg must be > 0"):
        recommend_dose_obese("DrugX", 100.0, 170.0, "male", -1.0)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="dosing_weight_strategy must be one of"):
        recommend_dose_obese("DrugX", 100.0, 170.0, "male", 1.0, dosing_weight_strategy="flat")


# ---------------------------------------------------------------------------
# recommend_dose_obese — ABW strategy bounds
# ---------------------------------------------------------------------------


def test_abw_dose_between_ibw_and_tbw_dose():
    """ABW dose should be between IBW dose and TBW dose for an obese patient."""
    drug = "Vancomycin"
    tbw, h, sex, mg_kg = 130.0, 170.0, "male", 15.0
    ibw_result = recommend_dose_obese(drug, tbw, h, sex, mg_kg, dosing_weight_strategy="ibw")
    abw_result = recommend_dose_obese(drug, tbw, h, sex, mg_kg, dosing_weight_strategy="abw")
    tbw_result = recommend_dose_obese(drug, tbw, h, sex, mg_kg, dosing_weight_strategy="tbw")
    assert (
        ibw_result.recommended_dose_mg
        <= abw_result.recommended_dose_mg
        <= tbw_result.recommended_dose_mg
    )


# ---------------------------------------------------------------------------
# recommend_dose_obese — dose cap
# ---------------------------------------------------------------------------


def test_cap_applied():
    result = recommend_dose_obese(
        "DrugX", 130.0, 175.0, "male", 10.0, dosing_weight_strategy="tbw", max_dose_mg=500.0
    )
    assert result.dose_cap_applied is True
    assert result.recommended_dose_mg == 500.0


def test_cap_not_applied_when_under():
    result = recommend_dose_obese(
        "DrugX", 70.0, 175.0, "male", 1.0, dosing_weight_strategy="tbw", max_dose_mg=500.0
    )
    assert result.dose_cap_applied is False
    assert result.recommended_dose_mg < 500.0


def test_no_cap_no_capping():
    result = recommend_dose_obese("DrugX", 130.0, 175.0, "male", 10.0, max_dose_mg=None)
    assert result.dose_cap_applied is False


# ---------------------------------------------------------------------------
# recommend_dose_obese — different strategies give different weights
# ---------------------------------------------------------------------------


def test_different_strategies_give_different_weights():
    tbw, h, sex = 130.0, 170.0, "female"
    weights = {}
    for strat in ("ibw", "abw", "lbw", "tbw"):
        r = recommend_dose_obese("D", tbw, h, sex, 1.0, dosing_weight_strategy=strat)
        weights[strat] = r.dosing_weight_kg
    # TBW must be the largest; IBW the smallest (for obese patient)
    assert weights["tbw"] == tbw
    assert weights["ibw"] < weights["abw"] < weights["tbw"]


def test_recommended_dose_positive():
    result = recommend_dose_obese("DrugA", 90.0, 175.0, "male", 5.0)
    assert result.recommended_dose_mg > 0


def test_dosing_weight_kg_matches_strategy():
    bw = compute_body_weights(120.0, 170.0, "female")
    result = recommend_dose_obese(
        "DrugA", 120.0, 170.0, "female", 2.0, dosing_weight_strategy="lbw"
    )
    assert abs(result.dosing_weight_kg - bw.lbw_kg) < 1e-6
