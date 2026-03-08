"""Tests for Phase 416 — cocrystal_solubility.py"""

import pytest

from omega_pbpk.prediction.cocrystal_solubility import (
    CocrystalSolubilityResult,
    predict_cocrystal_solubility,
    screen_coformers,
)

# --- Fixtures ---


def base_result():
    return predict_cocrystal_solubility(
        drug_name="DrugA",
        coformer_name="CoformerB",
        drug_pka=4.5,
        coformer_pka=9.0,
        drug_solubility_mg_mL=0.05,
        coformer_solubility_mg_mL=50.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )


# --- Return type and field preservation ---


def test_return_type():
    result = base_result()
    assert isinstance(result, CocrystalSolubilityResult)


def test_drug_name_preserved():
    result = base_result()
    assert result.drug_name == "DrugA"


def test_coformer_name_preserved():
    result = base_result()
    assert result.coformer_name == "CoformerB"


def test_drug_pka_preserved():
    result = base_result()
    assert result.drug_pka == 4.5


def test_drug_solubility_preserved():
    result = base_result()
    assert result.drug_solubility_mg_mL == 0.05


# --- Enhancement ratio ---


def test_enhancement_ratio_at_least_one():
    result = base_result()
    assert result.enhancement_ratio >= 1.0


def test_cocrystal_solubility_ge_drug_solubility():
    result = base_result()
    assert result.cocrystal_solubility_mg_mL >= result.drug_solubility_mg_mL


def test_enhancement_capped_at_100():
    # Extreme solubility ratio should cap at 100x
    result = predict_cocrystal_solubility(
        drug_name="PoorDrug",
        coformer_name="HighSolCoformer",
        drug_pka=4.0,
        coformer_pka=10.0,
        drug_solubility_mg_mL=0.001,
        coformer_solubility_mg_mL=10000.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    assert result.enhancement_ratio <= 100.0


def test_acid_base_higher_than_neutral_neutral():
    r_salt = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.0,
        coformer_pka=8.0,
        drug_solubility_mg_mL=0.1,
        coformer_solubility_mg_mL=10.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    r_neutral = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.0,
        coformer_pka=6.0,
        drug_solubility_mg_mL=0.1,
        coformer_solubility_mg_mL=10.0,
        ionization_type_drug="neutral",
        ionization_type_coformer="neutral",
        temperature_C=25.0,
    )
    assert r_salt.enhancement_ratio > r_neutral.enhancement_ratio


# --- Temperature effect ---


def test_higher_temperature_higher_solubility():
    r25 = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.5,
        coformer_pka=9.0,
        drug_solubility_mg_mL=0.05,
        coformer_solubility_mg_mL=50.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    r37 = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.5,
        coformer_pka=9.0,
        drug_solubility_mg_mL=0.05,
        coformer_solubility_mg_mL=50.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=37.0,
    )
    assert r37.cocrystal_solubility_mg_mL > r25.cocrystal_solubility_mg_mL


# --- Eutectic pH ---


def test_eutectic_ph_acid_drug():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.0,
        coformer_pka=8.0,
        drug_solubility_mg_mL=0.1,
        coformer_solubility_mg_mL=10.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    # For acid drug: eutectic_ph = (drug_pka + coformer_pka) / 2
    expected = (4.0 + 8.0) / 2.0
    assert result.eutectic_ph == pytest.approx(expected, rel=1e-6)


# --- Classification ---


def test_solubility_class_poor():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=5.0,
        coformer_pka=5.0,
        drug_solubility_mg_mL=0.05,
        coformer_solubility_mg_mL=0.06,
        ionization_type_drug="acid",
        ionization_type_coformer="acid",
        temperature_C=25.0,
    )
    assert result.solubility_class == "poor"


def test_solubility_class_good():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.5,
        coformer_pka=9.0,
        drug_solubility_mg_mL=0.5,
        coformer_solubility_mg_mL=100.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    assert result.solubility_class == "good"


def test_dissolution_improvement_none():
    # Same type, equal solubilities -> enhancement = 1.0 -> "none"
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=5.0,
        coformer_pka=5.0,
        drug_solubility_mg_mL=1.0,
        coformer_solubility_mg_mL=1.0,
        ionization_type_drug="acid",
        ionization_type_coformer="acid",
        temperature_C=25.0,
    )
    assert result.dissolution_improvement == "none"


def test_dissolution_improvement_major():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.0,
        coformer_pka=9.0,
        drug_solubility_mg_mL=0.001,
        coformer_solubility_mg_mL=100.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    assert result.dissolution_improvement == "major"


def test_cocrystal_class_acidic():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.0,
        coformer_pka=2.0,
        drug_solubility_mg_mL=0.1,
        coformer_solubility_mg_mL=5.0,
        ionization_type_drug="base",
        ionization_type_coformer="acid",
        temperature_C=25.0,
    )
    assert result.cocrystal_class == "acidic_coformer"


def test_cocrystal_class_basic():
    result = predict_cocrystal_solubility(
        drug_name="D",
        coformer_name="C",
        drug_pka=4.5,
        coformer_pka=9.5,
        drug_solubility_mg_mL=0.05,
        coformer_solubility_mg_mL=50.0,
        ionization_type_drug="acid",
        ionization_type_coformer="base",
        temperature_C=25.0,
    )
    assert result.cocrystal_class == "basic_coformer"


# --- screen_coformers ---


def test_screen_coformers_returns_list():
    coformers = [
        {"name": "CF1", "pka": 2.0, "solubility_mg_mL": 10.0, "ionization_type": "acid"},
        {"name": "CF2", "pka": 9.0, "solubility_mg_mL": 50.0, "ionization_type": "base"},
    ]
    results = screen_coformers("DrugA", 4.5, 0.05, "acid", coformers)
    assert isinstance(results, list)


def test_screen_coformers_correct_length():
    coformers = [
        {"name": "CF1", "pka": 2.0, "solubility_mg_mL": 10.0, "ionization_type": "acid"},
        {"name": "CF2", "pka": 9.0, "solubility_mg_mL": 50.0, "ionization_type": "base"},
        {"name": "CF3", "pka": 6.0, "solubility_mg_mL": 5.0, "ionization_type": "neutral"},
    ]
    results = screen_coformers("DrugA", 4.5, 0.05, "acid", coformers)
    assert len(results) == 3


def test_screen_coformers_sorted_descending():
    coformers = [
        {"name": "CF1", "pka": 2.0, "solubility_mg_mL": 1.0, "ionization_type": "acid"},
        {"name": "CF2", "pka": 9.5, "solubility_mg_mL": 200.0, "ionization_type": "base"},
        {"name": "CF3", "pka": 6.0, "solubility_mg_mL": 5.0, "ionization_type": "neutral"},
    ]
    results = screen_coformers("DrugA", 4.5, 0.05, "acid", coformers)
    for i in range(1, len(results)):
        assert results[i - 1].enhancement_ratio >= results[i].enhancement_ratio


# --- Validation errors ---


def test_validation_drug_solubility_zero():
    with pytest.raises(ValueError):
        predict_cocrystal_solubility("D", "C", 4.5, 9.0, 0.0, 10.0, "acid", "base")


def test_validation_coformer_solubility_zero():
    with pytest.raises(ValueError):
        predict_cocrystal_solubility("D", "C", 4.5, 9.0, 0.1, 0.0, "acid", "base")


def test_validation_coformer_solubility_negative():
    with pytest.raises(ValueError):
        predict_cocrystal_solubility("D", "C", 4.5, 9.0, 0.1, -5.0, "acid", "base")


def test_validation_temperature_too_low():
    with pytest.raises(ValueError):
        predict_cocrystal_solubility(
            "D", "C", 4.5, 9.0, 0.1, 10.0, "acid", "base", temperature_C=3.0
        )


def test_validation_temperature_too_high():
    with pytest.raises(ValueError):
        predict_cocrystal_solubility(
            "D", "C", 4.5, 9.0, 0.1, 10.0, "acid", "base", temperature_C=81.0
        )
