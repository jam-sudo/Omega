"""
Tests for Phase 982 — Drug Class Predictor (prediction/drug_class_predictor.py)
"""

import pytest

from omega_pbpk.prediction.drug_class_predictor import (
    DrugClassPredictionResult,
    predict_drug_class,
    screen_drug_classes,
)

_VALID_CLASSES = {
    "CNS",
    "cardiovascular",
    "antibiotic",
    "NSAID",
    "antifungal",
    "diuretic",
    "anticancer",
    "antihistamine",
    "other",
}

_VALID_BCS = {"I", "II", "III", "IV"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_drug_class_prediction_result():
    result = predict_drug_class("Ibuprofen", mw=206.3, logp=3.5)
    assert isinstance(result, DrugClassPredictionResult)


# ---------------------------------------------------------------------------
# predicted_class
# ---------------------------------------------------------------------------


def test_predicted_class_in_valid_classes():
    result = predict_drug_class("DrugA", mw=300.0, logp=2.0)
    assert result.predicted_class in _VALID_CLASSES


# ---------------------------------------------------------------------------
# class_probabilities
# ---------------------------------------------------------------------------


def test_class_probabilities_is_dict():
    result = predict_drug_class("DrugB", mw=300.0, logp=2.0)
    assert isinstance(result.class_probabilities, dict)


def test_class_probabilities_has_all_classes():
    result = predict_drug_class("DrugB", mw=300.0, logp=2.0)
    assert set(result.class_probabilities.keys()) == _VALID_CLASSES


def test_class_probabilities_sum_to_one():
    result = predict_drug_class("DrugB", mw=300.0, logp=2.0)
    total = sum(result.class_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# BCS class
# ---------------------------------------------------------------------------


def test_bcs_class_valid():
    result = predict_drug_class("DrugC", mw=300.0, logp=2.0)
    assert result.bcs_class in _VALID_BCS


def test_bcs_class_I_for_low_logp_low_psa():
    result = predict_drug_class("DrugC", mw=300.0, logp=0.5, psa=60.0)
    assert result.bcs_class == "I"


def test_bcs_class_II_for_high_logp_low_psa():
    result = predict_drug_class("DrugD", mw=400.0, logp=4.0, psa=60.0)
    assert result.bcs_class == "II"


# ---------------------------------------------------------------------------
# Lipinski & Veber
# ---------------------------------------------------------------------------


def test_lipinski_pass_bool():
    result = predict_drug_class("DrugE", mw=300.0, logp=2.0)
    assert isinstance(result.lipinski_pass, bool)


def test_veber_pass_bool():
    result = predict_drug_class("DrugF", mw=300.0, logp=2.0)
    assert isinstance(result.veber_pass, bool)


def test_lipinski_pass_for_drug_like():
    # MW<500, logP<5, HBD=2<5, HBA=4<10 => should pass
    result = predict_drug_class(
        "Ibuprofen", mw=206.3, logp=3.5, hbd=1, hba=2
    )
    assert result.lipinski_pass is True


def test_lipinski_fail_for_large_mw():
    result = predict_drug_class("BigDrug", mw=600.0, logp=2.0, hbd=1, hba=2)
    assert result.lipinski_pass is False


# ---------------------------------------------------------------------------
# drug_likeness_score
# ---------------------------------------------------------------------------


def test_drug_likeness_score_in_range():
    result = predict_drug_class("DrugG", mw=300.0, logp=2.0)
    assert 0.0 <= result.drug_likeness_score <= 100.0


def test_drug_likeness_score_100_for_perfect():
    # Lipinski + Veber + logP 0-5 all pass
    result = predict_drug_class(
        "DrugH",
        mw=250.0,
        logp=2.0,
        hbd=1,
        hba=3,
        psa=70.0,
        rotbonds=5,
    )
    assert result.drug_likeness_score == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Class prediction logic
# ---------------------------------------------------------------------------


def test_acidic_low_pka_nsaid_profile():
    # Low pKa + logP 1-4 + small MW → NSAID or diuretic score high
    result = predict_drug_class(
        "Aspirin",
        mw=180.0,
        logp=1.2,
        pka=3.5,
        ionization_type="acid",
        psa=60.0,
        hbd=1,
        hba=3,
    )
    # NSAID or diuretic should have highest score
    top_classes = sorted(
        result.class_probabilities, key=lambda k: result.class_probabilities[k], reverse=True
    )
    assert top_classes[0] in {"NSAID", "diuretic"}


def test_cns_profile_predicted():
    # CNS: logP 2.5 (1-4), MW 300 (150-400), PSA 60 (<90), basic
    result = predict_drug_class(
        "CNSDrug",
        mw=300.0,
        logp=2.5,
        pka=8.5,
        ionization_type="base",
        psa=60.0,
        hbd=1,
        hba=3,
    )
    assert result.predicted_class == "CNS"


# ---------------------------------------------------------------------------
# screen_drug_classes
# ---------------------------------------------------------------------------


def test_screen_drug_classes_returns_list():
    compounds = [
        {"drug_name": "A", "mw": 300.0, "logp": 2.0},
        {"drug_name": "B", "mw": 250.0, "logp": 1.5},
    ]
    results = screen_drug_classes(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_drug_classes_sorted_by_drug_likeness():
    compounds = [
        {"drug_name": "A", "mw": 600.0, "logp": 6.0, "hbd": 6, "hba": 12, "rotbonds": 12},
        {"drug_name": "B", "mw": 250.0, "logp": 2.0, "hbd": 1, "hba": 3, "rotbonds": 4},
        {"drug_name": "C", "mw": 700.0, "logp": 7.0, "hbd": 8, "hba": 15, "rotbonds": 15},
    ]
    results = screen_drug_classes(compounds)
    scores = [r.drug_likeness_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_drug_class("Drug", mw=0.0, logp=2.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_drug_class("Drug", mw=-100.0, logp=2.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_drug_class("Drug", mw=300.0, logp=2.0, ionization_type="zwitterion")
