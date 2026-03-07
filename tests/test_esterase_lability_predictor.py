"""Tests for Phase 219 — Esterase Lability Predictor."""

import math

import pytest

from omega_pbpk.prediction.esterase_lability_predictor import (
    EsteraseLabilityResult,
    compare_ester_types,
    predict_esterase_lability,
)

VALID_STABILITY_CLASSES = {"very_labile", "labile", "moderate", "stable"}
VALID_PRODRUG_SUITABILITY = {"excellent", "good", "poor"}


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_esterase_lability("DrugA", "methyl_ester")
    assert isinstance(result, EsteraseLabilityResult)


def test_field_drug_name():
    result = predict_esterase_lability("DrugX", "ethyl_ester")
    assert result.drug_name == "DrugX"


def test_field_ester_type():
    result = predict_esterase_lability("DrugX", "carbonate")
    assert result.ester_type == "carbonate"


def test_field_steric_bulk():
    result = predict_esterase_lability("DrugX", "ethyl_ester", steric_bulk="moderate")
    assert result.steric_bulk == "moderate"


def test_field_electronic_effect():
    result = predict_esterase_lability(
        "DrugX", "ethyl_ester", electronic_effect="electron_withdrawing"
    )
    assert result.electronic_effect == "electron_withdrawing"


def test_frozen_dataclass():
    result = predict_esterase_lability("DrugX", "methyl_ester")
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "NewName"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Relative ordering of ester types (default modifiers)
# ---------------------------------------------------------------------------


def test_phenyl_ester_faster_than_methyl_ester():
    phenyl = predict_esterase_lability("Drug", "phenyl_ester")
    methyl = predict_esterase_lability("Drug", "methyl_ester")
    assert phenyl.t_half_plasma_min < methyl.t_half_plasma_min


def test_carbamate_most_stable():
    carbamate = predict_esterase_lability("Drug", "carbamate")
    for ester in ["methyl_ester", "ethyl_ester", "phenyl_ester", "carbonate"]:
        other = predict_esterase_lability("Drug", ester)
        assert carbamate.t_half_plasma_min > other.t_half_plasma_min


def test_tert_butyl_ester_longer_than_ethyl():
    tert = predict_esterase_lability("Drug", "tert_butyl_ester")
    ethyl = predict_esterase_lability("Drug", "ethyl_ester")
    assert tert.t_half_plasma_min > ethyl.t_half_plasma_min


# ---------------------------------------------------------------------------
# Steric bulk modifiers
# ---------------------------------------------------------------------------


def test_high_steric_bulk_longer_than_none():
    high = predict_esterase_lability("Drug", "methyl_ester", steric_bulk="high")
    none_ = predict_esterase_lability("Drug", "methyl_ester", steric_bulk="none")
    assert high.t_half_plasma_min > none_.t_half_plasma_min


def test_moderate_steric_bulk_between_none_and_high():
    high = predict_esterase_lability("Drug", "methyl_ester", steric_bulk="high")
    moderate = predict_esterase_lability("Drug", "methyl_ester", steric_bulk="moderate")
    none_ = predict_esterase_lability("Drug", "methyl_ester", steric_bulk="none")
    assert none_.t_half_plasma_min < moderate.t_half_plasma_min < high.t_half_plasma_min


def test_steric_high_factor_is_ten():
    none_ = predict_esterase_lability("Drug", "ethyl_ester", steric_bulk="none")
    high = predict_esterase_lability("Drug", "ethyl_ester", steric_bulk="high")
    assert abs(high.t_half_plasma_min / none_.t_half_plasma_min - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Electronic effect modifiers
# ---------------------------------------------------------------------------


def test_electron_withdrawing_shorter_than_neutral():
    ew = predict_esterase_lability("Drug", "ethyl_ester", electronic_effect="electron_withdrawing")
    neutral = predict_esterase_lability("Drug", "ethyl_ester", electronic_effect="neutral")
    assert ew.t_half_plasma_min < neutral.t_half_plasma_min


def test_electron_donating_longer_than_neutral():
    ed = predict_esterase_lability("Drug", "ethyl_ester", electronic_effect="electron_donating")
    neutral = predict_esterase_lability("Drug", "ethyl_ester", electronic_effect="neutral")
    assert ed.t_half_plasma_min > neutral.t_half_plasma_min


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_liver_half_life_is_plasma_divided_by_five():
    result = predict_esterase_lability("Drug", "ethyl_ester")
    assert abs(result.t_half_liver_min - result.t_half_plasma_min / 5.0) < 1e-9


def test_k_plasma_consistent_with_t_half():
    result = predict_esterase_lability("Drug", "methyl_ester")
    expected_k = math.log(2) / result.t_half_plasma_min
    assert abs(result.k_plasma_per_min - expected_k) < 1e-9


def test_stability_class_valid_values():
    for ester in [
        "methyl_ester",
        "ethyl_ester",
        "tert_butyl_ester",
        "phenyl_ester",
        "carbamate",
        "carbonate",
    ]:
        result = predict_esterase_lability("Drug", ester)
        assert result.stability_class in VALID_STABILITY_CLASSES


def test_prodrug_suitability_valid_values():
    for ester in [
        "methyl_ester",
        "ethyl_ester",
        "tert_butyl_ester",
        "phenyl_ester",
        "carbamate",
        "carbonate",
    ]:
        result = predict_esterase_lability("Drug", ester)
        assert result.prodrug_suitability in VALID_PRODRUG_SUITABILITY


def test_phenyl_ester_very_labile():
    result = predict_esterase_lability("Drug", "phenyl_ester")
    assert result.stability_class == "very_labile"
    assert result.prodrug_suitability == "excellent"


def test_carbamate_stable():
    result = predict_esterase_lability("Drug", "carbamate")
    assert result.stability_class == "stable"
    assert result.prodrug_suitability == "poor"


# ---------------------------------------------------------------------------
# compare_ester_types
# ---------------------------------------------------------------------------


def test_compare_ester_types_returns_six():
    results = compare_ester_types("Drug")
    assert len(results) == 6


def test_compare_ester_types_sorted_descending():
    results = compare_ester_types("Drug")
    for i in range(len(results) - 1):
        assert results[i].t_half_plasma_min >= results[i + 1].t_half_plasma_min


def test_compare_ester_types_all_types_present():
    results = compare_ester_types("Drug")
    ester_types = {r.ester_type for r in results}
    expected = {
        "methyl_ester",
        "ethyl_ester",
        "tert_butyl_ester",
        "phenyl_ester",
        "carbamate",
        "carbonate",
    }
    assert ester_types == expected


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_ester_type_raises():
    with pytest.raises(ValueError, match="Invalid ester_type"):
        predict_esterase_lability("Drug", "unknown_ester")


def test_invalid_steric_bulk_raises():
    with pytest.raises(ValueError, match="Invalid steric_bulk"):
        predict_esterase_lability("Drug", "methyl_ester", steric_bulk="extreme")


def test_invalid_electronic_effect_raises():
    with pytest.raises(ValueError, match="Invalid electronic_effect"):
        predict_esterase_lability("Drug", "methyl_ester", electronic_effect="resonance")
