"""
Tests for Phase 432 — Bioisostere Predictor
"""

import pytest

from omega_pbpk.prediction.bioisostere_predictor import (
    BioisostereResult,
    predict_bioisostere,
    suggest_bioisosteres,
)


# ------------------------------------------------------------------
# Return type
# ------------------------------------------------------------------
def test_return_type():
    result = predict_bioisostere("aspirin", "COOH", "tetrazole")
    assert isinstance(result, BioisostereResult)


# ------------------------------------------------------------------
# Field preservation
# ------------------------------------------------------------------
def test_drug_name_preserved():
    result = predict_bioisostere("ibuprofen", "OH", "F")
    assert result.drug_name == "ibuprofen"


def test_original_group_preserved():
    result = predict_bioisostere("drug_x", "COOH", "tetrazole")
    assert result.original_group == "COOH"


def test_replacement_group_preserved():
    result = predict_bioisostere("drug_x", "COOH", "tetrazole")
    assert result.replacement_group == "tetrazole"


# ------------------------------------------------------------------
# Known database values
# ------------------------------------------------------------------
def test_cooh_tetrazole_metabolic_major():
    result = predict_bioisostere("drug", "COOH", "tetrazole")
    assert result.metabolic_stability_improvement == "major"


def test_oh_f_permeability_increased():
    result = predict_bioisostere("drug", "OH", "F")
    assert result.permeability_impact == "increased"


def test_ester_amide_metabolic_major():
    result = predict_bioisostere("drug", "ester", "amide")
    assert result.metabolic_stability_improvement == "major"


def test_cooh_tetrazole_logp_delta():
    result = predict_bioisostere("drug", "COOH", "tetrazole")
    assert result.logp_delta == 0.0


def test_oh_f_pka_delta():
    result = predict_bioisostere("drug", "OH", "F")
    assert result.pka_delta == -3.0


def test_ch3_cf3_logp_delta():
    result = predict_bioisostere("drug", "CH3", "CF3")
    assert result.logp_delta == 1.5


# ------------------------------------------------------------------
# predicted_f_oral_change > 0
# ------------------------------------------------------------------
def test_f_oral_change_positive():
    result = predict_bioisostere("drug", "COOH", "tetrazole")
    assert result.predicted_f_oral_change > 0.0


def test_f_oral_change_cooh_tetrazole_value():
    result = predict_bioisostere("drug", "COOH", "tetrazole")
    assert abs(result.predicted_f_oral_change - 1.3) < 1e-9


# ------------------------------------------------------------------
# Category correctly assigned
# ------------------------------------------------------------------
def test_category_non_classical_cooh_tetrazole():
    result = predict_bioisostere("drug", "COOH", "tetrazole")
    assert result.category == "non_classical"


def test_category_classical_oh_f():
    result = predict_bioisostere("drug", "OH", "F")
    assert result.category == "classical"


# ------------------------------------------------------------------
# suggest_bioisosteres
# ------------------------------------------------------------------
def test_suggest_multiple_groups():
    results = suggest_bioisosteres("drug", ["COOH", "OH"])
    orig_groups = {r.original_group for r in results}
    assert "COOH" in orig_groups
    assert "OH" in orig_groups


def test_suggest_sorted_descending():
    results = suggest_bioisosteres("drug", ["COOH", "OH", "ester", "CH3"])
    f_values = [r.predicted_f_oral_change for r in results]
    assert f_values == sorted(f_values, reverse=True)


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------
def test_invalid_pair_raises_value_error():
    with pytest.raises(ValueError):
        predict_bioisostere("drug", "COOH", "nonexistent_group")


def test_invalid_original_group_raises_value_error():
    with pytest.raises(ValueError):
        predict_bioisostere("drug", "NO_SUCH", "tetrazole")


# ------------------------------------------------------------------
# Empty functional groups
# ------------------------------------------------------------------
def test_empty_functional_groups_returns_empty():
    results = suggest_bioisosteres("drug", [])
    assert results == []
