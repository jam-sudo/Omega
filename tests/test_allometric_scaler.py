"""Tests for prediction/allometric_scaler.py — Phase 762."""

import math

import pytest

from omega_pbpk.prediction.allometric_scaler import (
    AllometricScalingResult,
    multi_species_scaling,
    predict_human_pk_allometric,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_SPECIES_DATA = [{"species": "rat", "cl_mL_per_min_per_kg": 50.0, "vd_L_per_kg": 2.0}]

MULTI_SPECIES_DATA = [
    {"species": "mouse", "cl_mL_per_min_per_kg": 80.0, "vd_L_per_kg": 3.0},
    {"species": "rat", "cl_mL_per_min_per_kg": 50.0, "vd_L_per_kg": 2.0},
    {"species": "dog", "cl_mL_per_min_per_kg": 20.0, "vd_L_per_kg": 1.5},
]

TWO_SPECIES_DATA = [
    {"species": "rat", "cl_mL_per_min_per_kg": 50.0, "vd_L_per_kg": 2.0},
    {"species": "dog", "cl_mL_per_min_per_kg": 20.0, "vd_L_per_kg": 1.5},
]


@pytest.fixture
def single_result():
    return predict_human_pk_allometric("DrugA", SINGLE_SPECIES_DATA)


@pytest.fixture
def multi_result():
    return predict_human_pk_allometric("DrugB", MULTI_SPECIES_DATA)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_allometric_scaling_result(single_result):
    assert isinstance(single_result, AllometricScalingResult)


def test_drug_name_preserved(single_result):
    assert single_result.drug_name == "DrugA"


def test_bw_human_kg_default(single_result):
    assert single_result.bw_human_kg == 70.0


def test_bw_human_kg_custom():
    result = predict_human_pk_allometric("X", SINGLE_SPECIES_DATA, bw_human_kg=60.0)
    assert result.bw_human_kg == 60.0


# ---------------------------------------------------------------------------
# Predicted values positivity
# ---------------------------------------------------------------------------


def test_cl_human_L_per_h_positive(single_result):
    assert single_result.cl_human_L_per_h > 0.0


def test_vd_human_L_positive(single_result):
    assert single_result.vd_human_L > 0.0


def test_t_half_human_h_positive(single_result):
    assert single_result.t_half_human_h > 0.0


def test_cl_human_mL_per_min_per_kg_positive(single_result):
    assert single_result.cl_human_mL_per_min_per_kg > 0.0


def test_vd_human_L_per_kg_positive(single_result):
    assert single_result.vd_human_L_per_kg > 0.0


# ---------------------------------------------------------------------------
# Single-species: fixed exponents
# ---------------------------------------------------------------------------


def test_single_species_cl_exponent(single_result):
    assert abs(single_result.cl_exponent - 0.75) < 1e-9


def test_single_species_vd_exponent(single_result):
    assert abs(single_result.vd_exponent - 1.0) < 1e-9


def test_single_species_confidence_low(single_result):
    assert single_result.confidence == "low"


def test_single_species_n_species_used(single_result):
    assert single_result.n_species_used == 1


# ---------------------------------------------------------------------------
# Multi-species: OLS regression
# ---------------------------------------------------------------------------


def test_multi_species_confidence_high(multi_result):
    assert multi_result.confidence == "high"


def test_two_species_confidence_moderate():
    result = predict_human_pk_allometric("X", TWO_SPECIES_DATA)
    assert result.confidence == "moderate"


def test_multi_species_n_species_used(multi_result):
    assert multi_result.n_species_used == 3


def test_two_species_n_species_used():
    result = predict_human_pk_allometric("X", TWO_SPECIES_DATA)
    assert result.n_species_used == 2


# ---------------------------------------------------------------------------
# Unit consistency: cl_human_L_per_h formula
# ---------------------------------------------------------------------------


def test_cl_unit_conversion_single(single_result):
    """cl_human_L_per_h = cl_mL_per_min_per_kg * bw * 60 / 1000."""
    expected = single_result.cl_human_mL_per_min_per_kg * 70.0 * 60.0 / 1000.0
    assert abs(single_result.cl_human_L_per_h - expected) < 1e-9


def test_cl_unit_conversion_multi(multi_result):
    expected = multi_result.cl_human_mL_per_min_per_kg * 70.0 * 60.0 / 1000.0
    assert abs(multi_result.cl_human_L_per_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Monotonicity: higher animal CL → higher human CL
# ---------------------------------------------------------------------------


def test_higher_rat_cl_higher_human_cl():
    result_low = predict_human_pk_allometric(
        "LowCL", [{"species": "rat", "cl_mL_per_min_per_kg": 20.0, "vd_L_per_kg": 2.0}]
    )
    result_high = predict_human_pk_allometric(
        "HighCL", [{"species": "rat", "cl_mL_per_min_per_kg": 80.0, "vd_L_per_kg": 2.0}]
    )
    assert result_high.cl_human_L_per_h > result_low.cl_human_L_per_h


# ---------------------------------------------------------------------------
# multi_species_scaling alias
# ---------------------------------------------------------------------------


def test_multi_species_scaling_alias():
    result = multi_species_scaling("DrugC", MULTI_SPECIES_DATA)
    assert isinstance(result, AllometricScalingResult)
    assert result.drug_name == "DrugC"
    assert result.n_species_used == 3


# ---------------------------------------------------------------------------
# t½ formula
# ---------------------------------------------------------------------------


def test_t_half_formula(single_result):
    """t½ = ln(2) * Vd / CL."""
    expected = math.log(2) * single_result.vd_human_L / single_result.cl_human_L_per_h
    assert abs(single_result.t_half_human_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty(single_result):
    assert len(single_result.notes) > 0


def test_notes_nonempty_multi(multi_result):
    assert len(multi_result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_empty_preclinical_data():
    with pytest.raises(ValueError):
        predict_human_pk_allometric("X", [])


def test_validation_cl_zero():
    with pytest.raises(ValueError):
        predict_human_pk_allometric(
            "X",
            [{"species": "rat", "cl_mL_per_min_per_kg": 0.0, "vd_L_per_kg": 2.0}],
        )


def test_validation_cl_negative():
    with pytest.raises(ValueError):
        predict_human_pk_allometric(
            "X",
            [{"species": "rat", "cl_mL_per_min_per_kg": -5.0, "vd_L_per_kg": 2.0}],
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError):
        predict_human_pk_allometric(
            "X",
            [{"species": "rat", "cl_mL_per_min_per_kg": 50.0, "vd_L_per_kg": 0.0}],
        )


def test_validation_bw_human_zero():
    with pytest.raises(ValueError):
        predict_human_pk_allometric("X", SINGLE_SPECIES_DATA, bw_human_kg=0.0)
