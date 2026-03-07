"""Tests for enhanced allometric scaling (Phase 824)."""

import pytest

from omega_pbpk.prediction.allometry_enhanced import (
    AllometryResult,
    allometric_scale,
    multi_species_scale,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_allometric_scale_return_type():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert isinstance(result, AllometryResult)


def test_multi_species_scale_return_type():
    species_data = [
        {"species": "rat", "bw_kg": 0.25, "value": 0.5},
        {"species": "dog", "bw_kg": 10.0, "value": 8.0},
    ]
    result = multi_species_scale("DrugA", "CL", species_data, 70.0)
    assert isinstance(result, AllometryResult)


# ---------------------------------------------------------------------------
# Allometric exponent tests
# ---------------------------------------------------------------------------


def test_cl_exponent_is_0_75():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert abs(result.allometric_exponent - 0.75) < 1e-9


def test_vd_exponent_is_1_0():
    result = allometric_scale("DrugA", "Vd", "rat", 0.25, 0.25, 70.0, "none")
    assert abs(result.allometric_exponent - 1.0) < 1e-9


def test_t_half_exponent_is_0_25():
    result = allometric_scale("DrugA", "t_half", "rat", 1.5, 0.25, 70.0, "none")
    assert abs(result.allometric_exponent - 0.25) < 1e-9


def test_unknown_parameter_uses_default_exponent():
    result = allometric_scale("DrugA", "ka", "rat", 0.5, 0.25, 70.0, "none")
    assert abs(result.allometric_exponent - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# Scaling correctness
# ---------------------------------------------------------------------------


def test_predicted_human_value_positive():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.predicted_human_value > 0.0


def test_human_larger_animal_higher_cl():
    """Human BW >> rat BW so predicted CL should be higher than animal CL."""
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.predicted_human_value > result.animal_value


def test_cl_scaling_manual_calculation():
    """Verify manual: Y_human = Y_animal * (70 / 0.25)^0.75."""
    animal_cl = 0.5
    animal_bw = 0.25
    human_bw = 70.0
    expected = animal_cl * (human_bw / animal_bw) ** 0.75
    result = allometric_scale("DrugA", "CL", "rat", animal_cl, animal_bw, human_bw, "none")
    assert abs(result.predicted_human_value - expected) < 1e-9


def test_vd_scaling_manual_calculation():
    """Verify manual: Y_human = Y_animal * (70 / 10)^1.0."""
    animal_vd = 5.0
    animal_bw = 10.0
    human_bw = 70.0
    expected = animal_vd * (human_bw / animal_bw) ** 1.0
    result = allometric_scale("DrugA", "Vd", "dog", animal_vd, animal_bw, human_bw, "none")
    assert abs(result.predicted_human_value - expected) < 1e-9


# ---------------------------------------------------------------------------
# Prediction interval tests
# ---------------------------------------------------------------------------


def test_prediction_interval_lower_less_than_predicted():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.prediction_interval_lower < result.corrected_human_value


def test_prediction_interval_upper_greater_than_predicted():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.prediction_interval_upper > result.corrected_human_value


def test_prediction_interval_ratio():
    """Lower = predicted / 1.5, upper = predicted * 1.5."""
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    pred = result.corrected_human_value
    assert abs(result.prediction_interval_lower - pred / 1.5) < 1e-9
    assert abs(result.prediction_interval_upper - pred * 1.5) < 1e-9


# ---------------------------------------------------------------------------
# Correction methods
# ---------------------------------------------------------------------------


def test_mlp_correction_applied_to_cl():
    result_none = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    result_mlp = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "mlp")
    # MLP correction changes the corrected value
    assert result_mlp.correction_method == "mlp"
    assert result_mlp.corrected_human_value != result_none.corrected_human_value


def test_mlp_correction_not_applied_to_vd():
    """MLP correction should not apply to Vd; correction_method stays 'none'."""
    result = allometric_scale("DrugA", "Vd", "rat", 0.25, 0.25, 70.0, "mlp")
    assert result.correction_method == "none"
    assert abs(result.corrected_human_value - result.predicted_human_value) < 1e-9


def test_brain_weight_correction_applied_to_vd():
    result_none = allometric_scale("DrugA", "Vd", "rat", 0.25, 0.25, 70.0, "none")
    result_bw = allometric_scale("DrugA", "Vd", "rat", 0.25, 0.25, 70.0, "brain_weight")
    assert result_bw.correction_method == "brain_weight"
    assert result_bw.corrected_human_value != result_none.corrected_human_value


def test_brain_weight_correction_not_applied_to_cl():
    """Brain weight correction should not apply to CL; correction_method stays 'none'."""
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "brain_weight")
    assert result.correction_method == "none"
    assert abs(result.corrected_human_value - result.predicted_human_value) < 1e-9


def test_no_correction_method_none():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.correction_method == "none"
    assert abs(result.corrected_human_value - result.predicted_human_value) < 1e-9


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_cl_units():
    result = allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 70.0, "none")
    assert result.units == "L/h"


def test_vd_units():
    result = allometric_scale("DrugA", "Vd", "rat", 0.25, 0.25, 70.0, "none")
    assert result.units == "L"


def test_t_half_units():
    result = allometric_scale("DrugA", "t_half", "rat", 1.5, 0.25, 70.0, "none")
    assert result.units == "h"


# ---------------------------------------------------------------------------
# Multi-species scaling
# ---------------------------------------------------------------------------


def test_multi_species_two_species():
    species_data = [
        {"species": "rat", "bw_kg": 0.25, "value": 0.5},
        {"species": "dog", "bw_kg": 10.0, "value": 8.0},
    ]
    result = multi_species_scale("DrugA", "CL", species_data, 70.0)
    assert isinstance(result, AllometryResult)
    assert result.predicted_human_value > 0.0


def test_multi_species_three_species():
    species_data = [
        {"species": "mouse", "bw_kg": 0.02, "value": 0.1},
        {"species": "rat", "bw_kg": 0.25, "value": 0.8},
        {"species": "dog", "bw_kg": 10.0, "value": 12.0},
    ]
    result = multi_species_scale("DrugA", "CL", species_data, 70.0)
    assert result.allometric_exponent > 0.0
    assert result.predicted_human_value > 0.0


def test_multi_species_correction_is_none():
    """Multi-species scaling always uses correction_method='none'."""
    species_data = [
        {"species": "rat", "bw_kg": 0.25, "value": 0.5},
        {"species": "dog", "bw_kg": 10.0, "value": 8.0},
    ]
    result = multi_species_scale("DrugA", "CL", species_data, 70.0)
    assert result.correction_method == "none"


def test_multi_species_prediction_interval_valid():
    species_data = [
        {"species": "rat", "bw_kg": 0.25, "value": 0.5},
        {"species": "dog", "bw_kg": 10.0, "value": 8.0},
    ]
    result = multi_species_scale("DrugA", "CL", species_data, 70.0)
    assert result.prediction_interval_lower < result.predicted_human_value
    assert result.prediction_interval_upper > result.predicted_human_value


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_animal_value_zero_raises():
    with pytest.raises(ValueError, match="animal_value"):
        allometric_scale("DrugA", "CL", "rat", 0.0, 0.25, 70.0, "none")


def test_animal_value_negative_raises():
    with pytest.raises(ValueError, match="animal_value"):
        allometric_scale("DrugA", "CL", "rat", -1.0, 0.25, 70.0, "none")


def test_animal_bw_zero_raises():
    with pytest.raises(ValueError, match="animal_bw_kg"):
        allometric_scale("DrugA", "CL", "rat", 0.5, 0.0, 70.0, "none")


def test_animal_bw_negative_raises():
    with pytest.raises(ValueError, match="animal_bw_kg"):
        allometric_scale("DrugA", "CL", "rat", 0.5, -0.25, 70.0, "none")


def test_human_bw_zero_raises():
    with pytest.raises(ValueError, match="human_bw_kg"):
        allometric_scale("DrugA", "CL", "rat", 0.5, 0.25, 0.0, "none")


def test_multi_species_invalid_value_raises():
    species_data = [{"species": "rat", "bw_kg": 0.25, "value": 0.0}]
    with pytest.raises(ValueError):
        multi_species_scale("DrugA", "CL", species_data, 70.0)


def test_multi_species_invalid_bw_raises():
    species_data = [{"species": "rat", "bw_kg": 0.0, "value": 0.5}]
    with pytest.raises(ValueError):
        multi_species_scale("DrugA", "CL", species_data, 70.0)


def test_multi_species_empty_raises():
    with pytest.raises(ValueError):
        multi_species_scale("DrugA", "CL", [], 70.0)
