"""Tests for Phase 1102 — Drug Microsomal Stability Prediction."""

import pytest

from omega_pbpk.prediction.microsomal_stability import (
    MicrosomalStabilityResult,
    predict_microsomal_stability,
    screen_compounds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def high_stability():
    """t½=90 min → high stability, low clearance."""
    return predict_microsomal_stability("DrugHigh", t_half_in_vitro_min=90.0)


@pytest.fixture
def moderate_stability():
    """t½=45 min → moderate stability."""
    return predict_microsomal_stability("DrugMod", t_half_in_vitro_min=45.0)


@pytest.fixture
def low_stability():
    """t½=10 min → low stability, high clearance."""
    return predict_microsomal_stability("DrugLow", t_half_in_vitro_min=10.0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(high_stability):
    assert isinstance(high_stability, MicrosomalStabilityResult)


def test_frozen_dataclass(high_stability):
    with pytest.raises((AttributeError, TypeError)):
        high_stability.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLint direction: shorter t½ → higher CLint
# ---------------------------------------------------------------------------


def test_shorter_half_life_higher_clint_in_vitro():
    short = predict_microsomal_stability("Short", t_half_in_vitro_min=10.0)
    long = predict_microsomal_stability("Long", t_half_in_vitro_min=90.0)
    assert short.clint_in_vitro_uL_per_min_per_mg > long.clint_in_vitro_uL_per_min_per_mg


def test_shorter_half_life_higher_cl_hepatic():
    short = predict_microsomal_stability("Short", t_half_in_vitro_min=10.0)
    long = predict_microsomal_stability("Long", t_half_in_vitro_min=90.0)
    assert short.cl_hepatic_L_per_h > long.cl_hepatic_L_per_h


def test_shorter_half_life_higher_clint_liver():
    short = predict_microsomal_stability("Short", t_half_in_vitro_min=10.0)
    long = predict_microsomal_stability("Long", t_half_in_vitro_min=90.0)
    assert short.clint_liver_L_per_h > long.clint_liver_L_per_h


# ---------------------------------------------------------------------------
# Physicochemical constraints
# ---------------------------------------------------------------------------


def test_cl_hepatic_le_q_hepatic(low_stability):
    """Hepatic CL cannot exceed hepatic blood flow."""
    assert low_stability.cl_hepatic_L_per_h <= 90.0 + 1e-9


def test_cl_hepatic_le_custom_q_hepatic():
    result = predict_microsomal_stability("Drug", t_half_in_vitro_min=5.0, q_hepatic_L_per_h=60.0)
    assert result.cl_hepatic_L_per_h <= 60.0 + 1e-9


def test_hepatic_extraction_ratio_in_0_1(low_stability):
    assert 0.0 <= low_stability.hepatic_extraction_ratio <= 1.0


def test_f_oral_in_0_1(low_stability):
    assert 0.0 <= low_stability.f_oral_predicted <= 1.0


def test_f_oral_in_0_1_high(high_stability):
    assert 0.0 <= high_stability.f_oral_predicted <= 1.0


def test_hepatic_extraction_ratio_in_0_1_high(high_stability):
    assert 0.0 <= high_stability.hepatic_extraction_ratio <= 1.0


# ---------------------------------------------------------------------------
# Metabolic stability classification
# ---------------------------------------------------------------------------


def test_metabolic_stability_valid(high_stability, moderate_stability, low_stability):
    valid = {"high", "moderate", "low"}
    assert high_stability.metabolic_stability in valid
    assert moderate_stability.metabolic_stability in valid
    assert low_stability.metabolic_stability in valid


def test_long_half_life_high_stability(high_stability):
    """t½=90 min > 60 min → 'high'."""
    assert high_stability.metabolic_stability == "high"


def test_short_half_life_low_stability(low_stability):
    """t½=10 min < 30 min → 'low'."""
    assert low_stability.metabolic_stability == "low"


def test_moderate_half_life_moderate_stability(moderate_stability):
    """t½=45 min → 'moderate'."""
    assert moderate_stability.metabolic_stability == "moderate"


# ---------------------------------------------------------------------------
# Drug class classification
# ---------------------------------------------------------------------------


def test_drug_class_valid(high_stability, low_stability):
    valid = {"low_clearance", "intermediate", "high_clearance"}
    assert high_stability.drug_class in valid
    assert low_stability.drug_class in valid


def test_high_stability_low_clearance_class():
    """t½=180 min → E_h<0.3 → low_clearance class."""
    result = predict_microsomal_stability("DrugVeryHigh", t_half_in_vitro_min=180.0)
    assert result.drug_class == "low_clearance"


def test_low_stability_high_clearance_class(low_stability):
    assert low_stability.drug_class == "high_clearance"


# ---------------------------------------------------------------------------
# screen_compounds
# ---------------------------------------------------------------------------


def test_screen_compounds_count():
    compounds = [
        {"name": "A", "t_half_min": 10.0},
        {"name": "B", "t_half_min": 45.0},
        {"name": "C", "t_half_min": 90.0},
    ]
    results = screen_compounds(compounds)
    assert len(results) == 3


def test_screen_compounds_sorted_ascending():
    compounds = [
        {"name": "A", "t_half_min": 10.0},
        {"name": "B", "t_half_min": 45.0},
        {"name": "C", "t_half_min": 90.0},
    ]
    results = screen_compounds(compounds)
    cl_values = [r.cl_hepatic_L_per_h for r in results]
    for i in range(len(cl_values) - 1):
        assert cl_values[i] <= cl_values[i + 1] + 1e-9, f"Not sorted ascending: {cl_values}"


def test_screen_compounds_returns_list():
    compounds = [{"name": "X", "t_half_min": 30.0}]
    results = screen_compounds(compounds)
    assert isinstance(results, list)


def test_screen_compounds_correct_type():
    compounds = [{"name": "X", "t_half_min": 30.0}]
    results = screen_compounds(compounds)
    assert all(isinstance(r, MicrosomalStabilityResult) for r in results)


def test_screen_compounds_with_fu_mic():
    compounds = [
        {"name": "A", "t_half_min": 20.0, "fu_mic": 0.5},
        {"name": "B", "t_half_min": 20.0, "fu_mic": 1.0},
    ]
    results = screen_compounds(compounds)
    # fu_mic=0.5 reduces effective CLint → lower CLhep → comes first
    assert results[0].fu_mic == 0.5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_half_life():
    with pytest.raises(ValueError, match="t_half"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=-10.0)


def test_validation_zero_half_life():
    with pytest.raises(ValueError, match="t_half"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=0.0)


def test_validation_negative_protein_conc():
    with pytest.raises(ValueError, match="protein_conc"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=30.0, protein_conc_mg_per_mL=-1.0)


def test_validation_negative_volume():
    with pytest.raises(ValueError, match="volume"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=30.0, volume_incubation_uL=-10.0)


def test_validation_fu_mic_zero():
    with pytest.raises(ValueError, match="fu_mic"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=30.0, fu_mic=0.0)


def test_validation_fu_mic_above_one():
    with pytest.raises(ValueError, match="fu_mic"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=30.0, fu_mic=1.5)


def test_validation_negative_q_hepatic():
    with pytest.raises(ValueError, match="q_hepatic"):
        predict_microsomal_stability("Drug", t_half_in_vitro_min=30.0, q_hepatic_L_per_h=-10.0)
