"""Tests for Phase 500 — Bile Salt Solubilization Model."""

import math

import pytest

from omega_pbpk.prediction.bile_salt_solubilization import (
    BileSaltSolubilizationResult,
    compare_food_states,
    predict_bile_solubilization,
)

# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_bile_solubilization("TestDrug", 3.0, 0.05, "fasted")
    assert isinstance(result, BileSaltSolubilizationResult)


def test_required_fields():
    result = predict_bile_solubilization("DrugA", 2.5, 0.01, "fed")
    assert hasattr(result, "drug_name")
    assert hasattr(result, "logP")
    assert hasattr(result, "intrinsic_solubility_mg_mL")
    assert hasattr(result, "food_state")
    assert hasattr(result, "bile_salt_conc_mM")
    assert hasattr(result, "bile_enhancement_mg_mL")
    assert hasattr(result, "total_solubility_mg_mL")
    assert hasattr(result, "enhancement_fold")
    assert hasattr(result, "bcs_relevant")
    assert hasattr(result, "notes")


def test_field_values_stored():
    result = predict_bile_solubilization("Fenofibrate", 5.2, 0.002, "fasted")
    assert result.drug_name == "Fenofibrate"
    assert result.logP == 5.2
    assert result.intrinsic_solubility_mg_mL == 0.002
    assert result.food_state == "fasted"


# ---------------------------------------------------------------------------
# Bile salt concentration by food state
# ---------------------------------------------------------------------------


def test_fasted_bile_conc():
    result = predict_bile_solubilization("Drug", 3.0, 0.01, "fasted")
    assert math.isclose(result.bile_salt_conc_mM, 3.0)


def test_fed_bile_conc():
    result = predict_bile_solubilization("Drug", 3.0, 0.01, "fed")
    assert math.isclose(result.bile_salt_conc_mM, 15.0)


# ---------------------------------------------------------------------------
# Lipophilic drug: fed > fasted
# ---------------------------------------------------------------------------


def test_lipophilic_fed_greater_than_fasted():
    """Fed state has more bile salts → higher solubilization for lipophilic drug."""
    fed = predict_bile_solubilization("Lipophilic", 4.0, 0.01, "fed")
    fasted = predict_bile_solubilization("Lipophilic", 4.0, 0.01, "fasted")
    assert fed.total_solubility_mg_mL > fasted.total_solubility_mg_mL


def test_lipophilic_fed_enhancement_greater():
    fed = predict_bile_solubilization("Lipophilic", 4.0, 0.01, "fed")
    fasted = predict_bile_solubilization("Lipophilic", 4.0, 0.01, "fasted")
    assert fed.bile_enhancement_mg_mL > fasted.bile_enhancement_mg_mL


# ---------------------------------------------------------------------------
# Hydrophilic drug: no enhancement
# ---------------------------------------------------------------------------


def test_hydrophilic_no_enhancement_logP_zero():
    """logP=0 <= 1 → partition_factor=0 → no bile enhancement."""
    result = predict_bile_solubilization("Hydrophilic", 0.0, 1.0, "fed")
    assert math.isclose(result.bile_enhancement_mg_mL, 0.0)


def test_hydrophilic_no_enhancement_logP_one():
    """logP=1.0 → partition_factor=0 → no bile enhancement."""
    result = predict_bile_solubilization("BorderLine", 1.0, 0.5, "fed")
    assert math.isclose(result.bile_enhancement_mg_mL, 0.0)


def test_hydrophilic_negative_logP_no_enhancement():
    result = predict_bile_solubilization("VeryHydrophilic", -1.0, 2.0, "fed")
    assert math.isclose(result.bile_enhancement_mg_mL, 0.0)


# ---------------------------------------------------------------------------
# total_solubility >= intrinsic_solubility always
# ---------------------------------------------------------------------------


def test_total_ge_intrinsic_fasted():
    result = predict_bile_solubilization("Drug", 2.0, 0.05, "fasted")
    assert result.total_solubility_mg_mL >= result.intrinsic_solubility_mg_mL


def test_total_ge_intrinsic_fed():
    result = predict_bile_solubilization("Drug", 0.5, 0.5, "fed")
    assert result.total_solubility_mg_mL >= result.intrinsic_solubility_mg_mL


def test_total_ge_intrinsic_hydrophilic():
    result = predict_bile_solubilization("Hydro", -2.0, 10.0, "fed")
    assert result.total_solubility_mg_mL >= result.intrinsic_solubility_mg_mL


# ---------------------------------------------------------------------------
# enhancement_fold calculation
# ---------------------------------------------------------------------------


def test_enhancement_fold_definition():
    result = predict_bile_solubilization("Drug", 3.0, 0.02, "fed")
    expected_fold = result.total_solubility_mg_mL / result.intrinsic_solubility_mg_mL
    assert math.isclose(result.enhancement_fold, expected_fold, rel_tol=1e-9)


def test_enhancement_fold_at_least_one():
    result = predict_bile_solubilization("Drug", 1.5, 0.1, "fasted")
    assert result.enhancement_fold >= 1.0


def test_enhancement_fold_one_for_hydrophilic():
    """No bile enhancement → fold should be exactly 1.0."""
    result = predict_bile_solubilization("Hydro", 0.0, 1.0, "fed")
    assert math.isclose(result.enhancement_fold, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# BCS relevance flag
# ---------------------------------------------------------------------------


def test_bcs_relevant_low_solubility():
    """Solubility < 0.1 mg/mL → bcs_relevant True."""
    result = predict_bile_solubilization("PoorSol", 3.0, 0.05, "fasted")
    assert result.bcs_relevant is True


def test_bcs_not_relevant_high_solubility():
    """Solubility >= 0.1 mg/mL → bcs_relevant False."""
    result = predict_bile_solubilization("GoodSol", 1.0, 0.5, "fasted")
    assert result.bcs_relevant is False


def test_bcs_boundary_below():
    result = predict_bile_solubilization("BoundaryLow", 2.0, 0.099, "fasted")
    assert result.bcs_relevant is True


def test_bcs_boundary_at_threshold():
    result = predict_bile_solubilization("BoundaryAt", 2.0, 0.1, "fasted")
    assert result.bcs_relevant is False


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validate_intrinsic_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_bile_solubilization("Drug", 3.0, 0.0, "fasted")


def test_validate_intrinsic_solubility_negative():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_bile_solubilization("Drug", 3.0, -0.1, "fasted")


def test_validate_food_state_invalid():
    with pytest.raises(ValueError, match="food_state"):
        predict_bile_solubilization("Drug", 3.0, 0.1, "unknown")


# ---------------------------------------------------------------------------
# compare_food_states
# ---------------------------------------------------------------------------


def test_compare_food_states_returns_list():
    results = compare_food_states("Drug", 3.0, 0.05)
    assert isinstance(results, list)


def test_compare_food_states_length_two():
    results = compare_food_states("Drug", 3.0, 0.05)
    assert len(results) == 2


def test_compare_food_states_order():
    """First element is fasted, second is fed."""
    results = compare_food_states("Drug", 3.0, 0.05)
    assert results[0].food_state == "fasted"
    assert results[1].food_state == "fed"


def test_compare_food_states_types():
    results = compare_food_states("Drug", 2.5, 0.02)
    for r in results:
        assert isinstance(r, BileSaltSolubilizationResult)


def test_compare_food_states_fed_higher_for_lipophilic():
    results = compare_food_states("Fenofibrate", 5.0, 0.002)
    fasted, fed = results
    assert fed.total_solubility_mg_mL >= fasted.total_solubility_mg_mL
