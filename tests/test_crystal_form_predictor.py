"""Tests for Phase 804 — Crystal Form Prediction."""

import math

import pytest

from omega_pbpk.prediction.crystal_form_predictor import (
    CrystalFormResult,
    compare_polymorphs,
    predict_crystal_form_properties,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_crystal_form_properties("CompA")
    assert isinstance(result, CrystalFormResult)


def test_result_is_frozen():
    result = predict_crystal_form_properties("CompA")
    with pytest.raises((AttributeError, TypeError)):
        result.solubility_ratio = 5.0  # type: ignore[misc]


def test_fields_are_strings_or_floats():
    result = predict_crystal_form_properties("CompA", polymorph_name="Form I")
    assert isinstance(result.compound_name, str)
    assert isinstance(result.polymorph_name, str)
    assert isinstance(result.melting_point_celsius, float)
    assert isinstance(result.enthalpy_of_fusion_kJ_mol, float)
    assert isinstance(result.solubility_ratio, float)
    assert isinstance(result.dissolution_rate_ratio, float)
    assert isinstance(result.thermodynamic_stability, str)
    assert isinstance(result.bioavailability_advantage, float)
    assert isinstance(result.shelf_life_stability, str)
    assert isinstance(result.formulation_recommendation, str)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Reference form
# ---------------------------------------------------------------------------


def test_reference_form_solubility_ratio_is_one():
    result = predict_crystal_form_properties("CompA", is_reference_form=True, is_amorphous=False)
    assert abs(result.solubility_ratio - 1.0) < 1e-9


def test_reference_form_stability_is_stable():
    result = predict_crystal_form_properties("CompA", is_reference_form=True)
    assert result.thermodynamic_stability == "stable"


# ---------------------------------------------------------------------------
# Amorphous form
# ---------------------------------------------------------------------------


def test_amorphous_solubility_ratio_positive():
    result = predict_crystal_form_properties("CompA", is_amorphous=True, is_reference_form=False)
    assert result.solubility_ratio > 1.0


def test_amorphous_solubility_ratio_is_30():
    result = predict_crystal_form_properties("CompA", is_amorphous=True, is_reference_form=False)
    assert abs(result.solubility_ratio - 30.0) < 1e-9


def test_amorphous_stability_is_metastable():
    result = predict_crystal_form_properties("CompA", is_amorphous=True, is_reference_form=False)
    assert result.thermodynamic_stability == "metastable"


def test_amorphous_low_logp_shelf_life_risk():
    result = predict_crystal_form_properties(
        "CompA", is_amorphous=True, logp=0.5, is_reference_form=False
    )
    assert result.shelf_life_stability == "risk_of_conversion"


# ---------------------------------------------------------------------------
# Lower MP polymorph has higher solubility
# ---------------------------------------------------------------------------


def test_lower_mp_polymorph_higher_solubility():
    """A polymorph with lower MP than 150°C reference should have higher solubility."""
    low_mp = predict_crystal_form_properties(
        "CompA",
        polymorph_name="Form II",
        melting_point_celsius=100.0,
        enthalpy_of_fusion_kJ_mol=20.0,
        is_reference_form=False,
        is_amorphous=False,
    )
    high_mp = predict_crystal_form_properties(
        "CompA",
        polymorph_name="Form III",
        melting_point_celsius=180.0,
        enthalpy_of_fusion_kJ_mol=20.0,
        is_reference_form=False,
        is_amorphous=False,
    )
    assert low_mp.solubility_ratio > high_mp.solubility_ratio


# ---------------------------------------------------------------------------
# Dissolution rate
# ---------------------------------------------------------------------------


def test_dissolution_rate_sqrt_of_solubility():
    result = predict_crystal_form_properties("CompA", is_amorphous=True, is_reference_form=False)
    expected_diss = math.sqrt(result.solubility_ratio)
    assert abs(result.dissolution_rate_ratio - expected_diss) < 1e-9


# ---------------------------------------------------------------------------
# Bioavailability advantage
# ---------------------------------------------------------------------------


def test_ba_advantage_reference_form_is_one():
    result = predict_crystal_form_properties("CompA", is_reference_form=True)
    # Reference: solubility_ratio=1.0, BA advantage = 1^0.5 = 1.0 (logP=2 < 3)
    assert abs(result.bioavailability_advantage - 1.0) < 1e-6


def test_ba_advantage_permeability_limited():
    """logP > 3 → exponent 0.3 instead of 0.5."""
    result_high_logp = predict_crystal_form_properties(
        "CompA", is_amorphous=True, logp=4.0, is_reference_form=False
    )
    result_low_logp = predict_crystal_form_properties(
        "CompA", is_amorphous=True, logp=1.0, is_reference_form=False
    )
    # Higher logP → smaller exponent → lower BA advantage for same solubility ratio
    assert result_low_logp.bioavailability_advantage > result_high_logp.bioavailability_advantage


def test_ba_advantage_gte_one_for_amorphous():
    result = predict_crystal_form_properties("CompA", is_amorphous=True, is_reference_form=False)
    assert result.bioavailability_advantage >= 1.0


# ---------------------------------------------------------------------------
# Thermodynamic stability values
# ---------------------------------------------------------------------------


def test_thermodynamic_stability_values():
    valid_values = {"stable", "metastable", "unstable"}
    for mp, is_ref, is_amor in [
        (150.0, True, False),
        (120.0, False, False),
        (80.0, False, False),
        (150.0, False, True),
    ]:
        result = predict_crystal_form_properties(
            "CompA", melting_point_celsius=mp, is_reference_form=is_ref, is_amorphous=is_amor
        )
        assert result.thermodynamic_stability in valid_values


def test_low_mp_polymorph_can_be_unstable():
    result = predict_crystal_form_properties(
        "CompA", melting_point_celsius=80.0, is_reference_form=False, is_amorphous=False
    )
    assert result.thermodynamic_stability == "unstable"


# ---------------------------------------------------------------------------
# Shelf life stability
# ---------------------------------------------------------------------------


def test_low_mp_shelf_life_unstable():
    result = predict_crystal_form_properties(
        "CompA", melting_point_celsius=80.0, is_reference_form=False
    )
    assert result.shelf_life_stability == "unstable"


def test_high_mp_stable_shelf_life():
    result = predict_crystal_form_properties(
        "CompA", melting_point_celsius=200.0, is_reference_form=True
    )
    assert result.shelf_life_stability == "stable"


# ---------------------------------------------------------------------------
# compare_polymorphs
# ---------------------------------------------------------------------------


def test_compare_polymorphs_returns_list():
    forms = [
        {"polymorph_name": "Form I", "mp": 150.0, "dhf": 20.0, "is_reference": True},
        {"polymorph_name": "Amorphous", "mp": 150.0, "dhf": 20.0, "is_amorphous": True},
        {"polymorph_name": "Form II", "mp": 120.0, "dhf": 15.0},
    ]
    results = compare_polymorphs("CompA", forms)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_polymorphs_sorted_by_solubility_desc():
    forms = [
        {"polymorph_name": "Form I", "mp": 150.0, "dhf": 20.0, "is_reference": True},
        {"polymorph_name": "Amorphous", "mp": 150.0, "dhf": 20.0, "is_amorphous": True},
        {"polymorph_name": "Form II", "mp": 100.0, "dhf": 15.0},
    ]
    results = compare_polymorphs("CompA", forms)
    for i in range(len(results) - 1):
        assert results[i].solubility_ratio >= results[i + 1].solubility_ratio


def test_compare_polymorphs_amorphous_first():
    forms = [
        {"polymorph_name": "Form I", "mp": 150.0, "dhf": 20.0, "is_reference": True},
        {"polymorph_name": "Amorphous", "mp": 150.0, "dhf": 20.0, "is_amorphous": True},
    ]
    results = compare_polymorphs("CompA", forms)
    assert results[0].polymorph_name == "Amorphous"


def test_compare_polymorphs_empty_input():
    results = compare_polymorphs("CompA", [])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError):
        predict_crystal_form_properties("CompA", mw_da=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError):
        predict_crystal_form_properties("CompA", mw_da=-100.0)
