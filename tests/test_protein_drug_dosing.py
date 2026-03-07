"""Tests for Phase 669 — Protein Drug Dosing."""

import math

import pytest

from omega_pbpk.clinical.protein_drug_dosing import (
    ProteinDrugDosingResult,
    adjust_dose_for_protein_binding,
    screen_protein_binding_scenarios,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def make_result(
    scenario: str = "hypoalbuminemia",
    albumin: float = 2.0,
    fup_normal: float = 0.1,
    dose_mg: float = 100.0,
) -> ProteinDrugDosingResult:
    return adjust_dose_for_protein_binding(
        drug_name="TestDrug",
        dose_normal_mg=dose_mg,
        fup_normal=fup_normal,
        clinical_scenario=scenario,
        albumin_g_dL=albumin,
    )


# ---------------------------------------------------------------------------
# Basic return type and field sanity
# ---------------------------------------------------------------------------


def test_returns_protein_drug_dosing_result():
    result = make_result()
    assert isinstance(result, ProteinDrugDosingResult)


def test_dose_adjusted_mg_positive():
    result = make_result()
    assert result.dose_adjusted_mg > 0


def test_dose_adjustment_factor_positive():
    result = make_result()
    assert result.dose_adjustment_factor > 0


def test_fup_adjusted_in_range():
    result = make_result()
    assert 0 < result.fup_adjusted <= 1


def test_css_total_adjusted_mg_L_positive():
    result = make_result()
    assert result.css_total_adjusted_mg_L > 0


def test_clinical_recommendation_nonempty():
    result = make_result()
    assert isinstance(result.clinical_recommendation, str)
    assert len(result.clinical_recommendation) > 0


def test_notes_nonempty():
    result = make_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Scenario-specific correctness
# ---------------------------------------------------------------------------


def test_hypoalbuminemia_fup_adjusted_greater_than_normal():
    """Less albumin → more free drug → fup_adjusted > fup_normal."""
    result = make_result(scenario="hypoalbuminemia", albumin=2.0, fup_normal=0.1)
    assert result.fup_adjusted > result.fup_normal


def test_hypoalbuminemia_dose_adjusted_less_than_normal():
    """More free drug → need less total drug to achieve same free CSS."""
    result = make_result(scenario="hypoalbuminemia", albumin=2.0, fup_normal=0.1)
    assert result.dose_adjusted_mg < result.dose_normal_mg


def test_normal_scenario_fup_unchanged():
    """Normal scenario: fup_adjusted equals fup_normal."""
    result = make_result(scenario="normal", albumin=4.0, fup_normal=0.2)
    assert math.isclose(result.fup_adjusted, result.fup_normal, rel_tol=1e-9)


def test_normal_scenario_adjustment_factor_approx_one():
    """Normal scenario: dose_adjustment_factor ≈ 1.0."""
    result = make_result(scenario="normal", albumin=4.0, fup_normal=0.2)
    assert math.isclose(result.dose_adjustment_factor, 1.0, rel_tol=1e-9)


def test_uremia_fup_adjusted_less_than_normal():
    """Uremia elevates AGP → more binding → less free drug → fup_adjusted < fup_normal."""
    result = make_result(scenario="uremia", albumin=4.0, fup_normal=0.2)
    assert result.fup_adjusted < result.fup_normal


def test_uremia_dose_adjusted_greater_than_normal():
    """Less free drug → need more total drug."""
    result = make_result(scenario="uremia", albumin=4.0, fup_normal=0.2)
    assert result.dose_adjusted_mg > result.dose_normal_mg


def test_pregnancy_fup_adjusted_greater_than_normal():
    """Pregnancy dilution of albumin → more free drug."""
    result = make_result(scenario="pregnancy", albumin=3.0, fup_normal=0.1)
    assert result.fup_adjusted > result.fup_normal


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------


def test_risk_of_toxicity_when_fup_adjusted_greater():
    """Unadjusted dose exposes to more free drug → toxicity risk."""
    result = make_result(scenario="hypoalbuminemia", albumin=2.0, fup_normal=0.1)
    assert result.risk_of_toxicity is True


def test_risk_of_subtherapeutic_when_fup_adjusted_less():
    """Unadjusted dose gives less free drug → subtherapeutic risk."""
    result = make_result(scenario="uremia", albumin=4.0, fup_normal=0.2)
    assert result.risk_of_subtherapeutic is True


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    results = screen_protein_binding_scenarios(
        drug_name="Drug",
        dose_mg=200.0,
        fup_normal=0.1,
    )
    # Default 4 scenarios
    assert len(results) == 4


def test_screen_sorted_by_dose_adjustment_factor_descending():
    results = screen_protein_binding_scenarios(
        drug_name="Drug",
        dose_mg=200.0,
        fup_normal=0.1,
    )
    factors = [r.dose_adjustment_factor for r in results]
    assert factors == sorted(factors, reverse=True)


def test_screen_custom_scenarios():
    results = screen_protein_binding_scenarios(
        drug_name="Drug",
        dose_mg=100.0,
        fup_normal=0.15,
        scenarios=["hypoalbuminemia", "uremia"],
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_fup_normal_zero_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_protein_binding(
            drug_name="Drug",
            dose_normal_mg=100.0,
            fup_normal=0.0,
        )


def test_validation_dose_normal_zero_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_protein_binding(
            drug_name="Drug",
            dose_normal_mg=0.0,
            fup_normal=0.1,
        )


def test_validation_albumin_zero_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_protein_binding(
            drug_name="Drug",
            dose_normal_mg=100.0,
            fup_normal=0.1,
            albumin_g_dL=0.0,
        )


def test_validation_unknown_scenario_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_protein_binding(
            drug_name="Drug",
            dose_normal_mg=100.0,
            fup_normal=0.1,
            clinical_scenario="dialysis",
        )
