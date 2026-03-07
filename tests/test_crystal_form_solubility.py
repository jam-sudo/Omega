"""Tests for Phase 938 — Crystal Form Solubility (crystal_form_solubility.py)."""

import pytest

from omega_pbpk.prediction.crystal_form_solubility import (
    CrystalFormSolubilityResult,
    compare_crystal_forms,
    predict_crystal_form_solubility,
)

# ── Basic return type ──────────────────────────────────────────────────────────


def test_return_type():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert isinstance(result, CrystalFormSolubilityResult)


def test_frozen_dataclass():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    with pytest.raises((AttributeError, TypeError)):
        result.logp = 99.0


def test_intrinsic_solubility_positive():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.intrinsic_solubility_mg_mL > 0


def test_ph_adjusted_solubility_positive():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.ph_adjusted_solubility_mg_mL > 0


def test_relative_to_form_I_positive():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_II"
    )
    assert result.relative_to_form_I > 0


def test_dissolution_advantage_valid():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.dissolution_advantage in {"high", "moderate", "low"}


def test_stability_risk_valid():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.stability_risk in {"low", "moderate", "high"}


def test_notes_nonempty():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_crystal_form_solubility(
        drug_name="Ibuprofen", logp=3.5, pka=4.4, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.drug_name == "Ibuprofen"


# ── Form-specific properties ───────────────────────────────────────────────────


def test_amorphous_highest_in_compare():
    results = compare_crystal_forms("DrugX", logp=2.0, pka=4.5, ionization_type="neutral")
    assert results[0].crystal_form == "amorphous"


def test_form_I_relative_to_form_I_equals_one():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="neutral", crystal_form="Form_I"
    )
    assert abs(result.relative_to_form_I - 1.0) < 1e-9


def test_amorphous_stability_risk_high():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="amorphous"
    )
    assert result.stability_risk == "high"


def test_form_I_stability_risk_low():
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I"
    )
    assert result.stability_risk == "low"


def test_co_crystal_dissolution_advantage_high():
    """co_crystal has 5x relative to Form_I -> dissolution_advantage == 'high'."""
    result = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="neutral", crystal_form="co_crystal"
    )
    assert result.dissolution_advantage == "high"


def test_hydrate_lower_solubility_than_form_I():
    hydrate = predict_crystal_form_solubility(
        drug_name="DrugX", logp=2.0, pka=4.5, ionization_type="neutral", crystal_form="hydrate"
    )
    assert hydrate.relative_to_form_I < 1.0


# ── pH effects ─────────────────────────────────────────────────────────────────


def test_acid_high_ph_higher_solubility_than_low_ph():
    """Acid dissolves better at high pH (more ionized)."""
    high_ph = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I", ph=7.4
    )
    low_ph = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I", ph=1.2
    )
    assert high_ph.ph_adjusted_solubility_mg_mL > low_ph.ph_adjusted_solubility_mg_mL


def test_base_low_ph_higher_solubility_than_high_ph():
    """Base dissolves better at low pH (more ionized)."""
    low_ph = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=8.5, ionization_type="base", crystal_form="Form_I", ph=2.0
    )
    high_ph = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=8.5, ionization_type="base", crystal_form="Form_I", ph=7.4
    )
    assert low_ph.ph_adjusted_solubility_mg_mL > high_ph.ph_adjusted_solubility_mg_mL


def test_neutral_ph_independent():
    """Neutral drug: pH has no effect on solubility."""
    ph68 = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=7.0, ionization_type="neutral", crystal_form="Form_I", ph=6.8
    )
    ph12 = predict_crystal_form_solubility(
        "DrugX", logp=2.0, pka=7.0, ionization_type="neutral", crystal_form="Form_I", ph=1.2
    )
    # For neutral: ph_adjusted = intrinsic, so both should be equal
    assert abs(ph68.ph_adjusted_solubility_mg_mL - ph12.ph_adjusted_solubility_mg_mL) < 1e-9


# ── compare_crystal_forms ──────────────────────────────────────────────────────


def test_compare_returns_7_results():
    results = compare_crystal_forms("DrugX", logp=2.0, pka=4.5, ionization_type="acid")
    assert len(results) == 7


def test_compare_sorted_by_ph_adjusted_solubility_descending():
    results = compare_crystal_forms("DrugX", logp=2.0, pka=4.5, ionization_type="acid")
    sols = [r.ph_adjusted_solubility_mg_mL for r in results]
    assert sols == sorted(sols, reverse=True)


# ── Validation ─────────────────────────────────────────────────────────────────


def test_validation_invalid_crystal_form():
    with pytest.raises(ValueError, match="crystal_form"):
        predict_crystal_form_solubility(
            "DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_IV"
        )


def test_validation_ph_negative():
    with pytest.raises(ValueError, match="ph"):
        predict_crystal_form_solubility(
            "DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I", ph=-1.0
        )


def test_validation_ph_above_14():
    with pytest.raises(ValueError, match="ph"):
        predict_crystal_form_solubility(
            "DrugX", logp=2.0, pka=4.5, ionization_type="acid", crystal_form="Form_I", ph=15.0
        )


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_crystal_form_solubility(
            "DrugX", logp=2.0, pka=4.5, ionization_type="zwitterion", crystal_form="Form_I"
        )
