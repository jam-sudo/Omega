"""Tests for Phase 1030 — Drug Ionization at Physiological pH."""

import pytest

from omega_pbpk.prediction.drug_ionization_at_ph import (
    IonizationAtPhResult,
    compare_ph_conditions,
    predict_ionization_at_ph,
)

# --- Return type and frozen dataclass ---


def test_returns_ionization_at_ph_result():
    result = predict_ionization_at_ph("TestDrug", pka=4.5, ionization_type="acid")
    assert isinstance(result, IonizationAtPhResult)


def test_result_is_frozen():
    result = predict_ionization_at_ph("TestDrug", pka=4.5, ionization_type="acid")
    with pytest.raises((TypeError, AttributeError)):
        result.fraction_ionized = 0.99  # type: ignore[misc]


# --- fraction_ionized + fraction_unionized == 1 ---


def test_fractions_sum_to_one_acid():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert abs(result.fraction_ionized + result.fraction_unionized - 1.0) < 1e-9


def test_fractions_sum_to_one_base():
    result = predict_ionization_at_ph("Amine", pka=9.0, ionization_type="base", ph=7.4)
    assert abs(result.fraction_ionized + result.fraction_unionized - 1.0) < 1e-9


def test_fractions_sum_to_one_neutral():
    result = predict_ionization_at_ph("Neutral", pka=7.0, ionization_type="neutral", ph=7.4)
    assert abs(result.fraction_ionized + result.fraction_unionized - 1.0) < 1e-9


# --- fraction_ionized in [0, 1] ---


def test_fraction_ionized_in_range_acid():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert 0.0 <= result.fraction_ionized <= 1.0


def test_fraction_ionized_in_range_base():
    result = predict_ionization_at_ph("Amine", pka=9.0, ionization_type="base", ph=7.4)
    assert 0.0 <= result.fraction_ionized <= 1.0


# --- Acid at pH > pKa: fraction_ionized > 0.5 ---


def test_acid_high_ph_mostly_ionized():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert result.fraction_ionized > 0.5


# --- Base at pH < pKa: fraction_ionized > 0.5 ---


def test_base_low_ph_mostly_ionized():
    result = predict_ionization_at_ph("Amine", pka=9.0, ionization_type="base", ph=2.0)
    assert result.fraction_ionized > 0.5


# --- Neutral: fraction_ionized == 0 ---


def test_neutral_fraction_ionized_zero():
    result = predict_ionization_at_ph("Neutral", pka=7.0, ionization_type="neutral", ph=7.4)
    assert result.fraction_ionized == 0.0


# --- Charge state ---


def test_charge_state_anionic_acid_high_ph():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert result.charge_state == "anionic"


def test_charge_state_cationic_base_low_ph():
    result = predict_ionization_at_ph("Amine", pka=9.0, ionization_type="base", ph=2.0)
    assert result.charge_state == "cationic"


def test_charge_state_neutral_for_neutral_type():
    result = predict_ionization_at_ph("Neutral", pka=7.0, ionization_type="neutral", ph=7.4)
    assert result.charge_state == "neutral"


def test_charge_state_zwitterionic_for_zwitterion():
    result = predict_ionization_at_ph("AminoAcid", pka=6.0, ionization_type="zwitterion", ph=7.4)
    assert result.charge_state == "zwitterionic"


# --- Solubility fold change ---


def test_solubility_fold_change_ge_1_acid_above_pka():
    # Acid ionized above pKa → more soluble than neutral form
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert result.solubility_fold_change >= 1.0


# --- Permeability factor in [0, 1] ---


def test_permeability_factor_in_range():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert 0.0 <= result.permeability_factor <= 1.0


# --- Absorption class ---


def test_absorption_class_valid():
    result = predict_ionization_at_ph("Aspirin", pka=3.5, ionization_type="acid", ph=7.4)
    assert result.absorption_class in {"high", "moderate", "low"}


# --- compare_ph_conditions ---


def test_compare_ph_conditions_length():
    ph_values = [1.0, 4.0, 7.4, 9.0, 12.0]
    results = compare_ph_conditions("Aspirin", pka=3.5, ionization_type="acid", ph_values=ph_values)
    assert len(results) == len(ph_values)


def test_compare_ph_conditions_returns_correct_types():
    ph_values = [2.0, 7.4]
    results = compare_ph_conditions("Aspirin", pka=3.5, ionization_type="acid", ph_values=ph_values)
    for r in results:
        assert isinstance(r, IonizationAtPhResult)


# --- Higher logP → higher permeability_factor ---


def test_higher_logp_higher_permeability():
    low_logp = predict_ionization_at_ph(
        "LowLogP", pka=7.0, ionization_type="acid", ph=7.4, logp_neutral=0.5
    )
    high_logp = predict_ionization_at_ph(
        "HighLogP", pka=7.0, ionization_type="acid", ph=7.4, logp_neutral=4.0
    )
    assert high_logp.permeability_factor >= low_logp.permeability_factor


# --- Validation errors ---


def test_pka_out_of_range_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_ionization_at_ph("DrugX", pka=-1.0, ionization_type="acid")


def test_ph_out_of_range_raises():
    with pytest.raises(ValueError, match="ph"):
        predict_ionization_at_ph("DrugX", pka=7.0, ionization_type="acid", ph=15.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_ionization_at_ph("DrugX", pka=7.0, ionization_type="amphoteric")
