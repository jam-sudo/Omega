"""Tests for Phase 970 — drug pKa prediction."""

import pytest

from omega_pbpk.prediction.drug_pka_prediction import (
    PKAPredictionResult,
    predict_pka,
    screen_pka_profiles,
)

# ---------------------------------------------------------------------------
# Return type and basic checks
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_pka("DrugA", logp=1.5, mw=300.0, functional_group="carboxylic_acid")
    assert isinstance(result, PKAPredictionResult)


def test_frozen_dataclass():
    result = predict_pka("DrugA", logp=1.5, mw=300.0, functional_group="carboxylic_acid")
    with pytest.raises((AttributeError, TypeError)):
        result.pka_predicted = 99.0  # type: ignore[misc]


def test_pka_predicted_positive():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="aliphatic_amine")
    assert result.pka_predicted > 0


def test_pka_type_valid():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert result.pka_type in {"acid", "base", "neutral"}


def test_fraction_ionized_physiological_in_range():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert 0.0 <= result.fraction_ionized_at_physiological_ph <= 1.0


def test_fraction_ionized_gastric_in_range():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert 0.0 <= result.fraction_ionized_at_gastric_ph <= 1.0


def test_fraction_ionized_intestinal_in_range():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert 0.0 <= result.fraction_ionized_at_intestinal_ph <= 1.0


def test_effective_solubility_ratio_positive():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert result.effective_solubility_ratio_intestinal > 0


def test_notes_nonempty():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_pka("Ibuprofen", logp=3.5, mw=206.0, functional_group="carboxylic_acid")
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Functional group specific behaviour
# ---------------------------------------------------------------------------


def test_carboxylic_acid_type():
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="carboxylic_acid")
    assert result.pka_type == "acid"


def test_aliphatic_amine_type():
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="aliphatic_amine")
    assert result.pka_type == "base"


def test_carboxylic_acid_mostly_ionized_at_physiological_ph():
    """pKa ~4.2 → at pH 7.4 should be mostly ionized (> 0.9)."""
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="carboxylic_acid")
    assert result.fraction_ionized_at_physiological_ph > 0.9


def test_aliphatic_amine_mostly_ionized_at_gastric_ph():
    """pKa ~9.5 → at pH 1.2 amine is mostly protonated (ionized fraction > 0.9)."""
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="aliphatic_amine")
    assert result.fraction_ionized_at_gastric_ph > 0.9


# ---------------------------------------------------------------------------
# screen_pka_profiles
# ---------------------------------------------------------------------------


def test_screen_pka_profiles_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 200.0, "functional_group": "carboxylic_acid"},
        {"drug_name": "B", "logp": 2.0, "mw": 300.0, "functional_group": "aliphatic_amine"},
        {"drug_name": "C", "logp": 0.5, "mw": 150.0, "functional_group": "phenol"},
    ]
    results = screen_pka_profiles(compounds)
    assert len(results) == 3


def test_screen_pka_profiles_sorted():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 200.0, "functional_group": "aliphatic_amine"},
        {"drug_name": "B", "logp": 1.0, "mw": 200.0, "functional_group": "carboxylic_acid"},
        {"drug_name": "C", "logp": 1.0, "mw": 200.0, "functional_group": "imidazole"},
    ]
    results = screen_pka_profiles(compounds)
    pkas = [r.pka_predicted for r in results]
    assert pkas == sorted(pkas)


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def test_logp_correction_shifts_pka():
    """Higher logP → higher predicted pKa."""
    r_low = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="carboxylic_acid")
    r_high = predict_pka("DrugA", logp=5.0, mw=200.0, functional_group="carboxylic_acid")
    assert r_high.pka_predicted > r_low.pka_predicted


def test_mw_correction_large_molecule():
    """MW > 500 should add +0.3 to pKa compared to same compound with MW <= 500."""
    r_small = predict_pka("DrugA", logp=1.0, mw=300.0, functional_group="carboxylic_acid")
    r_large = predict_pka("DrugA", logp=1.0, mw=600.0, functional_group="carboxylic_acid")
    assert abs(r_large.pka_predicted - r_small.pka_predicted - 0.3) < 1e-9


def test_unknown_functional_group_near_7():
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="unknown")
    assert abs(result.pka_predicted - 7.0) < 1.0  # ~7.0 with logP correction of 0


def test_unknown_is_neutral():
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="unknown")
    assert result.pka_type == "neutral"


# ---------------------------------------------------------------------------
# charge_at_physiological_ph
# ---------------------------------------------------------------------------


def test_charge_at_physiological_ph_is_float():
    result = predict_pka("DrugA", logp=1.0, mw=250.0, functional_group="carboxylic_acid")
    assert isinstance(result.charge_at_physiological_ph, float)


def test_acid_charge_negative_at_physiological():
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="carboxylic_acid")
    # Mostly ionized at pH 7.4 → charge should be negative
    assert result.charge_at_physiological_ph < 0


def test_base_charge_positive_at_physiological_low_pka():
    """Aromatic amine pKa ~4.6 < 7.4 → mostly unprotonated at physiological pH → small + charge."""
    result = predict_pka("DrugA", logp=0.0, mw=200.0, functional_group="aromatic_amine")
    # pKa < pH → mostly uncharged, but charge_at_physiological_ph should be >= 0
    assert result.charge_at_physiological_ph >= 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_pka("DrugA", logp=1.0, mw=0.0, functional_group="carboxylic_acid")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_pka("DrugA", logp=1.0, mw=-100.0, functional_group="carboxylic_acid")


def test_validation_unknown_group():
    with pytest.raises(ValueError, match="functional_group"):
        predict_pka("DrugA", logp=1.0, mw=200.0, functional_group="ether_oxygen")
