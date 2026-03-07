"""Tests for Phase 934: Drug Permeability Classifier."""

import pytest
from omega_pbpk.prediction.drug_permeability_classifier import (
    DrugPermeabilityClassResult,
    classify_permeability,
    screen_permeability_class,
)


# ---- Basic return type and properties ----

def test_return_type():
    result = classify_permeability("AspIrin", logp=1.5, mw=180.0, hbd=1, hba=4, rotatable_bonds=3)
    assert isinstance(result, DrugPermeabilityClassResult)


def test_frozen_dataclass():
    result = classify_permeability("DrugA", logp=1.5, mw=200.0, hbd=1, hba=3, rotatable_bonds=2)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "OtherDrug"  # type: ignore[misc]


def test_permeability_class_valid():
    result = classify_permeability("DrugB", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.permeability_class in {"high", "medium", "low"}


def test_lipinski_violations_nonneg():
    result = classify_permeability("DrugC", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.lipinski_violations >= 0


def test_fails_ro5_is_bool():
    result = classify_permeability("DrugD", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert isinstance(result.fails_ro5, bool)


def test_oral_bioavailability_risk_valid():
    result = classify_permeability("DrugE", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.oral_bioavailability_risk in {"low", "moderate", "high"}


def test_veber_compliant_is_bool():
    result = classify_permeability("DrugF", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert isinstance(result.veber_compliant, bool)


def test_bbb_likelihood_valid():
    result = classify_permeability("DrugG", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.bbb_likelihood in {"likely", "unlikely"}


def test_psa_estimated_nonneg():
    result = classify_permeability("DrugH", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.psa_estimated >= 0


def test_notes_nonempty():
    result = classify_permeability("DrugI", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.notes and len(result.notes) > 0


def test_drug_name_preserved():
    result = classify_permeability("MyDrug", logp=1.5, mw=300.0, hbd=1, hba=4, rotatable_bonds=3)
    assert result.drug_name == "MyDrug"


# ---- Classification correctness ----

def test_high_logp_high_mw_fails_ro5():
    # logP=6 (>5 violation), MW=600 (>500 violation) → 2 violations → fails_ro5
    result = classify_permeability("BigDrug", logp=6.0, mw=600.0, hbd=2, hba=5, rotatable_bonds=5)
    assert result.fails_ro5 is True


def test_low_mw_logp2_low_hbd_high_perm():
    # logP=2, MW=200, PSA should be low, HBD=1 → "high"
    result = classify_permeability("SmallDrug", logp=2.0, mw=200.0, hbd=1, hba=3, rotatable_bonds=3)
    assert result.permeability_class == "high"


def test_high_mw_many_violations():
    # MW=700, logP=6 → fails_ro5 should be True or many violations
    result = classify_permeability("HeavyDrug", logp=6.0, mw=700.0, hbd=2, hba=5, rotatable_bonds=5)
    assert result.lipinski_violations >= 2
    assert result.fails_ro5 is True


def test_high_psa_not_veber():
    # To get PSA > 140 we need appropriate MW and low logP
    # PSA = MW * 0.12 * (1 - 0.08*logP) + 5
    # Try MW=1200, logP=0 → PSA = 1200*0.12*(1-0) + 5 = 144 + 5 = 149 > 140
    result = classify_permeability("BigPSA", logp=0.0, mw=1200.0, hbd=3, hba=8, rotatable_bonds=5)
    assert result.veber_compliant is False


def test_low_psa_rotb_le10_veber():
    # Low MW, moderate logP → low PSA; rotatable_bonds=5
    result = classify_permeability("SmallMol", logp=3.0, mw=200.0, hbd=1, hba=2, rotatable_bonds=5)
    assert result.veber_compliant is True


def test_bbb_likely():
    # logP=2, MW=300, PSA will be moderate
    result = classify_permeability("CNSDrug", logp=2.0, mw=300.0, hbd=1, hba=3, rotatable_bonds=4)
    assert result.bbb_likelihood == "likely"


def test_bbb_unlikely_high_mw():
    # MW=700 → fails MW < 450
    result = classify_permeability("LargeDrug", logp=2.0, mw=700.0, hbd=2, hba=5, rotatable_bonds=6)
    assert result.bbb_likelihood == "unlikely"


# ---- screen_permeability_class ----

def test_screen_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 2.0, "mw": 200.0, "hbd": 1, "hba": 3, "rotatable_bonds": 3},
        {"drug_name": "B", "logp": 6.0, "mw": 600.0, "hbd": 6, "hba": 11, "rotatable_bonds": 15},
        {"drug_name": "C", "logp": 1.5, "mw": 400.0, "hbd": 2, "hba": 6, "rotatable_bonds": 6},
    ]
    results = screen_permeability_class(compounds)
    assert len(results) == 3


def test_screen_sorted_high_before_low():
    compounds = [
        {"drug_name": "Low", "logp": 6.0, "mw": 700.0, "hbd": 6, "hba": 12, "rotatable_bonds": 15},
        {"drug_name": "High", "logp": 2.0, "mw": 200.0, "hbd": 1, "hba": 2, "rotatable_bonds": 3},
        {"drug_name": "Med", "logp": 1.5, "mw": 450.0, "hbd": 2, "hba": 6, "rotatable_bonds": 7},
    ]
    results = screen_permeability_class(compounds)
    classes = [r.permeability_class for r in results]
    order = {"high": 0, "medium": 1, "low": 2}
    sorted_classes = sorted(classes, key=lambda c: order[c])
    assert classes == sorted_classes


# ---- Validation errors ----

def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        classify_permeability("X", logp=1.0, mw=0.0, hbd=1, hba=2, rotatable_bonds=3)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        classify_permeability("X", logp=1.0, mw=-100.0, hbd=1, hba=2, rotatable_bonds=3)


def test_validation_hbd_negative_raises():
    with pytest.raises(ValueError, match="hbd"):
        classify_permeability("X", logp=1.0, mw=300.0, hbd=-1, hba=2, rotatable_bonds=3)


def test_validation_hba_negative_raises():
    with pytest.raises(ValueError, match="hba"):
        classify_permeability("X", logp=1.0, mw=300.0, hbd=1, hba=-1, rotatable_bonds=3)


def test_validation_rotatable_bonds_negative_raises():
    with pytest.raises(ValueError, match="rotatable_bonds"):
        classify_permeability("X", logp=1.0, mw=300.0, hbd=1, hba=2, rotatable_bonds=-1)


def test_validation_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        classify_permeability("X", logp=1.0, mw=300.0, hbd=1, hba=2, rotatable_bonds=3,
                              ionization_type="zwitterion")


# ---- Ro5 / oral risk consistency ----

def test_fails_ro5_when_2_violations():
    # logP=6 (>5) and MW=600 (>500) → 2 violations
    result = classify_permeability("Test", logp=6.0, mw=600.0, hbd=1, hba=5, rotatable_bonds=5)
    assert result.fails_ro5 is True
    assert result.lipinski_violations >= 2


def test_oral_risk_high_when_fails_ro5():
    result = classify_permeability("RiskyDrug", logp=6.0, mw=600.0, hbd=6, hba=11, rotatable_bonds=12)
    assert result.fails_ro5 is True
    assert result.oral_bioavailability_risk == "high"
