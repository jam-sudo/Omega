"""Platinum reference schema validation tests."""

import pytest

from omega_pbpk.data.platinum_schema import (
    ValidationError,
    load_platinum_reference,
    save_platinum_reference,
    validate_entry,
)


def _valid_entry() -> dict:
    return {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "dose_mg": 100.0,
        "cmax_mg_L": 1.74,
        "source_type": "fda_label",
        "source_id": "NDA 020863",
        "fasted_confidence": "confirmed_fasted",
        "formulation": "IR",
        "route": "oral",
        "population": "healthy",
        "single_dose": True,
        "tuning_contaminated": False,
        "nonlinear_pk": False,
        "data_quality": "fda_label_exact",
    }


def test_valid_entry_passes():
    entry = validate_entry("caffeine", _valid_entry())
    assert entry.drug_name == "caffeine"
    assert entry.cmax_mg_L == 1.74


def test_missing_smiles_fails():
    d = _valid_entry()
    del d["smiles"]
    with pytest.raises(ValidationError, match="smiles"):
        validate_entry("caffeine", d)


def test_bad_cmax_dose_ratio_fails():
    d = _valid_entry()
    d["cmax_mg_L"] = 500.0  # ratio 5.0, above 1.0 threshold
    with pytest.raises(ValidationError, match="ratio"):
        validate_entry("caffeine", d)


def test_invalid_fasted_confidence_fails():
    d = _valid_entry()
    d["fasted_confidence"] = "unknown"
    with pytest.raises(ValidationError, match="fasted_confidence"):
        validate_entry("caffeine", d)


def test_non_oral_route_fails():
    d = _valid_entry()
    d["route"] = "iv"
    with pytest.raises(ValidationError, match="route"):
        validate_entry("caffeine", d)


def test_roundtrip_json(tmp_path):
    ref = {"caffeine": _valid_entry()}
    path = tmp_path / "test_ref.json"
    save_platinum_reference(ref, path)
    loaded = load_platinum_reference(path)
    assert "caffeine" in loaded
    assert loaded["caffeine"]["cmax_mg_L"] == 1.74


def test_optional_auc_field():
    d = _valid_entry()
    d["auc_mg_h_L"] = 14.2
    entry = validate_entry("caffeine", d)
    assert entry.auc_mg_h_L == 14.2
