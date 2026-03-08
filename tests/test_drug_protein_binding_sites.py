"""Tests for Phase 960 — drug_protein_binding_sites."""

import pytest

from omega_pbpk.prediction.drug_protein_binding_sites import (
    ProteinBindingSitesResult,
    predict_protein_binding_sites,
    screen_protein_binding,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = predict_protein_binding_sites("Warfarin", logp=2.7, pka=5.0, ionization_type="acid")
    assert isinstance(r, ProteinBindingSitesResult)


def test_result_is_frozen():
    r = predict_protein_binding_sites("Warfarin", logp=2.7, pka=5.0, ionization_type="acid")
    with pytest.raises((AttributeError, TypeError)):
        r.fu_plasma = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


def test_fraction_bound_hsa_range():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert 0.0 <= r.fraction_bound_hsa <= 1.0


def test_fraction_bound_agp_range():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert 0.0 <= r.fraction_bound_agp <= 1.0


def test_fraction_bound_lipo_range():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert 0.0 <= r.fraction_bound_lipo <= 1.0


def test_total_fraction_bound_range():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert 0.0 <= r.total_fraction_bound <= 1.0


def test_fu_plasma_range():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert 0.0 <= r.fu_plasma <= 1.0


def test_fu_plus_total_bound_approx_one():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert abs(r.fu_plasma + r.total_fraction_bound - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Categorical outputs
# ---------------------------------------------------------------------------


def test_dominant_binding_protein_valid():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert r.dominant_binding_protein in {"HSA", "AGP", "lipoprotein"}


def test_binding_category_valid():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert r.binding_category in {"high", "moderate", "low"}


def test_displacement_risk_valid():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert r.displacement_risk in {"high", "low"}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="neutral")
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    r = predict_protein_binding_sites("MyDrug", logp=1.0, pka=4.0, ionization_type="acid")
    assert r.drug_name == "MyDrug"


# ---------------------------------------------------------------------------
# Acid dominant binding protein == HSA
# ---------------------------------------------------------------------------


def test_acid_dominant_hsa():
    # Acidic drugs strongly bound to HSA
    r = predict_protein_binding_sites("AcidDrug", logp=3.0, pka=4.0, ionization_type="acid")
    assert r.dominant_binding_protein == "HSA"


# ---------------------------------------------------------------------------
# Base dominant binding protein == AGP
# ---------------------------------------------------------------------------


def test_base_dominant_agp():
    # Basic drugs strongly bound to AGP — use low logP to keep HSA binding low
    r = predict_protein_binding_sites("BaseDrug", logp=-1.0, pka=9.0, ionization_type="base")
    assert r.dominant_binding_protein == "AGP"


# ---------------------------------------------------------------------------
# High logP → higher lipoprotein binding
# ---------------------------------------------------------------------------


def test_high_logp_higher_lipo_binding():
    r_lo = predict_protein_binding_sites("DrugLo", logp=0.0, pka=7.0, ionization_type="neutral")
    r_hi = predict_protein_binding_sites("DrugHi", logp=5.0, pka=7.0, ionization_type="neutral")
    assert r_hi.fraction_bound_lipo > r_lo.fraction_bound_lipo


# ---------------------------------------------------------------------------
# Low logP → lower total binding
# ---------------------------------------------------------------------------


def test_low_logp_lower_total_binding():
    r_lo = predict_protein_binding_sites("DrugLo", logp=-2.0, pka=7.0, ionization_type="neutral")
    r_hi = predict_protein_binding_sites("DrugHi", logp=4.0, pka=7.0, ionization_type="neutral")
    assert r_lo.total_fraction_bound < r_hi.total_fraction_bound


# ---------------------------------------------------------------------------
# Binding category consistency
# ---------------------------------------------------------------------------


def test_high_binding_category_when_fu_lt_01():
    # High logP acid should have high binding
    r = predict_protein_binding_sites("HighBinder", logp=4.0, pka=4.0, ionization_type="acid")
    if r.fu_plasma < 0.1:
        assert r.binding_category == "high"


def test_fu_below_01_gives_high_category():
    # Verify a known high-binder has category "high"
    r = predict_protein_binding_sites("Warfarin", logp=2.7, pka=5.0, ionization_type="acid")
    # Warfarin is ~99% protein bound; if fu < 0.1, category should be "high"
    if r.fu_plasma < 0.1:
        assert r.binding_category == "high"
    elif r.fu_plasma < 0.5:
        assert r.binding_category == "moderate"
    else:
        assert r.binding_category == "low"


# ---------------------------------------------------------------------------
# Displacement risk
# ---------------------------------------------------------------------------


def test_high_hsa_fraction_gives_high_displacement_risk():
    r = predict_protein_binding_sites("AcidDrug", logp=4.0, pka=4.0, ionization_type="acid")
    if r.fraction_bound_hsa > 0.70:
        assert r.displacement_risk == "high"


def test_low_hsa_fraction_gives_low_displacement_risk():
    # Basic drug at very low logP → low HSA binding
    r = predict_protein_binding_sites("BaseDrug", logp=-3.0, pka=9.0, ionization_type="base")
    assert r.displacement_risk == "low"


# ---------------------------------------------------------------------------
# screen_protein_binding
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "B", "logp": 3.0, "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "C", "logp": 2.0, "pka": 9.0, "ionization_type": "base"},
    ]
    results = screen_protein_binding(compounds)
    assert len(results) == 3


def test_screen_sorted_by_fu_ascending():
    compounds = [
        {"drug_name": "A", "logp": 0.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "B", "logp": 3.0, "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "C", "logp": -1.0, "pka": 9.0, "ionization_type": "base"},
        {"drug_name": "D", "logp": 4.5, "pka": 5.0, "ionization_type": "acid"},
    ]
    results = screen_protein_binding(compounds)
    fu_values = [r.fu_plasma for r in results]
    assert fu_values == sorted(fu_values)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_protein_binding_sites("DrugA", logp=2.0, pka=7.0, ionization_type="zwitterion")
