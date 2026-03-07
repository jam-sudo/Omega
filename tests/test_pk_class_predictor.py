"""Tests for pk_class_predictor module (Phase 698)."""

import pytest

from omega_pbpk.prediction.pk_class_predictor import (
    PKClassResult,
    generate_formulation_recommendation,
    predict_pk_class,
    screen_pk_classes,
)

_VALID_BA = {"high", "moderate", "low", "parenteral_only"}
_VALID_CLASSES = {"I", "II", "III", "IV", "V", "VI"}
_VALID_CLASS_NAMES = {
    "ideal_oral",
    "low_solubility",
    "low_permeability",
    "problematic",
    "biologic",
    "cns",
}


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_pk_class_result():
    result = predict_pk_class("DrugA", logP=2.0, mw=300, psa=50)
    assert isinstance(result, PKClassResult)


# ---------------------------------------------------------------------------
# Class I — ideal oral drug
# ---------------------------------------------------------------------------


def test_class_i_or_vi_for_typical_oral_drug():
    # logP=2, MW=300, PSA=50 — could be I or VI (BBB eligible)
    result = predict_pk_class("Ibuprofen-like", logP=2.0, mw=300, psa=50)
    assert result.pk_class in {"I", "VI"}


def test_class_i_explicit_low_psa_moderate_logp():
    # logP=1.5, MW=200, PSA=40 — within BBB range → VI
    result = predict_pk_class("Drug", logP=1.5, mw=200, psa=40)
    # BBB penetrant: 1<=logP<=5, mw<450, psa<90 → class VI
    assert result.pk_class == "VI"


def test_class_i_non_bbb_profile():
    # logP=0.5 (below 1), MW=300, PSA=50 → not BBB → Class I
    result = predict_pk_class("HydroPhilic", logP=0.5, mw=300, psa=50)
    assert result.pk_class == "I"


# ---------------------------------------------------------------------------
# Class II — low solubility
# ---------------------------------------------------------------------------


def test_class_ii_for_high_logp():
    # logP=5, MW=400, PSA=50 → Class II
    result = predict_pk_class("LipophilicDrug", logP=5.0, mw=400, psa=50)
    assert result.pk_class == "II"


def test_class_ii_logp_4_boundary():
    # logP=3.5 > 3 → Class II (unless BBB/MW override)
    result = predict_pk_class("Drug", logP=3.5, mw=400, psa=50)
    assert result.pk_class == "II"


# ---------------------------------------------------------------------------
# Class III — low permeability
# ---------------------------------------------------------------------------


def test_class_iii_for_high_psa():
    # logP=-1, MW=300, PSA=120 → Class III
    result = predict_pk_class("PolarDrug", logP=-1.0, mw=300, psa=120)
    assert result.pk_class == "III"


def test_class_iii_psa_over_100():
    result = predict_pk_class("Drug", logP=0.0, mw=250, psa=110)
    assert result.pk_class == "III"


# ---------------------------------------------------------------------------
# Class IV — problematic
# ---------------------------------------------------------------------------


def test_class_iv_high_logp_and_high_psa():
    # logP=6, MW=400, PSA=130 → Class IV
    result = predict_pk_class("Problematic", logP=6.0, mw=400, psa=130)
    assert result.pk_class == "IV"


def test_class_iv_oral_bioavailability_low():
    result = predict_pk_class("Problematic", logP=6.0, mw=400, psa=130)
    assert result.oral_bioavailability_prediction == "low"


# ---------------------------------------------------------------------------
# Class V — biologic
# ---------------------------------------------------------------------------


def test_class_v_for_large_mw():
    result = predict_pk_class("mAb", logP=2.0, mw=1500, psa=100)
    assert result.pk_class == "V"


def test_class_v_parenteral_only():
    result = predict_pk_class("Biologic", logP=2.0, mw=2000, psa=200)
    assert result.oral_bioavailability_prediction == "parenteral_only"


# ---------------------------------------------------------------------------
# Class VI — CNS
# ---------------------------------------------------------------------------


def test_class_vi_cns_profile():
    # logP=3, MW=300, PSA=60 → BBB penetrant and no high logP override → VI
    result = predict_pk_class("CNSDrug", logP=3.0, mw=300, psa=60)
    assert result.pk_class == "VI"


# ---------------------------------------------------------------------------
# Class name and BA prediction validity
# ---------------------------------------------------------------------------


def test_pk_class_name_in_expected_set():
    result = predict_pk_class("Drug", logP=2.0, mw=300, psa=50)
    assert result.pk_class_name in _VALID_CLASS_NAMES


def test_oral_bioavailability_prediction_in_expected_set():
    result = predict_pk_class("Drug", logP=2.0, mw=300, psa=50)
    assert result.oral_bioavailability_prediction in _VALID_BA


def test_notes_nonempty():
    result = predict_pk_class("Drug", logP=2.0, mw=300, psa=50)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Formulation recommendation
# ---------------------------------------------------------------------------


def test_generate_formulation_recommendation_nonempty():
    result = predict_pk_class("Drug", logP=2.0, mw=300, psa=50)
    rec = generate_formulation_recommendation(result)
    assert isinstance(rec, str)
    assert len(rec) > 0


def test_class_ii_recommendation_contains_key_terms():
    result = predict_pk_class("LipophilicDrug", logP=5.0, mw=400, psa=50)
    assert result.pk_class == "II"
    rec = generate_formulation_recommendation(result)
    assert any(kw in rec.lower() for kw in ["solid dispersion", "nano", "lipid"])


def test_class_i_recommendation_standard_tablet():
    result = predict_pk_class("HydroPhilic", logP=0.5, mw=300, psa=50)
    assert result.pk_class == "I"
    rec = generate_formulation_recommendation(result)
    assert "oral" in rec.lower() or "tablet" in rec.lower() or "capsule" in rec.lower()


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


def test_screen_pk_classes_sorted_ascending():
    compounds = [
        {"compound_name": "A", "logP": 6.0, "mw": 400, "psa": 130},  # IV
        {"compound_name": "B", "logP": 0.5, "mw": 300, "psa": 50},  # I
        {"compound_name": "C", "logP": -1.0, "mw": 300, "psa": 120},  # III
        {"compound_name": "D", "logP": 5.0, "mw": 400, "psa": 50},  # II
    ]
    results = screen_pk_classes(compounds)
    classes = [r.pk_class for r in results]
    assert classes == sorted(classes)


def test_screen_pk_classes_empty_returns_empty():
    results = screen_pk_classes([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_pk_class("Drug", logP=2.0, mw=0, psa=50)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_pk_class("Drug", logP=2.0, mw=-100, psa=50)


def test_validation_psa_negative_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_pk_class("Drug", logP=2.0, mw=300, psa=-10)
