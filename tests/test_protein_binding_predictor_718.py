"""Tests for Phase 718 additions to omega_pbpk.prediction.protein_binding_predictor.

Tests the albumin/AGP mechanistic binding predictor functions:
predict_albumin_binding, predict_agp_binding, predict_total_fup, screen_protein_binding,
ProteinBindingResult.
"""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.protein_binding_predictor import (
    ProteinBindingResult,
    predict_agp_binding,
    predict_albumin_binding,
    predict_total_fup,
    screen_protein_binding,
)

# ---------------------------------------------------------------------------
# predict_albumin_binding
# ---------------------------------------------------------------------------


def test_predict_albumin_binding_returns_dict():
    result = predict_albumin_binding(logP=2.0, mw=350.0, psa=60.0)
    assert isinstance(result, dict)


def test_albumin_binding_has_required_keys():
    result = predict_albumin_binding(logP=2.0, mw=350.0, psa=60.0)
    assert "fub_albumin" in result
    assert "kd_uM" in result
    assert "bound_fraction" in result
    assert "notes" in result


def test_albumin_fub_in_range():
    result = predict_albumin_binding(logP=3.0, mw=400.0, psa=80.0)
    assert 0.0 < result["fub_albumin"] < 1.0


def test_high_logP_lower_fub_albumin():
    """More lipophilic → more albumin binding → lower fub."""
    r_low = predict_albumin_binding(logP=0.0, mw=300.0, psa=60.0)
    r_high = predict_albumin_binding(logP=5.0, mw=300.0, psa=60.0)
    assert r_high["fub_albumin"] < r_low["fub_albumin"]


def test_low_logP_higher_fub_albumin():
    """Very low logP → less bound → higher fub_albumin."""
    result = predict_albumin_binding(logP=-2.0, mw=200.0, psa=100.0)
    assert result["fub_albumin"] > 0.5


def test_albumin_notes_nonempty():
    result = predict_albumin_binding(logP=2.0, mw=350.0, psa=60.0)
    assert len(result["notes"]) > 0


def test_albumin_bound_fraction_consistent():
    result = predict_albumin_binding(logP=3.0, mw=400.0, psa=50.0)
    assert result["bound_fraction"] == pytest.approx(1.0 - result["fub_albumin"], rel=1e-9)


# ---------------------------------------------------------------------------
# predict_agp_binding
# ---------------------------------------------------------------------------


def test_predict_agp_binding_returns_dict():
    result = predict_agp_binding(logP=2.0)
    assert isinstance(result, dict)


def test_agp_binding_has_required_keys():
    result = predict_agp_binding(logP=2.0)
    assert "fub_agp" in result
    assert "kd_agp_uM" in result
    assert "notes" in result


def test_agp_fub_in_range():
    result = predict_agp_binding(logP=3.0)
    assert 0.0 < result["fub_agp"] < 1.0


def test_basic_drug_lower_fub_agp():
    """Basic drug (high pKa) should bind more to AGP → lower fub_agp."""
    r_neutral = predict_agp_binding(logP=3.0, pka_base=None)
    r_basic = predict_agp_binding(logP=3.0, pka_base=9.0)
    assert r_basic["fub_agp"] < r_neutral["fub_agp"]


def test_agp_notes_nonempty():
    result = predict_agp_binding(logP=2.0, pka_base=8.5)
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# predict_total_fup
# ---------------------------------------------------------------------------


def test_predict_total_fup_returns_protein_binding_result():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0)
    assert isinstance(result, ProteinBindingResult)


def test_fup_total_in_range():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0)
    assert 0.0 < result.fup_total <= 1.0


def test_protein_binding_pct_matches_fup():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0)
    expected_pct = (1.0 - result.fup_total) * 100.0
    assert result.protein_binding_pct == pytest.approx(expected_pct, rel=1e-6)


def test_binding_class_valid_value():
    result = predict_total_fup(logP=3.0, mw=400.0, psa=50.0)
    assert result.binding_class in {"low", "moderate", "high", "very_high"}


def test_high_logP_high_binding_class():
    """Very lipophilic drug should be 'high' or 'very_high' binding."""
    result = predict_total_fup(logP=6.0, mw=500.0, psa=20.0)
    assert result.binding_class in {"high", "very_high"}


def test_total_fup_notes_nonempty():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0, compound_name="DrugA")
    assert len(result.notes) > 0


def test_result_is_frozen_dataclass():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0)
    with pytest.raises((AttributeError, TypeError)):
        result.fup_total = 0.5  # type: ignore[misc]


def test_ddi_risk_valid_value():
    result = predict_total_fup(logP=2.0, mw=350.0, psa=60.0)
    assert result.drug_drug_interaction_risk in {"low", "moderate", "high"}


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_total_fup(logP=2.0, mw=0.0, psa=60.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_total_fup(logP=2.0, mw=-100.0, psa=60.0)


def test_validation_psa_negative_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_total_fup(logP=2.0, mw=350.0, psa=-1.0)


# ---------------------------------------------------------------------------
# screen_protein_binding
# ---------------------------------------------------------------------------


def test_screen_protein_binding_returns_list():
    compounds = [
        {"logP": 1.0, "mw": 200.0, "psa": 80.0, "compound_name": "A"},
        {"logP": 4.0, "mw": 450.0, "psa": 30.0, "compound_name": "B"},
        {"logP": 2.5, "mw": 350.0, "psa": 60.0, "compound_name": "C"},
    ]
    results = screen_protein_binding(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_ascending_by_fup():
    compounds = [
        {"logP": 0.5, "mw": 200.0, "psa": 100.0, "compound_name": "LowBinder"},
        {"logP": 5.0, "mw": 500.0, "psa": 20.0, "compound_name": "HighBinder"},
        {"logP": 2.5, "mw": 350.0, "psa": 60.0, "compound_name": "MidBinder"},
    ]
    results = screen_protein_binding(compounds)
    fup_values = [r.fup_total for r in results]
    assert fup_values == sorted(fup_values)


def test_screen_all_are_protein_binding_result():
    compounds = [
        {"logP": 2.0, "mw": 300.0, "psa": 70.0},
        {"logP": 3.5, "mw": 400.0, "psa": 40.0},
    ]
    results = screen_protein_binding(compounds)
    for r in results:
        assert isinstance(r, ProteinBindingResult)
