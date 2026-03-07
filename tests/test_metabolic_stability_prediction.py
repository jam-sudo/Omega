"""Tests for Phase 996 — metabolic stability prediction."""

import pytest

from omega_pbpk.prediction.metabolic_stability_prediction import (
    MetabolicStabilityResult,
    predict_metabolic_stability,
    screen_metabolic_stability,
)

_VALID_PATHWAYS = {"CYP3A4", "CYP2D6", "CYP2C9", "esterase", "other"}
_VALID_CLASSES = {"stable", "moderate", "unstable"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(logp=1.0, mw=350.0, psa=80.0, **kwargs):
    return predict_metabolic_stability(
        drug_name="TestDrug", logp=logp, mw=mw, psa=psa, **kwargs
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_metabolic_stability_result():
    assert isinstance(_run(), MetabolicStabilityResult)


def test_drug_name_preserved():
    result = predict_metabolic_stability("Ibuprofen", logp=3.5, mw=206.0)
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Core PK outputs
# ---------------------------------------------------------------------------

def test_clint_positive():
    assert _run().clint_uL_min_per_mg_protein > 0.0


def test_t_half_positive():
    assert _run().t_half_min > 0.0


def test_extraction_ratio_in_range():
    er = _run().extraction_ratio
    assert 0.0 < er < 1.0


def test_fu_mic_in_range():
    fu = _run().fu_mic
    assert 0.0 < fu <= 1.0


def test_predicted_cls_positive():
    assert _run().predicted_cls_mL_min_per_kg > 0.0


# ---------------------------------------------------------------------------
# Categorical outputs
# ---------------------------------------------------------------------------

def test_stability_class_valid():
    assert _run().stability_class in _VALID_CLASSES


def test_major_metabolic_pathway_valid():
    assert _run().major_metabolic_pathway in _VALID_PATHWAYS


# ---------------------------------------------------------------------------
# Pathway assignment rules
# ---------------------------------------------------------------------------

def test_ester_gives_esterase_pathway():
    result = _run(has_ester=True)
    assert result.major_metabolic_pathway == "esterase"


def test_basic_amine_gives_cyp2d6():
    result = _run(logp=0.5, ionization_type="base", pka=9.0, has_ester=False)
    assert result.major_metabolic_pathway == "CYP2D6"


def test_acid_pka_range_gives_cyp2c9():
    result = _run(logp=0.5, ionization_type="acid", pka=4.5, has_ester=False)
    assert result.major_metabolic_pathway == "CYP2C9"


def test_high_logp_gives_cyp3a4():
    result = _run(logp=3.0, ionization_type="neutral", has_ester=False)
    assert result.major_metabolic_pathway == "CYP3A4"


# ---------------------------------------------------------------------------
# Stability class thresholds
# ---------------------------------------------------------------------------

def test_stable_drug_t_half_over_60():
    """Low CLint drug should have t½ > 60 min (stable)."""
    # Force a low CLint: low logP, neutral, large MW, high PSA, no ester
    result = _run(logp=-1.0, mw=500.0, psa=150.0, ionization_type="neutral", has_ester=False)
    assert result.t_half_min > 60.0
    assert result.stability_class == "stable"


def test_unstable_drug_t_half_under_30():
    """High CLint drug should have t½ < 30 min (unstable)."""
    # Force a high CLint: high logP, basic amine, small MW, ester
    result = _run(logp=4.0, mw=200.0, psa=30.0, ionization_type="base", pka=9.0, has_ester=True)
    assert result.t_half_min < 30.0
    assert result.stability_class == "unstable"


def test_high_logp_increases_clint_vs_low():
    high_logp = _run(logp=4.0, mw=350.0, psa=60.0, ionization_type="neutral")
    low_logp = _run(logp=-1.0, mw=350.0, psa=60.0, ionization_type="neutral")
    assert high_logp.clint_uL_min_per_mg_protein > low_logp.clint_uL_min_per_mg_protein


# ---------------------------------------------------------------------------
# screen_metabolic_stability
# ---------------------------------------------------------------------------

def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 300.0},
        {"drug_name": "B", "logp": 4.0, "mw": 200.0, "has_ester": True},
        {"drug_name": "C", "logp": -1.0, "mw": 450.0},
    ]
    results = screen_metabolic_stability(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_descending_t_half():
    compounds = [
        {"drug_name": "High_CLint", "logp": 4.0, "mw": 200.0, "has_ester": True},
        {"drug_name": "Low_CLint", "logp": -1.0, "mw": 500.0, "psa": 150.0},
        {"drug_name": "Mid_CLint", "logp": 2.0, "mw": 350.0},
    ]
    results = screen_metabolic_stability(compounds)
    t_halves = [r.t_half_min for r in results]
    assert t_halves == sorted(t_halves, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_metabolic_stability("X", logp=1.0, mw=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_metabolic_stability("X", logp=1.0, mw=-100.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_metabolic_stability("X", logp=1.0, mw=300.0, ionization_type="zwitterion")
