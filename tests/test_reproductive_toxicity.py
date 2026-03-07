"""Tests for reproductive toxicity risk assessment module."""

import pytest

from omega_pbpk.risk.reproductive_toxicity import (
    ReproductiveToxicityResult,
    assess_reproductive_toxicity,
    screen_reproductive_toxicity,
)

# --- Helper ---


def _assess(name="TestDrug", mw=300.0, logP=1.5, **features):
    return assess_reproductive_toxicity(name, mw, logP, features)


# --- Basic return type tests ---


def test_returns_result_type():
    result = _assess()
    assert isinstance(result, ReproductiveToxicityResult)


def test_no_alerts_score_zero():
    result = _assess()
    assert result.reproductive_toxicity_score == 0.0


def test_no_alerts_risk_class_none():
    result = _assess()
    assert result.risk_class == "none"


def test_no_alerts_no_structural_alerts():
    result = _assess()
    assert result.n_structural_alerts == 0


# --- Structural alert scoring ---


def test_thalidomide_like_raises_score():
    result = _assess(n_thalidomide_like=1)
    assert result.reproductive_toxicity_score >= 45.0


def test_thalidomide_like_moderate_or_higher():
    result = _assess(n_thalidomide_like=1)
    assert result.risk_class in {"moderate", "high", "very_high"}


def test_heavy_metal_single_score():
    result = _assess(n_heavy_metals=1)
    assert result.reproductive_toxicity_score >= 40.0


def test_nitrosamine_score():
    result = _assess(n_nitrosamines=1)
    assert result.reproductive_toxicity_score >= 35.0


def test_retinoid_like_score():
    result = _assess(n_retinoid_like=1)
    assert result.reproductive_toxicity_score >= 30.0


def test_alkylating_score():
    result = _assess(n_alkylating=1)
    assert result.reproductive_toxicity_score >= 25.0


def test_estrogen_like_score():
    result = _assess(n_estrogen_like=1)
    assert result.reproductive_toxicity_score >= 20.0


def test_planar_aromatic_score():
    result = _assess(is_planar_aromatic=True)
    assert result.reproductive_toxicity_score >= 10.0


# --- Pregnancy category ---


def test_pregnancy_category_in_valid_set():
    result = _assess()
    assert result.pregnancy_category in {"A", "B", "C", "D", "X"}


def test_score_above_75_category_x():
    # Two heavy metals (80 pts) but no MW/logP penalty → score > 75
    result = _assess(mw=300.0, logP=1.5, n_heavy_metals=2)
    # 2 * 40 = 80, clamped to 100, no MW/logP adjustments
    assert result.pregnancy_category == "X"


def test_no_alerts_category_a():
    result = _assess()
    assert result.pregnancy_category == "A"


def test_moderate_score_category_c():
    # n_retinoid_like=1 → 30 pts, fits "C" range (25 < score <= 50)
    result = _assess(n_retinoid_like=1)
    assert result.pregnancy_category == "C"


# --- Derived flags ---


def test_placental_transfer_concern_small_lipophilic():
    result = assess_reproductive_toxicity("Drug", mw=300.0, logP=2.0, smiles_features={})
    assert result.placental_transfer_concern is True


def test_no_placental_concern_large_mw():
    result = assess_reproductive_toxicity("Drug", mw=700.0, logP=2.0, smiles_features={})
    assert result.placental_transfer_concern is False


def test_no_placental_concern_low_logp():
    result = assess_reproductive_toxicity("Drug", mw=300.0, logP=-2.0, smiles_features={})
    assert result.placental_transfer_concern is False


def test_teratogenicity_flag_retinoid():
    result = _assess(n_retinoid_like=1)
    assert result.teratogenicity_flag is True


def test_teratogenicity_flag_thalidomide():
    result = _assess(n_thalidomide_like=1)
    assert result.teratogenicity_flag is True


def test_no_teratogenicity_flag_no_alerts():
    result = _assess()
    assert result.teratogenicity_flag is False


def test_requires_reproductive_study_when_score_above_20():
    # n_alkylating=1 → 25 pts > 20
    result = _assess(n_alkylating=1)
    assert result.requires_reproductive_study is True


def test_no_reproductive_study_required_when_score_low():
    result = _assess()
    assert result.requires_reproductive_study is False


# --- MW and logP adjustments ---


def test_high_mw_reduces_score():
    low_mw = assess_reproductive_toxicity(
        "Drug", mw=300.0, logP=1.0, smiles_features={"n_heavy_metals": 1}
    )
    high_mw = assess_reproductive_toxicity(
        "Drug", mw=900.0, logP=1.0, smiles_features={"n_heavy_metals": 1}
    )
    assert high_mw.reproductive_toxicity_score < low_mw.reproductive_toxicity_score


def test_very_high_mw_further_reduces():
    mw800 = assess_reproductive_toxicity(
        "Drug", mw=850.0, logP=1.0, smiles_features={"n_heavy_metals": 1}
    )
    mw1100 = assess_reproductive_toxicity(
        "Drug", mw=1100.0, logP=1.0, smiles_features={"n_heavy_metals": 1}
    )
    assert mw1100.reproductive_toxicity_score < mw800.reproductive_toxicity_score


# --- screen function ---


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "mw": 300.0, "logP": 1.5, "smiles_features": {}},
        {"compound_name": "B", "mw": 300.0, "logP": 1.5, "smiles_features": {"n_retinoid_like": 1}},
    ]
    results = screen_reproductive_toxicity(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {"compound_name": "Low", "mw": 300.0, "logP": 1.5, "smiles_features": {}},
        {
            "compound_name": "High",
            "mw": 300.0,
            "logP": 1.5,
            "smiles_features": {"n_heavy_metals": 2},
        },
        {
            "compound_name": "Mid",
            "mw": 300.0,
            "logP": 1.5,
            "smiles_features": {"n_estrogen_like": 1},
        },
    ]
    results = screen_reproductive_toxicity(compounds)
    scores = [r.reproductive_toxicity_score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- Validation ---


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError):
        assess_reproductive_toxicity("Drug", mw=0.0, logP=1.0, smiles_features={})


def test_invalid_mw_negative_raises():
    with pytest.raises(ValueError):
        assess_reproductive_toxicity("Drug", mw=-100.0, logP=1.0, smiles_features={})


def test_negative_count_raises():
    with pytest.raises(ValueError):
        assess_reproductive_toxicity(
            "Drug", mw=300.0, logP=1.0, smiles_features={"n_nitrosamines": -1}
        )


def test_notes_nonempty():
    result = _assess()
    assert len(result.notes) > 0


def test_notes_nonempty_with_alerts():
    result = _assess(n_thalidomide_like=1)
    assert "thalidomide" in result.notes.lower()
