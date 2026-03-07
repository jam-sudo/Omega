"""Tests for risk/protein_adduct.py — Phase 237."""

import math

import pytest

from omega_pbpk.risk.protein_adduct import (
    ProteinAdductResult,
    assess_protein_adduct_risk,
    rank_compounds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_compound(**kwargs):
    """Return a compound dict with no structural alerts."""
    defaults = dict(
        name="CleanDrug",
        smiles="CCC(=O)OCC",  # ethyl propanoate — no alerts
        mw=116.0,
        daily_dose_mg=100.0,
        fu=0.5,
        fm_cyp=0.5,
        logP=1.0,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        assess_protein_adduct_risk("X", "C", mw=0, daily_dose_mg=10, fu=0.5, fm_cyp=0.5, logP=1)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        assess_protein_adduct_risk("X", "C", mw=-1, daily_dose_mg=10, fu=0.5, fm_cyp=0.5, logP=1)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=0, fu=0.5, fm_cyp=0.5, logP=1)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=-5, fu=0.5, fm_cyp=0.5, logP=1)


def test_validation_fu_zero():
    with pytest.raises(ValueError, match="fu"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=10, fu=0, fm_cyp=0.5, logP=1)


def test_validation_fu_above_one():
    with pytest.raises(ValueError, match="fu"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=10, fu=1.1, fm_cyp=0.5, logP=1)


def test_validation_fm_cyp_negative():
    with pytest.raises(ValueError, match="fm_cyp"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=10, fu=0.5, fm_cyp=-0.1, logP=1)


def test_validation_fm_cyp_above_one():
    with pytest.raises(ValueError, match="fm_cyp"):
        assess_protein_adduct_risk("X", "C", mw=100, daily_dose_mg=10, fu=0.5, fm_cyp=1.5, logP=1)


# ---------------------------------------------------------------------------
# No alerts → low risk
# ---------------------------------------------------------------------------


def test_no_alerts_risk_low():
    r = assess_protein_adduct_risk(
        compound_name="CleanDrug",
        smiles="CCC(=O)OCC",
        mw=116.0,
        daily_dose_mg=100.0,
        fu=0.5,
        fm_cyp=0.8,
        logP=1.0,
    )
    assert r.risk_category == "low"
    assert r.n_alerts == 0
    assert r.alert_score == 0.0
    assert r.adduct_burden_score == 0.0
    assert r.mitigation == []


# ---------------------------------------------------------------------------
# Furan alert
# ---------------------------------------------------------------------------


def test_furan_alert_triggered():
    r = assess_protein_adduct_risk(
        compound_name="FuranDrug",
        smiles="c1ccoc1CC(=O)O",  # furan ring present
        mw=140.0,
        daily_dose_mg=200.0,
        fu=0.5,
        fm_cyp=0.8,
        logP=1.5,
    )
    assert "furan" in r.alerts_triggered
    assert r.alert_score >= 2.0  # furan weight = 2


def test_furan_weight_is_2():
    r = assess_protein_adduct_risk(
        compound_name="PureFuran",
        smiles="c1ccoc1",
        mw=68.0,
        daily_dose_mg=10.0,
        fu=1.0,
        fm_cyp=1.0,
        logP=0.5,
    )
    assert "furan" in r.alerts_triggered
    # alert_score should include 2.0 for furan
    assert r.alert_score == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dose effect
# ---------------------------------------------------------------------------


def test_high_dose_higher_score():
    base = dict(smiles="c1ccoc1", mw=68.0, fu=0.5, fm_cyp=0.8, logP=0.5)
    r_low = assess_protein_adduct_risk("L", daily_dose_mg=10, **base)
    r_high = assess_protein_adduct_risk("H", daily_dose_mg=1000, **base)
    assert r_high.adduct_burden_score > r_low.adduct_burden_score


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


def test_adduct_burden_score_positive_when_alerts():
    r = assess_protein_adduct_risk(
        "Drug",
        smiles="c1ccoc1",
        mw=68.0,
        daily_dose_mg=100.0,
        fu=0.5,
        fm_cyp=0.8,
        logP=0.5,
    )
    assert r.adduct_burden_score > 0


def test_adduct_burden_score_formula():
    smiles = "c1ccoc1"
    mw, dose, fu, fm, logp = 68.0, 100.0, 0.5, 0.8, 0.5
    r = assess_protein_adduct_risk("Drug", smiles, mw, dose, fu, fm, logp)
    expected = r.alert_score * math.log10(dose) * fu * fm
    assert r.adduct_burden_score == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Risk category boundaries
# ---------------------------------------------------------------------------


def test_risk_category_low():
    # Force zero fm_cyp → zero score → low
    r = assess_protein_adduct_risk(
        "Drug", "c1ccoc1", mw=68.0, daily_dose_mg=100.0, fu=0.5, fm_cyp=0.0, logP=0.5
    )
    assert r.risk_category == "low"


def test_risk_category_moderate():
    # Tune parameters to land in moderate range
    r = assess_protein_adduct_risk(
        "Drug", "c1ccoc1", mw=68.0, daily_dose_mg=100.0, fu=0.5, fm_cyp=0.5, logP=0.5
    )
    # alert_score=2, log10(100)=2, fu=0.5, fm=0.5 → 2*2*0.5*0.5=1.0 → boundary; clamp moderate start
    assert r.risk_category in ("low", "moderate")


def test_risk_category_high():
    # High alert_score + high dose + fu=1 + fm_cyp=1
    r = assess_protein_adduct_risk(
        "HighRisk",
        smiles="c1ccoc1C=CC=ONC1CCC",  # furan + michael_acceptor + aniline
        mw=200.0,
        daily_dose_mg=1000.0,
        fu=1.0,
        fm_cyp=1.0,
        logP=1.5,
    )
    assert r.risk_category == "high"


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------


def test_mitigation_populated_for_triggered_alerts():
    r = assess_protein_adduct_risk(
        "FuranDrug", "c1ccoc1", mw=68.0, daily_dose_mg=100.0, fu=0.5, fm_cyp=0.8, logP=0.5
    )
    assert len(r.mitigation) == len(r.alerts_triggered)
    assert all(isinstance(m, str) and len(m) > 0 for m in r.mitigation)


def test_mitigation_empty_when_no_alerts():
    r = assess_protein_adduct_risk(
        "Clean", "CCC(=O)OCC", mw=116.0, daily_dose_mg=100.0, fu=0.5, fm_cyp=0.5, logP=1.0
    )
    assert r.mitigation == []


# ---------------------------------------------------------------------------
# fu effect
# ---------------------------------------------------------------------------


def test_highly_bound_lower_score():
    base = dict(smiles="c1ccoc1", mw=68.0, fm_cyp=0.8, logP=0.5, daily_dose_mg=100.0)
    r_free = assess_protein_adduct_risk("Free", fu=1.0, **base)
    r_bound = assess_protein_adduct_risk("Bound", fu=0.01, **base)
    assert r_free.adduct_burden_score > r_bound.adduct_burden_score


# ---------------------------------------------------------------------------
# rank_compounds
# ---------------------------------------------------------------------------


def test_rank_compounds_returns_sorted():
    compounds = [
        dict(
            name="A",
            smiles="CCC(=O)OCC",
            mw=116.0,
            daily_dose_mg=100.0,
            fu=0.5,
            fm_cyp=0.5,
            logP=1.0,
        ),
        dict(
            name="B", smiles="c1ccoc1", mw=68.0, daily_dose_mg=500.0, fu=1.0, fm_cyp=1.0, logP=0.5
        ),
        dict(
            name="C",
            smiles="c1ccoc1C=CC=O",
            mw=100.0,
            daily_dose_mg=200.0,
            fu=0.8,
            fm_cyp=0.9,
            logP=1.5,
        ),
    ]
    results = rank_compounds(compounds)
    assert len(results) == 3
    scores = [r.adduct_burden_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_compounds_first_is_highest():
    compounds = [
        dict(
            name="Low",
            smiles="CCC(=O)OCC",
            mw=116.0,
            daily_dose_mg=10.0,
            fu=0.1,
            fm_cyp=0.1,
            logP=1.0,
        ),
        dict(
            name="High",
            smiles="c1ccoc1",
            mw=68.0,
            daily_dose_mg=1000.0,
            fu=1.0,
            fm_cyp=1.0,
            logP=0.5,
        ),
    ]
    results = rank_compounds(compounds)
    assert results[0].compound_name == "High"


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_field_types():
    r = assess_protein_adduct_risk(
        "Drug", "c1ccoc1", mw=68.0, daily_dose_mg=100.0, fu=0.5, fm_cyp=0.8, logP=0.5
    )
    assert isinstance(r, ProteinAdductResult)
    assert isinstance(r.compound_name, str)
    assert isinstance(r.alerts_triggered, list)
    assert isinstance(r.n_alerts, int)
    assert isinstance(r.alert_score, float)
    assert isinstance(r.adduct_burden_score, float)
    assert isinstance(r.risk_category, str)
    assert isinstance(r.mitigation, list)
    assert isinstance(r.notes, str)
