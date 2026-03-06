"""Tests for reactive metabolite / bioactivation risk assessment (Phase 156)."""

import pytest

from omega_pbpk.risk.reactive_metabolite import (
    ReactiveAlert,
    ReactiveMetaboliteResult,
    assess_reactive_metabolite,
    screen_reactive_metabolites,
)

SAFE_SMILES = "CC(=O)Oc1ccccc1"
FURAN_SMILES = "c1ccoc1"
ANILINE_SMILES = "Nc1ccccc1"
HYDRAZINE_SMILES = "CNN"
NITRO_SMILES = "c1ccc([N+](=O)[O-])cc1"
MICHAEL_SMILES = "C=CC=O"


def test_empty_smiles_raises():
    with pytest.raises(ValueError, match="SMILES"):
        assess_reactive_metabolite("X", "", mw=200.0, logP=2.0)


def test_blank_smiles_raises():
    with pytest.raises(ValueError, match="SMILES"):
        assess_reactive_metabolite("X", "   ", mw=200.0, logP=2.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="Molecular weight"):
        assess_reactive_metabolite("X", "CC", mw=-100.0, logP=1.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="Molecular weight"):
        assess_reactive_metabolite("X", "CC", mw=0.0, logP=1.0)


def test_returns_result():
    r = assess_reactive_metabolite("test", SAFE_SMILES, mw=180.0, logP=1.2)
    assert isinstance(r, ReactiveMetaboliteResult)


def test_alerts_list_not_empty():
    r = assess_reactive_metabolite("test", SAFE_SMILES, mw=180.0, logP=1.2)
    assert len(r.alerts) > 0
    assert all(isinstance(a, ReactiveAlert) for a in r.alerts)


def test_safe_compound_low_risk():
    r = assess_reactive_metabolite("safe", SAFE_SMILES, mw=180.0, logP=1.2)
    assert r.dili_risk == "low"


def test_furan_alert_triggered():
    r = assess_reactive_metabolite("furanoid", FURAN_SMILES, mw=68.0, logP=1.3)
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "furan" in triggered


def test_furan_score_positive():
    r = assess_reactive_metabolite("furanoid", FURAN_SMILES, mw=68.0, logP=1.3)
    assert r.alert_score > 0


def test_aniline_alert_triggered():
    r = assess_reactive_metabolite("aniline_cmpd", ANILINE_SMILES, mw=93.0, logP=0.9)
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "aniline" in triggered


def test_aniline_mitigation_suggested():
    r = assess_reactive_metabolite("aniline_cmpd", ANILINE_SMILES, mw=93.0, logP=0.9)
    assert len(r.mitigation_suggestions) > 0


def test_hydrazine_alert_triggered():
    r = assess_reactive_metabolite("hydrazine_cmpd", HYDRAZINE_SMILES, mw=60.0, logP=-0.5)
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "hydrazine" in triggered


def test_nitro_alert_triggered():
    r = assess_reactive_metabolite("nitro_cmpd", NITRO_SMILES, mw=123.0, logP=1.8)
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "nitro_aromatic" in triggered


def test_michael_alert_triggered():
    r = assess_reactive_metabolite("michael_cmpd", MICHAEL_SMILES, mw=70.0, logP=0.5)
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "michael_acceptor" in triggered


def test_high_dose_alert_triggered():
    r = assess_reactive_metabolite(
        "high_dose_cmpd", SAFE_SMILES, mw=180.0, logP=1.2, daily_dose_mg=200.0
    )
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "high_dose" in triggered


def test_low_dose_alert_not_triggered():
    r = assess_reactive_metabolite(
        "low_dose_cmpd", SAFE_SMILES, mw=180.0, logP=1.2, daily_dose_mg=10.0
    )
    triggered = {a.name for a in r.alerts if a.triggered}
    assert "high_dose" not in triggered


def test_n_alerts_count():
    r = assess_reactive_metabolite("furanoid", FURAN_SMILES, mw=68.0, logP=1.3)
    assert r.n_alerts == len([a for a in r.alerts if a.triggered])


def test_dose_burden_score_increases_with_dose():
    r_low = assess_reactive_metabolite(
        "cmpd", ANILINE_SMILES, mw=93.0, logP=0.9, daily_dose_mg=10.0
    )
    r_high = assess_reactive_metabolite(
        "cmpd", ANILINE_SMILES, mw=93.0, logP=0.9, daily_dose_mg=500.0
    )
    assert r_high.dose_burden_score > r_low.dose_burden_score


def test_dili_risk_values():
    r = assess_reactive_metabolite("cmpd", SAFE_SMILES, mw=180.0, logP=1.2)
    assert r.dili_risk in ("low", "moderate", "high")


def test_idiosyncratic_risk_values():
    r = assess_reactive_metabolite("cmpd", SAFE_SMILES, mw=180.0, logP=1.2)
    assert r.idiosyncratic_risk in ("low", "moderate", "high")


def test_structural_class_values():
    r = assess_reactive_metabolite("cmpd", SAFE_SMILES, mw=180.0, logP=1.2)
    assert r.structural_class in ("low_concern", "moderate_concern", "high_concern")


def test_screen_returns_list():
    compounds = [
        {"name": "furanoid", "smiles": FURAN_SMILES, "mw": 68.0, "logP": 1.3},
        {"name": "safe", "smiles": SAFE_SMILES, "mw": 180.0, "logP": 1.2},
    ]
    results = screen_reactive_metabolites(compounds)
    assert len(results) == 2
    assert all(isinstance(r, ReactiveMetaboliteResult) for r in results)


def test_screen_sorted_by_dose_burden():
    compounds = [
        {"name": "safe", "smiles": SAFE_SMILES, "mw": 180.0, "logP": 1.2, "daily_dose_mg": 5.0},
        {
            "name": "furanoid",
            "smiles": FURAN_SMILES,
            "mw": 68.0,
            "logP": 1.3,
            "daily_dose_mg": 200.0,
        },
    ]
    results = screen_reactive_metabolites(compounds)
    scores = [r.dose_burden_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_uses_default_dose():
    compounds = [{"name": "cmpd", "smiles": SAFE_SMILES, "mw": 180.0, "logP": 1.2}]
    results = screen_reactive_metabolites(compounds)
    assert results[0].daily_dose_mg == 50.0
