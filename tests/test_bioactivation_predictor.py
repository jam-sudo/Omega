"""Tests for bioactivation_predictor — Phase 483."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.bioactivation_predictor import (
    BioactivationResult,
    calculate_daily_bioactivation_burden,
    predict_bioactivation_risk,
    screen_compound_series,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_SMILES = "CC(=O)OC1=CC=CC=C1"  # aspirin-like, no alerts


# ---------------------------------------------------------------------------
# BioactivationResult dataclass
# ---------------------------------------------------------------------------


def test_result_is_dataclass():
    r = BioactivationResult(
        compound_name="test",
        smiles="C",
        daily_dose_mg=10.0,
    )
    assert r.compound_name == "test"
    assert r.n_alerts == 0
    assert r.risk_category == "low"


# ---------------------------------------------------------------------------
# predict_bioactivation_risk — low risk path
# ---------------------------------------------------------------------------


def test_no_alerts_low_dose_is_low_risk():
    r = predict_bioactivation_risk("safe", SAFE_SMILES, 10.0)
    assert r.risk_category == "low"
    assert r.risk_score == 0.0
    assert r.n_alerts == 0
    assert r.structural_alerts == []


def test_low_dose_no_fm_zero_burden():
    r = predict_bioactivation_risk("X", "CCC", 50.0, fm_cyp3a4=0.0)
    assert r.bioactivation_burden == 0.0
    assert r.risk_score == 0.0


# ---------------------------------------------------------------------------
# Structural alert detection
# ---------------------------------------------------------------------------


def test_thiophene_detected():
    smiles_thiophene = "c1ccsc1CC(=O)O"  # thiophene ring
    r = predict_bioactivation_risk("T", smiles_thiophene, 100.0, fm_cyp3a4=0.5)
    assert "thiophene" in r.structural_alerts


def test_furan_detected():
    smiles_furan = "c1ccoc1C(=O)O"
    r = predict_bioactivation_risk("F", smiles_furan, 100.0, fm_cyp3a4=0.5)
    assert "furan" in r.structural_alerts


def test_aniline_detected():
    smiles_aniline = "nc1ccccc1"
    r = predict_bioactivation_risk("A", smiles_aniline, 200.0, fm_cyp1a2=0.8)
    assert "aniline" in r.structural_alerts


def test_michael_acceptor_detected():
    smiles_ma = "c=cc=o"
    r = predict_bioactivation_risk("MA", smiles_ma, 100.0, fm_cyp3a4=0.5)
    assert "michael_acceptor" in r.structural_alerts


def test_quinone_precursor_detected():
    smiles_q = "oc1ccccc1"
    r = predict_bioactivation_risk("Q", smiles_q, 100.0, fm_cyp3a4=0.5)
    assert "quinone_precursor" in r.structural_alerts


def test_nitro_aromatic_detected():
    smiles_nitro = "c[n+](=o)[o-]c1ccccc1"
    r = predict_bioactivation_risk("N", smiles_nitro, 100.0, fm_cyp1a2=0.7)
    assert "nitro_aromatic" in r.structural_alerts


def test_epoxide_precursor_from_features():
    feats = {"mw": 300.0, "logp": 2.5}
    r = predict_bioactivation_risk("E", "CCC", 100.0, fm_cyp3a4=0.5, molecular_features=feats)
    assert "epoxide_precursor" in r.structural_alerts


def test_epoxide_precursor_not_detected_high_mw():
    feats = {"mw": 500.0, "logp": 2.5}  # mw >= 400 → no alert
    r = predict_bioactivation_risk("E2", "CCC", 100.0, fm_cyp3a4=0.5, molecular_features=feats)
    assert "epoxide_precursor" not in r.structural_alerts


def test_epoxide_precursor_not_detected_low_logp():
    feats = {"mw": 300.0, "logp": 1.0}  # logp <= 1.5 → no alert
    r = predict_bioactivation_risk("E3", "CCC", 100.0, fm_cyp3a4=0.5, molecular_features=feats)
    assert "epoxide_precursor" not in r.structural_alerts


# ---------------------------------------------------------------------------
# High risk scenario
# ---------------------------------------------------------------------------


def test_furan_high_dose_high_fm_very_high_risk():
    smiles_furan = "c1ccoc1CC"
    r = predict_bioactivation_risk("HighRisk", smiles_furan, 1000.0, fm_cyp3a4=0.9)
    assert r.risk_category in ("high", "very_high")
    assert r.risk_score > 25.0


def test_multiple_alerts_increase_score():
    # Both furan and nitro-aromatic
    smiles_multi = "c1ccoc1c[n+](=o)[o-]"
    r = predict_bioactivation_risk("Multi", smiles_multi, 500.0, fm_cyp3a4=0.8)
    assert r.n_alerts >= 2
    assert r.risk_score > 0.0


# ---------------------------------------------------------------------------
# Scoring mechanics
# ---------------------------------------------------------------------------


def test_risk_score_in_range():
    smiles = "c1ccsc1"
    r = predict_bioactivation_risk("S", smiles, 200.0, fm_cyp3a4=1.0)
    assert 0.0 <= r.risk_score <= 100.0


def test_bioactivation_burden_positive_when_alerts():
    smiles = "c1ccsc1"
    r = predict_bioactivation_risk("S", smiles, 100.0, fm_cyp3a4=0.5)
    assert r.bioactivation_burden > 0.0


def test_total_fm_cyp_capped_at_one():
    r = predict_bioactivation_risk("X", "CCC", 100.0, fm_cyp3a4=0.5, fm_cyp2d6=0.4, fm_cyp1a2=0.4)
    assert r.total_fm_cyp <= 1.0


def test_n_alerts_matches_structural_alerts():
    r = predict_bioactivation_risk("S", "c1ccsc1", 100.0, fm_cyp3a4=0.5)
    assert r.n_alerts == len(r.structural_alerts)


# ---------------------------------------------------------------------------
# Mitigation strategies
# ---------------------------------------------------------------------------


def test_mitigation_nonempty_when_alerts():
    r = predict_bioactivation_risk("S", "c1ccsc1", 100.0, fm_cyp3a4=0.5)
    assert len(r.mitigation_strategies) > 0


def test_mitigation_present_when_no_alerts():
    r = predict_bioactivation_risk("Safe", SAFE_SMILES, 10.0)
    assert isinstance(r.mitigation_strategies, list)
    assert len(r.mitigation_strategies) >= 1


# ---------------------------------------------------------------------------
# calculate_daily_bioactivation_burden
# ---------------------------------------------------------------------------


def test_burden_calculation_correct():
    val = calculate_daily_bioactivation_burden(
        daily_dose_mg=100.0,
        bioactivation_fraction=0.5,
        fm_cyp=0.8,
    )
    expected = 0.5 * math.log10(101.0) * 0.8
    assert abs(val - expected) < 1e-9


def test_burden_zero_when_fm_zero():
    val = calculate_daily_bioactivation_burden(100.0, 0.5, 0.0)
    assert val == 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_daily_dose_raises():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        predict_bioactivation_risk("X", "C", -10.0)


def test_invalid_fm_cyp3a4_raises():
    with pytest.raises(ValueError, match="fm_cyp3a4"):
        predict_bioactivation_risk("X", "C", 100.0, fm_cyp3a4=1.5)


def test_invalid_fm_cyp2d6_raises():
    with pytest.raises(ValueError, match="fm_cyp2d6"):
        predict_bioactivation_risk("X", "C", 100.0, fm_cyp2d6=-0.1)


def test_burden_invalid_dose_raises():
    with pytest.raises(ValueError):
        calculate_daily_bioactivation_burden(0.0, 0.5, 0.5)


def test_burden_invalid_fraction_raises():
    with pytest.raises(ValueError):
        calculate_daily_bioactivation_burden(100.0, 1.5, 0.5)


# ---------------------------------------------------------------------------
# screen_compound_series
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    compounds = [
        {"compound_name": "Low", "smiles": SAFE_SMILES, "daily_dose_mg": 10.0},
        {
            "compound_name": "High",
            "smiles": "c1ccoc1",
            "daily_dose_mg": 1000.0,
            "fm_cyp3a4": 0.9,
        },
        {
            "compound_name": "Medium",
            "smiles": "c1ccsc1",
            "daily_dose_mg": 100.0,
            "fm_cyp3a4": 0.3,
        },
    ]
    results = screen_compound_series(compounds)
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_returns_all_compounds():
    compounds = [
        {"compound_name": f"C{i}", "smiles": "CCC", "daily_dose_mg": 50.0} for i in range(5)
    ]
    results = screen_compound_series(compounds)
    assert len(results) == 5


def test_screen_empty_list():
    results = screen_compound_series([])
    assert results == []
