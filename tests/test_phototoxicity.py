"""Tests for phototoxicity prediction module."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.phototoxicity import (
    PhototoxResult,
    assess_phototoxicity,
    get_phototoxic_alerts_summary,
    screen_phototoxicity,
)

# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------


def test_assess_returns_phototox_result():
    result = assess_phototoxicity("aspirin", "CC(=O)Oc1ccccc1C(=O)O", mw=180.16, logP=1.2)
    assert isinstance(result, PhototoxResult)


def test_empty_smiles_raises_valueerror():
    with pytest.raises(ValueError, match="SMILES"):
        assess_phototoxicity("bad", "", mw=100, logP=1.0)


def test_mw_zero_raises_valueerror():
    with pytest.raises(ValueError, match="Molecular weight"):
        assess_phototoxicity("bad", "CCCCCC", mw=0, logP=1.0)


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------


def test_simple_alkane_negative():
    result = assess_phototoxicity("hexane", "CCCCCC", mw=86.18, logP=3.9)
    assert result.phototoxicity_prediction == "negative"


def test_naphthalene_triggers_alert():
    result = assess_phototoxicity("naphthalene", "c1ccc2ccccc2c1", mw=128.17, logP=4.0)
    triggered = [a for a in result.alerts if a.triggered]
    assert len(triggered) >= 1


def test_high_logP_triggers_alert():
    result = assess_phototoxicity("lipophilic", "c1ccccc1CCCCCCCCC", mw=200, logP=5.0)
    high_logp = [a for a in result.alerts if a.name == "high_logP" and a.triggered]
    assert len(high_logp) == 1


def test_furan_aromatic_triggers_alert():
    result = assess_phototoxicity("furan_cpd", "c1ccoc1", mw=68.07, logP=1.3)
    furan = [a for a in result.alerts if a.name == "furan_aromatic" and a.triggered]
    assert len(furan) == 1


# ---------------------------------------------------------------------------
# Score and classification
# ---------------------------------------------------------------------------


def test_alert_score_equals_sum_of_weights():
    result = assess_phototoxicity("test", "c1ccc2ccccc2c1", mw=128.17, logP=4.0)
    triggered = [a for a in result.alerts if a.triggered]
    assert result.alert_score == pytest.approx(sum(a.weight for a in triggered))


def test_phototoxicity_probability_in_range():
    result = assess_phototoxicity("test", "c1ccccc1", mw=78.11, logP=2.1)
    assert 0 <= result.phototoxicity_probability <= 1


def test_ich_s10_flag_valid():
    result = assess_phototoxicity("test", "CCCCCC", mw=86.18, logP=1.0)
    assert result.ich_s10_flag in {"concern", "no_concern", "inconclusive"}


def test_skin_risk_valid():
    result = assess_phototoxicity("test", "c1ccccc1", mw=78.11, logP=2.1)
    assert result.skin_risk in {"low", "moderate", "high"}


def test_eye_risk_valid():
    result = assess_phototoxicity("test", "c1ccccc1", mw=78.11, logP=2.1)
    assert result.eye_risk in {"low", "moderate", "high"}


def test_photo_dri_gte_alert_score():
    result = assess_phototoxicity("test", "c1ccc2ccccc2c1", mw=128.17, logP=4.0)
    assert result.photo_dri >= result.alert_score


def test_molar_absorptivity_proxy_non_negative():
    result = assess_phototoxicity("test", "CCCCCC", mw=86.18, logP=1.0)
    assert result.molar_absorptivity_proxy >= 0


# ---------------------------------------------------------------------------
# screen_phototoxicity
# ---------------------------------------------------------------------------


def test_screen_returns_same_length():
    compounds = [
        {"name": "a", "smiles": "CCCCCC", "mw": 86, "logP": 1.0},
        {"name": "b", "smiles": "c1ccccc1", "mw": 78, "logP": 2.0},
    ]
    results = screen_phototoxicity(compounds)
    assert len(results) == 2


def test_screen_sorted_by_photo_dri_descending():
    compounds = [
        {"name": "low", "smiles": "CCCCCC", "mw": 86, "logP": 1.0},
        {"name": "high", "smiles": "c1ccc2ccccc2c1", "mw": 128, "logP": 5.0},
    ]
    results = screen_phototoxicity(compounds)
    assert results[0].photo_dri >= results[1].photo_dri


# ---------------------------------------------------------------------------
# get_phototoxic_alerts_summary
# ---------------------------------------------------------------------------


def test_alerts_summary_has_enough_keys():
    summary = get_phototoxic_alerts_summary()
    assert isinstance(summary, dict)
    assert len(summary) >= 5


# ---------------------------------------------------------------------------
# Prediction values
# ---------------------------------------------------------------------------


def test_phototoxicity_prediction_valid_values():
    result = assess_phototoxicity("test", "CCCCCC", mw=86, logP=1.0)
    assert result.phototoxicity_prediction in {"positive", "negative", "inconclusive"}


def test_compound_name_preserved():
    result = assess_phototoxicity("MyCompound", "CCCCCC", mw=86, logP=1.0)
    assert result.compound_name == "MyCompound"


def test_high_alert_compound_not_low_risk():
    # Naphthalene with high logP should trigger multiple alerts → not low risk
    result = assess_phototoxicity("risky", "c1ccc2ccccc2c1", mw=200, logP=5.0)
    assert result.skin_risk != "low" or result.eye_risk != "low"


def test_n_alerts_equals_triggered_count():
    result = assess_phototoxicity("test", "c1ccccc1", mw=200, logP=2.0)
    triggered = [a for a in result.alerts if a.triggered]
    assert result.n_alerts == len(triggered)


def test_low_logP_no_alerts_skin_risk_low():
    result = assess_phototoxicity("safe", "CCCCCC", mw=86, logP=0.5)
    assert result.skin_risk == "low"
