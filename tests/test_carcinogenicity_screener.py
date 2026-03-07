"""Tests for Phase 364 — Carcinogenicity Structural Alert Screener."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.carcinogenicity_screener import (
    CarcinogenicityScreenResult,
    screen_carcinogenicity,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal features dict
# ---------------------------------------------------------------------------


def _no_alerts(logP: float = 2.0, mw_Da: float = 300.0) -> dict:
    return {
        "has_nitrosamine": False,
        "has_aromatic_amine": False,
        "has_aflatoxin_scaffold": False,
        "has_alkylating_group": False,
        "has_intercalating_scaffold": False,
        "has_epoxide": False,
        "has_michael_acceptor": False,
        "has_polycyclic_aromatic": False,
        "has_hydrazine": False,
        "has_azo_dye": False,
        "logP": logP,
        "mw_Da": mw_Da,
    }


def _with_alerts(*alert_keys: str, logP: float = 2.0, mw_Da: float = 300.0) -> dict:
    features = _no_alerts(logP=logP, mw_Da=mw_Da)
    for key in alert_keys:
        features[key] = True
    return features


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert isinstance(result, CarcinogenicityScreenResult)


def test_alerts_present_is_list():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert isinstance(result.alerts_present, list)


def test_drug_name_preserved():
    result = screen_carcinogenicity("TestCompound", _no_alerts())
    assert result.drug_name == "TestCompound"


# ---------------------------------------------------------------------------
# No alerts scenario
# ---------------------------------------------------------------------------


def test_no_alerts_score_zero():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert result.carcinogenicity_score == 0.0


def test_no_alerts_class_low_concern():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert result.ich_m7_class == "low_concern"


def test_no_alerts_empty_list():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert result.alerts_present == []


def test_no_alerts_not_cohort_of_concern():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert result.cohort_of_concern is False


def test_no_alerts_no_mutagenicity_testing():
    result = screen_carcinogenicity("SafeDrug", _no_alerts())
    assert result.needs_mutagenicity_testing is False


# ---------------------------------------------------------------------------
# Cohort of concern
# ---------------------------------------------------------------------------


def test_nitrosamine_triggers_cohort_of_concern():
    features = _with_alerts("has_nitrosamine")
    result = screen_carcinogenicity("NitrosamineDrug", features)
    assert result.cohort_of_concern is True


def test_aflatoxin_triggers_cohort_of_concern():
    features = _with_alerts("has_aflatoxin_scaffold")
    result = screen_carcinogenicity("AflatoxinDrug", features)
    assert result.cohort_of_concern is True


def test_alkylating_triggers_cohort_of_concern():
    features = _with_alerts("has_alkylating_group")
    result = screen_carcinogenicity("AlkylDrug", features)
    assert result.cohort_of_concern is True
    assert result.ich_m7_class == "cohort_of_concern"


def test_cohort_of_concern_has_lowest_regulatory_limit():
    features = _with_alerts("has_nitrosamine")
    result = screen_carcinogenicity("NitrosamineDrug", features)
    assert result.regulatory_limit_ug_per_day == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_in_0_100():
    # Even with all alerts, score must be capped at 100
    features = _with_alerts(
        "has_nitrosamine",
        "has_aromatic_amine",
        "has_aflatoxin_scaffold",
        "has_alkylating_group",
        "has_intercalating_scaffold",
        "has_epoxide",
        "has_michael_acceptor",
        "has_polycyclic_aromatic",
        "has_hydrazine",
        "has_azo_dye",
    )
    result = screen_carcinogenicity("AllAlerts", features)
    assert 0.0 <= result.carcinogenicity_score <= 100.0


def test_multiple_alerts_higher_score_than_single():
    single = screen_carcinogenicity("Single", _with_alerts("has_aromatic_amine"))
    multiple = screen_carcinogenicity("Multi", _with_alerts("has_aromatic_amine", "has_epoxide"))
    assert multiple.carcinogenicity_score > single.carcinogenicity_score


def test_high_concern_class():
    # aromatic_amine(20) + epoxide(20) + hydrazine(20) = 60 → high_concern
    features = _with_alerts("has_aromatic_amine", "has_epoxide", "has_hydrazine")
    result = screen_carcinogenicity("HighRisk", features)
    assert result.ich_m7_class == "high_concern"


def test_moderate_concern_class():
    # michael_acceptor(10) + azo_dye(10) = 20 → moderate_concern
    features = _with_alerts("has_michael_acceptor", "has_azo_dye")
    result = screen_carcinogenicity("ModRisk", features)
    assert result.ich_m7_class == "moderate_concern"


# ---------------------------------------------------------------------------
# Mutagenicity testing
# ---------------------------------------------------------------------------


def test_mutagenicity_testing_when_score_gte_20():
    # hydrazine = 20 → needs testing
    features = _with_alerts("has_hydrazine")
    result = screen_carcinogenicity("HydrazineDrug", features)
    assert result.needs_mutagenicity_testing is True


def test_no_mutagenicity_testing_when_score_lt_20():
    # michael_acceptor = 10 → no testing needed
    features = _with_alerts("has_michael_acceptor")
    result = screen_carcinogenicity("MichaelDrug", features)
    assert result.needs_mutagenicity_testing is False


# ---------------------------------------------------------------------------
# Regulatory limit always > 0
# ---------------------------------------------------------------------------


def test_regulatory_limit_gt_zero():
    for features in [
        _no_alerts(),
        _with_alerts("has_nitrosamine"),
        _with_alerts("has_aromatic_amine", "has_epoxide", "has_hydrazine"),
    ]:
        result = screen_carcinogenicity("Drug", features)
        assert result.regulatory_limit_ug_per_day > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_mw_zero():
    features = _no_alerts(mw_Da=0.0)
    with pytest.raises(ValueError, match="mw_Da"):
        screen_carcinogenicity("Drug", features)


def test_error_mw_negative():
    features = _no_alerts(mw_Da=-100.0)
    with pytest.raises(ValueError, match="mw_Da"):
        screen_carcinogenicity("Drug", features)


def test_error_logp_too_low():
    features = _no_alerts(logP=-6.0)
    with pytest.raises(ValueError, match="logP"):
        screen_carcinogenicity("Drug", features)


def test_error_logp_too_high():
    features = _no_alerts(logP=11.0)
    with pytest.raises(ValueError, match="logP"):
        screen_carcinogenicity("Drug", features)
