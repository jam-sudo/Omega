"""Tests for Phase 747 — Drug Metabolite Toxicity Predictor."""

import pytest

from omega_pbpk.risk.metabolite_toxicity_risk import (
    MetaboliteToxicityResult,
    assess_metabolite_toxicity,
    screen_metabolite_toxicity,
)

VALID_CONCERN_VALUES = {"acceptable", "caution", "significant concern"}
VALID_DILI_LEVELS = {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result_dataclass():
    result = assess_metabolite_toxicity("DrugA", daily_dose_mg=50.0)
    assert isinstance(result, MetaboliteToxicityResult)


def test_compound_name_preserved():
    result = assess_metabolite_toxicity("TestDrug", daily_dose_mg=10.0)
    assert result.compound_name == "TestDrug"


# ---------------------------------------------------------------------------
# Low-risk scenario
# ---------------------------------------------------------------------------


def test_no_alerts_low_dose_low_dili():
    result = assess_metabolite_toxicity(
        "SafeDrug",
        daily_dose_mg=10.0,
        has_quinone_forming=False,
        has_epoxide_forming=False,
    )
    assert result.dili_risk_level == "low"


def test_no_alerts_low_dose_acceptable_concern():
    result = assess_metabolite_toxicity("SafeDrug2", daily_dose_mg=10.0)
    assert result.overall_safety_concern == "acceptable"


# ---------------------------------------------------------------------------
# Quinone / high dose → high DILI
# ---------------------------------------------------------------------------


def test_quinone_high_dose_high_dili():
    result = assess_metabolite_toxicity(
        "QuinoneDrug",
        daily_dose_mg=600.0,
        has_quinone_forming=True,
    )
    assert result.dili_risk_level == "high"


def test_quinone_alone_raises_hepatic_score():
    result_no_q = assess_metabolite_toxicity("NoQ", daily_dose_mg=50.0)
    result_q = assess_metabolite_toxicity("WithQ", daily_dose_mg=50.0, has_quinone_forming=True)
    assert result_q.hepatic_risk_score > result_no_q.hepatic_risk_score


# ---------------------------------------------------------------------------
# Epoxide forming
# ---------------------------------------------------------------------------


def test_epoxide_forming_increases_hepatic_risk():
    result_base = assess_metabolite_toxicity("Base", daily_dose_mg=50.0)
    result_ep = assess_metabolite_toxicity("Epoxide", daily_dose_mg=50.0, has_epoxide_forming=True)
    assert result_ep.hepatic_risk_score > result_base.hepatic_risk_score


def test_epoxide_high_dose_significant_concern():
    result = assess_metabolite_toxicity(
        "EpoxideDrug",
        daily_dose_mg=600.0,
        has_epoxide_forming=True,
    )
    assert result.overall_safety_concern == "significant concern"


# ---------------------------------------------------------------------------
# Renal risk
# ---------------------------------------------------------------------------


def test_high_fe_renal_increases_renal_score():
    result_low = assess_metabolite_toxicity("LowRenal", daily_dose_mg=50.0, fe_renal=0.1)
    result_high = assess_metabolite_toxicity("HighRenal", daily_dose_mg=50.0, fe_renal=0.8)
    assert result_high.renal_risk_score > result_low.renal_risk_score


def test_nephrotoxic_flag_high_renal_risk():
    result = assess_metabolite_toxicity(
        "NephDrug",
        daily_dose_mg=50.0,
        has_nephrotoxic_metabolite_flag=True,
    )
    assert result.renal_risk_score >= 40


def test_nephrotoxic_flag_with_high_fe_renal_very_high_score():
    result = assess_metabolite_toxicity(
        "NephDrug2",
        daily_dose_mg=50.0,
        fe_renal=0.8,
        has_nephrotoxic_metabolite_flag=True,
    )
    assert result.renal_risk_score >= 65


# ---------------------------------------------------------------------------
# Risk level strings
# ---------------------------------------------------------------------------


def test_dili_risk_level_valid():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0)
    assert result.dili_risk_level in VALID_DILI_LEVELS


def test_renal_risk_level_valid():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0)
    assert result.renal_risk_level in VALID_DILI_LEVELS


def test_overall_safety_concern_valid():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0)
    assert result.overall_safety_concern in VALID_CONCERN_VALUES


# ---------------------------------------------------------------------------
# phase2_protection_factor
# ---------------------------------------------------------------------------


def test_phase2_protection_factor_matches_input():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0, phase2_coverage=0.85)
    assert result.phase2_protection_factor == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "daily_dose_mg": 10.0},
        {"compound_name": "B", "daily_dose_mg": 600.0, "has_quinone_forming": True},
    ]
    results = screen_metabolite_toxicity(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_hepatic_desc():
    compounds = [
        {"compound_name": "Low", "daily_dose_mg": 10.0},
        {"compound_name": "High", "daily_dose_mg": 600.0, "has_quinone_forming": True},
        {"compound_name": "Mid", "daily_dose_mg": 200.0, "has_epoxide_forming": True},
    ]
    results = screen_metabolite_toxicity(compounds)
    scores = [r.hepatic_risk_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_zero_dose_raises():
    with pytest.raises(ValueError):
        assess_metabolite_toxicity("D", daily_dose_mg=0.0)


def test_validation_negative_dose_raises():
    with pytest.raises(ValueError):
        assess_metabolite_toxicity("D", daily_dose_mg=-10.0)


def test_validation_zero_mw_raises():
    with pytest.raises(ValueError):
        assess_metabolite_toxicity("D", daily_dose_mg=50.0, mw_da=0.0)


def test_validation_invalid_fm_raises():
    with pytest.raises(ValueError):
        assess_metabolite_toxicity("D", daily_dose_mg=50.0, fm_cyp3a4=1.5)


def test_validation_invalid_fe_renal_raises():
    with pytest.raises(ValueError):
        assess_metabolite_toxicity("D", daily_dose_mg=50.0, fe_renal=-0.1)


# ---------------------------------------------------------------------------
# Notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0)
    assert len(result.notes) > 0


def test_recommendation_nonempty():
    result = assess_metabolite_toxicity("D", daily_dose_mg=50.0)
    assert len(result.recommendation) > 0
