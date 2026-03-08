"""Tests for Phase 551 - Drug Crystallization Risk in Formulation."""

import math

import pytest

from omega_pbpk.risk.crystallization_risk import (
    CrystallizationRiskResult,
    predict_crystallization_risk,
    screen_crystallization,
)


def test_return_type():
    r = predict_crystallization_risk("DrugA", logP=3.0, mw_Da=400.0, melting_point_C=200.0)
    assert isinstance(r, CrystallizationRiskResult)


def test_drug_name_preserved():
    r = predict_crystallization_risk("TestDrug", logP=2.0, mw_Da=300.0, melting_point_C=150.0)
    assert r.drug_name == "TestDrug"


def test_ssr_at_least_1():
    r = predict_crystallization_risk("LowMelt", logP=1.0, mw_Da=200.0, melting_point_C=20.0)
    assert r.supersaturation_ratio >= 1.0


def test_ssr_capped_at_100():
    r = predict_crystallization_risk("HighMelt", logP=2.0, mw_Da=300.0, melting_point_C=500.0)
    assert r.supersaturation_ratio <= 100.0


def test_ssr_calculation():
    r = predict_crystallization_risk("Exact25", logP=2.0, mw_Da=300.0, melting_point_C=25.0)
    assert abs(r.supersaturation_ratio - 1.0) < 1e-6


def test_amorphous_solubility_boost():
    r = predict_crystallization_risk(
        "Boost",
        logP=2.0,
        mw_Da=300.0,
        melting_point_C=150.0,
        crystalline_solubility_mg_mL=0.05,
    )
    expected_ssr = math.exp(0.05 * (150.0 - 25.0))
    ssr = min(100.0, max(1.0, expected_ssr))
    assert abs(r.solubility_amorphous_mg_mL - 0.05 * ssr) < 1e-6


def test_k_cryst_in_range():
    for logp in [-1.0, 0.5, 1.0, 3.0, 5.0, 10.0]:
        r = predict_crystallization_risk("K", logP=logp, mw_Da=300.0, melting_point_C=150.0)
        assert 0.01 <= r.crystallization_rate_constant <= 2.0


def test_k_cryst_increases_with_logp():
    r_low = predict_crystallization_risk("Low", logP=1.0, mw_Da=300.0, melting_point_C=150.0)
    r_high = predict_crystallization_risk("High", logP=5.0, mw_Da=300.0, melting_point_C=150.0)
    assert r_high.crystallization_rate_constant >= r_low.crystallization_rate_constant


def test_induction_time_in_range():
    r = predict_crystallization_risk("Ind", logP=3.0, mw_Da=400.0, melting_point_C=200.0)
    assert 0.1 <= r.induction_time_h <= 48.0


def test_risk_score_in_range():
    for logp in [-2.0, 0.0, 2.0, 5.0, 8.0]:
        r = predict_crystallization_risk("RS", logP=logp, mw_Da=300.0, melting_point_C=150.0)
        assert 0.0 <= r.risk_score <= 100.0


def test_risk_class_high():
    r = predict_crystallization_risk("HighRisk", logP=5.0, mw_Da=500.0, melting_point_C=250.0)
    assert r.risk_class == "high"
    assert r.risk_score > 60.0


def test_risk_class_low():
    r = predict_crystallization_risk("LowRisk", logP=0.5, mw_Da=200.0, melting_point_C=50.0)
    assert r.risk_class == "low"
    assert r.risk_score <= 30.0


def test_risk_class_moderate():
    # logP=1.5, mp=100 → risk ~30+0+12.6 ≈ 42.6 → moderate
    r = predict_crystallization_risk("ModRisk", logP=1.5, mw_Da=300.0, melting_point_C=100.0)
    assert r.risk_class == "moderate"


def test_recommendation_high():
    r = predict_crystallization_risk("RecHigh", logP=5.0, mw_Da=500.0, melting_point_C=250.0)
    assert "HPMC-AS" in r.formulation_recommendation


def test_recommendation_moderate():
    r = predict_crystallization_risk("RecMod", logP=1.5, mw_Da=300.0, melting_point_C=100.0)
    assert "amorphous" in r.formulation_recommendation.lower()


def test_recommendation_low():
    r = predict_crystallization_risk("RecLow", logP=0.5, mw_Da=200.0, melting_point_C=50.0)
    assert "crystalline" in r.formulation_recommendation.lower()


def test_dose_preserved():
    r = predict_crystallization_risk(
        "Dose", logP=2.0, mw_Da=300.0, melting_point_C=150.0, dose_mg=250.0
    )
    assert r.dose_mg == 250.0


def test_default_dose():
    r = predict_crystallization_risk("DefDose", logP=2.0, mw_Da=300.0, melting_point_C=150.0)
    assert r.dose_mg == 100.0


def test_notes_not_empty():
    r = predict_crystallization_risk("Notes", logP=2.0, mw_Da=300.0, melting_point_C=150.0)
    assert len(r.notes) > 0


def test_screen_crystallization_sorted():
    compounds = [
        {"drug_name": "A", "logP": 1.0, "mw_Da": 200.0, "melting_point_C": 100.0},
        {"drug_name": "B", "logP": 5.0, "mw_Da": 500.0, "melting_point_C": 250.0},
        {"drug_name": "C", "logP": 3.0, "mw_Da": 350.0, "melting_point_C": 180.0},
    ]
    results = screen_crystallization(compounds)
    assert len(results) == 3
    for i in range(1, len(results)):
        assert results[i - 1].risk_score >= results[i].risk_score


def test_screen_crystallization_returns_list():
    compounds = [
        {"drug_name": "X", "logP": 2.0, "mw_Da": 300.0, "melting_point_C": 150.0},
    ]
    results = screen_crystallization(compounds)
    assert isinstance(results, list)
    assert all(isinstance(r, CrystallizationRiskResult) for r in results)


def test_frozen_result():
    r = predict_crystallization_risk("Frozen", logP=2.0, mw_Da=300.0, melting_point_C=150.0)
    with pytest.raises(AttributeError):
        r.risk_score = 999.0
