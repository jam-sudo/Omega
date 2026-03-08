"""Tests for Phase 314 — Drug-Induced Kidney Injury Risk Predictor."""

import pytest

from omega_pbpk.risk.diki_risk_predictor import (
    DIKIRiskResult,
    predict_diki_risk,
    screen_compounds,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    logp=1.5,
    mw=300.0,
    renal_clearance_fraction=0.4,
    plasma_protein_binding_pct=70.0,
    is_nephrotoxic_class=False,
    structural_alert_count_kidney=0,
    daily_dose_mg=200.0,
    fu_plasma=0.3,
)


def make_result(**overrides) -> DIKIRiskResult:
    kwargs = {**DEFAULT_KWARGS, **overrides}
    return predict_diki_risk(**kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, DIKIRiskResult)


def test_result_is_dataclass_instance():
    from dataclasses import fields

    result = make_result()
    field_names = {f.name for f in fields(result)}
    assert "risk_score" in field_names
    assert "mitigation_strategies" in field_names


# ---------------------------------------------------------------------------
# Risk score bounds
# ---------------------------------------------------------------------------


def test_risk_score_nonnegative():
    result = make_result()
    assert result.risk_score >= 0


def test_risk_score_maximum():
    # Maximize all factors
    result = make_result(
        logp=5.0,
        renal_clearance_fraction=0.8,
        plasma_protein_binding_pct=99.0,
        is_nephrotoxic_class=True,
        structural_alert_count_kidney=3,
        daily_dose_mg=2000.0,
        fu_plasma=0.01,
    )
    assert result.risk_score <= 120


def test_risk_score_zero_case():
    result = make_result(
        logp=1.0,
        renal_clearance_fraction=0.1,
        plasma_protein_binding_pct=50.0,
        is_nephrotoxic_class=False,
        structural_alert_count_kidney=0,
        daily_dose_mg=5.0,
        fu_plasma=0.5,
    )
    assert result.risk_score >= 0


# ---------------------------------------------------------------------------
# AKI likelihood percentage
# ---------------------------------------------------------------------------


def test_aki_likelihood_in_range():
    result = make_result()
    assert 0.0 <= result.aki_likelihood_pct <= 90.0


def test_aki_likelihood_max_90():
    result = make_result(
        logp=5.0,
        renal_clearance_fraction=0.9,
        plasma_protein_binding_pct=99.0,
        is_nephrotoxic_class=True,
        structural_alert_count_kidney=5,
        daily_dose_mg=5000.0,
        fu_plasma=0.01,
    )
    assert result.aki_likelihood_pct <= 90.0


# ---------------------------------------------------------------------------
# Nephrotoxic class effect
# ---------------------------------------------------------------------------


def test_nephrotoxic_class_increases_score():
    without = make_result(is_nephrotoxic_class=False)
    with_class = make_result(is_nephrotoxic_class=True)
    assert with_class.risk_score > without.risk_score


def test_nephrotoxic_class_score_delta():
    without = make_result(is_nephrotoxic_class=False)
    with_class = make_result(is_nephrotoxic_class=True)
    assert with_class.risk_score - without.risk_score == 35


# ---------------------------------------------------------------------------
# Renal clearance fraction effect
# ---------------------------------------------------------------------------


def test_high_renal_fraction_increases_score():
    low = make_result(renal_clearance_fraction=0.1)
    high = make_result(renal_clearance_fraction=0.8)
    assert high.risk_score > low.risk_score


def test_renal_score_tiers():
    result_low = make_result(renal_clearance_fraction=0.1)
    result_mid = make_result(renal_clearance_fraction=0.45)
    result_high = make_result(renal_clearance_fraction=0.75)
    assert result_high.risk_score > result_mid.risk_score > result_low.risk_score


# ---------------------------------------------------------------------------
# Risk class
# ---------------------------------------------------------------------------


def test_risk_class_valid_values():
    valid = {"high", "moderate", "low"}
    result = make_result()
    assert result.risk_class in valid


def test_risk_class_high():
    result = make_result(
        is_nephrotoxic_class=True,
        renal_clearance_fraction=0.9,
        structural_alert_count_kidney=2,
    )
    assert result.risk_class == "high"


def test_risk_class_low():
    result = make_result(
        logp=1.0,
        renal_clearance_fraction=0.1,
        is_nephrotoxic_class=False,
        structural_alert_count_kidney=0,
        daily_dose_mg=5.0,
        plasma_protein_binding_pct=50.0,
        fu_plasma=0.3,
    )
    assert result.risk_class == "low"


# ---------------------------------------------------------------------------
# Mitigation strategies
# ---------------------------------------------------------------------------


def test_adequate_hydration_always_present():
    for nephrotoxic in (True, False):
        result = make_result(is_nephrotoxic_class=nephrotoxic)
        assert "adequate_hydration" in result.mitigation_strategies


def test_nephrotoxic_class_adds_dose_reduce():
    result = make_result(is_nephrotoxic_class=True)
    assert "dose_reduce_in_renal_impairment" in result.mitigation_strategies


def test_high_renal_fraction_adds_gfr_adjust():
    result = make_result(renal_clearance_fraction=0.8)
    assert "adjust_dose_for_gfr" in result.mitigation_strategies


def test_high_dose_adds_dose_split():
    result = make_result(daily_dose_mg=2000.0)
    assert "consider_dose_split" in result.mitigation_strategies


def test_low_dose_no_dose_split():
    result = make_result(daily_dose_mg=50.0)
    assert "consider_dose_split" not in result.mitigation_strategies


# ---------------------------------------------------------------------------
# screen_compounds
# ---------------------------------------------------------------------------


def test_screen_compounds_returns_list():
    compounds = [DEFAULT_KWARGS.copy(), {**DEFAULT_KWARGS, "drug_name": "DrugB"}]
    results = screen_compounds(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_compounds_sorted_descending():
    compounds = [
        {
            **DEFAULT_KWARGS,
            "drug_name": "Low",
            "renal_clearance_fraction": 0.1,
            "is_nephrotoxic_class": False,
        },
        {
            **DEFAULT_KWARGS,
            "drug_name": "High",
            "renal_clearance_fraction": 0.9,
            "is_nephrotoxic_class": True,
        },
        {
            **DEFAULT_KWARGS,
            "drug_name": "Mid",
            "renal_clearance_fraction": 0.4,
            "is_nephrotoxic_class": False,
        },
    ]
    results = screen_compounds(compounds)
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_logp_low():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=-6.0)


def test_invalid_logp_high():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=11.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=-100.0)


def test_renal_fraction_out_of_range():
    with pytest.raises(ValueError, match="renal_clearance_fraction"):
        make_result(renal_clearance_fraction=1.5)


def test_zero_daily_dose_raises():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        make_result(daily_dose_mg=0.0)


def test_zero_fu_plasma_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        make_result(fu_plasma=0.0)


def test_negative_structural_alerts_raises():
    with pytest.raises(ValueError, match="structural_alert_count_kidney"):
        make_result(structural_alert_count_kidney=-1)
