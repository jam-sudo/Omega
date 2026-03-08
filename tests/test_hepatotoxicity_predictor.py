"""Tests for Phase 299 — Hepatotoxicity Risk Predictor."""

import pytest

from omega_pbpk.risk.hepatotoxicity_predictor import (
    HepatotoxicityRiskResult,
    predict_hepatotoxicity_risk,
    screen_compounds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_compound(
    drug_name="TestDrug",
    logp=2.0,
    mw=350.0,
    daily_dose_mg=50.0,
    fu_plasma=0.1,
    has_reactive_metabolite=False,
    has_mitochondrial_liability=False,
    structural_alert_count=1,
):
    return dict(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        fu_plasma=fu_plasma,
        has_reactive_metabolite=has_reactive_metabolite,
        has_mitochondrial_liability=has_mitochondrial_liability,
        structural_alert_count=structural_alert_count,
    )


def _predict(**kwargs):
    defaults = _make_compound()
    defaults.update(kwargs)
    return predict_hepatotoxicity_risk(**defaults)


# ---------------------------------------------------------------------------
# Basic return type & structure
# ---------------------------------------------------------------------------


def test_returns_hepatotoxicity_risk_result():
    result = _predict()
    assert isinstance(result, HepatotoxicityRiskResult)


def test_risk_score_non_negative():
    result = _predict()
    assert result.risk_score >= 0


def test_dili_likelihood_in_range():
    result = _predict()
    assert 0 <= result.dili_likelihood_pct <= 95


def test_risk_class_valid_values():
    for logp, dose in [(0.5, 5), (2.0, 50), (4.0, 150)]:
        result = _predict(logp=logp, daily_dose_mg=dose)
        assert result.risk_class in {"high", "moderate", "low"}


def test_mitigation_strategies_is_list():
    result = _predict()
    assert isinstance(result.mitigation_strategies, list)


def test_mitigation_strategies_non_empty():
    result = _predict()
    assert len(result.mitigation_strategies) > 0


def test_monitor_liver_enzymes_always_present():
    for rm in [True, False]:
        for mito in [True, False]:
            result = _predict(has_reactive_metabolite=rm, has_mitochondrial_liability=mito)
            assert "monitor_liver_enzymes" in result.mitigation_strategies


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def test_higher_dose_higher_risk_score():
    low = _predict(daily_dose_mg=5)
    high = _predict(daily_dose_mg=200)
    assert high.risk_score > low.risk_score


def test_reactive_metabolite_increases_risk_score():
    without = _predict(has_reactive_metabolite=False)
    with_ = _predict(has_reactive_metabolite=True)
    assert with_.risk_score > without.risk_score


def test_mitochondrial_liability_increases_risk_score():
    without = _predict(has_mitochondrial_liability=False)
    with_ = _predict(has_mitochondrial_liability=True)
    assert with_.risk_score > without.risk_score


def test_high_logp_increases_risk_score():
    low = _predict(logp=0.5)
    high = _predict(logp=4.0)
    assert high.risk_score > low.risk_score


def test_structural_alerts_increase_risk_score():
    zero = _predict(structural_alert_count=0)
    three = _predict(structural_alert_count=3)
    assert three.risk_score > zero.risk_score


def test_structural_alerts_capped_at_30():
    three = _predict(structural_alert_count=3)
    ten = _predict(structural_alert_count=10)
    assert (
        three.risk_score == ten.risk_score - (min(30, 10 * 10) - min(30, 3 * 10))
        or three.risk_score == ten.risk_score
    )


def test_high_mw_adds_penalty():
    low_mw = _predict(mw=300)
    high_mw = _predict(mw=600)
    assert high_mw.risk_score > low_mw.risk_score


def test_low_fu_adds_ppb_penalty():
    high_fu = _predict(fu_plasma=0.5)
    low_fu = _predict(fu_plasma=0.005)
    assert low_fu.risk_score > high_fu.risk_score


def test_risk_score_high_compound():
    """Compound with many risk factors should be classified high."""
    result = _predict(
        logp=4.5,
        daily_dose_mg=200,
        has_reactive_metabolite=True,
        has_mitochondrial_liability=True,
        structural_alert_count=3,
    )
    assert result.risk_class == "high"
    assert result.dili_likelihood_pct > 50


def test_risk_score_low_compound():
    """Compound with few risk factors should be classified low."""
    result = _predict(
        logp=0.0,
        daily_dose_mg=5,
        fu_plasma=0.5,
        mw=200,
        has_reactive_metabolite=False,
        has_mitochondrial_liability=False,
        structural_alert_count=0,
    )
    assert result.risk_class == "low"


def test_reactive_metabolite_adds_avoid_bioactivation_strategy():
    result = _predict(has_reactive_metabolite=True)
    assert "avoid_bioactivation_pathways" in result.mitigation_strategies


def test_high_dose_adds_dose_reduction_strategy():
    result = _predict(daily_dose_mg=150)
    assert "consider_dose_reduction" in result.mitigation_strategies


def test_high_logp_adds_hydrophilicity_strategy():
    result = _predict(logp=4.0)
    assert "improve_hydrophilicity" in result.mitigation_strategies


def test_mitochondrial_adds_mtdna_strategy():
    result = _predict(has_mitochondrial_liability=True)
    assert "assess_mtDNA_integrity" in result.mitigation_strategies


# ---------------------------------------------------------------------------
# Screen compounds
# ---------------------------------------------------------------------------


def test_screen_compounds_sorted_descending():
    compounds = [
        _make_compound(drug_name="Low", logp=0.5, daily_dose_mg=5),
        _make_compound(
            drug_name="High",
            logp=4.5,
            daily_dose_mg=200,
            has_reactive_metabolite=True,
        ),
        _make_compound(drug_name="Mid", logp=2.0, daily_dose_mg=50, structural_alert_count=2),
    ]
    results = screen_compounds(compounds)
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].drug_name == "High"


def test_screen_compounds_returns_all():
    compounds = [_make_compound(drug_name=f"Drug{i}") for i in range(5)]
    results = screen_compounds(compounds)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        _predict(logp=11.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        _predict(logp=-6.0)


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        _predict(mw=0.0)


def test_invalid_daily_dose_zero():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        _predict(daily_dose_mg=0.0)


def test_invalid_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        _predict(fu_plasma=0.0)


def test_invalid_fu_plasma_gt_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        _predict(fu_plasma=1.5)


def test_invalid_structural_alert_count_negative():
    with pytest.raises(ValueError, match="structural_alert_count"):
        _predict(structural_alert_count=-1)
