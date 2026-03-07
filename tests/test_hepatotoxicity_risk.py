"""Tests for hepatotoxicity (DILI) risk prediction (Phase 978)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.hepatotoxicity_risk import (
    HepatotoxicityRiskResult,
    predict_hepatotoxicity_risk,
    screen_hepatotoxicity,
)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


def test_returns_result_type():
    r = predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0)
    assert isinstance(r, HepatotoxicityRiskResult)


def test_drug_name_preserved():
    r = predict_hepatotoxicity_risk("aspirin", logp=1.2, mw=180.0)
    assert r.drug_name == "aspirin"


def test_dili_score_in_valid_range():
    r = predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0)
    assert 0.0 <= r.dili_score <= 100.0


def test_dili_score_capped_at_100():
    # All risk factors present → capped at 100
    r = predict_hepatotoxicity_risk(
        "danger_drug",
        logp=5.0,
        mw=600.0,
        daily_dose_mg=500.0,
        has_reactive_metabolite=True,
        has_mitochondrial_liability=True,
        has_bsep_inhibition=True,
        ionization_type="acid",
    )
    assert r.dili_score <= 100.0


def test_dili_risk_valid_category():
    r = predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0)
    assert r.dili_risk in {"low", "moderate", "high"}


def test_bsep_inhibition_risk_valid():
    r = predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0)
    assert r.bsep_inhibition_risk in {"low", "moderate", "high"}


def test_mitochondrial_liability_risk_valid():
    r = predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0)
    assert r.mitochondrial_liability_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Reactive metabolite
# ---------------------------------------------------------------------------


def test_reactive_metabolite_higher_score():
    r_clean = predict_hepatotoxicity_risk("clean_drug", logp=1.0, mw=250.0)
    r_reactive = predict_hepatotoxicity_risk(
        "reactive_drug", logp=1.0, mw=250.0, has_reactive_metabolite=True
    )
    assert r_reactive.dili_score > r_clean.dili_score


# ---------------------------------------------------------------------------
# Dose-dependent risk
# ---------------------------------------------------------------------------


def test_high_dose_dose_dependent_true():
    r = predict_hepatotoxicity_risk("drug_a", logp=1.0, mw=250.0, daily_dose_mg=200.0)
    assert r.dose_dependent_risk is True


def test_low_dose_dose_dependent_false():
    r = predict_hepatotoxicity_risk("drug_a", logp=1.0, mw=250.0, daily_dose_mg=10.0)
    assert r.dose_dependent_risk is False


def test_exactly_100mg_dose_dependent_false():
    r = predict_hepatotoxicity_risk("drug_a", logp=1.0, mw=250.0, daily_dose_mg=100.0)
    assert r.dose_dependent_risk is False


def test_slightly_above_100mg_dose_dependent_true():
    r = predict_hepatotoxicity_risk("drug_a", logp=1.0, mw=250.0, daily_dose_mg=100.1)
    assert r.dose_dependent_risk is True


# ---------------------------------------------------------------------------
# BSEP inhibition risk
# ---------------------------------------------------------------------------


def test_bsep_flag_gives_high_risk():
    r = predict_hepatotoxicity_risk(
        "drug_a", logp=2.0, mw=300.0, has_bsep_inhibition=True
    )
    assert r.bsep_inhibition_risk == "high"


def test_acidic_lipophilic_moderate_bsep():
    r = predict_hepatotoxicity_risk(
        "drug_a", logp=4.0, mw=300.0, ionization_type="acid"
    )
    assert r.bsep_inhibition_risk == "moderate"


def test_neutral_low_logp_low_bsep():
    r = predict_hepatotoxicity_risk(
        "drug_a", logp=1.0, mw=200.0, ionization_type="neutral"
    )
    assert r.bsep_inhibition_risk == "low"


# ---------------------------------------------------------------------------
# Mitochondrial liability risk
# ---------------------------------------------------------------------------


def test_mito_flag_gives_high_mito_risk():
    r = predict_hepatotoxicity_risk(
        "drug_a", logp=2.0, mw=300.0, has_mitochondrial_liability=True
    )
    assert r.mitochondrial_liability_risk == "high"


def test_very_high_logp_moderate_mito():
    r = predict_hepatotoxicity_risk("drug_a", logp=5.0, mw=300.0)
    assert r.mitochondrial_liability_risk == "moderate"


# ---------------------------------------------------------------------------
# Risk categories by score
# ---------------------------------------------------------------------------


def test_all_risk_factors_high_dili():
    r = predict_hepatotoxicity_risk(
        "danger_drug",
        logp=4.0,
        mw=600.0,
        daily_dose_mg=200.0,
        has_reactive_metabolite=True,
        has_mitochondrial_liability=True,
        has_bsep_inhibition=True,
        ionization_type="acid",
    )
    assert r.dili_risk == "high"


def test_clean_drug_low_dili():
    r = predict_hepatotoxicity_risk(
        "safe_drug",
        logp=1.0,
        mw=200.0,
        daily_dose_mg=5.0,
    )
    assert r.dili_risk == "low"


# ---------------------------------------------------------------------------
# screen_hepatotoxicity
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 200.0},
        {"drug_name": "B", "logp": 4.0, "mw": 500.0, "daily_dose_mg": 200.0},
    ]
    results = screen_hepatotoxicity(compounds)
    assert isinstance(results, list)


def test_screen_sorted_descending():
    compounds = [
        {"drug_name": "low_risk", "logp": 0.5, "mw": 150.0, "daily_dose_mg": 5.0},
        {
            "drug_name": "high_risk",
            "logp": 5.0,
            "mw": 600.0,
            "daily_dose_mg": 300.0,
            "has_reactive_metabolite": True,
        },
        {"drug_name": "medium_risk", "logp": 3.5, "mw": 400.0, "daily_dose_mg": 50.0},
    ]
    results = screen_hepatotoxicity(compounds)
    scores = [r.dili_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_returns_correct_count():
    compounds = [
        {"drug_name": f"drug_{i}", "logp": float(i), "mw": 200.0 + i * 50}
        for i in range(5)
    ]
    results = screen_hepatotoxicity(compounds)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=-100.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        predict_hepatotoxicity_risk("drug_a", logp=2.0, mw=300.0, daily_dose_mg=0.0)


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_hepatotoxicity_risk(
            "drug_a", logp=2.0, mw=300.0, ionization_type="zwitterion"
        )
