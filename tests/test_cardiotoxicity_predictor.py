"""Tests for Phase 840 — Drug-Induced Cardiotoxicity Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.cardiotoxicity_predictor import (
    CardiotoxicityResult,
    predict_cardiotoxicity,
    screen_cardiotoxicity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _low_risk_drug(name: str = "SafeDrug") -> CardiotoxicityResult:
    """A neutral, hydrophilic drug with minimal cardiotoxicity risk."""
    return predict_cardiotoxicity(
        drug_name=name,
        logp=-1.0,
        mw_da=200.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )


def _high_risk_drug(name: str = "RiskyDrug") -> CardiotoxicityResult:
    """A lipophilic basic drug with structural alerts — high cardiotoxicity risk."""
    return predict_cardiotoxicity(
        drug_name=name,
        logp=4.0,
        mw_da=550.0,
        pka=9.0,
        has_amino_group=True,
        has_chloro=True,
        has_aromatic=True,
        drug_class="antiarrhythmic",
        cyp_inhibitor=True,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _low_risk_drug()
    assert isinstance(result, CardiotoxicityResult)


def test_result_is_frozen():
    """CardiotoxicityResult is frozen=True — should raise on attribute set."""
    result = _low_risk_drug()
    with pytest.raises(Exception):
        result.risk_score = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Risk score bounds
# ---------------------------------------------------------------------------


def test_risk_score_in_range_low():
    result = _low_risk_drug()
    assert 0.0 <= result.risk_score <= 100.0


def test_risk_score_in_range_high():
    result = _high_risk_drug()
    assert 0.0 <= result.risk_score <= 100.0


def test_high_risk_greater_than_low_risk():
    low = _low_risk_drug()
    high = _high_risk_drug()
    assert high.risk_score > low.risk_score


# ---------------------------------------------------------------------------
# Lipophilic basic drug → higher score than neutral hydrophilic
# ---------------------------------------------------------------------------


def test_basic_lipophilic_higher_score_than_neutral_hydrophilic():
    """logP>3 and pKa>7 triggers +20 hERG bonus."""
    neutral = predict_cardiotoxicity(
        drug_name="Neutral",
        logp=0.0,
        mw_da=300.0,
        pka=4.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    basic_lipophilic = predict_cardiotoxicity(
        drug_name="BasicLipophilic",
        logp=4.0,
        mw_da=300.0,
        pka=9.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    assert basic_lipophilic.risk_score > neutral.risk_score


# ---------------------------------------------------------------------------
# Drug class scoring
# ---------------------------------------------------------------------------


def test_antiarrhythmic_higher_than_other():
    antiarrhythmic = predict_cardiotoxicity(
        drug_name="Antiarrhythmic",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="antiarrhythmic",
        cyp_inhibitor=False,
    )
    other = predict_cardiotoxicity(
        drug_name="Other",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    assert antiarrhythmic.risk_score > other.risk_score


def test_antihistamine_higher_than_other():
    antihistamine = predict_cardiotoxicity(
        drug_name="Antihistamine",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="antihistamine",
        cyp_inhibitor=False,
    )
    other = predict_cardiotoxicity(
        drug_name="Other",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    assert antihistamine.risk_score > other.risk_score


# ---------------------------------------------------------------------------
# Structural alerts
# ---------------------------------------------------------------------------


def test_aminoalkyl_alert_detected():
    result = predict_cardiotoxicity(
        drug_name="AminoLipophilic",
        logp=3.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=True,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    assert "aminoalkyl_side_chain" in result.structural_alerts
    assert result.structural_alert_count >= 1


def test_halogenated_aromatic_alert_detected():
    result = predict_cardiotoxicity(
        drug_name="ChloroAromatic",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=True,
        has_aromatic=True,
        drug_class="other",
        cyp_inhibitor=False,
    )
    assert "halogenated_aromatic" in result.structural_alerts


def test_no_alerts_for_clean_drug():
    result = _low_risk_drug()
    assert result.structural_alert_count == 0
    assert result.structural_alerts == ""


# ---------------------------------------------------------------------------
# Overall risk valid values
# ---------------------------------------------------------------------------


def test_overall_risk_valid_values():
    valid = {"low", "moderate", "high", "critical"}
    for drug in [_low_risk_drug(), _high_risk_drug()]:
        assert drug.overall_risk in valid


def test_low_risk_drug_overall_risk():
    result = _low_risk_drug()
    # Base score = 10, no additional risk factors → score = 10 → "low"
    assert result.overall_risk == "low"


def test_high_risk_drug_overall_risk():
    result = _high_risk_drug()
    assert result.overall_risk in {"high", "critical"}


# ---------------------------------------------------------------------------
# Monitoring recommendation correlates with risk
# ---------------------------------------------------------------------------


def test_monitoring_recommendation_low_risk():
    result = _low_risk_drug()
    assert result.monitoring_recommendation == "standard monitoring"


def test_monitoring_recommendation_high_risk():
    result = _high_risk_drug()
    assert result.monitoring_recommendation == "cardiac monitoring required"


# ---------------------------------------------------------------------------
# CYP inhibitor effect
# ---------------------------------------------------------------------------


def test_cyp_inhibitor_raises_score():
    base = predict_cardiotoxicity(
        drug_name="Base",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=False,
    )
    with_cyp = predict_cardiotoxicity(
        drug_name="WithCYP",
        logp=1.0,
        mw_da=300.0,
        pka=5.0,
        has_amino_group=False,
        has_chloro=False,
        has_aromatic=False,
        drug_class="other",
        cyp_inhibitor=True,
    )
    assert with_cyp.risk_score > base.risk_score


# ---------------------------------------------------------------------------
# Contraindications
# ---------------------------------------------------------------------------


def test_contraindications_herg_high():
    """hERG high → contraindications should mention QT-prolonging drugs."""
    result = _high_risk_drug()
    if result.herg_risk == "high":
        assert "QT-prolonging drugs" in result.contraindications


def test_contraindications_low_risk_empty():
    result = _low_risk_drug()
    # Low risk drug should have no hERG contraindications
    if result.herg_risk != "high":
        assert result.contraindications == ""


# ---------------------------------------------------------------------------
# screen_cardiotoxicity
# ---------------------------------------------------------------------------


def test_screen_cardiotoxicity_sorted_descending():
    candidates = [
        {
            "drug_name": "SafeDrug",
            "logp": -1.0,
            "mw_da": 200.0,
            "pka": 5.0,
            "has_amino_group": False,
            "has_chloro": False,
            "has_aromatic": False,
            "drug_class": "other",
            "cyp_inhibitor": False,
        },
        {
            "drug_name": "ModerateRisk",
            "logp": 2.5,
            "mw_da": 350.0,
            "pka": 7.5,
            "has_amino_group": False,
            "has_chloro": False,
            "has_aromatic": True,
            "drug_class": "antihistamine",
            "cyp_inhibitor": False,
        },
        {
            "drug_name": "HighRisk",
            "logp": 4.0,
            "mw_da": 550.0,
            "pka": 9.0,
            "has_amino_group": True,
            "has_chloro": True,
            "has_aromatic": True,
            "drug_class": "antiarrhythmic",
            "cyp_inhibitor": True,
        },
    ]
    results = screen_cardiotoxicity(candidates)
    assert len(results) == 3
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_cardiotoxicity_return_type():
    candidates = [
        {
            "drug_name": "DrugA",
            "logp": 1.0,
            "mw_da": 300.0,
            "pka": 6.0,
            "has_amino_group": False,
            "has_chloro": False,
            "has_aromatic": False,
            "drug_class": "other",
            "cyp_inhibitor": False,
        }
    ]
    results = screen_cardiotoxicity(candidates)
    assert isinstance(results, list)
    assert isinstance(results[0], CardiotoxicityResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_logp_low_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_cardiotoxicity(
            "Drug",
            logp=-6.0,
            mw_da=300.0,
            pka=7.0,
            has_amino_group=False,
            has_chloro=False,
            has_aromatic=False,
            drug_class="other",
        )


def test_invalid_logp_high_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_cardiotoxicity(
            "Drug",
            logp=11.0,
            mw_da=300.0,
            pka=7.0,
            has_amino_group=False,
            has_chloro=False,
            has_aromatic=False,
            drug_class="other",
        )


def test_invalid_pka_low_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_cardiotoxicity(
            "Drug",
            logp=1.0,
            mw_da=300.0,
            pka=-1.0,
            has_amino_group=False,
            has_chloro=False,
            has_aromatic=False,
            drug_class="other",
        )


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_cardiotoxicity(
            "Drug",
            logp=1.0,
            mw_da=300.0,
            pka=15.0,
            has_amino_group=False,
            has_chloro=False,
            has_aromatic=False,
            drug_class="other",
        )


def test_invalid_drug_class_raises():
    with pytest.raises(ValueError, match="drug_class"):
        predict_cardiotoxicity(
            "Drug",
            logp=1.0,
            mw_da=300.0,
            pka=7.0,
            has_amino_group=False,
            has_chloro=False,
            has_aromatic=False,
            drug_class="unknown_class",
        )
