"""Tests for Phase 964 — ROS burden prediction."""

import pytest

from omega_pbpk.prediction.reactive_oxygen_species_burden import (
    ROSBurdenResult,
    predict_ros_burden,
    screen_ros_risk,
)

_VALID_CATEGORIES = {"high", "moderate", "low"}
_VALID_MECHANISMS = {
    "quinone_redox_cycling",
    "nitro_reduction_activation",
    "cyp_oxidative_metabolism",
    "minimal_ros_generation",
}


def _make_basic(**kwargs):
    defaults = dict(
        drug_name="test_drug",
        logp=1.0,
        mw=300.0,
        pka=7.0,
        ionization_type="neutral",
        has_quinone=False,
        has_nitro_group=False,
    )
    defaults.update(kwargs)
    return predict_ros_burden(**defaults)


# ── Return-type checks ───────────────────────────────────────────────────────


def test_return_type():
    result = _make_basic()
    assert isinstance(result, ROSBurdenResult)


def test_frozen_dataclass():
    result = _make_basic()
    with pytest.raises((AttributeError, TypeError)):
        result.ros_score = 99.0  # type: ignore[misc]


def test_ros_score_in_range():
    result = _make_basic()
    assert 0 <= result.ros_score <= 100


def test_ros_risk_category_valid():
    result = _make_basic()
    assert result.ros_risk_category in _VALID_CATEGORIES


def test_gsh_depletion_risk_valid():
    result = _make_basic()
    assert result.gsh_depletion_risk in _VALID_CATEGORIES


def test_mitochondrial_toxicity_index_in_range():
    result = _make_basic()
    assert 0 <= result.mitochondrial_toxicity_index <= 100


def test_oxidative_stress_mechanism_nonempty():
    result = _make_basic()
    assert isinstance(result.oxidative_stress_mechanism, str)
    assert len(result.oxidative_stress_mechanism) > 0


def test_dili_ros_risk_is_bool():
    result = _make_basic()
    assert isinstance(result.dili_ros_risk, bool)


def test_notes_nonempty():
    result = _make_basic()
    assert result.notes != ""


def test_drug_name_preserved():
    result = _make_basic(drug_name="doxorubicin")
    assert result.drug_name == "doxorubicin"


# ── Scoring logic ────────────────────────────────────────────────────────────


def test_quinone_raises_score():
    result = _make_basic(has_quinone=True)
    # base 10 + quinone 35 = 45 minimum
    assert result.ros_score > 40


def test_quinone_risk_category_high_or_moderate():
    result = _make_basic(has_quinone=True)
    assert result.ros_risk_category in {"high", "moderate"}


def test_nitro_raises_score():
    result = _make_basic(has_nitro_group=True)
    # base 10 + nitro 25 = 35 minimum
    assert result.ros_score >= 35


def test_no_alerts_neutral_low_logp_low_risk():
    result = _make_basic(logp=1.0, mw=300.0, ionization_type="neutral", pka=7.0)
    assert result.ros_risk_category == "low"


def test_dili_risk_when_score_above_50():
    # quinone (35) + logP>3 and mw<400 (15) + base 10 = 60 → dili_ros_risk True
    result = _make_basic(has_quinone=True, logp=4.0, mw=300.0, ionization_type="neutral", pka=7.0)
    assert result.dili_ros_risk is True


def test_dili_risk_false_when_score_low():
    result = _make_basic(logp=1.0, mw=300.0, ionization_type="neutral")
    # score = 10, not > 50
    assert result.dili_ros_risk is False


# ── Mechanism assignment ─────────────────────────────────────────────────────


def test_quinone_mechanism():
    result = _make_basic(has_quinone=True)
    assert result.oxidative_stress_mechanism == "quinone_redox_cycling"


def test_nitro_mechanism_no_quinone():
    result = _make_basic(has_nitro_group=True, has_quinone=False)
    assert "nitro" in result.oxidative_stress_mechanism


def test_cyp_mechanism_when_score_above_30():
    # logP=4 + mw=300 + base pKa=10 → +15+10+10 = 35 → score 45
    result = _make_basic(
        logp=4.0,
        mw=300.0,
        ionization_type="base",
        pka=10.0,
        has_quinone=False,
        has_nitro_group=False,
    )
    assert result.ros_score > 30
    assert result.oxidative_stress_mechanism == "cyp_oxidative_metabolism"


# ── screen_ros_risk ───────────────────────────────────────────────────────────


def _sample_compounds():
    return [
        {"drug_name": "A", "logp": 1.0, "mw": 300.0, "pka": 7.0, "ionization_type": "neutral"},
        {
            "drug_name": "B",
            "logp": 4.0,
            "mw": 250.0,
            "pka": 7.0,
            "ionization_type": "neutral",
            "has_quinone": True,
        },
        {
            "drug_name": "C",
            "logp": 2.0,
            "mw": 350.0,
            "pka": 9.5,
            "ionization_type": "base",
            "has_nitro_group": True,
        },
    ]


def test_screen_returns_correct_length():
    compounds = _sample_compounds()
    results = screen_ros_risk(compounds)
    assert len(results) == len(compounds)


def test_screen_sorted_descending():
    compounds = _sample_compounds()
    results = screen_ros_risk(compounds)
    scores = [r.ros_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_returns_ros_burden_results():
    compounds = _sample_compounds()
    results = screen_ros_risk(compounds)
    for r in results:
        assert isinstance(r, ROSBurdenResult)


# ── Validation ────────────────────────────────────────────────────────────────


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_ros_burden("x", logp=1.0, mw=0.0, pka=7.0, ionization_type="neutral")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_ros_burden("x", logp=1.0, mw=-100.0, pka=7.0, ionization_type="neutral")


def test_validation_invalid_ionization():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_ros_burden("x", logp=1.0, mw=300.0, pka=7.0, ionization_type="zwitterion")
