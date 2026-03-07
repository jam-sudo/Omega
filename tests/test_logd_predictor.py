"""Tests for Phase 670 — LogD Predictor."""

import math

import pytest

from omega_pbpk.prediction.logd_predictor import (
    LogDPrediction,
    logD_pH_profile,
    predict_logD,
    screen_logD,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_neutral(logP: float = 2.5) -> LogDPrediction:
    return predict_logD("Neutral", logP=logP)


def make_acid(logP: float = 2.5, pka_acid: float = 4.0) -> LogDPrediction:
    return predict_logD("AcidDrug", logP=logP, pka_acid=pka_acid)


def make_base(logP: float = 2.5, pka_base: float = 9.0) -> LogDPrediction:
    return predict_logD("BaseDrug", logP=logP, pka_base=pka_base)


# ---------------------------------------------------------------------------
# Basic return type and finite fields
# ---------------------------------------------------------------------------


def test_returns_logd_prediction():
    result = make_neutral()
    assert isinstance(result, LogDPrediction)


def test_logD_at_pH74_finite():
    result = make_neutral()
    assert math.isfinite(result.logD_at_pH74)


def test_logD_at_pH_gastric_finite():
    result = make_neutral()
    assert math.isfinite(result.logD_at_pH_gastric)


def test_logD_at_pH_intestinal_finite():
    result = make_neutral()
    assert math.isfinite(result.logD_at_pH_intestinal)


def test_notes_nonempty():
    result = make_neutral()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Neutral compound (no pKa)
# ---------------------------------------------------------------------------


def test_neutral_logD_equals_logP_at_all_pH():
    logP = 3.0
    result = predict_logD("Neutral", logP=logP)
    assert math.isclose(result.logD_at_pH74, logP, rel_tol=1e-9)
    assert math.isclose(result.logD_at_pH_gastric, logP, rel_tol=1e-9)
    assert math.isclose(result.logD_at_pH_intestinal, logP, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Acid behaviour
# ---------------------------------------------------------------------------


def test_acid_logD_pH74_less_than_logP():
    """Acid (pKa=4) is mostly ionized at pH 7.4 => logD < logP."""
    result = make_acid(logP=2.5, pka_acid=4.0)
    assert result.logD_at_pH74 < result.logP


def test_acid_logD_gastric_greater_than_pH74():
    """Acid is un-ionized at low pH => logD closer to logP at pH 1.5."""
    result = make_acid(logP=2.5, pka_acid=4.0)
    assert result.logD_at_pH_gastric > result.logD_at_pH74


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


def test_base_logD_pH74_less_than_logP():
    """Base (pKa=9) is mostly ionized at pH 7.4 => logD < logP."""
    result = make_base(logP=2.5, pka_base=9.0)
    assert result.logD_at_pH74 < result.logP


def test_base_logD_low_pH_less_than_high_pH():
    """Base is more ionized (lower logD) at low pH."""
    result = make_base(logP=2.5, pka_base=9.0)
    assert result.logD_at_pH_gastric < result.logD_at_pH74


# ---------------------------------------------------------------------------
# Classification fields
# ---------------------------------------------------------------------------


def test_membrane_permeability_class_valid_values():
    result = make_neutral()
    assert result.membrane_permeability_class in {"high", "moderate", "low"}


def test_cns_penetration_likelihood_valid_values():
    result = make_neutral()
    assert result.cns_penetration_likelihood in {"high", "moderate", "low"}


def test_gi_absorption_prediction_valid_values():
    result = make_neutral()
    assert result.gi_absorption_prediction in {"excellent", "good", "poor"}


def test_ionization_at_pH74_in_range():
    result = make_acid()
    assert 0.0 <= result.ionization_at_pH74 <= 1.0


# ---------------------------------------------------------------------------
# High logP neutral compound
# ---------------------------------------------------------------------------


def test_high_logP_neutral_high_permeability():
    result = predict_logD("HighLogP", logP=3.0)
    assert result.membrane_permeability_class == "high"


def test_high_logP_neutral_good_cns():
    result = predict_logD("HighLogP", logP=2.0)
    # logD=2.0 in [1,3] => "high" CNS
    assert result.cns_penetration_likelihood == "high"


# ---------------------------------------------------------------------------
# Highly ionized at pH 7.4 => low permeability
# ---------------------------------------------------------------------------


def test_highly_ionized_acid_low_permeability():
    """Acid with pKa=2 => logD at pH 7.4 deeply negative => low permeability."""
    result = predict_logD("StrongAcid", logP=0.0, pka_acid=2.0)
    assert result.membrane_permeability_class == "low"


# ---------------------------------------------------------------------------
# logD_pH_profile
# ---------------------------------------------------------------------------


def test_logD_pH_profile_returns_list_of_tuples():
    profile = logD_pH_profile("Drug", logP=2.0, pka_acid=4.0)
    assert isinstance(profile, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in profile)


def test_logD_pH_profile_default_length():
    profile = logD_pH_profile("Drug", logP=2.0)
    # Default pH_range = [1..10] => 10 entries
    assert len(profile) == 10


def test_logD_pH_profile_custom_length():
    custom_pH = [1.0, 3.0, 5.0, 7.0, 9.0]
    profile = logD_pH_profile("Drug", logP=2.0, pH_range=custom_pH)
    assert len(profile) == len(custom_pH)


# ---------------------------------------------------------------------------
# screen_logD
# ---------------------------------------------------------------------------


def test_screen_logD_sorted_descending():
    compounds = [
        {"compound_name": "Low", "logP": -1.0},
        {"compound_name": "High", "logP": 3.0},
        {"compound_name": "Mid", "logP": 1.5},
    ]
    results = screen_logD(compounds)
    vals = [r.logD_at_pH74 for r in results]
    assert vals == sorted(vals, reverse=True)


def test_screen_logD_empty_returns_empty():
    assert screen_logD([]) == []


# ---------------------------------------------------------------------------
# Zwitterion
# ---------------------------------------------------------------------------


def test_zwitterion_logD_computed():
    """Compound with both pka_acid and pka_base => logD computed (not equal to logP)."""
    result = predict_logD("Zwitterion", logP=2.0, pka_acid=4.0, pka_base=9.0)
    assert math.isfinite(result.logD_at_pH74)
    # acid pKa=4 => ionized at pH 7.4; base pKa=9 => also ionized => logD < logP
    assert result.logD_at_pH74 < result.logP


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_pka_acid_negative_raises():
    with pytest.raises(ValueError):
        predict_logD("Drug", logP=2.0, pka_acid=-1.0)


def test_validation_pka_base_too_large_raises():
    with pytest.raises(ValueError):
        predict_logD("Drug", logP=2.0, pka_base=15.0)
