"""Tests for Phase 994: Formulation excipient interaction prediction."""

import pytest

from omega_pbpk.prediction.formulation_excipient_interaction import (
    ExcipientInteractionResult,
    predict_excipient_interaction,
    screen_excipient_compatibility,
)

_ALL_EXCIPIENTS = [
    "MCC",
    "lactose",
    "HPMC",
    "PVP",
    "magnesium_stearate",
    "croscarmellose",
    "starch",
    "talc",
]

_VALID_INTERACTION_TYPES = {"none", "physical", "chemical", "electrostatic"}
_VALID_SEVERITIES = {"none", "minor", "moderate", "major"}
_VALID_RECOMMENDATIONS = {"compatible", "use with caution", "avoid"}


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_excipient_interaction_result():
    result = predict_excipient_interaction("TestDrug", "MCC")
    assert isinstance(result, ExcipientInteractionResult)


# ---------------------------------------------------------------------------
# Score in valid range
# ---------------------------------------------------------------------------


def test_compatibility_score_in_range():
    for exc in _ALL_EXCIPIENTS:
        result = predict_excipient_interaction("Drug", exc)
        assert 0.0 <= result.compatibility_score <= 100.0, exc


# ---------------------------------------------------------------------------
# Enumerated field values
# ---------------------------------------------------------------------------


def test_interaction_type_valid():
    for exc in _ALL_EXCIPIENTS:
        result = predict_excipient_interaction("Drug", exc)
        assert result.interaction_type in _VALID_INTERACTION_TYPES, exc


def test_severity_valid():
    for exc in _ALL_EXCIPIENTS:
        result = predict_excipient_interaction("Drug", exc)
        assert result.interaction_severity in _VALID_SEVERITIES, exc


def test_recommendation_valid():
    for exc in _ALL_EXCIPIENTS:
        result = predict_excipient_interaction("Drug", exc)
        assert result.recommendation in _VALID_RECOMMENDATIONS, exc


# ---------------------------------------------------------------------------
# Specific rule checks
# ---------------------------------------------------------------------------


def test_primary_amine_lactose_major_severity():
    result = predict_excipient_interaction("PrimaryAmineDrug", "lactose", has_primary_amine=True)
    assert result.interaction_severity == "major"


def test_primary_amine_lactose_avoid():
    result = predict_excipient_interaction("PrimaryAmineDrug", "lactose", has_primary_amine=True)
    assert result.recommendation == "avoid"


def test_acid_drug_magnesium_stearate_reduced_score():
    acid_result = predict_excipient_interaction(
        "AcidDrug", "magnesium_stearate", drug_ionization_type="acid"
    )
    neutral_result = predict_excipient_interaction(
        "NeutralDrug", "magnesium_stearate", drug_ionization_type="neutral"
    )
    assert acid_result.compatibility_score < neutral_result.compatibility_score


def test_mcc_perfect_score():
    result = predict_excipient_interaction("AnyDrug", "MCC")
    assert result.compatibility_score == pytest.approx(100.0)


def test_talc_high_score():
    result = predict_excipient_interaction("AnyDrug", "talc")
    assert result.compatibility_score == pytest.approx(98.0)


def test_mcc_no_interaction():
    result = predict_excipient_interaction("AnyDrug", "MCC")
    assert result.interaction_type == "none"
    assert result.interaction_severity == "none"


def test_carboxylic_acid_magnesium_stearate_reduced_score():
    result = predict_excipient_interaction(
        "CarboxylicAcidDrug",
        "magnesium_stearate",
        has_carboxylic_acid=True,
    )
    assert result.compatibility_score < 80.0


def test_high_logp_hpmc_reduced_score():
    high_logp = predict_excipient_interaction("LipophilicDrug", "HPMC", drug_logp=5.0)
    low_logp = predict_excipient_interaction("HydrophilicDrug", "HPMC", drug_logp=1.0)
    assert high_logp.compatibility_score < low_logp.compatibility_score


# ---------------------------------------------------------------------------
# screen_excipient_compatibility
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_excipient_compatibility("TestDrug", _ALL_EXCIPIENTS)
    assert isinstance(results, list)
    assert len(results) == len(_ALL_EXCIPIENTS)


def test_screen_sorted_descending():
    results = screen_excipient_compatibility("TestDrug", _ALL_EXCIPIENTS)
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_all_are_results():
    results = screen_excipient_compatibility("TestDrug", _ALL_EXCIPIENTS)
    for r in results:
        assert isinstance(r, ExcipientInteractionResult)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_excipient_raises_value_error():
    with pytest.raises(ValueError, match="excipient_name"):
        predict_excipient_interaction("Drug", "unknownExcipient")


def test_invalid_ionization_type_raises_value_error():
    with pytest.raises(ValueError, match="drug_ionization_type"):
        predict_excipient_interaction("Drug", "MCC", drug_ionization_type="zwitterion")
