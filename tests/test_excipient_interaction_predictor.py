"""Tests for Phase 451: Excipient-drug interaction predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.excipient_interaction_predictor import (
    SUPPORTED_EXCIPIENTS,
    ExcipientEffectResult,
    predict_excipient_effect,
    screen_excipients,
)


# ---------------------------------------------------------------------------
# Supported excipient list
# ---------------------------------------------------------------------------


def test_supported_excipients_count():
    assert len(SUPPORTED_EXCIPIENTS) >= 8


def test_supported_excipients_includes_required():
    required = {"HPMC", "PVP", "SDS", "Poloxamer 407", "Lactose", "MCC",
                "Stearic acid", "Tween 80"}
    assert required.issubset(set(SUPPORTED_EXCIPIENTS))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_excipient_effect_result():
    result = predict_excipient_effect(2.0, 300.0, "MCC")
    assert isinstance(result, ExcipientEffectResult)


# ---------------------------------------------------------------------------
# Excipient name is preserved
# ---------------------------------------------------------------------------


def test_excipient_name_preserved():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert result.excipient_name == "PVP"


# ---------------------------------------------------------------------------
# Compatibility score in [0, 100]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_compatibility_score_range(excipient):
    result = predict_excipient_effect(3.0, 350.0, excipient)
    assert 0.0 <= result.compatibility_score <= 100.0


# ---------------------------------------------------------------------------
# Permeability effect valid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_permeability_effect_valid(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.permeability_effect in ("enhance", "neutral", "reduce")


# ---------------------------------------------------------------------------
# Stability effect valid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_stability_effect_valid(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.stability_effect in ("positive", "neutral", "negative")


# ---------------------------------------------------------------------------
# Solubility fold change > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_solubility_fold_change_positive(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.solubility_fold_change > 0.0


# ---------------------------------------------------------------------------
# SDS enhances solubility for high-logP drug
# ---------------------------------------------------------------------------


def test_sds_enhances_solubility_high_logp():
    result = predict_excipient_effect(5.0, 300.0, "SDS")
    assert result.solubility_fold_change > 1.0


# ---------------------------------------------------------------------------
# Surfactants enhance permeability
# ---------------------------------------------------------------------------


def test_sds_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "SDS")
    assert result.permeability_effect == "enhance"


def test_tween80_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "Tween 80")
    assert result.permeability_effect == "enhance"


def test_poloxamer_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "Poloxamer 407")
    assert result.permeability_effect == "enhance"


# ---------------------------------------------------------------------------
# Stearic acid reduces permeability
# ---------------------------------------------------------------------------


def test_stearic_acid_reduces_permeability():
    result = predict_excipient_effect(2.0, 300.0, "Stearic acid")
    assert result.permeability_effect == "reduce"


# ---------------------------------------------------------------------------
# Stearic acid reduces solubility (fold change < 1.1)
# ---------------------------------------------------------------------------


def test_stearic_acid_solubility_below_one():
    result = predict_excipient_effect(2.0, 300.0, "Stearic acid")
    assert result.solubility_fold_change < 1.1


# ---------------------------------------------------------------------------
# MCC positive stability effect
# ---------------------------------------------------------------------------


def test_mcc_positive_stability():
    result = predict_excipient_effect(2.0, 300.0, "MCC")
    assert result.stability_effect == "positive"


# ---------------------------------------------------------------------------
# Lactose negative stability effect
# ---------------------------------------------------------------------------


def test_lactose_negative_stability():
    result = predict_excipient_effect(2.0, 300.0, "Lactose")
    assert result.stability_effect == "negative"


# ---------------------------------------------------------------------------
# High logP drug + surfactant → greater fold change than low logP
# ---------------------------------------------------------------------------


def test_high_logp_greater_fold_change_sds():
    high = predict_excipient_effect(6.0, 300.0, "SDS")
    low = predict_excipient_effect(1.0, 300.0, "SDS")
    assert high.solubility_fold_change > low.solubility_fold_change


def test_high_logp_greater_fold_change_tween80():
    high = predict_excipient_effect(5.0, 300.0, "Tween 80")
    low = predict_excipient_effect(0.0, 300.0, "Tween 80")
    assert high.solubility_fold_change > low.solubility_fold_change


# ---------------------------------------------------------------------------
# Invalid excipient raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_excipient_raises():
    with pytest.raises(ValueError, match="Unknown excipient"):
        predict_excipient_effect(2.0, 300.0, "Kryptonite")


def test_invalid_excipient_empty_string_raises():
    with pytest.raises(ValueError):
        predict_excipient_effect(2.0, 300.0, "")


# ---------------------------------------------------------------------------
# pKa amine + Lactose lowers compatibility
# ---------------------------------------------------------------------------


def test_lactose_amine_pka_lowers_score():
    without_pka = predict_excipient_effect(2.0, 300.0, "Lactose")
    with_pka = predict_excipient_effect(2.0, 300.0, "Lactose", drug_pka=9.0)
    assert with_pka.compatibility_score < without_pka.compatibility_score


# ---------------------------------------------------------------------------
# screen_excipients returns list of correct length (all excipients)
# ---------------------------------------------------------------------------


def test_screen_all_excipients_length():
    results = screen_excipients(2.0, 300.0)
    assert len(results) == len(SUPPORTED_EXCIPIENTS)


# ---------------------------------------------------------------------------
# screen_excipients returns sorted by compatibility_score descending
# ---------------------------------------------------------------------------


def test_screen_sorted_by_score():
    results = screen_excipients(3.0, 300.0)
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# screen_excipients subset
# ---------------------------------------------------------------------------


def test_screen_subset_excipients():
    subset = ["SDS", "MCC", "Lactose"]
    results = screen_excipients(3.0, 300.0, excipients=subset)
    assert len(results) == 3
    names = {r.excipient_name for r in results}
    assert names == set(subset)


# ---------------------------------------------------------------------------
# screen_excipients invalid excipient raises ValueError
# ---------------------------------------------------------------------------


def test_screen_invalid_excipient_raises():
    with pytest.raises(ValueError):
        screen_excipients(2.0, 300.0, excipients=["SDS", "InvalidExcipient"])


# ---------------------------------------------------------------------------
# Mechanism and notes are non-empty strings
# ---------------------------------------------------------------------------


def test_mechanism_nonempty():
    result = predict_excipient_effect(2.0, 300.0, "HPMC")
    assert isinstance(result.mechanism, str) and len(result.mechanism) > 0


def test_notes_nonempty():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# HPMC improves solubility vs pure drug (fold change >= 1.0)
# ---------------------------------------------------------------------------


def test_hpmc_solubility_improvement_low_logp():
    result = predict_excipient_effect(0.0, 300.0, "HPMC")
    # HPMC should provide at least some solubility improvement
    assert result.solubility_fold_change >= 1.0


# ---------------------------------------------------------------------------
# PVP stability positive
# ---------------------------------------------------------------------------


def test_pvp_stability_positive():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert result.stability_effect == "positive"
