"""Tests for Phase 973 — drug_absorption_food_effect."""

import pytest

from omega_pbpk.prediction.drug_absorption_food_effect import (
    FoodEffectResult,
    predict_food_effect,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_food_effect_result():
    result = predict_food_effect("TestDrug", logp=1.0, pka=7.0)
    assert isinstance(result, FoodEffectResult)


def test_result_is_frozen():
    result = predict_food_effect("TestDrug", logp=1.0, pka=7.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fa bounds
# ---------------------------------------------------------------------------


def test_fa_fasted_in_range():
    result = predict_food_effect("Drug", logp=2.0, pka=6.0, ionization_type="acid")
    assert 0.0 <= result.fa_fasted <= 1.0


def test_fa_fed_in_range():
    result = predict_food_effect("Drug", logp=2.0, pka=6.0, ionization_type="acid")
    assert 0.0 <= result.fa_fed <= 1.0


# ---------------------------------------------------------------------------
# Lipophilic acid — positive food effect (bile enhancement)
# ---------------------------------------------------------------------------


def test_lipophilic_acid_positive_food_effect():
    """Lipophilic acid (logP=3, pKa=5): bile enhances solubility → positive food effect."""
    result = predict_food_effect(
        "LipophilicAcid",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        dose_mg=200.0,
        solubility_mg_mL=0.1,
    )
    assert result.food_effect_classification == "positive"
    assert result.auc_ratio_fed_fasted > 1.25


def test_lipophilic_acid_fed_higher_than_fasted():
    result = predict_food_effect(
        "LipophilicAcid",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        dose_mg=200.0,
        solubility_mg_mL=0.1,
    )
    assert result.fa_fed >= result.fa_fasted


# ---------------------------------------------------------------------------
# Highly basic drug — pH effect
# ---------------------------------------------------------------------------


def test_highly_basic_drug_neutral_fraction():
    """Highly basic drug (pKa=9): mostly ionized at both gastric pHs."""
    result = predict_food_effect(
        "StrongBase",
        logp=1.0,
        pka=9.0,
        ionization_type="base",
        dose_mg=100.0,
        solubility_mg_mL=1.0,
    )
    # At fed pH=4, more neutral fraction for a base with pKa=9 → fa_fed >= fa_fasted
    # (fed pH=4 > fasted pH=2, so more neutral at fasted for a base — actually opposite)
    # alpha_base(pH=2, pKa=9) = 1/(1+10^(2-9))=1/(1+1e-7) ≈ 1
    # alpha_base(pH=4, pKa=9) = 1/(1+10^(4-9))=1/(1+1e-5) ≈ 1
    # Both nearly 1 at these pHs since pKa >> pH
    assert result.fa_fasted > 0
    assert result.fa_fed > 0
    # bile enhancement: logp=1 → bile_factor = 1+1*0.5=1.5
    assert result.bile_enhancement_factor == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------


def test_neutral_drug_auc_ratio_near_one_with_bile():
    """Neutral drug: ionization unchanged, only bile effect applies."""
    result = predict_food_effect(
        "NeutralDrug",
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
        dose_mg=100.0,
        solubility_mg_mL=5.0,
    )
    # logP=0 → bile_factor = 1.0 (no enhancement), neutral fraction = 1.0 both states
    assert result.bile_enhancement_factor == pytest.approx(1.0)
    assert result.auc_ratio_fed_fasted == pytest.approx(1.0, abs=1e-6)


def test_neutral_drug_classification_none():
    result = predict_food_effect(
        "NeutralDrug",
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
    )
    assert result.food_effect_classification == "none"


# ---------------------------------------------------------------------------
# Classification completeness
# ---------------------------------------------------------------------------


def test_classification_in_valid_set():
    result = predict_food_effect("Drug", logp=1.0, pka=7.0)
    assert result.food_effect_classification in {"positive", "negative", "none"}


def test_recommendation_in_valid_set():
    result = predict_food_effect("Drug", logp=1.0, pka=7.0)
    assert result.recommendation in {"take with food", "take fasted", "no preference"}


def test_positive_classification_gives_take_with_food():
    result = predict_food_effect(
        "LipophilicAcid",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        dose_mg=200.0,
        solubility_mg_mL=0.1,
    )
    if result.food_effect_classification == "positive":
        assert result.recommendation == "take with food"


def test_none_classification_gives_no_preference():
    result = predict_food_effect(
        "NeutralDrug",
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
    )
    if result.food_effect_classification == "none":
        assert result.recommendation == "no preference"


# ---------------------------------------------------------------------------
# AUC ratio positive
# ---------------------------------------------------------------------------


def test_auc_ratio_is_positive():
    result = predict_food_effect("Drug", logp=2.0, pka=6.0)
    assert result.auc_ratio_fed_fasted > 0.0


# ---------------------------------------------------------------------------
# Bile enhancement factor >= 1.0
# ---------------------------------------------------------------------------


def test_bile_enhancement_factor_gte_one_high_logp():
    result = predict_food_effect("Drug", logp=4.0, pka=7.0)
    assert result.bile_enhancement_factor >= 1.0


def test_bile_enhancement_factor_gte_one_negative_logp():
    """Negative logP → bile factor stays at 1.0 (no enhancement)."""
    result = predict_food_effect("Drug", logp=-2.0, pka=7.0)
    assert result.bile_enhancement_factor >= 1.0
    assert result.bile_enhancement_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stored fields correctness
# ---------------------------------------------------------------------------


def test_stored_drug_name():
    result = predict_food_effect("Aspirin", logp=1.2, pka=3.5, ionization_type="acid")
    assert result.drug_name == "Aspirin"


def test_stored_logp_pka():
    result = predict_food_effect("Drug", logp=2.5, pka=6.5, ionization_type="base")
    assert result.logp == pytest.approx(2.5)
    assert result.pka == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_too_high():
    with pytest.raises(ValueError, match="pka"):
        predict_food_effect("Drug", logp=1.0, pka=15.0)


def test_invalid_pka_too_low():
    with pytest.raises(ValueError, match="pka"):
        predict_food_effect("Drug", logp=1.0, pka=-1.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_food_effect("Drug", logp=1.0, pka=7.0, ionization_type="zwitterion")


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        predict_food_effect("Drug", logp=11.0, pka=7.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        predict_food_effect("Drug", logp=-6.0, pka=7.0)
