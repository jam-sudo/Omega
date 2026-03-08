"""Tests for Drug-Excipient Compatibility Predictor — Phase 1069."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_excipient_compatibility_p1069 import (
    DrugExcipientCompatibilityResult,
    predict_compatibility,
    screen_excipients,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compat(**kwargs) -> DrugExcipientCompatibilityResult:
    """Call predict_compatibility with sensible defaults."""
    defaults = dict(
        drug_name="DrugA",
        excipient_name="ExcipientX",
        drug_pka=7.0,
        excipient_pka=None,
        drug_has_amine=False,
        excipient_is_reducing_sugar=False,
        drug_oxidation_sensitive=False,
        excipient_peroxide_level="low",
        drug_hygroscopic=False,
        excipient_hygroscopic=False,
    )
    defaults.update(kwargs)
    return predict_compatibility(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _compat()
    assert isinstance(result, DrugExcipientCompatibilityResult)


def test_drug_name_stored():
    result = _compat(drug_name="Aspirin")
    assert result.drug_name == "Aspirin"


def test_excipient_name_stored():
    result = _compat(excipient_name="Lactose")
    assert result.excipient_name == "Lactose"


# ---------------------------------------------------------------------------
# Compatible excipient (no risks) → score 100
# ---------------------------------------------------------------------------


def test_fully_compatible_score_is_100():
    result = _compat()
    assert result.compatibility_score == pytest.approx(100.0)


def test_fully_compatible_category():
    result = _compat()
    assert result.risk_category == "compatible"


def test_fully_compatible_no_risks():
    result = _compat()
    assert result.maillard_risk is False
    assert result.oxidation_risk is False
    assert result.acid_base_risk is False
    assert result.hygroscopic_risk is False


# ---------------------------------------------------------------------------
# Maillard reaction risk
# ---------------------------------------------------------------------------


def test_maillard_risk_detected():
    result = _compat(drug_has_amine=True, excipient_is_reducing_sugar=True)
    assert result.maillard_risk is True


def test_maillard_reduces_score():
    compatible = _compat()
    maillard = _compat(drug_has_amine=True, excipient_is_reducing_sugar=True)
    assert maillard.compatibility_score < compatible.compatibility_score


def test_no_maillard_without_amine():
    result = _compat(drug_has_amine=False, excipient_is_reducing_sugar=True)
    assert result.maillard_risk is False


def test_no_maillard_without_reducing_sugar():
    result = _compat(drug_has_amine=True, excipient_is_reducing_sugar=False)
    assert result.maillard_risk is False


# ---------------------------------------------------------------------------
# Oxidation risk
# ---------------------------------------------------------------------------


def test_oxidation_risk_high_peroxide():
    result = _compat(drug_oxidation_sensitive=True, excipient_peroxide_level="high")
    assert result.oxidation_risk is True


def test_oxidation_risk_medium_peroxide():
    result = _compat(drug_oxidation_sensitive=True, excipient_peroxide_level="medium")
    assert result.oxidation_risk is True


def test_no_oxidation_risk_low_peroxide():
    result = _compat(drug_oxidation_sensitive=True, excipient_peroxide_level="low")
    assert result.oxidation_risk is False


def test_no_oxidation_risk_not_sensitive():
    result = _compat(drug_oxidation_sensitive=False, excipient_peroxide_level="high")
    assert result.oxidation_risk is False


def test_high_peroxide_penalty_greater_than_medium():
    r_high = _compat(drug_oxidation_sensitive=True, excipient_peroxide_level="high")
    r_med = _compat(drug_oxidation_sensitive=True, excipient_peroxide_level="medium")
    assert r_high.compatibility_score < r_med.compatibility_score


# ---------------------------------------------------------------------------
# Acid-base risk
# ---------------------------------------------------------------------------


def test_acid_base_risk_when_pka_close():
    result = _compat(drug_pka=7.0, excipient_pka=7.5)  # delta=0.5 < 2
    assert result.acid_base_risk is True


def test_no_acid_base_risk_when_excipient_pka_none():
    result = _compat(drug_pka=7.0, excipient_pka=None)
    assert result.acid_base_risk is False


def test_no_acid_base_risk_when_pka_far_apart():
    result = _compat(drug_pka=7.0, excipient_pka=12.0)  # delta=5 > 2
    assert result.acid_base_risk is False


def test_acid_base_risk_exactly_at_threshold_not_triggered():
    # delta == 2.0 should not trigger (strict < 2)
    result = _compat(drug_pka=7.0, excipient_pka=9.0)  # delta=2.0
    assert result.acid_base_risk is False


# ---------------------------------------------------------------------------
# Hygroscopic risk
# ---------------------------------------------------------------------------


def test_hygroscopic_risk_both_hygroscopic():
    result = _compat(drug_hygroscopic=True, excipient_hygroscopic=True)
    assert result.hygroscopic_risk is True


def test_no_hygroscopic_risk_only_drug():
    result = _compat(drug_hygroscopic=True, excipient_hygroscopic=False)
    assert result.hygroscopic_risk is False


def test_no_hygroscopic_risk_only_excipient():
    result = _compat(drug_hygroscopic=False, excipient_hygroscopic=True)
    assert result.hygroscopic_risk is False


# ---------------------------------------------------------------------------
# risk_category mapping
# ---------------------------------------------------------------------------


def test_risk_category_compatible():
    result = _compat()
    assert result.risk_category == "compatible"


def test_risk_category_minor_concern():
    # One small penalty (hygroscopic = 20) → score = 80 → minor_concern
    result = _compat(drug_hygroscopic=True, excipient_hygroscopic=True)
    assert result.risk_category == "minor_concern"


def test_risk_category_moderate_concern():
    # Acid-base (30) + hygroscopic (20) = 50 → score = 50 → moderate_concern
    result = _compat(
        drug_pka=7.0,
        excipient_pka=7.5,
        drug_hygroscopic=True,
        excipient_hygroscopic=True,
    )
    assert result.risk_category == "moderate_concern"


def test_risk_category_incompatible():
    # Maillard (35) + oxidation high (40) = 75 → score = 25 → incompatible
    result = _compat(
        drug_has_amine=True,
        excipient_is_reducing_sugar=True,
        drug_oxidation_sensitive=True,
        excipient_peroxide_level="high",
    )
    assert result.risk_category == "incompatible"


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


def test_score_in_0_to_100():
    result = _compat(
        drug_has_amine=True,
        excipient_is_reducing_sugar=True,
        drug_oxidation_sensitive=True,
        excipient_peroxide_level="high",
        drug_pka=7.0,
        excipient_pka=7.5,
        drug_hygroscopic=True,
        excipient_hygroscopic=True,
    )
    assert 0.0 <= result.compatibility_score <= 100.0


# ---------------------------------------------------------------------------
# screen_excipients
# ---------------------------------------------------------------------------


def test_screen_excipients_returns_list():
    excipients = [
        {"name": "Lactose", "is_reducing_sugar": True},
        {"name": "MCC", "is_reducing_sugar": False},
    ]
    results = screen_excipients("DrugA", excipients)
    assert isinstance(results, list)
    assert all(isinstance(r, DrugExcipientCompatibilityResult) for r in results)


def test_screen_excipients_sorted_descending():
    excipients = [
        {"name": "Lactose", "is_reducing_sugar": True},
        {"name": "MCC"},
        {"name": "HPMC"},
    ]
    results = screen_excipients("DrugA", excipients, drug_has_amine=True)
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_excipients_count():
    excipients = [{"name": f"Exc{i}"} for i in range(5)]
    results = screen_excipients("DrugB", excipients)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# recommended_action non-empty
# ---------------------------------------------------------------------------


def test_recommended_action_nonempty():
    result = _compat()
    assert isinstance(result.recommended_action, str)
    assert len(result.recommended_action) > 0


def test_recommended_action_nonempty_with_risks():
    result = _compat(drug_has_amine=True, excipient_is_reducing_sugar=True)
    assert len(result.recommended_action) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_drug_name_raises_value_error():
    with pytest.raises(ValueError, match="drug_name"):
        predict_compatibility(drug_name="", excipient_name="MCC")


def test_empty_excipient_name_raises_value_error():
    with pytest.raises(ValueError, match="excipient_name"):
        predict_compatibility(drug_name="DrugA", excipient_name="")
