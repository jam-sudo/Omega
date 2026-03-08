"""Tests for Phase 249: Excipient-drug interaction predictor."""

import pytest

from omega_pbpk.prediction.excipient_drug_interaction import (
    ExcipientInteractionResult,
    predict_excipient_interaction,
    screen_excipient_compatibility,
)

_VALID_AB_RISKS = {"none", "low", "moderate", "high"}
_VALID_COMPAT_CLASSES = {"compatible", "marginal", "incompatible"}


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_excipient_interaction_result(self):
        result = predict_excipient_interaction("DrugA", ["hydroxyl"], "lactose")
        assert isinstance(result, ExcipientInteractionResult)

    def test_drug_name_stored(self):
        result = predict_excipient_interaction("Metformin", ["amine"], "MCC")
        assert result.drug_name == "Metformin"

    def test_excipient_stored(self):
        result = predict_excipient_interaction("DrugB", [], "HPMC")
        assert result.excipient == "HPMC"

    def test_functional_groups_stored_as_tuple(self):
        result = predict_excipient_interaction("DrugC", ["amine", "hydroxyl"], "starch")
        assert isinstance(result.drug_functional_groups, tuple)
        assert "amine" in result.drug_functional_groups


# ---------------------------------------------------------------------------
# Maillard risk tests
# ---------------------------------------------------------------------------


class TestMaillardRisk:
    def test_amine_drug_plus_lactose_has_maillard_risk(self):
        result = predict_excipient_interaction("Drug", ["amine"], "lactose")
        assert result.maillard_risk_score > 0

    def test_neutral_drug_plus_lactose_has_no_maillard_risk(self):
        result = predict_excipient_interaction("Drug", ["hydroxyl"], "lactose")
        assert result.maillard_risk_score == 0.0

    def test_amine_drug_plus_mcc_has_no_maillard_risk(self):
        # MCC is not a reducing sugar
        result = predict_excipient_interaction("Drug", ["amine"], "MCC")
        assert result.maillard_risk_score == 0.0

    def test_amine_drug_plus_starch_has_maillard_risk(self):
        # Starch is a reducing sugar
        result = predict_excipient_interaction("Drug", ["amine"], "starch")
        assert result.maillard_risk_score > 0

    def test_maillard_risk_lactose_greater_than_starch(self):
        # lactose amine_reactivity=0.8 > starch 0.3
        r_lac = predict_excipient_interaction("Drug", ["amine"], "lactose")
        r_sta = predict_excipient_interaction("Drug", ["amine"], "starch")
        assert r_lac.maillard_risk_score > r_sta.maillard_risk_score


# ---------------------------------------------------------------------------
# Compatibility class and score tests
# ---------------------------------------------------------------------------


class TestCompatibilityClass:
    def test_compatibility_class_valid_values(self):
        result = predict_excipient_interaction("Drug", ["amine"], "lactose")
        assert result.compatibility_class in _VALID_COMPAT_CLASSES

    def test_overall_compatibility_in_range(self):
        result = predict_excipient_interaction(
            "Drug", ["amine", "carboxylic_acid"], "magnesium_stearate"
        )
        assert 0.0 <= result.overall_compatibility_score <= 100.0

    def test_compatible_class_score_threshold(self):
        result = predict_excipient_interaction("Drug", [], "HPMC")
        if result.compatibility_class == "compatible":
            assert result.overall_compatibility_score >= 70.0

    def test_incompatible_class_score_threshold(self):
        # amine + lactose: high maillard risk -> lower score
        result = predict_excipient_interaction("Drug", ["amine"], "lactose")
        if result.compatibility_class == "incompatible":
            assert result.overall_compatibility_score < 50.0

    def test_marginal_class_score_range(self):
        for exc in ["starch", "silicon_dioxide", "PVP"]:
            result = predict_excipient_interaction("Drug", ["amine"], exc)
            if result.compatibility_class == "marginal":
                assert 50.0 <= result.overall_compatibility_score < 70.0


# ---------------------------------------------------------------------------
# Acid-base incompatibility tests
# ---------------------------------------------------------------------------


class TestAcidBaseIncompatibility:
    def test_acid_base_incompatibility_valid_values(self):
        for exc in ["lactose", "MCC", "magnesium_stearate", "silicon_dioxide"]:
            result = predict_excipient_interaction("Drug", ["amine"], exc)
            assert result.acid_base_incompatibility_risk in _VALID_AB_RISKS

    def test_carboxylic_acid_plus_metal_salt_high_risk(self):
        result = predict_excipient_interaction("Drug", ["carboxylic_acid"], "magnesium_stearate")
        assert result.acid_base_incompatibility_risk == "high"

    def test_amine_plus_acid_base_reactive_moderate_risk(self):
        result = predict_excipient_interaction("Drug", ["amine"], "silicon_dioxide")
        assert result.acid_base_incompatibility_risk == "moderate"

    def test_no_reactive_groups_no_acid_base_risk(self):
        result = predict_excipient_interaction("Drug", ["hydroxyl"], "HPMC")
        assert result.acid_base_incompatibility_risk == "none"


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------


class TestScreenCompatibility:
    def test_screen_returns_list(self):
        results = screen_excipient_compatibility("Drug", ["amine"])
        assert isinstance(results, list)

    def test_screen_returns_seven_excipients(self):
        results = screen_excipient_compatibility("Drug", ["hydroxyl"])
        assert len(results) == 7

    def test_screen_sorted_descending(self):
        results = screen_excipient_compatibility("Drug", ["amine", "carboxylic_acid"])
        scores = [r.overall_compatibility_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_screen_all_return_types(self):
        results = screen_excipient_compatibility("Drug", [])
        for r in results:
            assert isinstance(r, ExcipientInteractionResult)


# ---------------------------------------------------------------------------
# Recommended alternative tests
# ---------------------------------------------------------------------------


class TestRecommendedAlternative:
    def test_incompatible_has_alternative(self):
        # amine + lactose is likely incompatible -> should suggest alternative
        result = predict_excipient_interaction("Drug", ["amine"], "lactose")
        if result.compatibility_class == "incompatible":
            assert result.recommended_alternative != ""

    def test_compatible_no_alternative(self):
        result = predict_excipient_interaction("Drug", [], "HPMC")
        if result.compatibility_class == "compatible":
            assert result.recommended_alternative == ""


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_excipient_raises(self):
        with pytest.raises(ValueError, match="Unknown excipient"):
            predict_excipient_interaction("Drug", ["amine"], "nonexistent_excipient")

    def test_invalid_drug_charge_raises(self):
        with pytest.raises(ValueError, match="drug_charge"):
            predict_excipient_interaction("Drug", ["amine"], "lactose", drug_charge="unknown")

    def test_empty_drug_name_raises(self):
        with pytest.raises((ValueError, Exception)):
            predict_excipient_interaction("", ["amine"], "lactose")
