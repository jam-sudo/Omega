"""Tests for Phase 175: salt_screening module."""

import pytest

from omega_pbpk.prediction.salt_screening import (
    SaltFormResult,
    predict_salt_solubility,
    screen_salt_forms,
)

# ---------------------------------------------------------------------------
# predict_salt_solubility — basic correctness
# ---------------------------------------------------------------------------


class TestPredictSaltSolubility:
    def test_acid_sodium_viable(self):
        """Acid drug + sodium counterion should form viable salt (delta_pKa >> 3)."""
        result = predict_salt_solubility(
            drug_name="TestAcid",
            pka=4.0,
            ionization_type="acid",
            counterion_name="sodium",
            intrinsic_solubility_mg_mL=0.1,
        )
        assert result.salt_viable is True
        assert result.solubility_enhancement > 1.0
        assert result.predicted_solubility_mg_mL > 0.1

    def test_acid_sodium_counterion_type(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 0.5)
        assert result.counterion_type == "cation"
        assert result.counterion_name == "sodium"

    def test_delta_pka_calculation(self):
        """sodium pKa_counterion=14; drug pKa=4 → delta=10."""
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        assert abs(result.delta_pka - 10.0) < 1e-9

    def test_low_delta_pka_not_viable(self):
        """Acid drug pKa=12 + meglumine (pKa=9.5) → delta=2.5 < 3 → not viable."""
        result = predict_salt_solubility("Drug", 12.0, "acid", "meglumine", 1.0)
        assert result.salt_viable is False
        # No real enhancement expected
        assert result.solubility_enhancement == pytest.approx(1.0)

    def test_not_viable_solubility_unchanged(self):
        result = predict_salt_solubility("Drug", 12.0, "acid", "meglumine", 2.0)
        assert result.predicted_solubility_mg_mL == pytest.approx(2.0)

    def test_viable_predicted_solubility(self):
        """Sodium: factor=3.0, delta=10 → enhancement=3.0*min(10,10/3)=10."""
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        expected_enhancement = 3.0 * min(10.0, 10.0 / 3.0)
        assert result.solubility_enhancement == pytest.approx(expected_enhancement, rel=1e-6)
        assert result.predicted_solubility_mg_mL == pytest.approx(expected_enhancement, rel=1e-6)

    def test_base_hydrochloride_viable(self):
        """Base drug pKa=8 + hydrochloride (pKa=-7) → delta=15 → viable."""
        result = predict_salt_solubility("BaseDrug", 8.0, "base", "hydrochloride", 0.5)
        assert result.salt_viable is True
        assert result.counterion_type == "anion"

    def test_base_mesylate_viable(self):
        result = predict_salt_solubility("BaseDrug", 8.0, "base", "mesylate", 0.3)
        assert result.salt_viable is True
        assert result.counterion_name == "mesylate"

    def test_regulatory_preferred(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        assert result.regulatory_status == "preferred"

    def test_regulatory_acceptable(self):
        result = predict_salt_solubility("BaseDrug", 7.0, "base", "tosylate", 1.0)
        assert result.regulatory_status == "acceptable"

    def test_hygroscopicity_calcium_high(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "calcium", 1.0)
        assert result.hygroscopicity_risk == "high"

    def test_hygroscopicity_meglumine_moderate(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "meglumine", 1.0)
        assert result.hygroscopicity_risk == "moderate"

    def test_hygroscopicity_sodium_low(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        assert result.hygroscopicity_risk == "low"

    def test_hygroscopicity_lysine_moderate(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "lysine", 1.0)
        assert result.hygroscopicity_risk == "moderate"

    def test_result_is_frozen_dataclass(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "modified"  # type: ignore[misc]

    def test_ionization_type_stored(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        assert result.ionization_type == "acid"

    def test_pka_stored(self):
        result = predict_salt_solubility("Drug", 4.0, "acid", "sodium", 1.0)
        assert result.pka_drug == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# screen_salt_forms
# ---------------------------------------------------------------------------


class TestScreenSaltForms:
    def test_neutral_drug_returns_empty(self):
        results = screen_salt_forms("Neutral", 7.0, "neutral", 1.0, 2.0)
        assert results == []

    def test_acid_drug_returns_only_cation_counterions(self):
        results = screen_salt_forms("AcidDrug", 4.0, "acid", 1.0, 2.0)
        assert len(results) > 0
        for r in results:
            assert r.counterion_type == "cation"

    def test_base_drug_returns_only_anion_counterions(self):
        results = screen_salt_forms("BaseDrug", 8.0, "base", 0.5, 1.0)
        assert len(results) > 0
        for r in results:
            assert r.counterion_type == "anion"

    def test_sorted_descending_by_solubility_enhancement(self):
        results = screen_salt_forms("AcidDrug", 4.0, "acid", 1.0, 2.0)
        enhancements = [r.solubility_enhancement for r in results]
        assert enhancements == sorted(enhancements, reverse=True)

    def test_acid_counterion_count(self):
        """Five cation counterions defined."""
        results = screen_salt_forms("AcidDrug", 4.0, "acid", 1.0, 2.0)
        assert len(results) == 5

    def test_base_counterion_count(self):
        """Five anion counterions defined."""
        results = screen_salt_forms("BaseDrug", 8.0, "base", 0.5, 1.0)
        assert len(results) == 5

    def test_higher_delta_pka_higher_enhancement(self):
        """Acid drug with lower pKa has higher delta_pKa vs sodium → higher enhancement."""
        results_low = screen_salt_forms("Drug1", 2.0, "acid", 1.0, 1.0)
        results_high = screen_salt_forms("Drug2", 10.0, "acid", 1.0, 1.0)

        # For sodium (pKa=14): delta_pKa(2.0)=12 vs delta_pKa(10.0)=4
        def get_sodium(results):
            return next(r for r in results if r.counterion_name == "sodium")

        r_low = get_sodium(results_low)
        r_high = get_sodium(results_high)
        assert r_low.solubility_enhancement > r_high.solubility_enhancement

    def test_all_results_are_salt_form_result(self):
        results = screen_salt_forms("AcidDrug", 4.0, "acid", 1.0, 2.0)
        for r in results:
            assert isinstance(r, SaltFormResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_pka_below_zero_raises(self):
        with pytest.raises(ValueError, match="pka"):
            predict_salt_solubility("Drug", -1.0, "acid", "sodium", 1.0)

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pka"):
            predict_salt_solubility("Drug", 15.0, "acid", "sodium", 1.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_salt_solubility("Drug", 4.0, "zwitterion", "sodium", 1.0)

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            predict_salt_solubility("Drug", 4.0, "acid", "sodium", -1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            predict_salt_solubility("Drug", 4.0, "acid", "sodium", 0.0)

    def test_unknown_counterion_raises(self):
        with pytest.raises(ValueError, match="counterion_name"):
            predict_salt_solubility("Drug", 4.0, "acid", "unicorn", 1.0)

    def test_screen_logP_out_of_range_raises(self):
        with pytest.raises(ValueError, match="logP"):
            screen_salt_forms("Drug", 4.0, "acid", 1.0, 15.0)

    def test_screen_pka_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pka"):
            screen_salt_forms("Drug", -0.5, "acid", 1.0, 2.0)
