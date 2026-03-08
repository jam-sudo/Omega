"""Tests for Phase 471 — Drug Interaction with Food Components."""

import pytest

from omega_pbpk.prediction.drug_food_interaction import (
    DrugFoodInteractionResult,
    predict_drug_food_interaction,
    screen_food_interactions,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert isinstance(result, DrugFoodInteractionResult)

    def test_fields_present(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "food_item")
        assert hasattr(result, "drug_class")
        assert hasattr(result, "interaction_detected")
        assert hasattr(result, "auc_ratio")
        assert hasattr(result, "fa_ratio")
        assert hasattr(result, "clinical_severity")
        assert hasattr(result, "mechanism")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "notes")

    def test_frozen_dataclass(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        with pytest.raises((AttributeError, TypeError)):
            result.auc_ratio = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Grapefruit + CYP3A4_substrate
# ---------------------------------------------------------------------------


class TestGrapefruit:
    def test_auc_ratio(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert result.auc_ratio == 3.0

    def test_severity_major(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert result.clinical_severity == "major"

    def test_recommendation_avoid(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert result.recommendation == "avoid"

    def test_interaction_detected_true(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert result.interaction_detected is True

    def test_fa_ratio_unchanged(self):
        result = predict_drug_food_interaction("simvastatin", "grapefruit", "CYP3A4_substrate")
        assert result.fa_ratio == 1.0

    def test_non_cyp3a4_no_interaction(self):
        result = predict_drug_food_interaction("metformin", "grapefruit", "biguanide")
        assert result.interaction_detected is False
        assert result.auc_ratio == 1.0
        assert result.clinical_severity == "none"


# ---------------------------------------------------------------------------
# Dairy + fluoroquinolone / tetracycline
# ---------------------------------------------------------------------------


class TestDairy:
    def test_fluoroquinolone_auc_ratio(self):
        result = predict_drug_food_interaction("ciprofloxacin", "dairy", "fluoroquinolone")
        assert result.auc_ratio == 0.5

    def test_fluoroquinolone_fa_ratio(self):
        result = predict_drug_food_interaction("ciprofloxacin", "dairy", "fluoroquinolone")
        assert result.fa_ratio == 0.5

    def test_tetracycline_auc_ratio(self):
        result = predict_drug_food_interaction("doxycycline", "dairy", "tetracycline")
        assert result.auc_ratio == 0.5

    def test_dairy_severity_moderate(self):
        result = predict_drug_food_interaction("ciprofloxacin", "dairy", "fluoroquinolone")
        assert result.clinical_severity == "moderate"

    def test_dairy_recommendation_separate(self):
        result = predict_drug_food_interaction("ciprofloxacin", "dairy", "fluoroquinolone")
        assert result.recommendation == "separate_by_2h"

    def test_dairy_interaction_detected(self):
        result = predict_drug_food_interaction("ciprofloxacin", "dairy", "fluoroquinolone")
        assert result.interaction_detected is True

    def test_dairy_non_affected_drug(self):
        result = predict_drug_food_interaction("metformin", "dairy", "biguanide")
        assert result.interaction_detected is False
        assert result.auc_ratio == 1.0
        assert result.clinical_severity == "none"


# ---------------------------------------------------------------------------
# High fiber
# ---------------------------------------------------------------------------


class TestHighFiber:
    def test_high_fiber_affects_all(self):
        result = predict_drug_food_interaction("warfarin", "high_fiber", "anticoagulant")
        assert result.interaction_detected is True
        assert result.auc_ratio == 0.8
        assert result.fa_ratio == 0.8

    def test_high_fiber_severity_minor(self):
        result = predict_drug_food_interaction("warfarin", "high_fiber", "anticoagulant")
        assert result.clinical_severity == "minor"

    def test_high_fiber_recommendation_monitor(self):
        result = predict_drug_food_interaction("warfarin", "high_fiber", "anticoagulant")
        assert result.recommendation == "monitor"


# ---------------------------------------------------------------------------
# Alcohol acute
# ---------------------------------------------------------------------------


class TestAlcoholAcute:
    def test_cyp2e1_auc_increase(self):
        result = predict_drug_food_interaction("acetaminophen", "alcohol_acute", "CYP2E1_substrate")
        assert result.auc_ratio == 2.0
        assert result.clinical_severity == "moderate"

    def test_cns_depressant_major(self):
        result = predict_drug_food_interaction("diazepam", "alcohol_acute", "CNS_depressant")
        assert result.clinical_severity == "major"
        assert result.auc_ratio == 1.0
        assert result.interaction_detected is True
        assert result.recommendation == "avoid"

    def test_alcohol_acute_non_affected(self):
        result = predict_drug_food_interaction("lisinopril", "alcohol_acute", "ACE_inhibitor")
        assert result.interaction_detected is False


# ---------------------------------------------------------------------------
# Alcohol chronic
# ---------------------------------------------------------------------------


class TestAlcoholChronic:
    def test_cyp2e1_induction_decreases_auc(self):
        result = predict_drug_food_interaction(
            "acetaminophen", "alcohol_chronic", "CYP2E1_substrate"
        )
        assert result.auc_ratio == 0.5
        assert result.clinical_severity == "moderate"
        assert result.recommendation == "monitor"

    def test_alcohol_chronic_non_affected(self):
        result = predict_drug_food_interaction(
            "atorvastatin", "alcohol_chronic", "CYP3A4_substrate"
        )
        assert result.interaction_detected is False
        assert result.auc_ratio == 1.0


# ---------------------------------------------------------------------------
# Tyramine foods + MAOI
# ---------------------------------------------------------------------------


class TestTyramine:
    def test_maoi_contraindicated(self):
        result = predict_drug_food_interaction("phenelzine", "tyramine_foods", "MAOI")
        assert result.clinical_severity == "contraindicated"
        assert result.recommendation == "avoid"
        assert result.interaction_detected is True
        assert result.auc_ratio == 1.0

    def test_non_maoi_no_interaction(self):
        result = predict_drug_food_interaction("citalopram", "tyramine_foods", "SSRI")
        assert result.interaction_detected is False
        assert result.clinical_severity == "none"


# ---------------------------------------------------------------------------
# No interaction
# ---------------------------------------------------------------------------


class TestNoInteraction:
    def test_no_interaction_severity_none(self):
        result = predict_drug_food_interaction("metformin", "grapefruit", "biguanide")
        assert result.clinical_severity == "none"
        assert result.interaction_detected is False

    def test_no_interaction_ratios_one(self):
        result = predict_drug_food_interaction("metformin", "grapefruit", "biguanide")
        assert result.auc_ratio == 1.0
        assert result.fa_ratio == 1.0

    def test_no_interaction_recommendation(self):
        result = predict_drug_food_interaction("metformin", "grapefruit", "biguanide")
        assert result.recommendation == "no_action"


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenFoodInteractions:
    def test_returns_list(self):
        results = screen_food_interactions("simvastatin", "CYP3A4_substrate")
        assert isinstance(results, list)

    def test_screens_all_six_items(self):
        results = screen_food_interactions("simvastatin", "CYP3A4_substrate")
        assert len(results) == 6

    def test_sorted_by_auc_ratio_descending(self):
        results = screen_food_interactions("simvastatin", "CYP3A4_substrate")
        ratios = [r.auc_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_grapefruit_first_for_cyp3a4(self):
        results = screen_food_interactions("simvastatin", "CYP3A4_substrate")
        assert results[0].food_item == "grapefruit"
        assert results[0].auc_ratio == 3.0

    def test_custom_food_items(self):
        results = screen_food_interactions(
            "ciprofloxacin", "fluoroquinolone", food_items=["grapefruit", "dairy"]
        )
        assert len(results) == 2

    def test_custom_food_items_sorted(self):
        results = screen_food_interactions(
            "acetaminophen",
            "CYP2E1_substrate",
            food_items=["alcohol_acute", "alcohol_chronic", "high_fiber"],
        )
        ratios = [r.auc_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_results_are_correct_type(self):
        results = screen_food_interactions("warfarin", "anticoagulant")
        for r in results:
            assert isinstance(r, DrugFoodInteractionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_food_item_raises(self):
        with pytest.raises(ValueError, match="Invalid food_item"):
            predict_drug_food_interaction("simvastatin", "pizza", "CYP3A4_substrate")

    def test_empty_drug_class_raises(self):
        with pytest.raises(ValueError, match="drug_class"):
            predict_drug_food_interaction("simvastatin", "grapefruit", "")

    def test_whitespace_drug_class_raises(self):
        with pytest.raises(ValueError, match="drug_class"):
            predict_drug_food_interaction("simvastatin", "grapefruit", "   ")

    def test_screen_invalid_food_item_raises(self):
        with pytest.raises(ValueError):
            screen_food_interactions("drug", "class", food_items=["invalid_item"])
