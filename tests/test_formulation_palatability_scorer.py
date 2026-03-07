"""Tests for formulation palatability scorer (Phase 273)."""

import pytest

from omega_pbpk.prediction.formulation_palatability_scorer import (
    PalatabilityResult,
    optimize_palatability,
    score_palatability,
)


class TestReturnType:
    def test_returns_palatability_result(self):
        result = score_palatability("DrugA", "bitter")
        assert isinstance(result, PalatabilityResult)

    def test_drug_name_preserved(self):
        result = score_palatability("MyDrug", "sweet")
        assert result.drug_name == "MyDrug"

    def test_taste_category_preserved(self):
        result = score_palatability("DrugA", "sour")
        assert result.taste_category == "sour"

    def test_masking_strategy_preserved(self):
        result = score_palatability("DrugA", "bitter", masking_strategy="coating")
        assert result.masking_strategy == "coating"

    def test_formulation_form_preserved(self):
        result = score_palatability("DrugA", "bitter", formulation_form="chewable")
        assert result.formulation_form == "chewable"


class TestBaseScores:
    def test_bitter_base_score(self):
        result = score_palatability("DrugA", "bitter", masking_strategy="none")
        assert result.base_palatability_score == pytest.approx(20.0)

    def test_sweet_base_score(self):
        result = score_palatability("DrugA", "sweet", masking_strategy="none")
        assert result.base_palatability_score == pytest.approx(90.0)

    def test_sour_base_score(self):
        result = score_palatability("DrugA", "sour", masking_strategy="none")
        assert result.base_palatability_score == pytest.approx(40.0)

    def test_salty_base_score(self):
        result = score_palatability("DrugA", "salty", masking_strategy="none")
        assert result.base_palatability_score == pytest.approx(60.0)

    def test_tasteless_base_score(self):
        result = score_palatability("DrugA", "tasteless", masking_strategy="none")
        assert result.base_palatability_score == pytest.approx(95.0)


class TestMaskedScore:
    def test_no_masking_keeps_base_score(self):
        result = score_palatability("DrugA", "bitter", masking_strategy="none")
        assert result.masked_palatability_score == pytest.approx(result.base_palatability_score)

    def test_masking_improves_score(self):
        base_result = score_palatability("DrugA", "bitter", masking_strategy="none")
        masked_result = score_palatability("DrugA", "bitter", masking_strategy="coating")
        assert masked_result.masked_palatability_score >= base_result.masked_palatability_score

    def test_bitter_coating_score(self):
        # bitter(20) + coating(0.3): masked = 20 + 80 * 0.7 = 76
        result = score_palatability("DrugA", "bitter", masking_strategy="coating")
        assert result.masked_palatability_score == pytest.approx(76.0)

    def test_coating_better_than_none_for_bitter(self):
        none_result = score_palatability("DrugA", "bitter", masking_strategy="none")
        coating_result = score_palatability("DrugA", "bitter", masking_strategy="coating")
        assert coating_result.masked_palatability_score > none_result.masked_palatability_score

    def test_sweet_has_higher_score_than_bitter(self):
        sweet_result = score_palatability("DrugA", "sweet")
        bitter_result = score_palatability("DrugA", "bitter")
        assert sweet_result.masked_palatability_score > bitter_result.masked_palatability_score


class TestPatientAcceptance:
    def test_bitter_no_masking_low_acceptance(self):
        result = score_palatability("DrugA", "bitter", masking_strategy="none")
        # base_score=20 < 40 → 20%
        assert result.patient_acceptance_pct == pytest.approx(20.0)

    def test_sweet_high_acceptance(self):
        result = score_palatability("DrugA", "sweet")
        # base_score=90 > 70 → 90%
        assert result.patient_acceptance_pct == pytest.approx(90.0)

    def test_acceptance_valid_values(self):
        for taste in ("bitter", "sweet", "sour", "salty", "tasteless"):
            result = score_palatability("DrugA", taste)
            assert result.patient_acceptance_pct in (20.0, 60.0, 90.0)


class TestBitternessIndex:
    def test_sweet_bitterness_zero(self):
        result = score_palatability("DrugA", "sweet")
        assert result.bitterness_index == pytest.approx(0.0)

    def test_tasteless_bitterness_zero(self):
        result = score_palatability("DrugA", "tasteless")
        assert result.bitterness_index == pytest.approx(0.0)

    def test_bitter_bitterness_index(self):
        result = score_palatability("DrugA", "bitter")
        # 100 - 20 = 80
        assert result.bitterness_index == pytest.approx(80.0)


class TestRecommendations:
    def test_recommended_for_children_when_high_score(self):
        result = score_palatability("DrugA", "sweet")
        assert result.recommended_for_children is True

    def test_not_recommended_for_children_when_bitter_no_masking(self):
        result = score_palatability("DrugA", "bitter", masking_strategy="none")
        assert result.recommended_for_children is False

    def test_recommended_for_elderly_threshold(self):
        # bitter + coating = 76 >= 60 → elderly True
        result = score_palatability("DrugA", "bitter", masking_strategy="coating")
        assert result.recommended_for_elderly is True


class TestFeasibility:
    def test_sweet_is_feasible(self):
        result = score_palatability("DrugA", "sweet")
        assert result.formulation_feasibility == "feasible"

    def test_bitter_no_masking_feasibility(self):
        # base_score=20 (no masking) < 30 → not_recommended
        result = score_palatability("DrugA", "bitter", masking_strategy="none")
        assert result.formulation_feasibility == "not_recommended"

    def test_feasibility_valid_values(self):
        for taste in ("bitter", "sweet", "sour", "salty", "tasteless"):
            for strategy in ("none", "sweetener", "flavor", "cyclodextrin", "coating"):
                result = score_palatability("DrugA", taste, masking_strategy=strategy)
                assert result.formulation_feasibility in (
                    "feasible",
                    "challenging",
                    "not_recommended",
                )


class TestOptimizePalatability:
    def test_returns_list(self):
        results = optimize_palatability("DrugA", "bitter")
        assert isinstance(results, list)

    def test_returns_five_strategies(self):
        results = optimize_palatability("DrugA", "bitter")
        assert len(results) == 5

    def test_sorted_descending(self):
        results = optimize_palatability("DrugA", "bitter")
        scores = [r.masked_palatability_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_correct_type(self):
        results = optimize_palatability("DrugA", "sour")
        for r in results:
            assert isinstance(r, PalatabilityResult)

    def test_best_masking_for_bitter(self):
        results = optimize_palatability("DrugA", "bitter")
        # coating (factor=0.3) gives highest score for bitter
        assert results[0].masking_strategy == "coating"


class TestValidation:
    def test_invalid_taste_raises(self):
        with pytest.raises(ValueError, match="taste_category"):
            score_palatability("DrugA", "umami")

    def test_invalid_masking_raises(self):
        with pytest.raises(ValueError, match="masking_strategy"):
            score_palatability("DrugA", "bitter", masking_strategy="magic")

    def test_invalid_formulation_form_raises(self):
        with pytest.raises(ValueError, match="formulation_form"):
            score_palatability("DrugA", "bitter", formulation_form="injection")
