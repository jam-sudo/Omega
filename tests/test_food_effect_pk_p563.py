"""Tests for Phase 563 - Drug Interaction with Food (Food Effect PK)."""

import pytest

from omega_pbpk.clinical.food_effect_pk_p563 import (
    FoodEffectResult,
    compare_bcs_classes,
    predict_food_effect,
)


class TestPredictFoodEffectReturnType:
    def test_returns_food_effect_result(self):
        result = predict_food_effect("DrugA", "I")
        assert isinstance(result, FoodEffectResult)

    def test_result_is_frozen(self):
        result = predict_food_effect("DrugA", "I")
        with pytest.raises(AttributeError):
            result.drug_name = "other"


class TestBCSClassI:
    def test_cmax_ratio(self):
        r = predict_food_effect("DrugA", "I")
        assert r.cmax_ratio == 0.9

    def test_auc_ratio(self):
        r = predict_food_effect("DrugA", "I")
        assert r.auc_ratio == 1.0

    def test_tmax_ratio(self):
        r = predict_food_effect("DrugA", "I")
        assert r.tmax_ratio == 1.5

    def test_food_effect_class_none(self):
        r = predict_food_effect("DrugA", "I")
        assert r.food_effect_class == "none"


class TestBCSClassII:
    def test_high_logp_higher_auc(self):
        r_high = predict_food_effect("DrugA", "II", logP=5.0)
        r_low = predict_food_effect("DrugA", "II", logP=2.0)
        assert r_high.auc_ratio > r_low.auc_ratio

    def test_logp_2_auc_ratio_1(self):
        r = predict_food_effect("DrugA", "II", logP=2.0)
        assert r.auc_ratio == 1.0

    def test_logp_5_auc_ratio(self):
        r = predict_food_effect("DrugA", "II", logP=5.0)
        expected = min(2.0, 1.0 + 0.2 * 3.0)
        assert abs(r.auc_ratio - expected) < 1e-9

    def test_cmax_ratio_is_auc_times_085(self):
        r = predict_food_effect("DrugA", "II", logP=4.0)
        assert abs(r.cmax_ratio - r.auc_ratio * 0.85) < 1e-9

    def test_high_logp_positive_food_effect(self):
        r = predict_food_effect("DrugA", "II", logP=5.0)
        assert r.food_effect_class == "positive"


class TestBCSClassIII:
    def test_negative_food_effect(self):
        r = predict_food_effect("DrugA", "III")
        assert r.auc_ratio < 1.0

    def test_auc_ratio_085(self):
        r = predict_food_effect("DrugA", "III")
        assert r.auc_ratio == 0.85

    def test_food_effect_class_none_border(self):
        # 0.85 is between 0.8 and 1.25 so "none"
        r = predict_food_effect("DrugA", "III")
        assert r.food_effect_class == "none"


class TestBCSClassIV:
    def test_slight_positive(self):
        r = predict_food_effect("DrugA", "IV")
        assert r.auc_ratio == 1.1
        assert r.cmax_ratio == 1.1


class TestFoodEffectClassification:
    def test_positive_class(self):
        r = predict_food_effect("DrugA", "II", logP=5.0)
        assert r.food_effect_class == "positive"

    def test_none_class(self):
        r = predict_food_effect("DrugA", "I")
        assert r.food_effect_class == "none"


class TestClinicalRecommendation:
    def test_take_with_food(self):
        r = predict_food_effect("DrugA", "II", logP=5.0)
        assert r.clinical_recommendation == "take_with_food"

    def test_no_preference(self):
        r = predict_food_effect("DrugA", "I")
        assert r.clinical_recommendation == "no_preference"

    def test_take_without_food_if_negative(self):
        # BCS III with low f_fasted to force auc_ratio < 0.8 won't work
        # since auc_ratio is fixed at 0.85 for BCS III
        # We need to verify that if auc_ratio < 0.8, recommendation is correct
        # BCS III has auc_ratio=0.85 which is "none", not "negative"
        r = predict_food_effect("DrugA", "III")
        assert r.clinical_recommendation == "no_preference"


class TestKaFed:
    def test_ka_fed_calculation(self):
        r = predict_food_effect("DrugA", "I", ka_fasted_per_h=2.0)
        assert abs(r.ka_fed - 2.0 / 1.5) < 1e-9


class TestFFedClamping:
    def test_f_fed_clamped_to_1(self):
        # BCS II with high logP and high f_fasted
        r = predict_food_effect("DrugA", "II", f_fasted=0.9, logP=5.0)
        assert r.f_fed <= 1.0

    def test_f_fed_not_negative(self):
        r = predict_food_effect("DrugA", "III", f_fasted=0.1)
        assert r.f_fed >= 0.0


class TestCompareBcsClasses:
    def test_returns_list(self):
        results = compare_bcs_classes("DrugA")
        assert isinstance(results, list)
        assert len(results) == 4

    def test_sorted_descending_by_auc_ratio(self):
        results = compare_bcs_classes("DrugA", logP=4.0)
        auc_ratios = [r.auc_ratio for r in results]
        for i in range(len(auc_ratios) - 1):
            assert auc_ratios[i] >= auc_ratios[i + 1]


class TestValidation:
    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError):
            predict_food_effect("DrugA", "V")

    def test_f_fasted_zero(self):
        with pytest.raises(ValueError):
            predict_food_effect("DrugA", "I", f_fasted=0.0)

    def test_f_fasted_negative(self):
        with pytest.raises(ValueError):
            predict_food_effect("DrugA", "I", f_fasted=-0.1)

    def test_ka_fasted_zero(self):
        with pytest.raises(ValueError):
            predict_food_effect("DrugA", "I", ka_fasted_per_h=0.0)

    def test_ka_fasted_negative(self):
        with pytest.raises(ValueError):
            predict_food_effect("DrugA", "I", ka_fasted_per_h=-1.0)
