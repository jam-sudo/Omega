"""Tests for Phase 314 — Pharmacokinetic food interaction simulator."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.food_interaction_simulator import (
    FoodInteractionResult,
    MealComposition,
    classify_food_effect,
    food_effect_on_pk,
    meal_composition,
)

# ---------------------------------------------------------------------------
# meal_composition tests
# ---------------------------------------------------------------------------


class TestMealComposition:
    def test_returns_dataclass(self):
        result = meal_composition("fasted")
        assert isinstance(result, MealComposition)

    def test_frozen(self):
        result = meal_composition("fasted")
        with pytest.raises((AttributeError, TypeError)):
            result.fat_g = 99.0  # type: ignore[misc]

    def test_fasted_zero_macros(self):
        m = meal_composition("fasted")
        assert m.fat_g == 0.0
        assert m.protein_g == 0.0
        assert m.carb_g == 0.0
        assert m.total_kcal == 0.0

    def test_fasted_ge_t50(self):
        m = meal_composition("fasted")
        assert m.gastric_emptying_t50_h == pytest.approx(0.5)

    def test_fasted_bile_low(self):
        m = meal_composition("fasted")
        assert m.bile_conc_mM == pytest.approx(1.0)

    def test_high_fat_fda_macros(self):
        m = meal_composition("high_fat_fda")
        assert m.fat_g == pytest.approx(50.0)
        assert m.protein_g == pytest.approx(25.0)
        assert m.carb_g == pytest.approx(58.0)
        assert m.total_kcal == pytest.approx(1000.0)

    def test_high_fat_fda_ge_t50(self):
        m = meal_composition("high_fat_fda")
        assert m.gastric_emptying_t50_h == pytest.approx(3.0)

    def test_high_fat_fda_bile(self):
        m = meal_composition("high_fat_fda")
        assert m.bile_conc_mM == pytest.approx(8.0)

    def test_standard_fda_macros(self):
        m = meal_composition("standard_fda")
        assert m.fat_g == pytest.approx(20.0)
        assert m.carb_g == pytest.approx(40.0)
        assert m.total_kcal == pytest.approx(400.0)

    def test_low_fat_macros(self):
        m = meal_composition("low_fat")
        assert m.fat_g == pytest.approx(5.0)
        assert m.total_kcal == pytest.approx(300.0)

    def test_ge_t50_ordering(self):
        """High fat > standard > low fat > fasted."""
        t_fasted = meal_composition("fasted").gastric_emptying_t50_h
        t_low = meal_composition("low_fat").gastric_emptying_t50_h
        t_std = meal_composition("standard_fda").gastric_emptying_t50_h
        t_high = meal_composition("high_fat_fda").gastric_emptying_t50_h
        assert t_fasted < t_low < t_std < t_high

    def test_unknown_meal_raises(self):
        with pytest.raises(ValueError, match="Unknown meal_type"):
            meal_composition("ultra_fatty")

    def test_meal_type_stored(self):
        m = meal_composition("low_fat")
        assert m.meal_type == "low_fat"


# ---------------------------------------------------------------------------
# classify_food_effect tests
# ---------------------------------------------------------------------------


class TestClassifyFoodEffect:
    def test_no_effect(self):
        assert classify_food_effect(1.0, 1.0) == "none"

    def test_no_effect_boundary(self):
        assert classify_food_effect(0.80, 0.80) == "none"

    def test_no_effect_upper_boundary(self):
        assert classify_food_effect(1.25, 1.25) == "none"

    def test_strong_auc_very_low(self):
        assert classify_food_effect(0.3, 1.0) == "strong"

    def test_strong_auc_very_high(self):
        assert classify_food_effect(3.0, 1.5) == "strong"

    def test_moderate_auc_low(self):
        result = classify_food_effect(0.65, 1.0)
        assert result == "moderate"

    def test_moderate_auc_high(self):
        result = classify_food_effect(1.7, 1.2)
        assert result == "moderate"

    def test_weak_slightly_outside(self):
        result = classify_food_effect(1.3, 1.0)
        assert result in ("none", "weak")


# ---------------------------------------------------------------------------
# food_effect_on_pk tests
# ---------------------------------------------------------------------------


class TestFoodEffectOnPk:
    """Integration tests for food_effect_on_pk."""

    def _basic_result(self, meal_type="high_fat_fda", bcs_class=2):
        return food_effect_on_pk(
            drug_name="test_drug",
            dose_mg=100.0,
            bcs_class=bcs_class,
            logP=3.0,
            pka=7.0,
            solubility_water_mg_mL=0.01,
            cl_L_per_h=5.0,
            vd_L=50.0,
            meal_type=meal_type,
            papp_cm_s=1e-6,
        )

    def test_returns_result_type(self):
        result = self._basic_result()
        assert isinstance(result, FoodInteractionResult)

    def test_frozen_dataclass(self):
        result = self._basic_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_auc_ratio_positive(self):
        result = self._basic_result()
        assert result.auc_ratio > 0.0

    def test_cmax_ratio_positive(self):
        result = self._basic_result()
        assert result.cmax_ratio > 0.0

    def test_tmax_values_positive(self):
        result = self._basic_result()
        assert result.tmax_fed_h >= 0.0
        assert result.tmax_fasted_h >= 0.0

    def test_bcs2_food_increases_auc(self):
        """BCS II drugs typically benefit from food (higher AUC)."""
        result = self._basic_result(bcs_class=2)
        assert result.auc_ratio > 1.0

    def test_bcs1_food_effect_mild(self):
        """BCS I: food effect should be weak or none."""
        result = food_effect_on_pk(
            drug_name="bcs1_drug",
            dose_mg=100.0,
            bcs_class=1,
            logP=1.0,
            pka=7.0,
            solubility_water_mg_mL=10.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            meal_type="high_fat_fda",
            papp_cm_s=20e-6,
        )
        assert result.food_effect_class in ("none", "weak", "moderate")

    def test_tmax_fed_later_than_fasted(self):
        """Fed state should have delayed Tmax due to slower gastric emptying."""
        result = self._basic_result(bcs_class=2)
        assert result.tmax_fed_h >= result.tmax_fasted_h

    def test_food_effect_class_string(self):
        result = self._basic_result()
        assert result.food_effect_class in ("none", "weak", "moderate", "strong")

    def test_concentration_arrays_same_length(self):
        result = self._basic_result()
        assert len(result.times_h) == len(result.c_plasma_fed)
        assert len(result.times_h) == len(result.c_plasma_fasted)

    def test_concentrations_non_negative(self):
        result = self._basic_result()
        assert all(c >= 0.0 for c in result.c_plasma_fed)
        assert all(c >= 0.0 for c in result.c_plasma_fasted)

    def test_drug_name_stored(self):
        result = food_effect_on_pk(
            drug_name="ketoconazole",
            dose_mg=200.0,
            bcs_class=2,
            logP=4.3,
            pka=6.5,
            solubility_water_mg_mL=0.004,
            cl_L_per_h=3.0,
            vd_L=200.0,
        )
        assert result.drug_name == "ketoconazole"

    def test_meal_type_stored(self):
        result = self._basic_result(meal_type="standard_fda")
        assert result.meal_type == "standard_fda"

    def test_fasted_meal_no_change(self):
        """Fasted vs fasted should give auc_ratio ~ 1.0."""
        result = food_effect_on_pk(
            drug_name="drug",
            dose_mg=100.0,
            bcs_class=1,
            logP=1.0,
            pka=7.0,
            solubility_water_mg_mL=5.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            meal_type="fasted",
            papp_cm_s=15e-6,
        )
        assert result.auc_ratio == pytest.approx(1.0, abs=0.05)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_pk("d", -10, 1, 1.0, 7.0, 1.0, 5.0, 50.0)

    def test_invalid_bcs_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_pk("d", 100, 5, 1.0, 7.0, 1.0, 5.0, 50.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_pk("d", 100, 1, 1.0, 7.0, 1.0, 0.0, 50.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_pk("d", 100, 1, 1.0, 7.0, 1.0, 5.0, -1.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_pk("d", 100, 1, 1.0, 7.0, -1.0, 5.0, 50.0)

    def test_bcs_class_stored(self):
        result = self._basic_result(bcs_class=3)
        assert result.bcs_class == 3

    def test_notes_nonempty(self):
        result = self._basic_result()
        assert isinstance(result.notes, str)
