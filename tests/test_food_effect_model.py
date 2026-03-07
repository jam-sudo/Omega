"""Tests for Phase 568 — Food Effect Model."""

import pytest

from omega_pbpk.clinical.food_effect_model import (
    fasted_fed_recommendation,
    food_effect_on_absorption,
    food_effect_on_solubility,
    high_fat_meal_effect,
)

# ---------------------------------------------------------------------------
# food_effect_on_solubility
# ---------------------------------------------------------------------------


class TestFoodEffectOnSolubility:
    def test_lipophilic_fed_higher_than_acid(self):
        lip = food_effect_on_solubility("lipophilic", logP=5.0, food_state="fed")
        acid = food_effect_on_solubility("acid", logP=5.0, food_state="fed")
        assert lip["solubility_fold_change"] > acid["solubility_fold_change"]

    def test_fasted_fold_is_1(self):
        r = food_effect_on_solubility("lipophilic", logP=4.0, food_state="fasted")
        assert r["solubility_fold_change"] == pytest.approx(1.0)

    def test_lipophilic_high_logP_capped_at_8(self):
        r = food_effect_on_solubility("lipophilic", logP=20.0, food_state="fed")
        assert r["solubility_fold_change"] == pytest.approx(8.0)

    def test_acid_fed_below_1(self):
        r = food_effect_on_solubility("acid", logP=1.0, food_state="fed")
        assert r["solubility_fold_change"] < 1.0

    def test_base_fed_above_1(self):
        r = food_effect_on_solubility("base", logP=1.0, food_state="fed")
        assert r["solubility_fold_change"] > 1.0

    def test_neutral_fed_fold_1_2(self):
        r = food_effect_on_solubility("neutral", logP=1.0, food_state="fed")
        assert r["solubility_fold_change"] == pytest.approx(1.2)

    def test_returns_all_keys(self):
        r = food_effect_on_solubility("base", logP=2.0, food_state="fed")
        for k in ("drug_type", "logP", "food_state", "solubility_fold_change", "notes"):
            assert k in r

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_solubility("unknown", logP=2.0)

    def test_invalid_food_state_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_solubility("acid", logP=2.0, food_state="breakfast")

    def test_lipophilic_logP_scaling(self):
        r3 = food_effect_on_solubility("lipophilic", logP=3.0, food_state="fed")
        r5 = food_effect_on_solubility("lipophilic", logP=5.0, food_state="fed")
        assert r5["solubility_fold_change"] > r3["solubility_fold_change"]


# ---------------------------------------------------------------------------
# food_effect_on_absorption
# ---------------------------------------------------------------------------


class TestFoodEffectOnAbsorption:
    def test_fa_fed_higher_lipophilic(self):
        r = food_effect_on_absorption(
            fa_fasted=0.3,
            cmax_fasted=1.0,
            auc_fasted=10.0,
            drug_type="lipophilic",
            logP=4.0,
        )
        assert r["fa_fed"] > 0.3

    def test_fa_fed_capped_at_1(self):
        r = food_effect_on_absorption(
            fa_fasted=0.99,
            cmax_fasted=1.0,
            auc_fasted=10.0,
            drug_type="lipophilic",
            logP=8.0,
        )
        assert r["fa_fed"] <= 1.0

    def test_acid_auc_fed_below_fasted(self):
        r = food_effect_on_absorption(
            fa_fasted=0.8,
            cmax_fasted=2.0,
            auc_fasted=20.0,
            drug_type="acid",
            logP=1.0,
        )
        assert r["auc_fed"] < 20.0

    def test_base_auc_fed_above_fasted(self):
        r = food_effect_on_absorption(
            fa_fasted=0.5,
            cmax_fasted=1.0,
            auc_fasted=10.0,
            drug_type="base",
            logP=2.0,
        )
        assert r["auc_fed"] > 10.0

    def test_returns_all_keys(self):
        r = food_effect_on_absorption(
            fa_fasted=0.5,
            cmax_fasted=1.0,
            auc_fasted=10.0,
            drug_type="neutral",
            logP=1.0,
        )
        for k in (
            "fa_fed",
            "cmax_fed",
            "auc_fed",
            "food_effect_ratio_auc",
            "clinical_significance",
        ):
            assert k in r

    def test_invalid_fa_raises(self):
        with pytest.raises(ValueError):
            food_effect_on_absorption(
                fa_fasted=1.5,
                cmax_fasted=1.0,
                auc_fasted=10.0,
                drug_type="base",
                logP=2.0,
            )

    def test_food_effect_ratio_auc_for_major(self):
        r = food_effect_on_absorption(
            fa_fasted=0.3,
            cmax_fasted=1.0,
            auc_fasted=10.0,
            drug_type="lipophilic",
            logP=8.0,
        )
        assert r["clinical_significance"] == "major"


# ---------------------------------------------------------------------------
# high_fat_meal_effect
# ---------------------------------------------------------------------------


class TestHighFatMealEffect:
    def test_high_logP_auc_ratio_higher(self):
        r5 = high_fat_meal_effect(dose_mg=100.0, logP=5.0)
        r1 = high_fat_meal_effect(dose_mg=100.0, logP=1.0)
        assert r5["auc_ratio"] > r1["auc_ratio"]

    def test_tmax_delay_is_1(self):
        r = high_fat_meal_effect(dose_mg=100.0, logP=3.0)
        assert r["tmax_delay_h"] == pytest.approx(1.0)

    def test_auc_ratio_capped_at_5(self):
        r = high_fat_meal_effect(dose_mg=100.0, logP=100.0)
        assert r["auc_ratio"] == pytest.approx(5.0)

    def test_negative_logP_auc_ratio_is_1(self):
        r = high_fat_meal_effect(dose_mg=100.0, logP=-2.0)
        assert r["auc_ratio"] == pytest.approx(1.0)

    def test_clinical_significance_major(self):
        r = high_fat_meal_effect(dose_mg=100.0, logP=5.0)
        assert r["clinical_significance"] == "major"

    def test_clinical_significance_none_low_logP(self):
        r = high_fat_meal_effect(dose_mg=100.0, logP=0.5)
        assert r["clinical_significance"] == "none"

    def test_clinical_significance_moderate(self):
        # logP=2.5 → ratio = 1 + 2*(2.5/5) = 2.0, border → major
        # logP=1.875 → ratio = 1 + 2*(1.875/5) = 1.75, moderate
        r = high_fat_meal_effect(dose_mg=100.0, logP=1.875)
        assert r["clinical_significance"] == "moderate"

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            high_fat_meal_effect(dose_mg=0.0, logP=3.0)

    def test_returns_all_keys(self):
        r = high_fat_meal_effect(dose_mg=500.0, logP=2.0)
        for k in ("auc_ratio", "tmax_delay_h", "clinical_significance"):
            assert k in r


# ---------------------------------------------------------------------------
# fasted_fed_recommendation
# ---------------------------------------------------------------------------


class TestFastedFedRecommendation:
    def test_high_logP_low_solubility_take_with_food(self):
        s = fasted_fed_recommendation("DrugA", logP=4.0, solubility_mg_mL=0.05)
        assert s == "Take with food to improve absorption"

    def test_low_logP_high_solubility_empty_stomach(self):
        s = fasted_fed_recommendation("DrugB", logP=-1.0, solubility_mg_mL=5.0)
        assert s == "Take on empty stomach for fastest absorption"

    def test_neutral_recommendation(self):
        s = fasted_fed_recommendation("DrugC", logP=2.0, solubility_mg_mL=1.0)
        assert s == "May be taken with or without food"

    def test_high_logP_high_solubility_neutral(self):
        # logP > 3 but solubility >= 0.1 → neutral
        s = fasted_fed_recommendation("DrugD", logP=4.0, solubility_mg_mL=0.5)
        assert s == "May be taken with or without food"

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError):
            fasted_fed_recommendation("DrugE", logP=2.0, solubility_mg_mL=-1.0)

    def test_boundary_logP_3_solubility_below_0_1(self):
        # logP exactly 3 is not > 3, so should not trigger "take with food"
        s = fasted_fed_recommendation("DrugF", logP=3.0, solubility_mg_mL=0.05)
        assert s == "May be taken with or without food"

    def test_boundary_logP_slightly_above_3(self):
        s = fasted_fed_recommendation("DrugG", logP=3.1, solubility_mg_mL=0.05)
        assert s == "Take with food to improve absorption"
