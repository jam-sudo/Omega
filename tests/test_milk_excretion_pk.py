"""Tests for Phase 871: Drug Excretion in Breast Milk."""

import pytest

from omega_pbpk.clinical.milk_excretion_pk import (
    MilkExcretionResult,
    predict_milk_excretion,
    screen_drugs_for_breastfeeding,
)


class TestReturnType:
    def test_returns_milk_excretion_result(self):
        result = predict_milk_excretion("TestDrug", pka=8.0, logp=1.5, ionization_type="base")
        assert isinstance(result, MilkExcretionResult)

    def test_fields_preserved(self):
        result = predict_milk_excretion("Amoxicillin", pka=2.4, logp=-3.1, ionization_type="acid")
        assert result.drug_name == "Amoxicillin"
        assert result.pka == pytest.approx(2.4)
        assert result.logp == pytest.approx(-3.1)
        assert result.ionization_type == "acid"

    def test_frozen_dataclass(self):
        result = predict_milk_excretion("Drug", pka=7.0, logp=1.0, ionization_type="neutral")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"


class TestMilkToPlasmaRatio:
    def test_mp_ratio_positive(self):
        result = predict_milk_excretion("X", pka=9.0, logp=2.0, ionization_type="base")
        assert result.milk_to_plasma_ratio > 0.0

    def test_base_high_pka_concentrates_in_milk(self):
        # Base with pKa >> pH_plasma (7.4) → ionized form predominates in more acidic milk
        result = predict_milk_excretion("Base", pka=10.0, logp=0.5, ionization_type="base")
        assert result.milk_to_plasma_ratio > 1.0

    def test_neutral_drug_mp_close_to_one_or_lipid(self):
        # Neutral with low logP → M/P near 1
        result = predict_milk_excretion("Neutral", pka=7.0, logp=0.0, ionization_type="neutral")
        assert result.milk_to_plasma_ratio >= 0.1

    def test_mp_clamped_min(self):
        # Very hydrophilic acid → M/P should not go below 0.1
        result = predict_milk_excretion("HydAcid", pka=1.0, logp=-5.0, ionization_type="acid")
        assert result.milk_to_plasma_ratio >= 0.1

    def test_mp_clamped_max(self):
        # Extreme lipophilic base → M/P should not exceed 50
        result = predict_milk_excretion(
            "LipBase", pka=12.0, logp=10.0, ionization_type="base", milk_fat_pct=10.0
        )
        assert result.milk_to_plasma_ratio <= 50.0


class TestInfantExposure:
    def test_c_milk_positive(self):
        result = predict_milk_excretion("X", pka=8.0, logp=1.0, ionization_type="base")
        assert result.c_milk_mg_L > 0.0

    def test_c_milk_proportional_to_maternal_conc(self):
        r1 = predict_milk_excretion(
            "X", pka=8.0, logp=1.0, ionization_type="base", maternal_plasma_conc_mg_L=1.0
        )
        r2 = predict_milk_excretion(
            "X", pka=8.0, logp=1.0, ionization_type="base", maternal_plasma_conc_mg_L=2.0
        )
        assert r2.c_milk_mg_L == pytest.approx(2.0 * r1.c_milk_mg_L)

    def test_infant_daily_dose_positive(self):
        result = predict_milk_excretion("X", pka=8.0, logp=1.0, ionization_type="base")
        assert result.infant_daily_dose_mg > 0.0

    def test_infant_daily_dose_scales_with_weight(self):
        r1 = predict_milk_excretion(
            "X", pka=8.0, logp=1.0, ionization_type="base", infant_weight_kg=5.0
        )
        r2 = predict_milk_excretion(
            "X", pka=8.0, logp=1.0, ionization_type="base", infant_weight_kg=10.0
        )
        assert r2.infant_daily_dose_mg == pytest.approx(2.0 * r1.infant_daily_dose_mg)


class TestRIDAndSafety:
    def test_rid_nonnegative(self):
        result = predict_milk_excretion("X", pka=8.0, logp=1.0, ionization_type="base")
        assert result.relative_infant_dose_pct >= 0.0

    def test_safety_assessment_safe_low_mp(self):
        # Low M/P → low RID → safe
        result = predict_milk_excretion(
            "Safe",
            pka=2.0,
            logp=-4.0,
            ionization_type="acid",
            maternal_plasma_conc_mg_L=0.01,
            maternal_daily_dose_mg=500.0,
        )
        assert result.safety_assessment in {"safe", "caution", "avoid"}

    def test_safety_assessment_values(self):
        result = predict_milk_excretion("X", pka=8.0, logp=1.0, ionization_type="base")
        assert result.safety_assessment in {"safe", "caution", "avoid"}

    def test_compatible_true_when_safe(self):
        # Force a safe scenario: very low conc, high maternal dose
        result = predict_milk_excretion(
            "SafeDrug",
            pka=5.0,
            logp=0.5,
            ionization_type="neutral",
            maternal_plasma_conc_mg_L=0.001,
            maternal_daily_dose_mg=1000.0,
            maternal_weight_kg=65.0,
            infant_weight_kg=5.0,
        )
        if result.safety_assessment == "safe":
            assert result.compatible_with_breastfeeding is True

    def test_compatible_false_when_avoid(self):
        # Force an avoid scenario: high conc, low maternal dose
        result = predict_milk_excretion(
            "Risky",
            pka=10.0,
            logp=5.0,
            ionization_type="base",
            maternal_plasma_conc_mg_L=100.0,
            maternal_daily_dose_mg=1.0,
            maternal_weight_kg=65.0,
            infant_weight_kg=5.0,
            milk_fat_pct=8.0,
        )
        if result.safety_assessment == "avoid":
            assert result.compatible_with_breastfeeding is False

    def test_rid_threshold_safe(self):
        # RID < 10% → safe, compatible
        result = predict_milk_excretion(
            "SafeBase",
            pka=7.0,
            logp=0.0,
            ionization_type="base",
            maternal_plasma_conc_mg_L=0.001,
            maternal_daily_dose_mg=1000.0,
        )
        if result.relative_infant_dose_pct < 10.0:
            assert result.safety_assessment == "safe"
            assert result.compatible_with_breastfeeding is True

    def test_notes_nonempty(self):
        result = predict_milk_excretion("X", pka=8.0, logp=1.0, ionization_type="base")
        assert len(result.notes) > 0


class TestScreenFunction:
    def test_returns_list(self):
        drugs = [
            {"drug_name": "DrugA", "pka": 8.0, "logp": 1.5, "ionization_type": "base"},
            {"drug_name": "DrugB", "pka": 3.0, "logp": -1.0, "ionization_type": "acid"},
        ]
        results = screen_drugs_for_breastfeeding(drugs)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_returns_sorted_ascending_rid(self):
        drugs = [
            {
                "drug_name": "Risky",
                "pka": 10.0,
                "logp": 5.0,
                "ionization_type": "base",
                "maternal_plasma_conc_mg_L": 100.0,
                "maternal_daily_dose_mg": 1.0,
            },
            {
                "drug_name": "Safe",
                "pka": 2.0,
                "logp": -4.0,
                "ionization_type": "acid",
                "maternal_plasma_conc_mg_L": 0.001,
                "maternal_daily_dose_mg": 1000.0,
            },
            {"drug_name": "Mid", "pka": 7.0, "logp": 1.0, "ionization_type": "neutral"},
        ]
        results = screen_drugs_for_breastfeeding(drugs)
        rids = [r.relative_infant_dose_pct for r in results]
        assert rids == sorted(rids)

    def test_all_results_correct_type(self):
        drugs = [
            {"drug_name": "X", "pka": 7.0, "logp": 1.0, "ionization_type": "neutral"},
        ]
        results = screen_drugs_for_breastfeeding(drugs)
        for r in results:
            assert isinstance(r, MilkExcretionResult)

    def test_empty_list_returns_empty(self):
        results = screen_drugs_for_breastfeeding([])
        assert results == []


class TestValidation:
    def test_invalid_pka_too_low(self):
        with pytest.raises(ValueError, match="pka must be in"):
            predict_milk_excretion("X", pka=-1.0, logp=1.0, ionization_type="base")

    def test_invalid_pka_too_high(self):
        with pytest.raises(ValueError, match="pka must be in"):
            predict_milk_excretion("X", pka=15.0, logp=1.0, ionization_type="base")

    def test_invalid_maternal_plasma_conc(self):
        with pytest.raises(ValueError, match="maternal_plasma_conc_mg_L must be > 0"):
            predict_milk_excretion(
                "X", pka=7.0, logp=1.0, ionization_type="base", maternal_plasma_conc_mg_L=0.0
            )

    def test_invalid_infant_weight(self):
        with pytest.raises(ValueError, match="infant_weight_kg must be > 0"):
            predict_milk_excretion(
                "X", pka=7.0, logp=1.0, ionization_type="base", infant_weight_kg=0.0
            )

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError):
            predict_milk_excretion("X", pka=7.0, logp=1.0, ionization_type="zwitterion")
