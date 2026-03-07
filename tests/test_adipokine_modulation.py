"""Tests for Phase 918 — Drug Adipokine Modulation Score."""

import math
import pytest

from omega_pbpk.prediction.adipokine_modulation import (
    AdipokineModulationResult,
    predict_adipokine_modulation,
    screen_metabolic_risk,
)


class TestAdipokineModulationResultDataclass:
    def test_is_frozen(self):
        result = predict_adipokine_modulation("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_fields_present(self):
        result = predict_adipokine_modulation("TestDrug")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "drug_class")
        assert hasattr(result, "kp_adipose_estimate")
        assert hasattr(result, "leptin_effect_direction")
        assert hasattr(result, "adiponectin_effect_direction")
        assert hasattr(result, "tnf_alpha_effect_direction")
        assert hasattr(result, "metabolic_syndrome_risk_modifier")
        assert hasattr(result, "adipose_accumulation_index")
        assert hasattr(result, "notes")


class TestValidation:
    def test_invalid_drug_class_raises(self):
        with pytest.raises(ValueError, match="drug_class"):
            predict_adipokine_modulation("Drug", drug_class="unknown")

    def test_valid_drug_classes_accepted(self):
        for cls in ["antidiabetic", "statin", "NSAID", "antipsychotic", "other"]:
            result = predict_adipokine_modulation("Drug", drug_class=cls)
            assert result.drug_class == cls


class TestDefaultParameters:
    def test_returns_result_type(self):
        result = predict_adipokine_modulation("Generic")
        assert isinstance(result, AdipokineModulationResult)

    def test_drug_name_stored(self):
        result = predict_adipokine_modulation("Metformin")
        assert result.drug_name == "Metformin"

    def test_logp_stored(self):
        result = predict_adipokine_modulation("Drug", logp=3.5)
        assert result.logp == 3.5

    def test_mw_stored(self):
        result = predict_adipokine_modulation("Drug", mw_Da=500.0)
        assert result.mw_Da == 500.0

    def test_drug_class_stored(self):
        result = predict_adipokine_modulation("Drug", drug_class="statin")
        assert result.drug_class == "statin"

    def test_kp_adipose_calculation(self):
        # logp=2.0 → 10^(2.0*0.3) = 10^0.6
        result = predict_adipokine_modulation("Drug", logp=2.0)
        expected = 10.0 ** (2.0 * 0.3)
        assert result.kp_adipose_estimate == pytest.approx(expected, rel=1e-6)

    def test_kp_adipose_clamped_min(self):
        result = predict_adipokine_modulation("Drug", logp=-10.0)
        assert result.kp_adipose_estimate >= 0.1

    def test_kp_adipose_clamped_max(self):
        result = predict_adipokine_modulation("Drug", logp=10.0)
        assert result.kp_adipose_estimate <= 50.0


class TestAntidiabeticClass:
    def test_leptin_decreases(self):
        result = predict_adipokine_modulation("Metformin", drug_class="antidiabetic")
        assert result.leptin_effect_direction == "decrease"

    def test_adiponectin_increases(self):
        result = predict_adipokine_modulation("Metformin", drug_class="antidiabetic")
        assert result.adiponectin_effect_direction == "increase"

    def test_tnf_decreases(self):
        result = predict_adipokine_modulation("Metformin", drug_class="antidiabetic")
        assert result.tnf_alpha_effect_direction == "decrease"

    def test_metabolic_modifier(self):
        result = predict_adipokine_modulation("Metformin", drug_class="antidiabetic")
        assert result.metabolic_syndrome_risk_modifier == pytest.approx(0.7)

    def test_antidiabetic_notes(self):
        result = predict_adipokine_modulation("Metformin", drug_class="antidiabetic")
        assert "Antidiabetic" in result.notes
        assert "favorable adipokine profile" in result.notes


class TestStatinClass:
    def test_adiponectin_increases(self):
        result = predict_adipokine_modulation("Atorvastatin", drug_class="statin")
        assert result.adiponectin_effect_direction == "increase"

    def test_leptin_neutral(self):
        result = predict_adipokine_modulation("Atorvastatin", drug_class="statin")
        assert result.leptin_effect_direction == "neutral"

    def test_tnf_decreases(self):
        result = predict_adipokine_modulation("Atorvastatin", drug_class="statin")
        assert result.tnf_alpha_effect_direction == "decrease"

    def test_metabolic_modifier(self):
        result = predict_adipokine_modulation("Atorvastatin", drug_class="statin")
        assert result.metabolic_syndrome_risk_modifier == pytest.approx(0.85)


class TestNSAIDClass:
    def test_tnf_decreases(self):
        result = predict_adipokine_modulation("Ibuprofen", drug_class="NSAID")
        assert result.tnf_alpha_effect_direction == "decrease"

    def test_leptin_neutral(self):
        result = predict_adipokine_modulation("Ibuprofen", drug_class="NSAID")
        assert result.leptin_effect_direction == "neutral"

    def test_metabolic_modifier(self):
        result = predict_adipokine_modulation("Ibuprofen", drug_class="NSAID")
        assert result.metabolic_syndrome_risk_modifier == pytest.approx(0.9)


class TestAntipsychoticClass:
    def test_leptin_increases(self):
        result = predict_adipokine_modulation("Olanzapine", drug_class="antipsychotic")
        assert result.leptin_effect_direction == "increase"

    def test_adiponectin_decreases(self):
        result = predict_adipokine_modulation("Olanzapine", drug_class="antipsychotic")
        assert result.adiponectin_effect_direction == "decrease"

    def test_tnf_increases(self):
        result = predict_adipokine_modulation("Olanzapine", drug_class="antipsychotic")
        assert result.tnf_alpha_effect_direction == "increase"

    def test_metabolic_modifier(self):
        result = predict_adipokine_modulation("Olanzapine", drug_class="antipsychotic")
        assert result.metabolic_syndrome_risk_modifier == pytest.approx(1.4)

    def test_antipsychotic_notes(self):
        result = predict_adipokine_modulation("Olanzapine", drug_class="antipsychotic")
        assert "Antipsychotic class" in result.notes
        assert "metabolic syndrome" in result.notes


class TestAccumulationIndex:
    def test_accumulation_index_bounds(self):
        result = predict_adipokine_modulation("Drug", logp=2.0)
        assert 0.0 <= result.adipose_accumulation_index <= 100.0

    def test_accumulation_index_clamped_max(self):
        result = predict_adipokine_modulation("Drug", logp=10.0)
        assert result.adipose_accumulation_index == pytest.approx(100.0)

    def test_accumulation_index_increases_with_logp(self):
        r1 = predict_adipokine_modulation("Drug", logp=1.0)
        r2 = predict_adipokine_modulation("Drug", logp=3.0)
        assert r2.adipose_accumulation_index >= r1.adipose_accumulation_index


class TestOtherClass:
    def test_neutral_effects(self):
        result = predict_adipokine_modulation("Unknown", drug_class="other")
        assert result.leptin_effect_direction == "neutral"
        assert result.adiponectin_effect_direction == "neutral"
        assert result.tnf_alpha_effect_direction == "neutral"

    def test_neutral_modifier(self):
        result = predict_adipokine_modulation("Unknown", drug_class="other")
        assert result.metabolic_syndrome_risk_modifier == pytest.approx(1.0)

    def test_neutral_notes(self):
        result = predict_adipokine_modulation("Unknown", drug_class="other")
        assert "Neutral or favorable" in result.notes


class TestScreenMetabolicRisk:
    def test_sorted_by_risk_modifier_descending(self):
        candidates = [
            {"name": "Metformin", "logp": -0.5, "drug_class": "antidiabetic"},
            {"name": "Olanzapine", "logp": 2.8, "drug_class": "antipsychotic"},
            {"name": "Ibuprofen", "logp": 3.5, "drug_class": "NSAID"},
        ]
        results = screen_metabolic_risk(candidates)
        assert results[0].drug_name == "Olanzapine"
        modifiers = [r.metabolic_syndrome_risk_modifier for r in results]
        assert modifiers == sorted(modifiers, reverse=True)

    def test_default_mw_and_class(self):
        candidates = [{"name": "Drug", "logp": 1.0}]
        results = screen_metabolic_risk(candidates)
        assert results[0].mw_Da == 400.0
        assert results[0].drug_class == "other"

    def test_empty_list(self):
        results = screen_metabolic_risk([])
        assert results == []

    def test_single_candidate(self):
        candidates = [{"name": "Statin", "logp": 4.0, "drug_class": "statin"}]
        results = screen_metabolic_risk(candidates)
        assert len(results) == 1
        assert results[0].drug_name == "Statin"
