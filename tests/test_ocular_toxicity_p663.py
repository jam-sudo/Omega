"""Tests for Phase 663 - Drug Ocular Toxicity Prediction."""

import pytest

from omega_pbpk.risk.ocular_toxicity_p663 import (
    OcularToxicityResult,
    predict_ocular_toxicity,
    screen_ocular_toxicity,
)


def _default():
    return predict_ocular_toxicity(
        drug_name="Chloroquine",
        logP=4.6,
        mw_Da=319.9,
        pka=10.1,
        is_base=True,
        charge_at_ph7=1.0,
        amphiphilic=True,
    )


class TestFrozen:
    def test_return_type(self):
        r = _default()
        assert isinstance(r, OcularToxicityResult)

    def test_frozen_mutation_raises(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"

    def test_frozen_score_mutation_raises(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.overall_ocular_risk_score = 0.0


class TestMelaninBinding:
    def test_high_pka_base_high_melanin(self):
        r = predict_ocular_toxicity("DrugA", logP=3.0, mw_Da=300.0, pka=10.0, is_base=True)
        assert r.melanin_binding_score > 3.0

    def test_non_base_low_melanin(self):
        r = predict_ocular_toxicity("DrugB", logP=1.0, mw_Da=300.0, pka=5.0, is_base=False)
        assert r.melanin_binding_score == pytest.approx(0.3, abs=0.01)

    def test_base_low_pka_uses_logP(self):
        r = predict_ocular_toxicity("DrugC", logP=2.0, mw_Da=300.0, pka=6.0, is_base=True)
        assert r.melanin_binding_score == pytest.approx(0.6, abs=0.01)

    def test_melanin_clamped_0_10(self):
        r = predict_ocular_toxicity("DrugD", logP=-5.0, mw_Da=300.0, pka=6.0, is_base=False)
        assert r.melanin_binding_score >= 0.0
        r2 = predict_ocular_toxicity("DrugE", logP=10.0, mw_Da=300.0, pka=15.0, is_base=True)
        assert r2.melanin_binding_score <= 10.0


class TestCornealDeposition:
    def test_large_hydrophilic_high_corneal(self):
        r = predict_ocular_toxicity("DrugF", logP=-1.0, mw_Da=600.0)
        assert r.corneal_deposition_risk == pytest.approx(8.0, abs=0.01)

    def test_lipophilic_large_amphiphilic(self):
        r = predict_ocular_toxicity("DrugG", logP=4.0, mw_Da=450.0)
        assert r.corneal_deposition_risk == pytest.approx(6.0, abs=0.01)

    def test_small_default(self):
        r = predict_ocular_toxicity("DrugH", logP=1.0, mw_Da=200.0)
        assert 0.0 <= r.corneal_deposition_risk <= 5.0


class TestRetinalToxicity:
    def test_amphiphilic_increases_retinal(self):
        r1 = predict_ocular_toxicity("D1", logP=3.0, mw_Da=300.0, amphiphilic=False)
        r2 = predict_ocular_toxicity("D2", logP=3.0, mw_Da=300.0, amphiphilic=True)
        assert r2.retinal_toxicity_score > r1.retinal_toxicity_score

    def test_retinal_clamped(self):
        r = predict_ocular_toxicity("D3", logP=-5.0, mw_Da=200.0)
        assert r.retinal_toxicity_score >= 0.0
        r2 = predict_ocular_toxicity(
            "D4", logP=10.0, mw_Da=300.0, pka=15.0, is_base=True, amphiphilic=True
        )
        assert r2.retinal_toxicity_score <= 10.0


class TestLensAccumulation:
    def test_lens_positive_for_lipophilic(self):
        r = predict_ocular_toxicity("D5", logP=4.0, mw_Da=400.0)
        assert r.lens_accumulation_risk > 0.0

    def test_lens_clamped(self):
        r = predict_ocular_toxicity("D6", logP=-5.0, mw_Da=100.0)
        assert r.lens_accumulation_risk >= 0.0


class TestOverallRisk:
    def test_overall_in_range(self):
        r = _default()
        assert 0.0 <= r.overall_ocular_risk_score <= 10.0

    def test_risk_class_high(self):
        r = _default()
        # Chloroquine with high pKa and amphiphilic should be high risk
        assert r.risk_class == "high"

    def test_risk_class_low(self):
        r = predict_ocular_toxicity("SafeDrug", logP=-1.0, mw_Da=150.0, pka=5.0, is_base=False)
        assert r.risk_class == "low"

    def test_risk_class_moderate(self):
        r = predict_ocular_toxicity("ModDrug", logP=3.0, mw_Da=350.0, pka=9.0, is_base=True)
        assert r.risk_class == "moderate"


class TestMechanism:
    def test_primary_mechanism_is_valid(self):
        r = _default()
        assert r.primary_toxicity_mechanism in {
            "melanin_binding",
            "corneal_deposition",
            "retinal",
            "lens",
        }

    def test_highest_score_mechanism(self):
        # High pKa base → melanin binding should dominate
        r = predict_ocular_toxicity("DrugM", logP=1.0, mw_Da=200.0, pka=12.0, is_base=True)
        assert r.primary_toxicity_mechanism == "melanin_binding"


class TestMitigation:
    def test_high_risk_avoid(self):
        r = _default()
        assert r.mitigation_strategy == "avoid"

    def test_low_risk_standard(self):
        r = predict_ocular_toxicity("Safe", logP=-1.0, mw_Da=150.0, pka=5.0, is_base=False)
        assert r.mitigation_strategy == "standard_surveillance"

    def test_moderate_monitoring(self):
        r = predict_ocular_toxicity("Mod", logP=3.0, mw_Da=350.0, pka=9.0, is_base=True)
        assert r.mitigation_strategy == "monitoring_required"


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            predict_ocular_toxicity("Bad", logP=1.0, mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            predict_ocular_toxicity("Bad", logP=1.0, mw_Da=-100.0)


class TestScreen:
    def test_screen_sorted_descending(self):
        compounds = [
            {"drug_name": "A", "logP": -1.0, "mw_Da": 150.0, "pka": 5.0, "is_base": False},
            {
                "drug_name": "B",
                "logP": 4.6,
                "mw_Da": 320.0,
                "pka": 10.0,
                "is_base": True,
                "amphiphilic": True,
            },
            {"drug_name": "C", "logP": 2.0, "mw_Da": 300.0},
        ]
        results = screen_ocular_toxicity(compounds)
        assert len(results) == 3
        for i in range(len(results) - 1):
            assert results[i].overall_ocular_risk_score >= results[i + 1].overall_ocular_risk_score

    def test_screen_empty(self):
        assert screen_ocular_toxicity([]) == []
