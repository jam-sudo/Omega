"""Tests for Phase 491 — Ocular Bioavailability Predictor."""

import math

import pytest

from omega_pbpk.prediction.ocular_bioavailability_p491 import (
    OcularBioavailabilityResult,
    predict_ocular_bioavailability,
    screen_ophthalmic_candidates,
)


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = predict_ocular_bioavailability("TestDrug", logP=2.5, mw_Da=300.0)
        assert isinstance(result, OcularBioavailabilityResult)

    def test_result_is_frozen(self):
        result = predict_ocular_bioavailability("TestDrug", logP=2.5, mw_Da=300.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = predict_ocular_bioavailability("Timolol", logP=2.5, mw_Da=300.0)
        assert result.drug_name == "Timolol"

    def test_logp_preserved(self):
        result = predict_ocular_bioavailability("Drug", logP=1.8, mw_Da=250.0)
        assert result.logP == 1.8

    def test_mw_preserved(self):
        result = predict_ocular_bioavailability("Drug", logP=2.0, mw_Da=350.0)
        assert result.mw_Da == 350.0


class TestOptimalLogP:
    def test_logp_25_is_highest_kp_vs_low_logp(self):
        """logP=2.5 should yield higher Kp than logP=0 for same MW."""
        optimal = predict_ocular_bioavailability("Opt", logP=2.5, mw_Da=300.0)
        low = predict_ocular_bioavailability("Low", logP=0.0, mw_Da=300.0)
        assert optimal.kp_cornea_cm_per_h > low.kp_cornea_cm_per_h

    def test_logp_25_is_highest_kp_vs_high_logp(self):
        optimal = predict_ocular_bioavailability("Opt", logP=2.5, mw_Da=300.0)
        high = predict_ocular_bioavailability("High", logP=5.0, mw_Da=300.0)
        assert optimal.kp_cornea_cm_per_h > high.kp_cornea_cm_per_h

    def test_logp_symmetric_penalty(self):
        """Same distance from 2.5 should give identical Kp."""
        plus = predict_ocular_bioavailability("Plus", logP=4.5, mw_Da=300.0)
        minus = predict_ocular_bioavailability("Minus", logP=0.5, mw_Da=300.0)
        assert math.isclose(plus.kp_cornea_cm_per_h, minus.kp_cornea_cm_per_h, rel_tol=1e-6)

    def test_kp_cornea_positive(self):
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=300.0)
        assert result.kp_cornea_cm_per_h > 0


class TestMWEffect:
    def test_higher_mw_lower_kp(self):
        low_mw = predict_ocular_bioavailability("Small", logP=2.5, mw_Da=200.0)
        high_mw = predict_ocular_bioavailability("Large", logP=2.5, mw_Da=600.0)
        assert low_mw.kp_cornea_cm_per_h > high_mw.kp_cornea_cm_per_h

    def test_kp_monotonically_decreasing_with_mw(self):
        mws = [100.0, 200.0, 300.0, 500.0, 800.0]
        kps = [
            predict_ocular_bioavailability("D", logP=2.5, mw_Da=mw).kp_cornea_cm_per_h for mw in mws
        ]
        for i in range(1, len(kps)):
            assert kps[i] < kps[i - 1]


class TestIonization:
    def test_neutral_f_unionized_is_one(self):
        result = predict_ocular_bioavailability(
            "Drug", logP=2.5, mw_Da=300.0, ionization_type="neutral"
        )
        assert result.f_unionized == 1.0

    def test_ionized_acid_lower_kp_eff(self):
        """An acidic drug at pH 7.4 with pKa=5 is mostly ionized => lower kp_eff."""
        neutral = predict_ocular_bioavailability(
            "N", logP=2.5, mw_Da=300.0, ionization_type="neutral"
        )
        ionized = predict_ocular_bioavailability(
            "A", logP=2.5, mw_Da=300.0, pKa=5.0, ionization_type="acid"
        )
        assert ionized.kp_effective_cm_per_h < neutral.kp_effective_cm_per_h

    def test_ionized_base_lower_kp_eff(self):
        """Basic drug with pKa=10 (mostly protonated at pH 7.4) => low f_unionized."""
        neutral = predict_ocular_bioavailability(
            "N", logP=2.5, mw_Da=300.0, ionization_type="neutral"
        )
        ionized = predict_ocular_bioavailability(
            "B", logP=2.5, mw_Da=300.0, pKa=10.0, ionization_type="base"
        )
        assert ionized.kp_effective_cm_per_h < neutral.kp_effective_cm_per_h

    def test_f_unionized_in_range(self):
        for itype in ["acid", "base", "neutral"]:
            result = predict_ocular_bioavailability(
                "Drug", logP=2.5, mw_Da=300.0, pKa=7.4, ionization_type=itype
            )
            assert 0.0 <= result.f_unionized <= 1.0


class TestBioavailabilityRange:
    def test_f_ocular_aqueous_in_range(self):
        for logP in [-1.0, 0.0, 2.5, 4.0, 6.0]:
            result = predict_ocular_bioavailability("Drug", logP=logP, mw_Da=300.0)
            assert 0.0 <= result.f_ocular_aqueous <= 0.05

    def test_f_ocular_aqueous_max_is_0_05(self):
        """Even with maximum permeability, F_ocular <= 5%."""
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=100.0)
        assert result.f_ocular_aqueous <= 0.05

    def test_f_ocular_vitreous_smaller_than_aqueous(self):
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=300.0)
        assert result.f_ocular_vitreous < result.f_ocular_aqueous

    def test_kp_effective_clamped(self):
        # Very large logP and tiny MW
        result = predict_ocular_bioavailability("Extreme", logP=100.0, mw_Da=10.0)
        assert result.kp_effective_cm_per_h >= 1e-8
        assert result.kp_effective_cm_per_h <= 0.1


class TestPenetrationClass:
    def test_class_poor_below_0001(self):
        """Force kp_eff < 0.001 by using extreme logP and high MW."""
        result = predict_ocular_bioavailability(
            "Drug", logP=10.0, mw_Da=800.0, pKa=5.0, ionization_type="acid"
        )
        # kp_eff will be very small
        assert result.aqueous_penetration_class == "poor"

    def test_class_good_at_optimal(self):
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=100.0)
        # kp_eff should be high enough for "good"
        assert result.aqueous_penetration_class in ("moderate", "good")

    def test_classification_consistent_with_kp_eff(self):
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=300.0)
        kp = result.kp_effective_cm_per_h
        if kp < 0.001:
            assert result.aqueous_penetration_class == "poor"
        elif kp <= 0.01:
            assert result.aqueous_penetration_class == "moderate"
        else:
            assert result.aqueous_penetration_class == "good"

    def test_dosing_recommendation_poor_is_tid(self):
        result = predict_ocular_bioavailability(
            "Drug", logP=10.0, mw_Da=800.0, pKa=5.0, ionization_type="acid"
        )
        assert result.aqueous_penetration_class == "poor"
        assert result.dosing_recommendation == "TID"

    def test_dosing_recommendation_mapping(self):
        # At logP=2.5, MW=100, should be "good" => "QD"
        result = predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=100.0)
        if result.aqueous_penetration_class == "good":
            assert result.dosing_recommendation == "QD"
        elif result.aqueous_penetration_class == "moderate":
            assert result.dosing_recommendation == "BID"
        else:
            assert result.dosing_recommendation == "TID"


class TestScreenFunction:
    def _make_compounds(self):
        return [
            {"drug_name": "DrugA", "logP": 4.0, "mw_Da": 500.0},
            {"drug_name": "DrugB", "logP": 2.5, "mw_Da": 300.0},
            {"drug_name": "DrugC", "logP": 0.0, "mw_Da": 800.0},
        ]

    def test_screen_returns_list(self):
        results = screen_ophthalmic_candidates(self._make_compounds())
        assert isinstance(results, list)

    def test_screen_returns_correct_count(self):
        results = screen_ophthalmic_candidates(self._make_compounds())
        assert len(results) == 3

    def test_screen_sorted_descending(self):
        results = screen_ophthalmic_candidates(self._make_compounds())
        for i in range(1, len(results)):
            assert results[i - 1].kp_effective_cm_per_h >= results[i].kp_effective_cm_per_h

    def test_screen_all_result_types(self):
        results = screen_ophthalmic_candidates(self._make_compounds())
        for r in results:
            assert isinstance(r, OcularBioavailabilityResult)

    def test_screen_with_optional_params(self):
        compounds = [
            {
                "drug_name": "Acid",
                "logP": 2.5,
                "mw_Da": 300.0,
                "pKa": 5.0,
                "ionization_type": "acid",
            },
            {"drug_name": "Neutral", "logP": 2.5, "mw_Da": 300.0},
        ]
        results = screen_ophthalmic_candidates(compounds)
        # Neutral should have higher kp_eff (sorted first)
        assert results[0].drug_name == "Neutral"


class TestValidation:
    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw_Da must be > 0"):
            predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=0.0)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError, match="mw_Da must be > 0"):
            predict_ocular_bioavailability("Drug", logP=2.5, mw_Da=-100.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type must be one of"):
            predict_ocular_bioavailability(
                "Drug", logP=2.5, mw_Da=300.0, ionization_type="zwitterion"
            )

    def test_screen_propagates_validation_error(self):
        compounds = [{"drug_name": "Bad", "logP": 2.5, "mw_Da": -1.0}]
        with pytest.raises(ValueError):
            screen_ophthalmic_candidates(compounds)
