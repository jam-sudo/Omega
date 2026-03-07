"""Tests for biopharmaceutics/dose_number.py — BCS Dose Number (Phase 244)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.dose_number import (
    BCSRiskResult,
    bcs_risk_assessment,
    dose_escalation_fa,
    dose_number,
    predict_fa_from_do,
)

# ---------------------------------------------------------------------------
# dose_number tests
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_formula_correct(self):
        """Do = dose / (250 * sol)."""
        do = dose_number(dose_mg=250.0, solubility_mg_mL=1.0, v_gi_mL=250.0)
        assert do == pytest.approx(1.0, rel=1e-6)

    def test_do_below_one_low_dose(self):
        """Small dose relative to solubility gives Do < 1."""
        do = dose_number(dose_mg=10.0, solubility_mg_mL=1.0, v_gi_mL=250.0)
        assert do < 1.0
        assert do == pytest.approx(0.04, rel=1e-6)

    def test_do_above_one_high_dose(self):
        """Large dose relative to solubility gives Do > 1."""
        do = dose_number(dose_mg=500.0, solubility_mg_mL=0.5, v_gi_mL=250.0)
        assert do > 1.0
        assert do == pytest.approx(4.0, rel=1e-6)

    def test_default_v_gi_250_mL(self):
        """Default GI volume should be 250 mL per FDA BCS guidance."""
        do_default = dose_number(dose_mg=100.0, solubility_mg_mL=0.5)
        do_explicit = dose_number(dose_mg=100.0, solubility_mg_mL=0.5, v_gi_mL=250.0)
        assert do_default == pytest.approx(do_explicit, rel=1e-9)

    def test_non_positive_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            dose_number(0.0, 1.0)

    def test_non_positive_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            dose_number(100.0, 0.0)

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            dose_number(100.0, -0.5)


# ---------------------------------------------------------------------------
# predict_fa_from_do tests
# ---------------------------------------------------------------------------


class TestPredictFaFromDo:
    def test_low_do_high_fa(self):
        """Do < 1 with high permeability gives high fa."""
        fa = predict_fa_from_do(do=0.1, peff_cm_s=5e-4)
        assert fa > 0.8

    def test_high_do_low_fa(self):
        """Do = 5 with same permeability gives lower fa."""
        fa_low_do = predict_fa_from_do(do=0.5, peff_cm_s=5e-4)
        fa_high_do = predict_fa_from_do(do=5.0, peff_cm_s=5e-4)
        assert fa_low_do > fa_high_do

    def test_fa_in_range_zero_to_one(self):
        """fa should always be in [0, 1]."""
        for do in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            fa = predict_fa_from_do(do, peff_cm_s=1e-5)
            assert 0.0 <= fa <= 1.0

    def test_fa_below_one_high_do(self):
        """Very high Do should give fa well below 1."""
        fa = predict_fa_from_do(do=100.0, peff_cm_s=1e-3)
        assert fa < 0.1

    def test_fa_not_zero_for_moderate_do(self):
        """Moderate Do should still give some absorption."""
        fa = predict_fa_from_do(do=2.0, peff_cm_s=1e-4)
        assert fa > 0.0

    def test_fa_do_below_one_same_as_permeability_limited(self):
        """For Do < 1, fa equals the permeability-limited value."""
        import math

        peff = 3e-4
        tau = 3.0 * 3600.0
        r = 1.75
        fa_perm = 1.0 - math.exp(-2.0 * peff * tau / r)
        fa = predict_fa_from_do(do=0.3, peff_cm_s=peff)
        assert fa == pytest.approx(fa_perm, abs=1e-6)

    def test_fa_do_above_one_scales_with_do(self):
        """For Do > 1, fa = fa_perm / Do (linear relationship)."""
        import math

        peff = 3e-4
        tau = 3.0 * 3600.0
        r = 1.75
        fa_perm = 1.0 - math.exp(-2.0 * peff * tau / r)
        do = 3.0
        fa = predict_fa_from_do(do=do, peff_cm_s=peff)
        expected = fa_perm / do
        assert fa == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# bcs_risk_assessment tests
# ---------------------------------------------------------------------------


class TestBCSRiskAssessment:
    def test_bcs_class_i_low_do_high_perm(self):
        """Do < 1, Peff >= 2e-4 → BCS Class I."""
        result = bcs_risk_assessment(
            drug_name="DrugI",
            dose_mg=10.0,
            solubility_mg_mL_pH68=1.0,  # Do=0.04
            peff_cm_s=5e-4,
            logP=1.5,
            mw=250.0,
        )
        assert result.bcs_class == "I"
        assert result.biowaiver_eligible is True
        assert result.dissolution_limited is False
        assert result.permeability_limited is False
        assert result.risk_level == "low"

    def test_bcs_class_ii_high_do_high_perm(self):
        """Do >= 1, Peff >= 2e-4 → BCS Class II."""
        result = bcs_risk_assessment(
            drug_name="DrugII",
            dose_mg=500.0,
            solubility_mg_mL_pH68=0.5,  # Do=4.0
            peff_cm_s=5e-4,
            logP=3.0,
            mw=320.0,
        )
        assert result.bcs_class == "II"
        assert result.biowaiver_eligible is False
        assert result.dissolution_limited is True
        assert result.permeability_limited is False
        assert result.risk_level == "moderate"

    def test_bcs_class_iii_low_do_low_perm(self):
        """Do < 1, Peff < 2e-4 → BCS Class III."""
        result = bcs_risk_assessment(
            drug_name="DrugIII",
            dose_mg=10.0,
            solubility_mg_mL_pH68=1.0,  # Do=0.04
            peff_cm_s=5e-5,
            logP=0.5,
            mw=200.0,
        )
        assert result.bcs_class == "III"
        assert result.biowaiver_eligible is False
        assert result.dissolution_limited is False
        assert result.permeability_limited is True
        assert result.risk_level == "moderate"

    def test_bcs_class_iv_high_do_low_perm(self):
        """Do >= 1, Peff < 2e-4 → BCS Class IV."""
        result = bcs_risk_assessment(
            drug_name="DrugIV",
            dose_mg=500.0,
            solubility_mg_mL_pH68=0.1,  # Do=20.0
            peff_cm_s=5e-5,
            logP=5.0,
            mw=450.0,
        )
        assert result.bcs_class == "IV"
        assert result.biowaiver_eligible is False
        assert result.dissolution_limited is True
        assert result.permeability_limited is True
        assert result.risk_level == "high"

    def test_result_is_bcs_risk_result(self):
        result = bcs_risk_assessment("Drug", 100.0, 1.0, 5e-4, 2.0, 300.0)
        assert isinstance(result, BCSRiskResult)

    def test_dose_number_computed_correctly(self):
        """dose_number field = dose / (250 * sol)."""
        result = bcs_risk_assessment("Drug", 250.0, 1.0, 5e-4, 2.0, 300.0)
        expected_do = 250.0 / (250.0 * 1.0)  # = 1.0
        assert result.dose_number == pytest.approx(expected_do, rel=1e-6)

    def test_predicted_fa_in_range(self):
        """predicted_fa must be in [0, 1]."""
        result = bcs_risk_assessment("Drug", 500.0, 0.1, 5e-5, 3.0, 400.0)
        assert 0.0 <= result.predicted_fa <= 1.0

    def test_recommendation_populated(self):
        result = bcs_risk_assessment("Drug", 100.0, 1.0, 5e-4, 2.0, 300.0)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0

    def test_validation_non_positive_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            bcs_risk_assessment("Drug", 0.0, 1.0, 5e-4, 2.0, 300.0)

    def test_validation_non_positive_solubility(self):
        with pytest.raises(ValueError, match="solubility"):
            bcs_risk_assessment("Drug", 100.0, 0.0, 5e-4, 2.0, 300.0)

    def test_validation_non_positive_peff(self):
        with pytest.raises(ValueError, match="peff"):
            bcs_risk_assessment("Drug", 100.0, 1.0, 0.0, 2.0, 300.0)

    def test_notes_is_list(self):
        result = bcs_risk_assessment("Drug", 100.0, 1.0, 5e-4, 2.0, 300.0)
        assert isinstance(result.notes, list)


# ---------------------------------------------------------------------------
# dose_escalation_fa tests
# ---------------------------------------------------------------------------


class TestDoseEscalationFa:
    def test_sorted_by_dose_ascending(self):
        """Results should be sorted by dose_mg ascending."""
        doses = [500.0, 50.0, 200.0, 10.0]
        results = dose_escalation_fa("Drug", solubility_mg_mL=0.5, peff_cm_s=5e-4, doses_mg=doses)
        dose_values = [r.dose_mg for r in results]
        assert dose_values == sorted(dose_values)

    def test_fa_decreases_with_dose_dissolution_limited(self):
        """For dissolution-limited drug (low sol), fa should decrease as dose increases."""
        doses = [100.0, 500.0, 1000.0]
        results = dose_escalation_fa("Drug", solubility_mg_mL=0.1, peff_cm_s=5e-4, doses_mg=doses)
        # Sort by dose
        results_sorted = sorted(results, key=lambda r: r.dose_mg)
        fa_values = [r.predicted_fa for r in results_sorted]
        # fa should be non-increasing as dose increases (all above Do=1 threshold)
        for i in range(len(fa_values) - 1):
            assert fa_values[i] >= fa_values[i + 1] - 1e-9

    def test_returns_correct_count(self):
        """Should return one result per dose."""
        doses = [10.0, 50.0, 100.0, 250.0, 500.0]
        results = dose_escalation_fa("Drug", 0.5, 5e-4, doses)
        assert len(results) == 5

    def test_empty_doses_raises(self):
        with pytest.raises(ValueError, match="doses_mg"):
            dose_escalation_fa("Drug", 0.5, 5e-4, [])

    def test_non_positive_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            dose_escalation_fa("Drug", 0.0, 5e-4, [100.0])

    def test_non_positive_peff_raises(self):
        with pytest.raises(ValueError, match="peff"):
            dose_escalation_fa("Drug", 0.5, 0.0, [100.0])

    def test_result_type_in_list(self):
        """Each element should be a BCSRiskResult."""
        results = dose_escalation_fa("Drug", 0.5, 5e-4, [100.0, 200.0])
        for r in results:
            assert isinstance(r, BCSRiskResult)
