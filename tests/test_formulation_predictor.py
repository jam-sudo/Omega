"""
Tests for Phase 588 — prediction/formulation_predictor.py
"""

import pytest

from omega_pbpk.prediction.formulation_predictor import (
    compare_formulations,
    formulation_bioavailability_estimate,
    recommend_formulation,
)

# ---------------------------------------------------------------------------
# recommend_formulation
# ---------------------------------------------------------------------------


class TestRecommendFormulation:
    def test_poor_solubility_high_logP_gives_lipid_based(self):
        result = recommend_formulation(
            logP=4.5,
            pka=7.0,
            drug_type="neutral",
            dose_mg=50.0,
            solubility_mg_mL=0.05,
            t_half_h=6.0,
        )
        assert result["recommended_formulation"] == "lipid_based"

    def test_short_half_life_gives_extended_release(self):
        result = recommend_formulation(
            logP=1.5,
            pka=7.0,
            drug_type="neutral",
            dose_mg=100.0,
            solubility_mg_mL=5.0,
            t_half_h=2.0,
        )
        assert result["recommended_formulation"] == "extended_release"

    def test_long_half_life_low_dose_gives_ir(self):
        result = recommend_formulation(
            logP=2.0,
            pka=7.0,
            drug_type="neutral",
            dose_mg=50.0,
            solubility_mg_mL=5.0,
            t_half_h=24.0,
        )
        assert result["recommended_formulation"] == "immediate_release"

    def test_highly_hydrophilic_good_solubility_gives_solution(self):
        result = recommend_formulation(
            logP=-1.5,
            pka=7.0,
            drug_type="neutral",
            dose_mg=20.0,
            solubility_mg_mL=50.0,
            t_half_h=6.0,
        )
        assert result["recommended_formulation"] == "solution"

    def test_bioavailability_concern_true_when_low_solubility(self):
        result = recommend_formulation(
            logP=3.0, pka=7.0, drug_type="neutral", dose_mg=50.0, solubility_mg_mL=0.5, t_half_h=6.0
        )
        assert result["bioavailability_concern"] is True

    def test_bioavailability_concern_false_when_good_solubility(self):
        result = recommend_formulation(
            logP=1.0,
            pka=7.0,
            drug_type="neutral",
            dose_mg=50.0,
            solubility_mg_mL=10.0,
            t_half_h=6.0,
        )
        assert result["bioavailability_concern"] is False

    def test_salt_form_suggested_for_acid(self):
        result = recommend_formulation(
            logP=2.0, pka=4.5, drug_type="acid", dose_mg=100.0, solubility_mg_mL=2.0, t_half_h=8.0
        )
        assert result["salt_form_suggested"] is True

    def test_salt_form_suggested_for_base(self):
        result = recommend_formulation(
            logP=2.0, pka=9.0, drug_type="base", dose_mg=100.0, solubility_mg_mL=2.0, t_half_h=8.0
        )
        assert result["salt_form_suggested"] is True

    def test_salt_form_not_suggested_for_neutral(self):
        result = recommend_formulation(
            logP=2.0,
            pka=7.0,
            drug_type="neutral",
            dose_mg=100.0,
            solubility_mg_mL=5.0,
            t_half_h=6.0,
        )
        assert result["salt_form_suggested"] is False

    def test_alternative_formulations_is_list(self):
        result = recommend_formulation(
            logP=2.0,
            pka=7.0,
            drug_type="neutral",
            dose_mg=100.0,
            solubility_mg_mL=5.0,
            t_half_h=6.0,
        )
        assert isinstance(result["alternative_formulations"], list)

    def test_rationale_is_nonempty_string(self):
        result = recommend_formulation(
            logP=2.0,
            pka=7.0,
            drug_type="neutral",
            dose_mg=100.0,
            solubility_mg_mL=5.0,
            t_half_h=6.0,
        )
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"]) > 0

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            recommend_formulation(
                logP=2.0,
                pka=7.0,
                drug_type="neutral",
                dose_mg=0.0,
                solubility_mg_mL=5.0,
                t_half_h=6.0,
            )

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            recommend_formulation(
                logP=2.0,
                pka=7.0,
                drug_type="zwitterion",
                dose_mg=50.0,
                solubility_mg_mL=5.0,
                t_half_h=6.0,
            )


# ---------------------------------------------------------------------------
# formulation_bioavailability_estimate
# ---------------------------------------------------------------------------


class TestFormulationBioavailabilityEstimate:
    def test_solution_full_fa_when_dose_dissolved(self):
        result = formulation_bioavailability_estimate(
            formulation="solution", logP=1.0, solubility_mg_mL=10.0, dose_mg=50.0, peff_cm_s=1e-4
        )
        assert abs(result["fa_predicted"] - 1.0) < 1e-9

    def test_solution_limited_fa_when_dose_exceeds_solubility(self):
        result = formulation_bioavailability_estimate(
            formulation="solution", logP=1.0, solubility_mg_mL=0.01, dose_mg=500.0, peff_cm_s=1e-4
        )
        assert result["fa_predicted"] < 1.0

    def test_lipid_based_fa_above_06_for_high_logP(self):
        result = formulation_bioavailability_estimate(
            formulation="lipid_based", logP=4.0, solubility_mg_mL=0.05, dose_mg=50.0, peff_cm_s=1e-4
        )
        assert result["fa_predicted"] > 0.6

    def test_lipid_based_vs_ir_high_logP(self):
        """Lipid-based should give higher fa than IR for a poorly soluble drug."""
        fa_lb = formulation_bioavailability_estimate(
            formulation="lipid_based",
            logP=4.0,
            solubility_mg_mL=0.01,
            dose_mg=100.0,
            peff_cm_s=5e-5,
        )["fa_predicted"]
        fa_ir = formulation_bioavailability_estimate(
            formulation="immediate_release",
            logP=4.0,
            solubility_mg_mL=0.01,
            dose_mg=100.0,
            peff_cm_s=5e-5,
        )["fa_predicted"]
        assert fa_lb > fa_ir

    def test_er_fa_less_than_ir(self):
        """Extended release should have lower fa due to absorption window."""
        fa_er = formulation_bioavailability_estimate(
            formulation="extended_release",
            logP=2.0,
            solubility_mg_mL=5.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
        )["fa_predicted"]
        fa_ir = formulation_bioavailability_estimate(
            formulation="immediate_release",
            logP=2.0,
            solubility_mg_mL=5.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
        )["fa_predicted"]
        assert fa_er < fa_ir

    def test_fa_between_0_and_1(self):
        for f in [
            "solution",
            "immediate_release",
            "lipid_based",
            "extended_release",
            "nanoparticle",
        ]:
            result = formulation_bioavailability_estimate(
                formulation=f, logP=2.0, solubility_mg_mL=1.0, dose_mg=100.0, peff_cm_s=1e-4
            )
            assert 0.0 <= result["fa_predicted"] <= 1.0, f"fa out of range for {f}"

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError, match="Invalid formulation"):
            formulation_bioavailability_estimate(
                formulation="magic_pill",
                logP=2.0,
                solubility_mg_mL=1.0,
                dose_mg=100.0,
                peff_cm_s=1e-4,
            )

    def test_returns_correct_keys(self):
        result = formulation_bioavailability_estimate(
            formulation="immediate_release",
            logP=2.0,
            solubility_mg_mL=5.0,
            dose_mg=50.0,
            peff_cm_s=1e-4,
        )
        assert "formulation" in result
        assert "fa_predicted" in result
        assert "note" in result


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_five_entries(self):
        results = compare_formulations(
            drug_name="TestDrug",
            logP=2.0,
            solubility_mg_mL=1.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
            t_half_h=6.0,
        )
        assert len(results) == 5

    def test_sorted_by_fa_descending(self):
        results = compare_formulations(
            drug_name="TestDrug",
            logP=2.0,
            solubility_mg_mL=1.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
            t_half_h=6.0,
        )
        fas = [r["fa_predicted"] for r in results]
        assert fas == sorted(fas, reverse=True)

    def test_each_entry_has_required_keys(self):
        results = compare_formulations(
            drug_name="TestDrug",
            logP=2.0,
            solubility_mg_mL=1.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
            t_half_h=6.0,
        )
        for r in results:
            assert "formulation" in r
            assert "fa_predicted" in r
            assert "relative_advantage_vs_IR" in r

    def test_ir_relative_advantage_is_zero(self):
        results = compare_formulations(
            drug_name="TestDrug",
            logP=2.0,
            solubility_mg_mL=1.0,
            dose_mg=100.0,
            peff_cm_s=1e-4,
            t_half_h=6.0,
        )
        ir_entry = next(r for r in results if r["formulation"] == "immediate_release")
        assert abs(ir_entry["relative_advantage_vs_IR"]) < 1e-9
