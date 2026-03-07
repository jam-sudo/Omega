"""Tests for formulation optimizer (Phase 404)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.formulation_optimizer import (
    FormulationRecommendation,
    formulation_risk_assessment,
    recommend_formulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_kwargs(**overrides):
    """Return default valid kwargs for recommend_formulation."""
    base = {
        "drug_name": "TestDrug",
        "mw": 300.0,
        "logP": 2.0,
        "psa": 80.0,
        "solubility_mg_mL": 1.0,
        "permeability_cm_s": 5e-6,
        "dose_mg": 100.0,
        "bcs_class": "I",
        "target_tmax_h": 2.0,
        "target_duration_h": 8.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# recommend_formulation — input validation
# ---------------------------------------------------------------------------


class TestRecommendFormulationValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            recommend_formulation(**_default_kwargs(drug_name=""))

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            recommend_formulation(**_default_kwargs(mw=0.0))

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            recommend_formulation(**_default_kwargs(mw=-100.0))

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            recommend_formulation(**_default_kwargs(psa=-1.0))

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            recommend_formulation(**_default_kwargs(solubility_mg_mL=-0.01))

    def test_negative_permeability_raises(self):
        with pytest.raises(ValueError, match="permeability_cm_s"):
            recommend_formulation(**_default_kwargs(permeability_cm_s=-1e-6))

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            recommend_formulation(**_default_kwargs(dose_mg=0.0))

    def test_invalid_bcs_class_raises(self):
        with pytest.raises(ValueError, match="bcs_class"):
            recommend_formulation(**_default_kwargs(bcs_class="V"))

    def test_zero_target_tmax_raises(self):
        with pytest.raises(ValueError, match="target_tmax_h"):
            recommend_formulation(**_default_kwargs(target_tmax_h=0.0))

    def test_zero_target_duration_raises(self):
        with pytest.raises(ValueError, match="target_duration_h"):
            recommend_formulation(**_default_kwargs(target_duration_h=0.0))


# ---------------------------------------------------------------------------
# recommend_formulation — functional behaviour
# ---------------------------------------------------------------------------


class TestRecommendFormulationFunctional:
    def test_returns_correct_type(self):
        r = recommend_formulation(**_default_kwargs())
        assert isinstance(r, FormulationRecommendation)

    def test_drug_name_preserved(self):
        r = recommend_formulation(**_default_kwargs(drug_name="Ibuprofen"))
        assert r.drug_name == "Ibuprofen"

    def test_bcs_class_preserved(self):
        r = recommend_formulation(**_default_kwargs(bcs_class="II"))
        assert r.bcs_class == "II"

    def test_bcs_I_primary_strategy(self):
        r = recommend_formulation(**_default_kwargs(bcs_class="I"))
        assert "IR" in r.primary_strategy or "tablet" in r.primary_strategy.lower()

    def test_bcs_II_primary_strategy_contains_asd(self):
        r = recommend_formulation(**_default_kwargs(bcs_class="II"))
        assert "ASD" in r.primary_strategy or "nanoparticle" in r.primary_strategy.lower()

    def test_bcs_III_primary_strategy_contains_enhancer(self):
        r = recommend_formulation(**_default_kwargs(bcs_class="III"))
        assert "enhancer" in r.primary_strategy.lower() or "prodrug" in r.primary_strategy.lower()

    def test_bcs_IV_primary_strategy_contains_asd_or_enhancer(self):
        r = recommend_formulation(**_default_kwargs(bcs_class="IV"))
        primary = r.primary_strategy.lower()
        assert "asd" in primary or "enhancer" in primary or "lipid" in primary

    def test_extended_release_preferred_for_long_duration(self):
        r = recommend_formulation(**_default_kwargs(target_duration_h=24.0))
        strat = (
            r.primary_strategy + r.secondary_strategy + r.tertiary_strategy
        ).lower()
        assert "er" in strat or "extended" in strat or "mr" in strat

    def test_risk_flags_is_tuple(self):
        r = recommend_formulation(**_default_kwargs())
        assert isinstance(r.risk_flags, tuple)

    def test_very_low_solubility_flag(self):
        r = recommend_formulation(**_default_kwargs(solubility_mg_mL=0.005))
        assert "very_low_solubility_flag" in r.risk_flags

    def test_no_very_low_solubility_flag_normal(self):
        r = recommend_formulation(**_default_kwargs(solubility_mg_mL=1.0))
        assert "very_low_solubility_flag" not in r.risk_flags

    def test_high_lipophilicity_flag(self):
        r = recommend_formulation(**_default_kwargs(logP=6.0))
        assert "high_lipophilicity_flag" in r.risk_flags

    def test_no_high_lipophilicity_flag_normal(self):
        r = recommend_formulation(**_default_kwargs(logP=2.0))
        assert "high_lipophilicity_flag" not in r.risk_flags

    def test_all_four_bcs_classes_work(self):
        for cls in ("I", "II", "III", "IV"):
            r = recommend_formulation(**_default_kwargs(bcs_class=cls))
            assert isinstance(r, FormulationRecommendation)

    def test_rationale_nonempty(self):
        r = recommend_formulation(**_default_kwargs())
        assert len(r.rationale) > 10

    def test_notes_nonempty(self):
        r = recommend_formulation(**_default_kwargs())
        assert len(r.notes) > 0

    def test_three_strategies_returned(self):
        r = recommend_formulation(**_default_kwargs())
        assert r.primary_strategy
        assert r.secondary_strategy
        assert r.tertiary_strategy

    def test_target_tmax_and_duration_preserved(self):
        r = recommend_formulation(**_default_kwargs(target_tmax_h=3.0, target_duration_h=12.0))
        assert abs(r.target_tmax_h - 3.0) < 0.01
        assert abs(r.target_duration_h - 12.0) < 0.01


# ---------------------------------------------------------------------------
# formulation_risk_assessment
# ---------------------------------------------------------------------------


class TestFormulationRiskAssessment:
    def test_returns_dict_with_expected_keys(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 1.0)
        assert "risk_flags" in result
        assert "overall_risk" in result
        assert "rationale" in result

    def test_no_flags_is_low_risk(self):
        # MW=300, logP=2, PSA=80, solubility=1.0 → no flags
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 1.0)
        assert result["overall_risk"] == "low"
        assert result["risk_flags"] == []

    def test_one_flag_is_moderate_risk(self):
        # Only high logP
        result = formulation_risk_assessment("Drug", 300.0, 6.0, 80.0, 1.0)
        assert result["overall_risk"] == "moderate"

    def test_two_flags_is_high_risk(self):
        # Very low solubility + high logP
        result = formulation_risk_assessment("Drug", 300.0, 6.0, 80.0, 0.005)
        assert result["overall_risk"] == "high"

    def test_very_low_solubility_flag_appears(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 0.005)
        assert "very_low_solubility_flag" in result["risk_flags"]

    def test_high_logp_flag_appears(self):
        result = formulation_risk_assessment("Drug", 300.0, 6.0, 80.0, 1.0)
        assert "high_lipophilicity_flag" in result["risk_flags"]

    def test_high_mw_flag_appears(self):
        result = formulation_risk_assessment("Drug", 600.0, 2.0, 80.0, 1.0)
        assert "high_mw_flag" in result["risk_flags"]

    def test_high_psa_flag_appears(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 150.0, 1.0)
        assert "high_psa_flag" in result["risk_flags"]

    def test_rationale_nonempty(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 1.0)
        assert len(result["rationale"]) > 5

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            formulation_risk_assessment("", 300.0, 2.0, 80.0, 1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            formulation_risk_assessment("Drug", 0.0, 2.0, 80.0, 1.0)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            formulation_risk_assessment("Drug", 300.0, 2.0, -1.0, 1.0)

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, -0.1)

    def test_risk_flags_is_list(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 1.0)
        assert isinstance(result["risk_flags"], list)

    def test_overall_risk_valid_values(self):
        result = formulation_risk_assessment("Drug", 300.0, 2.0, 80.0, 1.0)
        assert result["overall_risk"] in ("low", "moderate", "high")
