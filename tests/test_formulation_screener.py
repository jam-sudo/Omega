"""Tests for formulation strategy screener -- Phase 460."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.formulation_screener import (
    FormulationScreenResult,
    estimate_bioavailability_improvement,
    get_formulation_recommendation,
    screen_formulations,
)


# ---------------------------------------------------------------------------
# screen_formulations
# ---------------------------------------------------------------------------


class TestScreenFormulations:
    """Tests for screen_formulations."""

    def test_returns_formulation_screen_result(self):
        result = screen_formulations("DrugA", bcs_class=1, logP=1.5, mw=300.0)
        assert isinstance(result, FormulationScreenResult)

    def test_bcs1_primary_is_simple_tablet(self):
        """BCS I drugs should favour simple tablet/capsule."""
        result = screen_formulations("DrugA", bcs_class=1, logP=1.5, mw=300.0)
        assert result.primary_recommendation == "simple_tablet_capsule"

    def test_bcs2_high_logP_primary_is_lipophilic_strategy(self):
        """BCS II high-logP drugs should get ASD, SEDDS, or lipid-based as primary."""
        result = screen_formulations("DrugB", bcs_class=2, logP=5.5, mw=450.0)
        lipophilic_strategies = {
            "amorphous_solid_dispersion",
            "SEDDS_SMEDDS",
            "lipid_based_formulation",
            "nanosuspension",
        }
        assert result.primary_recommendation in lipophilic_strategies

    def test_bcs4_highest_risk(self):
        """BCS IV should always have 'high' risk level."""
        result = screen_formulations("DrugD", bcs_class=4, logP=2.0, mw=400.0)
        assert result.risk_level == "high"

    def test_bcs1_low_risk(self):
        result = screen_formulations("DrugA", bcs_class=1, logP=1.5, mw=300.0)
        assert result.risk_level == "low"

    def test_strategies_sorted_by_score_descending(self):
        result = screen_formulations("DrugB", bcs_class=2, logP=4.0, mw=400.0)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    def test_scores_in_valid_range(self):
        result = screen_formulations("DrugC", bcs_class=3, logP=0.5, mw=350.0)
        for score in result.scores:
            assert 0.0 <= score <= 100.0

    def test_strategies_and_scores_same_length(self):
        result = screen_formulations("DrugX", bcs_class=2, logP=3.0, mw=380.0)
        assert len(result.strategies) == len(result.scores)

    def test_primary_recommendation_in_strategies(self):
        result = screen_formulations("DrugX", bcs_class=2, logP=3.0, mw=380.0)
        assert result.primary_recommendation == result.strategies[0]

    def test_expected_f_improvement_bounded(self):
        result = screen_formulations("DrugD", bcs_class=4, logP=2.0, mw=400.0)
        assert 0.0 <= result.expected_f_improvement <= 0.95

    def test_drug_name_preserved(self):
        result = screen_formulations("Warfarin", bcs_class=1, logP=2.7, mw=308.0)
        assert result.drug_name == "Warfarin"

    def test_bcs_class_preserved(self):
        result = screen_formulations("DrugB", bcs_class=2, logP=4.0, mw=400.0)
        assert result.bcs_class == 2

    def test_invalid_bcs_class_zero_raises(self):
        with pytest.raises(ValueError):
            screen_formulations("Drug", bcs_class=0, logP=2.0, mw=300.0)

    def test_invalid_bcs_class_five_raises(self):
        with pytest.raises(ValueError):
            screen_formulations("Drug", bcs_class=5, logP=2.0, mw=300.0)

    def test_notes_is_non_empty_string(self):
        result = screen_formulations("DrugA", bcs_class=1, logP=1.5, mw=300.0)
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_solubility_adjusts_asd_score_for_bcs2(self):
        """Very low solubility should boost ASD score for BCS II."""
        result_low_sol = screen_formulations(
            "Drug", bcs_class=2, logP=3.0, mw=350.0, solubility_mg_mL=0.001
        )
        result_normal_sol = screen_formulations(
            "Drug", bcs_class=2, logP=3.0, mw=350.0, solubility_mg_mL=1.0
        )
        idx_low = result_low_sol.strategies.index("amorphous_solid_dispersion")
        idx_normal = result_normal_sol.strategies.index("amorphous_solid_dispersion")
        # ASD should rank higher (lower index) for very low solubility
        assert idx_low <= idx_normal

    def test_bcs3_permeation_enhancer_appears_high(self):
        """For BCS III, permeation_enhancers should score well."""
        result = screen_formulations("DrugC", bcs_class=3, logP=0.5, mw=300.0)
        pe_idx = result.strategies.index("permeation_enhancers")
        # Should be in top 3
        assert pe_idx < 3


# ---------------------------------------------------------------------------
# get_formulation_recommendation
# ---------------------------------------------------------------------------


class TestGetFormulationRecommendation:
    """Tests for get_formulation_recommendation."""

    def test_returns_dict_with_required_keys(self):
        rec = get_formulation_recommendation(bcs_class=1, logP=1.5, mw=300.0)
        assert "primary" in rec
        assert "alternatives" in rec
        assert "rationale" in rec

    def test_primary_is_string(self):
        rec = get_formulation_recommendation(bcs_class=2, logP=4.0, mw=400.0)
        assert isinstance(rec["primary"], str) and len(rec["primary"]) > 0

    def test_alternatives_is_list(self):
        rec = get_formulation_recommendation(bcs_class=2, logP=4.0, mw=400.0)
        assert isinstance(rec["alternatives"], list)

    def test_alternatives_not_include_primary(self):
        rec = get_formulation_recommendation(bcs_class=2, logP=4.0, mw=400.0)
        assert rec["primary"] not in rec["alternatives"]

    def test_rationale_is_non_empty_string(self):
        rec = get_formulation_recommendation(bcs_class=3, logP=0.5, mw=300.0)
        assert isinstance(rec["rationale"], str) and len(rec["rationale"]) > 0

    def test_bcs1_primary_simple_tablet(self):
        rec = get_formulation_recommendation(bcs_class=1, logP=1.5, mw=300.0)
        assert rec["primary"] == "simple_tablet_capsule"

    def test_invalid_bcs_raises(self):
        with pytest.raises(ValueError):
            get_formulation_recommendation(bcs_class=5, logP=2.0, mw=300.0)


# ---------------------------------------------------------------------------
# estimate_bioavailability_improvement
# ---------------------------------------------------------------------------


class TestEstimateBioavailabilityImprovement:
    """Tests for estimate_bioavailability_improvement."""

    def test_improvement_capped_at_095(self):
        """Bioavailability cannot exceed 0.95."""
        f = estimate_bioavailability_improvement(
            current_f=0.90,
            formulation_strategy="amorphous_solid_dispersion",
            bcs_class=2,
            logP=4.0,
        )
        assert f <= 0.95

    def test_no_improvement_simple_tablet_bcs1(self):
        """Simple tablet for BCS I should not change F substantially."""
        f = estimate_bioavailability_improvement(
            current_f=0.80,
            formulation_strategy="simple_tablet_capsule",
            bcs_class=1,
            logP=2.0,
        )
        assert abs(f - 0.80) < 0.05

    def test_asd_improves_bcs2_bioavailability(self):
        """ASD should improve bioavailability for BCS II drugs."""
        f = estimate_bioavailability_improvement(
            current_f=0.30,
            formulation_strategy="amorphous_solid_dispersion",
            bcs_class=2,
            logP=4.0,
        )
        assert f > 0.30

    def test_returns_float(self):
        f = estimate_bioavailability_improvement(
            current_f=0.50,
            formulation_strategy="nanosuspension",
            bcs_class=2,
            logP=3.0,
        )
        assert isinstance(f, float)

    def test_invalid_current_f_above_one(self):
        with pytest.raises(ValueError):
            estimate_bioavailability_improvement(
                current_f=1.1,
                formulation_strategy="simple_tablet_capsule",
                bcs_class=1,
                logP=2.0,
            )

    def test_invalid_current_f_negative(self):
        with pytest.raises(ValueError):
            estimate_bioavailability_improvement(
                current_f=-0.1,
                formulation_strategy="simple_tablet_capsule",
                bcs_class=1,
                logP=2.0,
            )

    def test_invalid_bcs_class_raises(self):
        with pytest.raises(ValueError):
            estimate_bioavailability_improvement(
                current_f=0.50,
                formulation_strategy="simple_tablet_capsule",
                bcs_class=0,
                logP=2.0,
            )

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown formulation_strategy"):
            estimate_bioavailability_improvement(
                current_f=0.50,
                formulation_strategy="magic_pills",
                bcs_class=1,
                logP=2.0,
            )

    def test_result_at_least_current_f_for_beneficial_strategy(self):
        """Any improvement strategy for BCS 2 should not decrease F."""
        for strat in ["amorphous_solid_dispersion", "nanosuspension", "SEDDS_SMEDDS"]:
            f = estimate_bioavailability_improvement(
                current_f=0.30,
                formulation_strategy=strat,
                bcs_class=2,
                logP=3.5,
            )
            assert f >= 0.30, f"{strat} reduced bioavailability"
