"""Tests for protein/mAb drug formulation optimization — Phase 507."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.protein_formulation import (
    ExcipientCompatibilityResult,
    FormulationRecommendationResult,
    formulation_recommendation,
    optimize_buffer_system,
    predict_aggregation_propensity,
    predict_excipient_compatibility,
)

# ---------------------------------------------------------------------------
# predict_excipient_compatibility
# ---------------------------------------------------------------------------


class TestPredictExcipientCompatibility:
    def test_returns_correct_type(self):
        result = predict_excipient_compatibility("mAb-A", 7.0, 150.0)
        assert isinstance(result, ExcipientCompatibilityResult)

    def test_compatibility_score_in_range(self):
        result = predict_excipient_compatibility("mAb-A", 7.0, 150.0)
        assert 0.0 <= result.compatibility_score <= 100.0

    def test_recommended_stabilizers_non_empty(self):
        result = predict_excipient_compatibility("mAb-A", 7.0, 150.0, ph=7.4)
        assert len(result.recommended_stabilizers) > 0

    def test_near_pi_high_aggregation_risk_sucrose_added(self):
        # pH very close to pI → near pI → sucrose + polysorbate should be in stabilizers
        result = predict_excipient_compatibility("mAb-B", 7.4, 100.0, ph=7.4)
        stabilizers = result.recommended_stabilizers
        assert "sucrose" in stabilizers

    def test_far_from_pi_lower_aggregation_risk(self):
        # pH 5, pI 9: distance 4 → good charge separation
        result = predict_excipient_compatibility("mAb-C", 9.0, 100.0, ph=5.0)
        assert result.compatibility_score > 60.0

    def test_near_pi_polysorbate_added(self):
        result = predict_excipient_compatibility("mAb-D", 7.5, 100.0, ph=7.4)
        assert "polysorbate 80" in result.recommended_stabilizers

    def test_high_mw_trehalose_or_sucrose_added(self):
        result = predict_excipient_compatibility("LargeMab", 7.0, 200.0, ph=5.0)
        stabilizers = result.recommended_stabilizers
        assert "trehalose" in stabilizers or "sucrose" in stabilizers

    def test_ph_4_to_6_histidine_or_acetate_buffer(self):
        result = predict_excipient_compatibility("mAb-E", 9.0, 150.0, ph=5.0)
        buffers = result.recommended_buffers
        assert any(b in buffers for b in ("histidine", "acetate"))

    def test_ph_6_to_8_phosphate_or_histidine_buffer(self):
        result = predict_excipient_compatibility("mAb-F", 9.5, 150.0, ph=7.0)
        buffers = result.recommended_buffers
        assert any(b in buffers for b in ("phosphate", "histidine"))

    def test_ph_above_8_tris_buffer(self):
        result = predict_excipient_compatibility("mAb-G", 5.0, 150.0, ph=8.5)
        buffers = result.recommended_buffers
        assert "Tris" in buffers

    def test_net_charge_positive_when_ph_below_pi(self):
        result = predict_excipient_compatibility("mAb-H", 9.0, 100.0, ph=5.0)
        assert result.net_charge > 0.0

    def test_net_charge_negative_when_ph_above_pi(self):
        result = predict_excipient_compatibility("mAb-I", 5.0, 100.0, ph=7.4)
        assert result.net_charge < 0.0

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            predict_excipient_compatibility("X", 7.0, 0.0)

    def test_invalid_pi_raises(self):
        with pytest.raises(ValueError, match="isoelectric_point"):
            predict_excipient_compatibility("X", -1.0, 150.0)

    def test_notes_non_empty(self):
        result = predict_excipient_compatibility("mAb-J", 7.0, 150.0)
        assert len(result.notes) > 0

    def test_protein_name_preserved(self):
        result = predict_excipient_compatibility("TestProtein", 7.0, 150.0)
        assert result.protein_name == "TestProtein"


# ---------------------------------------------------------------------------
# optimize_buffer_system
# ---------------------------------------------------------------------------


class TestOptimizeBufferSystem:
    def test_returns_dict(self):
        result = optimize_buffer_system(7.4, 8.0)
        assert isinstance(result, dict)

    def test_target_ph_7_4_recommends_phosphate(self):
        result = optimize_buffer_system(7.4, 9.0)
        assert result["primary_buffer"] == "phosphate"

    def test_target_ph_5_recommends_histidine_or_acetate(self):
        result = optimize_buffer_system(5.0, 9.0)
        assert result["primary_buffer"] in ("histidine", "acetate")

    def test_target_ph_above_8_recommends_tris(self):
        result = optimize_buffer_system(8.5, 5.0)
        assert result["primary_buffer"] == "Tris"

    def test_near_pi_warning_in_recommendation(self):
        result = optimize_buffer_system(7.0, 7.2)
        assert "aggregation" in result.get("ph_recommendation", "").lower()

    def test_invalid_target_ph_raises(self):
        with pytest.raises(ValueError, match="target_ph"):
            optimize_buffer_system(-1.0, 7.0)

    def test_invalid_protein_pi_raises(self):
        with pytest.raises(ValueError, match="protein_pi"):
            optimize_buffer_system(7.4, 0.0)

    def test_outside_stability_range_adds_warning(self):
        result = optimize_buffer_system(2.0, 7.0, stability_range=(4.0, 9.0))
        assert "warning" in result


# ---------------------------------------------------------------------------
# predict_aggregation_propensity
# ---------------------------------------------------------------------------


class TestPredictAggregationPropensity:
    def test_near_pi_high_score(self):
        score = predict_aggregation_propensity(150.0, 7.4, ph=7.4)
        assert score > 0.6

    def test_far_from_pi_low_score(self):
        score = predict_aggregation_propensity(150.0, 9.5, ph=5.0)
        assert score < 0.5

    def test_score_in_range(self):
        score = predict_aggregation_propensity(150.0, 7.0, ph=7.4)
        assert 0.0 <= score <= 1.0

    def test_high_temperature_increases_score(self):
        low = predict_aggregation_propensity(150.0, 9.0, ph=5.0, temperature_C=4.0)
        high = predict_aggregation_propensity(150.0, 9.0, ph=5.0, temperature_C=40.0)
        assert high > low

    def test_high_concentration_increases_score(self):
        low = predict_aggregation_propensity(150.0, 9.0, ph=5.0, concentration_mg_mL=1.0)
        high = predict_aggregation_propensity(150.0, 9.0, ph=5.0, concentration_mg_mL=100.0)
        assert high > low

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            predict_aggregation_propensity(0.0, 7.0)

    def test_invalid_pi_raises(self):
        with pytest.raises(ValueError, match="isoelectric_point"):
            predict_aggregation_propensity(150.0, 0.0)


# ---------------------------------------------------------------------------
# formulation_recommendation
# ---------------------------------------------------------------------------


class TestFormulationRecommendation:
    def test_returns_correct_type(self):
        result = formulation_recommendation("mAb-A", 150.0, 7.0, 10.0, route="iv")
        assert isinstance(result, FormulationRecommendationResult)

    def test_sc_high_conc_viscosity_strategy(self):
        result = formulation_recommendation("mAb-B", 150.0, 8.5, 100.0, route="sc")
        assert "viscosity" in result.viscosity_strategy.lower()
        assert "hyaluronidase" in result.viscosity_strategy.lower()

    def test_iv_route_viscosity_message(self):
        result = formulation_recommendation("mAb-C", 150.0, 7.0, 10.0, route="iv")
        assert "IV" in result.viscosity_strategy

    def test_overall_score_in_range(self):
        result = formulation_recommendation("mAb-D", 150.0, 8.5, 10.0, route="iv")
        assert 0.0 <= result.overall_score <= 100.0

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            formulation_recommendation("X", 150.0, 7.0, 10.0, route="oral")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            formulation_recommendation("X", 0.0, 7.0, 10.0)

    def test_invalid_pi_raises(self):
        with pytest.raises(ValueError, match="pi"):
            formulation_recommendation("X", 150.0, 0.0, 10.0)

    def test_invalid_concentration_raises(self):
        with pytest.raises(ValueError, match="target_concentration"):
            formulation_recommendation("X", 150.0, 7.0, 0.0)

    def test_stabilizers_non_empty(self):
        result = formulation_recommendation("mAb-E", 150.0, 7.0, 10.0)
        assert len(result.stabilizers) > 0

    def test_sc_route_preserved(self):
        result = formulation_recommendation("mAb-F", 150.0, 7.0, 10.0, route="sc")
        assert result.route == "sc"
