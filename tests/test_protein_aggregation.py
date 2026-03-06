"""Tests for protein aggregation and amyloid prediction module."""

import pytest

from omega_pbpk.risk.protein_aggregation import (
    ProteinAggregationResult,
    assess_protein_aggregation,
    optimize_formulation,
)


def _default():
    return assess_protein_aggregation("mAb-X", mw_kDa=150.0)


class TestAssessProteinAggregation:
    def test_returns_result(self):
        r = _default()
        assert isinstance(r, ProteinAggregationResult)

    def test_mw_kda_zero_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=0)

    def test_mw_kda_negative_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=-1)

    def test_concentration_zero_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=150, concentration_mg_mL=0)

    def test_concentration_negative_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=150, concentration_mg_mL=-5)

    def test_ph_zero_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=150, ph=0)

    def test_ph_14_raises(self):
        with pytest.raises(ValueError):
            assess_protein_aggregation("x", mw_kDa=150, ph=14)

    def test_score_in_range(self):
        r = _default()
        assert 0 <= r.aggregation_propensity_score <= 100

    def test_risk_valid(self):
        r = _default()
        assert r.aggregation_risk in {"low", "moderate", "high"}

    def test_high_conc_higher_score(self):
        low = assess_protein_aggregation("x", mw_kDa=150, concentration_mg_mL=10)
        high = assess_protein_aggregation("x", mw_kDa=150, concentration_mg_mL=100)
        assert high.aggregation_propensity_score > low.aggregation_propensity_score

    def test_ph_far_from_neutral_higher_score(self):
        neutral = assess_protein_aggregation("x", mw_kDa=150, ph=7.0)
        acidic = assess_protein_aggregation("x", mw_kDa=150, ph=4.0)
        assert acidic.aggregation_propensity_score > neutral.aggregation_propensity_score

    def test_sucrose_reduces_score(self):
        no_exc = assess_protein_aggregation("x", mw_kDa=150)
        with_suc = assess_protein_aggregation("x", mw_kDa=150, excipients=["sucrose"])
        assert with_suc.aggregation_propensity_score < no_exc.aggregation_propensity_score

    def test_thermal_stability_positive(self):
        r = _default()
        assert r.thermal_stability_proxy > 0

    def test_solubility_positive(self):
        r = _default()
        assert r.solubility_proxy_mg_mL > 0

    def test_shelf_life_4c_gt_25c(self):
        r = _default()
        assert r.shelf_life_months_at_4C > r.shelf_life_months_at_25C

    def test_recommended_ph_range_tuple(self):
        r = _default()
        assert isinstance(r.recommended_ph_range, tuple)
        assert len(r.recommended_ph_range) == 2
        assert all(isinstance(v, float) for v in r.recommended_ph_range)

    def test_formulation_recommendation_nonempty(self):
        r = _default()
        assert isinstance(r.formulation_recommendation, str)
        assert len(r.formulation_recommendation) > 0

    def test_iep_proxy_positive(self):
        r = _default()
        assert r.iep_proxy > 0

    def test_high_temperature_higher_score(self):
        t25 = assess_protein_aggregation("x", mw_kDa=150, temperature_C=25)
        t50 = assess_protein_aggregation("x", mw_kDa=150, temperature_C=50)
        assert t50.aggregation_propensity_score > t25.aggregation_propensity_score

    def test_excipients_preserved(self):
        r = assess_protein_aggregation("x", mw_kDa=150, excipients=["sucrose", "arginine"])
        assert r.excipients == ("sucrose", "arginine")

    def test_notes_contains_name(self):
        r = assess_protein_aggregation("TestProtein", mw_kDa=150)
        assert "TestProtein" in r.notes


class TestOptimizeFormulation:
    def test_returns_6_results(self):
        results = optimize_formulation("mAb", mw_kDa=150, concentration_mg_mL=10)
        assert len(results) == 6

    def test_sorted_by_score_ascending(self):
        results = optimize_formulation("mAb", mw_kDa=150, concentration_mg_mL=10)
        scores = [r.aggregation_propensity_score for r in results]
        assert scores == sorted(scores)
