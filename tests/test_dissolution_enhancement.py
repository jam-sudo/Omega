"""Tests for biopharmaceutics/dissolution_enhancement.py — Phase 454."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.dissolution_enhancement import (
    DissolutionEnhancementResult,
    cyclodextrin_complexation,
    predict_enhancement,
    rank_strategies,
)

# Reference drug profile: BCS II, high logP
_BCS2_LOGP = 4.5
_BCS2_MW = 400.0
_BCS2_SOL = 0.01  # mg/mL — poorly soluble

# BCS I drug: high solubility
_BCS1_LOGP = 1.0
_BCS1_MW = 300.0
_BCS1_SOL = 5.0  # mg/mL — readily soluble

_ALL_STRATEGIES = [
    "micronization",
    "nanosuspension",
    "solid_dispersion",
    "cyclodextrin",
    "self_emulsifying",
    "cocrystal",
]


class TestPredictEnhancement:
    def test_returns_dissolution_result(self):
        r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", "solid_dispersion")
        assert isinstance(r, DissolutionEnhancementResult)

    def test_enhancement_factor_positive_all_strategies(self):
        for strategy in _ALL_STRATEGIES:
            r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", strategy)
            assert r.enhancement_factor > 0, f"{strategy} should have positive factor"

    def test_strategy_stored_correctly(self):
        r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", "micronization")
        assert r.strategy == "micronization"

    def test_drug_logP_stored(self):
        r = predict_enhancement(3.5, 350.0, 0.05, "II", "nanosuspension")
        assert r.drug_logP == 3.5

    def test_drug_mw_stored(self):
        r = predict_enhancement(3.5, 350.0, 0.05, "II", "nanosuspension")
        assert r.drug_mw == 350.0

    def test_mechanism_non_empty(self):
        for strategy in _ALL_STRATEGIES:
            r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", strategy)
            assert r.mechanism, f"{strategy} mechanism should not be empty"

    def test_solid_dispersion_high_for_bcs2_high_logp(self):
        """Solid dispersion should give highest factor for BCS II high logP drug."""
        r = predict_enhancement(5.0, 400.0, 0.01, "II", "solid_dispersion")
        assert r.enhancement_factor >= 3.0

    def test_solid_dispersion_recommended_for_bcs2_high_logp(self):
        r = predict_enhancement(5.0, 400.0, 0.01, "II", "solid_dispersion")
        assert r.recommended is True

    def test_bcs1_all_strategies_low_benefit(self):
        """BCS I drug with high solubility should have low enhancement factors."""
        for strategy in _ALL_STRATEGIES:
            r = predict_enhancement(_BCS1_LOGP, _BCS1_MW, _BCS1_SOL, "I", strategy)
            assert r.enhancement_factor < 2.0, (
                f"{strategy} should have low enhancement for BCS I drug, "
                f"got {r.enhancement_factor}"
            )

    def test_bcs1_strategies_not_recommended(self):
        """BCS I drug: none of the strategies should be recommended."""
        for strategy in _ALL_STRATEGIES:
            r = predict_enhancement(_BCS1_LOGP, _BCS1_MW, _BCS1_SOL, "I", strategy)
            assert r.recommended is False, (
                f"{strategy} should not be recommended for BCS I drug"
            )

    def test_self_emulsifying_good_for_very_high_logp(self):
        r = predict_enhancement(6.0, 500.0, 0.005, "II", "self_emulsifying")
        assert r.enhancement_factor >= 3.0
        assert r.recommended is True

    def test_absorption_improvement_between_0_and_1(self):
        for strategy in _ALL_STRATEGIES:
            r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", strategy)
            assert 0.0 <= r.absorption_improvement <= 1.0

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", "magic_pill")

    def test_case_insensitive_strategy(self):
        """Strategy matching should be case-insensitive."""
        r = predict_enhancement(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II", "Micronization")
        assert r.strategy == "micronization"

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="drug_mw"):
            predict_enhancement(3.0, 0.0, 0.1, "II", "micronization")

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="drug_solubility_mg_mL"):
            predict_enhancement(3.0, 400.0, -1.0, "II", "micronization")

    def test_nanosuspension_scales_with_logp(self):
        """Higher logP → higher nanosuspension enhancement for BCS II."""
        r_low = predict_enhancement(2.0, 400.0, 0.01, "II", "nanosuspension")
        r_high = predict_enhancement(5.0, 400.0, 0.01, "II", "nanosuspension")
        assert r_high.enhancement_factor >= r_low.enhancement_factor


class TestRankStrategies:
    def test_returns_all_six_strategies(self):
        results = rank_strategies(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II")
        assert len(results) == 6

    def test_sorted_by_enhancement_factor_descending(self):
        results = rank_strategies(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II")
        factors = [r.enhancement_factor for r in results]
        assert factors == sorted(factors, reverse=True)

    def test_all_strategies_represented(self):
        results = rank_strategies(_BCS2_LOGP, _BCS2_MW, _BCS2_SOL, "II")
        strategies_in_result = {r.strategy for r in results}
        assert strategies_in_result == set(_ALL_STRATEGIES)

    def test_solid_dispersion_top_for_high_logp_bcs2(self):
        """For BCS II high-logP drug, solid_dispersion or SEDDS should be top."""
        results = rank_strategies(5.0, 400.0, 0.01, "II")
        top_strategy = results[0].strategy
        assert top_strategy in ("solid_dispersion", "self_emulsifying")

    def test_bcs1_results_all_low(self):
        results = rank_strategies(_BCS1_LOGP, _BCS1_MW, _BCS1_SOL, "I")
        assert all(r.enhancement_factor < 2.0 for r in results)


class TestCyclodextrinComplexation:
    def test_returns_dict_with_required_keys(self):
        result = cyclodextrin_complexation(3.0, 350.0)
        assert "complexation_efficiency" in result
        assert "solubility_fold_increase" in result
        assert "notes" in result

    def test_high_logp_gives_higher_complexation(self):
        """Higher logP → better hydrophobic CD complexation (within MW range)."""
        r_low = cyclodextrin_complexation(1.0, 350.0)
        r_high = cyclodextrin_complexation(4.0, 350.0)
        assert r_high["complexation_efficiency"] > r_low["complexation_efficiency"]

    def test_complexation_efficiency_between_0_and_1(self):
        for logp in [0.5, 2.0, 4.0, 6.0]:
            r = cyclodextrin_complexation(logp, 350.0)
            assert 0.0 <= r["complexation_efficiency"] <= 1.0

    def test_solubility_fold_increase_at_least_1(self):
        r = cyclodextrin_complexation(3.0, 350.0)
        assert r["solubility_fold_increase"] >= 1.0

    def test_mw_out_of_range_lowers_efficiency(self):
        """MW=1200 (too large) should give lower efficiency than MW=350."""
        r_ok = cyclodextrin_complexation(4.0, 350.0)
        r_big = cyclodextrin_complexation(4.0, 1200.0)
        assert r_big["complexation_efficiency"] < r_ok["complexation_efficiency"]

    def test_default_cd_type_hpbetacd(self):
        """Default cd_type should be HP-beta-CD."""
        r = cyclodextrin_complexation(3.0, 350.0)
        # Should not raise; verify result is valid
        assert r["complexation_efficiency"] > 0

    def test_beta_cd_type_accepted(self):
        r = cyclodextrin_complexation(3.0, 300.0, cd_type="beta-CD")
        assert r["complexation_efficiency"] > 0

    def test_gamma_cd_type_accepted(self):
        r = cyclodextrin_complexation(3.0, 500.0, cd_type="gamma-CD")
        assert r["complexation_efficiency"] > 0

    def test_invalid_cd_type_raises(self):
        with pytest.raises(ValueError, match="cd_type"):
            cyclodextrin_complexation(3.0, 350.0, cd_type="theta-CD")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="drug_mw"):
            cyclodextrin_complexation(3.0, 0.0)

    def test_notes_non_empty(self):
        r = cyclodextrin_complexation(3.0, 350.0)
        assert len(r["notes"]) > 0
