"""Tests for Phase 458 — LipophilicityOptimizer."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.lipophilicity_optimizer import (
    LogPModificationResult,
    optimal_logp_range,
    predict_pk_impact_of_logp,
    score_logp_modification,
)


# ---------------------------------------------------------------------------
# optimal_logp_range
# ---------------------------------------------------------------------------

class TestOptimalLogpRange:
    def test_returns_tuple(self):
        result = optimal_logp_range("systemic", "oral")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_min_less_than_max(self):
        for site in ("systemic", "cns", "topical", "inhaled", "oral_absorption"):
            lo, hi = optimal_logp_range(site, "oral")
            assert lo < hi, f"min >= max for site={site}"

    def test_systemic_oral_range(self):
        lo, hi = optimal_logp_range("systemic", "oral")
        assert lo == 1.0
        assert hi == 3.0

    def test_systemic_iv_range(self):
        lo, hi = optimal_logp_range("systemic", "iv")
        assert lo == 0.0
        assert hi == 2.0

    def test_cns_range(self):
        lo, hi = optimal_logp_range("cns", "oral")
        assert lo == 1.0
        assert hi == 3.0

    def test_topical_range(self):
        lo, hi = optimal_logp_range("topical", "topical")
        assert lo == 1.0
        assert hi == 4.0

    def test_inhaled_range(self):
        lo, hi = optimal_logp_range("inhaled", "inhaled")
        assert lo == 0.0
        assert hi == 2.0

    def test_oral_absorption_range(self):
        lo, hi = optimal_logp_range("oral_absorption", "oral")
        assert lo == 0.0
        assert hi == 5.0

    def test_invalid_target_site_raises(self):
        with pytest.raises(ValueError, match="Invalid target_site"):
            optimal_logp_range("blood", "oral")


# ---------------------------------------------------------------------------
# score_logp_modification
# ---------------------------------------------------------------------------

class TestScoreLogpModification:
    def test_returns_result_dataclass(self):
        result = score_logp_modification(5.0, 2.0)
        assert isinstance(result, LogPModificationResult)

    def test_moving_from_high_to_optimal_oral_is_favorable(self):
        """logP 5 → 2 for oral/systemic should be favorable."""
        result = score_logp_modification(5.0, 2.0, target_site="systemic", route="oral")
        assert result.overall_recommendation == "favorable"

    def test_moving_to_very_high_logp_is_unfavorable(self):
        """logP 5 → 7 for oral should be unfavorable."""
        result = score_logp_modification(5.0, 7.0, target_site="systemic", route="oral")
        assert result.overall_recommendation == "unfavorable"

    def test_score_delta_in_range(self):
        for current, proposed in [(0.0, 2.0), (2.0, 5.0), (1.0, 6.0), (3.0, 1.0)]:
            result = score_logp_modification(current, proposed)
            assert -1.0 <= result.score_delta <= 1.0, (
                f"score_delta={result.score_delta} out of range for {current}→{proposed}"
            )

    def test_delta_logp_is_correct(self):
        result = score_logp_modification(3.0, 1.5)
        assert abs(result.delta_logP - (-1.5)) < 1e-9

    def test_current_proposed_stored_correctly(self):
        result = score_logp_modification(4.0, 2.0)
        assert result.current_logP == 4.0
        assert result.proposed_logP == 2.0

    def test_cns_optimal_logp_returns_favorable(self):
        """Moving into CNS optimal range (1-3) should be favorable."""
        result = score_logp_modification(5.0, 2.0, target_site="cns", route="oral")
        assert result.overall_recommendation in ("favorable", "neutral")

    def test_cns_already_optimal_no_change_neutral(self):
        """Already optimal logP with no change."""
        result = score_logp_modification(2.0, 2.0, target_site="cns", route="oral")
        assert result.overall_recommendation in ("neutral", "favorable")
        assert abs(result.score_delta) < 0.2

    def test_absorption_change_label_type(self):
        result = score_logp_modification(1.0, 3.0)
        assert result.absorption_change in ("improve", "neutral", "worsen")

    def test_distribution_change_label_type(self):
        result = score_logp_modification(1.0, 3.0)
        assert result.distribution_change in ("improve", "neutral", "worsen")

    def test_clearance_change_label_type(self):
        result = score_logp_modification(1.0, 3.0)
        assert result.clearance_change in ("improve", "neutral", "worsen")

    def test_invalid_target_site_raises(self):
        with pytest.raises(ValueError, match="Invalid target_site"):
            score_logp_modification(2.0, 3.0, target_site="plasma")

    def test_result_is_frozen(self):
        """LogPModificationResult must be immutable (frozen=True)."""
        result = score_logp_modification(2.0, 3.0)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.score_delta = 0.99  # type: ignore[misc]

    def test_moving_logp_within_optimal_range_neutral_or_favorable(self):
        """Both 1.5 and 2.5 are in the oral optimal range; delta should be small."""
        result = score_logp_modification(1.5, 2.5, target_site="systemic", route="oral")
        assert abs(result.score_delta) < 0.5


# ---------------------------------------------------------------------------
# predict_pk_impact_of_logp
# ---------------------------------------------------------------------------

class TestPredictPkImpactOfLogP:
    def test_returns_dict_with_keys_for_each_logp(self):
        logP_values = [1.0, 2.0, 3.0]
        result = predict_pk_impact_of_logp(logP_values, mw=350.0)
        assert set(result.keys()) == {1.0, 2.0, 3.0}

    def test_each_entry_has_required_keys(self):
        result = predict_pk_impact_of_logp([2.0], mw=300.0)
        entry = result[2.0]
        for key in ("absorption", "distribution", "clearance", "overall_score"):
            assert key in entry, f"missing key: {key}"

    def test_overall_score_in_zero_one(self):
        for logP in [-2.0, 0.0, 2.0, 5.0, 8.0]:
            result = predict_pk_impact_of_logp([logP], mw=400.0)
            score = result[logP]["overall_score"]
            assert 0.0 <= score <= 1.0, f"out of range for logP={logP}"

    def test_low_logp_poor_oral_absorption(self):
        """Very negative logP → low absorption score for oral route."""
        result = predict_pk_impact_of_logp([-5.0], mw=300.0, route="oral")
        assert result[-5.0]["absorption"] < 0.7

    def test_high_logp_lower_clearance_score(self):
        """High logP → lower clearance score (high hepatic extraction risk)."""
        low_result = predict_pk_impact_of_logp([1.0], mw=300.0)
        high_result = predict_pk_impact_of_logp([6.0], mw=300.0)
        assert high_result[6.0]["clearance"] < low_result[1.0]["clearance"]

    def test_empty_logp_values_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            predict_pk_impact_of_logp([], mw=350.0)

    def test_mw_penalty_applied_for_large_mw(self):
        """High MW compound should have lower overall score than low MW."""
        low_mw = predict_pk_impact_of_logp([2.0], mw=200.0)[2.0]["overall_score"]
        high_mw = predict_pk_impact_of_logp([2.0], mw=800.0)[2.0]["overall_score"]
        assert high_mw < low_mw
