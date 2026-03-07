"""Tests for clinical/half_life_optimization.py — Phase 331."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.half_life_optimization import (
    HalfLifeTargetResult,
    OptimalRegimenResult,
    design_t_half_strategies,
    estimate_t_half,
    metabolic_stability_t_half_target,
    optimal_dosing_regimen,
    t_half_from_targets,
)

# ---------------------------------------------------------------------------
# estimate_t_half
# ---------------------------------------------------------------------------


class TestEstimateTHalf:
    def test_basic_formula(self):
        # t_half = 0.693 * Vd / CL
        result = estimate_t_half(cl_L_per_h=10.0, vd_L=100.0)
        assert math.isclose(result, 0.693 * 100.0 / 10.0, rel_tol=1e-6)

    def test_units_consistency(self):
        # doubling CL halves t_half
        t1 = estimate_t_half(cl_L_per_h=5.0, vd_L=50.0)
        t2 = estimate_t_half(cl_L_per_h=10.0, vd_L=50.0)
        assert math.isclose(t1, 2.0 * t2, rel_tol=1e-6)

    def test_doubling_vd_doubles_t_half(self):
        t1 = estimate_t_half(cl_L_per_h=5.0, vd_L=50.0)
        t2 = estimate_t_half(cl_L_per_h=5.0, vd_L=100.0)
        assert math.isclose(t2, 2.0 * t1, rel_tol=1e-6)

    def test_typical_small_molecule(self):
        # CL=30 L/h, Vd=200 L → t_half ≈ 4.62 h
        result = estimate_t_half(cl_L_per_h=30.0, vd_L=200.0)
        assert 4.0 < result < 5.5

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            estimate_t_half(cl_L_per_h=0.0, vd_L=100.0)

    def test_invalid_cl_negative(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            estimate_t_half(cl_L_per_h=-1.0, vd_L=100.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            estimate_t_half(cl_L_per_h=5.0, vd_L=0.0)

    def test_invalid_vd_negative(self):
        with pytest.raises(ValueError, match="vd_L"):
            estimate_t_half(cl_L_per_h=5.0, vd_L=-10.0)

    def test_returns_float(self):
        assert isinstance(estimate_t_half(1.0, 1.0), float)


# ---------------------------------------------------------------------------
# t_half_from_targets
# ---------------------------------------------------------------------------


class TestTHalfFromTargets:
    def test_once_daily_typical(self):
        # tau=24h, R=2 → t_half should be positive and in reasonable range
        result = t_half_from_targets(target_dosing_interval_h=24.0, desired_accumulation_ratio=2.0)
        assert result > 0

    def test_higher_ratio_longer_t_half(self):
        # Higher accumulation ratio → longer required t_half (less elimination per dose interval)
        t1 = t_half_from_targets(24.0, 1.5)
        t2 = t_half_from_targets(24.0, 3.0)
        assert t2 > t1

    def test_longer_interval_longer_t_half(self):
        # Same R, longer tau → longer t_half
        t1 = t_half_from_targets(12.0, 2.0)
        t2 = t_half_from_targets(24.0, 2.0)
        assert t2 > t1

    def test_roundtrip_accumulation(self):
        # Verify that the resulting t_half actually gives the target accumulation ratio
        tau = 24.0
        R_target = 2.0
        t_half = t_half_from_targets(tau, R_target)
        ke = 0.693 / t_half
        R_check = 1.0 / (1.0 - math.exp(-ke * tau))
        assert math.isclose(R_check, R_target, rel_tol=1e-5)

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError, match="target_dosing_interval_h"):
            t_half_from_targets(0.0, 2.0)

    def test_invalid_ratio_one(self):
        with pytest.raises(ValueError, match="desired_accumulation_ratio"):
            t_half_from_targets(24.0, 1.0)

    def test_invalid_ratio_less_than_one(self):
        with pytest.raises(ValueError, match="desired_accumulation_ratio"):
            t_half_from_targets(24.0, 0.5)

    def test_returns_float(self):
        assert isinstance(t_half_from_targets(24.0, 2.0), float)


# ---------------------------------------------------------------------------
# metabolic_stability_t_half_target
# ---------------------------------------------------------------------------


class TestMetabolicStabilityTHalfTarget:
    def test_oral_once_daily(self):
        result = metabolic_stability_t_half_target("oral", "once_daily")
        assert isinstance(result, HalfLifeTargetResult)
        assert result.recommended_t_half_min_h == 12.0
        assert result.recommended_t_half_max_h == 24.0

    def test_oral_twice_daily(self):
        result = metabolic_stability_t_half_target("oral", "twice_daily")
        assert result.recommended_t_half_min_h == 6.0
        assert result.recommended_t_half_max_h == 12.0

    def test_iv_once_daily(self):
        result = metabolic_stability_t_half_target("iv", "once_daily")
        assert result.recommended_t_half_min_h == 18.0
        assert result.recommended_t_half_max_h == 30.0

    def test_long_acting_injection(self):
        result = metabolic_stability_t_half_target("long_acting_injection", "once_daily")
        assert result.recommended_t_half_min_h == 120.0
        assert result.recommended_t_half_max_h == 336.0

    def test_route_stored(self):
        result = metabolic_stability_t_half_target("oral", "once_daily")
        assert result.route == "oral"
        assert result.dosing_frequency == "once_daily"

    def test_rationale_nonempty(self):
        result = metabolic_stability_t_half_target("oral", "once_daily")
        assert len(result.rationale) > 10

    def test_frozen_dataclass(self):
        result = metabolic_stability_t_half_target("oral", "once_daily")
        with pytest.raises((AttributeError, TypeError)):
            result.route = "iv"  # type: ignore[misc]

    def test_unknown_route_returns_default(self):
        result = metabolic_stability_t_half_target("rectal", "once_daily")
        assert isinstance(result, HalfLifeTargetResult)
        assert result.recommended_t_half_min_h > 0

    def test_min_less_than_max(self):
        for route, freq in [("oral", "once_daily"), ("oral", "twice_daily"), ("iv", "once_daily")]:
            result = metabolic_stability_t_half_target(route, freq)
            assert result.recommended_t_half_min_h < result.recommended_t_half_max_h


# ---------------------------------------------------------------------------
# design_t_half_strategies
# ---------------------------------------------------------------------------


class TestDesignTHalfStrategies:
    def test_longer_target_returns_list(self):
        strategies = design_t_half_strategies(current_t_half_h=4.0, target_t_half_h=24.0)
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_longer_target_mentions_reduce_cl(self):
        strategies = design_t_half_strategies(4.0, 24.0)
        combined = " ".join(strategies).lower()
        # Should mention strategies to extend half-life
        assert any(word in combined for word in ["metabolic", "clearance", "lipophil", "vd", "peg"])

    def test_shorter_target_returns_list(self):
        strategies = design_t_half_strategies(current_t_half_h=24.0, target_t_half_h=4.0)
        assert len(strategies) > 0

    def test_shorter_target_mentions_increase_cl(self):
        strategies = design_t_half_strategies(24.0, 4.0)
        combined = " ".join(strategies).lower()
        assert any(word in combined for word in ["metabolic", "clearance", "soft", "labil"])

    def test_equal_target_no_change_needed(self):
        strategies = design_t_half_strategies(12.0, 12.0)
        assert len(strategies) >= 1
        combined = " ".join(strategies).lower()
        assert "already" in combined or "no structural" in combined or "target" in combined

    def test_large_extension_mentions_pegylation(self):
        strategies = design_t_half_strategies(1.0, 200.0)
        combined = " ".join(strategies).lower()
        assert "peg" in combined or "depot" in combined or "fusion" in combined

    def test_invalid_current_zero(self):
        with pytest.raises(ValueError, match="current_t_half_h"):
            design_t_half_strategies(0.0, 12.0)

    def test_invalid_target_negative(self):
        with pytest.raises(ValueError, match="target_t_half_h"):
            design_t_half_strategies(12.0, -1.0)


# ---------------------------------------------------------------------------
# optimal_dosing_regimen
# ---------------------------------------------------------------------------


class TestOptimalDosingRegimen:
    def _run(self, t_half=12.0, window=(1.0, 10.0), dose=100.0, vd=50.0, cl=5.0):
        return optimal_dosing_regimen(
            t_half_h=t_half,
            therapeutic_window=window,
            dose_mg=dose,
            vd_L=vd,
            cl_L_per_h=cl,
        )

    def test_returns_result_type(self):
        result = self._run()
        assert isinstance(result, OptimalRegimenResult)

    def test_frozen_dataclass(self):
        result = self._run()
        with pytest.raises((AttributeError, TypeError)):
            result.t_half_h = 99.0  # type: ignore[misc]

    def test_interval_in_candidates(self):
        result = self._run()
        assert result.best_interval_h in [8.0, 12.0, 24.0, 48.0]

    def test_cmax_ge_cmin(self):
        result = self._run()
        assert result.predicted_cmax_mg_L >= result.predicted_cmin_mg_L

    def test_accumulation_ratio_ge_one(self):
        result = self._run()
        assert result.accumulation_ratio >= 1.0

    def test_in_window_flag_consistent(self):
        result = self._run(window=(0.0001, 1e6))  # very wide window
        # With a wide window, should be in window
        assert result.in_window is True

    def test_iv_route(self):
        result = optimal_dosing_regimen(
            t_half_h=12.0,
            therapeutic_window=(0.5, 20.0),
            dose_mg=100.0,
            vd_L=50.0,
            cl_L_per_h=5.0,
            route="iv",
        )
        assert isinstance(result, OptimalRegimenResult)
        assert result.predicted_cmax_mg_L > 0

    def test_notes_nonempty(self):
        result = self._run()
        assert len(result.notes) > 0

    def test_invalid_t_half(self):
        with pytest.raises(ValueError, match="t_half_h"):
            optimal_dosing_regimen(0.0, (1.0, 10.0), 100.0, 50.0, 5.0)

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            optimal_dosing_regimen(12.0, (1.0, 10.0), 100.0, 50.0, 0.0)

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            optimal_dosing_regimen(12.0, (1.0, 10.0), 100.0, 0.0, 5.0)

    def test_invalid_window_order(self):
        with pytest.raises(ValueError, match="therapeutic_window"):
            optimal_dosing_regimen(12.0, (10.0, 1.0), 100.0, 50.0, 5.0)

    def test_t_half_stored(self):
        result = self._run(t_half=8.0)
        assert result.t_half_h == 8.0
