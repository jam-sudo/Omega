"""Tests for RES (Reticuloendothelial System) uptake model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.res_uptake import (
    RESUptakeResult,
    compare_res_conditions,
    simulate_res_uptake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(**overrides) -> RESUptakeResult:
    kw = dict(
        compound_name="TestCompound",
        dose_mg=10.0,
        k_liver_per_h=0.3,
        k_spleen_per_h=0.1,
    )
    kw.update(overrides)
    return simulate_res_uptake(**kw)


# ---------------------------------------------------------------------------
# 1-5: Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-5.0)

    def test_k_liver_negative_raises(self):
        with pytest.raises(ValueError, match="k_liver"):
            _run(k_liver_per_h=-0.1)

    def test_k_spleen_negative_raises(self):
        with pytest.raises(ValueError, match="k_spleen"):
            _run(k_spleen_per_h=-0.1)

    def test_k_other_elim_zero_raises(self):
        with pytest.raises(ValueError, match="k_other_elim"):
            _run(k_other_elim_per_h=0.0)

    def test_k_other_elim_negative_raises(self):
        with pytest.raises(ValueError, match="k_other_elim"):
            _run(k_other_elim_per_h=-0.05)


# ---------------------------------------------------------------------------
# 6-8: Basic return type and structure
# ---------------------------------------------------------------------------


class TestBasicSimulation:
    def test_returns_res_uptake_result(self):
        r = _run()
        assert isinstance(r, RESUptakeResult)

    def test_result_fields_correct_types(self):
        r = _run()
        assert isinstance(r.compound_name, str)
        assert isinstance(r.dose_mg, float)
        assert isinstance(r.k_liver_per_h, float)
        assert isinstance(r.k_spleen_per_h, float)
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_plasma_mg_L, list)
        assert isinstance(r.a_liver_pct, list)
        assert isinstance(r.a_spleen_pct, list)
        assert isinstance(r.cmax_plasma_mg_L, float)
        assert isinstance(r.auc_plasma_mg_h_per_L, float)
        assert isinstance(r.liver_pct_final, float)
        assert isinstance(r.spleen_pct_final, float)
        assert isinstance(r.t_half_plasma_h, float)
        assert isinstance(r.res_clearance_fraction, float)
        assert isinstance(r.notes, str)

    def test_time_series_lengths_match(self):
        r = _run(t_end_h=24.0, dt_h=0.5)
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.a_liver_pct) == n
        assert len(r.a_spleen_pct) == n


# ---------------------------------------------------------------------------
# 9-11: No RES uptake — all eliminated via k_other_elim
# ---------------------------------------------------------------------------


class TestNoRESUptake:
    def test_k_liver_zero_k_spleen_zero_liver_pct_final_zero(self):
        r = _run(k_liver_per_h=0.0, k_spleen_per_h=0.0, k_other_elim_per_h=0.5)
        assert r.liver_pct_final == pytest.approx(0.0, abs=1e-6)
        assert r.spleen_pct_final == pytest.approx(0.0, abs=1e-6)

    def test_k_liver_zero_k_spleen_zero_res_fraction_zero(self):
        r = _run(k_liver_per_h=0.0, k_spleen_per_h=0.0, k_other_elim_per_h=0.5)
        assert r.res_clearance_fraction == pytest.approx(0.0, abs=1e-6)

    def test_k_liver_zero_k_spleen_zero_plasma_decreases(self):
        r = _run(k_liver_per_h=0.0, k_spleen_per_h=0.0, k_other_elim_per_h=0.3)
        c = r.c_plasma_mg_L
        # Plasma should decrease overall
        assert c[-1] < c[0]


# ---------------------------------------------------------------------------
# 12-13: High k_liver → significant liver accumulation
# ---------------------------------------------------------------------------


class TestHighLiverUptake:
    def test_high_k_liver_liver_pct_exceeds_50_at_24h(self):
        r = _run(
            k_liver_per_h=1.0,
            k_spleen_per_h=0.05,
            k_other_elim_per_h=0.05,
            t_end_h=48.0,
        )
        # At 24h: find index
        t = r.times_h
        idx = min(range(len(t)), key=lambda i: abs(t[i] - 24.0))
        assert r.a_liver_pct[idx] > 50.0

    def test_high_k_liver_res_fraction_above_zero(self):
        r = _run(k_liver_per_h=0.8, k_spleen_per_h=0.1, k_other_elim_per_h=0.05)
        assert r.res_clearance_fraction > 0.0


# ---------------------------------------------------------------------------
# 14: Plasma decreases monotonically
# ---------------------------------------------------------------------------


class TestMonotonicPlasma:
    def test_plasma_decreases_monotonically(self):
        r = _run(t_end_h=24.0, dt_h=0.1)
        c = r.c_plasma_mg_L
        for i in range(1, len(c)):
            assert c[i] <= c[i - 1] + 1e-10, f"Non-monotonic at step {i}"


# ---------------------------------------------------------------------------
# 15: Mass balance constraint
# ---------------------------------------------------------------------------


class TestMassBalance:
    def test_liver_spleen_pct_leq_100(self):
        r = _run(k_liver_per_h=0.5, k_spleen_per_h=0.3, k_other_elim_per_h=0.1)
        assert r.liver_pct_final + r.spleen_pct_final <= 100.0 + 1e-6


# ---------------------------------------------------------------------------
# 16: res_clearance_fraction in [0, 1]
# ---------------------------------------------------------------------------


class TestRESFraction:
    def test_res_clearance_fraction_in_unit_interval(self):
        for k_l, k_s in [(0.0, 0.0), (0.1, 0.05), (1.0, 0.5)]:
            r = _run(k_liver_per_h=k_l, k_spleen_per_h=k_s, k_other_elim_per_h=0.1)
            assert 0.0 <= r.res_clearance_fraction <= 1.0


# ---------------------------------------------------------------------------
# 17: t_half decreases with higher RES rates
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_decreases_with_higher_res_rates(self):
        r_low = _run(k_liver_per_h=0.05, k_spleen_per_h=0.02, k_other_elim_per_h=0.05)
        r_high = _run(k_liver_per_h=0.5, k_spleen_per_h=0.2, k_other_elim_per_h=0.05)
        assert r_high.t_half_plasma_h < r_low.t_half_plasma_h

    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_plasma_h > 0.0


# ---------------------------------------------------------------------------
# 18-19: compare_res_conditions
# ---------------------------------------------------------------------------


class TestCompareRESConditions:
    def test_compare_returns_correct_length(self):
        conditions = [
            {"k_liver_per_h": 0.1, "k_spleen_per_h": 0.05, "label": "low"},
            {"k_liver_per_h": 0.5, "k_spleen_per_h": 0.2, "label": "med"},
            {"k_liver_per_h": 1.0, "k_spleen_per_h": 0.4, "label": "high"},
        ]
        results = compare_res_conditions("NP", 10.0, conditions)
        assert len(results) == 3

    def test_compare_results_are_res_uptake_result(self):
        conditions = [
            {"k_liver_per_h": 0.2, "k_spleen_per_h": 0.1},
            {"k_liver_per_h": 0.4, "k_spleen_per_h": 0.2},
        ]
        results = compare_res_conditions("Compound", 5.0, conditions)
        for r in results:
            assert isinstance(r, RESUptakeResult)

    def test_compare_higher_k_liver_gives_faster_clearance(self):
        conditions = [
            {"k_liver_per_h": 0.1, "k_spleen_per_h": 0.05, "label": "slow"},
            {"k_liver_per_h": 1.0, "k_spleen_per_h": 0.05, "label": "fast"},
        ]
        results = compare_res_conditions("NP", 10.0, conditions, t_end_h=24.0)
        # Higher k_liver → more in liver at end
        assert results[1].liver_pct_final > results[0].liver_pct_final


# ---------------------------------------------------------------------------
# 20: cmax equals initial plasma concentration
# ---------------------------------------------------------------------------


class TestCmax:
    def test_cmax_equals_initial_plasma_concentration(self):
        dose = 10.0
        v = 3.0
        r = _run(dose_mg=dose, v_plasma_L=v)
        expected_c0 = dose / v
        assert r.cmax_plasma_mg_L == pytest.approx(expected_c0, rel=1e-4)

    def test_auc_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0.0
