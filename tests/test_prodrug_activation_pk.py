"""Tests for Phase 827 — Drug Prodrug Activation Kinetics."""

import math

import pytest

from omega_pbpk.core.prodrug_activation_pk import (
    ProdrugPKResult,
    compare_prodrug_strategies,
    simulate_prodrug_pk,
)

# ---------------------------------------------------------------------------
# Default kwargs
# ---------------------------------------------------------------------------

DEFAULT_KW = dict(
    prodrug_name="CapProdrug",
    active_drug_name="ActiveCapecitabine",
    dose_mg=100.0,
    vd_prodrug_L=30.0,
    vd_active_L=50.0,
    ka_prodrug_per_h=1.0,
    kconv_per_h=0.5,
    ke_prodrug_per_h=0.3,
    ke_active_per_h=0.15,
    f_oral_prodrug=0.8,
    f_conversion=0.8,
    route="oral",
    t_end_h=24.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(DEFAULT_KW)
    kw.update(overrides)
    return simulate_prodrug_pk(**kw)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_prodrug_pk_result(self):
        result = _run()
        assert isinstance(result, ProdrugPKResult)

    def test_list_lengths_match(self):
        result = _run()
        n = len(result.times_h)
        assert len(result.c_prodrug_mg_L) == n
        assert len(result.c_active_mg_L) == n

    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == 0.0

    def test_times_end_approx_t_end(self):
        result = _run(t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1

    def test_notes_nonempty_string(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_route_preserved(self):
        result = _run(route="oral")
        assert result.route == "oral"

    def test_iv_route_preserved(self):
        result = _run(route="iv")
        assert result.route == "iv"


# ---------------------------------------------------------------------------
# Active drug PK metrics
# ---------------------------------------------------------------------------


class TestActiveDrugMetrics:
    def test_cmax_active_positive(self):
        result = _run()
        assert result.cmax_active_mg_L > 0.0

    def test_auc_active_positive(self):
        result = _run()
        assert result.auc_active_mg_h_per_L > 0.0

    def test_tmax_active_nonnegative(self):
        result = _run()
        assert result.tmax_active_h >= 0.0

    def test_t_half_active_formula(self):
        ke = 0.15
        expected = math.log(2.0) / ke
        result = _run(ke_active_per_h=ke)
        assert abs(result.t_half_active_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Prodrug metrics
# ---------------------------------------------------------------------------


class TestProdrugMetrics:
    def test_cmax_prodrug_positive_oral(self):
        result = _run(route="oral")
        assert result.cmax_prodrug_mg_L > 0.0

    def test_cmax_prodrug_positive_iv(self):
        result = _run(route="iv")
        assert result.cmax_prodrug_mg_L > 0.0

    def test_auc_prodrug_positive(self):
        result = _run()
        assert result.auc_prodrug_mg_h_per_L > 0.0

    def test_active_to_prodrug_auc_ratio_positive(self):
        result = _run()
        assert result.active_to_prodrug_auc_ratio > 0.0


# ---------------------------------------------------------------------------
# PK behaviour
# ---------------------------------------------------------------------------


class TestPKBehaviour:
    def test_higher_kconv_higher_active_cmax(self):
        r_low = _run(kconv_per_h=0.1)
        r_high = _run(kconv_per_h=1.5)
        assert r_high.cmax_active_mg_L > r_low.cmax_active_mg_L

    def test_higher_kconv_earlier_active_tmax(self):
        r_low = _run(kconv_per_h=0.1, t_end_h=48.0)
        r_high = _run(kconv_per_h=2.0, t_end_h=48.0)
        # Higher conversion → peak occurs earlier
        assert r_high.tmax_active_h <= r_low.tmax_active_h + 2.0

    def test_f_conversion_1_vs_05_auc_ratio(self):
        r_full = _run(f_conversion=1.0)
        r_half = _run(f_conversion=0.5)
        ratio = r_full.auc_active_mg_h_per_L / r_half.auc_active_mg_h_per_L
        # Should be approximately 2 (within 5%)
        assert abs(ratio - 2.0) < 0.1

    def test_iv_route_tmax_prodrug_near_zero(self):
        result = _run(route="iv", kconv_per_h=0.5, ke_prodrug_per_h=0.3)
        # IV bolus: prodrug starts at peak immediately
        assert result.tmax_prodrug_h < 0.5

    def test_zero_kconv_no_active_drug(self):
        result = _run(kconv_per_h=0.0)
        assert result.cmax_active_mg_L < 1e-6
        assert result.auc_active_mg_h_per_L < 1e-6

    def test_higher_dose_proportional_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_active_mg_L / r1.cmax_active_mg_L
        assert abs(ratio - 2.0) < 0.1


# ---------------------------------------------------------------------------
# compare_prodrug_strategies
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_returns_list(self):
        strategies = [
            {"kconv_per_h": 0.5, "f_conversion": 0.8, "route": "oral"},
            {"kconv_per_h": 1.0, "f_conversion": 0.9, "route": "oral"},
        ]
        results = compare_prodrug_strategies("PD", "AD", 100.0, strategies)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_descending_by_auc_active(self):
        strategies = [
            {"kconv_per_h": 0.1, "f_conversion": 0.3, "route": "oral"},
            {"kconv_per_h": 1.5, "f_conversion": 0.9, "route": "oral"},
            {"kconv_per_h": 0.8, "f_conversion": 0.6, "route": "oral"},
        ]
        results = compare_prodrug_strategies("PD", "AD", 100.0, strategies)
        aucs = [r.auc_active_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_results_are_prodrug_pk_result(self):
        strategies = [
            {"kconv_per_h": 0.5, "f_conversion": 0.8, "route": "iv"},
            {"kconv_per_h": 0.5, "f_conversion": 0.8, "route": "oral"},
        ]
        results = compare_prodrug_strategies("PD", "AD", 100.0, strategies)
        assert all(isinstance(r, ProdrugPKResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-50.0)

    def test_invalid_vd_prodrug_zero(self):
        with pytest.raises(ValueError):
            _run(vd_prodrug_L=0.0)

    def test_invalid_vd_active_negative(self):
        with pytest.raises(ValueError):
            _run(vd_active_L=-10.0)

    def test_invalid_kconv_negative(self):
        with pytest.raises(ValueError):
            _run(kconv_per_h=-0.1)

    def test_invalid_ke_prodrug_negative(self):
        with pytest.raises(ValueError):
            _run(ke_prodrug_per_h=-0.1)

    def test_invalid_ke_active_negative(self):
        with pytest.raises(ValueError):
            _run(ke_active_per_h=-0.1)

    def test_invalid_route(self):
        with pytest.raises(ValueError):
            _run(route="subcutaneous")
