"""Tests for IV infusion PK model (Phase 117)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.iv_infusion import (
    InfusionResult,
    compare_bolus_vs_infusion,
    simulate_constant_infusion,
    simulate_loading_then_maintenance,
)


def _default(**kw):
    defaults = dict(
        drug_name="Drug",
        dose_mg=100.0,
        infusion_duration_h=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_constant_infusion(**defaults)


class TestConstantInfusion:
    def test_returns_result_type(self):
        assert isinstance(_default(), InfusionResult)

    def test_conc_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.conc_mg_L)

    def test_conc_rises_during_infusion(self):
        r = _default()
        idx_end_infusion = int(1.0 / 0.05)
        assert r.conc_mg_L[idx_end_infusion] > r.conc_mg_L[0]

    def test_conc_falls_after_infusion(self):
        r = _default(t_end_h=5.0)
        assert r.conc_mg_L[-1] < r.cmax

    def test_css_positive(self):
        r = _default()
        assert r.css_mg_L > 0.0

    def test_infusion_rate_equals_dose_over_duration(self):
        r = _default(dose_mg=200.0, infusion_duration_h=2.0)
        assert abs(r.infusion_rate_mg_h - 100.0) < 0.01

    def test_auc_positive(self):
        r = _default()
        assert r.auc_0_t > 0.0

    def test_css_90pct_positive(self):
        r = _default()
        assert r.css_90pct_h > 0.0

    def test_t_half_positive(self):
        r = _default()
        assert r.t_half_h > 0.0

    def test_post_infusion_auc_non_negative(self):
        r = _default(t_end_h=6.0)
        assert r.post_infusion_auc >= 0.0

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            _default(infusion_duration_h=0.0)

    def test_longer_infusion_lower_cmax(self):
        r_short = _default(infusion_duration_h=0.5)
        r_long = _default(infusion_duration_h=4.0)
        assert r_short.cmax > r_long.cmax


class TestLoadingMaintenance:
    def test_returns_result_type(self):
        r = simulate_loading_then_maintenance("Drug", 200.0, 50.0, 4.0, 5.0, 20.0, dt_h=0.1)
        assert isinstance(r, InfusionResult)

    def test_cmax_positive(self):
        r = simulate_loading_then_maintenance("Drug", 200.0, 50.0, 4.0, 5.0, 20.0, dt_h=0.1)
        assert r.cmax > 0.0


class TestCompareBolus:
    def test_returns_dict(self):
        result = compare_bolus_vs_infusion("Drug", 100.0, 5.0, 20.0)
        assert isinstance(result, dict)
        assert "bolus_5min" in result

    def test_bolus_higher_cmax_than_slow_infusion(self):
        result = compare_bolus_vs_infusion("Drug", 100.0, 5.0, 20.0, [2.0])
        assert result["bolus_5min"].cmax > result["2.0h_infusion"].cmax
