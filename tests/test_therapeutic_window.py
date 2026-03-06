"""Tests for therapeutic window analysis (Phase 104)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.clinical.therapeutic_window import (
    TherapeuticWindowResult,
    analyze_therapeutic_window,
    optimal_dose_for_window,
)


def _flat_conc(value=3.0, t_end=12.0, n=100):
    t = np.linspace(0, t_end, n + 1).tolist()
    c = [value] * (n + 1)
    return t, c


def _decaying_conc(c0=5.0, ke=0.3, t_end=12.0, n=100):
    t_arr = np.linspace(0, t_end, n + 1)
    c = (c0 * np.exp(-ke * t_arr)).tolist()
    return t_arr.tolist(), c


class TestAnalyzeTherapeuticWindow:
    def test_returns_result_type(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert isinstance(r, TherapeuticWindowResult)

    def test_always_in_range(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert r.pct_time_in_range > 90.0

    def test_always_above_toxic(self):
        t, c = _flat_conc(10.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert r.time_above_mtc_h > 0.0
        assert r.risk_level in ("caution", "toxic")

    def test_always_below_mec(self):
        t, c = _flat_conc(0.5)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert r.time_below_mec_h > 0.0

    def test_therapeutic_ratio(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=2.0, mtc_mg_L=10.0)
        assert abs(r.therapeutic_ratio - 5.0) < 0.01

    def test_cmax_cmin_correct(self):
        t, c = _decaying_conc(c0=5.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=0.5, mtc_mg_L=10.0)
        assert r.cmax_mg_L > r.cmin_mg_L

    def test_pct_time_in_range_0_to_100(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert 0.0 <= r.pct_time_in_range <= 100.0

    def test_risk_level_valid(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert r.risk_level in ("safe", "caution", "toxic")

    def test_recommendation_is_string(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert isinstance(r.recommendation, str)

    def test_invalid_mec_raises(self):
        t, c = _flat_conc(3.0)
        with pytest.raises(ValueError):
            analyze_therapeutic_window("Drug", t, c, mec_mg_L=0.0, mtc_mg_L=5.0)

    def test_mtc_leq_mec_raises(self):
        t, c = _flat_conc(3.0)
        with pytest.raises(ValueError):
            analyze_therapeutic_window("Drug", t, c, mec_mg_L=5.0, mtc_mg_L=3.0)

    def test_auc_therapeutic_non_negative(self):
        t, c = _flat_conc(3.0)
        r = analyze_therapeutic_window("Drug", t, c, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert r.auc_therapeutic >= 0.0


class TestOptimalDose:
    def test_returns_dict(self):
        t = np.linspace(0, 12, 50).tolist()
        conc_per_mg = [np.exp(-0.2 * ti) / 20.0 for ti in t]
        result = optimal_dose_for_window("Drug", t, conc_per_mg, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert "optimal_dose_mg" in result
        assert "pct_time_in_range" in result

    def test_optimal_dose_positive(self):
        t = np.linspace(0, 12, 50).tolist()
        conc_per_mg = [np.exp(-0.2 * ti) / 20.0 for ti in t]
        result = optimal_dose_for_window("Drug", t, conc_per_mg, mec_mg_L=1.0, mtc_mg_L=5.0)
        assert result["optimal_dose_mg"] > 0.0
