"""Tests for drug accumulation and steady-state analysis (Phase 74)."""
from __future__ import annotations

import pytest

from omega_pbpk.clinical.accumulation import (
    AccumulationResult,
    accumulation_ratio,
    simulate_accumulation,
    dosing_regimen_comparison,
)


def _default(**kw):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dosing_interval_h=12.0,
        n_doses=8,
    )
    defaults.update(kw)
    return simulate_accumulation(**defaults)


class TestAccumulationRatio:
    def test_returns_float_geq_1(self):
        assert accumulation_ratio(t_half_h=12.0, tau_h=12.0) >= 1.0

    def test_long_tau_low_accumulation(self):
        # tau >> t_half → minimal accumulation → R ≈ 1
        r = accumulation_ratio(t_half_h=6.0, tau_h=60.0)
        assert r < 1.5

    def test_short_tau_high_accumulation(self):
        # tau << t_half → large accumulation
        r = accumulation_ratio(t_half_h=24.0, tau_h=4.0)
        assert r > 3.0

    def test_formula(self):
        import math
        r = accumulation_ratio(t_half_h=12.0, tau_h=12.0)
        expected = 1.0 / (1.0 - math.exp(-0.693))
        assert abs(r - expected) < 0.1


class TestSimulateAccumulation:
    def test_returns_result_type(self):
        assert isinstance(_default(), AccumulationResult)

    def test_n_doses_to_ss_positive(self):
        assert _default().n_doses_to_ss > 0

    def test_accumulation_ratio_geq_1(self):
        assert _default().accumulation_ratio_r >= 1.0

    def test_css_max_positive(self):
        assert _default().css_max_mg_L > 0

    def test_css_min_positive(self):
        assert _default().css_min_mg_L >= 0

    def test_css_max_geq_min(self):
        r = _default()
        assert r.css_max_mg_L >= r.css_min_mg_L

    def test_loading_dose_geq_maintenance(self):
        r = _default()
        assert r.loading_dose_mg >= r.dose_mg

    def test_doses_profile_length(self):
        r = _default(n_doses=8)
        assert len(r.doses_profile) == 8

    def test_fluctuation_pct_positive(self):
        assert _default().fluctuation_pct >= 0

    def test_dose_scaling(self):
        r1 = _default(dose_mg=100.0)
        r2 = _default(dose_mg=200.0)
        assert r2.css_max_mg_L > r1.css_max_mg_L

    def test_time_to_ss_positive(self):
        assert _default().time_to_ss_h > 0

    def test_drug_name_preserved(self):
        r = _default(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"


class TestDosingRegimenComparison:
    def test_returns_list_of_4(self):
        r = dosing_regimen_comparison("Drug", 100.0, 5.0, 50.0)
        assert len(r) == 4

    def test_entries_have_keys(self):
        r = dosing_regimen_comparison("Drug", 100.0, 5.0, 50.0)
        for entry in r:
            for key in ("regimen", "tau_h", "dose_mg", "css_avg", "fluctuation_pct", "peak_trough_ratio"):
                assert key in entry, f"Missing key {key} in {entry}"

    def test_once_daily_highest_fluctuation(self):
        r = dosing_regimen_comparison("Drug", 100.0, 5.0, 50.0)
        # Sort by tau_h; longer tau = once daily = more fluctuation
        sorted_r = sorted(r, key=lambda x: x["tau_h"])
        # Once-daily (highest tau) should have highest fluctuation
        assert sorted_r[-1]["fluctuation_pct"] >= sorted_r[0]["fluctuation_pct"]

    def test_same_css_avg(self):
        r = dosing_regimen_comparison("Drug", 100.0, 5.0, 50.0)
        css_avgs = [e["css_avg"] for e in r]
        # All regimens should give same average concentration
        assert max(css_avgs) - min(css_avgs) < max(css_avgs) * 0.05  # within 5%
