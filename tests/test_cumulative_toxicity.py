"""Tests for cumulative toxicity and multicycle PK (Phase 108)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.cumulative_toxicity import (
    CumulativeToxicityResult,
    MulticyclePKResult,
    assess_cumulative_toxicity,
    simulate_multicycle_pk,
)


class TestMulticyclePK:
    def _run(self, **kw):
        defaults = dict(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
            n_cycles=3,
            cycle_interval_h=168.0,
            t_per_cycle_h=24.0,
            dt_h=0.5,
        )
        defaults.update(kw)
        return simulate_multicycle_pk(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), MulticyclePKResult)

    def test_conc_positive(self):
        r = self._run()
        assert max(r.conc_mg_L) > 0.0

    def test_trough_at_cycle_length(self):
        r = self._run(n_cycles=4)
        assert len(r.trough_at_cycle) == 4

    def test_css_max_geq_css_min(self):
        r = self._run()
        assert r.css_max >= r.css_min

    def test_accumulation_ratio_positive(self):
        r = self._run()
        assert r.accumulation_ratio > 0.0

    def test_invalid_n_cycles_raises(self):
        with pytest.raises(ValueError):
            simulate_multicycle_pk("Drug", 100.0, 5.0, 20.0, n_cycles=0)

    def test_times_h_length_matches_conc(self):
        r = self._run()
        assert len(r.times_h) == len(r.conc_mg_L)


class TestAssessCumulativeToxicity:
    def _run(self, **kw):
        defaults = dict(
            drug_name="Drug",
            cycle_doses_mg=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            cycle_aucs=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            auc_mtc=30.0,
        )
        defaults.update(kw)
        return assess_cumulative_toxicity(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), CumulativeToxicityResult)

    def test_toxicity_score_between_0_and_1(self):
        r = self._run()
        assert all(0.0 <= t <= 1.0 for t in r.toxicity_score)

    def test_cumulative_toxicity_between_0_and_1(self):
        r = self._run()
        assert 0.0 <= r.cumulative_toxicity <= 1.0

    def test_grade_valid_range(self):
        r = self._run()
        assert 0 <= r.predicted_grade <= 4

    def test_recommendation_is_string(self):
        r = self._run()
        assert isinstance(r.recommendation, str)

    def test_dlt_probability_between_0_and_1(self):
        r = self._run()
        assert 0.0 <= r.dlt_probability <= 1.0

    def test_high_auc_high_toxicity(self):
        r_low = self._run(cycle_aucs=[5.0] * 6)
        r_high = self._run(cycle_aucs=[50.0] * 6)
        assert r_high.cumulative_toxicity > r_low.cumulative_toxicity

    def test_empty_aucs_raises(self):
        with pytest.raises(ValueError):
            assess_cumulative_toxicity("Drug", [], [], auc_mtc=30.0)

    def test_invalid_auc_mtc_raises(self):
        with pytest.raises(ValueError):
            assess_cumulative_toxicity("Drug", [100.0], [20.0], auc_mtc=0.0)

    def test_cumulative_auc_is_sum(self):
        aucs = [10.0, 15.0, 20.0]
        r = assess_cumulative_toxicity("Drug", [100.0] * 3, aucs, auc_mtc=30.0)
        assert abs(r.cumulative_auc - 45.0) < 1e-6
