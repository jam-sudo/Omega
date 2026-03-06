"""Tests for medication adherence model (Phase 115)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.adherence_model import (
    AdherenceResult,
    compare_adherence_levels,
    simulate_adherence,
)


def _default(**kw):
    defaults = dict(
        drug_name="Drug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        dosing_interval_h=24.0,
        n_doses=7,
        adherence_rate=0.8,
        dt_h=0.5,
    )
    defaults.update(kw)
    return simulate_adherence(**defaults)


class TestAdherenceResult:
    def test_returns_result_type(self):
        assert isinstance(_default(), AdherenceResult)

    def test_conc_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.conc_mg_L)

    def test_perfect_conc_positive(self):
        r = _default()
        assert max(r.conc_perfect_mg_L) > 0.0

    def test_times_and_conc_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.conc_mg_L) == len(r.conc_perfect_mg_L)

    def test_auc_actual_leq_perfect(self):
        r = _default(adherence_rate=0.7)
        assert r.auc_actual <= r.auc_perfect + 1e-6

    def test_auc_ratio_between_0_and_1_for_low_adherence(self):
        r = _default(adherence_rate=0.5)
        assert 0.0 <= r.auc_ratio <= 1.0 + 1e-6

    def test_perfect_adherence_ratio_near_1(self):
        r = _default(adherence_rate=1.0)
        assert abs(r.auc_ratio - 1.0) < 0.05

    def test_n_doses_taken_leq_planned(self):
        r = _default()
        assert r.n_doses_taken <= r.n_doses_planned

    def test_dose_times_plus_missed_equals_planned(self):
        r = _default()
        assert len(r.dose_times_h) + len(r.missed_dose_times_h) == r.n_doses_planned

    def test_therapeutic_failure_risk_valid(self):
        r = _default()
        assert r.therapeutic_failure_risk in ("low", "moderate", "high")

    def test_high_adherence_low_failure_risk(self):
        r = _default(adherence_rate=1.0)
        assert r.therapeutic_failure_risk == "low"

    def test_low_adherence_high_failure_risk(self):
        r = _default(adherence_rate=0.3, seed=1)
        assert r.therapeutic_failure_risk in ("moderate", "high")

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_invalid_adherence_raises(self):
        with pytest.raises(ValueError):
            _default(adherence_rate=1.5)

    def test_time_below_mec_non_negative(self):
        r = simulate_adherence("Drug", 100.0, 5.0, 20.0, n_doses=7, mec_mg_L=1.0, dt_h=0.5)
        assert r.time_below_mec_h >= 0.0


class TestCompareAdherenceLevels:
    def test_returns_list(self):
        results = compare_adherence_levels("Drug", 100.0, 5.0, 20.0, n_doses=5, dt_h=1.0)
        assert isinstance(results, list)

    def test_higher_adherence_higher_auc(self):
        results = compare_adherence_levels(
            "Drug", 100.0, 5.0, 20.0, adherence_levels=[0.5, 1.0], n_doses=5, dt_h=1.0
        )
        r_low, r_high = results
        assert r_high.auc_actual >= r_low.auc_actual - 1e-6
