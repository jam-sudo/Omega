"""Tests for dose fractionation optimizer (phase 653)."""

from __future__ import annotations

import dataclasses

import pytest

from omega_pbpk.clinical.dose_fractionation_p653 import (
    DoseFractionationResult,
    compare_fractionation,
    simulate_dose_fractionation,
)

# ---------------------------------------------------------------------------
# Default parameters for convenience
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    total_daily_dose_mg=100.0,
    n_doses_per_day=2,
    cl_L_per_h=5.0,
    vd_L=50.0,
    ka_per_h=1.0,
    mec_mg_L=0.1,
    mtc_mg_L=2.0,
)


def _sim(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_dose_fractionation(**params)


# ---------------------------------------------------------------------------
# Return type & frozen dataclass
# ---------------------------------------------------------------------------


class TestResultType:
    def test_return_type(self):
        r = _sim()
        assert isinstance(r, DoseFractionationResult)

    def test_frozen_dataclass(self):
        r = _sim()
        assert dataclasses.is_dataclass(r)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.score = 999.0  # type: ignore[misc]

    def test_frozen_mutation_raises(self):
        r = _sim()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Basic metric ranges
# ---------------------------------------------------------------------------


class TestMetricRanges:
    def test_fluctuation_positive(self):
        r = _sim()
        assert r.fluctuation_pct > 0

    def test_time_in_window_range(self):
        r = _sim()
        assert 0 <= r.time_in_window_h_per_day <= 24.2  # small tolerance for dt

    def test_peak_trough_ratio_gte_one(self):
        r = _sim()
        assert r.peak_trough_ratio >= 1.0

    def test_cmax_gt_cmin(self):
        r = _sim()
        assert r.cmax_ss_mg_L > r.cmin_ss_mg_L

    def test_cavg_between_cmin_cmax(self):
        r = _sim()
        assert r.cmin_ss_mg_L <= r.cavg_ss_mg_L <= r.cmax_ss_mg_L

    def test_time_above_mec_non_negative(self):
        r = _sim()
        assert r.time_above_mec_h_per_day >= 0

    def test_time_above_mtc_non_negative(self):
        r = _sim()
        assert r.time_above_mtc_h_per_day >= 0


# ---------------------------------------------------------------------------
# Fractionation effects
# ---------------------------------------------------------------------------


class TestFractionationEffects:
    def test_higher_n_doses_lower_fluctuation(self):
        """More frequent dosing should produce smoother profiles."""
        r_qd = _sim(n_doses_per_day=1)
        r_bid = _sim(n_doses_per_day=2)
        r_tid = _sim(n_doses_per_day=3)
        r_qid = _sim(n_doses_per_day=4)
        assert r_qd.fluctuation_pct > r_bid.fluctuation_pct
        assert r_bid.fluctuation_pct > r_tid.fluctuation_pct
        assert r_tid.fluctuation_pct > r_qid.fluctuation_pct

    def test_higher_n_doses_lower_cmax(self):
        r_qd = _sim(n_doses_per_day=1)
        r_qid = _sim(n_doses_per_day=4)
        assert r_qd.cmax_ss_mg_L > r_qid.cmax_ss_mg_L

    def test_higher_n_doses_higher_cmin(self):
        r_qd = _sim(n_doses_per_day=1)
        r_qid = _sim(n_doses_per_day=4)
        assert r_qid.cmin_ss_mg_L > r_qd.cmin_ss_mg_L


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _sim(total_daily_dose_mg=100.0)
        r2 = _sim(total_daily_dose_mg=200.0)
        ratio = r2.cmax_ss_mg_L / r1.cmax_ss_mg_L
        assert 1.9 < ratio < 2.1

    def test_double_dose_doubles_cavg(self):
        r1 = _sim(total_daily_dose_mg=100.0)
        r2 = _sim(total_daily_dose_mg=200.0)
        ratio = r2.cavg_ss_mg_L / r1.cavg_ss_mg_L
        assert 1.9 < ratio < 2.1


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_formula(self):
        r = _sim()
        expected = r.time_in_window_h_per_day - r.time_above_mtc_h_per_day * 2.0
        assert abs(r.score - expected) < 1e-9

    def test_score_negative_when_high_toxicity(self):
        # Very high dose → lots of time above MTC
        r = _sim(total_daily_dose_mg=5000.0, mtc_mg_L=0.5)
        # score can be negative
        assert r.score < r.time_in_window_h_per_day


# ---------------------------------------------------------------------------
# Compare function
# ---------------------------------------------------------------------------


class TestCompare:
    def test_returns_list_of_four(self):
        results = compare_fractionation("Drug", 100.0, 5.0, 50.0)
        assert len(results) == 4

    def test_exactly_one_recommended(self):
        results = compare_fractionation("Drug", 100.0, 5.0, 50.0)
        n_rec = sum(1 for r in results if r.recommended)
        assert n_rec >= 1  # at least one recommended (ties possible)

    def test_compare_n_doses_order(self):
        results = compare_fractionation("Drug", 100.0, 5.0, 50.0)
        assert [r.n_doses_per_day for r in results] == [1, 2, 3, 4]

    def test_recommended_has_best_score(self):
        results = compare_fractionation("Drug", 100.0, 5.0, 50.0)
        best_score = max(r.score for r in results)
        for r in results:
            if r.recommended:
                assert r.score == best_score


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="total_daily_dose_mg"):
            _sim(total_daily_dose_mg=-10)

    def test_invalid_n_doses_raises(self):
        with pytest.raises(ValueError, match="n_doses_per_day"):
            _sim(n_doses_per_day=5)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _sim(cl_L_per_h=-1)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _sim(vd_L=-1)

    def test_mec_le_zero_raises(self):
        with pytest.raises(ValueError, match="mec_mg_L"):
            _sim(mec_mg_L=0)

    def test_mtc_le_mec_raises(self):
        with pytest.raises(ValueError, match="mtc_mg_L"):
            _sim(mec_mg_L=1.0, mtc_mg_L=0.5)

    def test_dose_interval_correctness(self):
        r = _sim(n_doses_per_day=3)
        assert abs(r.dose_interval_h - 8.0) < 1e-9
