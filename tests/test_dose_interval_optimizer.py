"""Tests for dosing interval optimizer module (Phase 380)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.dose_interval_optimizer import (
    DoseIntervalResult,
    compute_accumulation_ratio,
    optimize_dose_interval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default(**kw) -> DoseIntervalResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        target_trough_mg_L=0.5,
        target_peak_mg_L=5.0,
        intervals_to_test=[6.0, 8.0, 12.0, 24.0],
        n_doses=10,
        ka_per_h=1.0,
        f_oral=1.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return optimize_dose_interval(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dose_interval_result(self):
        assert isinstance(_default(), DoseIntervalResult)

    def test_intervals_tested_matches_input(self):
        r = _default(intervals_to_test=[6.0, 12.0, 24.0])
        assert r.intervals_tested == [6.0, 12.0, 24.0]

    def test_troughs_same_length_as_intervals(self):
        r = _default(intervals_to_test=[6.0, 8.0, 12.0])
        assert len(r.troughs) == 3

    def test_peaks_same_length_as_intervals(self):
        r = _default(intervals_to_test=[6.0, 8.0, 12.0])
        assert len(r.peaks) == 3

    def test_fluctuations_same_length_as_intervals(self):
        r = _default(intervals_to_test=[6.0, 8.0, 12.0])
        assert len(r.fluctuations) == 3

    def test_optimal_interval_in_tested_list(self):
        r = _default()
        assert r.optimal_interval_h in r.intervals_tested

    def test_drug_name_preserved(self):
        r = _default(drug_name="DrugA")
        assert r.drug_name == "DrugA"

    def test_dose_preserved(self):
        r = _default(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# PK metric validity
# ---------------------------------------------------------------------------

class TestPKMetrics:
    def test_troughs_non_negative(self):
        r = _default()
        assert all(t >= 0 for t in r.troughs)

    def test_peaks_positive(self):
        r = _default()
        assert all(p > 0 for p in r.peaks)

    def test_peak_geq_trough_each_interval(self):
        r = _default()
        for trough, peak in zip(r.troughs, r.peaks, strict=True):
            assert peak >= trough - 1e-9

    def test_fluctuations_non_negative(self):
        r = _default()
        assert all(f >= 0 for f in r.fluctuations)

    def test_optimal_trough_matches_intervals(self):
        r = _default()
        idx = r.intervals_tested.index(r.optimal_interval_h)
        assert r.optimal_trough == pytest.approx(r.troughs[idx], abs=1e-9)

    def test_optimal_peak_matches_intervals(self):
        r = _default()
        idx = r.intervals_tested.index(r.optimal_interval_h)
        assert r.optimal_peak == pytest.approx(r.peaks[idx], abs=1e-9)

    def test_optimal_fluctuation_matches_intervals(self):
        r = _default()
        idx = r.intervals_tested.index(r.optimal_interval_h)
        assert r.optimal_fluctuation == pytest.approx(r.fluctuations[idx], abs=1e-9)


# ---------------------------------------------------------------------------
# Optimal selection logic
# ---------------------------------------------------------------------------

class TestOptimalSelection:
    def test_meets_criteria_when_one_interval_fits(self):
        """With a wide target range, at least one interval should meet criteria."""
        r = _default(
            target_trough_mg_L=0.01,
            target_peak_mg_L=1000.0,
            intervals_to_test=[8.0, 12.0],
        )
        assert r.meets_criteria is True

    def test_meets_criteria_false_when_impossible(self):
        """If target peak is unrealistically low, no interval meets criteria."""
        r = _default(
            target_trough_mg_L=0.0,
            target_peak_mg_L=1e-6,  # essentially impossible
            intervals_to_test=[6.0, 12.0, 24.0],
        )
        assert r.meets_criteria is False

    def test_fallback_to_min_fluctuation(self):
        """When no interval meets criteria, pick minimum fluctuation."""
        r = _default(
            target_trough_mg_L=0.0,
            target_peak_mg_L=1e-6,
            intervals_to_test=[6.0, 12.0, 24.0],
        )
        # Optimal fluctuation should be the min
        assert r.optimal_fluctuation == pytest.approx(min(r.fluctuations), abs=1e-9)

    def test_shorter_interval_higher_trough(self):
        """Shorter dosing interval generally leads to higher trough at steady state."""
        r = _default(intervals_to_test=[6.0, 24.0])
        idx_6 = r.intervals_tested.index(6.0)
        idx_24 = r.intervals_tested.index(24.0)
        assert r.troughs[idx_6] >= r.troughs[idx_24]

    def test_shorter_interval_lower_fluctuation(self):
        """Shorter intervals generally give lower peak-trough fluctuation."""
        r = _default(intervals_to_test=[6.0, 24.0])
        idx_6 = r.intervals_tested.index(6.0)
        idx_24 = r.intervals_tested.index(24.0)
        assert r.fluctuations[idx_6] <= r.fluctuations[idx_24] + 1.0  # allow small tolerance

    def test_notes_non_empty(self):
        r = _default()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_drug_name(self):
        with pytest.raises(ValueError):
            _default(drug_name="")

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _default(dose_mg=-100.0)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError):
            _default(vd_L=0.0)

    def test_invalid_trough_negative(self):
        with pytest.raises(ValueError):
            _default(target_trough_mg_L=-1.0)

    def test_invalid_peak_zero(self):
        with pytest.raises(ValueError):
            _default(target_peak_mg_L=0.0)

    def test_trough_geq_peak_raises(self):
        with pytest.raises(ValueError):
            _default(target_trough_mg_L=5.0, target_peak_mg_L=5.0)

    def test_invalid_n_doses(self):
        with pytest.raises(ValueError):
            _default(n_doses=0)

    def test_invalid_ka_negative(self):
        with pytest.raises(ValueError):
            _default(ka_per_h=-1.0)

    def test_invalid_f_oral_zero(self):
        with pytest.raises(ValueError):
            _default(f_oral=0.0)

    def test_invalid_f_oral_above_one(self):
        with pytest.raises(ValueError):
            _default(f_oral=1.1)

    def test_invalid_dt_h_zero(self):
        with pytest.raises(ValueError):
            _default(dt_h=0.0)

    def test_empty_intervals_raises(self):
        with pytest.raises(ValueError):
            _default(intervals_to_test=[])

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            _default(intervals_to_test=[-6.0, 12.0])


# ---------------------------------------------------------------------------
# Accumulation ratio
# ---------------------------------------------------------------------------

class TestAccumulationRatio:
    def test_accumulation_ratio_greater_than_one(self):
        """Accumulation ratio must be > 1."""
        r = compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=50.0, interval_h=8.0)
        assert r > 1.0

    def test_shorter_interval_higher_accumulation(self):
        """Shorter interval → more accumulation."""
        r_short = compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=50.0, interval_h=4.0)
        r_long = compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=50.0, interval_h=24.0)
        assert r_short > r_long

    def test_known_value(self):
        """Validate against analytic formula: R = 1/(1-exp(-ke*tau))."""
        ke = 5.0 / 50.0  # = 0.1
        tau = 12.0
        expected = 1.0 / (1.0 - math.exp(-ke * tau))
        got = compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=50.0, interval_h=tau)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError):
            compute_accumulation_ratio(cl_L_per_h=0.0, vd_L=50.0, interval_h=12.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError):
            compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=0.0, interval_h=12.0)

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError):
            compute_accumulation_ratio(cl_L_per_h=5.0, vd_L=50.0, interval_h=0.0)

    def test_single_interval_input(self):
        """optimize_dose_interval works with a single interval."""
        r = optimize_dose_interval(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            target_trough_mg_L=0.1,
            target_peak_mg_L=10.0,
            intervals_to_test=[12.0],
            n_doses=5,
            dt_h=0.2,
        )
        assert r.optimal_interval_h == pytest.approx(12.0)
