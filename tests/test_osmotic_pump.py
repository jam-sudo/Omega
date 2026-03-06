"""Tests for biopharmaceutics/osmotic_pump.py (Phase 213)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.osmotic_pump import (
    OsmoticPumpResult,
    compare_release_durations,
    simulate_osmotic_pump,
)

# ---------------------------------------------------------------------------
# Helper: default parameters for quick simulations
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    release_duration_h=8.0,
    ka_per_h=1.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def _sim(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_osmotic_pump(**params)


# ---------------------------------------------------------------------------
# 1. Input validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-10.0)

    def test_zero_release_duration_raises(self):
        with pytest.raises(ValueError, match="release_duration_h"):
            _sim(release_duration_h=0.0)

    def test_negative_release_duration_raises(self):
        with pytest.raises(ValueError, match="release_duration_h"):
            _sim(release_duration_h=-1.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            _sim(ka_per_h=0.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _sim(cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _sim(vd_L=0.0)


# ---------------------------------------------------------------------------
# 2. Return type and field checks
# ---------------------------------------------------------------------------
class TestResultFields:
    def setup_method(self):
        self.res = _sim()

    def test_returns_dataclass(self):
        assert isinstance(self.res, OsmoticPumpResult)

    def test_drug_name(self):
        assert self.res.drug_name == "TestDrug"

    def test_times_h_is_list(self):
        assert isinstance(self.res.times_h, list)

    def test_c_plasma_is_list(self):
        assert isinstance(self.res.c_plasma_mg_L, list)

    def test_m_tablet_is_list(self):
        assert isinstance(self.res.m_tablet_mg, list)

    def test_c_gi_is_list(self):
        assert isinstance(self.res.c_gi_mg_mL, list)

    def test_lengths_consistent(self):
        r = self.res
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.m_tablet_mg)
        assert len(r.times_h) == len(r.c_gi_mg_mL)

    def test_notes_is_str(self):
        assert isinstance(self.res.notes, str)

    def test_cmax_positive(self):
        assert self.res.cmax_mg_L > 0.0

    def test_auc_positive(self):
        assert self.res.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 3. Physical / pharmacokinetic properties
# ---------------------------------------------------------------------------
class TestPKProperties:
    def test_peak_trough_ratio_gte_one(self):
        res = _sim()
        assert res.peak_trough_ratio >= 1.0 - 1e-9

    def test_fluctuation_pct_nonnegative(self):
        res = _sim()
        assert res.fluctuation_pct >= 0.0

    def test_release_complete_h_approx_duration(self):
        duration = 8.0
        res = _sim(release_duration_h=duration, t_end_h=24.0)
        # tablet should be exhausted near the release_duration_h
        assert abs(res.release_complete_h - duration) < 2.0  # within 2 h

    def test_tablet_empty_at_end(self):
        res = _sim(release_duration_h=4.0, t_end_h=24.0)
        # by end of simulation tablet should be empty
        assert res.m_tablet_mg[-1] < res.dose_mg * 0.01

    def test_plasma_starts_at_zero(self):
        res = _sim()
        assert res.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Short vs long release duration
# ---------------------------------------------------------------------------
class TestReleaseControlEffect:
    """Longer release duration → lower Cmax and lower fluctuation."""

    def setup_method(self):
        self.short = _sim(release_duration_h=2.0)
        self.long = _sim(release_duration_h=16.0)

    def test_shorter_duration_higher_cmax(self):
        assert self.short.cmax_mg_L > self.long.cmax_mg_L

    def test_shorter_duration_higher_fluctuation(self):
        assert self.short.fluctuation_pct > self.long.fluctuation_pct

    def test_shorter_duration_higher_ptr(self):
        assert self.short.peak_trough_ratio > self.long.peak_trough_ratio


# ---------------------------------------------------------------------------
# 5. Dose linearity
# ---------------------------------------------------------------------------
def test_double_dose_approx_double_cmax():
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    assert ratio == pytest.approx(2.0, rel=0.1)


def test_double_dose_approx_double_auc():
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.1)


# ---------------------------------------------------------------------------
# 6. compare_release_durations
# ---------------------------------------------------------------------------
class TestCompareReleaseDurations:
    def setup_method(self):
        self.durations = [2.0, 8.0, 16.0]
        self.results = compare_release_durations(
            drug_name="DrugX",
            dose_mg=100.0,
            durations_h=self.durations,
            ka_per_h=1.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_correct_number_of_results(self):
        assert len(self.results) == len(self.durations)

    def test_sorted_by_peak_trough_ratio_ascending(self):
        ptrs = [r.peak_trough_ratio for r in self.results]
        assert ptrs == sorted(ptrs)

    def test_all_results_are_osmotic_pump_results(self):
        for r in self.results:
            assert isinstance(r, OsmoticPumpResult)

    def test_auc_similar_across_durations(self):
        """Same dose should give similar AUC regardless of release rate."""
        aucs = [r.auc_mg_h_per_L for r in self.results]
        auc_mean = sum(aucs) / len(aucs)
        for auc in aucs:
            # within 30% of mean (some variation due to transit loss)
            assert abs(auc - auc_mean) / auc_mean < 0.30

    def test_empty_durations_raises(self):
        with pytest.raises(ValueError):
            compare_release_durations(
                drug_name="X",
                dose_mg=100.0,
                durations_h=[],
                ka_per_h=1.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
            )
