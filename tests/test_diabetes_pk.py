"""Tests for diabetes drug PK simulation (Phase 393)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.diabetes_pk import (
    InsulinPKResult,
    MetforminPKResult,
    simulate_insulin_pk,
    simulate_metformin_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insulin(**kw):
    defaults = dict(
        dose_units=10.0,
        insulin_type="rapid",
        cl_L_per_h=0.8,
        vd_L=12.0,
        t_end_h=8.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_insulin_pk(**defaults)


def _metformin(**kw):
    defaults = dict(
        dose_mg=500.0,
        cl_L_per_h=30.0,
        vd_L=350.0,
        interval_h=12.0,
        n_doses=5,
        dt_h=0.5,
    )
    defaults.update(kw)
    return simulate_metformin_pk(**defaults)


# ---------------------------------------------------------------------------
# InsulinPKResult type checks
# ---------------------------------------------------------------------------


class TestInsulinResultType:
    def test_returns_result_object(self):
        r = _insulin()
        assert isinstance(r, InsulinPKResult)

    def test_times_and_conc_same_length(self):
        r = _insulin()
        assert len(r.times_h) == len(r.c_plasma_mU_L)

    def test_times_starts_at_zero(self):
        r = _insulin()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_conc_non_negative(self):
        r = _insulin()
        assert all(c >= 0 for c in r.c_plasma_mU_L)

    def test_cmax_positive(self):
        r = _insulin()
        assert r.cmax_mU_L > 0.0

    def test_tmax_within_simulation(self):
        r = _insulin()
        assert 0.0 <= r.tmax_h <= 8.0

    def test_duration_above_threshold_non_negative(self):
        r = _insulin()
        assert r.duration_above_threshold_h >= 0.0

    def test_notes_is_string(self):
        r = _insulin()
        assert isinstance(r.notes, str)

    def test_insulin_type_stored(self):
        r = _insulin(insulin_type="regular")
        assert r.insulin_type == "regular"

    def test_dose_units_stored(self):
        r = _insulin(dose_units=20.0)
        assert r.dose_units == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Insulin type differences
# ---------------------------------------------------------------------------


class TestInsulinTypes:
    def test_rapid_peaks_earlier_than_regular(self):
        r_rapid = _insulin(insulin_type="rapid", dt_h=0.05)
        r_regular = _insulin(insulin_type="regular", dt_h=0.05)
        assert r_rapid.tmax_h < r_regular.tmax_h

    def test_regular_peaks_earlier_than_nph(self):
        r_regular = _insulin(insulin_type="regular", t_end_h=10.0, dt_h=0.1)
        r_nph = _insulin(insulin_type="NPH", t_end_h=10.0, dt_h=0.1)
        assert r_regular.tmax_h < r_nph.tmax_h

    def test_nph_peaks_earlier_than_glargine(self):
        r_nph = _insulin(insulin_type="NPH", t_end_h=24.0, dt_h=0.2)
        r_glargine = _insulin(insulin_type="glargine", t_end_h=24.0, dt_h=0.2)
        assert r_nph.tmax_h <= r_glargine.tmax_h

    def test_higher_dose_higher_cmax(self):
        r_low = _insulin(dose_units=5.0)
        r_high = _insulin(dose_units=20.0)
        assert r_high.cmax_mU_L > r_low.cmax_mU_L

    def test_all_insulin_types_run(self):
        for itype in ("rapid", "regular", "NPH", "glargine"):
            r = _insulin(insulin_type=itype, t_end_h=24.0, dt_h=0.5)
            assert r.cmax_mU_L > 0.0


# ---------------------------------------------------------------------------
# Insulin validation errors
# ---------------------------------------------------------------------------


class TestInsulinValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_units"):
            simulate_insulin_pk(dose_units=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_insulin_pk(dose_units=-5.0)

    def test_invalid_insulin_type_raises(self):
        with pytest.raises(ValueError, match="insulin_type"):
            simulate_insulin_pk(dose_units=10.0, insulin_type="ultralente")

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_insulin_pk(dose_units=10.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_insulin_pk(dose_units=10.0, vd_L=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_insulin_pk(dose_units=10.0, t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_insulin_pk(dose_units=10.0, dt_h=0.0)


# ---------------------------------------------------------------------------
# MetforminPKResult type checks
# ---------------------------------------------------------------------------


class TestMetforminResultType:
    def test_returns_result_object(self):
        r = _metformin()
        assert isinstance(r, MetforminPKResult)

    def test_times_and_conc_same_length(self):
        r = _metformin()
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_conc_non_negative(self):
        r = _metformin()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_cmax_ss_positive(self):
        r = _metformin()
        assert r.cmax_ss > 0.0

    def test_trough_leq_cmax(self):
        r = _metformin()
        assert r.trough_ss <= r.cmax_ss

    def test_auc_tau_positive(self):
        r = _metformin()
        assert r.auc_tau > 0.0

    def test_dose_mg_stored(self):
        r = _metformin(dose_mg=1000.0)
        assert r.dose_mg == pytest.approx(1000.0)

    def test_n_doses_stored(self):
        r = _metformin(n_doses=3)
        assert r.n_doses == 3

    def test_notes_is_string(self):
        r = _metformin()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Metformin dose and steady-state behaviour
# ---------------------------------------------------------------------------


class TestMetforminBehaviour:
    def test_higher_dose_higher_cmax(self):
        r_low = _metformin(dose_mg=500.0)
        r_high = _metformin(dose_mg=2000.0)
        assert r_high.cmax_ss > r_low.cmax_ss

    def test_more_doses_increases_trough(self):
        r_few = _metformin(n_doses=1)
        r_many = _metformin(n_doses=10)
        assert r_many.trough_ss >= r_few.trough_ss

    def test_simulation_length_matches_n_doses_x_interval(self):
        r = _metformin(n_doses=4, interval_h=12.0)
        assert r.times_h[-1] == pytest.approx(4 * 12.0, rel=0.01)

    def test_single_dose_runs(self):
        r = _metformin(n_doses=1)
        assert r.cmax_ss > 0.0


# ---------------------------------------------------------------------------
# Metformin validation errors
# ---------------------------------------------------------------------------


class TestMetforminValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_metformin_pk(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_metformin_pk(dose_mg=-100.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_metformin_pk(dose_mg=500.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_metformin_pk(dose_mg=500.0, vd_L=0.0)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            simulate_metformin_pk(dose_mg=500.0, interval_h=0.0)

    def test_zero_n_doses_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_metformin_pk(dose_mg=500.0, n_doses=0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_metformin_pk(dose_mg=500.0, dt_h=0.0)
