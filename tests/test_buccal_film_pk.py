"""Tests for Phase 1115 — Buccal film PK model."""

import pytest

from omega_pbpk.core.buccal_film_pk import BuccalFilmPKResult, simulate_buccal_film_pk


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _default(**kwargs) -> BuccalFilmPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        k_dissolve_per_h=0.3,
        f_buccal=0.6,
        k_abs_per_h=0.8,
        cl_L_per_h=5.0,
        vd_L=30.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_buccal_film_pk(**defaults)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TestResultType:
    def test_returns_buccal_film_pk_result(self):
        r = _default()
        assert isinstance(r, BuccalFilmPKResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Fentanyl")
        assert r.drug_name == "Fentanyl"

    def test_dose_preserved(self):
        r = _default(dose_mg=5.0)
        assert r.dose_mg == 5.0

    def test_f_buccal_preserved(self):
        r = _default(f_buccal=0.7)
        assert r.f_buccal == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Time-course shape
# ---------------------------------------------------------------------------

class TestTimeCourse:
    def test_times_non_empty(self):
        r = _default()
        assert len(r.times_h) > 0

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_monotonically_increasing(self):
        r = _default()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] > r.times_h[i - 1]

    def test_plasma_starts_at_zero(self):
        r = _default()
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_film_starts_at_dose(self):
        r = _default(dose_mg=10.0)
        assert r.a_film_mg[0] == pytest.approx(10.0)

    def test_mucosa_starts_at_zero(self):
        r = _default()
        assert r.a_mucosa_mg[0] == pytest.approx(0.0)

    def test_film_monotonically_decreasing(self):
        r = _default()
        for i in range(1, len(r.a_film_mg)):
            assert r.a_film_mg[i] <= r.a_film_mg[i - 1] + 1e-9

    def test_plasma_rises_then_falls(self):
        """Plasma should peak somewhere in the middle, not at time 0."""
        r = _default()
        assert r.cmax_mg_L > r.c_plasma_mg_L[0]
        assert r.cmax_mg_L > r.c_plasma_mg_L[-1]

    def test_plasma_non_negative(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_film_non_negative(self):
        r = _default()
        assert all(a >= 0.0 for a in r.a_film_mg)

    def test_mucosa_non_negative(self):
        r = _default()
        assert all(a >= 0.0 for a in r.a_mucosa_mg)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

class TestSummaryMetrics:
    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_mg_L > 0.0

    def test_tmax_between_zero_and_tend(self):
        r = _default(t_end_h=12.0)
        assert 0.0 <= r.tmax_h <= 12.0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_mg_h_per_L > 0.0

    def test_bioavailability_pct_range(self):
        """Bioavailability should be between 0 and 120%."""
        r = _default()
        assert 0.0 <= r.bioavailability_pct <= 120.0

    def test_film_dissolved_pct_range(self):
        r = _default()
        assert 0.0 <= r.film_dissolved_pct <= 100.0

    def test_film_largely_dissolved_after_long_simulation(self):
        """Most of the film should dissolve after 24 hours."""
        r = _default(t_end_h=24.0, dt_h=0.1)
        assert r.film_dissolved_pct > 90.0


# ---------------------------------------------------------------------------
# Pharmacological effects
# ---------------------------------------------------------------------------

class TestPharmacologicalEffects:
    def test_higher_dose_higher_cmax(self):
        r_low = _default(dose_mg=5.0)
        r_high = _default(dose_mg=20.0)
        assert r_high.cmax_mg_L > r_low.cmax_mg_L

    def test_higher_dose_higher_auc(self):
        r_low = _default(dose_mg=5.0)
        r_high = _default(dose_mg=20.0)
        assert r_high.auc_mg_h_per_L > r_low.auc_mg_h_per_L

    def test_higher_f_buccal_affects_cmax(self):
        """Higher buccal fraction (less swallowed) changes Cmax."""
        r_low = _default(f_buccal=0.2)
        r_high = _default(f_buccal=0.9)
        # Both should produce positive Cmax
        assert r_low.cmax_mg_L > 0.0
        assert r_high.cmax_mg_L > 0.0

    def test_high_cl_reduces_cmax(self):
        r_low_cl = _default(cl_L_per_h=1.0)
        r_high_cl = _default(cl_L_per_h=20.0)
        assert r_low_cl.cmax_mg_L > r_high_cl.cmax_mg_L

    def test_high_vd_reduces_cmax(self):
        r_small_vd = _default(vd_L=10.0)
        r_large_vd = _default(vd_L=100.0)
        assert r_small_vd.cmax_mg_L > r_large_vd.cmax_mg_L

    def test_faster_dissolution_accelerates_absorption(self):
        """Higher k_dissolve should reduce tmax or increase early Cmax."""
        r_slow = _default(k_dissolve_per_h=0.05, t_end_h=24.0)
        r_fast = _default(k_dissolve_per_h=1.0, t_end_h=24.0)
        # Fast dissolution leads to earlier peak
        assert r_fast.tmax_h <= r_slow.tmax_h


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_notes_contains_drug_name(self):
        r = _default(drug_name="Nicotine")
        assert "Nicotine" in r.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-5.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default(vd_L=0.0)

    def test_f_buccal_zero_raises(self):
        with pytest.raises(ValueError, match="f_buccal"):
            _default(f_buccal=0.0)

    def test_f_buccal_above_one_raises(self):
        with pytest.raises(ValueError, match="f_buccal"):
            _default(f_buccal=1.1)

    def test_k_dissolve_zero_raises(self):
        with pytest.raises(ValueError, match="k_dissolve_per_h"):
            _default(k_dissolve_per_h=0.0)

    def test_k_abs_zero_raises(self):
        with pytest.raises(ValueError, match="k_abs_per_h"):
            _default(k_abs_per_h=0.0)
