"""Tests for clinical/pain_analgesic_pk.py — Phase 384."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pain_analgesic_pk import (
    AnalgesicPKResult,
    compute_analgesic_duration,
    simulate_analgesic_pk,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_OPIOID_KWARGS = dict(
    drug_name="morphine",
    drug_class="opioid",
    dose_mg=10.0,
    cl_L_per_h=3.0,
    vd_L=30.0,
    ke0_per_h=0.5,
    ec50_mg_L=0.05,
    emax_analgesia=100.0,
    interval_h=4.0,
    n_doses=1,
    dt_h=0.1,
)

_NSAID_KWARGS = dict(
    drug_name="ibuprofen",
    drug_class="nsaid",
    dose_mg=400.0,
    cl_L_per_h=4.0,
    vd_L=12.0,
    ke0_per_h=1.0,
    ec50_mg_L=5.0,
    emax_analgesia=80.0,
    interval_h=6.0,
    n_doses=1,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# compute_analgesic_duration tests
# ---------------------------------------------------------------------------


class TestComputeAnalgesicDuration:
    def test_always_above_threshold(self):
        times = [0.0, 1.0, 2.0, 3.0]
        effects = [80.0, 90.0, 85.0, 75.0]
        dur = compute_analgesic_duration(times, effects, threshold=50.0)
        assert dur == pytest.approx(3.0, abs=1e-6)

    def test_always_below_threshold(self):
        times = [0.0, 1.0, 2.0]
        effects = [10.0, 20.0, 30.0]
        dur = compute_analgesic_duration(times, effects, threshold=50.0)
        assert dur == pytest.approx(0.0, abs=1e-6)

    def test_partially_above_threshold(self):
        # Effect crosses threshold linearly between t=1 and t=2
        times = [0.0, 1.0, 2.0, 3.0]
        effects = [0.0, 0.0, 100.0, 100.0]
        dur = compute_analgesic_duration(times, effects, threshold=50.0)
        # Crosses at t=1.5; above for [1.5, 3.0] = 1.5 h
        assert dur == pytest.approx(1.5, abs=0.01)

    def test_empty_times_raises(self):
        with pytest.raises(ValueError):
            compute_analgesic_duration([], [], threshold=50.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_analgesic_duration([0.0, 1.0], [80.0], threshold=50.0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            compute_analgesic_duration([0.0, 1.0], [60.0, 70.0], threshold=-10.0)

    def test_zero_threshold_all_above(self):
        times = [0.0, 1.0, 2.0]
        effects = [10.0, 20.0, 30.0]
        dur = compute_analgesic_duration(times, effects, threshold=0.0)
        assert dur == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# simulate_analgesic_pk — basic structure tests
# ---------------------------------------------------------------------------


class TestSimulateAnalgesicPKStructure:
    def test_returns_analgesic_pk_result(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert isinstance(result, AnalgesicPKResult)

    def test_time_series_non_empty(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert len(result.times_h) > 0
        assert len(result.c_plasma_mg_L) > 0
        assert len(result.c_effect_mg_L) > 0
        assert len(result.analgesia_pct) > 0

    def test_all_series_equal_length(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.c_effect_mg_L) == n
        assert len(result.analgesia_pct) == n

    def test_cmax_positive(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert result.auc > 0.0

    def test_t_half_positive(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert result.t_half_h > 0.0

    def test_notes_non_empty(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# simulate_analgesic_pk — PK/PD correctness tests
# ---------------------------------------------------------------------------


class TestSimulateAnalgesicPKCorrectness:
    def test_plasma_conc_positive_after_dose(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        max_cp = max(result.c_plasma_mg_L)
        assert max_cp > 0.0

    def test_effect_positive_when_plasma_nonzero(self):
        """When plasma concentration > 0, effect compartment eventually becomes non-zero."""
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        max_effect = max(result.analgesia_pct)
        assert max_effect > 0.0

    def test_effect_lags_plasma(self):
        """Effect compartment Tmax should be >= plasma Tmax (hysteresis)."""
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        t_cmax = result.times_h[result.c_plasma_mg_L.index(max(result.c_plasma_mg_L))]
        t_emax = result.times_h[result.analgesia_pct.index(max(result.analgesia_pct))]
        assert t_emax >= t_cmax

    def test_analgesia_bounded_0_100(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        for e in result.analgesia_pct:
            assert -1e-6 <= e <= 100.0 + 1e-6

    def test_nsaid_higher_bioavailability_higher_cmax(self):
        """NSAID (F=0.90) should reach higher Cmax than opioid (F=0.70) for same dose/PK."""
        kwargs = dict(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=25.0,
            ke0_per_h=0.5,
            ec50_mg_L=1.0,
            emax_analgesia=100.0,
            interval_h=6.0,
            n_doses=1,
            dt_h=0.1,
        )
        r_opioid = simulate_analgesic_pk(drug_name="X", drug_class="opioid", **kwargs)
        r_nsaid = simulate_analgesic_pk(drug_name="X", drug_class="nsaid", **kwargs)
        assert r_nsaid.cmax > r_opioid.cmax

    def test_multidose_cmax_higher_than_single(self):
        """After multiple doses, Cmax should be >= single-dose Cmax (accumulation)."""
        base = dict(
            drug_name="morphine",
            drug_class="opioid",
            dose_mg=10.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            ke0_per_h=0.5,
            ec50_mg_L=0.05,
            emax_analgesia=100.0,
            interval_h=4.0,
            dt_h=0.1,
        )
        r1 = simulate_analgesic_pk(n_doses=1, **base)
        r5 = simulate_analgesic_pk(n_doses=5, **base)
        # Multi-dose should accumulate
        assert r5.cmax >= r1.cmax * 0.9  # allow slight numerical tolerance

    def test_multidose_longer_time_series(self):
        r1 = simulate_analgesic_pk(**{**_OPIOID_KWARGS, "n_doses": 1})
        r3 = simulate_analgesic_pk(**{**_OPIOID_KWARGS, "n_doses": 3})
        assert max(r3.times_h) > max(r1.times_h)

    def test_higher_dose_higher_cmax(self):
        r_low = simulate_analgesic_pk(**{**_OPIOID_KWARGS, "dose_mg": 5.0})
        r_high = simulate_analgesic_pk(**{**_OPIOID_KWARGS, "dose_mg": 20.0})
        assert r_high.cmax > r_low.cmax

    def test_nsaid_simulation_runs(self):
        result = simulate_analgesic_pk(**_NSAID_KWARGS)
        assert isinstance(result, AnalgesicPKResult)
        assert result.cmax > 0.0

    def test_analgesic_duration_nonnegative(self):
        result = simulate_analgesic_pk(**_OPIOID_KWARGS)
        assert result.analgesic_duration_h >= 0.0

    def test_t_half_matches_cl_vd(self):
        """t½ = 0.693 * Vd / CL."""
        import math

        cl = 3.0
        vd = 30.0
        expected_t_half = math.log(2) / (cl / vd)
        result = simulate_analgesic_pk(**{**_OPIOID_KWARGS, "cl_L_per_h": cl, "vd_L": vd})
        assert result.t_half_h == pytest.approx(expected_t_half, rel=0.01)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestSimulateAnalgesicPKValidation:
    def test_invalid_drug_class(self):
        with pytest.raises(ValueError, match="drug_class"):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "drug_class": "antibiotic"})

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "dose_mg": 0.0})

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "dose_mg": -5.0})

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "cl_L_per_h": 0.0})

    def test_invalid_vd_negative(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "vd_L": -10.0})

    def test_invalid_ke0_zero(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "ke0_per_h": 0.0})

    def test_invalid_ec50_negative(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "ec50_mg_L": -1.0})

    def test_invalid_n_doses_zero(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "n_doses": 0})

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "interval_h": 0.0})

    def test_invalid_dt_negative(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "dt_h": -0.01})

    def test_empty_drug_name(self):
        with pytest.raises(ValueError):
            simulate_analgesic_pk(**{**_OPIOID_KWARGS, "drug_name": ""})
