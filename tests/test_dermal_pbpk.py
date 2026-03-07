"""Tests for Phase 820 — Dermal Absorption PBPK (dermal_pbpk.py)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.dermal_pbpk import (
    DermalPKResult,
    estimate_kp,
    simulate_dermal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs) -> DermalPKResult:
    defaults = dict(
        drug_name="TestCompound",
        dose_mg_per_cm2=0.5,
        application_area_cm2=100.0,
        kp_cm_per_h=0.005,
        cl_L_per_h=5.0,
        vd_L=50.0,
        lag_time_h=1.0,
        f_sc_retention=0.1,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_dermal_absorption(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _sim()
    assert isinstance(result, DermalPKResult)


def test_drug_name_stored():
    result = _sim(drug_name="Estradiol")
    assert result.drug_name == "Estradiol"


def test_dose_per_cm2_stored():
    result = _sim(dose_mg_per_cm2=1.0)
    assert result.dose_mg_per_cm2 == pytest.approx(1.0)


def test_area_stored():
    result = _sim(application_area_cm2=50.0)
    assert result.application_area_cm2 == pytest.approx(50.0)


def test_notes_is_nonempty_string():
    result = _sim()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Total dose calculation
# ---------------------------------------------------------------------------


def test_total_dose_equals_dose_per_cm2_times_area():
    dose_per = 0.3
    area = 80.0
    result = _sim(dose_mg_per_cm2=dose_per, application_area_cm2=area)
    assert result.total_dose_mg == pytest.approx(dose_per * area, rel=1e-9)


def test_sc_retention_correct():
    dose_per = 0.5
    area = 100.0
    f_sc = 0.15
    result = _sim(dose_mg_per_cm2=dose_per, application_area_cm2=area, f_sc_retention=f_sc)
    expected_sc = dose_per * area * f_sc
    assert result.stratum_corneum_retention_mg == pytest.approx(expected_sc, rel=1e-9)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive_for_reasonable_kp():
    result = _sim(kp_cm_per_h=0.01, lag_time_h=0.5, t_end_h=24.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = _sim(kp_cm_per_h=0.01, lag_time_h=0.5, t_end_h=24.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_nonnegative():
    result = _sim()
    assert result.tmax_h >= 0.0


def test_lag_time_stored():
    result = _sim(lag_time_h=3.0)
    assert result.lag_time_h == pytest.approx(3.0)


def test_steady_state_flux_calculation():
    kp = 0.002
    dose_per = 0.5
    result = _sim(kp_cm_per_h=kp, dose_mg_per_cm2=dose_per)
    expected_flux = kp * dose_per
    assert result.steady_state_flux_mg_per_h_per_cm2 == pytest.approx(expected_flux, rel=1e-9)


# ---------------------------------------------------------------------------
# Time-series consistency
# ---------------------------------------------------------------------------


def test_times_and_concs_same_length():
    result = _sim(t_end_h=12.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_start_at_zero():
    result = _sim()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-10)


def test_all_concentrations_nonnegative():
    result = _sim()
    for c in result.c_plasma_mg_L:
        assert c >= 0.0


# ---------------------------------------------------------------------------
# Lag-time behaviour
# ---------------------------------------------------------------------------


def test_lag_time_delays_absorption():
    """Plasma concentration near zero during lag phase."""
    lag = 4.0
    result = _sim(lag_time_h=lag, t_end_h=24.0, dt_h=0.05)
    # All time points strictly before the lag should show ~zero plasma
    for t, c in zip(result.times_h, result.c_plasma_mg_L):
        if t < lag - 0.1:  # small tolerance for time step alignment
            assert c == pytest.approx(0.0, abs=1e-9), (
                f"Expected ~0 at t={t:.2f} h (before lag {lag} h), got {c}"
            )


def test_longer_lag_delays_tmax():
    """Longer lag time should produce a later or equal tmax."""
    short_lag = _sim(lag_time_h=0.5)
    long_lag = _sim(lag_time_h=4.0)
    assert long_lag.tmax_h >= short_lag.tmax_h


# ---------------------------------------------------------------------------
# Linear PK (dose proportionality)
# ---------------------------------------------------------------------------


def test_linear_pk_cmax_double_dose():
    """Doubling applied dose should approximately double Cmax."""
    r1 = _sim(dose_mg_per_cm2=0.5)
    r2 = _sim(dose_mg_per_cm2=1.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-3)


def test_linear_pk_auc_double_dose():
    """Doubling applied dose should approximately double AUC."""
    r1 = _sim(dose_mg_per_cm2=0.5)
    r2 = _sim(dose_mg_per_cm2=1.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-3)


# ---------------------------------------------------------------------------
# estimate_kp
# ---------------------------------------------------------------------------


def test_estimate_kp_returns_float():
    kp = estimate_kp(logp=2.0, mw_da=200.0)
    assert isinstance(kp, float)


def test_estimate_kp_high_logp_low_mw_higher_than_low_logp_high_mw():
    """High logP / low MW → higher permeability than low logP / high MW."""
    kp_high = estimate_kp(logp=4.0, mw_da=200.0)
    kp_low = estimate_kp(logp=0.0, mw_da=500.0)
    assert kp_high > kp_low


def test_estimate_kp_clamped_above_minimum():
    """Very low logP / high MW should still give the minimum Kp."""
    kp = estimate_kp(logp=-5.0, mw_da=2000.0)
    assert kp >= 1e-6


def test_estimate_kp_clamped_below_maximum():
    """Very high logP / low MW should still give at most the maximum Kp."""
    kp = estimate_kp(logp=10.0, mw_da=50.0)
    assert kp <= 0.1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg_per_cm2"):
        _sim(dose_mg_per_cm2=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg_per_cm2"):
        _sim(dose_mg_per_cm2=-1.0)


def test_raises_on_zero_kp():
    with pytest.raises(ValueError, match="kp_cm_per_h"):
        _sim(kp_cm_per_h=0.0)


def test_raises_on_zero_area():
    with pytest.raises(ValueError, match="application_area_cm2"):
        _sim(application_area_cm2=0.0)
