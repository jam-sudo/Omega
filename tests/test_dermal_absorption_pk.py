"""Tests for Phase 974 — dermal_absorption_pk."""

import pytest

from omega_pbpk.core.dermal_absorption_pk import (
    DermalAbsorptionResult,
    simulate_dermal_absorption,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_dermal_absorption_result():
    result = simulate_dermal_absorption("TestDrug", logp=2.0, mw=300.0)
    assert isinstance(result, DermalAbsorptionResult)


# ---------------------------------------------------------------------------
# cmax > 0
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0)
    assert result.cmax_dermis_mg_L > 0.0


# ---------------------------------------------------------------------------
# f_absorbed in [0, 1]
# ---------------------------------------------------------------------------


def test_f_absorbed_in_range():
    result = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_f_absorbed_in_range_hydrophilic():
    result = simulate_dermal_absorption("Drug", logp=-1.0, mw=200.0)
    assert 0.0 <= result.f_absorbed <= 1.0


# ---------------------------------------------------------------------------
# Lipophilic > hydrophilic absorption
# ---------------------------------------------------------------------------


def test_lipophilic_higher_f_absorbed_than_hydrophilic():
    lipo = simulate_dermal_absorption("Lipo", logp=3.0, mw=300.0, t_end_h=48.0)
    hydro = simulate_dermal_absorption("Hydro", logp=-1.0, mw=300.0, t_end_h=48.0)
    assert lipo.f_absorbed > hydro.f_absorbed


# ---------------------------------------------------------------------------
# AUC > 0
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0)
    assert result.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Plasma starts near zero
# ---------------------------------------------------------------------------


def test_plasma_starts_near_zero():
    result = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0)
    assert result.c_dermis_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Lag time increases with MW
# ---------------------------------------------------------------------------


def test_lag_time_increases_with_mw():
    r_low = simulate_dermal_absorption("LowMW", logp=2.0, mw=100.0)
    r_high = simulate_dermal_absorption("HighMW", logp=2.0, mw=600.0)
    assert r_high.lag_time_h > r_low.lag_time_h


def test_lag_time_capped_at_6h():
    """Very large MW should be capped at 6 h lag time."""
    result = simulate_dermal_absorption("HeavyDrug", logp=2.0, mw=100000.0)
    assert result.lag_time_h == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Linear PK: higher dose → proportionally higher Cmax
# ---------------------------------------------------------------------------


def test_higher_dose_higher_cmax():
    r1 = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, dose_mg_per_cm2=1.0)
    r2 = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, dose_mg_per_cm2=2.0)
    assert r2.cmax_dermis_mg_L > r1.cmax_dermis_mg_L


def test_cmax_scales_linearly_with_dose():
    r1 = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, dose_mg_per_cm2=1.0)
    r2 = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, dose_mg_per_cm2=3.0)
    ratio = r2.cmax_dermis_mg_L / r1.cmax_dermis_mg_L
    assert ratio == pytest.approx(3.0, rel=0.05)


# ---------------------------------------------------------------------------
# Larger area → higher Cmax
# ---------------------------------------------------------------------------


def test_larger_area_higher_cmax():
    r_small = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, area_cm2=50.0)
    r_large = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, area_cm2=200.0)
    assert r_large.cmax_dermis_mg_L > r_small.cmax_dermis_mg_L


# ---------------------------------------------------------------------------
# List lengths consistent
# ---------------------------------------------------------------------------


def test_list_lengths_consistent():
    result = simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, t_end_h=12.0, dt_h=0.5)
    assert len(result.times_h) == len(result.c_dermis_mg_L)
    assert len(result.times_h) == len(result.c_skin_barrier_mg_cm2)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        simulate_dermal_absorption("Drug", logp=2.0, mw=0.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        simulate_dermal_absorption("Drug", logp=2.0, mw=-100.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, vd_L=0.0)


def test_invalid_area_zero():
    with pytest.raises(ValueError, match="area_cm2"):
        simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, area_cm2=0.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_dermal_absorption("Drug", logp=2.0, mw=300.0, cl_L_per_h=0.0)
