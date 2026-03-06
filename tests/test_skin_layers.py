"""Tests for drug permeation through skin layers — Phase 232."""

from __future__ import annotations

import pytest

from omega_pbpk.core.skin_layers import (
    SkinLayerResult,
    lag_time,
    potts_guy_psc,
    simulate_skin_layers,
)


# ── potts_guy_psc ─────────────────────────────────────────────────────


def test_potts_guy_higher_logP_higher_psc():
    """Within a reasonable MW range, higher logP → higher P_sc."""
    p_low = potts_guy_psc(logP=0.0, mw=300.0)
    p_mid = potts_guy_psc(logP=2.0, mw=300.0)
    p_high = potts_guy_psc(logP=4.0, mw=300.0)
    assert p_low < p_mid < p_high


def test_potts_guy_high_mw_lower_psc():
    """High MW → lower P_sc (large molecules penetrate poorly)."""
    p_small = potts_guy_psc(logP=2.0, mw=100.0)
    p_large = potts_guy_psc(logP=2.0, mw=800.0)
    assert p_small > p_large


def test_potts_guy_positive():
    """P_sc must always be positive."""
    assert potts_guy_psc(logP=-2.0, mw=500.0) > 0.0
    assert potts_guy_psc(logP=3.0, mw=300.0) > 0.0


def test_potts_guy_formula():
    """Verify formula: log(P_sc) = -6.3 + 0.71*logP - 0.0061*MW."""
    logP, mw = 2.0, 300.0
    expected = 10 ** (-6.3 + 0.71 * logP - 0.0061 * mw)
    assert abs(potts_guy_psc(logP, mw) - expected) < 1e-15


# ── lag_time ─────────────────────────────────────────────────────────


def test_lag_time_smaller_psc_longer_lag():
    """Smaller P_sc → longer lag time (h_sc²/(6*P_sc))."""
    lag_fast = lag_time(p_sc_cm_h=1e-3)
    lag_slow = lag_time(p_sc_cm_h=1e-5)
    assert lag_slow > lag_fast


def test_lag_time_formula():
    """Verify lag = h² / (6 * P_sc)."""
    p = 1e-4
    h = 0.001
    expected = h**2 / (6.0 * p)
    assert abs(lag_time(p_sc_cm_h=p, h_sc_cm=h) - expected) < 1e-15


def test_lag_time_invalid_psc():
    """p_sc_cm_h <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="p_sc_cm_h"):
        lag_time(p_sc_cm_h=0.0)


def test_lag_time_positive():
    """Lag time must be positive."""
    assert lag_time(p_sc_cm_h=1e-4) > 0.0


# ── simulate_skin_layers ─────────────────────────────────────────────


def _default_sim(**kwargs):
    """Helper: run simulation with default sensible parameters."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg_cm2=1.0,
        application_area_cm2=10.0,
        logP=2.0,
        mw=300.0,
        cl_systemic_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_skin_layers(**defaults)


def test_simulate_f_absorbed_in_range():
    """f_absorbed_systemic must be in [0, 1]."""
    result = _default_sim()
    assert 0.0 <= result.f_absorbed_systemic <= 1.0


def test_simulate_f_metabolized_in_range():
    """f_metabolized_skin must be in [0, 1]."""
    result = _default_sim()
    assert 0.0 <= result.f_metabolized_skin <= 1.0


def test_simulate_c_plasma_starts_near_zero():
    """Plasma concentration at t=0 is 0 (drug starts in SC)."""
    result = _default_sim()
    assert result.c_plasma_mg_L[0] == 0.0


def test_simulate_high_cl_skin_more_metabolism():
    """Higher cl_skin → more skin metabolism → less systemic absorption."""
    res_low = _default_sim(cl_skin_per_h=0.001)
    res_high = _default_sim(cl_skin_per_h=1.0)
    assert res_high.f_metabolized_skin > res_low.f_metabolized_skin


def test_simulate_higher_logP_more_penetration():
    """Higher logP → higher P_sc → more SC penetration (higher AUC or Cmax)."""
    res_low = _default_sim(logP=0.0)
    res_high = _default_sim(logP=3.0)
    # Higher logP means more SC penetration → higher AUC
    assert res_high.auc_mg_h_per_L >= res_low.auc_mg_h_per_L


def test_simulate_result_types():
    """Result fields have correct types."""
    result = _default_sim()
    assert isinstance(result, SkinLayerResult)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.p_sc_cm_h, float)
    assert isinstance(result.lag_time_h, float)
    assert isinstance(result.cmax_mg_L, float)
    assert isinstance(result.tmax_h, float)
    assert isinstance(result.auc_mg_h_per_L, float)
    assert isinstance(result.f_absorbed_systemic, float)
    assert isinstance(result.f_metabolized_skin, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.m_sc_mg, list)
    assert isinstance(result.m_ve_mg, list)
    assert isinstance(result.m_dermis_mg, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.notes, list)


def test_simulate_invalid_dose():
    """dose_mg_cm2 <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="dose_mg_cm2"):
        _default_sim(dose_mg_cm2=0.0)


def test_simulate_invalid_area():
    """application_area_cm2 <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="application_area_cm2"):
        _default_sim(application_area_cm2=0.0)


def test_simulate_invalid_cl():
    """cl_systemic_L_per_h <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="cl_systemic_L_per_h"):
        _default_sim(cl_systemic_L_per_h=0.0)


def test_simulate_invalid_vd():
    """vd_L <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="vd_L"):
        _default_sim(vd_L=0.0)


def test_simulate_2x_area_approx_2x_auc():
    """Doubling application area approximately doubles AUC."""
    res_1x = _default_sim(application_area_cm2=10.0)
    res_2x = _default_sim(application_area_cm2=20.0)
    # AUC should roughly double (within 50% tolerance due to non-linear effects)
    ratio = res_2x.auc_mg_h_per_L / (res_1x.auc_mg_h_per_L + 1e-12)
    assert 1.0 < ratio < 4.0


def test_simulate_lag_time_stored():
    """Lag time is stored and corresponds to Potts-Guy P_sc."""
    result = _default_sim(logP=2.0, mw=300.0)
    p_sc = potts_guy_psc(logP=2.0, mw=300.0)
    expected_lag = (0.001**2) / (6.0 * p_sc)
    assert abs(result.lag_time_h - expected_lag) < 1e-10


def test_simulate_p_sc_stored():
    """P_sc in result matches potts_guy_psc."""
    result = _default_sim(logP=2.0, mw=300.0)
    expected = potts_guy_psc(logP=2.0, mw=300.0)
    assert abs(result.p_sc_cm_h - expected) < 1e-15


def test_simulate_mass_lists_length():
    """All time-series lists have the same length."""
    result = _default_sim(t_end_h=10.0, dt_h=0.5)
    n = len(result.times_h)
    assert len(result.m_sc_mg) == n
    assert len(result.m_ve_mg) == n
    assert len(result.m_dermis_mg) == n
    assert len(result.c_plasma_mg_L) == n
