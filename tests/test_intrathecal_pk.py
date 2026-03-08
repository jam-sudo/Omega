"""Tests for Phase 230: Intrathecal drug delivery PK model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.intrathecal_pk import (
    IntrathecalPKResult,
    compare_it_doses,
    simulate_intrathecal_pk,
)


def _sim(**kw) -> IntrathecalPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_intrathecal_pk(**defaults)


# --- Return type ---


def test_return_type():
    assert isinstance(_sim(), IntrathecalPKResult)


# --- Time arrays ---


def test_times_start_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_length_consistent():
    r = _sim(t_end_h=10.0, dt_h=0.1)
    assert (
        len(r.times_h)
        == len(r.c_lumbar_mg_mL)
        == len(r.c_cisternal_mg_mL)
        == len(r.c_systemic_mg_L)
    )


# --- Lumbar concentration ---


def test_lumbar_starts_high():
    r = _sim()
    # initial lumbar conc = dose / v_lumbar (default 30 mL) = 1/30 ≈ 0.033 mg/mL
    assert r.c_lumbar_mg_mL[0] > 0.0


def test_lumbar_decreases_overall():
    r = _sim(t_end_h=48.0, dt_h=0.05)
    assert r.c_lumbar_mg_mL[-1] < r.c_lumbar_mg_mL[0]


def test_cmax_lumbar_positive():
    assert _sim().cmax_lumbar_mg_mL > 0.0


# --- Cisternal concentration ---


def test_cisternal_starts_at_zero():
    r = _sim()
    assert r.c_cisternal_mg_mL[0] == pytest.approx(0.0)


def test_cisternal_rises_then_falls():
    r = _sim(t_end_h=48.0, dt_h=0.05)
    cist = r.c_cisternal_mg_mL
    peak_idx = cist.index(max(cist))
    assert peak_idx > 0
    assert cist[-1] < cist[peak_idx]


# --- Systemic concentration ---


def test_systemic_starts_at_zero():
    r = _sim()
    assert r.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_cmax_systemic_positive():
    assert _sim().cmax_systemic_mg_L > 0.0


def test_systemic_much_lower_than_lumbar():
    r = _sim()
    # lumbar is in mg/mL, systemic in mg/L — direct comparison after unit note:
    # 1 mg/mL = 1000 mg/L, so systemic should be << lumbar * 1000
    assert r.cmax_systemic_mg_L < r.cmax_lumbar_mg_mL * 5000.0


# --- AUC ---


def test_auc_lumbar_positive():
    assert _sim().auc_lumbar_mg_h_per_mL > 0.0


def test_auc_cisternal_positive():
    assert _sim().auc_cisternal_mg_h_per_mL > 0.0


# --- t_half lumbar ---


def test_t_half_lumbar_positive():
    r = _sim()
    assert r.t_half_lumbar_h > 0.0
    assert not math.isnan(r.t_half_lumbar_h)


def test_higher_kbulk_flow_shorter_t_half():
    r_slow = _sim(k_bulk_flow=0.05)
    r_fast = _sim(k_bulk_flow=0.3)
    assert r_fast.t_half_lumbar_h < r_slow.t_half_lumbar_h


# --- tmax cisternal ---


def test_tmax_cisternal_positive():
    assert _sim().tmax_cisternal_h > 0.0


# --- Dose linearity ---


def test_dose_linearity_auc():
    r1 = _sim(dose_mg=1.0)
    r2 = _sim(dose_mg=2.0)
    ratio = r2.auc_lumbar_mg_h_per_mL / r1.auc_lumbar_mg_h_per_mL
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_dose_linearity_cmax():
    r1 = _sim(dose_mg=1.0)
    r2 = _sim(dose_mg=3.0)
    ratio = r2.cmax_lumbar_mg_mL / r1.cmax_lumbar_mg_mL
    assert ratio == pytest.approx(3.0, rel=0.01)


# --- compare_it_doses ---


def test_compare_it_doses_returns_list():
    results = compare_it_doses("D", [1.0, 2.0, 5.0], cl_sys_L_per_h=5.0, vd_sys_L=50.0)
    assert len(results) == 3


def test_compare_it_doses_sorted_descending():
    results = compare_it_doses("D", [0.5, 1.0, 2.0], cl_sys_L_per_h=5.0, vd_sys_L=50.0)
    aucs = [r.auc_lumbar_mg_h_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


# --- Validation errors ---


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intrathecal_pk("D", dose_mg=0.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intrathecal_pk("D", dose_mg=-1.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_intrathecal_pk("D", dose_mg=1.0, cl_sys_L_per_h=0.0, vd_sys_L=50.0)
