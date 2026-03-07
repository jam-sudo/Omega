"""Tests for Phase 354 — Multi-Dose Accumulation Calculator."""

import pytest

from omega_pbpk.clinical.multi_dose_accumulation import (
    MultiDoseAccumulationResult,
    simulate_multi_dose_accumulation,
)

# Common defaults: IV bolus
DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    ka_per_h=None,
    f_oral=1.0,
    tau_h=12.0,
    n_doses=10,
    t_end_h=120.0,
    dt_h=0.05,
)


def _sim(**overrides):
    kw = dict(DEFAULT_KWARGS)
    kw.update(overrides)
    return simulate_multi_dose_accumulation(**kw)


# --- Return type and structure ---


def test_return_type():
    res = _sim()
    assert isinstance(res, MultiDoseAccumulationResult)


def test_array_lengths_match():
    res = _sim()
    assert len(res.times_h) == len(res.c_plasma_mg_L)


def test_times_start_at_zero():
    res = _sim()
    assert res.times_h[0] == 0.0


def test_dose_times_length_equals_n_doses():
    res = _sim(n_doses=7)
    assert len(res.dose_times_h) == 7


def test_first_dose_at_t_zero():
    res = _sim()
    assert res.dose_times_h[0] == 0.0


# --- Pharmacokinetic properties ---


def test_plasma_rises_with_each_dose():
    """Concentration should increase after each dose during accumulation."""
    res = _sim(n_doses=5, tau_h=6.0, t_end_h=60.0, dt_h=0.05)
    times = res.times_h
    conc = res.c_plasma_mg_L
    tau = 6.0
    idx1 = next(i for i, t in enumerate(times) if t >= tau * 0.5)
    idx2 = next(i for i, t in enumerate(times) if t >= tau * 1.5)
    assert conc[idx2] > conc[idx1]


def test_auc_ss_ge_auc_single():
    res = _sim()
    assert res.auc_ss_mg_h_per_L >= res.auc_single_dose_mg_h_per_L


def test_cmax_ss_ge_cmax_single():
    res = _sim()
    assert res.cmax_ss_mg_L >= res.cmax_single_mg_L


def test_accumulation_ratio_ge_one():
    res = _sim()
    assert res.accumulation_ratio >= 1.0


def test_t_half_positive():
    res = _sim()
    assert res.t_half_h > 0.0


def test_t_half_formula():
    """t_half = 0.693147 / ke, ke = cl/vd."""
    res = _sim(cl_L_per_h=5.0, vd_L=50.0)
    expected = 0.693147 / (5.0 / 50.0)
    assert abs(res.t_half_h - expected) < 0.001


def test_doses_to_steady_state_ge_one():
    res = _sim()
    assert res.doses_to_steady_state >= 1


# --- Tau effect on accumulation ---


def test_longer_tau_lower_accumulation():
    """Longer dosing interval → less accumulation."""
    short_tau = _sim(tau_h=6.0, t_end_h=120.0, n_doses=15).accumulation_ratio
    long_tau = _sim(tau_h=24.0, t_end_h=240.0, n_doses=10).accumulation_ratio
    assert short_tau > long_tau


# --- Dose linearity ---


def test_dose_linearity_cmax_single():
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    ratio = r2.cmax_single_mg_L / r1.cmax_single_mg_L
    assert abs(ratio - 2.0) < 0.05


def test_dose_linearity_auc_single():
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    ratio = r2.auc_single_dose_mg_h_per_L / r1.auc_single_dose_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05


# --- Oral route ---


def test_oral_route_runs():
    res = _sim(ka_per_h=1.5, f_oral=0.8, dose_mg=100.0)
    assert isinstance(res, MultiDoseAccumulationResult)


def test_oral_cmax_lower_than_iv():
    """Oral Cmax < IV Cmax for same dose (absorption limits peak)."""
    iv_cmax = _sim(ka_per_h=None, dose_mg=100.0).cmax_single_mg_L
    oral_cmax = _sim(ka_per_h=1.5, f_oral=1.0, dose_mg=100.0).cmax_single_mg_L
    assert oral_cmax < iv_cmax


def test_oral_f_oral_scales_cmax():
    r_full = _sim(ka_per_h=1.5, f_oral=1.0, dose_mg=100.0)
    r_half = _sim(ka_per_h=1.5, f_oral=0.5, dose_mg=100.0)
    assert r_full.cmax_single_mg_L > r_half.cmax_single_mg_L


# --- Validation errors ---


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-1.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _sim(cl_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        _sim(vd_L=0.0)


def test_error_tau_zero():
    with pytest.raises(ValueError, match="tau_h"):
        _sim(tau_h=0.0)


def test_error_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        _sim(n_doses=0)
