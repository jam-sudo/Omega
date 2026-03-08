"""Tests for transdermal drug permeation (Phase 560)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.transdermal_permeation import (
    TransdermalPermeationResult,
    simulate_transdermal_permeation,
)

# Common parameters for a typical transdermal drug (moderate logP, small MW)
BASE = dict(
    drug_name="TestDrug",
    dose_mg_per_cm2=1.0,
    application_area_cm2=10.0,
    logP=2.5,
    mw_Da=300.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=48.0,
    dt_h=0.1,
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_transdermal_permeation(**params)


# ---------- Return type and list lengths ----------


def test_return_type():
    r = _run()
    assert isinstance(r, TransdermalPermeationResult)


def test_list_lengths_consistent():
    r = _run()
    n = len(r.times_h)
    assert len(r.flux_mg_per_cm2_per_h) == n
    assert len(r.c_systemic_mg_L) == n
    assert n > 0


def test_times_start_at_zero():
    r = _run()
    assert r.times_h[0] == 0.0


# ---------- Flux behavior ----------


def test_flux_zero_before_lag_time():
    r = _run()
    assert r.lag_time_h > 0
    for i, t in enumerate(r.times_h):
        if t < r.lag_time_h - 0.05:
            assert r.flux_mg_per_cm2_per_h[i] == 0.0


def test_flux_positive_after_lag_time():
    r = _run()
    # Find first index well after lag time
    found_positive = False
    for i, t in enumerate(r.times_h):
        if t > r.lag_time_h + 1.0:
            if r.flux_mg_per_cm2_per_h[i] > 0:
                found_positive = True
                break
    assert found_positive, "Expected positive flux after lag time"


# ---------- Cmax > 0 ----------


def test_cmax_positive():
    r = _run()
    assert r.cmax_systemic_mg_L > 0


def test_tmax_positive():
    r = _run()
    assert r.tmax_systemic_h > 0


# ---------- Higher logP → higher Kp → higher flux ----------


def test_higher_logP_higher_auc():
    # Use low MW to avoid Kp clamping at minimum
    r_low = _run(logP=2.0, mw_Da=150.0)
    r_high = _run(logP=4.0, mw_Da=150.0)
    assert r_high.auc_systemic_mg_h_per_L > r_low.auc_systemic_mg_h_per_L


def test_higher_logP_higher_cmax():
    r_low = _run(logP=2.0, mw_Da=150.0)
    r_high = _run(logP=4.0, mw_Da=150.0)
    assert r_high.cmax_systemic_mg_L > r_low.cmax_systemic_mg_L


# ---------- Higher area → higher AUC ----------


def test_higher_area_higher_auc():
    r_small = _run(application_area_cm2=5.0)
    r_large = _run(application_area_cm2=20.0)
    assert r_large.auc_systemic_mg_h_per_L > r_small.auc_systemic_mg_h_per_L


def test_area_linearity():
    r1 = _run(application_area_cm2=10.0)
    r2 = _run(application_area_cm2=20.0)
    # AUC should roughly double (within 20% tolerance for nonlinear depletion)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert 1.5 < ratio < 2.5


# ---------- total_absorbed <= dose_total ----------


def test_total_absorbed_le_dose():
    r = _run()
    dose_total = r.dose_mg_per_cm2 * r.application_area_cm2
    assert r.total_absorbed_mg <= dose_total + 1e-9


# ---------- Bioavailability in [0, 100] ----------


def test_bioavailability_in_range():
    r = _run()
    assert 0.0 <= r.bioavailability_pct <= 100.0


def test_bioavailability_capped_at_100():
    # Even with extreme parameters, should not exceed 100
    r = _run(t_end_h=200.0, logP=4.0, mw_Da=100.0)
    assert r.bioavailability_pct <= 100.0


# ---------- Lag time ----------


def test_lag_time_positive():
    r = _run()
    assert r.lag_time_h > 0


def test_lag_time_within_bounds():
    r = _run()
    assert 0.01 <= r.lag_time_h <= 24.0


# ---------- Dose linearity ----------


def test_dose_linearity_cmax():
    r1 = _run(dose_mg_per_cm2=1.0)
    r2 = _run(dose_mg_per_cm2=2.0)
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert 1.5 < ratio < 2.5


def test_dose_linearity_auc():
    r1 = _run(dose_mg_per_cm2=1.0)
    r2 = _run(dose_mg_per_cm2=2.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert 1.5 < ratio < 2.5


# ---------- Validation errors ----------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg_per_cm2"):
        _run(dose_mg_per_cm2=0.0)


def test_invalid_area():
    with pytest.raises(ValueError, match="application_area_cm2"):
        _run(application_area_cm2=-1.0)


def test_invalid_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        _run(mw_Da=0.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=-5.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)
