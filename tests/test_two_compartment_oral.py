"""Tests for core/two_compartment_oral.py — Phase 591."""

import math

import pytest

from omega_pbpk.core.two_compartment_oral import (
    TwoCompartmentOralResult,
    compute_vss,
    estimate_terminal_half_life,
    simulate_two_cpt_oral,
)

# Shared default parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd1_L=20.0,
    vd2_L=30.0,
    q12_L_per_h=3.0,
    route="oral",
    ka_per_h=1.5,
    f_oral=1.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def run(**kwargs):
    params = {**DEFAULTS, **kwargs}
    return simulate_two_cpt_oral(**params)


# ---- Basic return type ----


def test_returns_result():
    res = run()
    assert isinstance(res, TwoCompartmentOralResult)


# ---- Oral-specific ----


def test_oral_tmax_positive():
    res = run(route="oral")
    assert res.tmax_h > 0


def test_oral_absorption_rises():
    """C1 should rise from 0 before reaching Tmax."""
    res = run(route="oral")
    tmax_idx = res.times_h.index(res.tmax_h)
    assert tmax_idx > 0
    assert res.c_central_mg_L[0] == 0.0


# ---- IV-specific ----


def test_iv_c_max_at_t0():
    res = run(route="iv", dose_mg=100.0)
    expected_c0 = 100.0 / DEFAULTS["vd1_L"]
    assert abs(res.c_central_mg_L[0] - expected_c0) < 1e-9


def test_iv_no_peripheral_initially():
    res = run(route="iv")
    assert res.c_peripheral_mg_L[0] == 0.0


# ---- VSS ----


def test_vss_correct():
    res = run(vd1_L=20.0, vd2_L=30.0)
    assert abs(res.vss_L - 50.0) < 1e-9


def test_compute_vss():
    assert abs(compute_vss(10.0, 20.0) - 30.0) < 1e-9


# ---- Dose proportionality ----


def test_dose_proportionality_cmax():
    res1 = run(dose_mg=100.0)
    res2 = run(dose_mg=200.0)
    ratio = res2.cmax_mg_L / res1.cmax_mg_L
    assert abs(ratio - 2.0) < 0.05


def test_dose_proportionality_auc():
    res1 = run(dose_mg=100.0)
    res2 = run(dose_mg=200.0)
    ratio = res2.auc_mg_h_L / res1.auc_mg_h_L
    assert abs(ratio - 2.0) < 0.05


# ---- Peripheral compartment ----


def test_peripheral_conc_nonzero():
    """After equilibration, peripheral > 0."""
    res = run(t_end_h=24.0)
    # Check that at least one peripheral concentration is meaningfully positive
    max_peripheral = max(res.c_peripheral_mg_L)
    assert max_peripheral > 0.01


# ---- f_oral ----


def test_f_oral_reduces_cmax():
    res_full = run(f_oral=1.0)
    res_half = run(f_oral=0.5)
    assert res_half.cmax_mg_L < res_full.cmax_mg_L


# ---- q12 distribution speed ----


def test_higher_q12_faster_distribution():
    """Higher Q12 → faster rise in peripheral compartment."""
    res_low = run(q12_L_per_h=1.0, t_end_h=6.0)
    res_high = run(q12_L_per_h=20.0, t_end_h=6.0)
    # At early times, higher Q12 → more peripheral
    idx_6h = min(len(res_low.c_peripheral_mg_L) - 1, 60)
    assert res_high.c_peripheral_mg_L[idx_6h] >= res_low.c_peripheral_mg_L[idx_6h]


# ---- Terminal half-life ----


def test_terminal_half_life_positive():
    res = run(t_end_h=48.0)
    assert res.t_half_terminal_h > 0


# ---- Time / length checks ----


def test_times_start_zero():
    res = run()
    assert res.times_h[0] == 0.0


def test_lengths_match():
    res = run()
    assert len(res.c_central_mg_L) == len(res.c_peripheral_mg_L) == len(res.times_h)


# ---- AUC ----


def test_auc_positive():
    res = run()
    assert res.auc_mg_h_L > 0


# ---- Notes ----


def test_notes_nonempty():
    res = run()
    assert len(res.notes) > 0


# ---- Validation errors ----


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        run(dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        run(cl_L_per_h=-1.0)


def test_invalid_vd1_raises():
    with pytest.raises(ValueError):
        run(vd1_L=0.0)


def test_invalid_vd2_raises():
    with pytest.raises(ValueError):
        run(vd2_L=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        run(route="subcutaneous")


def test_invalid_f_oral_raises():
    with pytest.raises(ValueError):
        run(f_oral=1.5)


# ---- estimate_terminal_half_life standalone ----


def test_estimate_terminal_half_life_basic():
    """Simple monoexponential decay should yield approximately correct t½."""
    lam = 0.2  # per hour
    t_half_expected = math.log(2) / lam
    times = [float(i) for i in range(0, 50)]
    concs = [10.0 * math.exp(-lam * t) for t in times]
    t_half = estimate_terminal_half_life(times, concs, frac=0.3)
    assert abs(t_half - t_half_expected) / t_half_expected < 0.05
