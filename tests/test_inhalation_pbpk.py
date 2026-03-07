"""Tests for core/inhalation_pbpk.py — Phase 633."""

import pytest

from omega_pbpk.core.inhalation_pbpk import (
    InhalationPKResult,
    compare_inhalation_devices,
    simulate_inhalation_pk,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_inhalation_pk_result():
    result = simulate_inhalation_pk("salbutamol", dose_mg=0.1)
    assert isinstance(result, InhalationPKResult)


def test_field_types():
    result = simulate_inhalation_pk("salbutamol", dose_mg=0.1)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.dose_mg, float)
    assert isinstance(result.f_lung_abs, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.cmax_mg_L, float)
    assert isinstance(result.tmax_h, float)
    assert isinstance(result.auc_mg_h_per_L, float)
    assert isinstance(result.t_half_h, float)
    assert isinstance(result.f_systemic, float)
    assert isinstance(result.lung_auc, float)
    assert isinstance(result.notes, str)


def test_cmax_positive():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0)
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0)
    assert result.auc_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Pharmacological sensitivity tests
# ---------------------------------------------------------------------------


def test_higher_f_lung_abs_gives_higher_cmax():
    res_high = simulate_inhalation_pk("drug", dose_mg=1.0, f_lung_abs=0.9)
    res_low = simulate_inhalation_pk("drug", dose_mg=1.0, f_lung_abs=0.3)
    assert res_high.cmax_mg_L > res_low.cmax_mg_L


def test_higher_k_lung_abs_gives_earlier_tmax():
    res_fast = simulate_inhalation_pk("drug", dose_mg=1.0, k_lung_abs_per_h=20.0)
    res_slow = simulate_inhalation_pk("drug", dose_mg=1.0, k_lung_abs_per_h=1.0)
    assert res_fast.tmax_h < res_slow.tmax_h


def test_zero_f_swallowed_no_crash():
    result = simulate_inhalation_pk("drug", dose_mg=1.0, f_swallowed=0.0)
    assert result.cmax_mg_L > 0


def test_double_dose_doubles_cmax():
    res1 = simulate_inhalation_pk("drug", dose_mg=1.0)
    res2 = simulate_inhalation_pk("drug", dose_mg=2.0)
    ratio = res2.cmax_mg_L / res1.cmax_mg_L
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2x Cmax, got ratio={ratio:.3f}"


def test_double_dose_doubles_auc():
    res1 = simulate_inhalation_pk("drug", dose_mg=1.0)
    res2 = simulate_inhalation_pk("drug", dose_mg=2.0)
    ratio = res2.auc_mg_h_per_L / res1.auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2x AUC, got ratio={ratio:.3f}"


def test_t_half_positive():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    assert result.t_half_h > 0


def test_f_systemic_in_range():
    result = simulate_inhalation_pk(
        "drug", dose_mg=1.0, f_lung_abs=0.9, f_swallowed=0.1, f_oral_swallowed=0.2
    )
    assert 0 < result.f_systemic <= 1.0


# ---------------------------------------------------------------------------
# compare_inhalation_devices tests
# ---------------------------------------------------------------------------


def test_compare_devices_returns_correct_length():
    devices = [
        {"f_lung_abs": 0.9},
        {"f_lung_abs": 0.5},
        {"f_lung_abs": 0.3},
    ]
    results = compare_inhalation_devices("drug", dose_mg=1.0, device_params=devices)
    assert len(results) == 3


def test_compare_devices_higher_f_lung_abs_higher_cmax():
    devices = [
        {"f_lung_abs": 0.9},
        {"f_lung_abs": 0.3},
    ]
    results = compare_inhalation_devices("drug", dose_mg=1.0, device_params=devices)
    assert results[0].cmax_mg_L > results[1].cmax_mg_L


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_inhalation_pk("drug", dose_mg=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_inhalation_pk("drug", dose_mg=1.0, cl_L_per_h=0.0)


def test_validation_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_inhalation_pk("drug", dose_mg=1.0, vd_L=-1.0)


def test_validation_f_lung_abs_too_large_raises():
    with pytest.raises(ValueError, match="f_lung_abs"):
        simulate_inhalation_pk("drug", dose_mg=1.0, f_lung_abs=1.5)


# ---------------------------------------------------------------------------
# Profile shape tests
# ---------------------------------------------------------------------------


def test_plasma_rises_then_falls():
    result = simulate_inhalation_pk(
        "drug", dose_mg=1.0, k_lung_abs_per_h=10.0, cl_L_per_h=5.0, t_end_h=12.0
    )
    conc = result.c_plasma_mg_L
    max_idx = conc.index(max(conc))
    # concentration before peak should be lower than peak
    assert conc[0] < conc[max_idx]
    # concentration after peak should decrease (check last value vs peak)
    assert conc[-1] < conc[max_idx]


def test_cmax_equals_max_of_list():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    assert result.cmax_mg_L == max(result.c_plasma_mg_L)


def test_tmax_matches_index_of_max():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    max_idx = result.c_plasma_mg_L.index(result.cmax_mg_L)
    assert abs(result.tmax_h - result.times_h[max_idx]) < 1e-9


def test_lung_auc_nonnegative():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    assert result.lung_auc >= 0
