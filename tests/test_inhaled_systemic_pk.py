"""Tests for Phase 321 — Inhaled Drug Systemic Absorption Model."""

import pytest

from omega_pbpk.core.inhaled_systemic_pk import (
    InhaledSystemicPKResult,
    compare_devices,
    simulate_inhaled_systemic_pk,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="BudesonideICS",
    dose_mg=1.0,
    fine_particle_fraction=0.5,
    lung_deposition_fraction=0.3,
    f_oral_swallowed=0.1,
    ka_lung_per_h=1.5,
    ka_gi_per_h=0.8,
    cl_sys_L_per_h=80.0,
    vd_sys_L=280.0,
    t_end_h=12.0,
    dt_h=0.05,
)


def base_result():
    return simulate_inhaled_systemic_pk(**DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    res = base_result()
    assert isinstance(res, InhaledSystemicPKResult)


def test_drug_name_preserved():
    res = base_result()
    assert res.drug_name == "BudesonideICS"


def test_dose_preserved():
    res = base_result()
    assert res.dose_mg == pytest.approx(1.0)


def test_fine_particle_fraction_preserved():
    res = base_result()
    assert res.fine_particle_fraction == pytest.approx(0.5)


def test_lung_deposition_fraction_preserved():
    res = base_result()
    assert res.lung_deposition_fraction == pytest.approx(0.3)


def test_f_oral_swallowed_preserved():
    res = base_result()
    assert res.f_oral_swallowed == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Time array consistency
# ---------------------------------------------------------------------------


def test_times_and_concentrations_same_length():
    res = base_result()
    assert len(res.times_h) == len(res.c_systemic_mg_L)


def test_times_start_at_zero():
    res = base_result()
    assert res.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    res = base_result()
    for i in range(1, len(res.times_h)):
        assert res.times_h[i] > res.times_h[i - 1]


def test_initial_concentration_zero():
    res = base_result()
    assert res.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Core PK metrics
# ---------------------------------------------------------------------------


def test_auc_systemic_positive():
    res = base_result()
    assert res.auc_systemic_mg_h_per_L > 0


def test_cmax_systemic_positive():
    res = base_result()
    assert res.cmax_systemic_mg_L > 0


def test_cmax_equals_max_concentration():
    res = base_result()
    assert res.cmax_systemic_mg_L == pytest.approx(max(res.c_systemic_mg_L))


def test_tmax_in_time_range():
    res = base_result()
    assert res.tmax_systemic_h >= 0
    assert res.tmax_systemic_h <= res.times_h[-1]


# ---------------------------------------------------------------------------
# Dose partition
# ---------------------------------------------------------------------------


def test_lung_dose_correct():
    res = base_result()
    assert res.lung_dose_mg == pytest.approx(1.0 * 0.3)


def test_swallowed_dose_correct():
    res = base_result()
    assert res.swallowed_dose_mg == pytest.approx(1.0 * 0.7)


def test_lung_plus_swallowed_equals_dose():
    res = base_result()
    assert res.lung_dose_mg + res.swallowed_dose_mg == pytest.approx(res.dose_mg)


# ---------------------------------------------------------------------------
# Lung contribution percentage
# ---------------------------------------------------------------------------


def test_lung_contribution_pct_in_range():
    res = base_result()
    assert 0 <= res.lung_contribution_pct <= 100


def test_lung_contribution_pct_formula():
    res = base_result()
    denom = res.lung_dose_mg + res.f_oral_swallowed * res.swallowed_dose_mg
    expected = res.lung_dose_mg / denom * 100
    assert res.lung_contribution_pct == pytest.approx(expected, rel=1e-4)


def test_higher_lung_deposition_increases_lung_contribution():
    kw_lo = {**DEFAULT_KWARGS, "lung_deposition_fraction": 0.1}
    kw_hi = {**DEFAULT_KWARGS, "lung_deposition_fraction": 0.8}
    res_lo = simulate_inhaled_systemic_pk(**kw_lo)
    res_hi = simulate_inhaled_systemic_pk(**kw_hi)
    assert res_hi.lung_contribution_pct > res_lo.lung_contribution_pct


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def test_higher_lung_deposition_raises_auc():
    kw_lo = {**DEFAULT_KWARGS, "lung_deposition_fraction": 0.1, "f_oral_swallowed": 0.5}
    kw_hi = {**DEFAULT_KWARGS, "lung_deposition_fraction": 0.8, "f_oral_swallowed": 0.5}
    res_lo = simulate_inhaled_systemic_pk(**kw_lo)
    res_hi = simulate_inhaled_systemic_pk(**kw_hi)
    assert res_hi.auc_systemic_mg_h_per_L > res_lo.auc_systemic_mg_h_per_L


def test_lower_f_oral_reduces_gi_contribution():
    """Lower f_oral_swallowed → less GI contribution → lower systemic AUC
    (when lung_deposition_fraction < 1 so swallowed fraction > 0)."""
    kw_hi = {**DEFAULT_KWARGS, "f_oral_swallowed": 0.9, "lung_deposition_fraction": 0.2}
    kw_lo = {**DEFAULT_KWARGS, "f_oral_swallowed": 0.01, "lung_deposition_fraction": 0.2}
    res_hi = simulate_inhaled_systemic_pk(**kw_hi)
    res_lo = simulate_inhaled_systemic_pk(**kw_lo)
    assert res_hi.auc_systemic_mg_h_per_L > res_lo.auc_systemic_mg_h_per_L


def test_higher_dose_increases_auc():
    kw_lo = {**DEFAULT_KWARGS, "dose_mg": 0.5}
    kw_hi = {**DEFAULT_KWARGS, "dose_mg": 2.0}
    res_lo = simulate_inhaled_systemic_pk(**kw_lo)
    res_hi = simulate_inhaled_systemic_pk(**kw_hi)
    assert res_hi.auc_systemic_mg_h_per_L > res_lo.auc_systemic_mg_h_per_L


# ---------------------------------------------------------------------------
# compare_devices
# ---------------------------------------------------------------------------


def test_compare_devices_returns_list():
    devices = [
        {"device_name": "pMDI", "fine_particle_fraction": 0.3, "lung_deposition_fraction": 0.15},
        {"device_name": "DPI", "fine_particle_fraction": 0.5, "lung_deposition_fraction": 0.35},
        {
            "device_name": "Nebulizer",
            "fine_particle_fraction": 0.7,
            "lung_deposition_fraction": 0.6,
        },
    ]
    result = compare_devices("Budesonide", 1.0, 80.0, 280.0, devices)
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_devices_sorted_descending():
    devices = [
        {"device_name": "pMDI", "fine_particle_fraction": 0.3, "lung_deposition_fraction": 0.15},
        {"device_name": "DPI", "fine_particle_fraction": 0.5, "lung_deposition_fraction": 0.35},
        {
            "device_name": "Nebulizer",
            "fine_particle_fraction": 0.7,
            "lung_deposition_fraction": 0.6,
        },
    ]
    result = compare_devices("Budesonide", 1.0, 80.0, 280.0, devices)
    aucs = [r["auc_systemic_mg_h_per_L"] for r in result]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_devices_contains_result_object():
    devices = [
        {"device_name": "X", "fine_particle_fraction": 0.4, "lung_deposition_fraction": 0.25},
    ]
    result = compare_devices("Drug", 1.0, 5.0, 50.0, devices)
    assert isinstance(result[0]["result"], InhaledSystemicPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "dose_mg": 0})


def test_validation_lung_deposition_out_of_range():
    with pytest.raises(ValueError, match="lung_deposition_fraction"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "lung_deposition_fraction": 1.5})


def test_validation_fine_particle_fraction_negative():
    with pytest.raises(ValueError, match="fine_particle_fraction"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "fine_particle_fraction": -0.1})


def test_validation_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral_swallowed"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "f_oral_swallowed": 0.0})


def test_validation_ka_lung_negative():
    with pytest.raises(ValueError, match="ka_lung_per_h"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "ka_lung_per_h": -1.0})


def test_validation_ka_gi_zero():
    with pytest.raises(ValueError, match="ka_gi_per_h"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "ka_gi_per_h": 0.0})


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "cl_sys_L_per_h": 0.0})


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "vd_sys_L": -10.0})


def test_validation_t_end_zero():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_inhaled_systemic_pk(**{**DEFAULT_KWARGS, "t_end_h": 0.0})
