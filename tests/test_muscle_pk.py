"""Tests for Phase 293 — Muscle Tissue PK Model."""

import pytest

from omega_pbpk.core.muscle_pk import (
    MusclePKResult,
    compare_muscle_distribution,
    simulate_muscle_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    route="iv_bolus",
    cl_sys_L_per_h=5.0,
    vd_plasma_L=10.0,
    kp_muscle=2.0,
    muscle_volume_L=28.0,
    muscle_blood_flow_L_per_h=75.0,
    t_end_h=24.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# Return type & structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert isinstance(result, MusclePKResult)


def test_drug_name_preserved():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.drug_name == "TestDrug"


def test_dose_preserved():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.dose_mg == 100.0


def test_route_preserved():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.route == "iv_bolus"


def test_kp_muscle_preserved():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.kp_muscle == 2.0


# ---------------------------------------------------------------------------
# Time array consistency
# ---------------------------------------------------------------------------


def test_time_array_consistency():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_muscle_mg_L)


def test_time_starts_at_zero():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.times_h[0] == 0.0


def test_time_ends_at_t_end():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)


# ---------------------------------------------------------------------------
# Plasma PK
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.cmax_plasma_mg_L > 0


def test_auc_plasma_positive():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.auc_plasma_mg_h_per_L > 0


def test_cmax_plasma_iv_at_t0():
    """For IV bolus, plasma Cmax should occur at t=0."""
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.c_plasma_mg_L[0] == pytest.approx(result.cmax_plasma_mg_L, rel=0.01)


# ---------------------------------------------------------------------------
# Muscle PK
# ---------------------------------------------------------------------------


def test_cmax_muscle_positive():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.cmax_muscle_mg_L > 0


def test_auc_muscle_positive():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.auc_muscle_mg_h_per_L > 0


def test_muscle_zero_at_t0_iv():
    """Muscle concentration starts at 0 for IV bolus."""
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.c_muscle_mg_L[0] == pytest.approx(0.0, abs=1e-6)


def test_tmax_muscle_after_t0_iv():
    """Muscle Tmax should be after t=0 for IV bolus."""
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.tmax_muscle_h > 0.0


def test_muscle_to_plasma_ratio_positive():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert result.muscle_to_plasma_ratio > 0


# ---------------------------------------------------------------------------
# kp_muscle effect
# ---------------------------------------------------------------------------


def test_higher_kp_muscle_higher_ratio():
    """Higher kp_muscle should lead to higher muscle_to_plasma_ratio."""
    kwargs_low = {**BASE_KWARGS, "kp_muscle": 0.5}
    kwargs_high = {**BASE_KWARGS, "kp_muscle": 5.0}
    result_low = simulate_muscle_pk(**kwargs_low)
    result_high = simulate_muscle_pk(**kwargs_high)
    assert result_high.muscle_to_plasma_ratio > result_low.muscle_to_plasma_ratio


def test_higher_kp_muscle_higher_auc_muscle():
    """Higher kp_muscle should lead to higher AUC in muscle."""
    kwargs_low = {**BASE_KWARGS, "kp_muscle": 0.5}
    kwargs_high = {**BASE_KWARGS, "kp_muscle": 5.0}
    result_low = simulate_muscle_pk(**kwargs_low)
    result_high = simulate_muscle_pk(**kwargs_high)
    assert result_high.auc_muscle_mg_h_per_L > result_low.auc_muscle_mg_h_per_L


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_route_works():
    kwargs = {**BASE_KWARGS, "route": "oral"}
    result = simulate_muscle_pk(**kwargs)
    assert isinstance(result, MusclePKResult)
    assert result.route == "oral"


def test_oral_plasma_zero_at_t0():
    """Oral: plasma concentration at t=0 is 0."""
    kwargs = {**BASE_KWARGS, "route": "oral"}
    result = simulate_muscle_pk(**kwargs)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-6)


def test_oral_tmax_muscle_later_than_iv():
    """Oral muscle Tmax should be later than IV bolus muscle Tmax."""
    result_iv = simulate_muscle_pk(**BASE_KWARGS)
    result_oral = simulate_muscle_pk(**{**BASE_KWARGS, "route": "oral"})
    assert result_oral.tmax_muscle_h > result_iv.tmax_muscle_h


def test_oral_cmax_plasma_positive():
    kwargs = {**BASE_KWARGS, "route": "oral"}
    result = simulate_muscle_pk(**kwargs)
    assert result.cmax_plasma_mg_L > 0


def test_oral_cmax_muscle_positive():
    kwargs = {**BASE_KWARGS, "route": "oral"}
    result = simulate_muscle_pk(**kwargs)
    assert result.cmax_muscle_mg_L > 0


# ---------------------------------------------------------------------------
# compare_muscle_distribution
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_muscle_distribution(
        drug_name="TestDrug",
        dose_mg=100.0,
        kp_muscle_list=[0.5, 1.0, 2.0, 5.0],
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
        dt_h=0.1,
    )
    assert isinstance(results, list)
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_muscle_distribution(
        drug_name="TestDrug",
        dose_mg=100.0,
        kp_muscle_list=[0.5, 1.0, 2.0, 5.0],
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
        dt_h=0.1,
    )
    aucs = [r.auc_muscle_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_muscle_pk_results():
    results = compare_muscle_distribution(
        drug_name="TestDrug",
        dose_mg=100.0,
        kp_muscle_list=[1.0, 3.0],
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
        dt_h=0.1,
    )
    for r in results:
        assert isinstance(r, MusclePKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk(**{**BASE_KWARGS, "dose_mg": 0.0})


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_muscle_pk(**{**BASE_KWARGS, "cl_sys_L_per_h": -1.0})


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_muscle_pk(**{**BASE_KWARGS, "vd_plasma_L": 0.0})


def test_invalid_kp_raises():
    with pytest.raises(ValueError, match="kp_muscle"):
        simulate_muscle_pk(**{**BASE_KWARGS, "kp_muscle": -0.5})


def test_invalid_muscle_volume_raises():
    with pytest.raises(ValueError, match="muscle_volume_L"):
        simulate_muscle_pk(**{**BASE_KWARGS, "muscle_volume_L": 0.0})


def test_invalid_blood_flow_raises():
    with pytest.raises(ValueError, match="muscle_blood_flow_L_per_h"):
        simulate_muscle_pk(**{**BASE_KWARGS, "muscle_blood_flow_L_per_h": -5.0})


def test_invalid_t_end_raises():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_muscle_pk(**{**BASE_KWARGS, "t_end_h": 0.0})


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_muscle_pk(**{**BASE_KWARGS, "route": "intramuscular"})


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = simulate_muscle_pk(**BASE_KWARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
