"""Tests for Phase 419 — bone_marrow_pk_p419.py"""

import math

import pytest

from omega_pbpk.core.bone_marrow_pk_p419 import (
    BoneMarrowPKResult,
    compare_marrow_penetration,
    simulate_bone_marrow_pk,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    route="iv",
    cl_sys_L_per_h=2.0,
    vd_plasma_L=5.0,
    vd_marrow_L=1.5,
    q_marrow_L_per_h=0.3,
    t_end_h=24.0,
    dt_h=0.05,
)


def iv_result(**overrides):
    kw = dict(DEFAULTS)
    kw.update(overrides)
    return simulate_bone_marrow_pk(**kw)


def oral_result(**overrides):
    kw = dict(DEFAULTS, route="oral")
    kw.update(overrides)
    return simulate_bone_marrow_pk(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = iv_result()
    assert isinstance(r, BoneMarrowPKResult)


# ---------------------------------------------------------------------------
# Array properties
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    r = iv_result()
    assert r.times_h[0] == pytest.approx(0.0)


def test_array_lengths_consistent():
    r = iv_result()
    assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_marrow_mg_L)


def test_arrays_nonempty():
    r = iv_result()
    assert len(r.times_h) > 1


# ---------------------------------------------------------------------------
# IV initial conditions
# ---------------------------------------------------------------------------


def test_iv_plasma_starts_near_dose_over_vd():
    r = iv_result(dose_mg=100.0, vd_plasma_L=5.0)
    expected = 100.0 / 5.0  # 20 mg/L
    assert r.c_plasma_mg_L[0] == pytest.approx(expected, rel=0.01)


def test_iv_marrow_starts_near_zero():
    r = iv_result()
    assert r.c_marrow_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Oral initial conditions and timing
# ---------------------------------------------------------------------------


def test_oral_plasma_starts_at_zero():
    r = oral_result()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_oral_marrow_starts_at_zero():
    r = oral_result()
    assert r.c_marrow_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_oral_marrow_peaks_after_plasma():
    r = oral_result()
    tmax_plasma = r.times_h[r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))]
    assert r.tmax_marrow_h >= tmax_plasma


# ---------------------------------------------------------------------------
# Cmax and AUC positivity
# ---------------------------------------------------------------------------


def test_cmax_marrow_positive_iv():
    r = iv_result()
    assert r.cmax_marrow_mg_L > 0.0


def test_cmax_marrow_positive_oral():
    r = oral_result()
    assert r.cmax_marrow_mg_L > 0.0


def test_auc_marrow_positive_iv():
    r = iv_result()
    assert r.auc_marrow_mg_h_per_L > 0.0


def test_auc_plasma_positive_iv():
    r = iv_result()
    assert r.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Marrow penetration
# ---------------------------------------------------------------------------


def test_marrow_penetration_pct_positive():
    r = iv_result()
    assert r.marrow_penetration_pct > 0.0


def test_higher_q_marrow_higher_auc_ratio():
    r_low = iv_result(q_marrow_L_per_h=0.1)
    r_high = iv_result(q_marrow_L_per_h=1.0)
    assert r_high.marrow_to_plasma_auc_ratio > r_low.marrow_to_plasma_auc_ratio


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    r1 = iv_result(dose_mg=100.0)
    r2 = iv_result(dose_mg=200.0)
    assert r2.cmax_marrow_mg_L == pytest.approx(2.0 * r1.cmax_marrow_mg_L, rel=0.01)


def test_dose_linearity_auc():
    r1 = iv_result(dose_mg=100.0)
    r2 = iv_result(dose_mg=200.0)
    assert r2.auc_marrow_mg_h_per_L == pytest.approx(2.0 * r1.auc_marrow_mg_h_per_L, rel=0.01)


# ---------------------------------------------------------------------------
# IV vs oral
# ---------------------------------------------------------------------------


def test_iv_higher_initial_plasma_than_oral():
    r_iv = iv_result(dose_mg=100.0)
    r_oral = oral_result(dose_mg=100.0)
    assert r_iv.c_plasma_mg_L[0] > r_oral.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# t_half_marrow
# ---------------------------------------------------------------------------


def test_t_half_marrow_positive():
    r = iv_result()
    assert r.t_half_marrow_h > 0.0


def test_t_half_marrow_finite():
    r = iv_result(q_marrow_L_per_h=0.3)
    assert math.isfinite(r.t_half_marrow_h)


# ---------------------------------------------------------------------------
# compare_marrow_penetration
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    coefficients = [0.1, 0.3, 0.6, 1.0]
    results = compare_marrow_penetration(
        drug_name="TestDrug",
        dose_mg=100.0,
        penetration_coefficients=coefficients,
        cl_sys_L_per_h=2.0,
        vd_plasma_L=5.0,
    )
    assert len(results) == len(coefficients)


def test_compare_sorted_descending():
    coefficients = [0.1, 0.5, 1.0]
    results = compare_marrow_penetration(
        drug_name="TestDrug",
        dose_mg=100.0,
        penetration_coefficients=coefficients,
        cl_sys_L_per_h=2.0,
        vd_plasma_L=5.0,
    )
    ratios = [r.marrow_to_plasma_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bone_marrow_pk(drug_name="X", dose_mg=0.0, route="iv", cl_sys_L_per_h=1.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bone_marrow_pk(drug_name="X", dose_mg=-10.0, route="iv", cl_sys_L_per_h=1.0)


def test_error_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_bone_marrow_pk(drug_name="X", dose_mg=100.0, route="iv", cl_sys_L_per_h=0.0)


def test_error_bad_route():
    with pytest.raises(ValueError, match="route"):
        simulate_bone_marrow_pk(
            drug_name="X", dose_mg=100.0, route="subcutaneous", cl_sys_L_per_h=1.0
        )
