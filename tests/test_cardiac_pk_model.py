"""
Tests for Phase 423: Cardiac PK Model
~24 tests covering result type, IV/oral behavior, metrics, dose linearity,
compare function, and validation errors.
"""

import pytest

from omega_pbpk.clinical.cardiac_pk_model import (
    CardiacPKResult,
    compare_cardiac_drugs,
    simulate_cardiac_pk,
)

# ─── Basic return type ──────────────────────────────────────────────────────


def test_return_type_iv():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert isinstance(result, CardiacPKResult)


def test_return_type_oral():
    result = simulate_cardiac_pk("sotalol", 80.0, "oral", cl_sys_L_per_h=4.0)
    assert isinstance(result, CardiacPKResult)


# ─── Time array ─────────────────────────────────────────────────────────────


def test_times_start_at_zero():
    result = simulate_cardiac_pk("digoxin", 0.25, "iv", cl_sys_L_per_h=2.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_arrays_consistent_length():
    result = simulate_cardiac_pk("lidocaine", 100.0, "iv", cl_sys_L_per_h=40.0)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_myocardium_mg_L) == n
    assert n > 1


# ─── IV initial condition ────────────────────────────────────────────────────


def test_iv_plasma_starts_at_dose_over_vd():
    dose = 200.0
    vd = 5.0
    result = simulate_cardiac_pk("amiodarone", dose, "iv", cl_sys_L_per_h=5.0, vd_plasma_L=vd)
    assert result.c_plasma_mg_L[0] == pytest.approx(dose / vd, rel=1e-6)


def test_iv_myocardium_starts_at_zero():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.c_myocardium_mg_L[0] == pytest.approx(0.0, abs=1e-12)


# ─── Oral initial conditions ─────────────────────────────────────────────────


def test_oral_plasma_starts_at_zero():
    result = simulate_cardiac_pk("sotalol", 80.0, "oral", cl_sys_L_per_h=4.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_oral_myocardium_starts_at_zero():
    result = simulate_cardiac_pk("sotalol", 80.0, "oral", cl_sys_L_per_h=4.0)
    assert result.c_myocardium_mg_L[0] == pytest.approx(0.0, abs=1e-12)


# ─── cmax / auc positive ────────────────────────────────────────────────────


def test_cmax_myocardium_positive():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.cmax_myocardium_mg_L > 0.0


def test_auc_myocardium_positive():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.auc_myocardium_mg_h_per_L > 0.0


def test_auc_plasma_positive():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.auc_plasma_mg_h_per_L > 0.0


# ─── myocardium_to_plasma_ratio / cardiac_index ──────────────────────────────


def test_myocardium_to_plasma_ratio_positive():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.myocardium_to_plasma_ratio > 0.0


def test_cardiac_index_equals_ratio_times_100():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.cardiac_index == pytest.approx(result.myocardium_to_plasma_ratio * 100, rel=1e-6)


# ─── t_half_myocardium ───────────────────────────────────────────────────────


def test_t_half_myocardium_positive():
    result = simulate_cardiac_pk("amiodarone", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result.t_half_myocardium_h > 0.0


def test_t_half_myocardium_formula():
    """t_half = 0.693 / k21 = 0.693 * vd_myo / q_cardiac"""
    vd_myo = 0.35
    q_cardiac = 0.4
    expected_t_half = 0.693 / (q_cardiac / vd_myo)
    result = simulate_cardiac_pk(
        "test", 100.0, "iv", cl_sys_L_per_h=5.0, vd_myocardium_L=vd_myo, q_cardiac_L_per_h=q_cardiac
    )
    assert result.t_half_myocardium_h == pytest.approx(expected_t_half, rel=1e-3)


# ─── Higher q_cardiac → better myocardium penetration ───────────────────────


def test_higher_q_cardiac_higher_myocardium_penetration():
    result_low = simulate_cardiac_pk("drug", 100.0, "iv", cl_sys_L_per_h=5.0, q_cardiac_L_per_h=0.1)
    result_high = simulate_cardiac_pk(
        "drug", 100.0, "iv", cl_sys_L_per_h=5.0, q_cardiac_L_per_h=1.0
    )
    assert result_high.cmax_myocardium_mg_L > result_low.cmax_myocardium_mg_L


# ─── Dose linearity ──────────────────────────────────────────────────────────


def test_dose_linearity_iv_cmax():
    result1 = simulate_cardiac_pk("drug", 100.0, "iv", cl_sys_L_per_h=5.0)
    result2 = simulate_cardiac_pk("drug", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result2.cmax_plasma_mg_L == pytest.approx(2 * result1.cmax_plasma_mg_L, rel=1e-6)


def test_dose_linearity_iv_auc():
    result1 = simulate_cardiac_pk("drug", 100.0, "iv", cl_sys_L_per_h=5.0)
    result2 = simulate_cardiac_pk("drug", 200.0, "iv", cl_sys_L_per_h=5.0)
    assert result2.auc_myocardium_mg_h_per_L == pytest.approx(
        2 * result1.auc_myocardium_mg_h_per_L, rel=1e-4
    )


# ─── compare_cardiac_drugs ───────────────────────────────────────────────────


def test_compare_cardiac_drugs_length():
    drug_list = [
        {
            "drug_name": "drug_A",
            "dose_mg": 100.0,
            "cl_sys_L_per_h": 5.0,
            "vd_plasma_L": 5.0,
            "vd_myocardium_L": 0.35,
            "q_cardiac_L_per_h": 0.4,
        },
        {
            "drug_name": "drug_B",
            "dose_mg": 100.0,
            "cl_sys_L_per_h": 3.0,
            "vd_plasma_L": 5.0,
            "vd_myocardium_L": 0.5,
            "q_cardiac_L_per_h": 0.8,
        },
        {
            "drug_name": "drug_C",
            "dose_mg": 100.0,
            "cl_sys_L_per_h": 8.0,
            "vd_plasma_L": 5.0,
            "vd_myocardium_L": 0.2,
            "q_cardiac_L_per_h": 0.2,
        },
    ]
    results = compare_cardiac_drugs(drug_list, route="iv")
    assert len(results) == 3


def test_compare_cardiac_drugs_sorted_descending():
    drug_list = [
        {
            "drug_name": "drug_A",
            "dose_mg": 100.0,
            "cl_sys_L_per_h": 5.0,
            "vd_plasma_L": 5.0,
            "vd_myocardium_L": 0.35,
            "q_cardiac_L_per_h": 0.1,
        },
        {
            "drug_name": "drug_B",
            "dose_mg": 100.0,
            "cl_sys_L_per_h": 5.0,
            "vd_plasma_L": 5.0,
            "vd_myocardium_L": 0.35,
            "q_cardiac_L_per_h": 2.0,
        },
    ]
    results = compare_cardiac_drugs(drug_list, route="iv")
    # drug_B has higher q_cardiac -> higher ratio -> should be first
    ratios = [r.myocardium_to_plasma_ratio for r in results]
    assert ratios[0] >= ratios[1]


# ─── Validation errors ───────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_pk("drug", 0.0, "iv", cl_sys_L_per_h=5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_pk("drug", -10.0, "iv", cl_sys_L_per_h=5.0)


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_cardiac_pk("drug", 100.0, "iv", cl_sys_L_per_h=0.0)


def test_validation_bad_route():
    with pytest.raises(ValueError, match="route"):
        simulate_cardiac_pk("drug", 100.0, "subcutaneous", cl_sys_L_per_h=5.0)
