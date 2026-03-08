"""
Tests for Phase 431 — Synovial Joint PK (p431 suffix)
"""

import pytest

from omega_pbpk.core.synovial_joint_pk_p431 import (
    SynovialPKResult,
    compare_routes_synovial,
    simulate_synovial_pk,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
DRUG = "ibuprofen"
DOSE = 400.0  # mg
CL = 5.0  # L/h
VD = 10.0  # L


def _sim(route="iv", dose=DOSE, cl=CL, vd=VD, **kwargs):
    return simulate_synovial_pk(
        drug_name=DRUG, dose_mg=dose, route=route, cl_sys_L_per_h=cl, vd_plasma_L=vd, **kwargs
    )


# ------------------------------------------------------------------
# Return type
# ------------------------------------------------------------------
def test_return_type_iv():
    result = _sim(route="iv")
    assert isinstance(result, SynovialPKResult)


def test_return_type_oral():
    result = _sim(route="oral")
    assert isinstance(result, SynovialPKResult)


def test_return_type_intraarticular():
    result = _sim(route="intraarticular")
    assert isinstance(result, SynovialPKResult)


# ------------------------------------------------------------------
# Array length consistency
# ------------------------------------------------------------------
def test_array_lengths_consistent():
    result = _sim(route="iv")
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_synovial_mg_L) == n


def test_times_start_at_zero():
    result = _sim(route="iv")
    assert result.times_h[0] == 0.0


def test_times_end_near_t_end():
    result = _sim(route="oral", t_end_h=12.0)
    # Last time should be close to t_end_h
    assert abs(result.times_h[-1] - 12.0) < 0.2


# ------------------------------------------------------------------
# Intraarticular: high synovial, low plasma initially
# ------------------------------------------------------------------
def test_intraarticular_synovial_starts_high():
    result = _sim(route="intraarticular", v_syn_L=0.003)
    # C_syn(0) = dose / V_syn
    expected = DOSE / 0.003
    assert abs(result.c_synovial_mg_L[0] - expected) < 1e-6


def test_intraarticular_plasma_starts_near_zero():
    result = _sim(route="intraarticular")
    assert result.c_plasma_mg_L[0] < 1e-9


# ------------------------------------------------------------------
# IV: plasma starts at dose/vd, synovial near zero
# ------------------------------------------------------------------
def test_iv_plasma_starts_at_dose_over_vd():
    result = _sim(route="iv", vd=VD)
    expected = DOSE / VD
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-6


def test_iv_synovial_starts_near_zero():
    result = _sim(route="iv")
    assert result.c_synovial_mg_L[0] < 1e-9


# ------------------------------------------------------------------
# cmax and auc > 0
# ------------------------------------------------------------------
def test_cmax_synovial_positive_iv():
    result = _sim(route="iv")
    assert result.cmax_synovial_mg_L > 0.0


def test_auc_synovial_positive_oral():
    result = _sim(route="oral")
    assert result.auc_synovial_mg_h_per_L > 0.0


def test_auc_plasma_positive_iv():
    result = _sim(route="iv")
    assert result.auc_plasma_mg_h_per_L > 0.0


# ------------------------------------------------------------------
# t_half_synovial > 0
# ------------------------------------------------------------------
def test_t_half_synovial_positive():
    result = _sim(route="iv")
    assert result.t_half_synovial_h > 0.0


def test_t_half_synovial_formula():
    # t_half = 0.693 / (q_syn/v_syn + k_elim_syn)
    q = 0.02
    v = 0.003
    k_e = 0.1
    expected = 0.693 / (q / v + k_e)
    result = _sim(route="iv", q_syn_L_per_h=q, v_syn_L=v, k_elim_syn_per_h=k_e)
    assert abs(result.t_half_synovial_h - expected) < 1e-6


# ------------------------------------------------------------------
# Inflammation boost: higher inflamed_factor → higher synovial AUC
# ------------------------------------------------------------------
def test_higher_inflamed_factor_increases_synovial():
    r_low = _sim(route="iv", inflamed_factor=1.0)
    r_high = _sim(route="iv", inflamed_factor=3.0)
    assert r_high.auc_synovial_mg_h_per_L > r_low.auc_synovial_mg_h_per_L


def test_inflamed_ratio_boost_stored():
    factor = 2.5
    result = _sim(route="iv", inflamed_factor=factor)
    assert result.inflamed_ratio_boost == factor


# ------------------------------------------------------------------
# Dose linearity: 2x dose → 2x cmax
# ------------------------------------------------------------------
def test_dose_linearity_cmax_plasma():
    r1 = _sim(route="iv", dose=100.0)
    r2 = _sim(route="iv", dose=200.0)
    ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
    assert abs(ratio - 2.0) < 1e-6


def test_dose_linearity_cmax_synovial():
    r1 = _sim(route="oral", dose=100.0)
    r2 = _sim(route="oral", dose=200.0)
    ratio = r2.cmax_synovial_mg_L / r1.cmax_synovial_mg_L
    assert abs(ratio - 2.0) < 0.01


# ------------------------------------------------------------------
# compare_routes_synovial
# ------------------------------------------------------------------
def test_compare_routes_returns_3():
    results = compare_routes_synovial(DRUG, DOSE, CL, VD)
    assert len(results) == 3


def test_compare_routes_sorted_descending():
    results = compare_routes_synovial(DRUG, DOSE, CL, VD)
    aucs = [r.auc_synovial_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_routes_contains_all_routes():
    results = compare_routes_synovial(DRUG, DOSE, CL, VD)
    routes = {r.route for r in results}
    assert routes == {"iv", "oral", "intraarticular"}


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(route="iv", dose=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(route="iv", dose=-10.0)


def test_validation_bad_route():
    with pytest.raises(ValueError, match="route"):
        _sim(route="subcutaneous")
