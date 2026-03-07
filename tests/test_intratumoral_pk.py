"""Tests for Phase 961 — intratumoral_pk module."""

import pytest

from omega_pbpk.core.intratumoral_pk import (
    IntratumoralPKResult,
    compare_routes,
    simulate_intratumoral_pk,
)


def make_result(route="systemic_iv", logp=2.0):
    return simulate_intratumoral_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        logp=logp,
        route=route,
        cl_sys_L_per_h=10.0,
        v_plasma_L=4.0,
        q_tumor_L_per_h=0.5,
        t_end_h=48.0,
        dt_h=0.1,
    )


def test_return_type():
    result = make_result()
    assert isinstance(result, IntratumoralPKResult)


def test_auc_plasma_positive():
    result = make_result()
    assert result.auc_plasma > 0


def test_auc_tv_positive():
    result = make_result()
    assert result.auc_tv > 0


def test_auc_tc_positive():
    result = make_result()
    assert result.auc_tc > 0


def test_tumor_to_plasma_ratio_positive():
    result = make_result()
    assert result.tumor_to_plasma_ratio > 0


def test_core_periphery_ratio_positive():
    result = make_result()
    assert result.core_periphery_ratio > 0


def test_cmax_plasma_positive():
    result = make_result()
    assert result.cmax_plasma > 0


def test_cmax_tv_positive():
    result = make_result()
    assert result.cmax_tv > 0


def test_cmax_tc_positive():
    result = make_result()
    assert result.cmax_tc > 0


def test_intratumoral_higher_auc_tc():
    """Direct intratumoral injection should yield higher core AUC."""
    r_it = make_result(route="intratumoral")
    r_iv = make_result(route="systemic_iv")
    assert r_it.auc_tc > r_iv.auc_tc


def test_intratumoral_lower_auc_plasma():
    """Intratumoral injection should yield lower systemic plasma AUC."""
    r_it = make_result(route="intratumoral")
    r_iv = make_result(route="systemic_iv")
    assert r_it.auc_plasma < r_iv.auc_plasma


def test_compare_routes_returns_two():
    results = compare_routes("TestDrug", dose_mg=100.0)
    assert len(results) == 2


def test_compare_routes_sorted_by_auc_tc_descending():
    results = compare_routes("TestDrug", dose_mg=100.0)
    assert results[0].auc_tc >= results[1].auc_tc


def test_compare_routes_intratumoral_first():
    results = compare_routes("TestDrug", dose_mg=100.0)
    assert results[0].route == "intratumoral"


def test_notes_non_empty():
    result = make_result()
    assert result.notes and len(result.notes) > 0


def test_times_length_matches_plasma():
    result = make_result()
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_length_matches_tv():
    result = make_result()
    assert len(result.times_h) == len(result.c_tv_mg_L)


def test_times_length_matches_tc():
    result = make_result()
    assert len(result.times_h) == len(result.c_tc_mg_L)


def test_high_logp_higher_tumor_to_plasma_ratio():
    """Lipophilic drug should partition more into tumor."""
    r_high = make_result(logp=4.0)
    r_low = make_result(logp=0.5)
    assert r_high.tumor_to_plasma_ratio > r_low.tumor_to_plasma_ratio


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intratumoral_pk("Drug", dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intratumoral_pk("Drug", dose_mg=-10.0)


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_validation_cl_sys_negative():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, cl_sys_L_per_h=-1.0)


def test_validation_v_plasma_zero():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, v_plasma_L=0.0)


def test_validation_v_plasma_negative():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, v_plasma_L=-4.0)


def test_validation_q_tumor_zero():
    with pytest.raises(ValueError, match="q_tumor_L_per_h"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, q_tumor_L_per_h=0.0)


def test_validation_q_tumor_negative():
    with pytest.raises(ValueError, match="q_tumor_L_per_h"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, q_tumor_L_per_h=-0.5)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_intratumoral_pk("Drug", dose_mg=100.0, route="oral")
