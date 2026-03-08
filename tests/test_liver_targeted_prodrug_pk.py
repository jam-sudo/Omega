"""Tests for Phase 322 — Prodrug Liver-Targeted Delivery."""

import math

import pytest

from omega_pbpk.core.liver_targeted_prodrug_pk import (
    LiverTargetedProdrugResult,
    compare_targeting_efficiencies,
    simulate_liver_targeted_prodrug,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="Sofosbuvir",
    dose_mg=400.0,
    route="iv_bolus",
    cl_prodrug_sys_L_per_h=50.0,
    vd_prodrug_L=100.0,
    f_hepatic_activation=0.8,
    k_hepatic_activation_per_h=0.5,
    cl_active_sys_L_per_h=10.0,
    vd_active_L=30.0,
    t_end_h=24.0,
    dt_h=0.02,
)

ORAL_KWARGS = {**DEFAULT_KWARGS, "route": "oral"}


def base_result():
    return simulate_liver_targeted_prodrug(**DEFAULT_KWARGS)


def oral_result():
    return simulate_liver_targeted_prodrug(**ORAL_KWARGS)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    assert isinstance(base_result(), LiverTargetedProdrugResult)


def test_return_type_oral():
    assert isinstance(oral_result(), LiverTargetedProdrugResult)


def test_drug_name_preserved():
    assert base_result().drug_name == "Sofosbuvir"


def test_route_preserved():
    assert base_result().route == "iv_bolus"
    assert oral_result().route == "oral"


def test_f_hepatic_activation_preserved():
    assert base_result().f_hepatic_activation == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Time array consistency
# ---------------------------------------------------------------------------


def test_times_and_concentrations_same_length():
    res = base_result()
    n = len(res.times_h)
    assert len(res.c_prodrug_mg_L) == n
    assert len(res.c_hepatic_active_mg_L) == n
    assert len(res.c_systemic_active_mg_L) == n


def test_times_start_at_zero():
    res = base_result()
    assert res.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    res = base_result()
    for i in range(1, len(res.times_h)):
        assert res.times_h[i] > res.times_h[i - 1]


# ---------------------------------------------------------------------------
# Core PK metrics
# ---------------------------------------------------------------------------


def test_auc_hepatic_positive_when_f_hepatic_nonzero():
    res = base_result()
    assert res.auc_hepatic_mg_h_per_L > 0


def test_auc_hepatic_zero_when_f_hepatic_zero():
    kw = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.0}
    res = simulate_liver_targeted_prodrug(**kw)
    assert res.auc_hepatic_mg_h_per_L == pytest.approx(0.0, abs=1e-6)


def test_auc_systemic_active_positive_when_f_hepatic_less_than_1():
    res = base_result()
    assert res.auc_systemic_active_mg_h_per_L > 0


def test_auc_systemic_active_zero_when_f_hepatic_1():
    kw = {**DEFAULT_KWARGS, "f_hepatic_activation": 1.0}
    res = simulate_liver_targeted_prodrug(**kw)
    assert res.auc_systemic_active_mg_h_per_L == pytest.approx(0.0, abs=1e-6)


def test_cmax_hepatic_positive():
    res = base_result()
    assert res.cmax_hepatic_mg_L > 0


def test_cmax_systemic_active_positive():
    res = base_result()
    assert res.cmax_systemic_active_mg_L > 0


# ---------------------------------------------------------------------------
# Targeting metrics
# ---------------------------------------------------------------------------


def test_targeting_efficiency_pct_formula():
    res = base_result()
    assert res.targeting_efficiency_pct == pytest.approx(0.8 * 100, rel=1e-4)


def test_targeting_efficiency_pct_for_oral():
    res = oral_result()
    assert res.targeting_efficiency_pct == pytest.approx(0.8 * 100, rel=1e-4)


def test_hepatic_to_systemic_ratio_positive():
    res = base_result()
    assert res.hepatic_to_systemic_ratio > 0


def test_hepatic_to_systemic_ratio_is_inf_when_f_hepatic_1():
    kw = {**DEFAULT_KWARGS, "f_hepatic_activation": 1.0}
    res = simulate_liver_targeted_prodrug(**kw)
    assert math.isinf(res.hepatic_to_systemic_ratio)


def test_higher_f_hepatic_increases_hepatic_auc():
    kw_lo = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.2}
    kw_hi = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.9}
    res_lo = simulate_liver_targeted_prodrug(**kw_lo)
    res_hi = simulate_liver_targeted_prodrug(**kw_hi)
    assert res_hi.auc_hepatic_mg_h_per_L > res_lo.auc_hepatic_mg_h_per_L


def test_higher_f_hepatic_reduces_systemic_active_auc():
    kw_lo = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.1}
    kw_hi = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.9}
    res_lo = simulate_liver_targeted_prodrug(**kw_lo)
    res_hi = simulate_liver_targeted_prodrug(**kw_hi)
    assert res_hi.auc_systemic_active_mg_h_per_L < res_lo.auc_systemic_active_mg_h_per_L


def test_higher_f_hepatic_increases_ratio():
    kw_lo = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.2}
    kw_hi = {**DEFAULT_KWARGS, "f_hepatic_activation": 0.8}
    res_lo = simulate_liver_targeted_prodrug(**kw_lo)
    res_hi = simulate_liver_targeted_prodrug(**kw_hi)
    assert res_hi.hepatic_to_systemic_ratio > res_lo.hepatic_to_systemic_ratio


# ---------------------------------------------------------------------------
# compare_targeting_efficiencies
# ---------------------------------------------------------------------------


def test_compare_targeting_returns_list():
    result = compare_targeting_efficiencies(
        drug_name="TestDrug",
        dose_mg=100.0,
        f_hepatic_list=[0.2, 0.5, 0.9],
        k_activation_per_h=0.5,
        cl_prodrug_sys_L_per_h=50.0,
        vd_prodrug_L=100.0,
        cl_active_sys_L_per_h=10.0,
        vd_active_L=30.0,
    )
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_targeting_sorted_descending():
    result = compare_targeting_efficiencies(
        drug_name="TestDrug",
        dose_mg=100.0,
        f_hepatic_list=[0.1, 0.5, 0.9],
        k_activation_per_h=0.5,
        cl_prodrug_sys_L_per_h=50.0,
        vd_prodrug_L=100.0,
        cl_active_sys_L_per_h=10.0,
        vd_active_L=30.0,
    )
    ratios = [
        r["hepatic_to_systemic_ratio"]
        for r in result
        if not math.isinf(r["hepatic_to_systemic_ratio"])
    ]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_targeting_result_objects():
    result = compare_targeting_efficiencies(
        drug_name="TestDrug",
        dose_mg=100.0,
        f_hepatic_list=[0.5],
        k_activation_per_h=0.5,
        cl_prodrug_sys_L_per_h=50.0,
        vd_prodrug_L=100.0,
        cl_active_sys_L_per_h=10.0,
        vd_active_L=30.0,
    )
    assert isinstance(result[0]["result"], LiverTargetedProdrugResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "dose_mg": 0})


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "route": "subcutaneous"})


def test_validation_cl_prodrug_zero():
    with pytest.raises(ValueError, match="cl_prodrug_sys_L_per_h"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "cl_prodrug_sys_L_per_h": 0.0})


def test_validation_vd_prodrug_negative():
    with pytest.raises(ValueError, match="vd_prodrug_L"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "vd_prodrug_L": -5.0})


def test_validation_f_hepatic_out_of_range():
    with pytest.raises(ValueError, match="f_hepatic_activation"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "f_hepatic_activation": 1.5})


def test_validation_k_activation_zero():
    with pytest.raises(ValueError, match="k_hepatic_activation_per_h"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "k_hepatic_activation_per_h": 0.0})


def test_validation_cl_active_zero():
    with pytest.raises(ValueError, match="cl_active_sys_L_per_h"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "cl_active_sys_L_per_h": 0.0})


def test_validation_vd_active_zero():
    with pytest.raises(ValueError, match="vd_active_L"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "vd_active_L": 0.0})


def test_validation_t_end_zero():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_liver_targeted_prodrug(**{**DEFAULT_KWARGS, "t_end_h": 0.0})
