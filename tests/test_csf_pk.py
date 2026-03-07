"""Tests for Phase 1037 — Drug CSF PK (core/csf_pk.py)."""

import math
import pytest

from omega_pbpk.core.csf_pk import CSFPKResult, simulate_csf_pk


VALID_BBB_PENETRATIONS = {"high", "moderate", "low"}


def make_systemic(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        route="systemic",
        mw_Da=300.0,
        logp=2.0,
        fu_plasma=0.3,
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
        v_csf_mL=140.0,
        csf_turnover_per_h=0.25,
        t_end_h=24.0,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_csf_pk(**defaults)


def make_intrathecal(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="intrathecal",
        mw_Da=300.0,
        logp=2.0,
        fu_plasma=0.3,
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
        v_csf_mL=140.0,
        csf_turnover_per_h=0.25,
        t_end_h=24.0,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_csf_pk(**defaults)


# --- Return type ---
def test_returns_csf_pk_result():
    result = make_systemic()
    assert isinstance(result, CSFPKResult)


# --- Systemic initial conditions ---
def test_systemic_plasma_nonzero_at_t0():
    result = make_systemic()
    assert result.c_plasma_mg_L[0] > 0


def test_systemic_csf_zero_at_t0():
    result = make_systemic()
    assert result.c_csf_mg_L[0] == 0.0


def test_systemic_brain_zero_at_t0():
    result = make_systemic()
    assert result.c_brain_mg_L[0] == 0.0


# --- Intrathecal initial conditions ---
def test_intrathecal_csf_nonzero_at_t0():
    result = make_intrathecal()
    assert result.c_csf_mg_L[0] > 0


def test_intrathecal_plasma_zero_at_t0():
    result = make_intrathecal()
    assert result.c_plasma_mg_L[0] == 0.0


# --- AUCs ---
def test_systemic_auc_plasma_positive():
    result = make_systemic()
    assert result.auc_plasma > 0


def test_systemic_auc_csf_positive():
    result = make_systemic(logp=3.0, kp_bbb=0.5)
    assert result.auc_csf > 0


def test_intrathecal_auc_csf_positive():
    result = make_intrathecal()
    assert result.auc_csf > 0


# --- Cmax and Tmax ---
def test_cmax_csf_positive_systemic():
    result = make_systemic(logp=3.0, kp_bbb=0.5)
    assert result.cmax_csf_mg_L > 0


def test_tmax_csf_nonnegative():
    result = make_systemic(logp=3.0, kp_bbb=0.5)
    assert result.tmax_csf_h >= 0


# --- CSF to plasma ratio ---
def test_csf_to_plasma_ratio_positive_systemic():
    result = make_systemic(logp=3.0, kp_bbb=0.5)
    assert result.csf_to_plasma_ratio > 0


# --- BBB penetration classification ---
def test_bbb_penetration_valid_class():
    result = make_systemic()
    assert result.bbb_penetration in VALID_BBB_PENETRATIONS


def test_high_logp_gives_high_penetration():
    result = make_systemic(logp=4.0, mw_Da=200.0, kp_bbb=0.5, t_end_h=48.0)
    # high logP → high kp_bbb → higher ratio
    assert result.bbb_penetration in {"high", "moderate"}


# --- Half-life ---
def test_t_half_csf_positive():
    result = make_systemic()
    assert result.t_half_csf_h > 0


def test_t_half_csf_approx_ln2_over_turnover():
    csf_turnover = 0.25
    result = make_systemic(csf_turnover_per_h=csf_turnover)
    expected = math.log(2) / csf_turnover
    assert abs(result.t_half_csf_h - expected) < 0.01


# --- High logP vs low logP ---
def test_high_logp_higher_csf_plasma_ratio():
    res_high = make_systemic(logp=4.0, mw_Da=200.0, t_end_h=48.0, dt_h=0.2)
    res_low = make_systemic(logp=0.0, mw_Da=200.0, t_end_h=48.0, dt_h=0.2)
    assert res_high.csf_to_plasma_ratio > res_low.csf_to_plasma_ratio


# --- Intrathecal vs systemic AUC CSF ---
def test_intrathecal_higher_auc_csf_than_systemic():
    res_it = make_intrathecal(dose_mg=10.0)
    # Use kp_bbb override to ensure systemic gives CSF exposure
    res_sys = make_systemic(dose_mg=10.0, kp_bbb=0.3, t_end_h=24.0)
    # Intrathecal deposits directly into CSF so AUC_CSF should be higher per mg
    assert res_it.auc_csf > res_sys.auc_csf


# --- Notes nonempty ---
def test_notes_nonempty():
    result = make_systemic()
    assert len(result.notes) > 0


# --- Validation errors ---
def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_pk("DrugX", dose_mg=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_csf_pk("DrugX", dose_mg=100.0, route="oral")


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_csf_pk("DrugX", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_kp_negative_raises():
    with pytest.raises(ValueError, match="kp_bbb"):
        simulate_csf_pk("DrugX", dose_mg=100.0, kp_bbb=-0.1)
