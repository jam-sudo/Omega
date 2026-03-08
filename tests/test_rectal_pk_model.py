"""Tests for Phase 453 — rectal_pk_model.py"""

import pytest

from omega_pbpk.core.rectal_pk_model import (
    RectalPKResult,
    compare_formulations_rectal,
    simulate_rectal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_suppository(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation="suppository",
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_rectal_pk(**defaults)


def make_enema(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation="enema",
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_rectal_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type_suppository():
    result = make_suppository()
    assert isinstance(result, RectalPKResult)


def test_return_type_enema():
    result = make_enema()
    assert isinstance(result, RectalPKResult)


# ---------------------------------------------------------------------------
# Array consistency
# ---------------------------------------------------------------------------


def test_arrays_consistent_length():
    result = make_suppository()
    assert len(result.times_h) == len(result.c_systemic_mg_L)


def test_times_start_at_zero():
    result = make_enema()
    assert result.times_h[0] == pytest.approx(0.0)


def test_systemic_starts_at_zero():
    result = make_suppository()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = make_suppository()
    assert result.cmax_systemic_mg_L > 0.0


def test_auc_positive():
    result = make_suppository()
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_tmax_positive():
    result = make_suppository()
    assert result.tmax_systemic_h > 0.0


def test_t_half_positive():
    result = make_suppository()
    assert result.t_half_systemic_h > 0.0


def test_t_half_formula():
    """t_half = 0.693 * Vd / CL"""
    cl = 10.0
    vd = 50.0
    result = make_suppository(cl_sys_L_per_h=cl, vd_sys_L=vd)
    expected = 0.693 * vd / cl
    assert result.t_half_systemic_h == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Bioavailability parameters
# ---------------------------------------------------------------------------


def test_f_rectal_between_0_and_1():
    result = make_suppository()
    assert 0.0 < result.f_rectal <= 1.0


def test_first_pass_bypass_fraction_is_0_5():
    result = make_suppository()
    assert result.first_pass_bypass_fraction == pytest.approx(0.5)


def test_first_pass_bypass_fraction_enema():
    result = make_enema()
    assert result.first_pass_bypass_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Formulation differences
# ---------------------------------------------------------------------------


def test_enema_shorter_lag_than_suppository():
    sup = make_suppository()
    enema = make_enema()
    assert enema.lag_time_h < sup.lag_time_h


def test_suppository_lag_time_is_0_5():
    result = make_suppository()
    assert result.lag_time_h == pytest.approx(0.5)


def test_enema_lag_time_is_0_1():
    result = make_enema()
    assert result.lag_time_h == pytest.approx(0.1)


def test_enema_earlier_tmax_than_suppository():
    """Enema has shorter lag → reaches Cmax earlier (smaller tmax).

    Since M_rectal is preserved during the lag period, Cmax values converge
    to the same value; the observable difference is tmax.
    """
    sup = make_suppository(ka_per_h=2.0)
    enema = make_enema(ka_per_h=2.0)
    assert enema.tmax_systemic_h < sup.tmax_systemic_h


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = make_suppository(drug_name="Paracetamol")
    assert result.drug_name == "Paracetamol"


# ---------------------------------------------------------------------------
# compare_formulations_rectal
# ---------------------------------------------------------------------------


def test_compare_returns_two_items():
    results = compare_formulations_rectal(
        drug_name="Drug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    assert len(results) == 2


def test_compare_sorted_by_auc_descending():
    results = compare_formulations_rectal(
        drug_name="Drug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    assert results[0].auc_systemic_mg_h_per_L >= results[1].auc_systemic_mg_h_per_L


def test_compare_contains_both_formulations():
    results = compare_formulations_rectal(
        drug_name="Drug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    formulations = {r.formulation for r in results}
    assert formulations == {"suppository", "enema"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_bad_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        simulate_rectal_pk(
            drug_name="X",
            dose_mg=100.0,
            formulation="tablet",
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
        )


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_pk(
            drug_name="X",
            dose_mg=0.0,
            formulation="suppository",
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
        )


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_pk(
            drug_name="X",
            dose_mg=-10.0,
            formulation="enema",
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
        )


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_rectal_pk(
            drug_name="X",
            dose_mg=100.0,
            formulation="suppository",
            cl_sys_L_per_h=0.0,
            vd_sys_L=50.0,
        )
