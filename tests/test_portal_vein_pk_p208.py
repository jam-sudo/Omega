"""Tests for Phase 208 — portal_vein_pk_p208.py (~26 tests)."""

import pytest

from omega_pbpk.core.portal_vein_pk_p208 import (
    PortalPKResult,
    assess_first_pass,
    simulate_portal_pk,
)

ASSESS_FIRST_PASS_KEYS = {
    "hepatic_extraction_ratio",
    "fg_gut_bioavailability",
    "fh_hepatic_bioavailability",
    "f_oral_total",
    "high_first_pass",
}

# Reference call: medium-extraction drug
_REF_ARGS = dict(
    drug_name="RefDrug",
    dose_mg=100.0,
    fu_plasma=0.5,
    cl_intrinsic_L_per_h=50.0,
    cl_systemic_L_per_h=10.0,
    vd_sys_L=50.0,
)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_portal_pk(**_REF_ARGS)
    assert isinstance(result, PortalPKResult)


def test_field_drug_name():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.drug_name == "RefDrug"


def test_field_dose_mg():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.dose_mg == 100.0


def test_field_fu_plasma():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.fu_plasma == 0.5


def test_field_cl_intrinsic():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.cl_intrinsic_L_per_h == 50.0


# ---------------------------------------------------------------------------
# Systemic starts at 0
# ---------------------------------------------------------------------------


def test_c_systemic_starts_at_zero():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.c_systemic_mg_L[0] == 0.0


def test_c_portal_starts_at_zero():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.c_portal_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Cmax > 0
# ---------------------------------------------------------------------------


def test_cmax_systemic_positive():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.cmax_systemic_mg_L > 0.0


def test_auc_systemic_positive():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# E_h in [0, 1]
# ---------------------------------------------------------------------------


def test_e_hepatic_range():
    result = simulate_portal_pk(**_REF_ARGS)
    assert 0.0 <= result.e_hepatic <= 1.0


def test_e_hepatic_zero_clint():
    result = simulate_portal_pk(
        drug_name="NoCL",
        dose_mg=100.0,
        fu_plasma=0.5,
        cl_intrinsic_L_per_h=0.0,
        cl_systemic_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    assert result.e_hepatic == pytest.approx(0.0, abs=1e-9)


def test_e_hepatic_high_clint():
    result = simulate_portal_pk(
        drug_name="HighCL",
        dose_mg=100.0,
        fu_plasma=1.0,
        cl_intrinsic_L_per_h=9000.0,
        cl_systemic_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    assert result.e_hepatic > 0.9


# ---------------------------------------------------------------------------
# High CLint → low f_oral (high first-pass)
# ---------------------------------------------------------------------------


def test_high_clint_low_f_oral():
    result = simulate_portal_pk(
        drug_name="HighFirstPass",
        dose_mg=100.0,
        fu_plasma=1.0,
        cl_intrinsic_L_per_h=9000.0,
        cl_systemic_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    assert result.f_oral < 0.3


def test_low_clint_high_f_oral():
    result = simulate_portal_pk(
        drug_name="LowFirstPass",
        dose_mg=100.0,
        fu_plasma=0.1,
        cl_intrinsic_L_per_h=0.1,
        cl_systemic_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    # f_oral should be reasonably high (not drastically reduced)
    assert result.f_oral > 0.5


# ---------------------------------------------------------------------------
# f_oral in (0, 1]
# ---------------------------------------------------------------------------


def test_f_oral_range():
    result = simulate_portal_pk(**_REF_ARGS)
    assert 0.0 < result.f_oral <= 1.0


# ---------------------------------------------------------------------------
# Times list integrity
# ---------------------------------------------------------------------------


def test_times_h_starts_at_zero():
    result = simulate_portal_pk(**_REF_ARGS)
    assert result.times_h[0] == 0.0


def test_times_h_length_matches_c_systemic():
    result = simulate_portal_pk(**_REF_ARGS)
    assert len(result.times_h) == len(result.c_systemic_mg_L)


def test_c_gut_length_matches_times():
    result = simulate_portal_pk(**_REF_ARGS)
    assert len(result.c_gut_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# assess_first_pass
# ---------------------------------------------------------------------------


def test_assess_first_pass_returns_dict():
    result = simulate_portal_pk(**_REF_ARGS)
    info = assess_first_pass(result)
    assert isinstance(info, dict)


def test_assess_first_pass_keys():
    result = simulate_portal_pk(**_REF_ARGS)
    info = assess_first_pass(result)
    assert set(info.keys()) == ASSESS_FIRST_PASS_KEYS


def test_assess_first_pass_high_first_pass_true():
    result = simulate_portal_pk(
        drug_name="HighFP",
        dose_mg=100.0,
        fu_plasma=1.0,
        cl_intrinsic_L_per_h=9000.0,
        cl_systemic_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    info = assess_first_pass(result)
    assert info["high_first_pass"] is True


def test_assess_first_pass_low_first_pass_false():
    result = simulate_portal_pk(
        drug_name="LowFP",
        dose_mg=100.0,
        fu_plasma=0.1,
        cl_intrinsic_L_per_h=0.1,
        cl_systemic_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    info = assess_first_pass(result)
    assert info["high_first_pass"] is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_pk(
            drug_name="Bad",
            dose_mg=0.0,
            fu_plasma=0.5,
            cl_intrinsic_L_per_h=10.0,
            cl_systemic_L_per_h=5.0,
            vd_sys_L=50.0,
        )


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_pk(
            drug_name="Bad",
            dose_mg=-10.0,
            fu_plasma=0.5,
            cl_intrinsic_L_per_h=10.0,
            cl_systemic_L_per_h=5.0,
            vd_sys_L=50.0,
        )


def test_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_portal_pk(
            drug_name="Bad",
            dose_mg=100.0,
            fu_plasma=0.0,
            cl_intrinsic_L_per_h=10.0,
            cl_systemic_L_per_h=5.0,
            vd_sys_L=50.0,
        )


def test_fu_above_1_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_portal_pk(
            drug_name="Bad",
            dose_mg=100.0,
            fu_plasma=1.5,
            cl_intrinsic_L_per_h=10.0,
            cl_systemic_L_per_h=5.0,
            vd_sys_L=50.0,
        )


def test_clint_negative_raises():
    with pytest.raises(ValueError, match="cl_intrinsic"):
        simulate_portal_pk(
            drug_name="Bad",
            dose_mg=100.0,
            fu_plasma=0.5,
            cl_intrinsic_L_per_h=-5.0,
            cl_systemic_L_per_h=5.0,
            vd_sys_L=50.0,
        )
