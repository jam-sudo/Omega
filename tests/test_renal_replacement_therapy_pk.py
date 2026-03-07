"""Tests for Phase 945 — Renal Replacement Therapy PK."""

import math
import pytest
from omega_pbpk.clinical.renal_replacement_therapy_pk import (
    CRRTPKResult, simulate_crrt_pk, compare_crrt_modes,
)


def make_default(**kwargs):
    defaults = dict(
        drug_name="VancomycinTest",
        dose_mg=1000.0,
        fu_plasma=0.3,
        cl_hepatic_L_per_h=1.0,
        cl_residual_renal_L_per_h=0.0,
        vd_L=50.0,
        crrt_mode="CVVHDF",
        effluent_rate_L_per_h=1.5,
        route="iv_bolus",
        infusion_rate_mg_h=0.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_crrt_pk(**defaults)


# --- Return type ---

def test_return_type():
    result = make_default()
    assert isinstance(result, CRRTPKResult)


# --- Basic PK metrics ---

def test_cmax_positive():
    result = make_default()
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = make_default()
    assert result.auc_mg_h_per_L > 0


def test_t_half_positive():
    result = make_default()
    assert result.t_half_h > 0


# --- CRRT clearance ---

def test_cl_crrt_positive_for_cvvhdf():
    result = make_default(crrt_mode="CVVHDF")
    assert result.cl_crrt_L_per_h > 0


def test_cl_crrt_positive_for_cvvh():
    result = make_default(crrt_mode="CVVH")
    assert result.cl_crrt_L_per_h > 0


def test_cl_crrt_positive_for_cvvhd():
    result = make_default(crrt_mode="CVVHD")
    assert result.cl_crrt_L_per_h > 0


def test_cl_crrt_zero_for_none():
    result = make_default(crrt_mode="none")
    assert result.cl_crrt_L_per_h == 0.0


# --- Fraction removed ---

def test_fraction_removed_in_range():
    result = make_default()
    assert 0.0 <= result.fraction_removed_by_crrt <= 1.0


def test_fraction_removed_zero_for_none():
    result = make_default(crrt_mode="none")
    assert result.fraction_removed_by_crrt == 0.0


# --- High fu_plasma → higher CL_CRRT ---

def test_high_fu_higher_cl_crrt():
    low = make_default(fu_plasma=0.1)
    high = make_default(fu_plasma=0.9)
    assert high.cl_crrt_L_per_h > low.cl_crrt_L_per_h


# --- compare_crrt_modes ---

def test_compare_returns_four_results():
    results = compare_crrt_modes("Vancomycin", dose_mg=1000.0)
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_crrt_modes("Vancomycin", dose_mg=1000.0)
    fracs = [r.fraction_removed_by_crrt for r in results]
    assert fracs == sorted(fracs, reverse=True)


def test_compare_none_has_lowest_fraction():
    results = compare_crrt_modes("Vancomycin", dose_mg=1000.0)
    none_result = next(r for r in results if r.crrt_mode == "none")
    assert none_result.fraction_removed_by_crrt == 0.0
    # "none" should be last since sorted descending
    assert results[-1].crrt_mode == "none"


# --- IV infusion starts at 0 ---

def test_iv_infusion_starts_at_zero():
    result = make_default(route="iv_infusion", infusion_rate_mg_h=50.0)
    assert result.c_plasma_mg_L[0] == 0.0


# --- IV bolus starts > 0 ---

def test_iv_bolus_starts_positive():
    result = make_default(route="iv_bolus")
    assert result.c_plasma_mg_L[0] > 0


# --- Notes non-empty ---

def test_notes_non_empty():
    result = make_default()
    assert len(result.notes) > 0


# --- Lengths match ---

def test_time_concentration_same_length():
    result = make_default()
    assert len(result.times_h) == len(result.c_plasma_mg_L)


# --- t_half validation ---

def test_t_half_matches_formula():
    result = make_default(
        fu_plasma=0.3,
        cl_hepatic_L_per_h=2.0,
        cl_residual_renal_L_per_h=0.0,
        vd_L=50.0,
        crrt_mode="CVVHDF",
        effluent_rate_L_per_h=1.5,
        replacement_flow_L_per_h=1.5,
        dialysate_flow_L_per_h=1.5,
    )
    # CL_CRRT for CVVHDF = fu * 0.5*(rep+dial) = 0.3 * 1.5 = 0.45
    cl_crrt = 0.3 * (0.5 * 1.5 + 0.5 * 1.5)
    cl_total = 2.0 + cl_crrt
    expected_t_half = math.log(2) * 50.0 / cl_total
    assert abs(result.t_half_h - expected_t_half) < 0.01


# --- Validation errors ---

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        make_default(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        make_default(dose_mg=-10.0)


def test_validation_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        make_default(fu_plasma=0.0)


def test_validation_fu_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        make_default(fu_plasma=1.1)


def test_validation_cl_hepatic_negative():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        make_default(cl_hepatic_L_per_h=-1.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        make_default(vd_L=0.0)


def test_validation_invalid_crrt_mode():
    with pytest.raises(ValueError, match="crrt_mode"):
        make_default(crrt_mode="SLEDD")


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        make_default(route="oral")
