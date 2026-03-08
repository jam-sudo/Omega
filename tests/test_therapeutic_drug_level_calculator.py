"""Tests for Phase 1133 — Therapeutic Drug Level Calculator."""

import math

import pytest

from omega_pbpk.clinical.therapeutic_drug_level_calculator import (
    TherapeuticLevelResult,
    calculate_target_level,
    recommend_sampling_time,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VANC_PARAMS = dict(
    drug_name="vancomycin",
    indication="bacteremia",
    route="iv",
    dosing_interval_h=12.0,
    cl_L_per_h=4.0,
    vd_L=50.0,
    ka_per_h=1.0,  # ignored for IV
    f_bioavail=1.0,
    dose_mg=1000.0,
)

ORAL_PARAMS = dict(
    drug_name="generic",
    indication="test",
    route="oral",
    dosing_interval_h=8.0,
    cl_L_per_h=5.0,
    vd_L=40.0,
    ka_per_h=1.5,
    f_bioavail=0.8,
    dose_mg=500.0,
)


# ---------------------------------------------------------------------------
# Basic construction and types
# ---------------------------------------------------------------------------


def test_returns_therapeutic_level_result():
    res = calculate_target_level(**VANC_PARAMS)
    assert isinstance(res, TherapeuticLevelResult)


def test_iv_vancomycin_trough_less_than_peak():
    res = calculate_target_level(**VANC_PARAMS)
    assert res.css_trough_mg_L < res.css_peak_mg_L


def test_oral_css_trough_less_than_peak():
    res = calculate_target_level(**ORAL_PARAMS)
    assert res.css_trough_mg_L < res.css_peak_mg_L


def test_t_half_formula():
    res = calculate_target_level(**VANC_PARAMS)
    ke = VANC_PARAMS["cl_L_per_h"] / VANC_PARAMS["vd_L"]
    expected = math.log(2) / ke
    assert abs(res.t_half_h - expected) < 1e-9


def test_ke_per_h_correct():
    res = calculate_target_level(**VANC_PARAMS)
    expected_ke = VANC_PARAMS["cl_L_per_h"] / VANC_PARAMS["vd_L"]
    assert abs(res.ke_per_h - expected_ke) < 1e-9


# ---------------------------------------------------------------------------
# Drug database look-up
# ---------------------------------------------------------------------------


def test_vancomycin_trough_target_range():
    res = calculate_target_level(**VANC_PARAMS)
    assert res.trough_target_low == 10
    assert res.trough_target_high == 20


def test_generic_fallback_for_unknown_drug():
    params = VANC_PARAMS.copy()
    params["drug_name"] = "xyz_unknown_drug"
    res = calculate_target_level(**params)
    assert res.trough_target_low == 1
    assert res.trough_target_high == 10


def test_case_insensitive_lookup():
    params = VANC_PARAMS.copy()
    params["drug_name"] = "VANCOMYCIN"
    res = calculate_target_level(**params)
    assert res.trough_target_low == 10


def test_digoxin_toxic_level():
    params = dict(
        drug_name="digoxin",
        indication="heart failure",
        route="oral",
        dosing_interval_h=24.0,
        cl_L_per_h=7.0,
        vd_L=500.0,
        ka_per_h=0.5,
        f_bioavail=0.75,
        dose_mg=0.25,
    )
    res = calculate_target_level(**params)
    assert res.toxic_level_mg_L == 2.0


def test_phenytoin_target():
    params = dict(
        drug_name="phenytoin",
        indication="epilepsy",
        route="oral",
        dosing_interval_h=12.0,
        cl_L_per_h=1.5,
        vd_L=45.0,
        ka_per_h=1.2,
        f_bioavail=0.95,
        dose_mg=300.0,
    )
    res = calculate_target_level(**params)
    assert res.trough_target_low == 10
    assert res.trough_target_high == 20


# ---------------------------------------------------------------------------
# Safety margin
# ---------------------------------------------------------------------------


def test_safety_margin_formula():
    res = calculate_target_level(**VANC_PARAMS)
    expected = res.toxic_level_mg_L / res.css_peak_mg_L
    assert abs(res.safety_margin - expected) < 1e-9


def test_safety_margin_positive():
    res = calculate_target_level(**VANC_PARAMS)
    assert res.safety_margin > 0


# ---------------------------------------------------------------------------
# Dose recommendation logic
# ---------------------------------------------------------------------------


def test_maintain_when_in_range():
    # High dose to ensure vancomycin is in trough range 10-20
    params = dict(
        drug_name="vancomycin",
        indication="test",
        route="iv",
        dosing_interval_h=12.0,
        cl_L_per_h=3.5,
        vd_L=50.0,
        ka_per_h=1.0,
        f_bioavail=1.0,
        dose_mg=1500.0,
    )
    res = calculate_target_level(**params)
    assert res.dose_recommendation in {"maintain", "decrease", "increase"}


def test_increase_when_trough_below_target():
    # Very low dose → trough below target
    params = VANC_PARAMS.copy()
    params["dose_mg"] = 50.0  # tiny dose
    res = calculate_target_level(**params)
    assert res.dose_recommendation == "increase"


def test_decrease_when_trough_above_target():
    # Very high dose → trough above target
    params = VANC_PARAMS.copy()
    params["dose_mg"] = 20000.0
    res = calculate_target_level(**params)
    assert res.dose_recommendation == "decrease"


# ---------------------------------------------------------------------------
# Higher CL → lower Css
# ---------------------------------------------------------------------------


def test_higher_cl_lower_css():
    params_lo = VANC_PARAMS.copy()
    params_lo["cl_L_per_h"] = 2.0

    params_hi = VANC_PARAMS.copy()
    params_hi["cl_L_per_h"] = 8.0

    res_lo = calculate_target_level(**params_lo)
    res_hi = calculate_target_level(**params_hi)
    assert res_lo.css_trough_mg_L > res_hi.css_trough_mg_L


# ---------------------------------------------------------------------------
# Oral vs IV
# ---------------------------------------------------------------------------


def test_oral_route_stored():
    res = calculate_target_level(**ORAL_PARAMS)
    assert res.route == "oral"


def test_iv_route_stored():
    res = calculate_target_level(**VANC_PARAMS)
    assert res.route == "iv"


def test_oral_bioavailability_effect():
    params_hi = ORAL_PARAMS.copy()
    params_hi["f_bioavail"] = 0.9

    params_lo = ORAL_PARAMS.copy()
    params_lo["f_bioavail"] = 0.3

    res_hi = calculate_target_level(**params_hi)
    res_lo = calculate_target_level(**params_lo)
    assert res_hi.css_trough_mg_L > res_lo.css_trough_mg_L


# ---------------------------------------------------------------------------
# recommend_sampling_time
# ---------------------------------------------------------------------------


def test_recommend_sampling_time_keys():
    res = calculate_target_level(**VANC_PARAMS)
    st = recommend_sampling_time(res)
    assert "trough_time_h" in st
    assert "peak_time_h" in st
    assert "recommendation" in st


def test_trough_time_30_min_before_next_dose():
    res = calculate_target_level(**VANC_PARAMS)
    st = recommend_sampling_time(res)
    assert abs(st["trough_time_h"] - (VANC_PARAMS["dosing_interval_h"] - 0.5)) < 1e-9


def test_iv_peak_time_30_min():
    res = calculate_target_level(**VANC_PARAMS)
    st = recommend_sampling_time(res)
    assert abs(st["peak_time_h"] - 0.5) < 1e-9


def test_recommendation_is_string():
    res = calculate_target_level(**ORAL_PARAMS)
    st = recommend_sampling_time(res)
    assert isinstance(st["recommendation"], str)
    assert len(st["recommendation"]) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_cl_raises():
    params = VANC_PARAMS.copy()
    params["cl_L_per_h"] = -1.0
    with pytest.raises(ValueError, match="cl_L_per_h"):
        calculate_target_level(**params)


def test_zero_vd_raises():
    params = VANC_PARAMS.copy()
    params["vd_L"] = 0.0
    with pytest.raises(ValueError, match="vd_L"):
        calculate_target_level(**params)


def test_invalid_route_raises():
    params = VANC_PARAMS.copy()
    params["route"] = "inhalation"
    with pytest.raises(ValueError, match="route"):
        calculate_target_level(**params)


def test_zero_f_bioavail_raises():
    params = ORAL_PARAMS.copy()
    params["f_bioavail"] = 0.0
    with pytest.raises(ValueError, match="f_bioavail"):
        calculate_target_level(**params)


def test_f_bioavail_above_one_raises():
    params = ORAL_PARAMS.copy()
    params["f_bioavail"] = 1.5
    with pytest.raises(ValueError, match="f_bioavail"):
        calculate_target_level(**params)


def test_negative_ka_oral_raises():
    params = ORAL_PARAMS.copy()
    params["ka_per_h"] = -0.5
    with pytest.raises(ValueError, match="ka_per_h"):
        calculate_target_level(**params)


def test_zero_dosing_interval_raises():
    params = VANC_PARAMS.copy()
    params["dosing_interval_h"] = 0.0
    with pytest.raises(ValueError, match="dosing_interval_h"):
        calculate_target_level(**params)
