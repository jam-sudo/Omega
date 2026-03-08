"""Tests for Phase 967 — Lithium PK model."""

import math

import pytest

from omega_pbpk.clinical.lithium_pk import (
    LithiumPKResult,
    find_therapeutic_dose,
    simulate_lithium_pk,
)


def test_return_type():
    result = simulate_lithium_pk()
    assert isinstance(result, LithiumPKResult)


def test_cmax_positive():
    result = simulate_lithium_pk()
    assert result.cmax_mg_L > 0


def test_cmin_nonnegative():
    result = simulate_lithium_pk()
    assert result.cmin_mg_L >= 0


def test_t_half_positive():
    result = simulate_lithium_pk()
    assert result.t_half_h > 0


def test_css_max_gt_css_min():
    result = simulate_lithium_pk()
    assert result.css_max_mg_L > result.css_min_mg_L


def test_in_therapeutic_range_is_bool():
    result = simulate_lithium_pk()
    assert isinstance(result.in_therapeutic_range, bool)


def test_crcl_positive():
    result = simulate_lithium_pk()
    assert result.crcl_ml_min > 0


def test_cl_Li_positive():
    result = simulate_lithium_pk()
    assert result.cl_Li_L_per_h > 0


def test_vd_positive():
    result = simulate_lithium_pk()
    assert result.vd_L > 0


def test_higher_scr_lower_crcl_higher_css():
    """Higher SCr → lower CrCl → lower CL_Li → higher CSS_min."""
    r_normal = simulate_lithium_pk(scr_mg_dL=1.0)
    r_high = simulate_lithium_pk(scr_mg_dL=2.5)
    assert r_high.crcl_ml_min < r_normal.crcl_ml_min
    assert r_high.cl_Li_L_per_h < r_normal.cl_Li_L_per_h
    assert r_high.css_min_mg_L > r_normal.css_min_mg_L


def test_female_lower_crcl_than_male():
    """Female sex → lower CrCl than male (same params)."""
    r_male = simulate_lithium_pk(sex="male")
    r_female = simulate_lithium_pk(sex="female")
    assert r_female.crcl_ml_min < r_male.crcl_ml_min


def test_find_therapeutic_dose_returns_dict():
    result = find_therapeutic_dose()
    assert isinstance(result, dict)


def test_find_therapeutic_dose_has_required_keys():
    result = find_therapeutic_dose()
    assert "recommended_dose_mg" in result
    assert "predicted_css_min_mg_L" in result
    assert "predicted_css_max_mg_L" in result
    assert "in_therapeutic_range" in result


def test_recommended_dose_positive():
    result = find_therapeutic_dose()
    assert result["recommended_dose_mg"] > 0


def test_in_therapeutic_range_bool_from_find():
    result = find_therapeutic_dose()
    assert isinstance(result["in_therapeutic_range"], bool)


def test_notes_nonempty():
    result = simulate_lithium_pk()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_times_h_nonempty():
    result = simulate_lithium_pk()
    assert len(result.times_h) > 0


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError):
        simulate_lithium_pk(dose_mg=0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError):
        simulate_lithium_pk(dose_mg=-100)


def test_validation_weight_kg_zero():
    with pytest.raises(ValueError):
        simulate_lithium_pk(weight_kg=0)


def test_validation_age_years_zero():
    with pytest.raises(ValueError):
        simulate_lithium_pk(age_years=0)


def test_validation_scr_zero():
    with pytest.raises(ValueError):
        simulate_lithium_pk(scr_mg_dL=0)


def test_validation_invalid_sex():
    with pytest.raises(ValueError):
        simulate_lithium_pk(sex="other")


def test_validation_n_doses_zero():
    with pytest.raises(ValueError):
        simulate_lithium_pk(n_doses=0)


def test_t_half_formula():
    """t_half = ln2 * Vd / CL (within 10%)."""
    result = simulate_lithium_pk()
    expected_t_half = math.log(2) * result.vd_L / result.cl_Li_L_per_h
    assert abs(result.t_half_h - expected_t_half) / expected_t_half < 0.10


def test_vd_proportional_to_weight():
    r_light = simulate_lithium_pk(weight_kg=50.0)
    r_heavy = simulate_lithium_pk(weight_kg=100.0)
    assert r_heavy.vd_L > r_light.vd_L


def test_css_min_increases_with_dose():
    r_low = simulate_lithium_pk(dose_mg=150)
    r_high = simulate_lithium_pk(dose_mg=600)
    assert r_high.css_min_mg_L > r_low.css_min_mg_L
