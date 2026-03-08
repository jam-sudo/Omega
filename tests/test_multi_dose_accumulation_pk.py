"""Tests for Phase 947 — Multi-dose accumulation PK."""

import pytest

from omega_pbpk.core.multi_dose_accumulation_pk import (
    MultiDoseAccumulationResult,
    find_steady_state_dose,
    simulate_accumulation,
)

# --- Basic return type and field presence ---


def test_return_type():
    result = simulate_accumulation("TestDrug", dose_mg=100.0)
    assert isinstance(result, MultiDoseAccumulationResult)


def test_css_max_positive():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.css_max_theoretical > 0


def test_css_min_positive():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.css_min_theoretical > 0


def test_css_max_greater_than_min():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.css_max_theoretical > result.css_min_theoretical


def test_accumulation_ratio_gte_one():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.accumulation_ratio >= 1.0


def test_time_to_90ss_positive():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.time_to_90ss_h > 0


def test_t_half_positive():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.t_half_h > 0


def test_fluctuation_non_negative():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.fluctuation_pct >= 0


def test_cmax_last_dose_positive():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert result.cmax_last_dose > 0


def test_times_non_empty():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert len(result.times_h) > 0


def test_c_plasma_same_length_as_times():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_notes_non_empty():
    result = simulate_accumulation("Drug", dose_mg=100.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# --- Physics / pharmacology checks ---


def test_short_interval_higher_accumulation():
    """Shorter dosing interval → more accumulation."""
    short = simulate_accumulation(
        "Drug", dose_mg=100.0, dosing_interval_h=4.0, cl_L_per_h=10.0, vd_L=50.0
    )
    long = simulate_accumulation(
        "Drug", dose_mg=100.0, dosing_interval_h=24.0, cl_L_per_h=10.0, vd_L=50.0
    )
    assert short.accumulation_ratio > long.accumulation_ratio


def test_accumulation_ratio_formula():
    """R_acc = Css_max / (dose/Vd) within 1%."""
    dose_mg = 100.0
    vd_L = 50.0
    result = simulate_accumulation(
        "Drug", dose_mg=dose_mg, vd_L=vd_L, dosing_interval_h=12.0, cl_L_per_h=5.0, route="iv"
    )
    c0 = dose_mg / vd_L
    expected_r_acc = result.css_max_theoretical / c0
    assert abs(result.accumulation_ratio - expected_r_acc) / expected_r_acc < 0.01


def test_long_t_half_high_accumulation():
    """Long t½ relative to dosing interval → high R_acc."""
    # ke very small → long t½
    result = simulate_accumulation(
        "LongHalf", dose_mg=100.0, cl_L_per_h=0.5, vd_L=50.0, dosing_interval_h=12.0
    )
    assert result.accumulation_ratio > 5.0


def test_n_doses_1_still_computes():
    """n_doses=1 produces valid accumulation_ratio."""
    result = simulate_accumulation("Drug", dose_mg=100.0, n_doses=1)
    assert result.accumulation_ratio >= 1.0


def test_iv_route_works():
    result = simulate_accumulation("Drug", dose_mg=100.0, route="iv")
    assert result.css_max_theoretical > 0


def test_oral_route_works():
    result = simulate_accumulation(
        "Drug", dose_mg=100.0, route="oral", ka_per_h=1.5, bioavailability=0.8
    )
    assert result.css_max_theoretical > 0


def test_simulate_washout_false():
    result = simulate_accumulation("Drug", dose_mg=100.0, simulate_washout=False)
    assert len(result.times_h) > 0


# --- find_steady_state_dose ---


def test_find_ss_dose_returns_dict():
    result = find_steady_state_dose("Drug", 1.0, 5.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result, dict)


def test_find_ss_dose_has_required_keys():
    result = find_steady_state_dose("Drug", 1.0, 5.0, cl_L_per_h=5.0, vd_L=50.0)
    assert "optimal_dose_mg" in result
    assert "predicted_css_min" in result
    assert "predicted_css_max" in result
    assert "within_therapeutic_window" in result


def test_find_ss_dose_optimal_dose_positive():
    result = find_steady_state_dose("Drug", 1.0, 5.0, cl_L_per_h=5.0, vd_L=50.0)
    assert result["optimal_dose_mg"] > 0


def test_find_ss_dose_within_window_is_bool():
    result = find_steady_state_dose("Drug", 1.0, 5.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result["within_therapeutic_window"], bool)


# --- Validation ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_accumulation("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_accumulation("Drug", dose_mg=-10.0)


def test_validation_dosing_interval_zero():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        simulate_accumulation("Drug", dose_mg=100.0, dosing_interval_h=0.0)


def test_validation_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_accumulation("Drug", dose_mg=100.0, n_doses=0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_accumulation("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_accumulation("Drug", dose_mg=100.0, vd_L=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_accumulation("Drug", dose_mg=100.0, route="subcutaneous")


def test_validation_bioavailability_zero():
    with pytest.raises(ValueError, match="bioavailability"):
        simulate_accumulation("Drug", dose_mg=100.0, bioavailability=0.0)
