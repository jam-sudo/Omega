"""Tests for Phase 932 — hemodialysis PK simulation (core/dialysis_pk.py)."""

import pytest
from omega_pbpk.core.dialysis_pk import (
    DialysisPKResult,
    simulate_dialysis_pk,
    compare_dialysis_vs_normal,
)


# --- Basic return type and field checks ---

def test_return_type_iv():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert isinstance(result, DialysisPKResult)


def test_return_type_oral():
    result = simulate_dialysis_pk("DrugB", dose_mg=100.0, route="oral")
    assert isinstance(result, DialysisPKResult)


def test_c_central_is_list_of_floats():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert isinstance(result.c_central_mg_L, list)
    assert all(isinstance(v, float) for v in result.c_central_mg_L)


def test_times_and_c_central_same_length():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert len(result.times_h) == len(result.c_central_mg_L)


def test_cmax_positive():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert result.auc_mg_h_per_L > 0


def test_tmax_non_negative():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert result.tmax_h >= 0


def test_fraction_removed_in_range():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert 0.0 <= result.fraction_removed_by_dialysis <= 1.0


def test_dialysis_sessions_at_least_one():
    # 168h simulation, sessions every 48h starting at t=0 → sessions at 0, 48, 96, 144 (4 sessions)
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv", t_end_h=168.0, dialysis_interval_h=48.0)
    assert result.dialysis_sessions_simulated >= 1


def test_dialysis_sessions_multiple():
    # 168h simulation with 48h interval starting at t=0 → at least 3 sessions
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv", t_end_h=168.0)
    assert result.dialysis_sessions_simulated >= 3


def test_t_half_interdialytic_positive():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert result.t_half_interdialytic_h > 0


def test_notes_non_empty():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, route="iv")
    assert isinstance(result.notes, str) and len(result.notes) > 0


# --- Route-specific initial conditions ---

def test_iv_initial_concentration_near_dose_over_vd():
    dose = 100.0
    vd = 20.0
    result = simulate_dialysis_pk("DrugA", dose_mg=dose, route="iv", vd_central_L=vd)
    expected = dose / vd
    # First concentration should be close to dose/Vd
    assert abs(result.c_central_mg_L[0] - expected) < 0.01 * expected + 0.01


def test_oral_initial_concentration_is_zero():
    result = simulate_dialysis_pk("DrugB", dose_mg=100.0, route="oral")
    assert result.c_central_mg_L[0] == 0.0


# --- Dialyzer clearance effect ---

def test_high_dialyzer_cl_higher_fraction_removed():
    high = simulate_dialysis_pk("DrugA", dose_mg=100.0, dialyzer_cl_L_per_h=20.0)
    low = simulate_dialysis_pk("DrugA", dose_mg=100.0, dialyzer_cl_L_per_h=5.0)
    assert high.fraction_removed_by_dialysis >= low.fraction_removed_by_dialysis


def test_zero_dialyzer_cl_no_removal():
    result = simulate_dialysis_pk("DrugA", dose_mg=100.0, dialyzer_cl_L_per_h=0.0)
    assert result.fraction_removed_by_dialysis == 0.0


# --- compare_dialysis_vs_normal ---

def test_compare_returns_dict_with_keys():
    result = compare_dialysis_vs_normal("DrugA", dose_mg=100.0)
    assert "dialysis" in result
    assert "normal_renal" in result
    assert "auc_ratio" in result


def test_compare_dialysis_key_is_result():
    result = compare_dialysis_vs_normal("DrugA", dose_mg=100.0)
    assert isinstance(result["dialysis"], DialysisPKResult)


def test_compare_normal_key_is_result():
    result = compare_dialysis_vs_normal("DrugA", dose_mg=100.0)
    assert isinstance(result["normal_renal"], DialysisPKResult)


def test_compare_auc_ratio_less_than_or_equal_one():
    """Dialysis removes extra drug → AUC_dialysis <= AUC_normal → ratio <= 1."""
    result = compare_dialysis_vs_normal(
        "DrugA", dose_mg=100.0,
        cl_hepatic_L_per_h=5.0,
        dialyzer_cl_L_per_h=12.0,
        normal_renal_cl_L_per_h=5.0,
    )
    assert result["auc_ratio"] <= 1.0


# --- Validation errors ---

def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dialysis_pk("DrugA", dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dialysis_pk("DrugA", dose_mg=-100.0)


def test_cl_hepatic_negative_raises():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        simulate_dialysis_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=-1.0)


def test_vd_central_zero_raises():
    with pytest.raises(ValueError, match="vd_central_L"):
        simulate_dialysis_pk("DrugA", dose_mg=100.0, vd_central_L=0.0)


def test_vd_central_negative_raises():
    with pytest.raises(ValueError, match="vd_central_L"):
        simulate_dialysis_pk("DrugA", dose_mg=100.0, vd_central_L=-10.0)


def test_dialyzer_cl_negative_raises():
    with pytest.raises(ValueError, match="dialyzer_cl_L_per_h"):
        simulate_dialysis_pk("DrugA", dose_mg=100.0, dialyzer_cl_L_per_h=-5.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_dialysis_pk("DrugA", dose_mg=100.0, route="sublingual")
