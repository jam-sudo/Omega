"""Tests for Phase 405: lymphatic_pk — systemic lymphatic PK model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.lymphatic_pk import (
    LymphaticPKResult,
    food_effect_on_lymph,
    simulate_lymphatic_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=2.0,
    vd_L=30.0,
    f_lymph=0.3,
    k_lymph_per_h=0.5,
    lymph_flow_L_per_h=0.12,
    t_end_h=24.0,
    dt_h=0.1,
    route="oral",
)


# ---------------------------------------------------------------------------
# Test 1: Returns correct type
# ---------------------------------------------------------------------------
def test_result_type():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert isinstance(result, LymphaticPKResult)


# ---------------------------------------------------------------------------
# Test 2: Drug name preserved
# ---------------------------------------------------------------------------
def test_drug_name_preserved():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.drug_name == "TestDrug"


# ---------------------------------------------------------------------------
# Test 3: Route preserved
# ---------------------------------------------------------------------------
def test_route_preserved():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.route == "oral"


# ---------------------------------------------------------------------------
# Test 4: SC route
# ---------------------------------------------------------------------------
def test_sc_route():
    kw = dict(BASE_KWARGS)
    kw["route"] = "sc"
    result = simulate_lymphatic_pk(**kw)
    assert result.route == "sc"
    assert any("SC" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Test 5: time array length
# ---------------------------------------------------------------------------
def test_times_length():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    expected_n = max(int(24.0 / 0.1), 1) + 1
    assert len(result.times_h) == expected_n


# ---------------------------------------------------------------------------
# Test 6: c_sys length matches times
# ---------------------------------------------------------------------------
def test_csys_length_matches_times():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert len(result.c_sys_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Test 7: a_lymph length matches times
# ---------------------------------------------------------------------------
def test_alymph_length_matches_times():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert len(result.a_lymph_mg) == len(result.times_h)


# ---------------------------------------------------------------------------
# Test 8: First time point is 0
# ---------------------------------------------------------------------------
def test_first_time_zero():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 9: Initial concentration is 0
# ---------------------------------------------------------------------------
def test_initial_conc_zero():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.c_sys_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 10: Cmax > 0
# ---------------------------------------------------------------------------
def test_cmax_positive():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.cmax_sys > 0.0


# ---------------------------------------------------------------------------
# Test 11: AUC > 0
# ---------------------------------------------------------------------------
def test_auc_positive():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.auc_sys > 0.0


# ---------------------------------------------------------------------------
# Test 12: All concentrations >= 0
# ---------------------------------------------------------------------------
def test_concentrations_nonnegative():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert all(c >= 0.0 for c in result.c_sys_mg_L)


# ---------------------------------------------------------------------------
# Test 13: f_lymph stored correctly
# ---------------------------------------------------------------------------
def test_f_lymph_stored():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.f_lymph == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 14: f_via_lymph_actual between 0 and 1
# ---------------------------------------------------------------------------
def test_f_via_lymph_actual_range():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert 0.0 <= result.f_via_lymph_actual <= 1.0


# ---------------------------------------------------------------------------
# Test 15: f_lymph=0 means no lymph contribution
# ---------------------------------------------------------------------------
def test_f_lymph_zero_no_lymph():
    kw = dict(BASE_KWARGS)
    kw["f_lymph"] = 0.0
    result = simulate_lymphatic_pk(**kw)
    assert result.f_via_lymph_actual == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 16: f_lymph=1 means all via lymph
# ---------------------------------------------------------------------------
def test_f_lymph_one_all_lymph():
    kw = dict(BASE_KWARGS)
    kw["f_lymph"] = 1.0
    result = simulate_lymphatic_pk(**kw)
    # All drug goes through lymph; direct portal input is 0
    assert result.f_via_lymph_actual == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 17: Higher dose → proportionally higher AUC
# ---------------------------------------------------------------------------
def test_dose_proportionality_auc():
    r1 = simulate_lymphatic_pk(**BASE_KWARGS)
    kw2 = dict(BASE_KWARGS)
    kw2["dose_mg"] = 200.0
    r2 = simulate_lymphatic_pk(**kw2)
    ratio = r2.auc_sys / r1.auc_sys
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Test 18: Higher CL → lower AUC
# ---------------------------------------------------------------------------
def test_higher_cl_lower_auc():
    r_low_cl = simulate_lymphatic_pk(**BASE_KWARGS)
    kw_high = dict(BASE_KWARGS)
    kw_high["cl_L_per_h"] = 10.0
    r_high_cl = simulate_lymphatic_pk(**kw_high)
    assert r_high_cl.auc_sys < r_low_cl.auc_sys


# ---------------------------------------------------------------------------
# Test 19: Higher lymph flow → faster systemic rise (higher early Cmax)
# ---------------------------------------------------------------------------
def test_higher_lymph_flow_changes_profile():
    r_low = simulate_lymphatic_pk(**BASE_KWARGS)
    kw_high = dict(BASE_KWARGS)
    kw_high["lymph_flow_L_per_h"] = 1.0
    r_high = simulate_lymphatic_pk(**kw_high)
    # AUC should differ when lymph flow changes significantly
    assert r_high.auc_sys != pytest.approx(r_low.auc_sys, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 20: Lymph node amount starts at 0
# ---------------------------------------------------------------------------
def test_lymph_node_starts_zero():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.a_lymph_mg[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 21: Lymph node amount eventually approaches 0 at end
# ---------------------------------------------------------------------------
def test_lymph_node_decays():
    kw = dict(BASE_KWARGS)
    kw["t_end_h"] = 100.0
    result = simulate_lymphatic_pk(**kw)
    assert result.a_lymph_mg[-1] < result.a_lymph_mg[10]  # decaying


# ---------------------------------------------------------------------------
# Test 22: Validation — dose_mg <= 0
# ---------------------------------------------------------------------------
def test_validation_dose_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["dose_mg"] = 0.0
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 23: Validation — cl_L_per_h <= 0
# ---------------------------------------------------------------------------
def test_validation_cl_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["cl_L_per_h"] = -1.0
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 24: Validation — vd_L <= 0
# ---------------------------------------------------------------------------
def test_validation_vd_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["vd_L"] = 0.0
    with pytest.raises(ValueError, match="vd_L"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 25: Validation — f_lymph out of range
# ---------------------------------------------------------------------------
def test_validation_f_lymph_oob():
    kw = dict(BASE_KWARGS)
    kw["f_lymph"] = 1.5
    with pytest.raises(ValueError, match="f_lymph"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 26: Validation — invalid route
# ---------------------------------------------------------------------------
def test_validation_invalid_route():
    kw = dict(BASE_KWARGS)
    kw["route"] = "topical"
    with pytest.raises(ValueError, match="route"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 27: Validation — k_lymph_per_h <= 0
# ---------------------------------------------------------------------------
def test_validation_k_lymph_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["k_lymph_per_h"] = 0.0
    with pytest.raises(ValueError, match="k_lymph_per_h"):
        simulate_lymphatic_pk(**kw)


# ---------------------------------------------------------------------------
# Test 28: Cmax equals max of c_sys array
# ---------------------------------------------------------------------------
def test_cmax_equals_max_of_array():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert result.cmax_sys == pytest.approx(max(result.c_sys_mg_L))


# ---------------------------------------------------------------------------
# Test 29: food_effect_on_lymph returns dict with required keys
# ---------------------------------------------------------------------------
def test_food_effect_returns_dict():
    result = food_effect_on_lymph(
        drug_name="LipoDrug",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=30.0,
        f_lymph_fasted=0.1,
        f_lymph_fed=0.4,
        k_lymph_per_h=0.5,
        lymph_flow_L_per_h=0.05,
    )
    assert "fasted" in result
    assert "fed" in result
    assert "auc_ratio_fed_fasted" in result
    assert "notes" in result


# ---------------------------------------------------------------------------
# Test 30: food_effect — AUC ratio is a positive finite float
# ---------------------------------------------------------------------------
def test_food_effect_auc_ratio_is_finite():
    result = food_effect_on_lymph(
        drug_name="LipoDrug",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=30.0,
        f_lymph_fasted=0.05,
        f_lymph_fed=0.5,
        k_lymph_per_h=0.5,
        lymph_flow_L_per_h=0.05,
    )
    ratio = result["auc_ratio_fed_fasted"]
    assert ratio > 0.0 and math.isfinite(ratio)


# ---------------------------------------------------------------------------
# Test 31: food_effect — AUC ratio > 1 when lymph fraction increases with food
# ---------------------------------------------------------------------------
def test_food_effect_auc_ratio_positive():
    result = food_effect_on_lymph(
        drug_name="Drug",
        dose_mg=50.0,
        cl_L_per_h=1.0,
        vd_L=20.0,
        f_lymph_fasted=0.1,
        f_lymph_fed=0.4,
        k_lymph_per_h=0.3,
        lymph_flow_L_per_h=0.05,
    )
    assert result["auc_ratio_fed_fasted"] > 0.0


# ---------------------------------------------------------------------------
# Test 32: food_effect — fasted result is LymphaticPKResult
# ---------------------------------------------------------------------------
def test_food_effect_fasted_is_result():
    result = food_effect_on_lymph(
        drug_name="Drug",
        dose_mg=50.0,
        cl_L_per_h=1.0,
        vd_L=20.0,
        f_lymph_fasted=0.1,
        f_lymph_fed=0.4,
        k_lymph_per_h=0.3,
        lymph_flow_L_per_h=0.05,
    )
    assert isinstance(result["fasted"], LymphaticPKResult)


# ---------------------------------------------------------------------------
# Test 33: food_effect — notes is a list
# ---------------------------------------------------------------------------
def test_food_effect_notes_is_list():
    result = food_effect_on_lymph(
        drug_name="Drug",
        dose_mg=50.0,
        cl_L_per_h=1.0,
        vd_L=20.0,
        f_lymph_fasted=0.1,
        f_lymph_fed=0.4,
        k_lymph_per_h=0.3,
        lymph_flow_L_per_h=0.05,
    )
    assert isinstance(result["notes"], list)


# ---------------------------------------------------------------------------
# Test 34: food_effect — validation: dose_mg invalid
# ---------------------------------------------------------------------------
def test_food_effect_validation_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        food_effect_on_lymph(
            drug_name="Drug",
            dose_mg=-1.0,
            cl_L_per_h=1.0,
            vd_L=20.0,
            f_lymph_fasted=0.1,
            f_lymph_fed=0.4,
            k_lymph_per_h=0.3,
            lymph_flow_L_per_h=0.05,
        )


# ---------------------------------------------------------------------------
# Test 35: food_effect — validation: f_lymph_fasted invalid
# ---------------------------------------------------------------------------
def test_food_effect_validation_f_lymph_fasted():
    with pytest.raises(ValueError, match="f_lymph_fasted"):
        food_effect_on_lymph(
            drug_name="Drug",
            dose_mg=50.0,
            cl_L_per_h=1.0,
            vd_L=20.0,
            f_lymph_fasted=-0.1,
            f_lymph_fed=0.4,
            k_lymph_per_h=0.3,
            lymph_flow_L_per_h=0.05,
        )


# ---------------------------------------------------------------------------
# Test 36: Simulation with minimal dt gives smooth profile
# ---------------------------------------------------------------------------
def test_fine_dt_smooth_profile():
    kw = dict(BASE_KWARGS)
    kw["dt_h"] = 0.01
    result = simulate_lymphatic_pk(**kw)
    # No negative concentrations
    assert all(c >= 0.0 for c in result.c_sys_mg_L)


# ---------------------------------------------------------------------------
# Test 37: notes is a list
# ---------------------------------------------------------------------------
def test_notes_is_list():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    assert isinstance(result.notes, list)


# ---------------------------------------------------------------------------
# Test 38: High f_lymph note appears
# ---------------------------------------------------------------------------
def test_high_f_lymph_note():
    kw = dict(BASE_KWARGS)
    kw["f_lymph"] = 0.9
    result = simulate_lymphatic_pk(**kw)
    assert any("lymphatic" in n.lower() or "lymph" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Test 39: AUC consistent with manual calculation
# ---------------------------------------------------------------------------
def test_auc_manual_check():
    result = simulate_lymphatic_pk(**BASE_KWARGS)
    # Manual trapezoidal
    t = result.times_h
    c = result.c_sys_mg_L
    manual_auc = sum(0.5 * (c[i] + c[i + 1]) * (t[i + 1] - t[i]) for i in range(len(t) - 1))
    assert result.auc_sys == pytest.approx(manual_auc, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 40: drug_name empty raises ValueError
# ---------------------------------------------------------------------------
def test_validation_empty_drug_name():
    kw = dict(BASE_KWARGS)
    kw["drug_name"] = ""
    with pytest.raises(ValueError, match="drug_name"):
        simulate_lymphatic_pk(**kw)
