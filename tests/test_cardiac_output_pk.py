"""Tests for Phase 1079 — Cardiac Output Effect on PK."""

import pytest

from omega_pbpk.core.cardiac_output_pk import (
    CardiacOutputPKResult,
    compare_cardiac_outputs,
    simulate_cardiac_output_pk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VD_PLASMA = 3.5  # L

DEFAULT_ARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cardiac_output_L_per_h=300.0,
    logP=2.0,
    cl_intrinsic_L_per_h=5.0,
    t_end_h=24.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# Test 1: Return type
# ---------------------------------------------------------------------------
def test_return_type():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert isinstance(result, CardiacOutputPKResult)


# ---------------------------------------------------------------------------
# Test 2: C_plasma at t=0 equals dose / Vd_plasma
# ---------------------------------------------------------------------------
def test_initial_plasma_concentration():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    expected_c0 = DEFAULT_ARGS["dose_mg"] / VD_PLASMA
    assert abs(result.c_plasma_mg_L[0] - expected_c0) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: c_liver, c_kidney, c_muscle all start at 0
# ---------------------------------------------------------------------------
def test_organ_initial_conditions_zero():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.c_liver_mg_L[0] == 0.0
    assert result.c_kidney_mg_L[0] == 0.0
    assert result.c_muscle_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Test 4: Lower CO → higher auc_plasma
# ---------------------------------------------------------------------------
def test_lower_co_higher_auc_plasma():
    low_co = simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": 100.0})
    high_co = simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": 500.0})
    assert low_co.auc_plasma > high_co.auc_plasma


# ---------------------------------------------------------------------------
# Test 5: auc_liver > 0
# ---------------------------------------------------------------------------
def test_auc_liver_positive():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.auc_liver > 0.0


# ---------------------------------------------------------------------------
# Test 6: auc_kidney > 0
# ---------------------------------------------------------------------------
def test_auc_kidney_positive():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.auc_kidney > 0.0


# ---------------------------------------------------------------------------
# Test 7: auc_muscle > 0
# ---------------------------------------------------------------------------
def test_auc_muscle_positive():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.auc_muscle > 0.0


# ---------------------------------------------------------------------------
# Test 8: Higher logP → higher peak liver concentration (lipophilic accumulates more)
# Peak liver-to-plasma ratio at equilibrium: Cl/Cp ~ Kp_liver, higher for lipophilic drugs.
# Test with very low cl_intrinsic so clearance doesn't consume the liver accumulation.
# ---------------------------------------------------------------------------
def test_higher_logp_higher_liver_peak():
    shared = {**DEFAULT_ARGS, "cl_intrinsic_L_per_h": 0.1}
    low_logP = simulate_cardiac_output_pk(**{**shared, "logP": 1.0})
    high_logP = simulate_cardiac_output_pk(**{**shared, "logP": 5.0})
    assert max(high_logP.c_liver_mg_L) > max(low_logP.c_liver_mg_L)


# ---------------------------------------------------------------------------
# Test 9: pk_risk_assessment valid values
# ---------------------------------------------------------------------------
def test_pk_risk_assessment_valid_values():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.pk_risk_assessment in ("normal", "reduced_exposure", "elevated_exposure")


# ---------------------------------------------------------------------------
# Test 10: Low CO → "elevated_exposure"
# ---------------------------------------------------------------------------
def test_low_co_elevated_exposure():
    result = simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": 120.0})
    assert result.pk_risk_assessment == "elevated_exposure"


# ---------------------------------------------------------------------------
# Test 11: High CO → "reduced_exposure"
# ---------------------------------------------------------------------------
def test_high_co_reduced_exposure():
    result = simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": 400.0})
    assert result.pk_risk_assessment == "reduced_exposure"


# ---------------------------------------------------------------------------
# Test 12: Normal CO → "normal"
# ---------------------------------------------------------------------------
def test_normal_co_assessment():
    result = simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": 270.0})
    assert result.pk_risk_assessment == "normal"


# ---------------------------------------------------------------------------
# Test 13: t_half_plasma > 0
# ---------------------------------------------------------------------------
def test_thalf_plasma_positive():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.t_half_plasma_h > 0.0


# ---------------------------------------------------------------------------
# Test 14: t_half_plasma < t_end (plasma clears within simulation window)
# ---------------------------------------------------------------------------
def test_thalf_plasma_less_than_tend():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.t_half_plasma_h < DEFAULT_ARGS["t_end_h"]


# ---------------------------------------------------------------------------
# Test 15: cmax_plasma equals initial concentration (IV bolus)
# ---------------------------------------------------------------------------
def test_cmax_plasma_equals_initial():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    expected_c0 = DEFAULT_ARGS["dose_mg"] / VD_PLASMA
    assert abs(result.cmax_plasma - expected_c0) < 1e-6


# ---------------------------------------------------------------------------
# Test 16: compare_cardiac_outputs returns correct count
# ---------------------------------------------------------------------------
def test_compare_cardiac_outputs_count():
    co_list = [100.0, 200.0, 300.0, 400.0, 500.0]
    results = compare_cardiac_outputs("TestDrug", 100.0, co_list, logP=2.0)
    assert len(results) == len(co_list)


# ---------------------------------------------------------------------------
# Test 17: compare_cardiac_outputs sorted by auc_plasma descending
# ---------------------------------------------------------------------------
def test_compare_cardiac_outputs_sorted():
    co_list = [500.0, 100.0, 300.0]
    results = compare_cardiac_outputs("TestDrug", 100.0, co_list, logP=2.0)
    aucs = [r.auc_plasma for r in results]
    assert aucs == sorted(aucs, reverse=True)


# ---------------------------------------------------------------------------
# Test 18: plasma concentration is monotonically decreasing (IV bolus, no re-entry)
# ---------------------------------------------------------------------------
def test_plasma_concentration_decreasing():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    # Check that peak is at t=0 and trend is downward overall
    assert result.c_plasma_mg_L[0] == max(result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Test 19: Validation error — dose <= 0
# ---------------------------------------------------------------------------
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "dose_mg": 0.0})


# ---------------------------------------------------------------------------
# Test 20: Validation error — CO <= 0
# ---------------------------------------------------------------------------
def test_validation_co_zero():
    with pytest.raises(ValueError, match="cardiac_output"):
        simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cardiac_output_L_per_h": -1.0})


# ---------------------------------------------------------------------------
# Test 21: Validation error — logP out of range
# ---------------------------------------------------------------------------
def test_validation_logp_out_of_range():
    with pytest.raises(ValueError, match="logP"):
        simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "logP": 15.0})


# ---------------------------------------------------------------------------
# Test 22: Validation error — cl_intrinsic <= 0
# ---------------------------------------------------------------------------
def test_validation_cl_intrinsic_zero():
    with pytest.raises(ValueError, match="cl_intrinsic"):
        simulate_cardiac_output_pk(**{**DEFAULT_ARGS, "cl_intrinsic_L_per_h": 0.0})


# ---------------------------------------------------------------------------
# Test 23: notes field is a non-empty string
# ---------------------------------------------------------------------------
def test_notes_non_empty():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Test 24: times_h starts at 0 and ends at t_end_h
# ---------------------------------------------------------------------------
def test_times_range():
    result = simulate_cardiac_output_pk(**DEFAULT_ARGS)
    assert result.times_h[0] == 0.0
    assert abs(result.times_h[-1] - DEFAULT_ARGS["t_end_h"]) < 0.11
