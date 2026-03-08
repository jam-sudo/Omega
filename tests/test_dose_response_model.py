"""Tests for Phase 519 — Dose-Response Relationship Model."""

import pytest

from omega_pbpk.clinical.dose_response_model import (
    DoseResponseResult,
    _emax_effect,
    _sigmoid_emax_effect,
    _threshold_effect,
    compare_models,
    compute_dose_response,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
EMAX = 100.0
EC50 = 1.0  # mg/L
F = 1.0
VD = 10.0  # L   → Cmax = dose / 10


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    assert isinstance(r, DoseResponseResult)


def test_field_drug_name():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    assert r.drug_name == DRUG


def test_field_model_type_emax():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    assert r.model_type == "emax"


def test_field_model_type_sigmoid():
    r = compute_dose_response(DRUG, "sigmoid_emax", EMAX, EC50, F, VD)
    assert r.model_type == "sigmoid_emax"


def test_field_model_type_threshold():
    r = compute_dose_response(DRUG, "threshold", EMAX, EC50, F, VD)
    assert r.model_type == "threshold"


def test_effect_values_length_matches_doses():
    doses = [5.0, 10.0, 50.0]
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD, doses_mg=doses)
    assert len(r.effect_values) == len(doses)
    assert len(r.cmax_values_mg_L) == len(doses)
    assert len(r.doses_mg) == len(doses)


def test_default_doses_length():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    assert len(r.doses_mg) == 6


# ---------------------------------------------------------------------------
# Emax model correctness
# ---------------------------------------------------------------------------


def test_emax_at_ec50_concentration():
    """E(EC50) = E0 + Emax/2."""
    e = _emax_effect(EC50, 0.0, EMAX, EC50)
    assert abs(e - EMAX / 2) < 1e-9


def test_emax_model_effect_at_ec50_dose():
    """When Cmax == EC50, effect ≈ E0 + Emax/2."""
    # dose = EC50 * Vd / F  → Cmax = EC50
    dose_at_ec50 = EC50 * VD / F
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD, doses_mg=[dose_at_ec50])
    assert abs(r.effect_values[0] - EMAX / 2) < 1e-9


def test_emax_approaches_emax_at_large_concentration():
    """Very high concentration → effect near Emax."""
    e = _emax_effect(1e9, 0.0, EMAX, EC50)
    assert e > 0.9999 * EMAX


def test_emax_at_zero_concentration():
    """Zero concentration → baseline effect."""
    e = _emax_effect(0.0, 5.0, EMAX, EC50)
    assert abs(e - 5.0) < 1e-12


# ---------------------------------------------------------------------------
# Sigmoid-Emax model
# ---------------------------------------------------------------------------


def test_sigmoid_at_ec50():
    """E(EC50) = E0 + Emax/2 regardless of hill coefficient."""
    e = _sigmoid_emax_effect(EC50, 0.0, EMAX, EC50, 3.0)
    assert abs(e - EMAX / 2) < 1e-9


def test_sigmoid_steeper_than_emax_for_n_gt_1():
    """Sigmoid (n>1) rises faster than Emax between EC50/2 and EC50*2."""
    c = EC50 * 2.0
    e_emax = _emax_effect(c, 0.0, EMAX, EC50)
    e_sig = _sigmoid_emax_effect(c, 0.0, EMAX, EC50, 3.0)
    assert e_sig > e_emax


def test_sigmoid_zero_concentration():
    e = _sigmoid_emax_effect(0.0, 10.0, EMAX, EC50, 2.0)
    assert abs(e - 10.0) < 1e-12


# ---------------------------------------------------------------------------
# Threshold model
# ---------------------------------------------------------------------------


def test_threshold_below_threshold():
    """Below threshold, effect == E0."""
    e = _threshold_effect(0.5, 0.0, EMAX, EC50, c_thresh=1.0)
    assert abs(e - 0.0) < 1e-12


def test_threshold_at_threshold():
    """At exactly the threshold, effect == E0."""
    e = _threshold_effect(1.0, 0.0, EMAX, EC50, c_thresh=1.0)
    assert abs(e - 0.0) < 1e-12


def test_threshold_above_threshold():
    """Above threshold, follows Emax with (C - threshold)."""
    # c = 2.0, thresh = 1.0 → delta = 1.0 = EC50 → E = Emax/2
    e = _threshold_effect(2.0, 0.0, EMAX, 1.0, 1.0)
    assert abs(e - EMAX / 2) < 1e-9


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_emax_effect_values_monotonically_increasing():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    evs = r.effect_values
    for i in range(1, len(evs)):
        assert evs[i] >= evs[i - 1], f"Not monotone at index {i}"


def test_sigmoid_effect_values_monotonically_increasing():
    r = compute_dose_response(DRUG, "sigmoid_emax", EMAX, EC50, F, VD, hill_coeff=3.0)
    evs = r.effect_values
    for i in range(1, len(evs)):
        assert evs[i] >= evs[i - 1], f"Not monotone at index {i}"


# ---------------------------------------------------------------------------
# ED50 dose
# ---------------------------------------------------------------------------


def test_ed50_in_dose_range():
    doses = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD, doses_mg=doses)
    assert r.ed50_mg >= doses[0]
    assert r.ed50_mg <= doses[-1]


def test_ed50_positive():
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD)
    assert r.ed50_mg > 0


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


def test_compare_models_returns_list_of_3():
    results = compare_models(DRUG, EMAX, EC50, F, VD)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_models_all_types():
    results = compare_models(DRUG, EMAX, EC50, F, VD)
    types = {r.model_type for r in results}
    assert types == {"emax", "sigmoid_emax", "threshold"}


def test_compare_models_different_effects():
    """The three models should produce at least some difference in effects."""
    results = compare_models(DRUG, EMAX, EC50, F, VD)
    # Compare effects across all doses: flatten all effect values and compare
    # sigmoid_emax with hill=2 will differ from emax at doses != EC50*Vd
    # Check that at least one dose index produces different values across models
    any_different = False
    for idx in range(len(results[0].effect_values)):
        vals = [round(r.effect_values[idx], 6) for r in results]
        if len(set(vals)) > 1:
            any_different = True
            break
    assert any_different, "All 3 models produce identical effect curves"


def test_all_models_are_DoseResponseResult():
    results = compare_models(DRUG, EMAX, EC50, F, VD)
    for r in results:
        assert isinstance(r, DoseResponseResult)


# ---------------------------------------------------------------------------
# Cmax calculation
# ---------------------------------------------------------------------------


def test_cmax_values_correct():
    doses = [10.0, 50.0]
    r = compute_dose_response(DRUG, "emax", EMAX, EC50, F, VD, doses_mg=doses)
    assert abs(r.cmax_values_mg_L[0] - F * 10.0 / VD) < 1e-12
    assert abs(r.cmax_values_mg_L[1] - F * 50.0 / VD) < 1e-12


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        compute_dose_response(DRUG, "bad_model", EMAX, EC50, F, VD)


def test_zero_emax_raises():
    with pytest.raises(ValueError, match="emax_effect"):
        compute_dose_response(DRUG, "emax", 0.0, EC50, F, VD)


def test_nonpositive_ec50_raises():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        compute_dose_response(DRUG, "emax", EMAX, 0.0, F, VD)


def test_invalid_f_oral_zero_raises():
    with pytest.raises(ValueError, match="f_oral"):
        compute_dose_response(DRUG, "emax", EMAX, EC50, 0.0, VD)


def test_invalid_f_oral_gt1_raises():
    with pytest.raises(ValueError, match="f_oral"):
        compute_dose_response(DRUG, "emax", EMAX, EC50, 1.1, VD)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        compute_dose_response(DRUG, "emax", EMAX, EC50, F, 0.0)
