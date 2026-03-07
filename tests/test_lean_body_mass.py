"""Tests for Phase 580 — lean_body_mass."""

import pytest

from omega_pbpk.clinical.lean_body_mass import (
    adjusted_body_weight,
    bmi,
    dose_body_weight_method,
    ideal_body_weight_devine,
    lean_body_mass_boer,
    lean_body_mass_james,
)

# ---------------------------------------------------------------------------
# lean_body_mass_james
# ---------------------------------------------------------------------------


def test_james_male_typical():
    lbm = lean_body_mass_james(180.0, 70.0, "male")
    # Expected ~55 kg range
    assert 45.0 < lbm < 70.0


def test_james_female_typical():
    lbm = lean_body_mass_james(165.0, 60.0, "female")
    assert 35.0 < lbm < 60.0


def test_james_male_greater_than_female_same_stats():
    lbm_m = lean_body_mass_james(175.0, 70.0, "male")
    lbm_f = lean_body_mass_james(175.0, 70.0, "female")
    assert lbm_m > lbm_f


def test_james_clamps_to_minimum():
    # Very short/heavy → formula could go negative
    lbm = lean_body_mass_james(100.0, 200.0, "female")
    assert lbm >= 10.0


def test_james_invalid_sex_raises():
    with pytest.raises(ValueError, match="sex"):
        lean_body_mass_james(170.0, 70.0, "unknown")


def test_james_zero_height_raises():
    with pytest.raises(ValueError, match="height_cm"):
        lean_body_mass_james(0.0, 70.0, "male")


# ---------------------------------------------------------------------------
# lean_body_mass_boer
# ---------------------------------------------------------------------------


def test_boer_male_typical():
    lbm = lean_body_mass_boer(180.0, 80.0, "male")
    assert 50.0 < lbm < 80.0


def test_boer_female_typical():
    lbm = lean_body_mass_boer(165.0, 65.0, "female")
    assert 35.0 < lbm < 65.0


def test_boer_male_greater_than_female_same_stats():
    lbm_m = lean_body_mass_boer(170.0, 70.0, "male")
    lbm_f = lean_body_mass_boer(170.0, 70.0, "female")
    assert lbm_m > lbm_f


def test_boer_clamps_to_minimum():
    lbm = lean_body_mass_boer(50.0, 5.0, "female")
    assert lbm >= 10.0


# ---------------------------------------------------------------------------
# bmi
# ---------------------------------------------------------------------------


def test_bmi_formula():
    result = bmi(70.0, 175.0)
    expected = 70.0 / (1.75**2)
    assert abs(result - expected) < 1e-9


def test_bmi_known_value():
    # 90 kg, 180 cm → 90 / 3.24 ≈ 27.78
    result = bmi(90.0, 180.0)
    assert abs(result - 27.78) < 0.01


# ---------------------------------------------------------------------------
# ideal_body_weight_devine
# ---------------------------------------------------------------------------


def test_ibw_male_typical():
    ibw = ideal_body_weight_devine(180.0, "male")
    # 180 cm → 70.87 in; IBW = 50 + 2.3*(70.87-60)
    assert 70.0 < ibw < 80.0


def test_ibw_female_typical():
    ibw = ideal_body_weight_devine(165.0, "female")
    assert 50.0 < ibw < 70.0


def test_ibw_male_greater_than_female_same_height():
    ibw_m = ideal_body_weight_devine(175.0, "male")
    ibw_f = ideal_body_weight_devine(175.0, "female")
    assert ibw_m > ibw_f


def test_ibw_clamps_to_minimum():
    ibw = ideal_body_weight_devine(100.0, "male")
    assert ibw >= 10.0


# ---------------------------------------------------------------------------
# adjusted_body_weight
# ---------------------------------------------------------------------------


def test_abw_obese_between_ibw_and_actual():
    ibw = 60.0
    actual = 100.0
    abw = adjusted_body_weight(ibw, actual)
    assert ibw < abw < actual


def test_abw_not_obese_returns_actual():
    ibw = 70.0
    actual = 65.0
    abw = adjusted_body_weight(ibw, actual)
    assert abw == actual


def test_abw_equal_weight_returns_actual():
    abw = adjusted_body_weight(70.0, 70.0)
    assert abw == 70.0


def test_abw_custom_correction_factor():
    ibw = 60.0
    actual = 100.0
    abw = adjusted_body_weight(ibw, actual, correction_factor=0.4)
    expected = 60.0 + 0.4 * 40.0  # 76.0
    assert abs(abw - expected) < 1e-9


# ---------------------------------------------------------------------------
# dose_body_weight_method
# ---------------------------------------------------------------------------


def test_dose_actual_method():
    res = dose_body_weight_method(2.0, 80.0, 65.0, 55.0, "actual")
    assert res["method"] == "actual"
    assert abs(res["weight_used_kg"] - 80.0) < 1e-9
    assert abs(res["dose_mg"] - 160.0) < 1e-9


def test_dose_ideal_method():
    res = dose_body_weight_method(2.0, 80.0, 65.0, 55.0, "ideal")
    assert res["method"] == "ideal"
    assert abs(res["weight_used_kg"] - 65.0) < 1e-9
    assert abs(res["dose_mg"] - 130.0) < 1e-9


def test_dose_lean_method():
    res = dose_body_weight_method(2.0, 80.0, 65.0, 55.0, "lean")
    assert abs(res["weight_used_kg"] - 55.0) < 1e-9
    assert abs(res["dose_mg"] - 110.0) < 1e-9


def test_dose_adjusted_method():
    res = dose_body_weight_method(2.0, 100.0, 60.0, 50.0, "adjusted")
    assert res["method"] == "adjusted"
    assert res["weight_used_kg"] > 60.0  # adjusted > IBW


def test_dose_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        dose_body_weight_method(2.0, 80.0, 65.0, 55.0, "unknown")
