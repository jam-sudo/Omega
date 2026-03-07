"""Tests for Phase 363 — Drug Solubility pH Profile."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.solubility_ph_profile_p363 import (
    SolubilityPHProfileResult,
    predict_solubility_ph_profile,
)

# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid")
    assert isinstance(result, SolubilityPHProfileResult)


def test_ph_values_length_default():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid")
    assert len(result.ph_values) == 22


def test_solubility_length_matches_ph():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid")
    assert len(result.solubility_mg_mL) == len(result.ph_values)


def test_custom_n_ph_points():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid", n_ph_points=10)
    assert len(result.ph_values) == 10
    assert len(result.solubility_mg_mL) == 10


def test_ph_range_starts_at_1():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid")
    assert math.isclose(result.ph_values[0], 1.0, abs_tol=1e-9)


def test_ph_range_ends_at_12():
    result = predict_solubility_ph_profile("TestDrug", 0.5, 4.0, "acid")
    assert math.isclose(result.ph_values[-1], 12.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# All solubility values >= intrinsic solubility
# ---------------------------------------------------------------------------


def test_all_solubility_gte_intrinsic_acid():
    s0 = 0.3
    result = predict_solubility_ph_profile("Acid", s0, 5.0, "acid")
    for s in result.solubility_mg_mL:
        assert s >= s0 - 1e-9, f"Solubility {s} below intrinsic {s0}"


def test_all_solubility_gte_intrinsic_base():
    s0 = 0.2
    result = predict_solubility_ph_profile("Base", s0, 8.0, "base")
    for s in result.solubility_mg_mL:
        assert s >= s0 - 1e-9, f"Solubility {s} below intrinsic {s0}"


def test_all_solubility_gte_intrinsic_neutral():
    s0 = 0.1
    result = predict_solubility_ph_profile("Neutral", s0, 7.0, "neutral")
    for s in result.solubility_mg_mL:
        assert s >= s0 - 1e-9


# ---------------------------------------------------------------------------
# Acid drug: high pH → higher solubility
# ---------------------------------------------------------------------------


def test_acid_higher_solubility_at_high_ph():
    result = predict_solubility_ph_profile("AcidDrug", 0.1, 5.0, "acid")
    # Solubility at pH 10 > solubility at pH 2
    ph_vals = result.ph_values
    sols = result.solubility_mg_mL
    idx_low = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 2.0))
    idx_high = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 10.0))
    assert sols[idx_high] > sols[idx_low]


def test_acid_ph_max_solubility_at_high_ph():
    # For an acid with pKa=5, solubility rises with pH.
    # The cap (1000*S0) is hit around pH 8; ph_max_solubility >= 7.0 is sufficient.
    result = predict_solubility_ph_profile("AcidDrug", 0.1, 5.0, "acid")
    assert result.ph_max_solubility >= 7.0


# ---------------------------------------------------------------------------
# Base drug: low pH → higher solubility
# ---------------------------------------------------------------------------


def test_base_higher_solubility_at_low_ph():
    result = predict_solubility_ph_profile("BaseDrug", 0.1, 8.0, "base")
    ph_vals = result.ph_values
    sols = result.solubility_mg_mL
    idx_low = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 2.0))
    idx_high = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 10.0))
    assert sols[idx_low] > sols[idx_high]


def test_base_ph_max_solubility_at_low_ph():
    result = predict_solubility_ph_profile("BaseDrug", 0.1, 8.0, "base")
    assert result.ph_max_solubility <= 3.0


def test_base_sol_gastric_greater_than_intestinal():
    # For a base, gastric pH (1.2) should have higher solubility than intestinal (6.8)
    result = predict_solubility_ph_profile("BaseDrug", 0.1, 9.0, "base")
    assert result.sol_gastric_ph12_mg_mL > result.sol_intestinal_ph68_mg_mL


# ---------------------------------------------------------------------------
# Neutral drug: constant solubility
# ---------------------------------------------------------------------------


def test_neutral_constant_solubility():
    s0 = 0.5
    result = predict_solubility_ph_profile("NeutralDrug", s0, 7.0, "neutral")
    for s in result.solubility_mg_mL:
        assert math.isclose(s, s0, rel_tol=1e-9)


def test_neutral_max_equals_min():
    result = predict_solubility_ph_profile("NeutralDrug", 0.5, 7.0, "neutral")
    assert math.isclose(result.max_solubility_mg_mL, result.min_solubility_mg_mL)


# ---------------------------------------------------------------------------
# max_sol >= min_sol
# ---------------------------------------------------------------------------


def test_max_sol_gte_min_sol_acid():
    result = predict_solubility_ph_profile("Acid", 0.1, 5.0, "acid")
    assert result.max_solubility_mg_mL >= result.min_solubility_mg_mL


def test_max_sol_gte_min_sol_base():
    result = predict_solubility_ph_profile("Base", 0.1, 8.0, "base")
    assert result.max_solubility_mg_mL >= result.min_solubility_mg_mL


# ---------------------------------------------------------------------------
# BCS solubility class
# ---------------------------------------------------------------------------


def test_bcs_class_good():
    # High intrinsic solubility -> good BCS class
    result = predict_solubility_ph_profile("GoodDrug", 5.0, 4.0, "neutral")
    assert result.bcs_solubility_class == "good"


def test_bcs_class_poor():
    # Very low solubility -> poor
    result = predict_solubility_ph_profile("PoorDrug", 0.01, 7.0, "neutral")
    assert result.bcs_solubility_class == "poor"


def test_bcs_class_borderline():
    result = predict_solubility_ph_profile("BorderDrug", 0.3, 7.0, "neutral")
    assert result.bcs_solubility_class == "borderline"


def test_bcs_class_valid_values():
    result = predict_solubility_ph_profile("Drug", 0.5, 5.0, "acid")
    assert result.bcs_solubility_class in {"good", "borderline", "poor"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_invalid_intrinsic_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_solubility_ph_profile("Drug", 0.0, 5.0, "acid")


def test_error_invalid_intrinsic_solubility_negative():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_solubility_ph_profile("Drug", -1.0, 5.0, "acid")


def test_error_pka_below_range():
    with pytest.raises(ValueError, match="pKa"):
        predict_solubility_ph_profile("Drug", 1.0, -0.5, "acid")


def test_error_pka_above_range():
    with pytest.raises(ValueError, match="pKa"):
        predict_solubility_ph_profile("Drug", 1.0, 14.1, "acid")


def test_error_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        predict_solubility_ph_profile("Drug", 1.0, 5.0, "amphoteric")


# ---------------------------------------------------------------------------
# Additional checks
# ---------------------------------------------------------------------------


def test_solubility_cap_applied():
    # With extreme pKa difference the cap should kick in
    result = predict_solubility_ph_profile("AcidDrug", 0.001, 1.0, "acid")
    cap = 1000.0 * 0.001
    for s in result.solubility_mg_mL:
        assert s <= cap + 1e-9


def test_drug_name_preserved():
    result = predict_solubility_ph_profile("Ibuprofen", 0.021, 4.4, "acid")
    assert result.drug_name == "Ibuprofen"


def test_pka_preserved():
    result = predict_solubility_ph_profile("Drug", 0.5, 6.5, "acid")
    assert math.isclose(result.pKa, 6.5)
