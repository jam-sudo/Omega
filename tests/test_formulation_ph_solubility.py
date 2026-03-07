"""Tests for Phase 1111 — pH-dependent solubility prediction."""

import math
import pytest

from omega_pbpk.prediction.formulation_ph_solubility import (
    PHSolubilityResult,
    find_max_solubility_ph,
    predict_ph_solubility,
)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------

def test_return_type():
    result = predict_ph_solubility("DrugA", s0_mg_mL=0.1, pka=5.0, ionization_type="acid")
    assert isinstance(result, PHSolubilityResult)


def test_field_drug_name():
    result = predict_ph_solubility("Ibuprofen", s0_mg_mL=0.05, pka=4.4, ionization_type="acid")
    assert result.drug_name == "Ibuprofen"


def test_field_s0():
    result = predict_ph_solubility("X", s0_mg_mL=2.5, pka=7.0, ionization_type="base")
    assert result.s0_mg_mL == pytest.approx(2.5)


def test_field_pka():
    result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=6.5, ionization_type="acid")
    assert result.pka == pytest.approx(6.5)


def test_field_ionization_type_preserved():
    for itype in ("acid", "base", "neutral"):
        result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=7.0, ionization_type=itype)
        assert result.ionization_type == itype


def test_ph_values_length():
    result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid", n_points=10)
    assert len(result.ph_values) == 10
    assert len(result.solubility_mg_mL) == 10


def test_ph_values_endpoints():
    result = predict_ph_solubility(
        "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid",
        ph_range_start=2.0, ph_range_end=8.0, n_points=7
    )
    assert result.ph_values[0] == pytest.approx(2.0)
    assert result.ph_values[-1] == pytest.approx(8.0)


def test_ph_values_tuple_type():
    result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid")
    assert isinstance(result.ph_values, tuple)
    assert isinstance(result.solubility_mg_mL, tuple)


def test_notes_not_empty():
    result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid")
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Acid ionization
# ---------------------------------------------------------------------------

def test_acid_solubility_increases_with_ph():
    result = predict_ph_subility_helper("acid", pka=5.0)
    sol = list(result.solubility_mg_mL)
    # solubility at pH 9 should be >> pH 1
    assert sol[-1] > sol[0]


def predict_ph_subility_helper(ionization_type, pka=5.0, s0=0.1):
    return predict_ph_solubility(
        "Test", s0_mg_mL=s0, pka=pka, ionization_type=ionization_type,
        ph_range_start=1.0, ph_range_end=9.0, n_points=9
    )


def test_acid_max_at_high_ph():
    result = predict_ph_subility_helper("acid", pka=5.0)
    assert result.ph_at_max_solubility == pytest.approx(result.ph_values[-1])


def test_acid_formula_value():
    # At pH=pKa for acid: S = S0 * (1 + 10^0) = 2*S0
    # Use pKa=5.0 and range [3,5] with n_points=3 → points [3,4,5]
    result = predict_ph_solubility(
        "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid",
        ph_range_start=3.0, ph_range_end=5.0, n_points=3
    )
    # pH=5.0 is last point: S = 1.0 * (1 + 10^(5-5)) = 2.0
    assert result.solubility_mg_mL[-1] == pytest.approx(2.0, rel=1e-6)
    # pH=3.0 (first point): S = 1.0*(1+10^(3-5)) = 1+0.01 = 1.01
    assert result.solubility_mg_mL[0] == pytest.approx(1.01, rel=1e-6)


def test_acid_max_gt_min():
    result = predict_ph_subility_helper("acid")
    assert result.max_solubility_mg_mL > result.min_solubility_mg_mL


def test_acid_ratio_gt_1():
    result = predict_ph_subility_helper("acid")
    assert result.solubility_ratio_max_min > 1.0


# ---------------------------------------------------------------------------
# Base ionization
# ---------------------------------------------------------------------------

def test_base_solubility_increases_low_ph():
    result = predict_ph_subility_helper("base", pka=7.0)
    sol = list(result.solubility_mg_mL)
    # pH 1 should have higher solubility than pH 9 for a base with pKa=7
    assert sol[0] > sol[-1]


def test_base_max_at_low_ph():
    result = predict_ph_subility_helper("base", pka=7.0)
    assert result.ph_at_max_solubility == pytest.approx(result.ph_values[0])


def test_base_formula_value():
    # At pH=pKa for base: S = S0 * (1 + 10^0) = 2*S0
    result = predict_ph_solubility(
        "X", s0_mg_mL=1.0, pka=5.0, ionization_type="base",
        ph_range_start=4.0, ph_range_end=6.0, n_points=3
    )
    # pH=5.0 is last point: S = 1.0*(1+10^(5-5))=2.0; pH=4: S=1*(1+10^1)=11
    assert result.solubility_mg_mL[0] == pytest.approx(11.0, rel=1e-6)


def test_base_ratio_gt_1():
    result = predict_ph_subility_helper("base", pka=7.0)
    assert result.solubility_ratio_max_min > 1.0


# ---------------------------------------------------------------------------
# Neutral ionization
# ---------------------------------------------------------------------------

def test_neutral_constant_solubility():
    result = predict_ph_solubility(
        "NeutralDrug", s0_mg_mL=0.5, pka=7.0, ionization_type="neutral"
    )
    sol = list(result.solubility_mg_mL)
    assert all(s == pytest.approx(0.5) for s in sol)


def test_neutral_ratio_equals_1():
    result = predict_ph_solubility(
        "NeutralDrug", s0_mg_mL=0.5, pka=7.0, ionization_type="neutral"
    )
    assert result.solubility_ratio_max_min == pytest.approx(1.0)


def test_neutral_max_equals_s0():
    result = predict_ph_solubility(
        "NeutralDrug", s0_mg_mL=2.0, pka=7.0, ionization_type="neutral"
    )
    assert result.max_solubility_mg_mL == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# find_max_solubility_ph
# ---------------------------------------------------------------------------

def test_find_max_solubility_ph_acid():
    result = predict_ph_solubility(
        "Acid", s0_mg_mL=0.1, pka=4.0, ionization_type="acid",
        ph_range_start=1.0, ph_range_end=10.0, n_points=19
    )
    ph_max = find_max_solubility_ph(result)
    assert ph_max == pytest.approx(10.0)


def test_find_max_solubility_ph_base():
    result = predict_ph_solubility(
        "Base", s0_mg_mL=0.1, pka=8.0, ionization_type="base",
        ph_range_start=1.0, ph_range_end=10.0, n_points=19
    )
    ph_max = find_max_solubility_ph(result)
    assert ph_max == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_s0_zero():
    with pytest.raises(ValueError, match="s0_mg_mL"):
        predict_ph_solubility("X", s0_mg_mL=0.0, pka=5.0, ionization_type="acid")


def test_invalid_s0_negative():
    with pytest.raises(ValueError, match="s0_mg_mL"):
        predict_ph_solubility("X", s0_mg_mL=-1.0, pka=5.0, ionization_type="acid")


def test_invalid_pka_negative():
    with pytest.raises(ValueError, match="pKa"):
        predict_ph_solubility("X", s0_mg_mL=1.0, pka=-1.0, ionization_type="acid")


def test_invalid_pka_above_14():
    with pytest.raises(ValueError, match="pKa"):
        predict_ph_solubility("X", s0_mg_mL=1.0, pka=15.0, ionization_type="acid")


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="zwitterion")


def test_invalid_ph_range_order():
    with pytest.raises(ValueError, match="ph_range_start"):
        predict_ph_solubility(
            "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid",
            ph_range_start=8.0, ph_range_end=2.0
        )


def test_invalid_ph_range_equal():
    with pytest.raises(ValueError, match="ph_range_start"):
        predict_ph_solubility(
            "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid",
            ph_range_start=5.0, ph_range_end=5.0
        )


def test_invalid_n_points_one():
    with pytest.raises(ValueError, match="n_points"):
        predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid", n_points=1)


def test_invalid_ph_out_of_range():
    with pytest.raises(ValueError):
        predict_ph_solubility(
            "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid",
            ph_range_start=-1.0, ph_range_end=10.0
        )


def test_max_min_fields_consistent():
    result = predict_ph_solubility(
        "X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid"
    )
    assert result.max_solubility_mg_mL >= result.min_solubility_mg_mL
    assert result.max_solubility_mg_mL == max(result.solubility_mg_mL)
    assert result.min_solubility_mg_mL == min(result.solubility_mg_mL)


def test_result_is_frozen():
    result = predict_ph_solubility("X", s0_mg_mL=1.0, pka=5.0, ionization_type="acid")
    with pytest.raises(Exception):
        result.drug_name = "Y"  # type: ignore[misc]
