"""Tests for Phase 240: combination_index_calculator."""

import pytest

from omega_pbpk.clinical.combination_index_calculator import (
    CombinationIndexResult,
    _d_alone,
    calculate_combination_index,
    scan_fa_range,
)

# ---------------------------------------------------------------------------
# Helpers / shared parameters
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug1_name="DrugA",
    drug2_name="DrugB",
    drug1_ic50=10.0,
    drug2_ic50=20.0,
    drug1_m=1.0,
    drug2_m=1.0,
    fa=0.5,
    effect_type="inhibition",
)


def make_result(**overrides):
    kw = dict(BASE_KWARGS)
    kw.update(overrides)
    # Provide doses if not overridden
    kw.setdefault("drug1_dose", 5.0)
    kw.setdefault("drug2_dose", 10.0)
    return calculate_combination_index(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, CombinationIndexResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.combination_index = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additive scenario: CI = 1.0
# ---------------------------------------------------------------------------


def test_ci_additive_exact():
    """Using exactly half of each D_alone gives CI = 1.0."""
    fa = 0.5
    ic50_1 = 10.0
    ic50_2 = 20.0
    m1 = 1.0
    m2 = 1.0

    d1_alone = _d_alone(ic50_1, m1, fa)
    d2_alone = _d_alone(ic50_2, m2, fa)

    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=d1_alone / 2.0,
        drug2_dose=d2_alone / 2.0,
        drug1_ic50=ic50_1,
        drug2_ic50=ic50_2,
        drug1_m=m1,
        drug2_m=m2,
        fa=fa,
    )
    assert abs(result.combination_index - 1.0) < 1e-10


def test_synergy_class_additive():
    fa = 0.5
    d1_alone = _d_alone(10.0, 1.0, fa)
    d2_alone = _d_alone(20.0, 1.0, fa)
    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=d1_alone / 2.0,
        drug2_dose=d2_alone / 2.0,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
        fa=fa,
    )
    assert result.synergy_class == "additive"


# ---------------------------------------------------------------------------
# Synergistic: CI < 1
# ---------------------------------------------------------------------------


def test_ci_synergistic():
    """Doses much lower than D_alone → CI < 1."""
    fa = 0.5
    d1_alone = _d_alone(10.0, 1.0, fa)
    d2_alone = _d_alone(20.0, 1.0, fa)
    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=d1_alone * 0.1,
        drug2_dose=d2_alone * 0.1,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
        fa=fa,
    )
    assert result.combination_index < 1.0
    assert result.synergy_class == "synergistic"


# ---------------------------------------------------------------------------
# Antagonistic: CI > 1
# ---------------------------------------------------------------------------


def test_ci_antagonistic():
    """Doses much higher than needed → CI > 1."""
    fa = 0.5
    d1_alone = _d_alone(10.0, 1.0, fa)
    d2_alone = _d_alone(20.0, 1.0, fa)
    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=d1_alone * 2.0,
        drug2_dose=d2_alone * 2.0,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
        fa=fa,
    )
    assert result.combination_index > 1.0
    assert result.synergy_class == "antagonistic"


# ---------------------------------------------------------------------------
# synergy_class valid values
# ---------------------------------------------------------------------------


def test_synergy_class_valid_values():
    result = make_result()
    assert result.synergy_class in ("synergistic", "additive", "antagonistic")


# ---------------------------------------------------------------------------
# DRI correctness
# ---------------------------------------------------------------------------


def test_dri1_formula():
    fa = 0.5
    ic50_1 = 10.0
    m1 = 1.0
    d1 = 3.0
    d1_alone = _d_alone(ic50_1, m1, fa)

    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=d1,
        drug2_dose=10.0,
        drug1_ic50=ic50_1,
        drug2_ic50=20.0,
        drug1_m=m1,
        fa=fa,
    )
    expected_dri1 = d1_alone / d1
    assert abs(result.dose_reduction_index_1 - expected_dri1) < 1e-10


def test_dri2_formula():
    fa = 0.5
    ic50_2 = 20.0
    m2 = 1.0
    d2 = 7.0
    d2_alone = _d_alone(ic50_2, m2, fa)

    result = calculate_combination_index(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=5.0,
        drug2_dose=d2,
        drug1_ic50=10.0,
        drug2_ic50=ic50_2,
        drug2_m=m2,
        fa=fa,
    )
    expected_dri2 = d2_alone / d2
    assert abs(result.dose_reduction_index_2 - expected_dri2) < 1e-10


# ---------------------------------------------------------------------------
# fa
# ---------------------------------------------------------------------------


def test_fa_within_range():
    result = make_result(fa=0.75)
    assert 0.0 < result.fa < 1.0


def test_fa_stored_correctly():
    result = make_result(fa=0.3)
    assert abs(result.fa - 0.3) < 1e-10


# ---------------------------------------------------------------------------
# Effect types
# ---------------------------------------------------------------------------


def test_effect_type_cytotoxicity():
    result = make_result(effect_type="cytotoxicity")
    assert result.effect_type == "cytotoxicity"


def test_effect_type_antimicrobial():
    result = make_result(effect_type="antimicrobial")
    assert result.effect_type == "antimicrobial"


# ---------------------------------------------------------------------------
# scan_fa_range
# ---------------------------------------------------------------------------


def test_scan_fa_range_default_length():
    results = scan_fa_range(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=5.0,
        drug2_dose=10.0,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
    )
    assert len(results) == 4


def test_scan_fa_range_sorted_ascending():
    results = scan_fa_range(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=5.0,
        drug2_dose=10.0,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
    )
    fa_values = [r.fa for r in results]
    assert fa_values == sorted(fa_values)


def test_scan_fa_range_custom_fa_values():
    custom = [0.1, 0.3, 0.6, 0.8, 0.95]
    results = scan_fa_range(
        drug1_name="A",
        drug2_name="B",
        drug1_dose=5.0,
        drug2_dose=10.0,
        drug1_ic50=10.0,
        drug2_ic50=20.0,
        fa_values=custom,
    )
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_drug1_dose_zero():
    with pytest.raises(ValueError, match="drug1_dose"):
        make_result(drug1_dose=0.0)


def test_validation_drug2_dose_negative():
    with pytest.raises(ValueError, match="drug2_dose"):
        make_result(drug2_dose=-1.0)


def test_validation_ic50_zero():
    with pytest.raises(ValueError, match="drug1_ic50"):
        make_result(drug1_ic50=0.0)


def test_validation_ic50_negative():
    with pytest.raises(ValueError, match="drug2_ic50"):
        make_result(drug2_ic50=-5.0)


def test_validation_fa_zero():
    with pytest.raises(ValueError, match="fa"):
        make_result(fa=0.0)


def test_validation_fa_one():
    with pytest.raises(ValueError, match="fa"):
        make_result(fa=1.0)


def test_validation_fa_above_one():
    with pytest.raises(ValueError, match="fa"):
        make_result(fa=1.5)


def test_validation_invalid_effect_type():
    with pytest.raises(ValueError, match="effect_type"):
        make_result(effect_type="unknown_effect")
