"""Tests for Phase 1104: Neonatal/Pediatric Physiological Parameters."""

import pytest

from omega_pbpk.clinical.neonatal_parameters import (
    NeonatalParameters,
    calculate_pediatric_dose,
    compare_age_groups,
    get_neonatal_parameters,
)

# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_return_type_neonate():
    result = get_neonatal_parameters("neonate")
    assert isinstance(result, NeonatalParameters)


def test_return_type_school():
    result = get_neonatal_parameters("school_6_12y")
    assert isinstance(result, NeonatalParameters)


def test_all_fields_present():
    result = get_neonatal_parameters("toddler_1_3y")
    for field in [
        "age_group",
        "body_weight_kg",
        "v_liver_L",
        "v_kidney_L",
        "cardiac_output_L_per_h",
        "q_liver_L_per_h",
        "q_kidney_L_per_h",
        "gfr_mL_per_min",
        "cyp3a4_activity_fraction",
        "ugt1a1_activity_fraction",
        "adult_cl_scaling_factor",
        "allometric_dose_factor",
        "notes",
    ]:
        assert hasattr(result, field)


# ---------------------------------------------------------------------------
# Organ volume trends
# ---------------------------------------------------------------------------


def test_v_liver_increases_with_age_absolutely():
    """Absolute liver volume should be larger for heavier/older children."""
    neo = get_neonatal_parameters("neonate")
    school = get_neonatal_parameters("school_6_12y")
    assert school.v_liver_L > neo.v_liver_L


def test_v_kidney_increases_absolutely_neonate_to_school():
    neo = get_neonatal_parameters("neonate")
    school = get_neonatal_parameters("school_6_12y")
    assert school.v_kidney_L > neo.v_kidney_L


def test_v_liver_positive_all_groups():
    for ag in [
        "neonate",
        "infant_1_6mo",
        "infant_6_12mo",
        "toddler_1_3y",
        "preschool_3_6y",
        "school_6_12y",
    ]:
        assert get_neonatal_parameters(ag).v_liver_L > 0.0


# ---------------------------------------------------------------------------
# GFR trends
# ---------------------------------------------------------------------------


def test_gfr_increases_from_neonate_to_school():
    neo = get_neonatal_parameters("neonate")
    school = get_neonatal_parameters("school_6_12y")
    assert school.gfr_mL_per_min > neo.gfr_mL_per_min


def test_gfr_positive_all_groups():
    for ag in [
        "neonate",
        "infant_1_6mo",
        "infant_6_12mo",
        "toddler_1_3y",
        "preschool_3_6y",
        "school_6_12y",
    ]:
        assert get_neonatal_parameters(ag).gfr_mL_per_min > 0.0


# ---------------------------------------------------------------------------
# CYP3A4 activity
# ---------------------------------------------------------------------------


def test_cyp3a4_low_for_neonate():
    result = get_neonatal_parameters("neonate")
    assert result.cyp3a4_activity_fraction < 1.0


def test_cyp3a4_full_for_school():
    result = get_neonatal_parameters("school_6_12y")
    assert result.cyp3a4_activity_fraction == 1.0


def test_cyp3a4_increases_with_age():
    groups = [
        "neonate",
        "infant_1_6mo",
        "infant_6_12mo",
        "toddler_1_3y",
        "preschool_3_6y",
        "school_6_12y",
    ]
    fractions = [get_neonatal_parameters(g).cyp3a4_activity_fraction for g in groups]
    assert fractions == sorted(fractions)


# ---------------------------------------------------------------------------
# UGT1A1 activity
# ---------------------------------------------------------------------------


def test_ugt1a1_very_low_neonate():
    result = get_neonatal_parameters("neonate")
    assert result.ugt1a1_activity_fraction < 0.05


def test_ugt1a1_full_for_school():
    result = get_neonatal_parameters("school_6_12y")
    assert result.ugt1a1_activity_fraction == 1.0


# ---------------------------------------------------------------------------
# Allometric dose factor
# ---------------------------------------------------------------------------


def test_allometric_dose_factor_less_than_one_for_neonate():
    result = get_neonatal_parameters("neonate")
    # (3.5/70)^0.75 << 1
    assert result.allometric_dose_factor < 1.0


def test_allometric_dose_factor_positive():
    for ag in ["neonate", "infant_1_6mo", "school_6_12y"]:
        assert get_neonatal_parameters(ag).allometric_dose_factor > 0.0


# ---------------------------------------------------------------------------
# Cardiac output
# ---------------------------------------------------------------------------


def test_cardiac_output_positive_all_groups():
    for ag in [
        "neonate",
        "infant_1_6mo",
        "infant_6_12mo",
        "toddler_1_3y",
        "preschool_3_6y",
        "school_6_12y",
    ]:
        assert get_neonatal_parameters(ag).cardiac_output_L_per_h > 0.0


def test_blood_flows_positive():
    result = get_neonatal_parameters("infant_1_6mo")
    assert result.q_liver_L_per_h > 0.0
    assert result.q_kidney_L_per_h > 0.0


# ---------------------------------------------------------------------------
# compare_age_groups
# ---------------------------------------------------------------------------


def test_compare_age_groups_returns_six():
    results = compare_age_groups()
    assert len(results) == 6


def test_compare_age_groups_all_neonatal_parameters():
    for r in compare_age_groups():
        assert isinstance(r, NeonatalParameters)


def test_compare_age_groups_sorted_by_bw_ascending():
    results = compare_age_groups()
    weights = [r.body_weight_kg for r in results]
    assert weights == sorted(weights)


def test_compare_age_groups_lightest_is_neonate():
    results = compare_age_groups()
    assert results[0].age_group == "neonate"


def test_compare_age_groups_heaviest_is_school():
    results = compare_age_groups()
    assert results[-1].age_group == "school_6_12y"


# ---------------------------------------------------------------------------
# calculate_pediatric_dose
# ---------------------------------------------------------------------------


def test_calculate_pediatric_dose_returns_dict():
    result = calculate_pediatric_dose(100.0, "neonate")
    assert isinstance(result, dict)


def test_calculate_pediatric_dose_required_keys():
    result = calculate_pediatric_dose(100.0, "neonate")
    for key in ["pediatric_dose_mg", "scaling_factor", "method", "age_group"]:
        assert key in result


def test_allometric_dose_less_than_adult_for_neonate():
    result = calculate_pediatric_dose(100.0, "neonate", scaling_method="allometric")
    assert result["pediatric_dose_mg"] < 100.0


def test_allometric_scaling_method_label():
    result = calculate_pediatric_dose(100.0, "infant_1_6mo", scaling_method="allometric")
    assert result["method"] == "allometric"


def test_mg_per_kg_scaling_works():
    result = calculate_pediatric_dose(100.0, "toddler_1_3y", scaling_method="mg_per_kg")
    # 13/70 ≈ 0.186
    assert result["pediatric_dose_mg"] < 100.0
    assert result["method"] == "mg_per_kg"


def test_organ_function_scaling_works():
    result = calculate_pediatric_dose(100.0, "neonate", scaling_method="organ_function")
    # CL factor very low for neonate → much smaller dose
    assert result["pediatric_dose_mg"] < 100.0
    assert result["method"] == "organ_function"


def test_organ_function_uses_cl_factor():
    params = get_neonatal_parameters("neonate")
    result = calculate_pediatric_dose(200.0, "neonate", scaling_method="organ_function")
    expected = 200.0 * params.adult_cl_scaling_factor
    assert abs(result["pediatric_dose_mg"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_age_group_raises_value_error():
    with pytest.raises(ValueError, match="age_group"):
        get_neonatal_parameters("adult")


def test_invalid_age_group_raises_value_error_empty():
    with pytest.raises(ValueError):
        get_neonatal_parameters("")


def test_invalid_scaling_method_raises():
    with pytest.raises(ValueError, match="scaling_method"):
        calculate_pediatric_dose(100.0, "neonate", scaling_method="weight_based")
