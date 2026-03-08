"""
Tests for Phase 606 — Paediatric Weight-Based Dosing
"""

import pytest

from omega_pbpk.clinical.paediatric_weight_dosing_p606 import (
    PaediatricWeightDosingResult,
    calculate_paediatric_dose,
    dose_across_ages,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
DRUG = "Amoxicillin"
ADULT_DOSE = 500.0  # mg
ADULT_WEIGHT = 70.0  # kg
T_HALF_ADULT = 1.5  # hours


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert isinstance(result, PaediatricWeightDosingResult)


def test_result_is_frozen():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    with pytest.raises((AttributeError, TypeError)):
        result.recommended_dose_mg = 999.0  # type: ignore[misc]


def test_fields_populated():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.drug_name == DRUG
    assert result.age_years == pytest.approx(5.0)
    assert result.weight_kg == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Age group classification
# ---------------------------------------------------------------------------


def test_neonate_classification():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.03, 2.0, T_HALF_ADULT)
    assert result.age_group == "neonate"


def test_infant_classification_lower():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.5, 8.0, T_HALF_ADULT)
    assert result.age_group == "infant"


def test_child_classification():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.age_group == "child"


def test_adolescent_classification():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 15.0, 55.0, T_HALF_ADULT)
    assert result.age_group == "adolescent"


def test_boundary_infant_at_0_08():
    # age exactly 0.08 → infant
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.08, 5.0, T_HALF_ADULT)
    assert result.age_group == "infant"


def test_boundary_child_at_2():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 2.0, 12.0, T_HALF_ADULT)
    assert result.age_group == "child"


def test_boundary_adolescent_at_12():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 12.0, 40.0, T_HALF_ADULT)
    assert result.age_group == "adolescent"


# ---------------------------------------------------------------------------
# CL / Vd scaling factors
# ---------------------------------------------------------------------------


def test_neonate_cl_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.03, 2.0, T_HALF_ADULT)
    assert result.cl_scaling_factor == pytest.approx(0.25)


def test_infant_cl_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.5, 8.0, T_HALF_ADULT)
    assert result.cl_scaling_factor == pytest.approx(1.5)


def test_child_cl_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.cl_scaling_factor == pytest.approx(1.2)


def test_adolescent_cl_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 15.0, 55.0, T_HALF_ADULT)
    assert result.cl_scaling_factor == pytest.approx(1.0)


def test_neonate_vd_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.03, 2.0, T_HALF_ADULT)
    assert result.vd_scaling_factor == pytest.approx(1.4)


def test_infant_vd_factor():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.5, 8.0, T_HALF_ADULT)
    assert result.vd_scaling_factor == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Dose calculation
# ---------------------------------------------------------------------------


def test_mg_per_kg_calculation_child():
    # adult_mg_per_kg = 500/70; child cl_factor=1.2
    adult_mk = ADULT_DOSE / ADULT_WEIGHT
    expected = adult_mk * 1.2
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.mg_per_kg == pytest.approx(expected)


def test_total_dose_equals_mg_per_kg_times_weight():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.total_dose_mg == pytest.approx(result.mg_per_kg * 18.0)


def test_max_dose_equals_adult_dose():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)
    assert result.max_dose_mg == pytest.approx(ADULT_DOSE)


def test_recommended_dose_capped_at_adult():
    # Heavy child: infant CL factor 1.5, weight 60kg
    # mg_per_kg = (500/70)*1.5 = 10.71, total = 10.71 * 60 = 642.9 > 500
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 1.0, 60.0, T_HALF_ADULT)
    assert result.recommended_dose_mg == pytest.approx(ADULT_DOSE)


def test_recommended_dose_not_capped_for_small_child():
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 10.0, T_HALF_ADULT)
    assert result.recommended_dose_mg < ADULT_DOSE
    assert result.recommended_dose_mg == pytest.approx(result.total_dose_mg)


def test_mg_per_kg_max_applied():
    max_mk = 2.0  # mg/kg
    result = calculate_paediatric_dose(
        DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT, mg_per_kg_max=max_mk
    )
    assert result.mg_per_kg == pytest.approx(max_mk)


def test_t_half_paediatric_neonate():
    # t_half * vd_factor / cl_factor = T_HALF_ADULT * 1.4 / 0.25
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 0.03, 2.0, T_HALF_ADULT)
    expected = T_HALF_ADULT * 1.4 / 0.25
    assert result.t_half_paediatric_h == pytest.approx(expected)


def test_dosing_interval_minimum_4h():
    # Adolescent with very short t_half: interval = max(4, t_half*1.5) = 4
    result = calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 15.0, 55.0, 1.0)
    assert result.dosing_interval_h == pytest.approx(max(4.0, 1.0 * 1.5))


# ---------------------------------------------------------------------------
# dose_across_ages
# ---------------------------------------------------------------------------


def test_dose_across_ages_returns_list():
    results = dose_across_ages(DRUG, ADULT_DOSE, ADULT_WEIGHT, T_HALF_ADULT)
    assert isinstance(results, list)


def test_dose_across_ages_four_entries():
    results = dose_across_ages(DRUG, ADULT_DOSE, ADULT_WEIGHT, T_HALF_ADULT)
    assert len(results) == 4


def test_dose_across_ages_age_groups():
    results = dose_across_ages(DRUG, ADULT_DOSE, ADULT_WEIGHT, T_HALF_ADULT)
    expected_groups = ["neonate", "infant", "child", "adolescent"]
    assert [r.age_group for r in results] == expected_groups


def test_dose_across_ages_correct_types():
    results = dose_across_ages(DRUG, ADULT_DOSE, ADULT_WEIGHT, T_HALF_ADULT)
    assert all(isinstance(r, PaediatricWeightDosingResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_age_raises():
    with pytest.raises(ValueError, match="age_years"):
        calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, -1.0, 10.0, T_HALF_ADULT)


def test_zero_weight_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 0.0, T_HALF_ADULT)


def test_zero_adult_dose_raises():
    with pytest.raises(ValueError, match="adult_dose_mg"):
        calculate_paediatric_dose(DRUG, 0.0, ADULT_WEIGHT, 5.0, 18.0, T_HALF_ADULT)


def test_zero_adult_weight_raises():
    with pytest.raises(ValueError, match="adult_weight_kg"):
        calculate_paediatric_dose(DRUG, ADULT_DOSE, 0.0, 5.0, 18.0, T_HALF_ADULT)


def test_zero_t_half_raises():
    with pytest.raises(ValueError, match="t_half_adult_h"):
        calculate_paediatric_dose(DRUG, ADULT_DOSE, ADULT_WEIGHT, 5.0, 18.0, 0.0)
