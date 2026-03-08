"""Tests for Phase 873: Pediatric Dose Scaling."""

import pytest

from omega_pbpk.clinical.pediatric_dose_scaling import (
    PediatricDoseScalingResult,
    dose_across_ages,
    scale_pediatric_dose,
)

# --- Basic return type and field presence ---


def test_returns_correct_type():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert isinstance(result, PediatricDoseScalingResult)


def test_fields_preserved():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.drug_name == "DrugA"
    assert result.child_bw_kg == 10.0
    assert result.child_pma_weeks == 60.0


def test_cyp_enzyme_stored():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0, cyp_enzyme="CYP2D6")
    assert result.cyp_enzyme == "CYP2D6"


# --- Maturation fraction ---


def test_maturation_fraction_in_range():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 5.0, 40.0)
    assert 0.0 < result.maturation_fraction < 1.0


def test_maturation_fraction_neonate_lt_adolescent():
    neonate = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 2.0, 35.0)
    adolescent = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 50.0, 700.0)
    assert neonate.maturation_fraction < adolescent.maturation_fraction


def test_maturation_fraction_increases_with_pma():
    r1 = scale_pediatric_dose("D", 5.0, 30.0, 100.0, 5.0, 30.0)
    r2 = scale_pediatric_dose("D", 5.0, 30.0, 100.0, 5.0, 60.0)
    r3 = scale_pediatric_dose("D", 5.0, 30.0, 100.0, 5.0, 200.0)
    assert r1.maturation_fraction < r2.maturation_fraction < r3.maturation_fraction


# --- CL values ---


def test_cl_mature_le_cl_allometric_for_young_children():
    """For children well below adult PMA, mature CL should be < allometric CL."""
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    # cl_mature = cl_allometric * (MF_child / MF_adult)
    # MF_child < MF_adult for young children, so cl_mature < cl_allometric
    assert result.cl_mature_L_per_h <= result.cl_allometric_L_per_h


def test_cl_allometric_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.cl_allometric_L_per_h > 0.0


def test_cl_mature_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.cl_mature_L_per_h > 0.0


def test_cl_ratio_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.cl_ratio_ped_to_adult > 0.0


def test_cl_ratio_lt_1_for_neonate():
    """Neonates should have lower CL ratio than 1 (underdeveloped enzymes)."""
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 3.0, 38.0, cyp_enzyme="CYP3A4")
    assert result.cl_ratio_ped_to_adult < 1.0


# --- Dose values ---


def test_recommended_dose_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.recommended_dose_mg > 0.0


def test_dose_per_kg_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.dose_per_kg_mg_per_kg > 0.0


def test_dose_per_kg_consistent():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    expected = result.recommended_dose_mg / result.child_bw_kg
    assert abs(result.dose_per_kg_mg_per_kg - expected) < 1e-6


def test_vd_scaled_positive():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 10.0, 60.0)
    assert result.vd_scaled_L > 0.0


# --- Age group assignment ---


def test_age_group_neonate():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 1.5, 35.0)
    assert result.age_group == "neonate"


def test_age_group_infant():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 5.0, 80.0)
    assert result.age_group == "infant"


def test_age_group_child():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 20.0, 400.0)
    assert result.age_group == "child"


def test_age_group_adolescent():
    result = scale_pediatric_dose("DrugA", 10.0, 50.0, 100.0, 50.0, 700.0)
    assert result.age_group == "adolescent"


# --- dose_across_ages ---


def test_dose_across_ages_sorted_by_pma():
    pairs = [(20.0, 300.0), (1.5, 35.0), (50.0, 700.0), (5.0, 80.0)]
    results = dose_across_ages("DrugA", 10.0, 100.0, pairs)
    pmas = [r.child_pma_weeks for r in results]
    assert pmas == sorted(pmas)


def test_dose_across_ages_returns_list():
    pairs = [(10.0, 60.0), (5.0, 40.0)]
    results = dose_across_ages("DrugA", 10.0, 100.0, pairs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_dose_across_ages_all_results_are_correct_type():
    pairs = [(10.0, 60.0), (30.0, 400.0)]
    results = dose_across_ages("DrugB", 8.0, 200.0, pairs)
    for r in results:
        assert isinstance(r, PediatricDoseScalingResult)


# --- Validation errors ---


def test_validation_negative_child_bw():
    with pytest.raises(ValueError, match="child_bw_kg must be > 0"):
        scale_pediatric_dose("D", 10.0, 50.0, 100.0, -1.0, 60.0)


def test_validation_zero_child_pma():
    with pytest.raises(ValueError, match="child_pma_weeks must be > 0"):
        scale_pediatric_dose("D", 10.0, 50.0, 100.0, 10.0, 0.0)


def test_validation_negative_adult_cl():
    with pytest.raises(ValueError, match="adult_cl_L_per_h must be > 0"):
        scale_pediatric_dose("D", -5.0, 50.0, 100.0, 10.0, 60.0)


def test_validation_zero_adult_dose():
    with pytest.raises(ValueError, match="adult_dose_mg must be > 0"):
        scale_pediatric_dose("D", 10.0, 50.0, 0.0, 10.0, 60.0)
