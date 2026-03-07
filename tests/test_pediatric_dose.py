"""Tests for clinical/pediatric_dose.py — Phase 170."""

import math

import pytest

from omega_pbpk.clinical.pediatric_dose import (
    PediatricDoseResult,
    calculate_pediatric_dose,
    dose_across_ages,
)

# ---------------------------------------------------------------------------
# Basic calculation
# ---------------------------------------------------------------------------


def test_basic_result_type():
    r = calculate_pediatric_dose(
        "DrugA", adult_dose_mg=100.0, child_age_years=10.0, child_weight_kg=30.0
    )
    assert isinstance(r, PediatricDoseResult)
    assert r.drug_name == "DrugA"


def test_adult_weight_child_gets_full_dose():
    """A child at adult weight and mature age should get close to adult dose."""
    r = calculate_pediatric_dose(
        "DrugA", adult_dose_mg=100.0, child_age_years=18.0, child_weight_kg=70.0
    )
    assert pytest.approx(r.pediatric_dose_mg, rel=0.01) == 100.0


def test_heavier_child_higher_dose():
    r1 = calculate_pediatric_dose(
        "D", adult_dose_mg=100.0, child_age_years=10.0, child_weight_kg=20.0
    )
    r2 = calculate_pediatric_dose(
        "D", adult_dose_mg=100.0, child_age_years=10.0, child_weight_kg=40.0
    )
    assert r2.pediatric_dose_mg > r1.pediatric_dose_mg


def test_dose_per_kg_computed():
    r = calculate_pediatric_dose(
        "D", adult_dose_mg=100.0, child_age_years=10.0, child_weight_kg=30.0
    )
    assert pytest.approx(r.dose_per_kg_mg_kg, rel=1e-6) == r.pediatric_dose_mg / 30.0


def test_t_half_positive():
    r = calculate_pediatric_dose(
        "D", adult_dose_mg=100.0, child_age_years=5.0, child_weight_kg=20.0
    )
    assert r.t_half_child_h > 0


def test_t_half_formula():
    r = calculate_pediatric_dose(
        "D", adult_dose_mg=100.0, child_age_years=10.0, child_weight_kg=30.0
    )
    expected = 0.693 * r.vd_child_L / r.cl_child_L_per_h
    assert pytest.approx(r.t_half_child_h, rel=1e-6) == expected


# ---------------------------------------------------------------------------
# Maturation
# ---------------------------------------------------------------------------


def test_young_child_lower_dose_hepatic():
    """Very young child (< 2y) should get lower dose due to CYP maturation."""
    r_young = calculate_pediatric_dose(
        "D", 100.0, child_age_years=0.5, child_weight_kg=7.0, elimination="hepatic"
    )
    r_older = calculate_pediatric_dose(
        "D", 100.0, child_age_years=5.0, child_weight_kg=20.0, elimination="hepatic"
    )
    # Dose per kg should be lower for very young due to maturation
    assert r_young.maturation_factor < r_older.maturation_factor


def test_hepatic_maturation_at_age_2_is_1():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=2.0, child_weight_kg=12.0, elimination="hepatic"
    )
    assert pytest.approx(r.maturation_factor, abs=1e-6) == 1.0


def test_hepatic_maturation_at_age_10_is_1():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=10.0, child_weight_kg=30.0, elimination="hepatic"
    )
    assert r.maturation_factor == 1.0


def test_hepatic_maturation_factor_at_half_year():
    """At 0.5 years: mat = 0.5 / (0.5 + 0.5) = 0.5."""
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=0.5, child_weight_kg=7.0, elimination="hepatic"
    )
    assert pytest.approx(r.maturation_factor, abs=1e-6) == 0.5


def test_renal_maturation_applied():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=1.0, child_weight_kg=10.0, elimination="renal"
    )
    expected_mat = 1.0 - math.exp(-0.5 * 1.0)
    assert pytest.approx(r.maturation_factor, rel=1e-6) == expected_mat


def test_renal_young_lower_clearance():
    r_young = calculate_pediatric_dose(
        "D", 100.0, child_age_years=0.5, child_weight_kg=7.0, elimination="renal"
    )
    r_older = calculate_pediatric_dose(
        "D", 100.0, child_age_years=10.0, child_weight_kg=30.0, elimination="renal"
    )
    assert r_young.maturation_factor < r_older.maturation_factor


def test_mixed_elimination():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=1.0, child_weight_kg=10.0, elimination="mixed"
    )
    hep = 1.0 / (1.0 + 0.5)
    ren = 1.0 - math.exp(-0.5 * 1.0)
    expected = 0.5 * (hep + ren)
    assert pytest.approx(r.maturation_factor, rel=1e-6) == expected


# ---------------------------------------------------------------------------
# Allometric scaling
# ---------------------------------------------------------------------------


def test_allometric_cl_scaling():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=10.0, child_weight_kg=30.0, cl_adult_L_per_h=10.0
    )
    # At age 10, hepatic maturation = 1.0
    expected_cl = 10.0 * (30.0 / 70.0) ** 0.75
    assert pytest.approx(r.cl_child_L_per_h, rel=1e-6) == expected_cl


def test_allometric_vd_scaling():
    r = calculate_pediatric_dose(
        "D", 100.0, child_age_years=10.0, child_weight_kg=30.0, vd_adult_L=70.0
    )
    expected_vd = 70.0 * (30.0 / 70.0)
    assert pytest.approx(r.vd_child_L, rel=1e-6) == expected_vd


# ---------------------------------------------------------------------------
# dose_across_ages
# ---------------------------------------------------------------------------


def test_dose_across_ages_count():
    ages = [1.0, 2.0, 5.0, 10.0, 15.0]
    results = dose_across_ages("D", 100.0, ages)
    assert len(results) == 5


def test_dose_across_ages_with_weights():
    ages = [1.0, 5.0, 10.0]
    weights = [10.0, 20.0, 30.0]
    results = dose_across_ages("D", 100.0, ages, child_weights_kg=weights)
    assert len(results) == 3
    assert results[0].child_weight_kg == 10.0
    assert results[2].child_weight_kg == 30.0


def test_dose_across_ages_dose_increases():
    ages = [1.0, 5.0, 10.0, 15.0]
    weights = [10.0, 20.0, 30.0, 55.0]
    results = dose_across_ages("D", 100.0, ages, child_weights_kg=weights)
    doses = [r.pediatric_dose_mg for r in results]
    assert doses == sorted(doses)


def test_dose_per_kg_varies_with_age():
    ages = [0.5, 2.0, 10.0]
    weights = [7.0, 12.0, 30.0]
    results = dose_across_ages("D", 100.0, ages, child_weights_kg=weights)
    dpk = [r.dose_per_kg_mg_kg for r in results]
    # Non-linear: dose_per_kg shouldn't be identical
    assert len(set(round(d, 4) for d in dpk)) > 1


def test_dose_across_ages_auto_weight():
    ages = [3.0, 6.0, 12.0]
    results = dose_across_ages("D", 100.0, ages)
    # Weights auto-estimated: max(3.5, 2.5*age + 8)
    assert results[0].child_weight_kg == pytest.approx(2.5 * 3.0 + 8.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_drug_name():
    with pytest.raises(ValueError, match="drug_name"):
        calculate_pediatric_dose("", 100.0, child_age_years=5.0, child_weight_kg=20.0)


def test_invalid_adult_dose():
    with pytest.raises(ValueError, match="adult_dose_mg"):
        calculate_pediatric_dose("D", -10.0, child_age_years=5.0, child_weight_kg=20.0)


def test_invalid_child_age():
    with pytest.raises(ValueError, match="child_age_years"):
        calculate_pediatric_dose("D", 100.0, child_age_years=-1.0, child_weight_kg=20.0)


def test_invalid_child_weight():
    with pytest.raises(ValueError, match="child_weight_kg"):
        calculate_pediatric_dose("D", 100.0, child_age_years=5.0, child_weight_kg=0)


def test_invalid_elimination():
    with pytest.raises(ValueError, match="elimination"):
        calculate_pediatric_dose(
            "D", 100.0, child_age_years=5.0, child_weight_kg=20.0, elimination="unknown"
        )


def test_dose_across_ages_empty():
    with pytest.raises(ValueError, match="ages_years"):
        dose_across_ages("D", 100.0, [])


def test_dose_across_ages_mismatched_weights():
    with pytest.raises(ValueError, match="length"):
        dose_across_ages("D", 100.0, [1.0, 2.0], child_weights_kg=[10.0])


def test_result_is_frozen():
    r = calculate_pediatric_dose("D", 100.0, child_age_years=5.0, child_weight_kg=20.0)
    with pytest.raises(AttributeError):
        r.pediatric_dose_mg = 999.0  # type: ignore[misc]
