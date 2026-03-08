"""Tests for Phase 1112 — BSA-based pediatric dosing."""

import math

import pytest

from omega_pbpk.clinical.pediatric_dose_bsa import (
    PediatricDoseBSAResult,
    calculate_bsa,
    dose_by_age_group,
    pediatric_dose_bsa,
)

# ---------------------------------------------------------------------------
# calculate_bsa — Mosteller formula
# ---------------------------------------------------------------------------


def test_bsa_mosteller_30kg_120cm():
    # sqrt(30 * 120 / 3600) = sqrt(1) = 1.0
    bsa = calculate_bsa(30.0, 120.0)
    assert bsa == pytest.approx(1.0, rel=1e-6)


def test_bsa_mosteller_70kg_170cm():
    expected = math.sqrt(70.0 * 170.0 / 3600.0)
    assert calculate_bsa(70.0, 170.0) == pytest.approx(expected, rel=1e-6)


def test_bsa_mosteller_10kg_75cm():
    expected = math.sqrt(10.0 * 75.0 / 3600.0)
    assert calculate_bsa(10.0, 75.0) == pytest.approx(expected, rel=1e-6)


def test_bsa_positive():
    assert calculate_bsa(20.0, 100.0) > 0


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0)
    assert isinstance(result, PediatricDoseBSAResult)


def test_field_drug_name():
    result = pediatric_dose_bsa("Amoxicillin", 500.0, 25.0, 110.0, 5.0)
    assert result.drug_name == "Amoxicillin"


def test_field_adult_dose_preserved():
    result = pediatric_dose_bsa("Drug", 250.0, 30.0, 120.0, 7.0)
    assert result.adult_dose_mg == pytest.approx(250.0)


def test_field_weight_preserved():
    result = pediatric_dose_bsa("Drug", 100.0, 22.5, 110.0, 6.0)
    assert result.weight_kg == pytest.approx(22.5)


def test_field_height_preserved():
    result = pediatric_dose_bsa("Drug", 100.0, 22.5, 110.0, 6.0)
    assert result.height_cm == pytest.approx(110.0)


def test_field_age_preserved():
    result = pediatric_dose_bsa("Drug", 100.0, 22.5, 110.0, 8.5)
    assert result.age_years == pytest.approx(8.5)


def test_field_bsa_m2_correct():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0)
    assert result.bsa_m2 == pytest.approx(1.0, rel=1e-6)


def test_field_adult_bsa_default():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0)
    assert result.adult_bsa_m2 == pytest.approx(1.73)


def test_field_adult_bsa_custom():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0, adult_bsa_m2=1.8)
    assert result.adult_bsa_m2 == pytest.approx(1.8)


def test_field_max_single_dose_equals_adult_dose():
    result = pediatric_dose_bsa("Drug", 300.0, 30.0, 120.0, 6.0)
    assert result.max_single_dose_mg == pytest.approx(300.0)


def test_result_is_frozen():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0)
    with pytest.raises(Exception):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proportional dosing
# ---------------------------------------------------------------------------


def test_proportional_dose():
    # BSA = 1.0 m², adult_bsa=1.73 → fraction = 1/1.73
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0, adult_bsa_m2=1.73)
    expected_dose = 100.0 * (1.0 / 1.73)
    assert result.recommended_dose_mg == pytest.approx(expected_dose, rel=1e-5)


def test_bsa_fraction_correct():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0, adult_bsa_m2=1.0)
    # BSA = 1.0, adult_bsa = 1.0 → fraction = 1.0
    assert result.bsa_fraction == pytest.approx(1.0, rel=1e-6)


def test_double_bsa_double_dose():
    # Two children: one at BSA=0.5, one at BSA=1.0 (same adult BSA)
    r1 = pediatric_dose_bsa(
        "Drug", 200.0, weight_kg=7.5, height_cm=60.0, age_years=0.5, adult_bsa_m2=2.0
    )
    r2 = pediatric_dose_bsa(
        "Drug", 200.0, weight_kg=30.0, height_cm=120.0, age_years=7.0, adult_bsa_m2=2.0
    )
    # BSA1 = sqrt(7.5*60/3600), BSA2 = sqrt(30*120/3600)=1.0
    # Ratio of doses should equal ratio of BSAs
    bsa1 = calculate_bsa(7.5, 60.0)
    bsa2 = calculate_bsa(30.0, 120.0)
    dose_ratio = r2.recommended_dose_mg / r1.recommended_dose_mg
    bsa_ratio = bsa2 / bsa1
    assert dose_ratio == pytest.approx(bsa_ratio, rel=1e-5)


# ---------------------------------------------------------------------------
# Dose capping
# ---------------------------------------------------------------------------


def test_dose_capped_when_bsa_exceeds_adult():
    # Child heavier than adult reference BSA → should cap
    # adult_bsa_m2 = 0.5 (very small reference), child BSA = 1.0
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 16.0, adult_bsa_m2=0.5)
    assert result.dose_capped is True
    assert result.recommended_dose_mg == pytest.approx(100.0)


def test_dose_not_capped_when_bsa_below_adult():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 8.0, adult_bsa_m2=1.73)
    assert result.dose_capped is False
    assert result.recommended_dose_mg < 100.0


def test_dose_capped_at_adult_dose():
    result = pediatric_dose_bsa("Drug", 50.0, 80.0, 180.0, 16.0, adult_bsa_m2=0.5)
    assert result.recommended_dose_mg == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Age group labels
# ---------------------------------------------------------------------------


def test_age_group_neonate():
    result = pediatric_dose_bsa("Drug", 10.0, 3.5, 50.0, 0.04)  # ~2 weeks
    assert result.age_group == "neonate"


def test_age_group_infant():
    result = pediatric_dose_bsa("Drug", 50.0, 8.0, 65.0, 1.0)
    assert result.age_group == "infant"


def test_age_group_toddler():
    result = pediatric_dose_bsa("Drug", 50.0, 14.0, 90.0, 3.0)
    assert result.age_group == "toddler"


def test_age_group_child():
    result = pediatric_dose_bsa("Drug", 100.0, 25.0, 115.0, 8.0)
    assert result.age_group == "child"


def test_age_group_adolescent():
    result = pediatric_dose_bsa("Drug", 200.0, 55.0, 160.0, 14.0)
    assert result.age_group == "adolescent"


def test_age_group_boundary_infant():
    # exactly 0.083 years = 1 month boundary
    result = pediatric_dose_bsa("Drug", 50.0, 5.0, 55.0, 0.083)
    assert result.age_group == "infant"


# ---------------------------------------------------------------------------
# dose_by_age_group function
# ---------------------------------------------------------------------------


def test_dose_by_age_group_same_as_pediatric_dose_bsa():
    r1 = pediatric_dose_bsa("Drug", 100.0, 25.0, 110.0, 6.0)
    r2 = dose_by_age_group("Drug", 100.0, 25.0, 110.0, 6.0)
    assert r1.recommended_dose_mg == pytest.approx(r2.recommended_dose_mg)
    assert r1.age_group == r2.age_group
    assert r1.bsa_m2 == pytest.approx(r2.bsa_m2)


def test_dose_by_age_group_return_type():
    result = dose_by_age_group("Drug", 100.0, 25.0, 110.0, 6.0)
    assert isinstance(result, PediatricDoseBSAResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_adult_dose_zero():
    with pytest.raises(ValueError, match="adult_dose_mg"):
        pediatric_dose_bsa("Drug", 0.0, 30.0, 120.0, 6.0)


def test_invalid_adult_dose_negative():
    with pytest.raises(ValueError, match="adult_dose_mg"):
        pediatric_dose_bsa("Drug", -100.0, 30.0, 120.0, 6.0)


def test_invalid_weight_zero():
    with pytest.raises(ValueError, match="weight_kg"):
        pediatric_dose_bsa("Drug", 100.0, 0.0, 120.0, 6.0)


def test_invalid_height_zero():
    with pytest.raises(ValueError, match="height_cm"):
        pediatric_dose_bsa("Drug", 100.0, 30.0, 0.0, 6.0)


def test_invalid_age_negative():
    with pytest.raises(ValueError, match="age_years"):
        pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, -1.0)


def test_invalid_adult_bsa_zero():
    with pytest.raises(ValueError, match="adult_bsa_m2"):
        pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0, adult_bsa_m2=0.0)


def test_notes_field_not_empty():
    result = pediatric_dose_bsa("Drug", 100.0, 30.0, 120.0, 6.0)
    assert len(result.notes) > 0
