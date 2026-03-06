"""Tests for omega_pbpk.core.pediatric_pbpk module."""

import math
import pytest
from omega_pbpk.core.pediatric_pbpk import (
    PediatricScalingResult,
    scale_to_pediatric,
    _cyp3a4_maturation,
    _gfr_maturation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
ADULT_CL = 10.0   # L/h
ADULT_VD = 50.0   # L
ADULT_DOSE = 100.0  # mg


# ---------------------------------------------------------------------------
# 1. Return type is PediatricScalingResult
# ---------------------------------------------------------------------------

def test_returns_pediatric_scaling_result():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5)
    assert isinstance(result, PediatricScalingResult)


# ---------------------------------------------------------------------------
# 2. Dataclass fields are populated
# ---------------------------------------------------------------------------

def test_result_fields_populated():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=20.0)
    assert result.drug_name == DRUG
    assert result.age_years == 5
    assert result.weight_kg == pytest.approx(20.0)
    assert result.adult_cl_L_per_h == pytest.approx(ADULT_CL)
    assert result.adult_vd_L == pytest.approx(ADULT_VD)
    assert result.scaling_method == "allometric_mat"


# ---------------------------------------------------------------------------
# 3. Pediatric CL <= adult CL for children (< 12 years)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age", [0.5, 1.0, 2.0, 5.0, 8.0, 11.0])
def test_pediatric_cl_less_than_adult_for_children(age):
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=age, weight_kg=20.0)
    assert result.pediatric_cl_L_per_h <= result.adult_cl_L_per_h


# ---------------------------------------------------------------------------
# 4. Pediatric Vd <= adult Vd for children (weight-based scaling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age,wt", [(1, 10), (5, 18), (8, 25), (11, 35)])
def test_pediatric_vd_less_than_adult_for_children(age, wt):
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=age, weight_kg=wt)
    assert result.pediatric_vd_L <= result.adult_vd_L


# ---------------------------------------------------------------------------
# 5. Half-life = 0.693 * Vd / CL
# ---------------------------------------------------------------------------

def test_t_half_formula():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=20.0)
    expected = 0.693 * result.pediatric_vd_L / result.pediatric_cl_L_per_h
    assert result.pediatric_t_half_h == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. Pediatric dose proportional to CL ratio
# ---------------------------------------------------------------------------

def test_pediatric_dose_proportional_to_cl_ratio():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=20.0)
    cl_ratio = result.pediatric_cl_L_per_h / result.adult_cl_L_per_h
    expected_dose = ADULT_DOSE * cl_ratio
    assert result.pediatric_dose_mg == pytest.approx(expected_dose, rel=1e-3)


# ---------------------------------------------------------------------------
# 7. weight_kg estimated if None
# ---------------------------------------------------------------------------

def test_weight_estimated_when_none():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=None)
    assert result.weight_kg is not None
    assert result.weight_kg > 0


# ---------------------------------------------------------------------------
# 8. Weight preserved when explicitly provided
# ---------------------------------------------------------------------------

def test_weight_preserved_when_provided():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=18.5)
    assert result.weight_kg == pytest.approx(18.5)


# ---------------------------------------------------------------------------
# 9. Adult age (~30y) -> pediatric CL approx adult CL
# ---------------------------------------------------------------------------

def test_adult_age_cl_approx_adult():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=30, weight_kg=70.0)
    assert result.pediatric_cl_L_per_h == pytest.approx(ADULT_CL, rel=0.20)


# ---------------------------------------------------------------------------
# 10. CYP3A4 maturation fraction in [0.05, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 30.0])
def test_cyp3a4_maturation_range(age):
    val = _cyp3a4_maturation(age)
    assert 0.05 <= val <= 1.0


# ---------------------------------------------------------------------------
# 11. GFR maturation fraction in [0.03, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 30.0])
def test_gfr_maturation_range(age):
    val = _gfr_maturation(age)
    assert 0.03 <= val <= 1.0


# ---------------------------------------------------------------------------
# 12. CYP3A4 maturation increases with age
# ---------------------------------------------------------------------------

def test_cyp3a4_maturation_increases_with_age():
    ages = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    values = [_cyp3a4_maturation(a) for a in ages]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"CYP3A4 maturation did not increase: age {ages[i]} -> {ages[i+1]}"
        )


# ---------------------------------------------------------------------------
# 13. GFR maturation increases with age
# ---------------------------------------------------------------------------

def test_gfr_maturation_increases_with_age():
    ages = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    values = [_gfr_maturation(a) for a in ages]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"GFR maturation did not increase: age {ages[i]} -> {ages[i+1]}"
        )


# ---------------------------------------------------------------------------
# 14. adult_cl <= 0 raises ValueError
# ---------------------------------------------------------------------------

def test_raises_on_zero_adult_cl():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, 0.0, ADULT_VD, ADULT_DOSE, age_years=5)


def test_raises_on_negative_adult_cl():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, -1.0, ADULT_VD, ADULT_DOSE, age_years=5)


# ---------------------------------------------------------------------------
# 15. adult_vd <= 0 raises ValueError
# ---------------------------------------------------------------------------

def test_raises_on_zero_adult_vd():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, ADULT_CL, 0.0, ADULT_DOSE, age_years=5)


def test_raises_on_negative_adult_vd():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, ADULT_CL, -10.0, ADULT_DOSE, age_years=5)


# ---------------------------------------------------------------------------
# 16. adult_dose <= 0 raises ValueError
# ---------------------------------------------------------------------------

def test_raises_on_nonpositive_adult_dose():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, 0.0, age_years=5)


# ---------------------------------------------------------------------------
# 17. age_years < 0 raises ValueError
# ---------------------------------------------------------------------------

def test_raises_on_negative_age():
    with pytest.raises(ValueError):
        scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=-1)


# ---------------------------------------------------------------------------
# 18. Adolescent (16y) CL closer to adult than infant (1y)
# ---------------------------------------------------------------------------

def test_adolescent_cl_closer_to_adult_than_infant():
    infant = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=1, weight_kg=10.0)
    adolescent = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=16, weight_kg=60.0)

    infant_diff = abs(infant.pediatric_cl_L_per_h - ADULT_CL)
    adolescent_diff = abs(adolescent.pediatric_cl_L_per_h - ADULT_CL)

    assert adolescent_diff < infant_diff, (
        f"Adolescent CL diff ({adolescent_diff:.3f}) should be < infant CL diff ({infant_diff:.3f})"
    )


# ---------------------------------------------------------------------------
# 19. Maturation fractions stored in result
# ---------------------------------------------------------------------------

def test_maturation_fractions_in_result():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5, weight_kg=20.0)
    assert 0.0 < result.cyp3a4_maturation_fraction <= 1.0
    assert 0.0 < result.renal_maturation_fraction <= 1.0


# ---------------------------------------------------------------------------
# 20. CL scales with weight^0.75 (allometric exponent check)
# ---------------------------------------------------------------------------

def test_cl_scales_allometrically():
    """Doubling weight at same age should scale CL by ~2^0.75."""
    wt1, wt2 = 20.0, 40.0
    r1 = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=10, weight_kg=wt1)
    r2 = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=10, weight_kg=wt2)

    ratio_cl = r2.pediatric_cl_L_per_h / r1.pediatric_cl_L_per_h
    expected_ratio = (wt2 / wt1) ** 0.75  # ~1.682

    assert ratio_cl == pytest.approx(expected_ratio, rel=0.05), (
        f"CL ratio {ratio_cl:.3f} should be ~{expected_ratio:.3f} (weight^0.75)"
    )


# ---------------------------------------------------------------------------
# 21. Newborn (age 0) does not raise, returns valid result
# ---------------------------------------------------------------------------

def test_newborn_age_zero_valid():
    result = scale_to_pediatric(DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=0, weight_kg=3.5)
    assert result.pediatric_cl_L_per_h > 0
    assert result.pediatric_vd_L > 0
    assert result.pediatric_t_half_h > 0
    assert result.pediatric_dose_mg > 0


# ---------------------------------------------------------------------------
# 22. fraction_cyp3a4 affects CL (higher CYP3A4 fraction in infant → lower CL)
# ---------------------------------------------------------------------------

def test_fraction_cyp3a4_affects_cl():
    r_high_cyp = scale_to_pediatric(
        DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=1, weight_kg=10.0,
        fraction_cyp3a4=0.9, fraction_renal=0.0,
    )
    r_low_cyp = scale_to_pediatric(
        DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=1, weight_kg=10.0,
        fraction_cyp3a4=0.1, fraction_renal=0.0,
    )
    # Infant has immature CYP3A4; higher fraction → more CL suppression
    assert r_high_cyp.pediatric_cl_L_per_h <= r_low_cyp.pediatric_cl_L_per_h


# ---------------------------------------------------------------------------
# 23. Scaling method stored correctly
# ---------------------------------------------------------------------------

def test_scaling_method_stored():
    result = scale_to_pediatric(
        DRUG, ADULT_CL, ADULT_VD, ADULT_DOSE, age_years=5,
        weight_kg=20.0, method="allometric_mat",
    )
    assert result.scaling_method == "allometric_mat"


# ---------------------------------------------------------------------------
# 24. CYP3A4 maturation at birth is near floor (heavily immature)
# ---------------------------------------------------------------------------

def test_cyp3a4_maturation_low_at_birth():
    val = _cyp3a4_maturation(0.0)
    assert val < 0.3, f"CYP3A4 maturation at birth {val:.3f} should be < 0.3"


# ---------------------------------------------------------------------------
# 25. GFR maturation at birth is near floor
# ---------------------------------------------------------------------------

def test_gfr_maturation_low_at_birth():
    val = _gfr_maturation(0.0)
    assert val <= 0.5, f"GFR maturation at birth {val:.3f} should be <=0.5"  # simplified model
