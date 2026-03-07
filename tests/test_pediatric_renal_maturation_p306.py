"""Tests for Phase 306 — pediatric_renal_maturation_p306 module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.pediatric_renal_maturation_p306 import (
    PediatricRenalCLResult,
    calculate_pediatric_renal_cl,
    compare_age_groups,
)

# ---------------------------------------------------------------------------
# Shared helpers / defaults
# ---------------------------------------------------------------------------

ADULT_CL = 5.0  # L/h
ADULT_WT = 70.0  # kg

NEONATE = dict(age_years=0.02, weight_kg=3.5)  # ~1 week old
INFANT = dict(age_years=0.5, weight_kg=7.0)  # 6 months
TODDLER = dict(age_years=2.0, weight_kg=12.0)  # 2 years
CHILD = dict(age_years=8.0, weight_kg=25.0)  # 8 years
ADOLESCENT_M = dict(age_years=15.0, weight_kg=55.0, sex="male")
ADOLESCENT_F = dict(age_years=15.0, weight_kg=55.0, sex="female")


def _calc(**kwargs):
    defaults = dict(
        drug_name="GentamicinB",
        adult_cl_renal_L_per_h=ADULT_CL,
        adult_weight_kg=ADULT_WT,
        sex="male",
    )
    defaults.update(kwargs)
    return calculate_pediatric_renal_cl(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pediatric_renal_cl_result(self):
        result = _calc(**NEONATE)
        assert isinstance(result, PediatricRenalCLResult)

    def test_is_frozen(self):
        result = _calc(**NEONATE)
        with pytest.raises((AttributeError, TypeError)):
            result.cl_renal_L_per_h = 99.0  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _calc(drug_name="Amoxicillin", **NEONATE)
        assert result.drug_name == "Amoxicillin"

    def test_age_years_preserved(self):
        result = _calc(**NEONATE)
        assert math.isclose(result.age_years, NEONATE["age_years"], rel_tol=1e-9)

    def test_weight_preserved(self):
        result = _calc(**NEONATE)
        assert math.isclose(result.weight_kg, NEONATE["weight_kg"], rel_tol=1e-9)

    def test_sex_preserved(self):
        result = _calc(**NEONATE, sex="female")
        assert result.sex == "female"


# ---------------------------------------------------------------------------
# cl_renal bounds
# ---------------------------------------------------------------------------


class TestCLBounds:
    def test_cl_renal_positive(self):
        result = _calc(**NEONATE)
        assert result.cl_renal_L_per_h > 0.0

    def test_maturation_factor_in_0_1(self):
        result = _calc(**NEONATE)
        assert 0.0 < result.maturation_factor <= 1.0

    def test_maturation_pct_in_0_100(self):
        result = _calc(**CHILD)
        assert 0.0 < result.maturation_pct <= 100.0

    def test_cl_renal_per_kg_positive(self):
        result = _calc(**NEONATE)
        assert result.cl_renal_per_kg_L_per_h_per_kg > 0.0

    def test_allometric_cl_positive(self):
        result = _calc(**NEONATE)
        assert result.allometric_cl_L_per_h > 0.0


# ---------------------------------------------------------------------------
# Maturation gradient
# ---------------------------------------------------------------------------


class TestMaturationGradient:
    def test_neonate_lower_mf_than_child(self):
        r_neonate = _calc(**NEONATE)
        r_child = _calc(**CHILD)
        assert r_neonate.maturation_factor < r_child.maturation_factor

    def test_older_child_higher_cl_than_neonate(self):
        # Use same weight to isolate maturation effect
        r_neonate = _calc(age_years=0.02, weight_kg=10.0)
        r_child = _calc(age_years=8.0, weight_kg=10.0)
        assert r_child.cl_renal_L_per_h > r_neonate.cl_renal_L_per_h

    def test_adult_age_has_high_maturation(self):
        result = _calc(age_years=25.0, weight_kg=70.0)
        assert result.maturation_factor > 0.95

    def test_neonate_maturation_well_below_1(self):
        result = _calc(**NEONATE)
        assert result.maturation_factor < 0.5


# ---------------------------------------------------------------------------
# Age class
# ---------------------------------------------------------------------------


class TestAgeClass:
    def test_neonate_class(self):
        result = _calc(**NEONATE)
        assert result.age_class == "neonate"

    def test_infant_class(self):
        result = _calc(**INFANT)
        assert result.age_class == "infant"

    def test_toddler_class(self):
        result = _calc(**TODDLER)
        assert result.age_class == "toddler"

    def test_child_class(self):
        result = _calc(**CHILD)
        assert result.age_class == "child"

    def test_adolescent_class(self):
        result = _calc(**ADOLESCENT_M)
        assert result.age_class == "adolescent"

    def test_age_class_valid_values(self):
        valid = {"neonate", "infant", "toddler", "child", "adolescent"}
        for age, wt in [(0.01, 3.0), (0.3, 6.0), (2.0, 12.0), (8.0, 25.0), (15.0, 55.0)]:
            result = _calc(age_years=age, weight_kg=wt)
            assert result.age_class in valid


# ---------------------------------------------------------------------------
# Sex correction
# ---------------------------------------------------------------------------


class TestSexCorrection:
    def test_female_adolescent_lower_cl_than_male(self):
        r_male = _calc(**ADOLESCENT_M)
        r_female = _calc(**ADOLESCENT_F)
        assert r_female.cl_renal_L_per_h < r_male.cl_renal_L_per_h

    def test_sex_correction_factor_for_female_adolescent(self):
        r_male = _calc(**ADOLESCENT_M)
        r_female = _calc(**ADOLESCENT_F)
        ratio = r_female.cl_renal_L_per_h / r_male.cl_renal_L_per_h
        assert math.isclose(ratio, 0.85, rel_tol=1e-6)

    def test_no_sex_correction_for_male_adolescent(self):
        r_male = _calc(**ADOLESCENT_M)
        # For male, sex_factor=1, so cl = allometric * MF
        assert math.isclose(
            r_male.cl_renal_L_per_h,
            r_male.allometric_cl_L_per_h * r_male.maturation_factor,
            rel_tol=1e-6,
        )

    def test_no_sex_correction_for_female_child(self):
        # Female < 12 years: no sex correction
        r_male = _calc(age_years=8.0, weight_kg=25.0, sex="male")
        r_female = _calc(age_years=8.0, weight_kg=25.0, sex="female")
        assert math.isclose(r_male.cl_renal_L_per_h, r_female.cl_renal_L_per_h, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# dose_adjustment_factor
# ---------------------------------------------------------------------------


class TestDoseAdjustmentFactor:
    def test_dose_adj_positive(self):
        result = _calc(**NEONATE)
        assert result.dose_adjustment_factor > 0.0

    def test_dose_adj_le_1_for_pediatric(self):
        # Pediatric patients (low weight) should generally have DAF < 1
        result = _calc(**NEONATE)
        assert result.dose_adjustment_factor <= 1.0

    def test_dose_adj_equals_cl_ratio(self):
        result = _calc(**CHILD)
        expected = result.cl_renal_L_per_h / ADULT_CL
        assert math.isclose(result.dose_adjustment_factor, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compare_age_groups
# ---------------------------------------------------------------------------


class TestCompareAgeGroups:
    def test_returns_list(self):
        result = compare_age_groups(
            drug_name="Gentamicin",
            adult_cl_renal_L_per_h=ADULT_CL,
            adult_weight_kg=ADULT_WT,
            sex="male",
            age_years_list=[0.02, 2.0, 8.0, 15.0],
            weight_kg_list=[3.5, 12.0, 25.0, 55.0],
        )
        assert isinstance(result, list)
        assert len(result) == 4

    def test_sorted_ascending_by_cl_renal(self):
        result = compare_age_groups(
            drug_name="Gentamicin",
            adult_cl_renal_L_per_h=ADULT_CL,
            adult_weight_kg=ADULT_WT,
            sex="male",
            age_years_list=[8.0, 0.02, 15.0, 2.0],
            weight_kg_list=[25.0, 3.5, 55.0, 12.0],
        )
        cl_values = [r.cl_renal_L_per_h for r in result]
        assert cl_values == sorted(cl_values)

    def test_all_results_are_correct_type(self):
        result = compare_age_groups(
            drug_name="Ampicillin",
            adult_cl_renal_L_per_h=ADULT_CL,
            adult_weight_kg=ADULT_WT,
            sex="female",
            age_years_list=[0.5, 5.0],
            weight_kg_list=[7.0, 20.0],
        )
        for r in result:
            assert isinstance(r, PediatricRenalCLResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_adult_cl_zero_raises(self):
        with pytest.raises(ValueError, match="adult_cl_renal_L_per_h"):
            _calc(adult_cl_renal_L_per_h=0.0, **NEONATE)

    def test_adult_cl_negative_raises(self):
        with pytest.raises(ValueError, match="adult_cl_renal_L_per_h"):
            _calc(adult_cl_renal_L_per_h=-1.0, **NEONATE)

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age_years"):
            _calc(age_years=-0.1, weight_kg=3.5)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            _calc(age_years=1.0, weight_kg=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            _calc(age_years=1.0, weight_kg=-5.0)

    def test_zero_adult_weight_raises(self):
        with pytest.raises(ValueError, match="adult_weight_kg"):
            calculate_pediatric_renal_cl(
                drug_name="D",
                adult_cl_renal_L_per_h=ADULT_CL,
                age_years=1.0,
                weight_kg=10.0,
                adult_weight_kg=0.0,
                sex="male",
            )

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            _calc(age_years=5.0, weight_kg=20.0, sex="other")

    def test_compare_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compare_age_groups(
                drug_name="D",
                adult_cl_renal_L_per_h=ADULT_CL,
                adult_weight_kg=ADULT_WT,
                sex="male",
                age_years_list=[1.0, 5.0],
                weight_kg_list=[10.0],
            )
