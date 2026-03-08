"""Tests for Phase 581 - Drug Distribution in Obesity."""

import pytest

from omega_pbpk.clinical.obesity_pk_p581 import (
    assess_obesity_pk,
    compare_obesity_levels,
)


class TestValidation:
    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="total_weight_kg"):
            assess_obesity_pk("Drug", -70, 170, "male", 2.0, 5.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="total_weight_kg"):
            assess_obesity_pk("Drug", 0, 170, "male", 2.0, 5.0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            assess_obesity_pk("Drug", 70, -170, "male", 2.0, 5.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            assess_obesity_pk("Drug", 70, 0, "male", 2.0, 5.0)

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            assess_obesity_pk("Drug", 70, 170, "other", 2.0, 5.0)


class TestIBW:
    def test_male_ibw(self):
        # IBW male = 50 + 2.3*(170/2.54 - 60)
        result = assess_obesity_pk("Drug", 70, 170, "male", 1.0, 5.0)
        height_in = 170 / 2.54
        expected_ibw = 50.0 + 2.3 * (height_in - 60.0)
        assert abs(result.ibw_kg - expected_ibw) < 0.01

    def test_female_ibw(self):
        result = assess_obesity_pk("Drug", 60, 160, "female", 1.0, 5.0)
        height_in = 160 / 2.54
        expected_ibw = 45.5 + 2.3 * (height_in - 60.0)
        assert abs(result.ibw_kg - expected_ibw) < 0.01

    def test_ibw_clamped_low(self):
        # Very short person
        result = assess_obesity_pk("Drug", 50, 130, "female", 1.0, 5.0)
        assert result.ibw_kg >= 40.0

    def test_ibw_clamped_high(self):
        # Very tall person
        result = assess_obesity_pk("Drug", 100, 220, "male", 1.0, 5.0)
        assert result.ibw_kg <= 120.0


class TestBMI:
    def test_normal_bmi(self):
        result = assess_obesity_pk("Drug", 70, 175, "male", 1.0, 5.0)
        expected_bmi = 70 / (1.75**2)
        assert abs(result.bmi - expected_bmi) < 0.01

    def test_obese_bmi(self):
        result = assess_obesity_pk("Drug", 130, 170, "male", 1.0, 5.0)
        assert result.bmi > 40.0


class TestObesityClassification:
    def test_normal(self):
        result = assess_obesity_pk("Drug", 65, 175, "male", 1.0, 5.0)
        assert result.obesity_class == "normal"

    def test_overweight(self):
        result = assess_obesity_pk("Drug", 85, 175, "male", 1.0, 5.0)
        assert result.obesity_class == "overweight"

    def test_obese_I(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 1.0, 5.0)
        assert result.obesity_class == "obese_I"

    def test_obese_II(self):
        result = assess_obesity_pk("Drug", 115, 175, "male", 1.0, 5.0)
        assert result.obesity_class == "obese_II"

    def test_obese_III(self):
        result = assess_obesity_pk("Drug", 140, 175, "male", 1.0, 5.0)
        assert result.obesity_class == "obese_III"


class TestDosingStrategy:
    def test_lipophilic_uses_abw(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 3.0, 5.0)
        assert result.dosing_strategy == "adjusted_body_weight"

    def test_hydrophilic_uses_lbw(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 1.0, 5.0)
        assert result.dosing_strategy == "lean_body_weight"

    def test_logP_boundary_uses_lbw(self):
        # logP exactly 2.0 should use LBW
        result = assess_obesity_pk("Drug", 100, 175, "male", 2.0, 5.0)
        assert result.dosing_strategy == "lean_body_weight"


class TestVdAdjusted:
    def test_lipophilic_vd_includes_adipose(self):
        result = assess_obesity_pk("Drug", 120, 175, "male", 4.0, 5.0, 0.5)
        ibw = result.ibw_kg
        expected_vd = 0.5 * ibw + max(0.0, 4.0 - 2.0) * 0.3 * (120 - ibw)
        assert abs(result.vd_adjusted_L - expected_vd) < 0.01

    def test_hydrophilic_vd_no_adipose(self):
        result = assess_obesity_pk("Drug", 120, 175, "male", 1.0, 5.0, 0.5)
        ibw = result.ibw_kg
        expected_vd = 0.5 * ibw
        assert abs(result.vd_adjusted_L - expected_vd) < 0.01


class TestClearanceAndHalfLife:
    def test_cl_adjusted(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 1.0, 5.0)
        expected_cl = 5.0 * (result.lbw_kg / result.ibw_kg)
        assert abs(result.cl_adjusted_L_per_h - expected_cl) < 0.01

    def test_half_life(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 1.0, 5.0)
        expected_t = 0.693 * result.vd_adjusted_L / result.cl_adjusted_L_per_h
        assert abs(result.t_half_adjusted_h - expected_t) < 0.01


class TestDoseAdjustmentFactor:
    def test_factor_is_dosing_weight_over_ibw(self):
        result = assess_obesity_pk("Drug", 100, 175, "male", 3.0, 5.0)
        expected = result.dosing_weight_kg / result.ibw_kg
        assert abs(result.dose_adjustment_factor - expected) < 0.001


class TestCompareObesityLevels:
    def test_returns_sorted_by_bmi(self):
        results = compare_obesity_levels("Drug", 175, "male", 3.0, 5.0, 0.5)
        bmis = [r.bmi for r in results]
        assert bmis == sorted(bmis)

    def test_default_weights(self):
        results = compare_obesity_levels("Drug", 175, "male", 3.0, 5.0, 0.5)
        assert len(results) == 5

    def test_custom_weights(self):
        results = compare_obesity_levels("Drug", 175, "male", 3.0, 5.0, 0.5, weights=[60, 80, 100])
        assert len(results) == 3

    def test_vd_increases_with_weight_lipophilic(self):
        results = compare_obesity_levels("Drug", 175, "male", 4.0, 5.0, 0.5)
        vds = [r.vd_adjusted_L for r in results]
        # Vd should generally increase with weight for lipophilic drugs
        assert vds[-1] > vds[0]


class TestResultDataclass:
    def test_frozen(self):
        result = assess_obesity_pk("Drug", 70, 175, "male", 1.0, 5.0)
        with pytest.raises(AttributeError):
            result.drug_name = "Other"

    def test_drug_name_preserved(self):
        result = assess_obesity_pk("TestDrug", 70, 175, "male", 1.0, 5.0)
        assert result.drug_name == "TestDrug"
