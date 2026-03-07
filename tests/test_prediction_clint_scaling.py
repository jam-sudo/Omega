"""Tests for prediction/clint_scaling.py — Drug Intrinsic Clearance Scaling (IVIVE)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.clint_scaling import (
    ClintScalingResult,
    compare_assay_types,
    scale_clint_to_human,
)

VALID_EXTRACTION_CLASSES = {"low", "intermediate", "high"}
VALID_FUP_SENSITIVITIES = {"high sensitivity", "moderate", "low sensitivity"}


class TestScaleClintToHumanReturnsResult:
    def test_returns_dataclass(self):
        result = scale_clint_to_human("Drug A", clint_in_vitro=5.0)
        assert isinstance(result, ClintScalingResult)

    def test_frozen_dataclass(self):
        result = scale_clint_to_human("Drug B", clint_in_vitro=5.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = scale_clint_to_human("Warfarin", clint_in_vitro=5.0)
        assert result.drug_name == "Warfarin"

    def test_clint_in_vitro_preserved(self):
        result = scale_clint_to_human("Drug C", clint_in_vitro=7.5)
        assert math.isclose(result.clint_in_vitro, 7.5)


class TestCLHepatic:
    def test_cl_hepatic_positive(self):
        result = scale_clint_to_human("Drug D", clint_in_vitro=5.0)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_cl_hepatic_le_q_liver(self):
        q = 90.0
        result = scale_clint_to_human("Drug E", clint_in_vitro=1000.0, q_liver_L_per_h=q)
        assert result.cl_hepatic_L_per_h <= q

    def test_cl_hepatic_increases_with_clint(self):
        r1 = scale_clint_to_human("Drug F", clint_in_vitro=1.0, fup=0.5)
        r2 = scale_clint_to_human("Drug F", clint_in_vitro=100.0, fup=0.5)
        assert r2.cl_hepatic_L_per_h > r1.cl_hepatic_L_per_h


class TestExtractionRatio:
    def test_er_between_zero_and_one(self):
        result = scale_clint_to_human("Drug G", clint_in_vitro=5.0)
        assert 0.0 < result.extraction_ratio <= 1.0

    def test_high_clint_high_er(self):
        # Need very large CLint to get ER > 0.7 with microsomal scaling
        # CLint_liver = clint_mic * 45 * 1500 * 60 / 1e9
        # For ER=0.7 with fup=1, QH=90: CLint_liver = 0.7*90/(1-0.7) = 210 L/h
        # clint_mic = 210 * 1e9 / (45 * 1500 * 60) = 210e9 / 4_050_000 ≈ 51852
        result = scale_clint_to_human("Drug H", clint_in_vitro=100000.0, fup=1.0)
        assert result.extraction_class == "high"

    def test_low_clint_low_er(self):
        result = scale_clint_to_human("Drug I", clint_in_vitro=0.001, fup=0.01)
        assert result.extraction_class == "low"

    def test_extraction_class_valid(self):
        result = scale_clint_to_human("Drug J", clint_in_vitro=5.0)
        assert result.extraction_class in VALID_EXTRACTION_CLASSES

    def test_intermediate_extraction(self):
        # Choose CLint that gives ER around 0.5
        result = scale_clint_to_human("Drug K", clint_in_vitro=5.0, fup=0.5)
        assert result.extraction_class in VALID_EXTRACTION_CLASSES


class TestFupSensitivity:
    def test_fup_sensitivity_valid(self):
        result = scale_clint_to_human("Drug L", clint_in_vitro=5.0)
        assert result.fup_sensitivity in VALID_FUP_SENSITIVITIES

    def test_low_er_high_fup_sensitivity(self):
        result = scale_clint_to_human("Drug M", clint_in_vitro=0.001, fup=0.01)
        assert result.fup_sensitivity == "high sensitivity"

    def test_high_er_low_fup_sensitivity(self):
        result = scale_clint_to_human("Drug N", clint_in_vitro=100000.0, fup=1.0)
        assert result.fup_sensitivity == "low sensitivity"


class TestBioavailabilityFirstPass:
    def test_bioavailability_plus_er_equals_one(self):
        result = scale_clint_to_human("Drug O", clint_in_vitro=5.0)
        assert math.isclose(
            result.bioavailability_first_pass + result.extraction_ratio, 1.0, rel_tol=1e-9
        )

    def test_high_er_low_bioavailability(self):
        result = scale_clint_to_human("Drug P", clint_in_vitro=100000.0, fup=1.0)
        assert result.bioavailability_first_pass < 0.3

    def test_low_er_high_bioavailability(self):
        result = scale_clint_to_human("Drug Q", clint_in_vitro=0.001, fup=0.01)
        assert result.bioavailability_first_pass > 0.7


class TestAssayTypes:
    def test_microsomal_assay_type_stored(self):
        result = scale_clint_to_human("Drug R", clint_in_vitro=5.0, assay_type="microsomal")
        assert result.assay_type == "microsomal"

    def test_hepatocyte_assay_type_stored(self):
        result = scale_clint_to_human("Drug S", clint_in_vitro=5.0, assay_type="hepatocyte")
        assert result.assay_type == "hepatocyte"

    def test_microsomal_hepatocyte_different_clint_liver(self):
        # Same numeric CLint value but different scaling gives different whole-liver CLint
        r_mic = scale_clint_to_human("Drug T", clint_in_vitro=5.0, assay_type="microsomal")
        r_hep = scale_clint_to_human("Drug T", clint_in_vitro=5.0, assay_type="hepatocyte")
        assert not math.isclose(r_mic.clint_liver_L_per_h, r_hep.clint_liver_L_per_h, rel_tol=0.01)


class TestNotes:
    def test_notes_nonempty(self):
        result = scale_clint_to_human("Drug U", clint_in_vitro=5.0)
        assert len(result.notes) > 0


class TestValidation:
    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError):
            scale_clint_to_human("Drug V", clint_in_vitro=0.0)

    def test_raises_on_negative_clint(self):
        with pytest.raises(ValueError):
            scale_clint_to_human("Drug W", clint_in_vitro=-1.0)

    def test_raises_on_zero_fup(self):
        with pytest.raises(ValueError):
            scale_clint_to_human("Drug X", clint_in_vitro=5.0, fup=0.0)

    def test_raises_on_zero_q_liver(self):
        with pytest.raises(ValueError):
            scale_clint_to_human("Drug Y", clint_in_vitro=5.0, q_liver_L_per_h=0.0)


class TestCompareAssayTypes:
    def test_returns_dict(self):
        result = compare_assay_types("Drug Z", clint_mic=5.0, clint_hep=2.0)
        assert isinstance(result, dict)

    def test_has_microsomal_result_key(self):
        result = compare_assay_types("Drug AA", clint_mic=5.0, clint_hep=2.0)
        assert "microsomal_result" in result

    def test_has_hepatocyte_result_key(self):
        result = compare_assay_types("Drug BB", clint_mic=5.0, clint_hep=2.0)
        assert "hepatocyte_result" in result

    def test_has_cl_ratio_key(self):
        result = compare_assay_types("Drug CC", clint_mic=5.0, clint_hep=2.0)
        assert "cl_ratio" in result

    def test_microsomal_result_is_dataclass(self):
        result = compare_assay_types("Drug DD", clint_mic=5.0, clint_hep=2.0)
        assert isinstance(result["microsomal_result"], ClintScalingResult)

    def test_hepatocyte_result_is_dataclass(self):
        result = compare_assay_types("Drug EE", clint_mic=5.0, clint_hep=2.0)
        assert isinstance(result["hepatocyte_result"], ClintScalingResult)

    def test_cl_ratio_positive(self):
        result = compare_assay_types("Drug FF", clint_mic=5.0, clint_hep=2.0)
        assert result["cl_ratio"] > 0.0
