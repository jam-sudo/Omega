"""Tests for Phase 190: body weight-based PK parameter scaling."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.body_weight_scaling import (
    BWScalingResult,
    dose_by_weight_band,
    scale_pk_by_weight,
)

# --- Basic smoke test ---

def test_scale_pk_returns_bwscaling_result():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=70.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
    )
    assert isinstance(result, BWScalingResult)
    assert result.drug_name == "Drug"


# --- Same weight → scaling factor = 1 ---

def test_same_weight_cl_factor_one():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=70.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
    )
    assert math.isclose(result.cl_scaling_factor, 1.0, rel_tol=1e-9)
    assert math.isclose(result.vd_scaling_factor, 1.0, rel_tol=1e-9)
    assert math.isclose(result.scaled_cl_L_per_h, 10.0, rel_tol=1e-9)
    assert math.isclose(result.scaled_vd_L, 50.0, rel_tol=1e-9)
    assert math.isclose(result.recommended_dose_mg, 500.0, rel_tol=1e-9)


# --- 2x weight allometric scaling ---

def test_double_weight_cl_factor_exponent_075():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=140.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
        cl_exponent=0.75,
    )
    expected_cl_factor = 2.0**0.75
    assert math.isclose(result.cl_scaling_factor, expected_cl_factor, rel_tol=1e-6)
    assert math.isclose(result.scaled_cl_L_per_h, 10.0 * expected_cl_factor, rel_tol=1e-6)


def test_double_weight_vd_linear_exponent():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=140.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
        vd_exponent=1.0,
    )
    # Linear scaling: factor = 2.0
    assert math.isclose(result.vd_scaling_factor, 2.0, rel_tol=1e-9)
    assert math.isclose(result.scaled_vd_L, 100.0, rel_tol=1e-9)


# --- Dose capping ---

def test_dose_capping_at_max():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=200.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
        max_dose_mg=600.0,
    )
    assert result.dose_capping is True
    assert result.recommended_dose_mg == 600.0


def test_dose_flooring_at_min():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=30.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
        min_dose_mg=400.0,
    )
    assert result.dose_capping is True
    assert result.recommended_dose_mg == 400.0


def test_no_dose_capping_within_range():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=80.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
        max_dose_mg=1000.0,
        min_dose_mg=100.0,
    )
    assert result.dose_capping is False


# --- t_half calculation ---

def test_scaled_t_half_computed_correctly():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=70.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
    )
    expected_t_half = math.log(2) * 50.0 / 10.0  # 0.693 * Vd / CL
    assert math.isclose(result.scaled_t_half_h, expected_t_half, rel_tol=1e-6)


# --- dose_by_weight_band ---

def test_dose_by_weight_band_returns_correct_length():
    bands = [(30.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 120.0)]
    results = dose_by_weight_band(
        drug_name="Drug",
        weight_bands_kg=bands,
        reference_weight_kg=70.0,
        reference_dose_mg=500.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
    )
    assert len(results) == len(bands)


def test_dose_by_weight_band_all_bwscaling_results():
    bands = [(40.0, 60.0), (60.0, 80.0)]
    results = dose_by_weight_band(
        drug_name="Drug",
        weight_bands_kg=bands,
        reference_weight_kg=70.0,
        reference_dose_mg=500.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
    )
    for r in results:
        assert isinstance(r, BWScalingResult)


def test_dose_by_weight_band_uses_midpoint():
    """Band (60, 80) → midpoint=70 → same weight as reference → factor=1."""
    results = dose_by_weight_band(
        drug_name="Drug",
        weight_bands_kg=[(60.0, 80.0)],
        reference_weight_kg=70.0,
        reference_dose_mg=500.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
    )
    assert math.isclose(results[0].cl_scaling_factor, 1.0, rel_tol=1e-9)


def test_dose_by_weight_band_empty():
    results = dose_by_weight_band(
        drug_name="Drug",
        weight_bands_kg=[],
        reference_weight_kg=70.0,
        reference_dose_mg=500.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
    )
    assert results == []


# --- Validation errors ---

def test_zero_reference_weight_raises():
    with pytest.raises(ValueError, match="reference_weight_kg"):
        scale_pk_by_weight(
            drug_name="Drug",
            reference_weight_kg=0.0,
            patient_weight_kg=70.0,
            reference_cl_L_per_h=10.0,
            reference_vd_L=50.0,
            reference_dose_mg=500.0,
        )


def test_negative_patient_weight_raises():
    with pytest.raises(ValueError, match="patient_weight_kg"):
        scale_pk_by_weight(
            drug_name="Drug",
            reference_weight_kg=70.0,
            patient_weight_kg=-10.0,
            reference_cl_L_per_h=10.0,
            reference_vd_L=50.0,
            reference_dose_mg=500.0,
        )


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="reference_cl_L_per_h"):
        scale_pk_by_weight(
            drug_name="Drug",
            reference_weight_kg=70.0,
            patient_weight_kg=70.0,
            reference_cl_L_per_h=0.0,
            reference_vd_L=50.0,
            reference_dose_mg=500.0,
        )


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="reference_vd_L"):
        scale_pk_by_weight(
            drug_name="Drug",
            reference_weight_kg=70.0,
            patient_weight_kg=70.0,
            reference_cl_L_per_h=10.0,
            reference_vd_L=0.0,
            reference_dose_mg=500.0,
        )


def test_zero_cl_exponent_raises():
    with pytest.raises(ValueError, match="cl_exponent"):
        scale_pk_by_weight(
            drug_name="Drug",
            reference_weight_kg=70.0,
            patient_weight_kg=70.0,
            reference_cl_L_per_h=10.0,
            reference_vd_L=50.0,
            reference_dose_mg=500.0,
            cl_exponent=0.0,
        )


# --- Dose proportional to CL ---

def test_dose_proportional_to_cl_factor():
    result = scale_pk_by_weight(
        drug_name="Drug",
        reference_weight_kg=70.0,
        patient_weight_kg=50.0,
        reference_cl_L_per_h=10.0,
        reference_vd_L=50.0,
        reference_dose_mg=500.0,
    )
    expected_dose = 500.0 * result.cl_scaling_factor
    assert math.isclose(result.recommended_dose_mg, expected_dose, rel_tol=1e-9)


# --- Stored reference values ---

def test_stored_reference_values():
    result = scale_pk_by_weight(
        drug_name="TestDrug",
        reference_weight_kg=70.0,
        patient_weight_kg=90.0,
        reference_cl_L_per_h=8.0,
        reference_vd_L=40.0,
        reference_dose_mg=250.0,
    )
    assert result.reference_dose_mg == 250.0
    assert result.reference_weight_kg == 70.0
    assert result.reference_cl_L_per_h == 8.0
    assert result.reference_vd_L == 40.0
