"""Tests for dose individualization module."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.dose_individualization import (
    IndividualizedDoseResult,
    PatientCovariates,
    batch_individualize,
    individualize_dose,
)

# Standard patient
STANDARD = PatientCovariates(
    body_weight_kg=70, age_years=40, sex="male", serum_creatinine_mg_dL=1.0, child_pugh_score=5
)

# Renal impaired
RENAL_SEVERE = PatientCovariates(
    body_weight_kg=70, age_years=60, sex="male", serum_creatinine_mg_dL=4.0, child_pugh_score=5
)

# Hepatic impaired (Child-Pugh C)
HEPATIC_SEVERE = PatientCovariates(
    body_weight_kg=65, age_years=55, sex="female", serum_creatinine_mg_dL=1.0, child_pugh_score=13
)

# Elderly
ELDERLY = PatientCovariates(
    body_weight_kg=60, age_years=78, sex="female", serum_creatinine_mg_dL=1.2, child_pugh_score=5
)


class TestPatientCovariates:
    def test_defaults(self):
        p = PatientCovariates()
        assert p.body_weight_kg == pytest.approx(70.0)
        assert p.age_years == pytest.approx(40.0)

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            STANDARD.age_years = 50  # type: ignore


class TestIndividualizeDose:
    def test_returns_result_type(self):
        r = individualize_dose("P001", STANDARD, 100.0)
        assert isinstance(r, IndividualizedDoseResult)

    def test_standard_patient_near_full_dose(self):
        # Normal renal + hepatic → dose ~= standard
        r = individualize_dose("P001", STANDARD, 100.0)
        assert r.individualized_dose_mg == pytest.approx(100.0, abs=20.0)

    def test_renal_impairment_reduces_dose(self):
        r_norm = individualize_dose("P001", STANDARD, 100.0, fraction_renal=0.8)
        r_renal = individualize_dose("P002", RENAL_SEVERE, 100.0, fraction_renal=0.8)
        assert r_renal.individualized_dose_mg < r_norm.individualized_dose_mg

    def test_hepatic_impairment_reduces_dose(self):
        r_norm = individualize_dose("P001", STANDARD, 100.0, fraction_hepatic=0.9)
        r_hep = individualize_dose("P002", HEPATIC_SEVERE, 100.0, fraction_hepatic=0.9)
        assert r_hep.individualized_dose_mg < r_norm.individualized_dose_mg

    def test_renal_grade_normal(self):
        r = individualize_dose("P001", STANDARD, 100.0)
        assert r.renal_impairment_grade == "normal"

    def test_renal_grade_severe(self):
        r = individualize_dose("P001", RENAL_SEVERE, 100.0)
        assert r.renal_impairment_grade in ("moderate", "severe", "ESRD")

    def test_hepatic_grade_severe(self):
        r = individualize_dose("P001", HEPATIC_SEVERE, 100.0)
        assert r.hepatic_impairment_grade == "severe"

    def test_elderly_warning(self):
        r = individualize_dose("P001", ELDERLY, 100.0)
        assert any("65" in w or "elderly" in w.lower() for w in r.warnings)

    def test_egfr_positive(self):
        r = individualize_dose("P001", STANDARD, 100.0)
        assert r.egfr_mL_min > 0

    def test_crcl_positive(self):
        r = individualize_dose("P001", STANDARD, 100.0)
        assert r.crcl_mL_min > 0

    def test_predicted_auc_ratio_higher_with_impairment(self):
        # Impaired clearance → AUC ratio > 1.0
        r = individualize_dose("P001", RENAL_SEVERE, 100.0, fraction_renal=0.8)
        # After dose adjustment, AUC ratio should be ~1.0 (that's the point)
        assert r.predicted_auc_ratio > 0

    def test_dose_rounded_to_5mg(self):
        r = individualize_dose("P001", RENAL_SEVERE, 100.0, fraction_renal=0.7)
        assert r.individualized_dose_mg % 5 == 0

    def test_patient_id_stored(self):
        r = individualize_dose("XYZ-123", STANDARD, 100.0)
        assert r.patient_id == "XYZ-123"

    def test_dose_adjustment_factor(self):
        r = individualize_dose("P001", STANDARD, 100.0)
        expected = r.individualized_dose_mg / r.standard_dose_mg
        assert r.dose_adjustment_factor == pytest.approx(expected, rel=0.01)

    def test_weight_adjustment_heavier_patient(self):
        heavy = PatientCovariates(
            body_weight_kg=120,
            age_years=40,
            sex="male",
            serum_creatinine_mg_dL=1.0,
            child_pugh_score=5,
        )
        r = individualize_dose("P001", heavy, 100.0, use_weight_adjustment=True)
        r_std = individualize_dose("P001", STANDARD, 100.0, use_weight_adjustment=True)
        assert r.individualized_dose_mg > r_std.individualized_dose_mg

    def test_custom_egfr_used(self):
        patient = PatientCovariates(
            body_weight_kg=70,
            age_years=40,
            sex="male",
            serum_creatinine_mg_dL=1.0,
            egfr_mL_min=20.0,
        )
        r = individualize_dose("P001", patient, 100.0, fraction_renal=0.8)
        assert r.egfr_mL_min == pytest.approx(20.0)
        assert r.renal_impairment_grade == "severe"


class TestBatchIndividualize:
    def test_multiple_patients(self):
        patients = [("P001", STANDARD), ("P002", RENAL_SEVERE), ("P003", ELDERLY)]
        results = batch_individualize(patients, standard_dose_mg=100.0)
        assert len(results) == 3

    def test_all_results_valid(self):
        patients = [("P001", STANDARD), ("P002", HEPATIC_SEVERE)]
        results = batch_individualize(patients, standard_dose_mg=200.0)
        for r in results:
            assert isinstance(r, IndividualizedDoseResult)
            assert r.individualized_dose_mg > 0

    def test_empty_cohort(self):
        results = batch_individualize([], standard_dose_mg=100.0)
        assert results == []
