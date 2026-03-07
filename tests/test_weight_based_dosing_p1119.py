"""Tests for Phase 1119 — weight_based_dosing_p1119."""

import math
import pytest

from omega_pbpk.clinical.weight_based_dosing_p1119 import (
    WeightBasedDoseResult,
    allometric_dose_adjustment,
    body_surface_area_dose,
    calculate_weight_based_dose,
)


# ---------------------------------------------------------------------------
# calculate_weight_based_dose
# ---------------------------------------------------------------------------

class TestCalculateWeightBasedDose:
    def test_basic_70kg(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=2.0, weight_kg=70.0)
        assert res.calculated_dose_mg == pytest.approx(140.0)
        assert res.final_dose_mg == pytest.approx(140.0)
        assert res.dose_capped is False
        assert res.cap_reason == "none"

    def test_dose_proportional_to_weight(self):
        res1 = calculate_weight_based_dose("DrugA", dose_per_kg_mg=3.0, weight_kg=50.0)
        res2 = calculate_weight_based_dose("DrugA", dose_per_kg_mg=3.0, weight_kg=100.0)
        assert res2.calculated_dose_mg == pytest.approx(2 * res1.calculated_dose_mg)

    def test_method_label(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=1.0, weight_kg=60.0)
        assert res.method == "mg_per_kg"

    def test_no_bsa(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=1.0, weight_kg=60.0)
        assert res.patient_bsa_m2 == 0.0

    def test_max_cap_applied(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=5.0, weight_kg=100.0, max_dose_mg=400.0)
        assert res.calculated_dose_mg == pytest.approx(500.0)
        assert res.final_dose_mg == pytest.approx(400.0)
        assert res.dose_capped is True
        assert res.cap_reason == "max_exceeded"

    def test_min_cap_applied(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=1.0, weight_kg=10.0, min_dose_mg=50.0)
        assert res.calculated_dose_mg == pytest.approx(10.0)
        assert res.final_dose_mg == pytest.approx(50.0)
        assert res.dose_capped is True
        assert res.cap_reason == "min_not_reached"

    def test_no_cap_intensity_100(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=2.0, weight_kg=70.0)
        assert res.dose_intensity_pct == pytest.approx(100.0)

    def test_max_cap_reduces_intensity(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=5.0, weight_kg=100.0, max_dose_mg=400.0)
        # 400/500 * 100 = 80%
        assert res.dose_intensity_pct == pytest.approx(80.0)

    def test_min_cap_increases_intensity_over_100(self):
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=1.0, weight_kg=10.0, min_dose_mg=50.0)
        # 50/10 * 100 = 500%
        assert res.dose_intensity_pct == pytest.approx(500.0)

    def test_drug_name_preserved(self):
        res = calculate_weight_based_dose("Paclitaxel", dose_per_kg_mg=2.0, weight_kg=70.0)
        assert res.drug_name == "Paclitaxel"

    def test_validation_dose_per_kg_zero(self):
        with pytest.raises(ValueError):
            calculate_weight_based_dose("DrugA", dose_per_kg_mg=0.0, weight_kg=70.0)

    def test_validation_dose_per_kg_negative(self):
        with pytest.raises(ValueError):
            calculate_weight_based_dose("DrugA", dose_per_kg_mg=-1.0, weight_kg=70.0)

    def test_validation_weight_zero(self):
        with pytest.raises(ValueError):
            calculate_weight_based_dose("DrugA", dose_per_kg_mg=2.0, weight_kg=0.0)

    def test_validation_weight_negative(self):
        with pytest.raises(ValueError):
            calculate_weight_based_dose("DrugA", dose_per_kg_mg=2.0, weight_kg=-70.0)

    def test_exact_max_boundary_no_cap(self):
        # exactly at max → should not cap
        res = calculate_weight_based_dose("DrugA", dose_per_kg_mg=2.0, weight_kg=70.0, max_dose_mg=140.0)
        assert res.dose_capped is False
        assert res.cap_reason == "none"


# ---------------------------------------------------------------------------
# allometric_dose_adjustment
# ---------------------------------------------------------------------------

class TestAllometricDoseAdjustment:
    def test_basic_scaling(self):
        # Same weight → same dose
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=70.0)
        assert res.final_dose_mg == pytest.approx(100.0)

    def test_method_label(self):
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=70.0)
        assert res.method == "allometric"

    def test_higher_weight_higher_dose(self):
        res_light = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=50.0)
        res_heavy = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=90.0)
        assert res_heavy.final_dose_mg > res_light.final_dose_mg

    def test_sub_proportional_scaling(self):
        # exponent 0.75 → doubling weight gives less than double dose
        res_ref = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=70.0)
        res_2x = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=140.0)
        assert res_2x.final_dose_mg < 2 * res_ref.final_dose_mg

    def test_exponent_0_75_formula(self):
        ref_dose = 200.0
        ref_wt = 70.0
        pt_wt = 85.0
        expected = ref_dose * (pt_wt / ref_wt) ** 0.75
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=ref_dose, reference_weight_kg=ref_wt, patient_weight_kg=pt_wt)
        assert res.calculated_dose_mg == pytest.approx(expected)

    def test_custom_exponent(self):
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=140.0, exponent=1.0)
        assert res.calculated_dose_mg == pytest.approx(200.0)

    def test_no_cap_by_default(self):
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=200.0)
        assert res.dose_capped is False

    def test_validation_ref_weight_zero(self):
        with pytest.raises(ValueError):
            allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=0.0, patient_weight_kg=70.0)

    def test_validation_patient_weight_zero(self):
        with pytest.raises(ValueError):
            allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=0.0)

    def test_intensity_100_no_cap(self):
        res = allometric_dose_adjustment("DrugB", reference_dose_mg=100.0, reference_weight_kg=70.0, patient_weight_kg=80.0)
        assert res.dose_intensity_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# body_surface_area_dose
# ---------------------------------------------------------------------------

class TestBodySurfaceAreaDose:
    def test_mosteller_bsa_calculation(self):
        # BSA = sqrt(70 * 170 / 3600)
        weight = 70.0
        height = 170.0
        expected_bsa = math.sqrt(weight * height / 3600.0)
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=weight, height_cm=height)
        assert res.patient_bsa_m2 == pytest.approx(expected_bsa)

    def test_basic_dose(self):
        weight = 70.0
        height = 170.0
        bsa = math.sqrt(weight * height / 3600.0)
        expected_dose = 50.0 * bsa
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=weight, height_cm=height)
        assert res.calculated_dose_mg == pytest.approx(expected_dose)

    def test_method_label(self):
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=70.0, height_cm=170.0)
        assert res.method == "bsa_based"

    def test_max_cap_applied(self):
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=1000.0, weight_kg=70.0, height_cm=170.0, max_dose_mg=100.0)
        assert res.final_dose_mg == pytest.approx(100.0)
        assert res.dose_capped is True
        assert res.cap_reason == "max_exceeded"

    def test_no_cap_intensity_100(self):
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=70.0, height_cm=170.0)
        assert res.dose_intensity_pct == pytest.approx(100.0)

    def test_validation_dose_per_m2_zero(self):
        with pytest.raises(ValueError):
            body_surface_area_dose("DrugC", dose_per_m2_mg=0.0, weight_kg=70.0, height_cm=170.0)

    def test_validation_weight_zero(self):
        with pytest.raises(ValueError):
            body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=0.0, height_cm=170.0)

    def test_validation_height_zero(self):
        with pytest.raises(ValueError):
            body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=70.0, height_cm=0.0)

    def test_result_is_dataclass(self):
        res = body_surface_area_dose("DrugC", dose_per_m2_mg=50.0, weight_kg=70.0, height_cm=170.0)
        assert isinstance(res, WeightBasedDoseResult)
