"""Tests for Phase 564 - Allometric Scaling for Pediatric Dosing."""

import pytest

from omega_pbpk.clinical.allometric_pediatric_dosing import (
    AllometricPediatricResult,
    _maturation_factor,
    scale_pediatric_dose,
    screen_age_groups,
)


class TestReturnType:
    def test_returns_allometric_result(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert isinstance(r, AllometricPediatricResult)

    def test_result_is_frozen(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        with pytest.raises(AttributeError):
            r.drug_name = "other"


class TestClearancScaling:
    def test_cl_ped_less_than_adult_low_weight(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert r.cl_pediatric_L_per_h < r.cl_adult_L_per_h

    def test_cl_ped_allometric_exponent(self):
        r = scale_pediatric_dose("DrugA", 10.0, 35.0, 10.0, 50.0, 100.0, use_maturation=False)
        expected = 10.0 * (35.0 / 70.0) ** 0.75
        assert abs(r.cl_pediatric_L_per_h - expected) < 1e-9


class TestVdScaling:
    def test_vd_ped_less_than_adult_low_weight(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert r.vd_pediatric_L < r.vd_adult_L

    def test_vd_linear_scaling(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        expected = 50.0 * (20.0 / 70.0)
        assert abs(r.vd_pediatric_L - expected) < 1e-9


class TestHalfLife:
    def test_t_half_positive(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert r.t_half_h > 0

    def test_t_half_formula(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        expected = 0.693 * r.vd_pediatric_L / r.cl_pediatric_L_per_h
        assert abs(r.t_half_h - expected) < 1e-9


class TestMaturationFactor:
    def test_mf_in_range(self):
        r = scale_pediatric_dose("DrugA", 0.5, 7.0, 10.0, 50.0, 100.0)
        assert 0 <= r.maturation_factor <= 1

    def test_infant_lower_mf_than_child(self):
        infant = scale_pediatric_dose("DrugA", 0.25, 5.0, 10.0, 50.0, 100.0)
        child = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert infant.maturation_factor < child.maturation_factor

    def test_child_over_2_mf_is_1(self):
        r = scale_pediatric_dose("DrugA", 3.0, 15.0, 10.0, 50.0, 100.0)
        assert r.maturation_factor == 1.0

    def test_neonate_mf_less_than_1(self):
        mf = _maturation_factor(0.0)
        assert mf < 1.0
        assert mf > 0.0


class TestDoseLinearity:
    def test_dose_doubles_with_double_auc(self):
        r1 = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 50.0)
        r2 = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert abs(r2.dose_total_mg - 2.0 * r1.dose_total_mg) < 1e-9

    def test_dose_per_kg_consistent(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0)
        assert abs(r.dose_per_kg_mg - r.dose_total_mg / 20.0) < 1e-9


class TestMaturationVsSimple:
    def test_no_maturation_higher_cl_for_neonates(self):
        r_mat = scale_pediatric_dose("DrugA", 0.25, 5.0, 10.0, 50.0, 100.0, use_maturation=True)
        r_simple = scale_pediatric_dose("DrugA", 0.25, 5.0, 10.0, 50.0, 100.0, use_maturation=False)
        assert r_simple.cl_pediatric_L_per_h > r_mat.cl_pediatric_L_per_h

    def test_no_maturation_mf_is_1(self):
        r = scale_pediatric_dose("DrugA", 0.25, 5.0, 10.0, 50.0, 100.0, use_maturation=False)
        assert r.maturation_factor == 1.0

    def test_scaling_method_with_maturation(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0, use_maturation=True)
        assert r.scaling_method == "with_maturation"

    def test_scaling_method_simple(self):
        r = scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 100.0, use_maturation=False)
        assert r.scaling_method == "simple_allometric"


class TestScreenAgeGroups:
    def test_returns_5_results(self):
        results = screen_age_groups("DrugA", 10.0, 50.0, 100.0)
        assert len(results) == 5

    def test_sorted_by_age_ascending(self):
        results = screen_age_groups("DrugA", 10.0, 50.0, 100.0)
        ages = [r.age_years for r in results]
        for i in range(len(ages) - 1):
            assert ages[i] <= ages[i + 1]

    def test_all_results_are_correct_type(self):
        results = screen_age_groups("DrugA", 10.0, 50.0, 100.0)
        for r in results:
            assert isinstance(r, AllometricPediatricResult)


class TestValidation:
    def test_weight_zero(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", 5.0, 0.0, 10.0, 50.0, 100.0)

    def test_weight_negative(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", 5.0, -1.0, 10.0, 50.0, 100.0)

    def test_age_negative(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", -1.0, 20.0, 10.0, 50.0, 100.0)

    def test_cl_zero(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", 5.0, 20.0, 0.0, 50.0, 100.0)

    def test_vd_zero(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 0.0, 100.0)

    def test_target_auc_zero(self):
        with pytest.raises(ValueError):
            scale_pediatric_dose("DrugA", 5.0, 20.0, 10.0, 50.0, 0.0)
