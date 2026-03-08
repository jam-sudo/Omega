"""Tests for Phase 567 - Drug Metabolism Age Scaling (Adult to Elderly)."""

import pytest

from omega_pbpk.clinical.elderly_metabolism_scaling import (
    ElderlyMetabolismResult,
    scale_elderly_metabolism,
    screen_age_effects,
)


def _make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        age_years=70,
        weight_kg=70.0,
        sex="male",
        cl_adult_L_per_h=10.0,
        vd_adult_L=50.0,
    )
    defaults.update(kwargs)
    return scale_elderly_metabolism(**defaults)


class TestReturnType:
    def test_returns_elderly_metabolism_result(self):
        r = _make_result()
        assert isinstance(r, ElderlyMetabolismResult)

    def test_all_fields_present(self):
        r = _make_result()
        assert r.drug_name == "TestDrug"
        assert r.age_years == 70
        assert r.weight_kg == 70.0
        assert r.sex == "male"
        assert isinstance(r.cl_adult_L_per_h, float)
        assert isinstance(r.cl_elderly_L_per_h, float)
        assert isinstance(r.vd_adult_L, float)
        assert isinstance(r.vd_elderly_L, float)
        assert isinstance(r.t_half_adult_h, float)
        assert isinstance(r.t_half_elderly_h, float)
        assert isinstance(r.dose_adjustment_factor, float)
        assert isinstance(r.gfr_elderly_mL_per_min, float)
        assert isinstance(r.cyp3a4_activity_factor, float)
        assert isinstance(r.aging_class, str)
        assert isinstance(r.notes, str)

    def test_frozen_dataclass(self):
        r = _make_result()
        with pytest.raises(AttributeError):
            r.drug_name = "other"


class TestGFR:
    def test_gfr_decreases_with_age(self):
        r50 = _make_result(age_years=50)
        r70 = _make_result(age_years=70)
        r85 = _make_result(age_years=85)
        assert r50.gfr_elderly_mL_per_min > r70.gfr_elderly_mL_per_min
        assert r70.gfr_elderly_mL_per_min > r85.gfr_elderly_mL_per_min

    def test_gfr_clamped_lower(self):
        r = _make_result(age_years=120)
        assert r.gfr_elderly_mL_per_min >= 10.0

    def test_gfr_young_adult(self):
        r = _make_result(age_years=25)
        assert r.gfr_elderly_mL_per_min == 120.0


class TestClearance:
    def test_cl_elderly_le_adult(self):
        r = _make_result(age_years=70)
        assert r.cl_elderly_L_per_h <= r.cl_adult_L_per_h

    def test_cl_elderly_positive(self):
        r = _make_result(age_years=85)
        assert r.cl_elderly_L_per_h > 0

    def test_cl_decreases_with_age(self):
        r50 = _make_result(age_years=50)
        r80 = _make_result(age_years=80)
        assert r50.cl_elderly_L_per_h > r80.cl_elderly_L_per_h


class TestHalfLife:
    def test_t_half_elderly_ge_adult(self):
        r = _make_result(age_years=75)
        assert r.t_half_elderly_h >= r.t_half_adult_h

    def test_t_half_positive(self):
        r = _make_result(age_years=70)
        assert r.t_half_adult_h > 0
        assert r.t_half_elderly_h > 0


class TestDoseAdjustment:
    def test_dose_adjustment_in_range(self):
        r = _make_result(age_years=70)
        assert 0 < r.dose_adjustment_factor <= 1.0

    def test_dose_adjustment_young_adult_near_one(self):
        r = _make_result(age_years=30)
        assert abs(r.dose_adjustment_factor - 1.0) < 0.01


class TestAgingClass:
    def test_middle_aged(self):
        r = _make_result(age_years=50)
        assert r.aging_class == "middle_aged"

    def test_young_old(self):
        r = _make_result(age_years=68)
        assert r.aging_class == "young_old"

    def test_old(self):
        r = _make_result(age_years=78)
        assert r.aging_class == "old"

    def test_oldest_old(self):
        r = _make_result(age_years=90)
        assert r.aging_class == "oldest_old"


class TestCYP3A4:
    def test_cyp3a4_in_range(self):
        r = _make_result(age_years=80)
        assert 0.5 <= r.cyp3a4_activity_factor <= 1.0

    def test_cyp3a4_decreases_with_age(self):
        r40 = _make_result(age_years=40)
        r80 = _make_result(age_years=80)
        assert r40.cyp3a4_activity_factor > r80.cyp3a4_activity_factor

    def test_cyp3a4_young_adult_is_one(self):
        r = _make_result(age_years=25)
        assert r.cyp3a4_activity_factor == 1.0


class TestSex:
    def test_male_accepted(self):
        r = _make_result(sex="male")
        assert r.sex == "male"

    def test_female_accepted(self):
        r = _make_result(sex="female")
        assert r.sex == "female"


class TestScreenAgeEffects:
    def test_returns_list(self):
        results = screen_age_effects("TestDrug", 10.0, 50.0)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_sorted_by_age_ascending(self):
        results = screen_age_effects("TestDrug", 10.0, 50.0)
        ages = [r.age_years for r in results]
        assert ages == sorted(ages)

    def test_custom_ages(self):
        results = screen_age_effects("TestDrug", 10.0, 50.0, ages=[60, 80])
        assert len(results) == 2


class TestValidation:
    def test_age_zero_raises(self):
        with pytest.raises(ValueError):
            _make_result(age_years=0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            _make_result(weight_kg=-5)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            _make_result(cl_adult_L_per_h=-1)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError):
            _make_result(vd_adult_L=-1)

    def test_invalid_fractions_raises(self):
        with pytest.raises(ValueError):
            scale_elderly_metabolism("X", 70, 70.0, "male", 10.0, 50.0, f_renal=0.8, f_hepatic=0.8)

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError):
            _make_result(sex="other")
