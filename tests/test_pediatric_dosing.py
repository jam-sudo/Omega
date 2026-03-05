"""Tests for pediatric dosing module."""

import pytest

from omega_pbpk.clinical.pediatric_dosing import (
    PediatricDosingResult,
    pediatric_dose,
    pediatric_dose_range,
    allometric_dose_scaling,
)


@pytest.mark.unit
class TestAllometricScaling:
    def test_adult_weight_gives_adult_dose(self):
        result = allometric_dose_scaling(100, 70, 70, 0.75)
        assert result == pytest.approx(100, rel=0.01)

    def test_lighter_child_lower_dose(self):
        dose_20 = allometric_dose_scaling(100, 70, 20, 0.75)
        dose_70 = allometric_dose_scaling(100, 70, 70, 0.75)
        assert dose_20 < dose_70

    def test_exponent_0_75(self):
        result = allometric_dose_scaling(100, 70, 20, 0.75)
        expected = 100 * (20 / 70) ** 0.75
        assert result == pytest.approx(expected, rel=0.001)

    def test_positive_dose(self):
        result = allometric_dose_scaling(50, 70, 10, 0.75)
        assert result > 0


@pytest.mark.unit
class TestPediatricDose:
    def test_returns_result_type(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert isinstance(result, PediatricDosingResult)

    def test_neonate_warning(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=0.1)
        warnings_lower = [w.lower() for w in result.warnings]
        assert any("neonat" in w or "infant" in w for w in warnings_lower)

    def test_child_dose_less_than_adult(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert result.combined_adjusted_dose_mg < 100

    def test_allometric_dose_positive(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert result.allometric_dose_mg > 0

    def test_cyp3a4_fraction_adult_between_0_1(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert 0 <= result.cyp3a4_fraction_adult <= 1

    def test_gfr_estimated_positive(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert result.gfr_estimated > 0

    def test_dose_per_kg_matches(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        expected = result.combined_adjusted_dose_mg / result.patient_weight_kg
        assert abs(result.dose_per_kg - expected) < 0.01

    def test_cyp3a4_adjustment_reduces_dose(self):
        high_cyp = pediatric_dose(adult_dose_mg=100, child_age_years=2, fraction_cyp3a4=0.9)
        low_cyp = pediatric_dose(adult_dose_mg=100, child_age_years=2, fraction_cyp3a4=0.0)
        assert high_cyp.combined_adjusted_dose_mg < low_cyp.combined_adjusted_dose_mg

    def test_renal_adjustment_reduces_dose(self):
        high_renal = pediatric_dose(adult_dose_mg=100, child_age_years=0.1, fraction_renal=0.9)
        low_renal = pediatric_dose(adult_dose_mg=100, child_age_years=0.1, fraction_renal=0.0)
        assert high_renal.combined_adjusted_dose_mg < low_renal.combined_adjusted_dose_mg

    def test_adult_patient_near_full_dose(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=18, child_weight_kg=70)
        assert result.combined_adjusted_dose_mg == pytest.approx(100, rel=0.20)

    def test_weight_estimated_from_age(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5, child_weight_kg=None)
        assert result.patient_weight_kg > 0

    def test_min_dose_floor(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=0.5, min_dose_mg=50)
        assert result.combined_adjusted_dose_mg >= 50


@pytest.mark.unit
class TestPediatricDoseRange:
    def test_returns_list(self):
        result = pediatric_dose_range(adult_dose_mg=100)
        assert isinstance(result, list)

    def test_has_7_entries(self):
        result = pediatric_dose_range(adult_dose_mg=100)
        assert len(result) == 7

    def test_entries_have_keys(self):
        result = pediatric_dose_range(adult_dose_mg=100)
        for entry in result:
            assert "age_years" in entry
            assert "weight_kg" in entry
            assert "dose_mg" in entry
            assert "dose_per_kg" in entry

    def test_older_child_higher_dose(self):
        result = pediatric_dose_range(adult_dose_mg=100)
        doses = [entry["dose_mg"] for entry in result]
        # Generally dose should increase with age; check first vs last
        assert doses[-1] > doses[0]
