"""Tests for pediatric weight/BSA-based dosing module (Phase 228)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pediatric_dosing import (
    PediatricDoseResult,
    PediatricDosingResult,
    allometric_dose_scaling,
    bsa_dubois,
    calculate_pediatric_dose,
    dose_frequency_adjustment,
    pediatric_dose,
    pediatric_dose_range,
)

# ---------------------------------------------------------------------------
# bsa_dubois tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBsaDubois:
    def test_adult_70kg_170cm_approx_1_73(self):
        """Standard adult: 70 kg, 170 cm → DuBois formula gives ~1.81 m²."""
        bsa = bsa_dubois(70.0, 170.0)
        # DuBois: 0.007184 * 70^0.425 * 170^0.725 ≈ 1.81 m²
        assert abs(bsa - 1.81) < 0.05

    def test_child_20kg_110cm_less_than_adult(self):
        child_bsa = bsa_dubois(20.0, 110.0)
        adult_bsa = bsa_dubois(70.0, 170.0)
        assert child_bsa < adult_bsa

    def test_bsa_positive(self):
        assert bsa_dubois(30.0, 120.0) > 0.0

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            bsa_dubois(0.0, 120.0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            bsa_dubois(20.0, -10.0)

    def test_larger_child_higher_bsa(self):
        bsa_small = bsa_dubois(15.0, 100.0)
        bsa_large = bsa_dubois(30.0, 130.0)
        assert bsa_large > bsa_small


# ---------------------------------------------------------------------------
# calculate_pediatric_dose tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculatePediatricDose:
    def test_returns_pediatric_dose_result(self):
        result = calculate_pediatric_dose(
            "amoxicillin", 500.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        assert isinstance(result, PediatricDoseResult)

    def test_bsa_method_same_bsa_gives_adult_dose(self):
        """When adult_bsa_m2 equals child BSA, dose equals adult dose."""
        child_bsa = bsa_dubois(70.0, 170.0)  # ~1.81 m²
        result = calculate_pediatric_dose(
            "drug_x", 100.0, adult_bsa_m2=child_bsa, weight_kg=70.0, height_cm=170.0, method="bsa"
        )
        assert abs(result.calculated_dose_mg - 100.0) < 0.1

    def test_bsa_method_smaller_child_lower_dose(self):
        result = calculate_pediatric_dose(
            "drug_x", 100.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        assert result.calculated_dose_mg < 100.0

    def test_weight_method_returns_result(self):
        result = calculate_pediatric_dose("paracetamol", 1000.0, weight_kg=25.0, method="weight")
        assert isinstance(result, PediatricDoseResult)
        assert result.calculated_dose_mg > 0.0

    def test_young_rule_age_12_gives_50pct(self):
        """Young's rule: age=12 → dose = adult * 12/(12+12) = 0.5 * adult."""
        result = calculate_pediatric_dose("drug_y", 200.0, age_years=12.0, method="young")
        assert abs(result.calculated_dose_mg - 100.0) < 1.0

    def test_clark_rule_68kg_gives_adult_dose(self):
        """Clark's rule: 68 kg → adult dose."""
        result = calculate_pediatric_dose("drug_z", 500.0, weight_kg=68.0, method="clark")
        assert abs(result.calculated_dose_mg - 500.0) < 1.0

    def test_max_dose_clamp_applied(self):
        result = calculate_pediatric_dose(
            "drug_a", 200.0, weight_kg=50.0, height_cm=160.0, method="bsa", max_dose_mg=80.0
        )
        assert result.adjusted_dose_mg <= 80.0

    def test_min_dose_clamp_applied(self):
        result = calculate_pediatric_dose(
            "drug_b", 200.0, weight_kg=5.0, height_cm=60.0, method="bsa", min_dose_mg=20.0
        )
        assert result.adjusted_dose_mg >= 20.0

    def test_was_clamped_true_when_clamped(self):
        result = calculate_pediatric_dose(
            "drug_c", 500.0, weight_kg=5.0, height_cm=60.0, method="bsa", max_dose_mg=10.0
        )
        assert result.was_clamped is True

    def test_was_clamped_false_when_not_clamped(self):
        result = calculate_pediatric_dose(
            "drug_d", 100.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        assert result.was_clamped is False

    def test_dose_per_kg_positive(self):
        result = calculate_pediatric_dose(
            "drug_e", 100.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        assert result.dose_per_kg is not None
        assert result.dose_per_kg > 0.0

    def test_dose_per_kg_consistent(self):
        result = calculate_pediatric_dose(
            "drug_f", 100.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        expected = result.adjusted_dose_mg / 20.0
        assert abs(result.dose_per_kg - expected) < 1e-6  # type: ignore[operator]

    def test_cautions_populated_for_age_under_2(self):
        result = calculate_pediatric_dose(
            "drug_g", 50.0, weight_kg=10.0, height_cm=80.0, age_years=1.0, method="bsa"
        )
        assert len(result.cautions) > 0

    def test_cautions_empty_for_older_child(self):
        result = calculate_pediatric_dose(
            "drug_h", 100.0, weight_kg=30.0, height_cm=130.0, age_years=8.0, method="bsa"
        )
        assert len(result.cautions) == 0

    def test_bsa_m2_set_for_bsa_method(self):
        result = calculate_pediatric_dose(
            "drug_i", 100.0, weight_kg=20.0, height_cm=110.0, method="bsa"
        )
        assert result.bsa_m2 is not None

    def test_bsa_m2_none_for_weight_method(self):
        result = calculate_pediatric_dose("drug_j", 100.0, weight_kg=20.0, method="weight")
        assert result.bsa_m2 is None

    def test_bsa_method_missing_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            calculate_pediatric_dose("drug_k", 100.0, height_cm=110.0, method="bsa")

    def test_bsa_method_missing_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            calculate_pediatric_dose("drug_l", 100.0, weight_kg=20.0, method="bsa")

    def test_young_method_missing_age_raises(self):
        with pytest.raises(ValueError, match="age_years"):
            calculate_pediatric_dose("drug_m", 100.0, method="young")

    def test_clark_method_missing_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            calculate_pediatric_dose("drug_n", 100.0, method="clark")

    def test_negative_adult_dose_raises(self):
        with pytest.raises(ValueError):
            calculate_pediatric_dose("drug_o", -50.0, weight_kg=20.0, method="weight")

    def test_drug_name_preserved(self):
        result = calculate_pediatric_dose("ibuprofen", 400.0, weight_kg=25.0, method="weight")
        assert result.drug_name == "ibuprofen"

    def test_method_preserved(self):
        result = calculate_pediatric_dose("drug_p", 100.0, weight_kg=20.0, method="clark")
        assert result.method == "clark"


# ---------------------------------------------------------------------------
# dose_frequency_adjustment tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDoseFrequencyAdjustment:
    def test_returns_float(self):
        result = dose_frequency_adjustment(5.0, 8.0)
        assert isinstance(result, float)

    def test_younger_gives_longer_interval(self):
        interval_young = dose_frequency_adjustment(0.1, 8.0)
        interval_older = dose_frequency_adjustment(5.0, 8.0)
        assert interval_young > interval_older

    def test_adult_near_adult_interval(self):
        """At age ~18, interval should be close to adult interval."""
        result = dose_frequency_adjustment(18.0, 8.0)
        assert result == pytest.approx(8.0, rel=0.1)

    def test_zero_age_raises(self):
        with pytest.raises(ValueError):
            dose_frequency_adjustment(0.0, 8.0)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError):
            dose_frequency_adjustment(5.0, 0.0)


# ---------------------------------------------------------------------------
# Legacy API tests (kept for backward compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyApi:
    def test_allometric_dose_scaling(self):
        result = allometric_dose_scaling(100, 70, 70, 0.75)
        assert result == pytest.approx(100, rel=0.01)

    def test_pediatric_dose_returns_legacy_result(self):
        result = pediatric_dose(adult_dose_mg=100, child_age_years=5)
        assert isinstance(result, PediatricDosingResult)

    def test_pediatric_dose_range_returns_list(self):
        result = pediatric_dose_range(adult_dose_mg=100)
        assert isinstance(result, list)
        assert len(result) == 7
