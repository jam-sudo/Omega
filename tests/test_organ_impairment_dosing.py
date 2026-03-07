"""Tests for Phase 809 — organ_impairment_dosing module."""

import dataclasses

import pytest

from omega_pbpk.clinical.organ_impairment_dosing import (
    OrganImpairmentDosingResult,
    calculate_organ_impairment_dose,
    child_pugh_class,
    screen_organ_impairment,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnTypeAndFrozen:
    def test_returns_correct_type(self):
        result = calculate_organ_impairment_dose(
            drug_name="DrugA",
            normal_dose_mg=100.0,
            normal_interval_h=12.0,
            fe_urine=0.5,
            fm_hepatic=0.4,
        )
        assert isinstance(result, OrganImpairmentDosingResult)

    def test_result_is_frozen(self):
        result = calculate_organ_impairment_dose(
            drug_name="DrugA",
            normal_dose_mg=100.0,
            normal_interval_h=12.0,
            fe_urine=0.5,
            fm_hepatic=0.4,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.drug_name = "Changed"  # type: ignore

    def test_all_fields_present(self):
        result = calculate_organ_impairment_dose(
            drug_name="DrugB",
            normal_dose_mg=200.0,
            normal_interval_h=24.0,
            fe_urine=0.3,
            fm_hepatic=0.6,
        )
        for field in dataclasses.fields(result):
            assert hasattr(result, field.name)


# ---------------------------------------------------------------------------
# Normal function → no adjustment
# ---------------------------------------------------------------------------


class TestNormalFunction:
    def test_normal_renal_hepatic_factor_1(self):
        result = calculate_organ_impairment_dose(
            drug_name="Drug",
            normal_dose_mg=100.0,
            normal_interval_h=12.0,
            fe_urine=0.5,
            fm_hepatic=0.5,
            egfr_ml_per_min=100.0,
            child_pugh_score=5,
        )
        assert result.renal_adjustment_factor == pytest.approx(1.0)
        assert result.hepatic_adjustment_factor == pytest.approx(1.0)
        assert result.combined_adjustment_factor == pytest.approx(1.0)
        assert result.recommended_dose_mg == pytest.approx(100.0)

    def test_normal_function_no_monitoring(self):
        result = calculate_organ_impairment_dose(
            drug_name="Drug",
            normal_dose_mg=100.0,
            normal_interval_h=12.0,
            fe_urine=0.5,
            fm_hepatic=0.5,
            egfr_ml_per_min=100.0,
            child_pugh_score=5,
        )
        assert result.monitoring_required is False
        assert result.contraindicated is False

    def test_renal_function_description_normal(self):
        result = calculate_organ_impairment_dose(
            drug_name="Drug",
            normal_dose_mg=100.0,
            normal_interval_h=12.0,
            fe_urine=0.5,
            fm_hepatic=0.5,
            egfr_ml_per_min=100.0,
            child_pugh_score=5,
        )
        assert result.renal_function == "normal"
        assert result.hepatic_function == "A"


# ---------------------------------------------------------------------------
# Renal impairment
# ---------------------------------------------------------------------------


class TestRenalImpairment:
    def test_severe_renal_reduces_dose(self):
        result_normal = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.8, 0.2, egfr_ml_per_min=100.0
        )
        result_severe = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.8, 0.2, egfr_ml_per_min=10.0
        )
        assert result_severe.recommended_dose_mg < result_normal.recommended_dose_mg

    def test_renal_function_categories(self):
        for egfr, expected in [
            (100.0, "normal"),
            (75.0, "mild"),
            (45.0, "moderate"),
            (20.0, "severe"),
            (5.0, "ESRD"),
        ]:
            result = calculate_organ_impairment_dose(
                "Drug", 100.0, 12.0, 0.5, 0.2, egfr_ml_per_min=egfr
            )
            assert result.renal_function == expected, f"eGFR={egfr}"

    def test_renal_factor_floor(self):
        """Even extreme renal failure should not go below 0.1."""
        result = calculate_organ_impairment_dose("Drug", 100.0, 12.0, 1.0, 0.0, egfr_ml_per_min=0.0)
        assert result.renal_adjustment_factor >= 0.1


# ---------------------------------------------------------------------------
# Hepatic impairment
# ---------------------------------------------------------------------------


class TestHepaticImpairment:
    def test_child_pugh_class_a(self):
        assert child_pugh_class(5) == "A"
        assert child_pugh_class(6) == "A"

    def test_child_pugh_class_b(self):
        assert child_pugh_class(7) == "B"
        assert child_pugh_class(9) == "B"

    def test_child_pugh_class_c(self):
        assert child_pugh_class(10) == "C"
        assert child_pugh_class(15) == "C"

    def test_hepatic_b_reduces_dose(self):
        result_a = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.1, 0.8, child_pugh_score=5
        )
        result_b = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.1, 0.8, child_pugh_score=8
        )
        assert result_b.recommended_dose_mg < result_a.recommended_dose_mg

    def test_hepatic_c_lower_than_b(self):
        result_b = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.1, 0.8, child_pugh_score=8
        )
        result_c = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.1, 0.8, child_pugh_score=12
        )
        assert result_c.recommended_dose_mg <= result_b.recommended_dose_mg


# ---------------------------------------------------------------------------
# Combined impairment
# ---------------------------------------------------------------------------


class TestCombinedImpairment:
    def test_combined_less_than_each_individually(self):
        result_renal = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.7, 0.0, egfr_ml_per_min=20.0, child_pugh_score=5
        )
        result_hepatic = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.0, 0.8, egfr_ml_per_min=100.0, child_pugh_score=10
        )
        result_combined = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.7, 0.8, egfr_ml_per_min=20.0, child_pugh_score=10
        )
        assert result_combined.combined_adjustment_factor <= result_renal.combined_adjustment_factor
        assert (
            result_combined.combined_adjustment_factor <= result_hepatic.combined_adjustment_factor
        )

    def test_combined_factor_clamped_at_1(self):
        result = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.0, 0.0, egfr_ml_per_min=100.0, child_pugh_score=5
        )
        assert result.combined_adjustment_factor <= 1.0

    def test_combined_factor_minimum(self):
        result = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 1.0, 1.0, egfr_ml_per_min=0.0, child_pugh_score=15
        )
        assert result.combined_adjustment_factor >= 0.1


# ---------------------------------------------------------------------------
# Contraindication logic
# ---------------------------------------------------------------------------


class TestContraindication:
    def test_contraindicated_severe_dual_impairment(self):
        result = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 1.0, 1.0, egfr_ml_per_min=5.0, child_pugh_score=12
        )
        # combined_factor would be very low → contraindicated
        assert result.contraindicated is True

    def test_not_contraindicated_normal(self):
        result = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.5, 0.3, egfr_ml_per_min=100.0, child_pugh_score=5
        )
        assert result.contraindicated is False

    def test_monitoring_required_below_0_7(self):
        result = calculate_organ_impairment_dose(
            "Drug", 100.0, 12.0, 0.9, 0.5, egfr_ml_per_min=20.0, child_pugh_score=9
        )
        assert result.monitoring_required is True


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fe_urine_too_high_raises(self):
        with pytest.raises(ValueError):
            calculate_organ_impairment_dose("Drug", 100.0, 12.0, 1.5, 0.5)

    def test_fe_urine_negative_raises(self):
        with pytest.raises(ValueError):
            calculate_organ_impairment_dose("Drug", 100.0, 12.0, -0.1, 0.5)

    def test_fm_hepatic_too_high_raises(self):
        with pytest.raises(ValueError):
            calculate_organ_impairment_dose("Drug", 100.0, 12.0, 0.5, 1.5)

    def test_fm_hepatic_negative_raises(self):
        with pytest.raises(ValueError):
            calculate_organ_impairment_dose("Drug", 100.0, 12.0, 0.5, -0.2)


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenOrganImpairment:
    def test_screen_returns_list(self):
        scenarios = [
            {"egfr_ml_per_min": 100.0, "child_pugh_score": 5},
            {"egfr_ml_per_min": 30.0, "child_pugh_score": 8},
            {"egfr_ml_per_min": 10.0, "child_pugh_score": 12},
        ]
        results = screen_organ_impairment("Drug", 100.0, 12.0, 0.6, 0.5, scenarios)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_screen_results_ordered(self):
        """More severe impairment → lower dose."""
        scenarios = [
            {"egfr_ml_per_min": 100.0, "child_pugh_score": 5},
            {"egfr_ml_per_min": 15.0, "child_pugh_score": 12},
        ]
        results = screen_organ_impairment("Drug", 100.0, 12.0, 0.6, 0.5, scenarios)
        assert results[1].recommended_dose_mg < results[0].recommended_dose_mg

    def test_screen_empty_scenarios(self):
        results = screen_organ_impairment("Drug", 100.0, 12.0, 0.5, 0.3, [])
        assert results == []
