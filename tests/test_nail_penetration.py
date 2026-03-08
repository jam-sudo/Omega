"""Tests for nail penetration prediction (Phase 910)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.nail_penetration import (
    NailPenetrationResult,
    compare_nail_formulations,
    predict_nail_penetration,
)

_NAIL_THICKNESS_CM = 0.05


class TestNailPenetrationResult:
    """Tests for the NailPenetrationResult dataclass."""

    def test_frozen_dataclass(self):
        result = predict_nail_penetration("Terbinafine")
        with pytest.raises((AttributeError, TypeError)):
            result.logp = 99.0

    def test_fields_present(self):
        result = predict_nail_penetration("Terbinafine")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "formulation")
        assert hasattr(result, "nail_permeability_coefficient_cm_h")
        assert hasattr(result, "predicted_nail_conc_ug_g")
        assert hasattr(result, "penetration_depth_fraction")
        assert hasattr(result, "antifungal_target_achievable")
        assert hasattr(result, "estimated_treatment_duration_weeks")
        assert hasattr(result, "nail_retention_score")
        assert hasattr(result, "notes")


class TestPredictNailPenetration:
    """Tests for predict_nail_penetration()."""

    def test_basic_call_returns_result(self):
        result = predict_nail_penetration("Terbinafine", logp=4.0, mw_Da=300.0)
        assert isinstance(result, NailPenetrationResult)
        assert result.drug_name == "Terbinafine"

    def test_formulation_stored(self):
        result = predict_nail_penetration("Drug", formulation="patch")
        assert result.formulation == "patch"

    def test_kp_formulation_lacquer_modifier(self):
        result_lac = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="lacquer")
        result_sol = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="solution")
        assert (
            abs(
                result_lac.nail_permeability_coefficient_cm_h
                / result_sol.nail_permeability_coefficient_cm_h
                - 1.5
            )
            < 0.01
        )

    def test_kp_formulation_patch_modifier(self):
        result_pat = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="patch")
        result_sol = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="solution")
        assert (
            abs(
                result_pat.nail_permeability_coefficient_cm_h
                / result_sol.nail_permeability_coefficient_cm_h
                - 2.0
            )
            < 0.01
        )

    def test_kp_formulation_cream_modifier(self):
        result_cre = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="cream")
        result_sol = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="solution")
        assert (
            abs(
                result_cre.nail_permeability_coefficient_cm_h
                / result_sol.nail_permeability_coefficient_cm_h
                - 0.5
            )
            < 0.01
        )

    def test_kp_clamped_min(self):
        result = predict_nail_penetration("Drug", logp=-20.0, mw_Da=3000.0)
        assert result.nail_permeability_coefficient_cm_h >= 1e-7

    def test_kp_clamped_max(self):
        result = predict_nail_penetration("Drug", logp=20.0, mw_Da=1.0, formulation="patch")
        assert result.nail_permeability_coefficient_cm_h <= 1e-2

    def test_predicted_nail_conc_formula(self):
        result = predict_nail_penetration(
            "Drug", logp=4.0, mw_Da=300.0, formulation="solution", application_dose_mg_cm2=1.0
        )
        flux = result.nail_permeability_coefficient_cm_h * 1.0
        expected = flux * 168.0 * 1000.0
        assert abs(result.predicted_nail_conc_ug_g - expected) < 0.01

    def test_predicted_nail_conc_clamped_min(self):
        result = predict_nail_penetration("Drug", logp=-30.0, mw_Da=5000.0)
        assert result.predicted_nail_conc_ug_g >= 0.001

    def test_predicted_nail_conc_clamped_max(self):
        result = predict_nail_penetration(
            "Drug", logp=20.0, mw_Da=1.0, formulation="patch", application_dose_mg_cm2=1e6
        )
        assert result.predicted_nail_conc_ug_g <= 1e5

    def test_penetration_depth_fraction_formula(self):
        result = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="solution")
        kp = result.nail_permeability_coefficient_cm_h
        expected = min(1.0, kp * 168.0 / _NAIL_THICKNESS_CM)
        assert abs(result.penetration_depth_fraction - expected) < 1e-9

    def test_penetration_depth_fraction_max_one(self):
        result = predict_nail_penetration("Drug", logp=20.0, mw_Da=1.0, formulation="patch")
        assert result.penetration_depth_fraction <= 1.0

    def test_antifungal_target_achievable_true(self):
        # High logp + patch should achieve high nail conc
        result = predict_nail_penetration(
            "Drug", logp=5.0, mw_Da=200.0, formulation="patch", application_dose_mg_cm2=5.0
        )
        assert result.antifungal_target_achievable is True

    def test_antifungal_target_achievable_false(self):
        # Very low kp
        result = predict_nail_penetration("Drug", logp=-20.0, mw_Da=3000.0, formulation="cream")
        assert result.antifungal_target_achievable is False

    def test_estimated_treatment_duration_min(self):
        result = predict_nail_penetration("Drug", logp=20.0, mw_Da=1.0, formulation="patch")
        assert result.estimated_treatment_duration_weeks >= 4

    def test_estimated_treatment_duration_formula(self):
        result = predict_nail_penetration("Drug", logp=2.0, mw_Da=400.0, formulation="solution")
        depth = max(0.01, result.penetration_depth_fraction)
        expected = max(4, int(12.0 / depth))
        assert result.estimated_treatment_duration_weeks == expected

    def test_nail_retention_score_lacquer_bonus(self):
        result = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="lacquer")
        expected = min(100.0, max(0.0, 4.0 * 10.0 + 50))
        assert abs(result.nail_retention_score - expected) < 0.01

    def test_nail_retention_score_cream_bonus(self):
        result = predict_nail_penetration("Drug", logp=4.0, mw_Da=300.0, formulation="cream")
        expected = min(100.0, max(0.0, 4.0 * 10.0 + 20))
        assert abs(result.nail_retention_score - expected) < 0.01

    def test_nail_retention_score_clamped(self):
        result = predict_nail_penetration("Drug", logp=20.0, mw_Da=300.0, formulation="patch")
        assert result.nail_retention_score <= 100.0
        assert result.nail_retention_score >= 0.0

    def test_notes_antifungal_achievable(self):
        result = predict_nail_penetration(
            "Drug", logp=5.0, mw_Da=200.0, formulation="patch", application_dose_mg_cm2=5.0
        )
        assert "onychomycosis" in result.notes.lower()

    def test_notes_occlusive_formulation(self):
        # Low logp so antifungal not achievable, but lacquer
        result = predict_nail_penetration("Drug", logp=-20.0, mw_Da=3000.0, formulation="lacquer")
        assert "occlusive" in result.notes.lower()

    def test_notes_standard_penetration(self):
        result = predict_nail_penetration("Drug", logp=-20.0, mw_Da=3000.0, formulation="solution")
        assert "lacquer" in result.notes.lower() or "standard" in result.notes.lower()

    def test_validation_mw_zero(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_nail_penetration("Drug", mw_Da=0.0)

    def test_validation_invalid_formulation(self):
        with pytest.raises(ValueError, match="formulation"):
            predict_nail_penetration("Drug", formulation="gel")

    def test_validation_dose_zero(self):
        with pytest.raises(ValueError, match="application_dose_mg_cm2"):
            predict_nail_penetration("Drug", application_dose_mg_cm2=0.0)


class TestCompareNailFormulations:
    """Tests for compare_nail_formulations()."""

    def test_returns_four_results(self):
        results = compare_nail_formulations("Terbinafine", logp=4.0, mw_Da=300.0)
        assert len(results) == 4

    def test_all_formulations_present(self):
        results = compare_nail_formulations("Terbinafine", logp=4.0, mw_Da=300.0)
        forms = {r.formulation for r in results}
        assert forms == {"lacquer", "solution", "patch", "cream"}

    def test_sorted_by_predicted_nail_conc_descending(self):
        results = compare_nail_formulations("Terbinafine", logp=4.0, mw_Da=300.0)
        concs = [r.predicted_nail_conc_ug_g for r in results]
        assert concs == sorted(concs, reverse=True)

    def test_patch_highest_concentration(self):
        results = compare_nail_formulations("Terbinafine", logp=4.0, mw_Da=300.0)
        # patch has modifier 2.0, highest among all
        assert results[0].formulation == "patch"
