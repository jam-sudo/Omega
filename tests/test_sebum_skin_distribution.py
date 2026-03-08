"""Tests for sebum and skin appendage distribution prediction (Phase 909)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.sebum_skin_distribution import (
    SebumSkinDistributionResult,
    predict_sebum_distribution,
    screen_follicular_drugs,
)


class TestSebumSkinDistributionResult:
    """Tests for the SebumSkinDistributionResult dataclass."""

    def test_frozen_dataclass(self):
        result = predict_sebum_distribution("TestDrug", logp=3.0)
        with pytest.raises((AttributeError, TypeError)):
            result.logp = 99.0

    def test_fields_present(self):
        result = predict_sebum_distribution("TestDrug", logp=3.0)
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "sebum_partition_coefficient")
        assert hasattr(result, "follicular_penetration_fraction")
        assert hasattr(result, "sebum_reservoir_t_half_h")
        assert hasattr(result, "topical_penetration_enhancement")
        assert hasattr(result, "acne_target_conc_achievable")
        assert hasattr(result, "predicted_sebum_conc_ug_g")
        assert hasattr(result, "plasma_conc_input_mg_L")
        assert hasattr(result, "notes")


class TestPredictSebumDistribution:
    """Tests for predict_sebum_distribution()."""

    def test_basic_call_returns_result(self):
        result = predict_sebum_distribution("Doxycycline", logp=0.0)
        assert isinstance(result, SebumSkinDistributionResult)
        assert result.drug_name == "Doxycycline"

    def test_logp3_partition_coefficient(self):
        result = predict_sebum_distribution("Drug", logp=3.0)
        expected = 10 ** (3.0 * 0.5)
        assert abs(result.sebum_partition_coefficient - expected) < 0.01

    def test_logp5_partition_coefficient(self):
        result = predict_sebum_distribution("Drug", logp=5.0)
        expected = min(1000.0, 10 ** (5.0 * 0.5))
        assert abs(result.sebum_partition_coefficient - expected) < 0.01

    def test_partition_coefficient_clamped_max(self):
        result = predict_sebum_distribution("Drug", logp=20.0)
        assert result.sebum_partition_coefficient == 1000.0

    def test_partition_coefficient_clamped_min(self):
        result = predict_sebum_distribution("Drug", logp=-10.0)
        assert result.sebum_partition_coefficient == 1.0

    def test_follicular_penetration_low_mw_positive_logp(self):
        result = predict_sebum_distribution("Drug", logp=3.0, mw_Da=300.0)
        expected = min(0.4, max(0.01, 0.1 + 3.0 * 0.05))
        assert abs(result.follicular_penetration_fraction - expected) < 1e-9

    def test_follicular_penetration_high_mw_scaled(self):
        result = predict_sebum_distribution("Drug", logp=3.0, mw_Da=600.0)
        base = min(0.4, max(0.01, 0.1 + 3.0 * 0.05))
        expected = max(0.01, base * 0.3)
        assert abs(result.follicular_penetration_fraction - expected) < 1e-9

    def test_follicular_penetration_clamped_max(self):
        result = predict_sebum_distribution("Drug", logp=10.0, mw_Da=200.0)
        assert result.follicular_penetration_fraction <= 0.4

    def test_follicular_penetration_clamped_min(self):
        result = predict_sebum_distribution("Drug", logp=-5.0, mw_Da=600.0)
        assert result.follicular_penetration_fraction >= 0.01

    def test_sebum_reservoir_half_life(self):
        result = predict_sebum_distribution("Drug", logp=3.0)
        pc = result.sebum_partition_coefficient
        expected = min(72.0, max(0.5, pc * 0.5))
        assert abs(result.sebum_reservoir_t_half_h - expected) < 0.001

    def test_sebum_reservoir_half_life_clamped_min(self):
        result = predict_sebum_distribution("Drug", logp=-10.0)
        assert result.sebum_reservoir_t_half_h == 0.5

    def test_sebum_reservoir_half_life_clamped_max(self):
        result = predict_sebum_distribution("Drug", logp=20.0)
        assert result.sebum_reservoir_t_half_h == 72.0

    def test_topical_penetration_enhancement(self):
        result = predict_sebum_distribution("Drug", logp=3.0)
        expected = 1.0 + result.follicular_penetration_fraction * 3.0
        assert abs(result.topical_penetration_enhancement - expected) < 1e-9

    def test_predicted_sebum_conc_formula(self):
        result = predict_sebum_distribution("Drug", logp=3.0, plasma_conc_mg_L=0.1)
        expected = result.sebum_partition_coefficient * 0.1 * 1000.0
        assert abs(result.predicted_sebum_conc_ug_g - expected) < 0.01

    def test_predicted_sebum_conc_clamped_max(self):
        result = predict_sebum_distribution("Drug", logp=20.0, plasma_conc_mg_L=1000.0)
        assert result.predicted_sebum_conc_ug_g <= 1e6

    def test_acne_target_achievable_true(self):
        result = predict_sebum_distribution("Drug", logp=3.0, plasma_conc_mg_L=1.0)
        assert result.acne_target_conc_achievable is True

    def test_acne_target_achievable_false(self):
        result = predict_sebum_distribution("Drug", logp=-5.0, plasma_conc_mg_L=0.0001)
        assert result.acne_target_conc_achievable is False

    def test_notes_acne_achievable(self):
        result = predict_sebum_distribution("Drug", logp=3.0, plasma_conc_mg_L=1.0)
        assert "acne" in result.notes.lower() or "folliculitis" in result.notes.lower()

    def test_notes_moderate_when_logp_low(self):
        result = predict_sebum_distribution("Drug", logp=1.0, plasma_conc_mg_L=0.0001)
        assert "moderate" in result.notes.lower()

    def test_notes_logp_gt4_not_achievable(self):
        # logp > 4 but plasma conc too low for acne target
        result = predict_sebum_distribution("Drug", logp=4.5, plasma_conc_mg_L=1e-10)
        assert "prolonged" in result.notes.lower() or "once-daily" in result.notes.lower()

    def test_validation_mw_zero(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_sebum_distribution("Drug", mw_Da=0.0)

    def test_validation_mw_negative(self):
        with pytest.raises(ValueError):
            predict_sebum_distribution("Drug", mw_Da=-1.0)

    def test_validation_plasma_conc_zero(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_sebum_distribution("Drug", plasma_conc_mg_L=0.0)

    def test_plasma_conc_stored(self):
        result = predict_sebum_distribution("Drug", plasma_conc_mg_L=0.5)
        assert result.plasma_conc_input_mg_L == 0.5


class TestScreenFollicularDrugs:
    """Tests for screen_follicular_drugs()."""

    def test_basic_screening(self):
        candidates = [
            {"name": "DrugA", "logp": 5.0},
            {"name": "DrugB", "logp": 2.0},
            {"name": "DrugC", "logp": 3.0},
        ]
        results = screen_follicular_drugs(candidates)
        assert len(results) == 3

    def test_sorted_by_sebum_partition_coefficient_descending(self):
        candidates = [
            {"name": "Low", "logp": 1.0},
            {"name": "High", "logp": 5.0},
            {"name": "Mid", "logp": 3.0},
        ]
        results = screen_follicular_drugs(candidates)
        pcs = [r.sebum_partition_coefficient for r in results]
        assert pcs == sorted(pcs, reverse=True)

    def test_optional_mw_default(self):
        candidates = [{"name": "Drug", "logp": 3.0}]
        results = screen_follicular_drugs(candidates)
        assert results[0].mw_Da == 350.0

    def test_optional_mw_provided(self):
        candidates = [{"name": "Drug", "logp": 3.0, "mw_Da": 500.0}]
        results = screen_follicular_drugs(candidates)
        assert results[0].mw_Da == 500.0

    def test_empty_list(self):
        results = screen_follicular_drugs([])
        assert results == []
