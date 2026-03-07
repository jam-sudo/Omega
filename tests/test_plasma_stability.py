"""Tests for plasma stability prediction module (Phase 892)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.prediction.plasma_stability import (
    PlasmaStabilityResult,
    predict_plasma_stability,
    screen_plasma_stability,
)


class TestPredictPlasmaStability:
    """Tests for predict_plasma_stability function."""

    def test_returns_dataclass_instance(self):
        result = predict_plasma_stability("StableDrug")
        assert isinstance(result, PlasmaStabilityResult)

    def test_drug_name_stored(self):
        result = predict_plasma_stability("TestDrug")
        assert result.drug_name == "TestDrug"

    def test_logp_stored(self):
        result = predict_plasma_stability("X", logp=3.5)
        assert result.logp == pytest.approx(3.5)

    def test_no_structural_alerts_stable(self):
        result = predict_plasma_stability("Stable", logp=2.0)
        assert result.structural_alerts == []
        assert result.hydrolysis_susceptibility == "low"
        assert result.plasma_t_half_h == pytest.approx(100.0)

    def test_ester_alert_added(self):
        result = predict_plasma_stability("X", has_ester=True)
        assert "ester" in result.structural_alerts

    def test_amide_alert_added(self):
        result = predict_plasma_stability("X", has_amide=True)
        assert "amide" in result.structural_alerts

    def test_lactone_alert_added(self):
        result = predict_plasma_stability("X", has_lactone=True)
        assert "lactone" in result.structural_alerts

    def test_carbonate_alert_added(self):
        result = predict_plasma_stability("X", has_carbonate=True)
        assert "carbonate" in result.structural_alerts

    def test_thioester_alert_added(self):
        result = predict_plasma_stability("X", has_thioester=True)
        assert "thioester" in result.structural_alerts

    def test_esterase_lability_score_base(self):
        # No flags, logp=2.0 → base 10 + 0 modifier = 10
        result = predict_plasma_stability("X", logp=2.0)
        assert result.esterase_lability_score == pytest.approx(10.0)

    def test_esterase_lability_score_with_ester(self):
        # base 10 + 35 = 45, logp=2.0 → no modifier
        result = predict_plasma_stability("X", logp=2.0, has_ester=True)
        assert result.esterase_lability_score == pytest.approx(45.0)

    def test_esterase_lability_score_clamped_to_100(self):
        # All flags + high logp → clamped to 100
        result = predict_plasma_stability(
            "X", logp=10.0, has_ester=True, has_thioester=True,
            has_lactone=True, has_carbonate=True, has_amide=True
        )
        assert result.esterase_lability_score == pytest.approx(100.0)

    def test_esterase_lability_score_clamped_to_0(self):
        # Very low logp → score may go below 0
        result = predict_plasma_stability("X", logp=-10.0)
        assert result.esterase_lability_score >= 0.0

    def test_plasma_t_half_ester_formula(self):
        logp = 2.0
        expected_t_half = max(0.5, 20.0 / (1.0 + logp * 0.2))
        result = predict_plasma_stability("X", logp=logp, has_ester=True)
        assert result.plasma_t_half_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_thioester_caps_t_half(self):
        result = predict_plasma_stability("X", has_thioester=True, logp=2.0)
        assert result.plasma_t_half_h <= 5.0

    def test_lactone_caps_t_half(self):
        result = predict_plasma_stability("X", has_lactone=True)
        assert result.plasma_t_half_h <= 10.0

    def test_stability_percentages_first_order(self):
        result = predict_plasma_stability("X", logp=2.0)
        t_half = result.plasma_t_half_h
        k = math.log(2) / t_half
        expected_1h = 100.0 * math.exp(-k * 1.0)
        expected_4h = 100.0 * math.exp(-k * 4.0)
        assert result.stability_at_37C_pct_1h == pytest.approx(expected_1h, rel=1e-6)
        assert result.stability_at_37C_pct_4h == pytest.approx(expected_4h, rel=1e-6)

    def test_high_lability_storage_temperature(self):
        # has_ester + thioester → score well above 60
        result = predict_plasma_stability("X", logp=5.0, has_ester=True, has_thioester=True)
        assert result.esterase_lability_score > 60.0
        assert result.recommended_storage_temp_C == pytest.approx(-80.0)
        assert result.hydrolysis_susceptibility == "high"

    def test_moderate_lability_storage_temperature(self):
        # has_ester alone at logp=2.0 → score ~45
        result = predict_plasma_stability("X", logp=2.0, has_ester=True)
        assert 30.0 < result.esterase_lability_score <= 60.0
        assert result.recommended_storage_temp_C == pytest.approx(-20.0)
        assert result.hydrolysis_susceptibility == "moderate"

    def test_low_lability_storage_temperature(self):
        result = predict_plasma_stability("X", logp=2.0)
        assert result.esterase_lability_score <= 30.0
        assert result.recommended_storage_temp_C == pytest.approx(4.0)
        assert result.hydrolysis_susceptibility == "low"

    def test_high_lability_handling_notes(self):
        result = predict_plasma_stability("X", logp=5.0, has_ester=True, has_thioester=True)
        assert "PMSF" in result.in_vitro_handling_notes or "esterase inhibitor" in result.in_vitro_handling_notes

    def test_moderate_lability_handling_notes(self):
        result = predict_plasma_stability("X", logp=2.0, has_ester=True)
        assert "ice" in result.in_vitro_handling_notes.lower()

    def test_low_lability_handling_notes(self):
        result = predict_plasma_stability("X", logp=2.0)
        assert "Standard handling" in result.in_vitro_handling_notes

    def test_high_susceptibility_notes(self):
        result = predict_plasma_stability("X", logp=5.0, has_ester=True, has_thioester=True)
        assert "Extensive plasma hydrolysis" in result.notes

    def test_moderate_susceptibility_notes(self):
        result = predict_plasma_stability("X", logp=2.0, has_ester=True)
        assert "Moderate stability" in result.notes

    def test_low_susceptibility_notes(self):
        result = predict_plasma_stability("X", logp=2.0)
        assert "Good plasma stability" in result.notes

    def test_frozen_dataclass(self):
        result = predict_plasma_stability("X")
        with pytest.raises(Exception):
            result.drug_name = "Y"  # type: ignore[misc]


class TestScreenPlasmaStability:
    """Tests for screen_plasma_stability function."""

    def test_returns_list(self):
        candidates = [
            {"name": "A", "logp": 2.0},
            {"name": "B", "logp": 3.0, "has_ester": True},
        ]
        results = screen_plasma_stability(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_t_half_descending(self):
        candidates = [
            {"name": "Labile", "logp": 2.0, "has_ester": True, "has_thioester": True},
            {"name": "Stable", "logp": 2.0},
            {"name": "Mid", "logp": 2.0, "has_ester": True},
        ]
        results = screen_plasma_stability(candidates)
        t_halves = [r.plasma_t_half_h for r in results]
        assert t_halves == sorted(t_halves, reverse=True)

    def test_optional_flags_default_false(self):
        candidates = [{"name": "X", "logp": 2.0}]
        result = screen_plasma_stability(candidates)[0]
        assert result.structural_alerts == []

    def test_optional_flags_overridden(self):
        candidates = [{"name": "X", "logp": 2.0, "has_ester": True, "has_amide": True}]
        result = screen_plasma_stability(candidates)[0]
        assert "ester" in result.structural_alerts
        assert "amide" in result.structural_alerts

    def test_empty_list(self):
        assert screen_plasma_stability([]) == []
