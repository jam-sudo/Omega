"""Tests for prediction/drug_stability_predictor.py — Phase 453."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.drug_stability_predictor import (
    StabilityStudyResult,
    accelerated_stability_study,
    arrhenius_rate,
    predict_shelf_life,
    q10_extrapolation,
)


class TestArrheniusRate:
    def test_same_temperature_returns_k_ref(self):
        """Same T as reference → rate should equal k_ref."""
        k = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_ref_C=25.0, t_C=25.0)
        assert math.isclose(k, 0.01, rel_tol=1e-9)

    def test_higher_temperature_gives_higher_rate(self):
        """40°C should give higher rate than 25°C."""
        k_25 = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_C=25.0)
        k_40 = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_C=40.0)
        assert k_40 > k_25

    def test_lower_temperature_gives_lower_rate(self):
        """4°C (refrigerator) should give lower rate than 25°C."""
        k_25 = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_C=25.0)
        k_4 = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_C=4.0)
        assert k_4 < k_25

    def test_higher_ea_more_temperature_sensitive(self):
        """Higher Ea → larger ratio k(40)/k(25)."""
        ratio_low_ea = (
            arrhenius_rate(0.01, ea_kJ_mol=50.0, t_C=40.0)
            / arrhenius_rate(0.01, ea_kJ_mol=50.0, t_C=25.0)
        )
        ratio_high_ea = (
            arrhenius_rate(0.01, ea_kJ_mol=120.0, t_C=40.0)
            / arrhenius_rate(0.01, ea_kJ_mol=120.0, t_C=25.0)
        )
        assert ratio_high_ea > ratio_low_ea

    def test_rate_positive(self):
        assert arrhenius_rate(0.005, ea_kJ_mol=70.0, t_C=60.0) > 0

    def test_k_ref_scales_result(self):
        """Doubling k_ref should double result."""
        k1 = arrhenius_rate(0.01, ea_kJ_mol=80.0, t_C=40.0)
        k2 = arrhenius_rate(0.02, ea_kJ_mol=80.0, t_C=40.0)
        assert math.isclose(k2, 2 * k1, rel_tol=1e-9)

    def test_invalid_k_ref_raises(self):
        with pytest.raises(ValueError, match="k_ref"):
            arrhenius_rate(0.0, ea_kJ_mol=80.0)

    def test_invalid_negative_k_ref_raises(self):
        with pytest.raises(ValueError, match="k_ref"):
            arrhenius_rate(-0.01, ea_kJ_mol=80.0)

    def test_invalid_ea_raises(self):
        with pytest.raises(ValueError, match="ea_kJ_mol"):
            arrhenius_rate(0.01, ea_kJ_mol=0.0)

    def test_negative_ea_raises(self):
        with pytest.raises(ValueError, match="ea_kJ_mol"):
            arrhenius_rate(0.01, ea_kJ_mol=-50.0)


class TestPredictShelfLife:
    def test_first_order_formula(self):
        """First-order shelf life = ln(100/(100-5)) / k at T=T_ref."""
        k_ref = 0.05  # per month at 25°C
        expected = math.log(100.0 / 95.0) / k_ref
        result = predict_shelf_life(k_ref, ea_kJ_mol=80.0, t_C=25.0, t_ref_C=25.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_higher_temperature_shorter_shelf_life(self):
        sl_25 = predict_shelf_life(0.01, ea_kJ_mol=80.0, t_C=25.0)
        sl_40 = predict_shelf_life(0.01, ea_kJ_mol=80.0, t_C=40.0)
        assert sl_40 < sl_25

    def test_larger_degradation_limit_longer_shelf_life(self):
        """Allowing 10% degradation gives longer shelf life than 5%."""
        sl_5 = predict_shelf_life(0.01, ea_kJ_mol=80.0, degradation_limit_pct=5.0)
        sl_10 = predict_shelf_life(0.01, ea_kJ_mol=80.0, degradation_limit_pct=10.0)
        assert sl_10 > sl_5

    def test_zero_order(self):
        """Zero-order shelf life = limit / k."""
        k_ref = 0.10
        limit = 5.0
        # At T_ref same as T, k_at_T = k_ref
        expected = limit / k_ref
        result = predict_shelf_life(
            k_ref, ea_kJ_mol=80.0, degradation_limit_pct=limit,
            t_C=25.0, t_ref_C=25.0, order=0
        )
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_positive_shelf_life(self):
        sl = predict_shelf_life(0.001, ea_kJ_mol=60.0)
        assert sl > 0

    def test_invalid_degradation_limit_zero(self):
        with pytest.raises(ValueError, match="degradation_limit_pct"):
            predict_shelf_life(0.01, ea_kJ_mol=80.0, degradation_limit_pct=0.0)

    def test_invalid_degradation_limit_100(self):
        with pytest.raises(ValueError, match="degradation_limit_pct"):
            predict_shelf_life(0.01, ea_kJ_mol=80.0, degradation_limit_pct=100.0)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order"):
            predict_shelf_life(0.01, ea_kJ_mol=80.0, order=2)


class TestQ10Extrapolation:
    def test_q10_at_same_temperature_returns_original(self):
        """At target=25°C, result equals t_half_25C_months."""
        result = q10_extrapolation(24.0, q10=2.0, target_temp_C=25.0)
        assert math.isclose(result, 24.0, rel_tol=1e-9)

    def test_10c_increase_halves_shelf_life(self):
        """With Q10=2.0 and +10°C, shelf life should halve."""
        result = q10_extrapolation(24.0, q10=2.0, target_temp_C=35.0)
        assert math.isclose(result, 12.0, rel_tol=1e-6)

    def test_20c_increase_quarters_shelf_life(self):
        """With Q10=2.0 and +20°C, shelf life quarters."""
        result = q10_extrapolation(24.0, q10=2.0, target_temp_C=45.0)
        assert math.isclose(result, 6.0, rel_tol=1e-6)

    def test_negative_delta_t_extends_shelf_life(self):
        """Cooling below 25°C extends shelf life."""
        result = q10_extrapolation(24.0, q10=2.0, target_temp_C=15.0)
        assert result > 24.0

    def test_refrigerator_q10(self):
        """4°C should significantly extend shelf life."""
        result = q10_extrapolation(24.0, q10=2.0, target_temp_C=4.0)
        assert result > 24.0 * 3

    def test_custom_q10(self):
        """Q10=3 should give t_shelf/3 at +10°C."""
        result = q10_extrapolation(30.0, q10=3.0, target_temp_C=35.0)
        assert math.isclose(result, 10.0, rel_tol=1e-6)

    def test_invalid_t_half_raises(self):
        with pytest.raises(ValueError, match="t_half_25C_months"):
            q10_extrapolation(0.0, q10=2.0)

    def test_invalid_q10_raises(self):
        with pytest.raises(ValueError, match="q10"):
            q10_extrapolation(24.0, q10=0.0)


class TestAcceleratedStabilityStudy:
    def test_returns_stability_study_result(self):
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0)
        assert isinstance(result, StabilityStudyResult)

    def test_default_temperatures_all_present(self):
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0)
        assert 25.0 in result.temperatures_C
        assert 40.0 in result.temperatures_C

    def test_custom_temperatures_returned(self):
        temps = [25.0, 40.0, 60.0]
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0, temperatures_C=temps)
        assert result.temperatures_C == temps

    def test_rate_constants_length_matches_temperatures(self):
        temps = [25.0, 30.0, 40.0]
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0, temperatures_C=temps)
        assert len(result.rate_constants) == 3

    def test_shelf_life_months_length_matches_temperatures(self):
        temps = [25.0, 40.0]
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0, temperatures_C=temps)
        assert len(result.shelf_life_months) == 2

    def test_higher_temperature_higher_rate_constant(self):
        """Rate constants should increase with temperature."""
        result = accelerated_stability_study(
            0.01, ea_kJ_mol=80.0, temperatures_C=[25.0, 40.0, 60.0]
        )
        assert result.rate_constants[0] < result.rate_constants[1] < result.rate_constants[2]

    def test_higher_temperature_lower_shelf_life(self):
        """Shelf life should decrease with temperature."""
        result = accelerated_stability_study(
            0.01, ea_kJ_mol=80.0, temperatures_C=[25.0, 40.0, 60.0]
        )
        assert (
            result.shelf_life_months[0]
            > result.shelf_life_months[1]
            > result.shelf_life_months[2]
        )

    def test_humidity_correction_increases_rate(self):
        r1 = accelerated_stability_study(0.01, ea_kJ_mol=80.0)
        r2 = accelerated_stability_study(0.01, ea_kJ_mol=80.0, humidity_correction=1.5)
        assert r2.rate_constants[0] > r1.rate_constants[0]

    def test_humidity_correction_in_notes(self):
        result = accelerated_stability_study(
            0.01, ea_kJ_mol=80.0, humidity_correction=1.5
        )
        assert "humidity" in result.notes.lower()

    def test_q10_factor_stored(self):
        result = accelerated_stability_study(0.01, ea_kJ_mol=80.0)
        assert result.q10_factor > 1.0

    def test_ea_stored(self):
        result = accelerated_stability_study(0.01, ea_kJ_mol=95.0)
        assert math.isclose(result.ea_kJ_mol, 95.0)

    def test_k_ref_stored(self):
        result = accelerated_stability_study(0.02, ea_kJ_mol=80.0)
        assert math.isclose(result.k_ref, 0.02)

    def test_invalid_humidity_correction_raises(self):
        with pytest.raises(ValueError, match="humidity_correction"):
            accelerated_stability_study(0.01, ea_kJ_mol=80.0, humidity_correction=0.5)

    def test_empty_temperatures_raises(self):
        with pytest.raises(ValueError, match="temperatures_C"):
            accelerated_stability_study(0.01, ea_kJ_mol=80.0, temperatures_C=[])
