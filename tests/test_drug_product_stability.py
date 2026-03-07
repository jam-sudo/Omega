"""Tests for prediction/drug_product_stability.py — Phase 546."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.drug_product_stability import (
    ICHStabilityResult,
    arrhenius_rate,
    degradation_profile,
    ich_stability_prediction,
    q10_extrapolation,
    shelf_life_first_order,
)

# ---------------------------------------------------------------------------
# arrhenius_rate
# ---------------------------------------------------------------------------


class TestArrheniusRate:
    def test_higher_temperature_gives_higher_k(self):
        k_low = arrhenius_rate(1e-4, 80.0, 25.0, 25.0)
        k_high = arrhenius_rate(1e-4, 80.0, 25.0, 40.0)
        assert k_high > k_low

    def test_at_reference_temperature_k_unchanged(self):
        k = arrhenius_rate(1e-4, 80.0, 25.0, 25.0)
        assert k == pytest.approx(1e-4, rel=1e-9)

    def test_lower_temperature_gives_lower_k(self):
        k_ref = arrhenius_rate(1e-4, 80.0, 25.0, 25.0)
        k_low = arrhenius_rate(1e-4, 80.0, 25.0, 10.0)
        assert k_low < k_ref

    def test_higher_ea_amplifies_temperature_effect(self):
        """Higher Ea → bigger ratio of k_high/k_low."""
        k_hi_ea = arrhenius_rate(1e-4, 120.0, 25.0, 40.0) / arrhenius_rate(1e-4, 120.0, 25.0, 25.0)
        k_lo_ea = arrhenius_rate(1e-4, 50.0, 25.0, 40.0) / arrhenius_rate(1e-4, 50.0, 25.0, 25.0)
        assert k_hi_ea > k_lo_ea

    def test_manual_calculation(self):
        k_ref = 1e-4
        ea = 80.0  # kJ/mol
        t_ref = 25.0 + 273.15
        t_store = 40.0 + 273.15
        expected = k_ref * math.exp(-(ea * 1000 / 8.314) * (1 / t_store - 1 / t_ref))
        result = arrhenius_rate(1e-4, 80.0, 25.0, 40.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_invalid_k_ref_zero(self):
        with pytest.raises(ValueError, match="k_ref"):
            arrhenius_rate(0.0, 80.0, 25.0, 40.0)

    def test_invalid_k_ref_negative(self):
        with pytest.raises(ValueError, match="k_ref"):
            arrhenius_rate(-1e-4, 80.0, 25.0, 40.0)

    def test_invalid_ea_zero(self):
        with pytest.raises(ValueError, match="ea_kJ_per_mol"):
            arrhenius_rate(1e-4, 0.0, 25.0, 40.0)


# ---------------------------------------------------------------------------
# shelf_life_first_order
# ---------------------------------------------------------------------------


class TestShelfLifeFirstOrder:
    def test_lower_k_gives_longer_shelf_life(self):
        t1 = shelf_life_first_order(1e-4)
        t2 = shelf_life_first_order(1e-3)
        assert t1 > t2

    def test_manual_calculation(self):
        k = 1e-4
        limit = 90.0
        expected = -math.log(limit / 100.0) / k
        result = shelf_life_first_order(k, limit)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_default_potency_limit_90(self):
        k = 1e-4
        t_default = shelf_life_first_order(k)
        t_explicit = shelf_life_first_order(k, 90.0)
        assert t_default == pytest.approx(t_explicit)

    def test_stricter_limit_shorter_shelf_life(self):
        t_95 = shelf_life_first_order(1e-4, 95.0)
        t_90 = shelf_life_first_order(1e-4, 90.0)
        assert t_95 < t_90

    def test_positive_shelf_life(self):
        t = shelf_life_first_order(1e-4, 90.0)
        assert t > 0.0

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k_per_h"):
            shelf_life_first_order(0.0)

    def test_invalid_potency_limit_100(self):
        with pytest.raises(ValueError, match="potency_limit_pct"):
            shelf_life_first_order(1e-4, 100.0)

    def test_invalid_potency_limit_zero(self):
        with pytest.raises(ValueError, match="potency_limit_pct"):
            shelf_life_first_order(1e-4, 0.0)


# ---------------------------------------------------------------------------
# q10_extrapolation
# ---------------------------------------------------------------------------


class TestQ10Extrapolation:
    def test_10c_increase_halves_shelf_life(self):
        t_ref = 24.0  # months
        t_result = q10_extrapolation(t_ref, 25.0, 35.0, q10=2.0)
        assert t_result == pytest.approx(t_ref / 2.0, rel=1e-9)

    def test_10c_decrease_doubles_shelf_life(self):
        t_ref = 24.0
        t_result = q10_extrapolation(t_ref, 25.0, 15.0, q10=2.0)
        assert t_result == pytest.approx(t_ref * 2.0, rel=1e-9)

    def test_same_temperature_unchanged(self):
        t_ref = 24.0
        t_result = q10_extrapolation(t_ref, 25.0, 25.0, q10=2.0)
        assert t_result == pytest.approx(t_ref, rel=1e-9)

    def test_higher_storage_temperature_shorter_shelf(self):
        t_result = q10_extrapolation(24.0, 25.0, 40.0, q10=2.0)
        assert t_result < 24.0

    def test_custom_q10(self):
        # Q10=3 at +10°C → shelf life / 3
        t_result = q10_extrapolation(24.0, 25.0, 35.0, q10=3.0)
        assert t_result == pytest.approx(24.0 / 3.0, rel=1e-9)

    def test_invalid_shelf_life_zero(self):
        with pytest.raises(ValueError, match="t_shelf_ref_months"):
            q10_extrapolation(0.0, 25.0, 40.0)

    def test_invalid_q10_zero(self):
        with pytest.raises(ValueError, match="q10"):
            q10_extrapolation(24.0, 25.0, 40.0, q10=0.0)


# ---------------------------------------------------------------------------
# ich_stability_prediction
# ---------------------------------------------------------------------------


class TestICHStabilityPrediction:
    def _default_result(self) -> ICHStabilityResult:
        return ich_stability_prediction("DrugA", k_25C_per_h=1e-4, ea_kJ_per_mol=80.0)

    def test_returns_dataclass(self):
        result = self._default_result()
        assert isinstance(result, ICHStabilityResult)

    def test_frozen_dataclass(self):
        result = self._default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_accelerated_shorter_than_long_term(self):
        result = self._default_result()
        assert result.accelerated_shelf_months < result.predicted_long_term_months

    def test_zone1_longer_than_zone2(self):
        # Zone I (21°C) → slower degradation → longer shelf life
        result = self._default_result()
        assert result.zone1_shelf_months > result.zone2_shelf_months

    def test_zone2_longer_than_zone3(self):
        result = self._default_result()
        assert result.zone2_shelf_months > result.zone3_shelf_months

    def test_predicted_long_term_equals_zone2(self):
        result = self._default_result()
        assert result.predicted_long_term_months == pytest.approx(result.zone2_shelf_months)

    def test_shelf_life_positive(self):
        result = self._default_result()
        assert result.zone1_shelf_months > 0
        assert result.zone2_shelf_months > 0
        assert result.accelerated_shelf_months > 0

    def test_notes_contains_drug_name(self):
        result = ich_stability_prediction("Aspirin", 1e-4, 80.0)
        assert "Aspirin" in result.notes

    def test_humidity_factor_reduces_shelf_life(self):
        r1 = ich_stability_prediction("DrugA", 1e-4, 80.0, humidity_acceleration_factor=1.0)
        r2 = ich_stability_prediction("DrugA", 1e-4, 80.0, humidity_acceleration_factor=2.0)
        # Higher humidity factor → higher k at humid conditions → shorter shelf
        assert r2.zone3_shelf_months < r1.zone3_shelf_months

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k_25C_per_h"):
            ich_stability_prediction("X", 0.0, 80.0)

    def test_invalid_ea_negative(self):
        with pytest.raises(ValueError, match="ea_kJ_per_mol"):
            ich_stability_prediction("X", 1e-4, -10.0)


# ---------------------------------------------------------------------------
# degradation_profile
# ---------------------------------------------------------------------------


class TestDegradationProfile:
    def _default_profile(self) -> dict:
        return degradation_profile("DrugA", 100.0, 1e-4)

    def test_returns_dict_with_required_keys(self):
        result = self._default_profile()
        assert "drug_name" in result
        assert "times_months" in result
        assert "potency_pct" in result
        assert "shelf_life_months" in result

    def test_potency_decreases_monotonically(self):
        result = self._default_profile()
        p = result["potency_pct"]
        for i in range(1, len(p)):
            assert p[i] <= p[i - 1]

    def test_initial_potency_preserved(self):
        result = degradation_profile("DrugA", 95.0, 1e-4)
        assert result["potency_pct"][0] == pytest.approx(95.0)

    def test_n_points(self):
        result = degradation_profile("DrugA", 100.0, 1e-4, n_points=7)
        assert len(result["times_months"]) == 7
        assert len(result["potency_pct"]) == 7

    def test_shelf_life_positive(self):
        result = self._default_profile()
        assert result["shelf_life_months"] > 0.0

    def test_high_k_shorter_shelf_life(self):
        r1 = degradation_profile("DrugA", 100.0, 1e-4)
        r2 = degradation_profile("DrugA", 100.0, 1e-3)
        assert r2["shelf_life_months"] < r1["shelf_life_months"]

    def test_drug_name_preserved(self):
        result = degradation_profile("Ibuprofen", 100.0, 1e-4)
        assert result["drug_name"] == "Ibuprofen"

    def test_first_time_point_zero(self):
        result = self._default_profile()
        assert result["times_months"][0] == pytest.approx(0.0)

    def test_last_time_point_equals_t_end(self):
        result = degradation_profile("DrugA", 100.0, 1e-4, t_end_months=24)
        assert result["times_months"][-1] == pytest.approx(24.0)

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k_per_h"):
            degradation_profile("X", 100.0, 0.0)

    def test_invalid_initial_potency_zero(self):
        with pytest.raises(ValueError, match="initial_potency_pct"):
            degradation_profile("X", 0.0, 1e-4)

    def test_invalid_initial_potency_above_100(self):
        with pytest.raises(ValueError, match="initial_potency_pct"):
            degradation_profile("X", 101.0, 1e-4)
