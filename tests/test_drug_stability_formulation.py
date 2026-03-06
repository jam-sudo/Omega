"""Tests for drug stability in formulation module (Phase 229)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.drug_stability import (
    AcceleratedStabilityResult,
    DegradationResult,
    accelerated_stability,
    arrhenius_rate,
    shelf_life,
    simulate_degradation,
)


# ---------------------------------------------------------------------------
# arrhenius_rate tests
# ---------------------------------------------------------------------------


class TestArrheniusRate:
    def test_higher_temp_gives_faster_rate(self):
        """Storage T > reference T should give k > k_ref."""
        k = arrhenius_rate(k_ref=0.01, ea_kJ_per_mol=80.0, t_ref_C=25.0, t_storage_C=40.0)
        assert k > 0.01

    def test_same_temperature_returns_k_ref(self):
        """Same temperature should return k_ref exactly."""
        k = arrhenius_rate(k_ref=0.005, ea_kJ_per_mol=75.0, t_ref_C=25.0, t_storage_C=25.0)
        assert math.isclose(k, 0.005, rel_tol=1e-10)

    def test_lower_temp_gives_slower_rate(self):
        """Storage T < reference T should give k < k_ref."""
        k = arrhenius_rate(k_ref=0.01, ea_kJ_per_mol=80.0, t_ref_C=40.0, t_storage_C=5.0)
        assert k < 0.01

    def test_higher_ea_gives_larger_factor(self):
        """Higher activation energy should amplify the temperature effect more."""
        k_low_ea = arrhenius_rate(k_ref=0.01, ea_kJ_per_mol=50.0, t_ref_C=25.0, t_storage_C=40.0)
        k_high_ea = arrhenius_rate(k_ref=0.01, ea_kJ_per_mol=100.0, t_ref_C=25.0, t_storage_C=40.0)
        assert k_high_ea > k_low_ea

    def test_return_type_is_float(self):
        k = arrhenius_rate(0.01, 80.0, 25.0, 40.0)
        assert isinstance(k, float)

    def test_known_value_is_positive(self):
        """Verify result is positive for 40C->25C with Ea=83 kJ/mol."""
        k_40 = 0.01
        ea = 83.0
        k_25 = arrhenius_rate(k_40, ea, t_ref_C=40.0, t_storage_C=25.0)
        assert k_25 < k_40  # must be slower at lower temp
        assert k_25 > 0


# ---------------------------------------------------------------------------
# shelf_life tests
# ---------------------------------------------------------------------------


class TestShelfLife:
    def test_first_order_known_solution(self):
        """first_order: t = -ln(0.9) / k."""
        k = 0.01
        expected = -math.log(0.9) / k
        result = shelf_life(k, target_pct=90.0, model="first_order")
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_zero_order_known_solution(self):
        """zero_order: t = (100 - target) / (k * 100)."""
        k = 0.001
        target = 90.0
        expected = (100.0 - target) / (k * 100.0)
        result = shelf_life(k, target_pct=target, model="zero_order")
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_shelf_life_is_positive(self):
        result = shelf_life(0.005, target_pct=90.0)
        assert result > 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            shelf_life(-0.01)

    def test_invalid_target_pct_zero_raises(self):
        with pytest.raises(ValueError):
            shelf_life(0.01, target_pct=0.0)

    def test_invalid_target_pct_100_raises(self):
        with pytest.raises(ValueError):
            shelf_life(0.01, target_pct=100.0)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            shelf_life(0.01, model="quadratic")


# ---------------------------------------------------------------------------
# simulate_degradation tests
# ---------------------------------------------------------------------------


class TestSimulateDegradation:
    def test_first_order_exponential_decay(self):
        """First-order: pct at time t should be close to 100*exp(-k*t)."""
        k = 0.005
        result = simulate_degradation(
            "Drug A", k_per_day=k, degradation_model="first_order", t_end_days=100.0
        )
        expected_at_100 = 100.0 * math.exp(-k * 100.0)
        assert math.isclose(result.pct_remaining[-1], expected_at_100, rel_tol=0.01)

    def test_zero_order_linear_decay(self):
        """Zero-order: percentage decreases linearly."""
        k = 0.0005
        result = simulate_degradation(
            "Drug B", k_per_day=k, degradation_model="zero_order", t_end_days=50.0, dt_days=1.0
        )
        # Verify roughly linear: check midpoint is ~halfway between start and end
        mid_idx = len(result.pct_remaining) // 2
        start = result.pct_remaining[0]
        end = result.pct_remaining[-1]
        mid = result.pct_remaining[mid_idx]
        assert abs(mid - (start + end) / 2) < 5.0  # within 5% of midpoint

    def test_pct_remaining_in_valid_range(self):
        result = simulate_degradation("Drug C", k_per_day=0.01, t_end_days=365.0)
        for pct in result.pct_remaining:
            assert 0.0 <= pct <= 100.0

    def test_pct_remaining_monotonically_decreasing(self):
        result = simulate_degradation("Drug D", k_per_day=0.01, t_end_days=180.0)
        pcts = result.pct_remaining
        for i in range(1, len(pcts)):
            assert pcts[i] <= pcts[i - 1] + 1e-9

    def test_first_element_equals_initial_pct(self):
        result = simulate_degradation("Drug E", initial_pct=95.0, k_per_day=0.01)
        assert math.isclose(result.pct_remaining[0], 95.0, rel_tol=1e-9)

    def test_autocatalytic_model_runs(self):
        result = simulate_degradation(
            "Drug F", k_per_day=0.005, degradation_model="autocatalytic", t_end_days=200.0
        )
        assert result.model == "autocatalytic"
        assert len(result.pct_remaining) > 0

    def test_result_fields_types(self):
        result = simulate_degradation("Drug G", k_per_day=0.01)
        assert isinstance(result, DegradationResult)
        assert isinstance(result.drug_name, str)
        assert isinstance(result.k_per_day, float)
        assert isinstance(result.model, str)
        assert isinstance(result.times_days, list)
        assert isinstance(result.pct_remaining, list)
        assert isinstance(result.t90_days, float)
        assert isinstance(result.shelf_life_days, float)

    def test_t90_within_simulation_range(self):
        result = simulate_degradation("Drug H", k_per_day=0.005, t_end_days=730.0)
        assert 0 <= result.t90_days <= 730.0

    def test_invalid_initial_pct_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_degradation("X", initial_pct=0.0)

    def test_invalid_initial_pct_over_100_raises(self):
        with pytest.raises(ValueError):
            simulate_degradation("X", initial_pct=101.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            simulate_degradation("X", k_per_day=0.0)

    def test_invalid_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_degradation("X", t_end_days=0.0)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            simulate_degradation("X", degradation_model="unknown_model")

    def test_last_pct_less_than_first(self):
        result = simulate_degradation("Drug I", k_per_day=0.01, t_end_days=365.0)
        assert result.pct_remaining[-1] < result.pct_remaining[0]


# ---------------------------------------------------------------------------
# accelerated_stability tests
# ---------------------------------------------------------------------------


class TestAcceleratedStability:
    def test_shelf_life_25C_greater_than_40C(self):
        """25 deg C shelf-life should be longer than 40 deg C shelf-life."""
        result = accelerated_stability("Drug J", k_40C_per_day=0.005, ea_kJ_per_mol=80.0)
        assert result.shelf_life_25C_days > result.shelf_life_40C_days

    def test_k_25C_less_than_k_40C(self):
        """Rate at 25 deg C should be slower than at 40 deg C."""
        result = accelerated_stability("Drug K", k_40C_per_day=0.01, ea_kJ_per_mol=75.0)
        assert result.k_25C_per_day < result.k_40C_per_day

    def test_acceleration_factor_greater_than_one(self):
        result = accelerated_stability("Drug L", k_40C_per_day=0.01, ea_kJ_per_mol=80.0)
        assert result.acceleration_factor > 1.0

    def test_result_fields_types(self):
        result = accelerated_stability("Drug M", k_40C_per_day=0.005, ea_kJ_per_mol=80.0)
        assert isinstance(result, AcceleratedStabilityResult)
        assert isinstance(result.drug_name, str)
        assert isinstance(result.k_40C_per_day, float)
        assert isinstance(result.ea_kJ_per_mol, float)
        assert isinstance(result.k_25C_per_day, float)
        assert isinstance(result.shelf_life_25C_days, float)
        assert isinstance(result.shelf_life_40C_days, float)
        assert isinstance(result.acceleration_factor, float)
        assert isinstance(result.notes, str)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            accelerated_stability("X", k_40C_per_day=0.0, ea_kJ_per_mol=80.0)

    def test_invalid_ea_raises(self):
        with pytest.raises(ValueError):
            accelerated_stability("X", k_40C_per_day=0.01, ea_kJ_per_mol=0.0)

    def test_notes_not_empty(self):
        result = accelerated_stability("Drug N", k_40C_per_day=0.01, ea_kJ_per_mol=80.0)
        assert len(result.notes) > 0

    def test_higher_ea_gives_larger_acceleration(self):
        """Higher Ea should give larger acceleration factor."""
        r1 = accelerated_stability("Drug", k_40C_per_day=0.01, ea_kJ_per_mol=60.0)
        r2 = accelerated_stability("Drug", k_40C_per_day=0.01, ea_kJ_per_mol=100.0)
        assert r2.acceleration_factor > r1.acceleration_factor
