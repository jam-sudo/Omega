"""Tests for analysis/hysteresis_analysis.py — Phase 309."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.analysis.hysteresis_analysis import (
    HysteresisResult,
    classify_hysteresis,
    compute_hysteresis_area,
    fit_effect_compartment_ke0,
    plot_hysteresis_loop_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clockwise_data():
    """Simulate a clockwise C-E loop (effect lags concentration)."""
    t = np.linspace(0, 12, 20)
    # Plasma conc: rise then fall (start at small positive value to satisfy validation)
    conc = np.where(t <= 4, t * 2.5 + 0.01, 10.0 * np.exp(-0.3 * (t - 4)))
    # Effect lags behind conc (ke0 delay)
    ke0 = 0.5
    ce = np.zeros(20)
    for i in range(19):
        dt = t[i + 1] - t[i]
        ce[i + 1] = max(ce[i] + ke0 * (conc[i] - ce[i]) * dt, 0.0)
    effect = 100 * ce / (2.0 + ce)
    return conc.tolist(), effect.tolist(), t.tolist()


def _counter_clockwise_data():
    """Simulate counter-clockwise C-E loop (sensitisation/indirect response).

    Counter-clockwise: ascending limb has HIGHER effect than descending limb
    at the same concentration (effect builds up faster than conc then wanes slowly).
    """
    # At same conc, ascending has higher effect → counter-clockwise in PK/PD convention
    conc = [1.0, 3.0, 5.0, 8.0, 10.0, 9.0, 7.0, 5.0, 3.0, 1.0]
    effect = [30.0, 60.0, 75.0, 85.0, 90.0, 50.0, 35.0, 25.0, 15.0, 10.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    return conc, effect, times


# ---------------------------------------------------------------------------
# compute_hysteresis_area
# ---------------------------------------------------------------------------


class TestComputeHysteresisArea:
    def test_returns_float(self):
        c, e, _ = _clockwise_data()
        area = compute_hysteresis_area(c, e)
        assert isinstance(area, float)

    def test_clockwise_negative_area(self):
        c, e, _ = _clockwise_data()
        area = compute_hysteresis_area(c, e)
        assert area < 0, f"Expected negative area for clockwise loop, got {area}"

    def test_counter_clockwise_positive_area(self):
        c, e, _ = _counter_clockwise_data()
        area = compute_hysteresis_area(c, e)
        assert area > 0, f"Expected positive area for counter-clockwise loop, got {area}"

    def test_collinear_near_zero(self):
        # Straight line C-E relationship: no loop
        conc = [1.0, 2.0, 3.0, 4.0]
        effect = [10.0, 20.0, 30.0, 40.0]
        area = compute_hysteresis_area(conc, effect)
        assert abs(area) < 1e-9

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compute_hysteresis_area([1, 2, 3, 4], [10, 20, 30])

    def test_validation_too_few_points(self):
        with pytest.raises(ValueError, match="At least 4"):
            compute_hysteresis_area([1, 2, 3], [10, 20, 30])

    def test_validation_zero_concentration(self):
        with pytest.raises(ValueError, match="must be > 0"):
            compute_hysteresis_area([0, 1, 2, 3], [0, 10, 20, 30])

    def test_validation_negative_concentration(self):
        with pytest.raises(ValueError, match="must be > 0"):
            compute_hysteresis_area([-1, 1, 2, 3], [0, 10, 20, 30])

    def test_validation_effect_out_of_range_high(self):
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            compute_hysteresis_area([1, 2, 3, 4], [10, 20, 30, 110])

    def test_validation_effect_out_of_range_low(self):
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            compute_hysteresis_area([1, 2, 3, 4], [-1, 20, 30, 40])

    def test_shoelace_known_rectangle(self):
        # Rectangle traversed clockwise: (1,0)->(1,50)->(5,50)->(5,0)->(1,0)
        c = [1.0, 1.0, 5.0, 5.0, 1.0]
        e = [0.0, 50.0, 50.0, 0.0, 0.0]
        area = compute_hysteresis_area(c, e)
        # Clockwise -> negative, |area| = 4 * 50 / 2 segments
        # Shoelace: signed area of the polygon
        assert abs(abs(area) - 200.0) < 1.0


# ---------------------------------------------------------------------------
# classify_hysteresis
# ---------------------------------------------------------------------------


class TestClassifyHysteresis:
    def test_returns_dataclass(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert isinstance(result, HysteresisResult)

    def test_clockwise_classification(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert result.hysteresis_type == "clockwise"
        assert result.hysteresis_area < 0

    def test_counter_clockwise_classification(self):
        c, e, t = _counter_clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert result.hysteresis_type == "counter_clockwise"
        assert result.hysteresis_area > 0

    def test_slopes_are_floats(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert isinstance(result.ascending_slope, float)
        assert isinstance(result.descending_slope, float)

    def test_t_lag_non_negative(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert result.t_lag_h >= 0.0

    def test_ke0_estimated_positive_when_lag_exists(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        if result.t_lag_h > 0:
            assert result.estimated_ke0_per_h > 0

    def test_ke0_formula(self):
        # If t_lag = ln(2)/ke0 then ke0 = ln(2)/t_lag
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        if 0 < result.t_lag_h < 1e6:
            expected_ke0 = math.log(2) / result.t_lag_h
            assert abs(result.estimated_ke0_per_h - expected_ke0) < 1e-9

    def test_notes_non_empty(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        assert len(result.notes) > 0

    def test_times_h_length_mismatch(self):
        c, e, _ = _clockwise_data()
        with pytest.raises(ValueError):
            classify_hysteresis(c, e, [0, 1, 2])

    def test_frozen_dataclass(self):
        c, e, t = _clockwise_data()
        result = classify_hysteresis(c, e, t)
        with pytest.raises((AttributeError, TypeError)):
            result.hysteresis_type = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fit_effect_compartment_ke0
# ---------------------------------------------------------------------------


class TestFitEffectCompartmentKe0:
    def test_returns_float(self):
        c, e, t = _clockwise_data()
        ke0 = fit_effect_compartment_ke0(t, c, e)
        assert isinstance(ke0, float)

    def test_ke0_in_valid_range(self):
        c, e, t = _clockwise_data()
        ke0 = fit_effect_compartment_ke0(t, c, e)
        assert 0.01 <= ke0 <= 10.0

    def test_with_ec50_init(self):
        c, e, t = _clockwise_data()
        ke0 = fit_effect_compartment_ke0(t, c, e, ec50_init=2.0)
        assert 0.01 <= ke0 <= 10.0

    def test_known_ke0_recovery(self):
        """Fit should recover ke0 near true value for synthetic data."""
        true_ke0 = 1.5
        t = np.linspace(0, 12, 40)
        # Plasma PK: IV bolus
        cp = 10.0 * np.exp(-0.3 * t)
        # Simulate effect compartment
        ce = np.zeros(40)
        for i in range(39):
            dt = t[i + 1] - t[i]
            ce[i + 1] = max(ce[i] + true_ke0 * (cp[i] - ce[i]) * dt, 0.0)
        emax = 100.0
        ec50 = 3.0
        effect = emax * ce / (ec50 + ce)

        ke0_fit = fit_effect_compartment_ke0(
            t.tolist(), cp.tolist(), effect.tolist(), emax=emax, ec50_init=ec50
        )
        # Allow ±50% error (simplified 1-parameter fit)
        assert abs(ke0_fit - true_ke0) / true_ke0 < 0.5

    def test_validation_mismatch(self):
        c, e, t = _clockwise_data()
        with pytest.raises(ValueError):
            fit_effect_compartment_ke0([0, 1, 2], c, e)

    def test_emax_parameter(self):
        c, e, t = _clockwise_data()
        # Should not raise regardless of emax
        ke0 = fit_effect_compartment_ke0(t, c, e, emax=1.0)
        assert ke0 > 0


# ---------------------------------------------------------------------------
# plot_hysteresis_loop_data
# ---------------------------------------------------------------------------


class TestPlotHysteresisLoopData:
    def test_returns_dict_with_expected_keys(self):
        c, e, _ = _clockwise_data()
        cmax_idx = int(np.argmax(c))
        asc = list(range(0, cmax_idx + 1))
        desc = list(range(cmax_idx, len(c)))
        result = plot_hysteresis_loop_data(c, e, asc, desc)
        assert set(result.keys()) == {
            "ascending_conc",
            "ascending_effect",
            "descending_conc",
            "descending_effect",
        }

    def test_ascending_values_correct(self):
        c = [1.0, 3.0, 6.0, 10.0, 7.0, 3.0, 1.0]
        e = [5.0, 15.0, 30.0, 50.0, 45.0, 25.0, 10.0]
        asc = [0, 1, 2, 3]
        desc = [3, 4, 5, 6]
        result = plot_hysteresis_loop_data(c, e, asc, desc)
        assert result["ascending_conc"] == [1.0, 3.0, 6.0, 10.0]
        assert result["ascending_effect"] == [5.0, 15.0, 30.0, 50.0]

    def test_descending_values_correct(self):
        c = [1.0, 3.0, 6.0, 10.0, 7.0, 3.0, 1.0]
        e = [5.0, 15.0, 30.0, 50.0, 45.0, 25.0, 10.0]
        asc = [0, 1, 2, 3]
        desc = [3, 4, 5, 6]
        result = plot_hysteresis_loop_data(c, e, asc, desc)
        assert result["descending_conc"] == [10.0, 7.0, 3.0, 1.0]

    def test_lengths_match(self):
        c, e, _ = _clockwise_data()
        cmax_idx = int(np.argmax(c))
        asc = list(range(0, cmax_idx + 1))
        desc = list(range(cmax_idx, len(c)))
        result = plot_hysteresis_loop_data(c, e, asc, desc)
        assert len(result["ascending_conc"]) == len(result["ascending_effect"])
        assert len(result["descending_conc"]) == len(result["descending_effect"])

    def test_returns_lists(self):
        c, e, _ = _clockwise_data()
        cmax_idx = int(np.argmax(c))
        asc = list(range(0, cmax_idx + 1))
        desc = list(range(cmax_idx, len(c)))
        result = plot_hysteresis_loop_data(c, e, asc, desc)
        for key in result:
            assert isinstance(result[key], list)
