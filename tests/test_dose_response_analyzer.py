"""Tests for Phase 457 — DoseResponseAnalyzer."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.analysis.dose_response_analyzer import (
    DoseResponseResult,
    calculate_ec50,
    calculate_selectivity_ratio,
    fit_dose_response,
    four_param_logistic,
    hill_equation,
    inhibition_constant,
)


# ---------------------------------------------------------------------------
# four_param_logistic
# ---------------------------------------------------------------------------

class TestFourParamLogistic:
    def test_at_ec50_returns_midpoint(self):
        """4PL at dose=EC50 → (top+bottom)/2."""
        bottom, top, ec50, hill = 0.0, 100.0, 10.0, 1.0
        result = four_param_logistic(ec50, bottom, top, ec50, hill)
        assert math.isclose(result, (top + bottom) / 2.0, rel_tol=1e-6)

    def test_at_ec50_asymmetric_baseline(self):
        bottom, top, ec50, hill = 10.0, 90.0, 5.0, 1.5
        result = four_param_logistic(ec50, bottom, top, ec50, hill)
        assert math.isclose(result, (top + bottom) / 2.0, rel_tol=1e-6)

    def test_dose_zero_returns_bottom(self):
        result = four_param_logistic(0, 5.0, 100.0, 10.0, 1.0)
        assert result == 5.0

    def test_very_high_dose_approaches_top(self):
        bottom, top, ec50, hill = 0.0, 100.0, 1.0, 2.0
        result = four_param_logistic(1e6, bottom, top, ec50, hill)
        assert result > 99.9

    def test_very_low_dose_approaches_bottom(self):
        bottom, top, ec50, hill = 0.0, 100.0, 100.0, 2.0
        result = four_param_logistic(1e-6, bottom, top, ec50, hill)
        assert result < 0.01

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            four_param_logistic(-1.0, 0.0, 100.0, 10.0, 1.0)

    def test_zero_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50 must be positive"):
            four_param_logistic(5.0, 0.0, 100.0, 0.0, 1.0)

    def test_hill_slope_effect(self):
        """Steeper Hill slope → sharper transition near EC50."""
        bottom, top, ec50 = 0.0, 100.0, 10.0
        # At dose=5 (half EC50), steeper slope → lower response
        r_shallow = four_param_logistic(5.0, bottom, top, ec50, 0.5)
        r_steep = four_param_logistic(5.0, bottom, top, ec50, 4.0)
        assert r_shallow > r_steep

    def test_returns_float(self):
        result = four_param_logistic(10.0, 0.0, 100.0, 10.0, 1.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# hill_equation
# ---------------------------------------------------------------------------

class TestHillEquation:
    def test_at_ec50_returns_half_emax(self):
        """Hill equation at C=EC50 → Emax/2."""
        emax, ec50, n = 80.0, 5.0, 1.0
        result = hill_equation(ec50, emax, ec50, n)
        assert math.isclose(result, emax / 2.0, rel_tol=1e-6)

    def test_at_ec50_different_n(self):
        """Hill equation at C=EC50 → Emax/2 regardless of n."""
        for n in (0.5, 1.0, 2.0, 4.0):
            result = hill_equation(5.0, 100.0, 5.0, n)
            assert math.isclose(result, 50.0, rel_tol=1e-6), f"failed for n={n}"

    def test_zero_concentration_returns_zero(self):
        assert hill_equation(0.0, 100.0, 5.0) == 0.0

    def test_large_concentration_approaches_emax(self):
        result = hill_equation(1e9, 100.0, 1.0, 2.0)
        assert result > 99.9

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            hill_equation(-1.0, 100.0, 5.0)

    def test_zero_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50 must be positive"):
            hill_equation(5.0, 100.0, 0.0)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            hill_equation(5.0, 100.0, 5.0, n=0.0)


# ---------------------------------------------------------------------------
# calculate_selectivity_ratio
# ---------------------------------------------------------------------------

class TestSelectivityRatio:
    def test_selective_compound_ratio_less_than_one(self):
        """EC50_target << EC50_off_target → ratio < 1 (selective)."""
        ratio = calculate_selectivity_ratio(ec50_target=0.1, ec50_off_target=100.0)
        assert ratio < 1.0
        assert math.isclose(ratio, 0.001, rel_tol=1e-6)

    def test_nonselective_compound_ratio_greater_than_one(self):
        ratio = calculate_selectivity_ratio(ec50_target=100.0, ec50_off_target=1.0)
        assert ratio > 1.0

    def test_equal_ec50_ratio_is_one(self):
        ratio = calculate_selectivity_ratio(5.0, 5.0)
        assert math.isclose(ratio, 1.0, rel_tol=1e-6)

    def test_invalid_target_ec50_raises(self):
        with pytest.raises(ValueError):
            calculate_selectivity_ratio(0.0, 10.0)

    def test_invalid_off_target_ec50_raises(self):
        with pytest.raises(ValueError):
            calculate_selectivity_ratio(1.0, -5.0)

    def test_returns_float(self):
        assert isinstance(calculate_selectivity_ratio(1.0, 10.0), float)


# ---------------------------------------------------------------------------
# inhibition_constant (Cheng-Prusoff)
# ---------------------------------------------------------------------------

class TestInhibitionConstant:
    def test_substrate_at_km_gives_ic50_over_2(self):
        """[S] = Km → Ki = IC50 / 2."""
        ic50 = 20.0
        ki = inhibition_constant(ic50, hill_slope=1.0, substrate_conc=10.0, km=10.0)
        assert math.isclose(ki, ic50 / 2.0, rel_tol=1e-6)

    def test_simplified_hill1_gives_ic50_over_2(self):
        """Without substrate data, hill_slope=1 → Ki = IC50/2."""
        ki = inhibition_constant(ic50=40.0, hill_slope=1.0)
        assert math.isclose(ki, 20.0, rel_tol=1e-6)

    def test_cheng_prusoff_substrate_zero(self):
        """[S]=0 → Ki = IC50 (no competitive correction)."""
        ki = inhibition_constant(10.0, hill_slope=1.0, substrate_conc=0.0, km=5.0)
        assert math.isclose(ki, 10.0, rel_tol=1e-6)

    def test_high_substrate_to_km_ratio_reduces_ki(self):
        """High [S]/Km → Ki << IC50."""
        ki = inhibition_constant(10.0, substrate_conc=90.0, km=10.0)
        assert ki < 2.0

    def test_invalid_ic50_raises(self):
        with pytest.raises(ValueError):
            inhibition_constant(-1.0)

    def test_invalid_km_raises(self):
        with pytest.raises(ValueError):
            inhibition_constant(10.0, substrate_conc=5.0, km=0.0)

    def test_invalid_hill_slope_raises(self):
        with pytest.raises(ValueError):
            inhibition_constant(10.0, hill_slope=0.0)

    def test_returns_float(self):
        assert isinstance(inhibition_constant(10.0), float)


# ---------------------------------------------------------------------------
# fit_dose_response / calculate_ec50
# ---------------------------------------------------------------------------

def _make_ideal_4pl(doses, bottom, top, ec50, hill):
    """Generate ideal (noiseless) 4PL responses."""
    return [four_param_logistic(d, bottom, top, ec50, hill) for d in doses]


class TestFitDoseResponse:
    def test_recovers_ec50_within_10_percent(self):
        """Fit on ideal 4PL data recovers EC50 within 10%."""
        true_ec50 = 10.0
        doses = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
        responses = _make_ideal_4pl(doses, 0.0, 100.0, true_ec50, 1.5)
        result = fit_dose_response(doses, responses)
        assert abs(result.ec50 - true_ec50) / true_ec50 < 0.10

    def test_r_squared_in_range(self):
        doses = [0.1, 1, 10, 100, 1000]
        responses = _make_ideal_4pl(doses, 5.0, 95.0, 50.0, 1.0)
        result = fit_dose_response(doses, responses)
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_near_one_for_ideal_data(self):
        doses = [0.1, 1, 10, 100, 1000]
        responses = _make_ideal_4pl(doses, 0.0, 100.0, 10.0, 1.0)
        result = fit_dose_response(doses, responses)
        assert result.r_squared > 0.98

    def test_fitted_responses_same_length_as_input(self):
        doses = [1, 3, 10, 30, 100]
        responses = _make_ideal_4pl(doses, 0.0, 100.0, 10.0, 1.0)
        result = fit_dose_response(doses, responses)
        assert len(result.fitted_responses) == len(doses)
        assert len(result.doses) == len(doses)

    def test_model_field_is_4pl(self):
        doses = [1, 10, 100]
        responses = _make_ideal_4pl(doses, 0.0, 100.0, 10.0, 1.0)
        result = fit_dose_response(doses, responses)
        assert result.model == "4pl"

    def test_emax_equals_top_minus_bottom(self):
        doses = [1, 10, 100]
        responses = _make_ideal_4pl(doses, 10.0, 80.0, 10.0, 1.0)
        result = fit_dose_response(doses, responses)
        assert math.isclose(result.emax, result.top - result.bottom, rel_tol=1e-6)

    def test_empty_doses_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_dose_response([], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_dose_response([1, 2, 3], [10, 20])

    def test_nonpositive_dose_raises(self):
        with pytest.raises(ValueError, match="positive"):
            fit_dose_response([0, 1, 10], [0, 50, 100])

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="unsupported model"):
            fit_dose_response([1, 10], [30, 80], model="3pl")

    def test_calculate_ec50_convenience_function(self):
        true_ec50 = 25.0
        doses = [0.1, 1, 10, 25, 100, 250, 1000]
        responses = _make_ideal_4pl(doses, 0.0, 100.0, true_ec50, 1.0)
        ec50 = calculate_ec50(doses, responses)
        assert abs(ec50 - true_ec50) / true_ec50 < 0.10

    def test_result_is_dose_response_result(self):
        doses = [1, 10, 100]
        responses = [10.0, 50.0, 90.0]
        result = fit_dose_response(doses, responses)
        assert isinstance(result, DoseResponseResult)

    def test_bottom_is_min_response(self):
        doses = [1, 10, 100]
        responses = [5.0, 50.0, 95.0]
        result = fit_dose_response(doses, responses)
        assert result.bottom == 5.0

    def test_top_is_max_response(self):
        doses = [1, 10, 100]
        responses = [5.0, 50.0, 95.0]
        result = fit_dose_response(doses, responses)
        assert result.top == 95.0
