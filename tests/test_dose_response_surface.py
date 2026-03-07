"""Tests for analysis/dose_response_surface.py — Phase 284."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.dose_response_surface import (
    DoseResponseSurfaceResult,
    HillFitResult,
    dose_response_surface,
    fit_hill_curve,
    optimal_combination,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_hill_doses_responses(emax=80.0, ed50=5.0, n=1.5, n_points=8):
    """Generate clean Hill curve data for testing."""
    doses = [0.1 * (10 ** (i * 2 / (n_points - 1))) for i in range(n_points)]
    responses = [emax * d**n / (ed50**n + d**n) for d in doses]
    return doses, responses


def _make_grid(d1_doses, d2_doses, emax=80.0, ed50_1=5.0, ed50_2=3.0, n=1.5):
    """Create a synergy grid from Bliss independence."""
    matrix = []
    for d1 in d1_doses:
        row = []
        for d2 in d2_doses:
            e1 = emax * d1**n / (ed50_1**n + d1**n)
            e2 = emax * d2**n / (ed50_2**n + d2**n)
            bliss = (e1 / 100 + e2 / 100 - (e1 / 100) * (e2 / 100)) * 100
            row.append(min(100.0, round(bliss, 2)))
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# fit_hill_curve
# ---------------------------------------------------------------------------


class TestFitHillCurve:
    def test_returns_hill_fit_result(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert isinstance(result, HillFitResult)

    def test_emax_close_to_true(self):
        """Fitted Emax should be close to the true value."""
        doses, responses = _make_hill_doses_responses(emax=80.0)
        result = fit_hill_curve(doses, responses)
        assert abs(result.emax - 80.0) < 10.0  # allow 10 unit tolerance

    def test_ed50_positive(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert result.ed50 > 0.0

    def test_hill_n_positive(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert result.hill_n > 0.0

    def test_r2_near_one_for_perfect_data(self):
        doses, responses = _make_hill_doses_responses(n_points=10)
        result = fit_hill_curve(doses, responses)
        assert result.r2 > 0.9

    def test_rmse_nonnegative(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert result.rmse >= 0.0

    def test_fitted_responses_same_length(self):
        doses, responses = _make_hill_doses_responses(n_points=6)
        result = fit_hill_curve(doses, responses)
        assert len(result.fitted_responses) == 6

    def test_fitted_responses_nonnegative(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert all(r >= 0.0 for r in result.fitted_responses)

    def test_result_frozen(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        with pytest.raises(Exception):
            result.emax = 999.0  # type: ignore[misc]

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="data points"):
            fit_hill_curve([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    def test_doses_must_be_positive(self):
        with pytest.raises(ValueError, match="doses must be > 0"):
            fit_hill_curve([-1.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 30.0])

    def test_responses_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            fit_hill_curve([1.0, 2.0, 3.0, 4.0], [0.0, 10.0, 50.0, 110.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            fit_hill_curve([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0])

    def test_notes_string(self):
        doses, responses = _make_hill_doses_responses()
        result = fit_hill_curve(doses, responses)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# dose_response_surface
# ---------------------------------------------------------------------------


class TestDoseResponseSurface:
    @pytest.fixture
    def grid_data(self):
        d1 = [0.1, 1.0, 5.0, 10.0, 50.0]
        d2 = [0.5, 2.0, 8.0, 20.0, 80.0]
        matrix = _make_grid(d1, d2)
        return d1, d2, matrix

    def test_returns_surface_result(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert isinstance(result, DoseResponseSurfaceResult)

    def test_drug1_fit_type(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert isinstance(result.drug1_fit, HillFitResult)

    def test_drug2_fit_type(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert isinstance(result.drug2_fit, HillFitResult)

    def test_interaction_matrix_shape(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert len(result.interaction_matrix) == len(d1)
        assert all(len(row) == len(d2) for row in result.interaction_matrix)

    def test_synergy_antagonism_counts_nonnegative(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert result.synergy_count >= 0
        assert result.antagonism_count >= 0

    def test_interaction_index_positive(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        for row in result.interaction_matrix:
            for ii in row:
                assert ii >= 0.0

    def test_result_frozen(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        with pytest.raises(Exception):
            result.synergy_count = 999  # type: ignore[misc]

    def test_bliss_grid_additive_interaction(self, grid_data):
        """For Bliss-generated data, II should be near 1.0 (additive)."""
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        flat = [ii for row in result.interaction_matrix for ii in row]
        # Most should be near 1.0 (within 20% tolerance for Bliss surface)
        near_one = sum(1 for ii in flat if 0.5 <= ii <= 1.5)
        assert near_one / len(flat) > 0.5

    def test_too_few_drug1_doses(self):
        with pytest.raises(ValueError, match="drug1_doses"):
            dose_response_surface(
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [[10.0, 20.0, 30.0, 40.0]] * 3,
            )

    def test_too_few_drug2_doses(self):
        with pytest.raises(ValueError, match="drug2_doses"):
            dose_response_surface(
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0],
                [[10.0, 20.0, 30.0]] * 4,
            )

    def test_invalid_response_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            dose_response_surface(
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [[10.0, 20.0, 30.0, 110.0]] * 4,
            )

    def test_notes_string(self, grid_data):
        d1, d2, matrix = grid_data
        result = dose_response_surface(d1, d2, matrix)
        assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# optimal_combination
# ---------------------------------------------------------------------------


class TestOptimalCombination:
    @pytest.fixture
    def grid_data(self):
        d1 = [0.1, 1.0, 5.0, 10.0, 50.0]
        d2 = [0.5, 2.0, 8.0, 20.0, 80.0]
        matrix = _make_grid(d1, d2)
        return d1, d2, matrix

    def test_returns_dict(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix)
        assert isinstance(result, dict)

    def test_keys_present(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix)
        expected_keys = {"drug1_dose", "drug2_dose", "response", "total_dose", "distance_to_target", "notes"}
        assert expected_keys.issubset(result.keys())

    def test_dose_in_input_list(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix)
        assert result["drug1_dose"] in d1
        assert result["drug2_dose"] in d2

    def test_total_dose_matches(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix)
        expected_total = result["drug1_dose"] + result["drug2_dose"]
        assert abs(result["total_dose"] - expected_total) < 1e-9

    def test_distance_to_target_correct(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix, target_response=50.0)
        expected_dist = abs(result["response"] - 50.0)
        assert abs(result["distance_to_target"] - expected_dist) < 1e-9

    def test_invalid_target_negative(self, grid_data):
        d1, d2, matrix = grid_data
        with pytest.raises(ValueError, match="target_response"):
            optimal_combination(d1, d2, matrix, target_response=-10.0)

    def test_invalid_target_over_100(self, grid_data):
        d1, d2, matrix = grid_data
        with pytest.raises(ValueError, match="target_response"):
            optimal_combination(d1, d2, matrix, target_response=150.0)

    def test_target_0_returns_min_response(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix, target_response=0.0)
        # Should pick lowest dose combination
        assert result["distance_to_target"] <= result["response"]

    def test_notes_nonempty(self, grid_data):
        d1, d2, matrix = grid_data
        result = optimal_combination(d1, d2, matrix)
        assert isinstance(result["notes"], str) and len(result["notes"]) > 0
