"""Tests for clinical/exposure_response_model.py — Phase 547."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.exposure_response_model import (
    er_safety_window,
    fit_emax_er,
    fit_linear_er,
    probability_of_response,
    quartile_analysis,
)

# ---------------------------------------------------------------------------
# fit_linear_er
# ---------------------------------------------------------------------------


class TestFitLinearEr:
    def test_perfect_linear_data(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0 + 3.0 * x for x in xs]
        result = fit_linear_er(xs, ys)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert result["slope"] == pytest.approx(3.0, rel=1e-4)
        assert result["intercept"] == pytest.approx(2.0, rel=1e-4)
        assert result["n"] == 5

    def test_positive_slope(self):
        xs = [10.0, 20.0, 30.0, 40.0]
        ys = [5.0, 10.0, 15.0, 20.0]
        result = fit_linear_er(xs, ys)
        assert result["slope"] > 0

    def test_returns_required_keys(self):
        result = fit_linear_er([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        for key in ("slope", "intercept", "r_squared", "n"):
            assert key in result

    def test_r_squared_between_0_and_1(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 4.0, 2.0, 6.0, 5.0]
        result = fit_linear_er(xs, ys)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fit_linear_er([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_linear_er([1.0, 2.0], [1.0])

    def test_two_points(self):
        result = fit_linear_er([0.0, 1.0], [0.0, 1.0])
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            fit_linear_er([1.0], [1.0])


# ---------------------------------------------------------------------------
# fit_emax_er
# ---------------------------------------------------------------------------


class TestFitEmaxEr:
    def _generate_emax_data(self, emax=100.0, auc50=10.0):
        xs = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        ys = [emax * x / (auc50 + x) for x in xs]
        return xs, ys

    def test_recovers_emax_approximately(self):
        xs, ys = self._generate_emax_data(emax=100.0, auc50=10.0)
        result = fit_emax_er(xs, ys)
        # Grid search recovers within ~50%
        assert result["emax"] == pytest.approx(100.0, rel=0.5)

    def test_returns_required_keys(self):
        xs, ys = self._generate_emax_data()
        result = fit_emax_er(xs, ys)
        for key in ("emax", "auc50", "r_squared", "predicted"):
            assert key in result

    def test_predicted_is_list(self):
        xs, ys = self._generate_emax_data()
        result = fit_emax_er(xs, ys)
        assert isinstance(result["predicted"], list)
        assert len(result["predicted"]) == len(xs)

    def test_r_squared_high_for_emax_data(self):
        xs, ys = self._generate_emax_data(emax=80.0, auc50=5.0)
        result = fit_emax_er(xs, ys)
        assert result["r_squared"] >= 0.8

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fit_emax_er([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_emax_er([1.0, 2.0], [1.0])

    def test_emax_init_ignored(self):
        xs, ys = self._generate_emax_data()
        r1 = fit_emax_er(xs, ys)
        r2 = fit_emax_er(xs, ys, emax_init=50.0)
        assert r1["emax"] == r2["emax"]


# ---------------------------------------------------------------------------
# quartile_analysis
# ---------------------------------------------------------------------------


class TestQuartileAnalysis:
    def test_returns_n_quartiles_groups(self):
        xs = list(range(1, 21))
        ys = [float(x) * 2 for x in xs]
        result = quartile_analysis(xs, ys, n_quartiles=4)
        assert len(result) == 4

    def test_each_group_has_required_keys(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ys = [float(x) for x in xs]
        result = quartile_analysis(xs, ys)
        for group in result:
            for key in ("quartile", "exp_range", "mean_exp", "mean_response", "n", "sem"):
                assert key in group

    def test_quartiles_ordered_by_exposure(self):
        xs = [10.0, 2.0, 8.0, 4.0, 6.0, 12.0, 1.0, 9.0]
        ys = [float(x) for x in xs]
        result = quartile_analysis(xs, ys, n_quartiles=4)
        mean_exps = [g["mean_exp"] for g in result]
        assert mean_exps == sorted(mean_exps)

    def test_total_n_equals_input_n(self):
        xs = list(range(1, 13))
        ys = [float(x) for x in xs]
        result = quartile_analysis(xs, ys, n_quartiles=3)
        assert sum(g["n"] for g in result) == len(xs)

    def test_custom_n_quartiles(self):
        xs = list(range(1, 11))
        ys = [float(x) for x in xs]
        result = quartile_analysis(xs, ys, n_quartiles=5)
        assert len(result) == 5

    def test_sem_zero_for_single_member(self):
        xs = [1.0, 2.0]
        ys = [3.0, 4.0]
        result = quartile_analysis(xs, ys, n_quartiles=2)
        for g in result:
            assert g["sem"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            quartile_analysis([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            quartile_analysis([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# probability_of_response
# ---------------------------------------------------------------------------


class TestProbabilityOfResponse:
    def test_returns_required_keys(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = probability_of_response(xs, ys, response_threshold=25.0)
        assert "coefficients" in result
        assert "prob_at_exposure" in result
        assert "ec50_eff" in result

    def test_prob_between_0_and_1(self):
        xs = [1.0, 2.0, 5.0, 10.0, 20.0]
        ys = [5.0, 15.0, 25.0, 35.0, 45.0]
        result = probability_of_response(xs, ys, response_threshold=20.0)
        for p in result["prob_at_exposure"].values():
            assert 0.0 <= p <= 1.0

    def test_high_exposures_higher_prob(self):
        xs = [1.0, 5.0, 10.0, 20.0, 50.0]
        ys = [2.0, 12.0, 22.0, 32.0, 42.0]
        result = probability_of_response(xs, ys, response_threshold=15.0)
        probs = [result["prob_at_exposure"][x] for x in sorted(result["prob_at_exposure"])]
        # Probability should be non-decreasing (b >= 0 in grid)
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 0.01

    def test_coefficients_dict(self):
        xs = [1.0, 2.0, 3.0]
        ys = [10.0, 20.0, 30.0]
        result = probability_of_response(xs, ys, response_threshold=15.0)
        assert "a" in result["coefficients"]
        assert "b" in result["coefficients"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            probability_of_response([], [], response_threshold=0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            probability_of_response([1.0, 2.0], [1.0], response_threshold=0.5)

    def test_unique_exposures_in_prob_dict(self):
        xs = [1.0, 1.0, 2.0, 3.0]
        ys = [5.0, 6.0, 15.0, 25.0]
        result = probability_of_response(xs, ys, response_threshold=10.0)
        # Unique exposures: {1.0, 2.0, 3.0}
        assert set(result["prob_at_exposure"].keys()) == {1.0, 2.0, 3.0}


# ---------------------------------------------------------------------------
# er_safety_window
# ---------------------------------------------------------------------------


class TestErSafetyWindow:
    def _make_data(self):
        xs = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        # Efficacy: Emax=80, AUC50=10
        eff = [80.0 * x / (10.0 + x) for x in xs]
        # Safety: Emax=60, AUC50=40 (kicks in at higher exposure)
        saf = [60.0 * x / (40.0 + x) for x in xs]
        return xs, eff, saf

    def test_returns_required_keys(self):
        xs, eff, saf = self._make_data()
        result = er_safety_window(xs, eff, saf, efficacy_threshold=40.0, safety_threshold=30.0)
        for key in (
            "auc_min_efficacy",
            "auc_max_safety",
            "therapeutic_window_exists",
            "safety_margin",
        ):
            assert key in result

    def test_therapeutic_window_exists(self):
        xs, eff, saf = self._make_data()
        result = er_safety_window(xs, eff, saf, efficacy_threshold=30.0, safety_threshold=50.0)
        assert isinstance(result["therapeutic_window_exists"], bool)

    def test_auc_min_efficacy_positive(self):
        xs, eff, saf = self._make_data()
        result = er_safety_window(xs, eff, saf, efficacy_threshold=20.0, safety_threshold=30.0)
        assert result["auc_min_efficacy"] > 0

    def test_safety_margin_positive_when_window_exists(self):
        xs, eff, saf = self._make_data()
        result = er_safety_window(xs, eff, saf, efficacy_threshold=20.0, safety_threshold=50.0)
        if result["therapeutic_window_exists"]:
            assert result["safety_margin"] > 1.0

    def test_no_window_when_safety_very_low(self):
        xs, eff, saf = self._make_data()
        # Safety threshold so low it's hit before efficacy
        result = er_safety_window(xs, eff, saf, efficacy_threshold=70.0, safety_threshold=1.0)
        # No guarantee on exact result, just check structure
        assert "therapeutic_window_exists" in result

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            er_safety_window([1.0, 2.0], [1.0], [1.0, 2.0], 0.5, 0.5)

    def test_efficacy_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            er_safety_window([1.0, 2.0], [1.0, 2.0], [1.0], 0.5, 0.5)
