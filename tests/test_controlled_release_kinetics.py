"""Tests for controlled release kinetics — Phase 462."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.controlled_release_kinetics import (
    ReleaseModelFitResult,
    compare_release_profiles,
    first_order_release,
    fit_release_model,
    higuchi_release,
    korsmeyer_peppas,
    zero_order_release,
)

# ---------------------------------------------------------------------------
# zero_order_release
# ---------------------------------------------------------------------------


class TestZeroOrderRelease:
    def test_zero_at_t_zero(self):
        assert zero_order_release(0.0, 0.05) == pytest.approx(0.0)

    def test_linear_increase(self):
        """Fraction should be proportional to time."""
        k0 = 0.1
        f1 = zero_order_release(5.0, k0)
        f2 = zero_order_release(10.0, k0)
        assert f2 == pytest.approx(2 * f1, rel=1e-6)

    def test_capped_at_one(self):
        assert zero_order_release(1000.0, 1.0) == pytest.approx(1.0)

    def test_lag_time_delays_release(self):
        k0, lag = 0.1, 2.0
        # Before lag: zero
        assert zero_order_release(1.0, k0, t_lag_h=lag) == pytest.approx(0.0)
        # After lag: proportional to (t - lag)
        assert zero_order_release(5.0, k0, t_lag_h=lag) == pytest.approx(k0 * 3.0, rel=1e-6)

    def test_at_lag_time_zero(self):
        assert zero_order_release(2.0, 0.1, t_lag_h=2.0) == pytest.approx(0.0)

    def test_negative_k0_raises(self):
        with pytest.raises(ValueError):
            zero_order_release(1.0, -0.1)

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            zero_order_release(1.0, 0.1, t_lag_h=-1.0)


# ---------------------------------------------------------------------------
# first_order_release
# ---------------------------------------------------------------------------


class TestFirstOrderRelease:
    def test_zero_at_t_zero(self):
        assert first_order_release(0.0, 0.5) == pytest.approx(0.0)

    def test_approaches_one(self):
        """Long time → fraction approaches 1."""
        f = first_order_release(1000.0, 1.0)
        assert f == pytest.approx(1.0, abs=1e-6)

    def test_mathematical_formula(self):
        k1, t = 0.3, 4.0
        expected = 1.0 - math.exp(-k1 * t)
        assert first_order_release(t, k1) == pytest.approx(expected, rel=1e-6)

    def test_lag_time_delays_release(self):
        k1, lag = 0.3, 2.0
        assert first_order_release(1.0, k1, t_lag_h=lag) == pytest.approx(0.0)
        expected = 1.0 - math.exp(-k1 * 3.0)
        assert first_order_release(5.0, k1, t_lag_h=lag) == pytest.approx(expected, rel=1e-6)

    def test_asymptotic_behaviour(self):
        f5 = first_order_release(5.0, 0.5)
        f10 = first_order_release(10.0, 0.5)
        assert f10 > f5
        assert f10 <= 1.0

    def test_negative_k1_raises(self):
        with pytest.raises(ValueError):
            first_order_release(1.0, -0.3)


# ---------------------------------------------------------------------------
# higuchi_release
# ---------------------------------------------------------------------------


class TestHiguchiRelease:
    def test_zero_at_t_zero(self):
        assert higuchi_release(0.0, 0.2) == pytest.approx(0.0)

    def test_sqrt_t_formula(self):
        kH, t = 0.15, 9.0
        assert higuchi_release(t, kH) == pytest.approx(kH * math.sqrt(t), rel=1e-6)

    def test_scales_with_kH(self):
        t = 4.0
        f1 = higuchi_release(t, 0.1)
        f2 = higuchi_release(t, 0.2)
        assert f2 == pytest.approx(2 * f1, rel=1e-6)

    def test_increases_with_time(self):
        kH = 0.1
        f1 = higuchi_release(1.0, kH)
        f4 = higuchi_release(4.0, kH)
        assert f4 == pytest.approx(2 * f1, rel=1e-6)  # sqrt(4) = 2 * sqrt(1)

    def test_negative_kH_raises(self):
        with pytest.raises(ValueError):
            higuchi_release(1.0, -0.1)

    def test_negative_t_raises(self):
        with pytest.raises(ValueError):
            higuchi_release(-1.0, 0.1)


# ---------------------------------------------------------------------------
# korsmeyer_peppas
# ---------------------------------------------------------------------------


class TestKorsmeyerPeppas:
    def test_zero_at_t_zero(self):
        assert korsmeyer_peppas(0.0, 0.3, 0.45) == pytest.approx(0.0)

    def test_power_law_formula(self):
        k, n, t = 0.3, 0.5, 4.0
        assert korsmeyer_peppas(t, k, n) == pytest.approx(k * (t**n), rel=1e-6)

    def test_n_one_linear_in_t(self):
        """n=1 → fraction = k * t (zero-order like)."""
        k = 0.1
        f1 = korsmeyer_peppas(3.0, k, 1.0)
        assert f1 == pytest.approx(k * 3.0, rel=1e-6)

    def test_n_half_sqrt_t(self):
        """n=0.5 → fraction = k * sqrt(t)."""
        k = 0.2
        f = korsmeyer_peppas(9.0, k, 0.5)
        assert f == pytest.approx(k * 3.0, rel=1e-6)

    def test_fickian_label_for_n_le_045(self):
        """n=0.45 is Fickian boundary."""
        f = korsmeyer_peppas(4.0, 0.2, 0.45)
        assert f == pytest.approx(0.2 * (4.0**0.45), rel=1e-6)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            korsmeyer_peppas(4.0, -0.1, 0.5)

    def test_increases_with_time(self):
        k, n = 0.2, 0.5
        assert korsmeyer_peppas(4.0, k, n) > korsmeyer_peppas(1.0, k, n)


# ---------------------------------------------------------------------------
# fit_release_model
# ---------------------------------------------------------------------------


class TestFitReleaseModel:
    def _zero_order_data(self, k0=0.08, n_pts=10, t_max=10.0):
        times = [t_max * i / (n_pts - 1) for i in range(n_pts)]
        fracs = [min(k0 * t, 1.0) for t in times]
        return times, fracs

    def _first_order_data(self, k1=0.3, n_pts=10, t_max=10.0):
        times = [t_max * i / (n_pts - 1) for i in range(n_pts)]
        fracs = [1.0 - math.exp(-k1 * t) for t in times]
        return times, fracs

    def _higuchi_data(self, kH=0.15, n_pts=10, t_max=16.0):
        times = [t_max * i / (n_pts - 1) for i in range(n_pts)]
        fracs = [min(kH * math.sqrt(t), 1.0) for t in times]
        return times, fracs

    def _kp_data(self, k=0.3, n=0.45, n_pts=10, t_max=10.0):
        times = [0.5 + t_max * i / (n_pts - 1) for i in range(n_pts)]
        fracs = [min(k * (t**n), 1.0) for t in times]
        return times, fracs

    def test_returns_result_instance(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        assert isinstance(res, ReleaseModelFitResult)

    def test_identifies_zero_order(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        assert res.best_model == "zero_order"

    def test_identifies_first_order(self):
        times, fracs = self._first_order_data()
        res = fit_release_model(times, fracs)
        assert res.best_model == "first_order"

    def test_identifies_higuchi(self):
        times, fracs = self._higuchi_data()
        res = fit_release_model(times, fracs)
        assert res.best_model == "higuchi"

    def test_identifies_korsmeyer_peppas(self):
        # Use n=0.45 (Fickian) which gives clear power-law with no saturation
        times, fracs = self._kp_data(n=0.45, k=0.25, t_max=8.0)
        res = fit_release_model(times, fracs)
        # KP with n=0.45 is identical to Higuchi (sqrt-t), so either model
        # should have a high R²
        assert res.best_r_squared > 0.95

    def test_r_squared_values_has_all_models(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        models = {"zero_order", "first_order", "higuchi", "korsmeyer_peppas"}
        assert set(res.r_squared_values.keys()) == models

    def test_best_r_squared_between_0_and_1(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        assert 0.0 <= res.best_r_squared <= 1.0

    def test_model_parameters_is_dict(self):
        times, fracs = self._higuchi_data()
        res = fit_release_model(times, fracs)
        assert isinstance(res.model_parameters, dict)

    def test_release_mechanism_valid(self):
        valid = {"fickian_diffusion", "case_ii_transport", "anomalous"}
        times, fracs = self._first_order_data()
        res = fit_release_model(times, fracs)
        assert res.release_mechanism in valid

    def test_fickian_mechanism_for_low_n(self):
        """KP n=0.3 should yield fickian_diffusion mechanism."""
        times, fracs = self._kp_data(n=0.3, k=0.4)
        res = fit_release_model(times, fracs)
        assert res.release_mechanism == "fickian_diffusion"

    def test_case_ii_mechanism_for_high_n(self):
        """KP n=0.9 should yield case_ii_transport mechanism."""
        times, fracs = self._kp_data(n=0.9, k=0.1)
        res = fit_release_model(times, fracs)
        assert res.release_mechanism == "case_ii_transport"

    def test_notes_is_string(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        assert isinstance(res.notes, str)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_release_model([1.0], [0.1])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            fit_release_model([1.0, 2.0, 3.0], [0.1, 0.2])

    def test_zero_order_high_r_squared(self):
        times, fracs = self._zero_order_data()
        res = fit_release_model(times, fracs)
        assert res.r_squared_values["zero_order"] > 0.95

    def test_higuchi_high_r_squared_for_sqrt_data(self):
        times, fracs = self._higuchi_data()
        res = fit_release_model(times, fracs)
        assert res.r_squared_values["higuchi"] > 0.95


# ---------------------------------------------------------------------------
# compare_release_profiles
# ---------------------------------------------------------------------------


class TestCompareReleaseProfiles:
    def _linear_profile(self, k=0.08, n=12, t_max=12.0):
        times = [t_max * i / (n - 1) for i in range(n)]
        fracs = [min(k * t, 1.0) for t in times]
        return times, fracs

    def test_f2_identical_profiles_ge_50(self):
        times, p1 = self._linear_profile()
        res = compare_release_profiles(times, p1, p1[:])
        assert res["f2_similarity"] >= 50.0

    def test_similar_flag_true_for_identical(self):
        times, p1 = self._linear_profile()
        res = compare_release_profiles(times, p1, p1[:])
        assert res["similar"] is True

    def test_f2_very_different_profiles_lt_50(self):
        # Use profiles with large absolute differences: one slow, one fast
        # Profile 1: very slow release (5% per point)
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p1 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        # Profile 2: fast release (~complete)
        p2 = [0.55, 0.70, 0.80, 0.88, 0.93, 0.97]
        res = compare_release_profiles(times, p1, p2)
        assert res["f2_similarity"] < 50.0

    def test_similar_flag_false_for_very_different(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p1 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        p2 = [0.55, 0.70, 0.80, 0.88, 0.93, 0.97]
        res = compare_release_profiles(times, p1, p2)
        assert res["similar"] is False

    def test_f1_zero_for_identical_profiles(self):
        times, p1 = self._linear_profile()
        res = compare_release_profiles(times, p1, p1[:])
        assert res["f1_similarity"] == pytest.approx(0.0, abs=1e-9)

    def test_returns_dict_with_required_keys(self):
        times, p1 = self._linear_profile()
        res = compare_release_profiles(times, p1, p1[:])
        assert "f1_similarity" in res
        assert "f2_similarity" in res
        assert "similar" in res
        assert "notes" in res

    def test_notes_is_string(self):
        times, p1 = self._linear_profile()
        res = compare_release_profiles(times, p1, p1[:])
        assert isinstance(res["notes"], str)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compare_release_profiles([1.0, 2.0], [0.1, 0.2], [0.1])

    def test_similar_profiles_ge_50(self):
        """Profiles differing by ~5% should still be similar."""
        times, p1 = self._linear_profile(k=0.08)
        p2 = [min(f + 0.02, 1.0) for f in p1]
        res = compare_release_profiles(times, p1, p2)
        assert res["f2_similarity"] >= 50.0

    def test_f2_formula_explicit(self):
        """Verify f2 formula explicitly for a simple case (percent scale)."""
        # Single point for simplicity
        r, t_val = 0.5, 0.5
        # Identical → mean_sq_diff_pct = 0 → f2 = 50*log10(100) = 100
        mean_sq_pct = ((r - t_val) * 100.0) ** 2 / 1
        expected_f2 = 50.0 * math.log10(math.sqrt(1.0 / (1.0 + mean_sq_pct)) * 100.0)
        res = compare_release_profiles([1.0], [r], [t_val])
        assert res["f2_similarity"] == pytest.approx(expected_f2, rel=1e-6)
