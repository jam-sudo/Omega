"""Tests for core/zero_order_release.py — Phase 573."""

import math

import pytest

from omega_pbpk.core.zero_order_release import (
    compare_release_profiles,
    first_order_release,
    mixed_order_release,
    weibull_release,
    zero_order_release,
)

# ---------------------------------------------------------------------------
# zero_order_release
# ---------------------------------------------------------------------------


class TestZeroOrderRelease:
    def test_basic_return_keys(self):
        result = zero_order_release(100.0, release_duration_h=10.0)
        assert "times_h" in result
        assert "cumulative_released_mg" in result
        assert "release_rate_mg_per_h" in result
        assert "f_released_at_end" in result

    def test_f_released_at_end_equals_one(self):
        result = zero_order_release(100.0, release_duration_h=10.0)
        assert abs(result["f_released_at_end"] - 1.0) < 0.02

    def test_cumulative_is_linear_during_release(self):
        """During the release window, cumulative should increase at constant rate."""
        result = zero_order_release(100.0, release_duration_h=10.0, dt_h=0.1)
        times = result["times_h"]
        cum = result["cumulative_released_mg"]
        # Check linearity: differences should be approx equal during release
        diffs = []
        for i in range(1, len(times)):
            if times[i - 1] < 10.0 and times[i] <= 10.0:
                diffs.append(cum[i] - cum[i - 1])
        if len(diffs) > 2:
            avg = sum(diffs) / len(diffs)
            for d in diffs:
                assert abs(d - avg) < 0.01 * avg + 1e-9

    def test_release_stops_after_duration(self):
        result = zero_order_release(100.0, release_duration_h=5.0, t_end_h=10.0, dt_h=0.1)
        times = result["times_h"]
        rates = result["release_rate_mg_per_h"]
        for i, t in enumerate(times):
            if t > 5.0:
                assert rates[i] == 0.0

    def test_default_t_end_is_1_5x_duration(self):
        result = zero_order_release(100.0, release_duration_h=10.0)
        assert result["times_h"][-1] >= 14.9  # 10 * 1.5

    def test_cumulative_monotonically_increasing(self):
        result = zero_order_release(50.0, release_duration_h=8.0)
        cum = result["cumulative_released_mg"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            zero_order_release(-10.0, release_duration_h=5.0)


# ---------------------------------------------------------------------------
# first_order_release
# ---------------------------------------------------------------------------


class TestFirstOrderRelease:
    def test_basic_return_keys(self):
        result = first_order_release(100.0, k_release_per_h=0.2)
        assert "times_h" in result
        assert "cumulative_released_mg" in result
        assert "a_remaining_mg" in result
        assert "f_released_at_end" in result

    def test_cumulative_monotonically_increasing(self):
        result = first_order_release(100.0, k_release_per_h=0.3)
        cum = result["cumulative_released_mg"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_approaches_one_asymptotically(self):
        """After long time, f_released should be close to 1."""
        result = first_order_release(100.0, k_release_per_h=1.0, t_end_h=20.0)
        assert result["f_released_at_end"] > 0.99

    def test_after_3_halflives_87pct_released(self):
        """t_half = ln2/k; after 3 t_half, >87% released."""
        k = 0.5
        t_half = math.log(2) / k
        t_3half = 3 * t_half
        result = first_order_release(100.0, k_release_per_h=k, t_end_h=t_3half + 0.1)
        assert result["f_released_at_end"] > 0.87

    def test_remaining_plus_cumulative_equals_dose(self):
        result = first_order_release(200.0, k_release_per_h=0.1, t_end_h=10.0)
        for i in range(len(result["times_h"])):
            total = result["a_remaining_mg"][i] + result["cumulative_released_mg"][i]
            assert abs(total - 200.0) < 1.0  # small numerical drift allowed

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            first_order_release(-50.0, k_release_per_h=0.2)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            first_order_release(100.0, k_release_per_h=0.0)


# ---------------------------------------------------------------------------
# mixed_order_release
# ---------------------------------------------------------------------------


class TestMixedOrderRelease:
    def test_basic_return_keys(self):
        result = mixed_order_release(100.0, k_zero=1.0, k_first=0.1)
        assert "times_h" in result
        assert "cumulative_released_mg" in result
        assert "a_remaining_mg" in result

    def test_cumulative_monotonically_increasing(self):
        result = mixed_order_release(100.0, k_zero=0.5, k_first=0.1)
        cum = result["cumulative_released_mg"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_mixed_order_releases_more_than_zero_order_alone(self):
        """Mixed (k_zero + k_first) releases more than pure zero-order."""
        t_end = 10.0
        mix = mixed_order_release(100.0, k_zero=1.0, k_first=0.2, t_end_h=t_end)
        pure_zero = mixed_order_release(100.0, k_zero=1.0, k_first=0.0, t_end_h=t_end)
        assert mix["cumulative_released_mg"][-1] >= pure_zero["cumulative_released_mg"][-1]

    def test_starts_at_zero(self):
        result = mixed_order_release(100.0, k_zero=1.0, k_first=0.1)
        assert result["cumulative_released_mg"][0] == 0.0

    def test_remaining_non_negative(self):
        result = mixed_order_release(50.0, k_zero=2.0, k_first=0.5, t_end_h=30.0)
        for a in result["a_remaining_mg"]:
            assert a >= 0.0

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            mixed_order_release(-10.0, k_zero=1.0, k_first=0.1)


# ---------------------------------------------------------------------------
# weibull_release
# ---------------------------------------------------------------------------


class TestWeibullRelease:
    def test_basic_return_keys(self):
        result = weibull_release(100.0, scale_h=8.0, shape=1.5)
        assert "times_h" in result
        assert "cumulative_released_mg" in result
        assert "f_released" in result

    def test_starts_at_zero(self):
        result = weibull_release(100.0, scale_h=8.0, shape=1.5)
        assert result["f_released"][0] == 0.0
        assert result["cumulative_released_mg"][0] == 0.0

    def test_shape_1_is_exponential(self):
        """With shape=1, Weibull = 1 - exp(-t/scale), i.e. first-order."""
        result = weibull_release(100.0, scale_h=5.0, shape=1.0, t_end_h=20.0, n_points=100)
        # At t=scale, F should be 1-exp(-1) ≈ 0.632
        scale = 5.0
        idx = None
        for i, t in enumerate(result["times_h"]):
            if abs(t - scale) < 0.3:
                idx = i
                break
        if idx is not None:
            assert abs(result["f_released"][idx] - (1 - math.exp(-1))) < 0.05

    def test_shape_greater_than_1_sigmoidal(self):
        """With shape>1, the release profile has an inflection point (sigmoidal)."""
        # For shape>1, initial release rate is slow (concave up initially)
        result = weibull_release(100.0, scale_h=8.0, shape=3.0, t_end_h=24.0, n_points=200)
        # At early times (t=1), release should be small
        f_early = result["f_released"][5]  # near t=1h
        f_mid = result["f_released"][100]  # around t=12h
        assert f_early < f_mid

    def test_cumulative_monotonically_increasing(self):
        result = weibull_release(100.0, scale_h=8.0, shape=1.5)
        cum = result["cumulative_released_mg"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            weibull_release(-10.0, scale_h=8.0, shape=1.5)


# ---------------------------------------------------------------------------
# compare_release_profiles
# ---------------------------------------------------------------------------


class TestCompareReleaseProfiles:
    def test_returns_three_entries(self):
        results = compare_release_profiles(100.0, t_end_h=24.0)
        assert len(results) == 3

    def test_model_names(self):
        results = compare_release_profiles(100.0)
        models = {r["model"] for r in results}
        assert "zero_order" in models
        assert "first_order" in models
        assert "weibull" in models

    def test_f_released_8h_between_zero_and_one(self):
        results = compare_release_profiles(100.0)
        for r in results:
            assert 0.0 <= r["f_released_8h"] <= 1.0

    def test_f_released_end_between_zero_and_one(self):
        results = compare_release_profiles(100.0)
        for r in results:
            assert 0.0 <= r["f_released_end"] <= 1.0

    def test_profile_is_list(self):
        results = compare_release_profiles(100.0)
        for r in results:
            assert isinstance(r["profile"], list)
            assert len(r["profile"]) > 0

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            compare_release_profiles(-100.0)
