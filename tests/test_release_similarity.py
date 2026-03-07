"""Tests for Phase 521 — Drug Release Profile Similarity."""

import math

import pytest

from omega_pbpk.biopharmaceutics.release_similarity import (
    DissolutionComparisonResult,
    compare_dissolution_profiles,
    f1_difference,
    f2_similarity,
    weibull_fit,
)

# ---------------------------------------------------------------------------
# f2_similarity
# ---------------------------------------------------------------------------


class TestF2Similarity:
    def test_identical_profiles_f2_100(self):
        ref = [20.0, 40.0, 60.0, 80.0, 95.0]
        f2 = f2_similarity(ref, ref)
        assert abs(f2 - 100.0) < 1e-6

    def test_ten_percent_difference_near_50(self):
        # Small uniform difference — f2 should be close to 50 or above
        ref = [20.0, 40.0, 60.0, 80.0, 100.0]
        tst = [18.0, 37.0, 56.0, 74.0, 91.0]  # ~6-9% difference
        f2 = f2_similarity(ref, tst)
        assert f2 >= 40.0  # meaningful similarity range

    def test_dissimilar_profiles_f2_below_50(self):
        ref = [20.0, 40.0, 60.0, 80.0, 100.0]
        tst = [5.0, 10.0, 20.0, 35.0, 50.0]  # very different
        f2 = f2_similarity(ref, tst)
        assert f2 < 50.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            f2_similarity([10.0, 20.0], [10.0])

    def test_value_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            f2_similarity([10.0, 110.0], [10.0, 50.0])

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            f2_similarity([-5.0, 50.0], [10.0, 50.0])

    def test_f2_always_positive(self):
        ref = [10.0, 20.0, 30.0]
        tst = [80.0, 90.0, 95.0]
        f2 = f2_similarity(ref, tst)
        assert f2 > 0

    def test_f2_single_timepoint(self):
        f2 = f2_similarity([50.0], [50.0])
        assert abs(f2 - 100.0) < 1e-6

    def test_f2_near_50_threshold(self):
        """Profiles with ~10% avg difference should yield f2 around 50."""
        ref = [30.0, 50.0, 70.0, 90.0]
        tst = [20.0, 40.0, 60.0, 80.0]
        f2 = f2_similarity(ref, tst)
        # 10 unit difference each, f2 = 50*log10(100/(1+25)^0.5)
        assert 40.0 <= f2 <= 60.0


# ---------------------------------------------------------------------------
# f1_difference
# ---------------------------------------------------------------------------


class TestF1Difference:
    def test_identical_profiles_f1_zero(self):
        ref = [20.0, 40.0, 60.0, 80.0]
        assert f1_difference(ref, ref) == 0.0

    def test_f1_acceptable_for_similar(self):
        ref = [20.0, 40.0, 60.0, 80.0]
        tst = [18.0, 38.0, 58.0, 78.0]
        f1 = f1_difference(ref, tst)
        assert f1 <= 15.0

    def test_f1_large_difference(self):
        ref = [20.0, 40.0, 60.0, 80.0]
        tst = [5.0, 10.0, 20.0, 40.0]
        f1 = f1_difference(ref, tst)
        assert f1 > 15.0

    def test_f1_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            f1_difference([10.0, 20.0], [10.0])

    def test_f1_value_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            f1_difference([10.0, 105.0], [10.0, 50.0])


# ---------------------------------------------------------------------------
# weibull_fit
# ---------------------------------------------------------------------------


class TestWeibullFit:
    def _make_weibull_data(self, alpha, beta, times):
        fracs = [1.0 - math.exp(-((t / alpha) ** beta)) for t in times]
        return fracs

    def test_returns_expected_keys(self):
        times = [1.0, 2.0, 4.0, 8.0, 12.0]
        fracs = self._make_weibull_data(4.0, 1.0, times)
        result = weibull_fit(times, fracs)
        assert set(result.keys()) == {"alpha", "beta", "r_squared", "td50"}

    def test_r_squared_high_for_clean_data(self):
        times = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        fracs = self._make_weibull_data(5.0, 1.2, times)
        result = weibull_fit(times, fracs)
        assert result["r_squared"] >= 0.90

    def test_alpha_positive(self):
        times = [0.5, 1.0, 2.0, 4.0, 8.0]
        fracs = self._make_weibull_data(3.0, 1.0, times)
        result = weibull_fit(times, fracs)
        assert result["alpha"] > 0

    def test_beta_positive(self):
        times = [0.5, 1.0, 2.0, 4.0, 8.0]
        fracs = self._make_weibull_data(3.0, 1.0, times)
        result = weibull_fit(times, fracs)
        assert result["beta"] > 0

    def test_td50_positive(self):
        times = [1.0, 2.0, 4.0, 8.0, 12.0]
        fracs = self._make_weibull_data(4.0, 1.0, times)
        result = weibull_fit(times, fracs)
        assert result["td50"] > 0

    def test_constant_data_r_squared_one(self):
        times = [1.0, 2.0, 3.0]
        fracs = [0.5, 0.5, 0.5]
        result = weibull_fit(times, fracs)
        assert result["r_squared"] == 1.0


# ---------------------------------------------------------------------------
# compare_dissolution_profiles
# ---------------------------------------------------------------------------


class TestCompareDissolutionProfiles:
    def test_identical_profiles_similar(self):
        ref_t = [15.0, 30.0, 45.0, 60.0]
        ref_p = [30.0, 55.0, 75.0, 90.0]
        result = compare_dissolution_profiles(ref_t, ref_p, ref_t, ref_p)
        assert isinstance(result, DissolutionComparisonResult)
        assert result.similar is True
        assert result.acceptable_f1 is True

    def test_dissimilar_profiles_flagged(self):
        ref_t = [15.0, 30.0, 45.0, 60.0]
        ref_p = [30.0, 55.0, 75.0, 90.0]
        test_p = [5.0, 15.0, 30.0, 50.0]
        result = compare_dissolution_profiles(ref_t, ref_p, ref_t, test_p)
        assert result.similar is False

    def test_returns_weibull_dicts(self):
        ref_t = [15.0, 30.0, 45.0, 60.0]
        ref_p = [30.0, 55.0, 75.0, 90.0]
        result = compare_dissolution_profiles(ref_t, ref_p, ref_t, ref_p)
        assert "alpha" in result.reference_weibull
        assert "alpha" in result.test_weibull

    def test_spec_pass(self):
        ref_t = [30.0, 60.0]
        ref_p = [60.0, 90.0]
        tst_p = [62.0, 91.0]
        result = compare_dissolution_profiles(
            ref_t,
            ref_p,
            ref_t,
            tst_p,
            specification_times=[30.0, 60.0],
            specification_min_pct=[50.0, 80.0],
        )
        assert "meets all specifications" in result.notes

    def test_spec_fail(self):
        ref_t = [30.0, 60.0]
        ref_p = [60.0, 90.0]
        tst_p = [40.0, 70.0]  # fails spec
        result = compare_dissolution_profiles(
            ref_t,
            ref_p,
            ref_t,
            tst_p,
            specification_times=[30.0],
            specification_min_pct=[50.0],
        )
        assert "fails spec" in result.notes

    def test_interpolation_different_timepoints(self):
        ref_t = [10.0, 20.0, 30.0]
        ref_p = [30.0, 60.0, 90.0]
        tst_t = [5.0, 15.0, 25.0, 35.0]
        tst_p = [20.0, 45.0, 70.0, 95.0]
        result = compare_dissolution_profiles(ref_t, ref_p, tst_t, tst_p)
        # Should not raise; f2 should be reasonable
        assert result.f2 > 0

    def test_f1_f2_values_present(self):
        ref_t = [30.0, 60.0, 90.0]
        ref_p = [50.0, 75.0, 90.0]
        result = compare_dissolution_profiles(ref_t, ref_p, ref_t, ref_p)
        assert result.f1 == 0.0
        assert abs(result.f2 - 100.0) < 1e-6
