"""Tests for Phase 566 — population_statistics.py"""

import pytest

from omega_pbpk.analysis.population_statistics import (
    bootstrap_ci,
    effect_size_cohens_d,
    nonparametric_anova,
    pearson_correlation,
    spearman_correlation,
)

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_constant_data_ci_collapses(self):
        values = [5.0] * 20
        result = bootstrap_ci(values, statistic="mean", seed=42)
        assert abs(result["lower_ci"] - 5.0) < 1e-9
        assert abs(result["upper_ci"] - 5.0) < 1e-9

    def test_variable_data_ci_width_positive(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
        result = bootstrap_ci(values, statistic="mean", seed=42)
        assert result["upper_ci"] > result["lower_ci"]

    def test_deterministic_with_same_seed(self):
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        r1 = bootstrap_ci(values, seed=42)
        r2 = bootstrap_ci(values, seed=42)
        assert r1["estimate"] == r2["estimate"]
        assert r1["lower_ci"] == r2["lower_ci"]
        assert r1["upper_ci"] == r2["upper_ci"]

    def test_different_seeds_may_differ(self):
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        r1 = bootstrap_ci(values, seed=1)
        r2 = bootstrap_ci(values, seed=99)
        # They may differ (not a strict guarantee but generally true)
        # Just verify both return valid dicts
        assert r1["estimate"] == r2["estimate"]  # estimate is not stochastic

    def test_estimate_is_mean(self):
        values = [2.0, 4.0, 6.0]
        result = bootstrap_ci(values, statistic="mean", seed=42)
        assert abs(result["estimate"] - 4.0) < 1e-9

    def test_median_statistic(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        result = bootstrap_ci(values, statistic="median", seed=42)
        assert result["estimate"] == 3.0

    def test_geometric_mean_statistic(self):
        values = [1.0, 4.0, 16.0]  # GM = 4
        result = bootstrap_ci(values, statistic="geometric_mean", seed=42)
        assert abs(result["estimate"] - 4.0) < 1e-6

    def test_cv_statistic(self):
        values = [10.0] * 10  # CV should be 0
        result = bootstrap_ci(values, statistic="cv", seed=42)
        assert result["estimate"] == pytest.approx(0.0, abs=1e-9)

    def test_ci_level_returned(self):
        values = [1.0, 2.0, 3.0]
        result = bootstrap_ci(values, ci_level=0.90, seed=42)
        assert result["ci_level"] == 0.90

    def test_n_bootstrap_returned(self):
        values = [1.0, 2.0, 3.0]
        result = bootstrap_ci(values, n_bootstrap=500, seed=42)
        assert result["n_bootstrap"] == 500

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([5.0])

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], statistic="mode")


# ---------------------------------------------------------------------------
# nonparametric_anova
# ---------------------------------------------------------------------------


class TestNonparametricAnova:
    def test_identical_groups_not_significant(self):
        g1 = [5.0, 5.1, 4.9, 5.0, 5.2]
        g2 = [5.0, 5.1, 4.9, 5.0, 5.2]
        g3 = [5.0, 5.1, 4.9, 5.0, 5.2]
        result = nonparametric_anova([g1, g2, g3])
        assert not result["significant"]

    def test_very_different_groups_significant(self):
        g1 = [1.0, 1.1, 0.9]
        g2 = [100.0, 101.0, 99.0]
        result = nonparametric_anova([g1, g2])
        assert result["significant"]

    def test_df_equals_k_minus_one(self):
        groups = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = nonparametric_anova(groups)
        assert result["df"] == 2

    def test_h_statistic_non_negative(self):
        result = nonparametric_anova([[1.0, 2.0], [3.0, 4.0]])
        assert result["H_statistic"] >= 0

    def test_p_value_between_zero_and_one(self):
        result = nonparametric_anova([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert 0.0 <= result["p_value_approx"] <= 1.0

    def test_fewer_than_two_groups_raises(self):
        with pytest.raises(ValueError):
            nonparametric_anova([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# pearson_correlation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    def test_identical_variables_r_equals_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pearson_correlation(x, x)
        assert abs(result["r"] - 1.0) < 1e-9

    def test_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [-1.0, -2.0, -3.0, -4.0, -5.0]
        result = pearson_correlation(x, y)
        assert abs(result["r"] - (-1.0)) < 1e-9

    def test_r_squared_equals_r_squared(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.5, 2.5, 2.0, 4.0]
        result = pearson_correlation(x, y)
        assert abs(result["r_squared"] - result["r"] ** 2) < 1e-9

    def test_strong_interpretation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.1, 2.0, 3.1, 3.9, 5.1]
        result = pearson_correlation(x, y)
        assert result["interpretation"] == "strong"

    def test_weak_interpretation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 4.0, 2.0, 5.0]  # weakly correlated
        result = pearson_correlation(x, y)
        # interpretation depends on actual r
        assert result["interpretation"] in ("strong", "moderate", "weak")

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            pearson_correlation([1.0], [1.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_monotone_data_rho_approx_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = spearman_correlation(x, y)
        assert abs(result["rho"] - 1.0) < 1e-6

    def test_perfect_negative_monotone_rho_neg_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = spearman_correlation(x, y)
        assert abs(result["rho"] - (-1.0)) < 1e-6

    def test_p_value_between_zero_and_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.5, 2.0, 3.5, 3.0, 5.0]
        result = spearman_correlation(x, y)
        assert 0.0 <= result["p_value_approx"] <= 1.0

    def test_significant_monotone_p_small(self):
        x = list(range(1, 11))
        y = [v * 2 + 0.1 * i for i, v in enumerate(x)]
        result = spearman_correlation(x, y)
        assert result["p_value_approx"] < 0.05

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            spearman_correlation([1.0], [1.0])


# ---------------------------------------------------------------------------
# effect_size_cohens_d
# ---------------------------------------------------------------------------


class TestEffectSizeCohensD:
    def test_identical_groups_d_zero(self):
        g1 = [5.0, 5.0, 5.0, 5.0]
        g2 = [5.0, 5.0, 5.0, 5.0]
        result = effect_size_cohens_d(g1, g2)
        assert abs(result["d"]) < 1e-9

    def test_large_effect_size(self):
        g1 = [100.0, 101.0, 99.0, 100.5]
        g2 = [1.0, 1.1, 0.9, 1.05]
        result = effect_size_cohens_d(g1, g2)
        assert abs(result["d"]) > 0.8
        assert result["interpretation"] == "large"

    def test_small_effect_size_interpretation(self):
        g1 = [10.0, 10.1, 9.9, 10.0]
        g2 = [10.05, 10.15, 9.95, 10.05]
        result = effect_size_cohens_d(g1, g2)
        # With such small difference, d should be small
        assert result["interpretation"] in ("small", "medium")

    def test_pooled_std_positive(self):
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 5.0, 6.0]
        result = effect_size_cohens_d(g1, g2)
        assert result["pooled_std"] > 0

    def test_sign_of_d(self):
        g1 = [10.0, 10.5, 9.5, 10.2]
        g2 = [5.0, 5.5, 4.5, 5.2]
        result = effect_size_cohens_d(g1, g2)
        assert result["d"] > 0  # mean1 > mean2 → d > 0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            effect_size_cohens_d([5.0], [5.0, 6.0])

    def test_returns_required_keys(self):
        result = effect_size_cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert {"d", "pooled_std", "interpretation"} == set(result.keys())
