"""Tests for analysis.variability_sources (Phase 391)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.variability_sources import (
    bootstrap_ci_cv,
    partition_variability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_identical_groups() -> list[list[float]]:
    """Three groups with identical values — no between-group variability."""
    return [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]


def make_separated_groups() -> list[list[float]]:
    """Three groups with very different means — high between-group variability."""
    return [[1.0, 1.1, 0.9], [5.0, 5.1, 4.9], [10.0, 10.1, 9.9]]


# ---------------------------------------------------------------------------
# VariabilityResult dataclass
# ---------------------------------------------------------------------------


class TestVariabilityResult:
    def test_is_frozen(self):
        vr = partition_variability("CL", [[1.0, 2.0], [3.0, 4.0]], ["A", "B"])
        with pytest.raises((AttributeError, TypeError)):
            vr.param_name = "new"  # type: ignore[misc]

    def test_has_required_fields(self):
        vr = partition_variability("CL", [[1.0, 2.0]], ["A"])
        assert hasattr(vr, "param_name")
        assert hasattr(vr, "source_names")
        assert hasattr(vr, "n_groups")
        assert hasattr(vr, "within_cv_pct")
        assert hasattr(vr, "between_cv_pct")
        assert hasattr(vr, "total_cv_pct")
        assert hasattr(vr, "variance_fraction_within")
        assert hasattr(vr, "variance_fraction_between")
        assert hasattr(vr, "notes")


# ---------------------------------------------------------------------------
# partition_variability — input validation
# ---------------------------------------------------------------------------


class TestPartitionVariabilityValidation:
    def test_empty_param_name_raises(self):
        with pytest.raises(ValueError, match="param_name"):
            partition_variability("", [[1.0, 2.0]], ["A"])

    def test_blank_param_name_raises(self):
        with pytest.raises(ValueError, match="param_name"):
            partition_variability("   ", [[1.0, 2.0]], ["A"])

    def test_empty_values_list_raises(self):
        with pytest.raises(ValueError, match="values_list"):
            partition_variability("CL", [], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            partition_variability("CL", [[1.0, 2.0], [3.0, 4.0]], ["only_one"])

    def test_empty_group_raises(self):
        with pytest.raises(ValueError):
            partition_variability("CL", [[1.0, 2.0], []], ["A", "B"])

    def test_nan_value_raises(self):
        with pytest.raises(ValueError):
            partition_variability("CL", [[1.0, float("nan")]], ["A"])

    def test_inf_value_raises(self):
        with pytest.raises(ValueError):
            partition_variability("CL", [[1.0, float("inf")]], ["A"])


# ---------------------------------------------------------------------------
# partition_variability — correctness
# ---------------------------------------------------------------------------


class TestPartitionVariabilityCorrectness:
    def test_fractions_sum_to_one(self):
        grps = make_separated_groups()
        vr = partition_variability("AUC", grps, ["Low", "Mid", "High"])
        total = vr.variance_fraction_within + vr.variance_fraction_between
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_fractions_sum_to_one_single_group(self):
        vr = partition_variability("AUC", [[1.0, 2.0, 3.0]], ["A"])
        total = vr.variance_fraction_within + vr.variance_fraction_between
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_between_fraction_high_when_groups_separated(self):
        grps = make_separated_groups()
        vr = partition_variability("CL", grps, ["A", "B", "C"])
        assert vr.variance_fraction_between > 0.9

    def test_within_fraction_high_when_groups_identical_means(self):
        grps = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
        vr = partition_variability("CL", grps, ["A", "B"])
        # Groups have same mean → between_SS = 0, within contains all variance
        assert vr.variance_fraction_between == pytest.approx(0.0, abs=1e-9)
        assert vr.variance_fraction_within == pytest.approx(1.0, abs=1e-9)

    def test_n_groups_correct(self):
        grps = make_separated_groups()
        vr = partition_variability("Vd", grps, ["X", "Y", "Z"])
        assert vr.n_groups == 3

    def test_param_name_preserved(self):
        vr = partition_variability("my_param", [[1.0, 2.0]], ["G1"])
        assert vr.param_name == "my_param"

    def test_source_names_preserved(self):
        vr = partition_variability("CL", [[1.0], [2.0]], ["site1", "site2"])
        assert vr.source_names == ["site1", "site2"]

    def test_total_cv_pct_positive(self):
        grps = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        vr = partition_variability("AUC", grps, ["A", "B"])
        assert vr.total_cv_pct > 0.0

    def test_total_cv_zero_for_constant_values(self):
        grps = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        vr = partition_variability("AUC", grps, ["A", "B"])
        assert vr.total_cv_pct == pytest.approx(0.0, abs=1e-9)
        assert vr.within_cv_pct == pytest.approx(0.0, abs=1e-9)
        assert vr.between_cv_pct == pytest.approx(0.0, abs=1e-9)

    def test_variance_fractions_in_unit_interval(self):
        grps = make_separated_groups()
        vr = partition_variability("Cmax", grps, ["A", "B", "C"])
        assert 0.0 <= vr.variance_fraction_within <= 1.0
        assert 0.0 <= vr.variance_fraction_between <= 1.0

    def test_notes_is_non_empty_string(self):
        vr = partition_variability("CL", [[1.0, 2.0]], ["A"])
        assert isinstance(vr.notes, str) and len(vr.notes) > 0

    def test_within_cv_pct_non_negative(self):
        grps = [[1.0, 1.5, 2.0], [3.0, 3.5, 4.0]]
        vr = partition_variability("F", grps, ["A", "B"])
        assert vr.within_cv_pct >= 0.0

    def test_between_cv_pct_non_negative(self):
        grps = [[1.0, 1.5, 2.0], [3.0, 3.5, 4.0]]
        vr = partition_variability("F", grps, ["A", "B"])
        assert vr.between_cv_pct >= 0.0

    def test_single_group_no_between_variance(self):
        vr = partition_variability("CL", [[1.0, 2.0, 3.0]], ["A"])
        assert vr.variance_fraction_between == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# bootstrap_ci_cv — input validation
# ---------------------------------------------------------------------------


class TestBootstrapCICVValidation:
    def test_single_value_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci_cv([5.0])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci_cv([])

    def test_n_bootstrap_zero_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_ci_cv([1.0, 2.0, 3.0], n_bootstrap=0)

    def test_ci_pct_zero_raises(self):
        with pytest.raises(ValueError, match="ci_pct"):
            bootstrap_ci_cv([1.0, 2.0, 3.0], ci_pct=0.0)

    def test_ci_pct_100_raises(self):
        with pytest.raises(ValueError, match="ci_pct"):
            bootstrap_ci_cv([1.0, 2.0, 3.0], ci_pct=100.0)

    def test_nan_value_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci_cv([1.0, float("nan"), 3.0])


# ---------------------------------------------------------------------------
# bootstrap_ci_cv — correctness
# ---------------------------------------------------------------------------


class TestBootstrapCICVCorrectness:
    def test_returns_tuple_of_two_floats(self):
        lo, hi = bootstrap_ci_cv([10.0, 20.0, 30.0, 25.0, 15.0])
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_le_upper(self):
        lo, hi = bootstrap_ci_cv([10.0, 20.0, 30.0, 25.0, 15.0])
        assert lo <= hi

    def test_ci_contains_observed_cv(self):
        values = [10.0, 15.0, 8.0, 12.0, 11.0, 9.0, 13.0, 16.0]
        # Observed CV%
        mean = sum(values) / len(values)
        n = len(values)
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        obs_cv = math.sqrt(var) / mean * 100.0
        lo, hi = bootstrap_ci_cv(values, n_bootstrap=500, ci_pct=90.0)
        assert lo <= obs_cv <= hi

    def test_reproducible_with_fixed_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = bootstrap_ci_cv(values, n_bootstrap=200)
        r2 = bootstrap_ci_cv(values, n_bootstrap=200)
        assert r1 == r2

    def test_wider_ci_for_lower_confidence(self):
        values = [10.0, 15.0, 8.0, 12.0, 11.0, 9.0, 13.0, 16.0, 14.0, 7.0]
        lo90, hi90 = bootstrap_ci_cv(values, ci_pct=90.0)
        lo50, hi50 = bootstrap_ci_cv(values, ci_pct=50.0)
        # 90% CI should be at least as wide as 50% CI
        assert (hi90 - lo90) >= (hi50 - lo50)

    def test_ci_non_negative(self):
        values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        lo, hi = bootstrap_ci_cv(values)
        assert lo >= 0.0
        assert hi >= 0.0
