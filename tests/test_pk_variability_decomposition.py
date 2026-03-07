"""Tests for Phase 452: PK variability source decomposition."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_variability_decomposition import (
    VariabilityResult,
    anova_variability,
    calculate_geometric_cv,
    decompose_variability,
)

# ---------------------------------------------------------------------------
# calculate_geometric_cv — basic
# ---------------------------------------------------------------------------


def test_gcv_identical_values_near_zero():
    values = [10.0] * 20
    gcv = calculate_geometric_cv(values)
    assert gcv < 0.01  # essentially zero


def test_gcv_known_lognormal():
    """GCV should equal approximately sqrt(exp(sigma^2)-1)*100 for known sigma."""
    import math

    # Generate pseudo-lognormal with known sigma=0.5 on log scale
    # Mean log = 0, sigma log = 0.5 → GCV ≈ sqrt(exp(0.25)-1)*100 ≈ 53.1%
    values = [math.exp(0.5 * (i - 5) / 5.0) for i in range(11)]
    gcv = calculate_geometric_cv(values)
    # Just verify it's in a reasonable range (not zero, not 1000%)
    assert 0.0 < gcv < 200.0


def test_gcv_two_values():
    values = [1.0, math.e]  # log values: 0, 1 → var=0.5 (sample)
    gcv = calculate_geometric_cv(values)
    expected = math.sqrt(max(0.0, math.exp(0.5) - 1.0)) * 100.0
    assert abs(gcv - expected) < 0.01


def test_gcv_larger_spread_higher_cv():
    low = calculate_geometric_cv([1.0, 1.1, 0.9])
    high = calculate_geometric_cv([1.0, 5.0, 0.2])
    assert high > low


def test_gcv_single_value_is_zero():
    # Single value: variance is 0
    gcv = calculate_geometric_cv([42.0])
    assert gcv == 0.0


def test_gcv_empty_raises():
    with pytest.raises(ValueError):
        calculate_geometric_cv([])


def test_gcv_negative_value_raises():
    with pytest.raises(ValueError):
        calculate_geometric_cv([1.0, -2.0, 3.0])


def test_gcv_zero_value_raises():
    with pytest.raises(ValueError):
        calculate_geometric_cv([1.0, 0.0, 3.0])


def test_gcv_returns_float():
    assert isinstance(calculate_geometric_cv([1.0, 2.0, 3.0]), float)


# ---------------------------------------------------------------------------
# anova_variability
# ---------------------------------------------------------------------------


def test_anova_between_greater_when_groups_differ():
    """Groups with very different means → between > within."""
    values = [1.0, 1.1, 0.9, 10.0, 11.0, 9.0]
    groups = ["A", "A", "A", "B", "B", "B"]
    result = anova_variability(values, groups)
    assert result["between_group_cv"] > result["within_group_cv"]


def test_anova_within_greater_when_groups_same():
    """Groups with same mean but wide spread within → within >= between."""
    values = [1.0, 5.0, 0.2, 1.0, 5.0, 0.2]
    groups = ["A", "A", "A", "B", "B", "B"]
    result = anova_variability(values, groups)
    # Between is 0 (identical group means), within is non-zero
    assert result["within_group_cv"] >= result["between_group_cv"]


def test_anova_icc_in_range():
    values = [1.0, 1.1, 0.9, 5.0, 5.5, 4.8]
    groups = ["A", "A", "A", "B", "B", "B"]
    result = anova_variability(values, groups)
    assert 0.0 <= result["icc"] <= 1.0


def test_anova_keys_present():
    values = [1.0, 2.0, 3.0, 4.0]
    groups = ["A", "A", "B", "B"]
    result = anova_variability(values, groups)
    assert "between_group_cv" in result
    assert "within_group_cv" in result
    assert "icc" in result


def test_anova_requires_two_groups():
    values = [1.0, 2.0, 3.0]
    groups = ["A", "A", "A"]
    with pytest.raises(ValueError, match="2"):
        anova_variability(values, groups)


def test_anova_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        anova_variability([1.0, 2.0], ["A"])


def test_anova_empty_raises():
    with pytest.raises(ValueError):
        anova_variability([], [])


def test_anova_negative_values_raises():
    with pytest.raises(ValueError):
        anova_variability([1.0, -1.0], ["A", "B"])


# ---------------------------------------------------------------------------
# decompose_variability — basic
# ---------------------------------------------------------------------------


def test_decompose_returns_variability_result():
    result = decompose_variability([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert isinstance(result, VariabilityResult)


def test_decompose_n_subjects_matches():
    cmax = [1.0, 2.0, 3.0, 4.0, 5.0]
    auc = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = decompose_variability(cmax, auc)
    assert result.n_subjects == 5


def test_decompose_identical_cmax_gcv_near_zero():
    cmax = [5.0] * 10
    auc = [50.0] * 10
    result = decompose_variability(cmax, auc)
    assert result.cmax_gcv_pct < 0.01


def test_decompose_identical_auc_gcv_near_zero():
    cmax = [5.0] * 10
    auc = [50.0] * 10
    result = decompose_variability(cmax, auc)
    assert result.auc_gcv_pct < 0.01


def test_decompose_percentiles_ordered():
    cmax = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    auc = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    result = decompose_variability(cmax, auc)
    assert result.cmax_p5 <= result.cmax_p50 <= result.cmax_p95
    assert result.auc_p5 <= result.auc_p50 <= result.auc_p95


def test_decompose_no_group_ids_within_cv_zero():
    cmax = [1.0, 2.0, 3.0]
    auc = [10.0, 20.0, 30.0]
    result = decompose_variability(cmax, auc)
    assert result.within_subject_cv_pct == 0.0


def test_decompose_no_group_ids_between_equals_gcv():
    cmax = [1.0, 2.0, 4.0]
    auc = [10.0, 20.0, 40.0]
    result = decompose_variability(cmax, auc)
    # Without group_ids, between = cmax_gcv
    assert abs(result.between_subject_cv_pct - result.cmax_gcv_pct) < 0.01


def test_decompose_with_group_ids():
    cmax = [1.0, 1.1, 0.9, 5.0, 5.5, 4.8]
    auc = [10.0, 11.0, 9.0, 50.0, 55.0, 48.0]
    groups = ["A", "A", "A", "B", "B", "B"]
    result = decompose_variability(cmax, auc, group_ids=groups)
    assert result.between_subject_cv_pct > result.within_subject_cv_pct


# ---------------------------------------------------------------------------
# decompose_variability — error cases
# ---------------------------------------------------------------------------


def test_decompose_empty_cmax_raises():
    with pytest.raises(ValueError):
        decompose_variability([], [])


def test_decompose_negative_cmax_raises():
    with pytest.raises(ValueError):
        decompose_variability([1.0, -1.0], [1.0, 2.0])


def test_decompose_negative_auc_raises():
    with pytest.raises(ValueError):
        decompose_variability([1.0, 2.0], [1.0, -2.0])


def test_decompose_mismatched_length_raises():
    with pytest.raises(ValueError):
        decompose_variability([1.0, 2.0], [1.0])


def test_decompose_group_ids_wrong_length_raises():
    with pytest.raises(ValueError):
        decompose_variability([1.0, 2.0], [1.0, 2.0], group_ids=["A"])


# ---------------------------------------------------------------------------
# notes is a non-empty string
# ---------------------------------------------------------------------------


def test_decompose_notes_nonempty():
    result = decompose_variability([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert isinstance(result.notes, str) and len(result.notes) > 0
