"""Tests for analysis/pk_population_summary.py — Phase 636."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_population_summary import (
    PopulationPKSummary,
    compute_bsv,
    summarize_pk_parameter,
    summarize_population_pk,
)

# ---------------------------------------------------------------------------
# summarize_pk_parameter — return type and correctness
# ---------------------------------------------------------------------------


def test_returns_population_pk_summary():
    result = summarize_pk_parameter([1.0, 2.0, 3.0, 4.0, 5.0], "cmax")
    assert isinstance(result, PopulationPKSummary)


def test_mean_correct_simple_list():
    result = summarize_pk_parameter([1.0, 2.0, 3.0, 4.0, 5.0], "cmax")
    assert result.mean == pytest.approx(3.0)


def test_median_odd_length():
    result = summarize_pk_parameter([1.0, 2.0, 3.0, 4.0, 5.0], "cmax")
    assert result.median == pytest.approx(3.0)


def test_median_even_length():
    result = summarize_pk_parameter([1.0, 2.0, 3.0, 4.0], "cmax")
    assert result.median == pytest.approx(2.5)


def test_sd_correct_for_known_values():
    # SD of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0 (population), sample SD = sqrt(variance ddof=1)
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    expected_sd = math.sqrt(var)
    result = summarize_pk_parameter(vals, "param")
    assert result.sd == pytest.approx(expected_sd, rel=1e-6)


def test_cv_pct_equals_sd_over_mean_times_100():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = summarize_pk_parameter(vals, "param")
    expected = result.sd / result.mean * 100.0
    assert result.cv_pct == pytest.approx(expected, rel=1e-6)


def test_percentile_ordering():
    vals = [float(i) for i in range(1, 21)]  # 1..20
    result = summarize_pk_parameter(vals, "param")
    assert result.p5 <= result.p25
    assert result.p25 <= result.median
    assert result.median <= result.p75
    assert result.p75 <= result.p95


def test_min_val_is_minimum():
    vals = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    result = summarize_pk_parameter(vals, "param")
    assert result.min_val == pytest.approx(min(vals))


def test_max_val_is_maximum():
    vals = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    result = summarize_pk_parameter(vals, "param")
    assert result.max_val == pytest.approx(max(vals))


def test_geometric_mean_positive_for_all_positive():
    vals = [1.0, 2.0, 4.0, 8.0]
    result = summarize_pk_parameter(vals, "param")
    assert result.geometric_mean > 0.0
    expected = math.exp(sum(math.log(v) for v in vals) / len(vals))
    assert result.geometric_mean == pytest.approx(expected, rel=1e-6)


def test_geometric_mean_zero_when_input_contains_zero():
    vals = [1.0, 0.0, 3.0, 4.0]
    result = summarize_pk_parameter(vals, "param")
    assert result.geometric_mean == 0.0


def test_n_subjects_equals_len_values():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = summarize_pk_parameter(vals, "param")
    assert result.n_subjects == 6


def test_notes_nonempty():
    result = summarize_pk_parameter([1.0, 2.0, 3.0], "cmax")
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_large_cv_population():
    """Values spanning 3 orders of magnitude → CV% > 100."""
    vals = [1.0, 10.0, 100.0]
    result = summarize_pk_parameter(vals, "auc")
    assert result.cv_pct > 100.0


def test_p95_equals_max_for_two_element_list():
    """For 2 elements, p95 should equal the maximum."""
    vals = [1.0, 10.0]
    result = summarize_pk_parameter(vals, "param")
    assert result.p95 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# summarize_population_pk
# ---------------------------------------------------------------------------


def test_summarize_population_pk_returns_dict():
    data = [{"cmax": 1.0, "auc": 10.0}, {"cmax": 2.0, "auc": 20.0}, {"cmax": 3.0, "auc": 30.0}]
    result = summarize_population_pk(data)
    assert isinstance(result, dict)


def test_summarize_population_pk_correct_keys():
    data = [{"cmax": 1.0, "auc": 10.0}, {"cmax": 2.0, "auc": 20.0}, {"cmax": 3.0, "auc": 30.0}]
    result = summarize_population_pk(data)
    assert "cmax" in result
    assert "auc" in result


def test_summarize_population_pk_values_are_summaries():
    data = [{"cmax": 1.0, "auc": 10.0}, {"cmax": 2.0, "auc": 20.0}, {"cmax": 3.0, "auc": 30.0}]
    result = summarize_population_pk(data)
    for v in result.values():
        assert isinstance(v, PopulationPKSummary)


def test_summarize_population_pk_filters_to_requested_parameters():
    data = [
        {"cmax": 1.0, "auc": 10.0, "tmax": 2.0},
        {"cmax": 2.0, "auc": 20.0, "tmax": 3.0},
        {"cmax": 3.0, "auc": 30.0, "tmax": 4.0},
    ]
    result = summarize_population_pk(data, parameters=["cmax"])
    assert list(result.keys()) == ["cmax"]
    assert "auc" not in result


# ---------------------------------------------------------------------------
# compute_bsv
# ---------------------------------------------------------------------------


def test_compute_bsv_returns_dict():
    result = compute_bsv([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(result, dict)


def test_compute_bsv_has_required_keys():
    result = compute_bsv([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "omega_sq" in result
    assert "cv_pct" in result
    assert "iqr" in result


def test_compute_bsv_omega_sq_positive_for_variable_population():
    result = compute_bsv([1.0, 2.0, 4.0, 8.0, 16.0])
    assert result["omega_sq"] > 0.0


def _hazen_pct(sorted_vals: list, pct: float) -> float:
    """Exclusive (Hazen) percentile — same formula as pk_population_summary._percentile."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = pct / 100.0 * (n + 1) - 1.0
    idx = max(0.0, min(float(n - 1), idx))
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi or hi >= n:
        return sorted_vals[min(lo, n - 1)]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def test_compute_bsv_iqr_equals_p75_minus_p25():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    result = compute_bsv(vals)
    sorted_vals = sorted(vals)
    expected_iqr = _hazen_pct(sorted_vals, 75) - _hazen_pct(sorted_vals, 25)
    assert result["iqr"] == pytest.approx(expected_iqr, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_summarize_pk_parameter_empty_raises():
    with pytest.raises(ValueError):
        summarize_pk_parameter([], "cmax")


def test_summarize_pk_parameter_single_raises():
    with pytest.raises(ValueError):
        summarize_pk_parameter([5.0], "cmax")


def test_compute_bsv_empty_raises():
    with pytest.raises(ValueError):
        compute_bsv([])


def test_compute_bsv_single_raises():
    with pytest.raises(ValueError):
        compute_bsv([1.0])
