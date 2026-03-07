"""Tests for analysis/variability_decomposition.py — Phase 647."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.variability_decomposition import (
    VariabilityDecomposition,
    compare_variability_sources,
    decompose_variability,
)

# ---------------------------------------------------------------------------
# Helpers to build test datasets
# ---------------------------------------------------------------------------


def _make_data(subject_values: dict[str, list[float]]) -> list[dict]:
    """Build observation list from {subject_id: [values by occasion]}."""
    rows = []
    for sid, vals in subject_values.items():
        for occ, v in enumerate(vals, start=1):
            rows.append({"subject_id": sid, "occasion": occ, "value": v})
    return rows


# ---------------------------------------------------------------------------
# Basic return type and structure tests
# ---------------------------------------------------------------------------


def test_returns_variability_decomposition():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert isinstance(result, VariabilityDecomposition)


def test_bsv_variance_nonnegative():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.bsv_variance >= 0.0


def test_wsv_variance_nonnegative():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.wsv_variance >= 0.0


def test_total_variance_positive():
    data = _make_data({"A": [1.0, 1.5], "B": [3.0, 3.5]})
    result = decompose_variability(data)
    assert result.total_variance > 0.0


def test_fractions_sum_to_one():
    data = _make_data({"A": [1.0, 1.2], "B": [2.0, 2.4]})
    result = decompose_variability(data)
    ruv_fraction = result.ruv_variance / result.total_variance
    total = result.bsv_fraction + result.wsv_fraction + ruv_fraction
    assert abs(total - 1.0) < 1e-9


def test_bsv_fraction_plus_wsv_fraction_le_1():
    data = _make_data({"A": [1.0, 1.2], "B": [2.0, 2.4]})
    result = decompose_variability(data)
    assert result.bsv_fraction + result.wsv_fraction <= 1.0 + 1e-9


def test_bsv_cv_pct_nonnegative():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.bsv_cv_pct >= 0.0


def test_wsv_cv_pct_nonnegative():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.wsv_cv_pct >= 0.0


def test_total_cv_pct_nonnegative():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.total_cv_pct >= 0.0


def test_icc_in_range_0_1():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert 0.0 <= result.icc <= 1.0


def test_interpretation_valid_value():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.interpretation in ("BSV-dominated", "WSV-dominated", "RUV-dominated")


def test_notes_nonempty():
    data = _make_data({"A": [1.0, 1.1], "B": [2.0, 2.1]})
    result = decompose_variability(data)
    assert result.notes != ""


# ---------------------------------------------------------------------------
# Scientific correctness tests
# ---------------------------------------------------------------------------


def test_high_inter_subject_variability_bsv_dominated():
    """Subjects with very different means → BSV should dominate."""
    data = _make_data(
        {
            "subj1": [0.5, 0.6, 0.55],
            "subj2": [5.0, 5.2, 4.9],
            "subj3": [50.0, 51.0, 49.0],
        }
    )
    result = decompose_variability(data)
    assert result.bsv_fraction > 0.5


def test_low_inter_subject_variability_bsv_near_zero():
    """All subjects same mean → BSV should be near 0."""
    # Same within-group measurements, same between-group means
    data = _make_data(
        {
            "A": [2.0, 2.0, 2.0],
            "B": [2.0, 2.0, 2.0],
            "C": [2.0, 2.0, 2.0],
        }
    )
    result = decompose_variability(data)
    assert result.bsv_fraction < 0.05


def test_single_occasion_wsv_is_zero():
    """Single observation per subject → WSV = 0."""
    data = _make_data({"A": [1.0], "B": [3.0], "C": [2.0]})
    result = decompose_variability(data)
    assert result.wsv_variance == 0.0


def test_icc_close_to_1_when_bsv_dominates():
    """When BSV >> WSV, ICC should be close to 1."""
    data = _make_data(
        {
            "low": [0.1, 0.11, 0.12],
            "high": [100.0, 101.0, 99.0],
        }
    )
    result = decompose_variability(data)
    assert result.icc > 0.8


def test_known_simple_case_2_subjects_same_within():
    """2 subjects with same within-subject variation but different means.

    Subject A: [1.0, 1.0] → mean = 1.0
    Subject B: [10.0, 10.0] → mean = 10.0

    Within-subject SS = 0 → WSV = 0 (or very small).
    BSV should dominate.
    """
    data = _make_data({"A": [1.0, 1.0], "B": [10.0, 10.0]})
    result = decompose_variability(data)
    assert result.bsv_variance > result.wsv_variance


def test_bsv_fraction_in_0_1():
    data = _make_data({"A": [1.0, 1.5], "B": [2.0, 2.5]})
    result = decompose_variability(data)
    assert 0.0 <= result.bsv_fraction <= 1.0


def test_wsv_fraction_in_0_1():
    data = _make_data({"A": [1.0, 1.5], "B": [2.0, 2.5]})
    result = decompose_variability(data)
    assert 0.0 <= result.wsv_fraction <= 1.0


# ---------------------------------------------------------------------------
# compare_variability_sources
# ---------------------------------------------------------------------------


def test_compare_variability_sources_returns_list():
    datasets = {
        "CL": _make_data({"A": [1.0, 1.1], "B": [5.0, 5.2]}),
        "Vd": _make_data({"A": [10.0, 11.0], "B": [12.0, 13.0]}),
    }
    result = compare_variability_sources(datasets)
    assert isinstance(result, list)
    assert len(result) == 2


def test_compare_variability_sources_sorted_desc():
    """Result should be sorted by total_cv_pct descending."""
    datasets = {
        "low_var": _make_data({"A": [1.0, 1.01], "B": [1.02, 1.03]}),
        "high_var": _make_data({"A": [1.0, 1.5], "B": [5.0, 8.0]}),
    }
    result = compare_variability_sources(datasets)
    assert result[0].total_cv_pct >= result[1].total_cv_pct


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_empty_data_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        decompose_variability([])


def test_single_subject_raises_value_error():
    data = _make_data({"only_one": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="subjects"):
        decompose_variability(data)


def test_non_positive_value_with_log_transform_raises():
    data = [
        {"subject_id": "A", "occasion": 1, "value": -1.0},
        {"subject_id": "B", "occasion": 1, "value": 2.0},
    ]
    with pytest.raises(ValueError, match="positive"):
        decompose_variability(data, log_transform=True)


def test_zero_value_with_log_transform_raises():
    data = [
        {"subject_id": "A", "occasion": 1, "value": 0.0},
        {"subject_id": "B", "occasion": 1, "value": 2.0},
    ]
    with pytest.raises(ValueError, match="positive"):
        decompose_variability(data, log_transform=True)
