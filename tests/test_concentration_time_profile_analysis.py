"""Tests for concentration_time_profile_analysis module (Phase 713)."""

import math

import pytest

from omega_pbpk.analysis.concentration_time_profile_analysis import (
    ProfileComparisonResult,
    compare_profiles,
    compute_fluctuation_index,
    compute_partial_auc,
    find_local_maxima,
    interpolate_concentration,
)

# ---------------------------------------------------------------------------
# find_local_maxima
# ---------------------------------------------------------------------------


def test_find_local_maxima_flat_returns_empty():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    concs = [1.0, 1.0, 1.0, 1.0, 1.0]
    peaks = find_local_maxima(times, concs)
    assert peaks == []


def test_find_local_maxima_single_peak():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    concs = [0.0, 1.0, 5.0, 1.0, 0.0]
    peaks = find_local_maxima(times, concs)
    assert len(peaks) == 1
    assert peaks[0]["time_h"] == 2.0
    assert peaks[0]["conc_mg_L"] == 5.0
    assert peaks[0]["index"] == 2


def test_find_local_maxima_double_hump_ehc():
    """Enterohepatic circulation double-hump profile."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    concs = [0.0, 2.0, 5.0, 3.0, 2.0, 4.0, 2.0, 1.0, 0.5]
    peaks = find_local_maxima(times, concs)
    assert len(peaks) == 2
    assert peaks[0]["time_h"] == 2.0
    assert peaks[1]["time_h"] == 5.0


def test_find_local_maxima_prominence_filter():
    """Tiny secondary peak below prominence threshold should be filtered out."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    concs = [0.0, 10.0, 0.5, 0.3, 0.5, 0.0]
    # cmax=10, min_prominence=0.05 → threshold=0.5; second peak at 0.5 is NOT > threshold
    peaks = find_local_maxima(times, concs, min_prominence=0.05)
    # Only first peak (10.0) qualifies; 0.5 is not > 0.5 (strict >)
    assert len(peaks) == 1
    assert peaks[0]["conc_mg_L"] == 10.0


def test_find_local_maxima_returns_list_of_dicts():
    times = [0.0, 1.0, 2.0, 3.0]
    concs = [0.0, 3.0, 2.0, 0.0]
    peaks = find_local_maxima(times, concs)
    assert isinstance(peaks, list)
    for p in peaks:
        assert "time_h" in p
        assert "conc_mg_L" in p
        assert "index" in p


# ---------------------------------------------------------------------------
# compute_partial_auc
# ---------------------------------------------------------------------------


def test_compute_partial_auc_full_range_equals_trapezoidal():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    concs = [0.0, 2.0, 4.0, 2.0, 0.0]
    total_auc = compute_partial_auc(times, concs, 0.0, 4.0)
    # Expected trapezoidal: (0+2)/2*1 + (2+4)/2*1 + (4+2)/2*1 + (2+0)/2*1 = 1+3+3+1 = 8
    assert math.isclose(total_auc, 8.0, rel_tol=1e-9)


def test_compute_partial_auc_partial_less_than_total():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    concs = [0.0, 2.0, 4.0, 2.0, 0.0]
    total = compute_partial_auc(times, concs, 0.0, 4.0)
    partial = compute_partial_auc(times, concs, 0.0, 2.0)
    assert partial < total


def test_compute_partial_auc_interpolation_at_boundary():
    """AUC from 0.5 to 2.5 should use linear interpolation at boundaries."""
    times = [0.0, 1.0, 2.0, 3.0]
    concs = [0.0, 2.0, 4.0, 2.0]
    auc = compute_partial_auc(times, concs, 0.5, 2.5)
    # c(0.5)=1, c(1.0)=2, c(2.0)=4, c(2.5)=3
    # (1+2)/2*0.5 + (2+4)/2*1 + (4+3)/2*0.5 = 0.75 + 3.0 + 1.75 = 5.5
    assert math.isclose(auc, 5.5, rel_tol=1e-9)


def test_compute_partial_auc_zero_when_t_start_equals_t_end():
    times = [0.0, 1.0, 2.0]
    concs = [1.0, 2.0, 3.0]
    auc = compute_partial_auc(times, concs, 1.0, 1.0)
    assert auc == 0.0


# ---------------------------------------------------------------------------
# compare_profiles
# ---------------------------------------------------------------------------


def test_compare_profiles_identical():
    times = [0.0, 1.0, 2.0, 4.0]
    concs = [0.0, 5.0, 3.0, 1.0]
    result = compare_profiles(times, concs, times, concs)
    assert math.isclose(result.auc_ratio, 1.0, rel_tol=1e-9)
    assert math.isclose(result.cmax_ratio, 1.0, rel_tol=1e-9)
    assert math.isclose(result.rmsd, 0.0, abs_tol=1e-10)


def test_compare_profiles_double_concentrations():
    times = [0.0, 1.0, 2.0, 4.0]
    concs_a = [0.0, 10.0, 6.0, 2.0]
    concs_b = [0.0, 5.0, 3.0, 1.0]
    result = compare_profiles(times, concs_a, times, concs_b)
    assert math.isclose(result.auc_ratio, 2.0, rel_tol=1e-9)
    assert math.isclose(result.cmax_ratio, 2.0, rel_tol=1e-9)


def test_compare_profiles_returns_dataclass():
    times = [0.0, 1.0, 2.0]
    concs = [0.0, 3.0, 1.0]
    result = compare_profiles(times, concs, times, concs)
    assert isinstance(result, ProfileComparisonResult)


def test_compare_profiles_similarity_similar():
    times = [0.0, 1.0, 2.0, 4.0]
    concs = [0.0, 5.0, 3.0, 1.0]
    result = compare_profiles(times, concs, times, concs)
    assert result.profile_similarity == "similar"


def test_compare_profiles_similarity_dissimilar():
    times = [0.0, 1.0, 2.0, 4.0]
    concs_a = [0.0, 50.0, 30.0, 10.0]
    concs_b = [0.0, 1.0, 0.5, 0.1]
    result = compare_profiles(times, concs_a, times, concs_b)
    # nRMSD should be >> 0.5
    assert result.profile_similarity == "dissimilar"


def test_compare_profiles_notes_nonempty():
    times = [0.0, 1.0, 2.0]
    concs = [0.0, 3.0, 1.0]
    result = compare_profiles(times, concs, times, concs)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# compute_fluctuation_index
# ---------------------------------------------------------------------------


def test_compute_fluctuation_index_constant():
    times = [0.0, 6.0, 12.0, 18.0, 24.0]
    concs = [5.0, 5.0, 5.0, 5.0, 5.0]
    fi = compute_fluctuation_index(times, concs)
    assert math.isclose(fi["ptf"], 0.0, abs_tol=1e-10)
    assert math.isclose(fi["swing"], 0.0, abs_tol=1e-10)
    assert math.isclose(fi["cavg"], 5.0, rel_tol=1e-9)


def test_compute_fluctuation_index_varying():
    times = [0.0, 6.0, 12.0, 18.0, 24.0]
    concs = [1.0, 5.0, 3.0, 2.0, 1.0]
    fi = compute_fluctuation_index(times, concs)
    assert fi["ptf"] > 0.0
    assert fi["swing"] > 0.0
    assert fi["cmax"] == 5.0
    assert fi["cmin"] == 1.0


def test_compute_fluctuation_index_keys():
    times = [0.0, 1.0, 2.0]
    concs = [2.0, 4.0, 2.0]
    fi = compute_fluctuation_index(times, concs)
    for key in ["cmax", "cmin", "cavg", "ptf", "swing"]:
        assert key in fi


# ---------------------------------------------------------------------------
# interpolate_concentration
# ---------------------------------------------------------------------------


def test_interpolate_at_exact_time_points():
    times = [0.0, 1.0, 2.0, 3.0]
    concs = [0.0, 2.0, 4.0, 3.0]
    for i, t in enumerate(times):
        assert math.isclose(interpolate_concentration(times, concs, t), concs[i], rel_tol=1e-9)


def test_interpolate_between_points():
    times = [0.0, 2.0, 4.0]
    concs = [0.0, 4.0, 2.0]
    # At t=1.0 (between 0 and 2): c = 0 + (1/2)*(4-0) = 2.0
    val = interpolate_concentration(times, concs, 1.0)
    assert math.isclose(val, 2.0, rel_tol=1e-9)


def test_interpolate_outside_range_returns_zero():
    times = [1.0, 2.0, 3.0]
    concs = [1.0, 2.0, 3.0]
    assert interpolate_concentration(times, concs, 0.0) == 0.0
    assert interpolate_concentration(times, concs, 5.0) == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unequal_length_raises_value_error():
    with pytest.raises(ValueError, match="same length"):
        find_local_maxima([0.0, 1.0, 2.0], [1.0, 2.0])


def test_too_short_raises_value_error():
    with pytest.raises(ValueError, match="at least 2"):
        compute_partial_auc([0.0], [1.0], 0.0, 1.0)


def test_non_increasing_times_raises_value_error():
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_fluctuation_index([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])
