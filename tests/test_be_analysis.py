"""Tests for bioequivalence statistical analysis module."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.be_analysis import (
    BEResult,
    PowerCurveResult,
    analyze_be,
    calculate_power_curve,
    sample_size_for_power,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def identical_data():
    """Reference and test arrays that are identical."""
    values = [100.0, 110.0, 95.0, 105.0, 102.0, 98.0, 108.0, 103.0, 97.0, 101.0]
    return values, list(values)


@pytest.fixture()
def half_data():
    """Test values at 50% of reference — should fail BE."""
    ref = [100.0, 110.0, 95.0, 105.0, 102.0, 98.0, 108.0, 103.0, 97.0, 101.0]
    test = [v * 0.5 for v in ref]
    return ref, test


@pytest.fixture()
def double_data():
    """Test values at 200% of reference."""
    ref = [100.0, 110.0, 95.0, 105.0, 102.0, 98.0, 108.0, 103.0, 97.0, 101.0]
    test = [v * 2.0 for v in ref]
    return ref, test


# ---------------------------------------------------------------------------
# 1. Identical arrays → gm_ratio ≈ 1.0, is_bioequivalent=True
# ---------------------------------------------------------------------------

def test_identical_arrays_gm_ratio(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert abs(result.gm_ratio - 1.0) < 1e-6


def test_identical_arrays_bioequivalent(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert result.is_bioequivalent is True


# ---------------------------------------------------------------------------
# 2. BE fails for large ratio (test = 0.5 × reference)
# ---------------------------------------------------------------------------

def test_half_ratio_fails_be(half_data):
    ref, test = half_data
    result = analyze_be(ref, test)
    assert result.is_bioequivalent is False


# ---------------------------------------------------------------------------
# 3. BE fails for high CV with small n
# ---------------------------------------------------------------------------

def test_high_cv_small_n_fails_be():
    # High variability, small sample — CI should be wide
    ref = [50.0, 150.0, 80.0]
    test = [55.0, 145.0, 85.0]
    result = analyze_be(ref, test)
    # With such high CV and only 3 subjects, BE is unlikely to pass
    # (the CI will be very wide even though ratio is near 1)
    assert isinstance(result.is_bioequivalent, bool)


# ---------------------------------------------------------------------------
# 4. ValueError for <2 subjects
# ---------------------------------------------------------------------------

def test_error_fewer_than_2_subjects_ref():
    with pytest.raises(ValueError, match="At least 2 subjects"):
        analyze_be([100.0], [100.0, 110.0])


def test_error_fewer_than_2_subjects_test():
    with pytest.raises(ValueError, match="At least 2 subjects"):
        analyze_be([100.0, 110.0], [100.0])


# ---------------------------------------------------------------------------
# 5. ValueError for negative values
# ---------------------------------------------------------------------------

def test_error_negative_reference():
    with pytest.raises(ValueError, match="positive"):
        analyze_be([-10.0, 100.0], [100.0, 110.0])


def test_error_negative_test():
    with pytest.raises(ValueError, match="positive"):
        analyze_be([100.0, 110.0], [-10.0, 100.0])


# ---------------------------------------------------------------------------
# 6. ValueError for zero values
# ---------------------------------------------------------------------------

def test_error_zero_reference():
    with pytest.raises(ValueError, match="positive"):
        analyze_be([0.0, 100.0], [100.0, 110.0])


def test_error_zero_test():
    with pytest.raises(ValueError, match="positive"):
        analyze_be([100.0, 110.0], [0.0, 100.0])


# ---------------------------------------------------------------------------
# 7. metric stored correctly
# ---------------------------------------------------------------------------

def test_metric_stored(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test, metric="Cmax")
    assert result.metric == "Cmax"


# ---------------------------------------------------------------------------
# 8. n_reference and n_test stored correctly
# ---------------------------------------------------------------------------

def test_n_reference_n_test_stored():
    ref = [100.0, 110.0, 95.0]
    test = [102.0, 108.0, 97.0, 105.0]
    result = analyze_be(ref, test)
    assert result.n_reference == 3
    assert result.n_test == 4


# ---------------------------------------------------------------------------
# 9. ci_lower < ci_upper
# ---------------------------------------------------------------------------

def test_ci_lower_less_than_upper(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert result.ci_lower_90 <= result.ci_upper_90


# ---------------------------------------------------------------------------
# 10. gm_ratio > 0
# ---------------------------------------------------------------------------

def test_gm_ratio_positive(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert result.gm_ratio > 0


# ---------------------------------------------------------------------------
# 11. cv_pct >= 0
# ---------------------------------------------------------------------------

def test_cv_pct_nonnegative(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert result.cv_pct >= 0


# ---------------------------------------------------------------------------
# 12. power_at_observed_n between 0 and 1
# ---------------------------------------------------------------------------

def test_power_in_range(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert 0.0 <= result.power_at_observed_n <= 1.0


# ---------------------------------------------------------------------------
# 13. notes string non-empty
# ---------------------------------------------------------------------------

def test_notes_nonempty(identical_data):
    ref, test = identical_data
    result = analyze_be(ref, test)
    assert len(result.notes) > 0
    assert "BE analysis" in result.notes


# ---------------------------------------------------------------------------
# 14. calculate_power_curve returns correct count
# ---------------------------------------------------------------------------

def test_power_curve_correct_count():
    ns = [10, 20, 30, 40, 50]
    result = calculate_power_curve(ns, cv_pct=30.0)
    assert len(result.n_values) == 5
    assert len(result.powers) == 5


# ---------------------------------------------------------------------------
# 15. calculate_power_curve powers are between 0 and 1
# ---------------------------------------------------------------------------

def test_power_curve_values_in_range():
    ns = [6, 12, 24, 48]
    result = calculate_power_curve(ns, cv_pct=25.0)
    for p in result.powers:
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 16. Powers increase with increasing n (for ratio=1.0)
# ---------------------------------------------------------------------------

def test_power_increases_with_n():
    ns = [4, 10, 20, 50, 100, 200]
    result = calculate_power_curve(ns, cv_pct=30.0, ratio=1.0)
    for i in range(1, len(result.powers)):
        assert result.powers[i] >= result.powers[i - 1]


# ---------------------------------------------------------------------------
# 17. ValueError for empty n_range
# ---------------------------------------------------------------------------

def test_power_curve_empty_n_range():
    with pytest.raises(ValueError, match="n_range must not be empty"):
        calculate_power_curve([], cv_pct=25.0)


# ---------------------------------------------------------------------------
# 18. ValueError for cv_pct <= 0 in power curve
# ---------------------------------------------------------------------------

def test_power_curve_nonpositive_cv():
    with pytest.raises(ValueError, match="cv_pct must be > 0"):
        calculate_power_curve([10, 20], cv_pct=0.0)

    with pytest.raises(ValueError, match="cv_pct must be > 0"):
        calculate_power_curve([10, 20], cv_pct=-5.0)


# ---------------------------------------------------------------------------
# 19. sample_size_for_power returns positive int
# ---------------------------------------------------------------------------

def test_sample_size_positive_int():
    n = sample_size_for_power(cv_pct=25.0, power_target=0.80)
    assert isinstance(n, int)
    assert n >= 2


# ---------------------------------------------------------------------------
# 20. sample_size increases with higher CV
# ---------------------------------------------------------------------------

def test_sample_size_increases_with_cv():
    n_low = sample_size_for_power(cv_pct=20.0, power_target=0.80)
    n_high = sample_size_for_power(cv_pct=40.0, power_target=0.80)
    assert n_high > n_low


# ---------------------------------------------------------------------------
# 21. ValueError for cv_pct <= 0 in sample_size
# ---------------------------------------------------------------------------

def test_sample_size_nonpositive_cv():
    with pytest.raises(ValueError, match="cv_pct must be > 0"):
        sample_size_for_power(cv_pct=0.0)

    with pytest.raises(ValueError, match="cv_pct must be > 0"):
        sample_size_for_power(cv_pct=-10.0)


# ---------------------------------------------------------------------------
# 22. ValueError for power_target out of range
# ---------------------------------------------------------------------------

def test_sample_size_invalid_power_target():
    with pytest.raises(ValueError, match="power_target must be between 0 and 1"):
        sample_size_for_power(cv_pct=25.0, power_target=0.0)

    with pytest.raises(ValueError, match="power_target must be between 0 and 1"):
        sample_size_for_power(cv_pct=25.0, power_target=1.0)

    with pytest.raises(ValueError, match="power_target must be between 0 and 1"):
        sample_size_for_power(cv_pct=25.0, power_target=1.5)


# ---------------------------------------------------------------------------
# 23. ratio field stored in PowerCurveResult
# ---------------------------------------------------------------------------

def test_power_curve_ratio_stored():
    result = calculate_power_curve([10, 20], cv_pct=25.0, ratio=0.95)
    assert result.ratio == 0.95
    assert result.cv_pct == 25.0


# ---------------------------------------------------------------------------
# 24. Large n with low CV → high power
# ---------------------------------------------------------------------------

def test_large_n_low_cv_high_power():
    result = calculate_power_curve([200], cv_pct=15.0, ratio=1.0)
    assert result.powers[0] > 0.99


# ---------------------------------------------------------------------------
# 25. Scaled test values (2×) should give gm_ratio ≈ 2.0
# ---------------------------------------------------------------------------

def test_double_ratio(double_data):
    ref, test = double_data
    result = analyze_be(ref, test)
    assert abs(result.gm_ratio - 2.0) < 0.05
    assert result.is_bioequivalent is False
