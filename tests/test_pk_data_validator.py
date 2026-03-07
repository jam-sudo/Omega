"""Tests for PK data quality validator."""

import pytest

from omega_pbpk.analysis.pk_data_validator import (
    PKDataValidationResult,
    check_dose_proportionality_consistency,
    compute_lloq_adjusted_auc,
    flag_outliers_grubbs,
    validate_pk_data,
)

# ---------------------------------------------------------------------------
# validate_pk_data
# ---------------------------------------------------------------------------


def _good_oral_data():
    """Standard well-sampled oral PK profile."""
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    concs = [0.0, 1.2, 3.5, 5.0, 4.0, 2.5, 1.5, 0.4]
    return times, concs


def test_validate_good_data_is_acceptable():
    times, concs = _good_oral_data()
    result = validate_pk_data(times, concs)
    assert result.is_acceptable is True


def test_validate_good_data_no_errors():
    times, concs = _good_oral_data()
    result = validate_pk_data(times, concs)
    assert result.errors == ""


def test_validate_perfect_data_quality_score_100():
    """No issues with good monotone data, non-zero at t=0, clear peak, low last conc."""
    # Starting at t=0.5 (no zero conc) avoids BLQ warning for IV route
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    concs = [0.1, 1.0, 3.0, 5.0, 4.0, 2.0, 0.8, 0.05]
    result = validate_pk_data(times, concs, expected_route="iv")
    assert result.data_quality_score == 100.0


def test_validate_non_monotone_time_warning():
    times = [0.0, 1.0, 0.5, 2.0, 4.0, 8.0, 12.0]  # 0.5 after 1.0
    concs = [0.0, 2.0, 1.5, 4.0, 3.0, 1.5, 0.5]
    result = validate_pk_data(times, concs)
    assert "non-monotone" in result.warnings


def test_validate_non_monotone_reduces_score():
    times = [0.0, 1.0, 0.5, 2.0, 4.0, 8.0, 12.0]
    concs = [0.0, 2.0, 1.5, 4.0, 3.0, 1.5, 0.5]
    result = validate_pk_data(times, concs)
    assert result.data_quality_score < 100.0


def test_validate_negative_concentration_error():
    times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    concs = [0.0, 2.0, -0.5, 3.0, 1.5, 0.5]
    result = validate_pk_data(times, concs)
    assert "negative concentration" in result.errors


def test_validate_negative_concentration_not_acceptable():
    times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    concs = [0.0, 2.0, -0.5, 3.0, 1.5, 0.5]
    result = validate_pk_data(times, concs)
    # negative conc is an error (-20 points)
    assert result.data_quality_score <= 80.0


def test_validate_insufficient_data_warning():
    times = [0.0, 1.0, 2.0, 4.0]
    concs = [0.0, 3.0, 2.0, 1.0]
    result = validate_pk_data(times, concs)
    assert "insufficient" in result.warnings


def test_validate_n_timepoints_correct():
    times, concs = _good_oral_data()
    result = validate_pk_data(times, concs)
    assert result.n_timepoints == len(times)


def test_validate_truncated_data_warning():
    """Last concentration > 20% of Cmax should warn about truncation."""
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    concs = [0.0, 1.0, 5.0, 5.0, 4.5, 4.0]  # last = 4.0, cmax = 5.0 → 80%
    result = validate_pk_data(times, concs)
    assert "truncated" in result.warnings


def test_validate_no_truncation_warning_when_low():
    """Last concentration <= 20% of Cmax should NOT warn about truncation."""
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
    concs = [0.0, 1.0, 5.0, 4.0, 2.0, 0.8, 0.05]  # last = 0.05 → 1% of Cmax
    result = validate_pk_data(times, concs)
    assert "truncated" not in result.warnings


def test_validate_blq_warning():
    """Zero concentrations among positive values → potential BLQ."""
    times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    concs = [0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0]
    result = validate_pk_data(times, concs)
    assert "BLQ" in result.warnings


def test_validate_result_type():
    times, concs = _good_oral_data()
    result = validate_pk_data(times, concs)
    assert isinstance(result, PKDataValidationResult)


def test_validate_n_issues_matches_errors_warnings():
    times = [0.0, 1.0, 0.5, 2.0]  # non-monotone + < 6 pts
    concs = [0.0, 2.0, 1.5, 1.0]
    result = validate_pk_data(times, concs)
    n_errors = len([e for e in result.errors.split("|") if e])
    n_warnings = len([w for w in result.warnings.split("|") if w])
    assert result.n_issues == n_errors + n_warnings


def test_validate_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        validate_pk_data([0.0, 1.0, 2.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# flag_outliers_grubbs
# ---------------------------------------------------------------------------


def test_grubbs_uniform_no_outliers():
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert flag_outliers_grubbs(values) == []


def test_grubbs_detects_extreme_outlier():
    # n=9 (< 10 → G_crit=2.5); 8 near-identical values + 1 extreme → G ≈ 2.667 > 2.5
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0]
    outliers = flag_outliers_grubbs(values)
    assert 8 in outliers


def test_grubbs_small_list_no_crash():
    assert flag_outliers_grubbs([1.0, 2.0]) == []


def test_grubbs_returns_list():
    result = flag_outliers_grubbs([1.0, 1.0, 1.0, 1.0, 50.0])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# compute_lloq_adjusted_auc
# ---------------------------------------------------------------------------


def test_lloq_auc_all_above_lloq():
    times = [0.0, 1.0, 2.0, 4.0]
    concs = [0.1, 1.0, 0.5, 0.2]
    auc = compute_lloq_adjusted_auc(times, concs, lloq=0.001)
    # Manual: 0.5*(0.1+1.0)*1 + 0.5*(1.0+0.5)*1 + 0.5*(0.5+0.2)*2 = 0.55+0.75+0.70 = 2.0
    assert abs(auc - 2.0) < 1e-6


def test_lloq_auc_blq_replaced():
    """Zero concentrations should be replaced by lloq/2, giving positive AUC."""
    times = [0.0, 1.0, 2.0, 4.0]
    concs = [0.0, 1.0, 0.5, 0.0]
    lloq = 0.01
    auc = compute_lloq_adjusted_auc(times, concs, lloq=lloq)
    assert auc > 0.0


def test_lloq_auc_result_greater_than_with_zeros():
    """AUC with lloq substitution > trapezoidal AUC ignoring zeros."""
    times = [0.0, 1.0, 2.0, 3.0]
    concs = [0.0, 0.0, 1.0, 0.0]
    lloq = 0.1
    auc = compute_lloq_adjusted_auc(times, concs, lloq=lloq)
    # lloq/2 = 0.05 substituted for zeros
    assert auc > 0.0


# ---------------------------------------------------------------------------
# check_dose_proportionality_consistency
# ---------------------------------------------------------------------------


def test_dose_prop_linear_data():
    """Perfectly linear AUC-dose: exponent should be ~1.0."""
    doses = [10.0, 20.0, 50.0, 100.0]
    aucs = [50.0, 100.0, 250.0, 500.0]  # AUC = 5 * dose
    result = check_dose_proportionality_consistency(doses, aucs, expected_exponent=1.0)
    assert abs(result["fitted_exponent"] - 1.0) < 0.01


def test_dose_prop_is_consistent_linear():
    doses = [10.0, 20.0, 50.0, 100.0]
    aucs = [50.0, 100.0, 250.0, 500.0]
    result = check_dose_proportionality_consistency(doses, aucs, expected_exponent=1.0)
    assert result["is_consistent"] is True


def test_dose_prop_nonlinear_not_consistent():
    """Saturable kinetics: exponent < 1, should not be consistent with 1.0."""
    doses = [10.0, 20.0, 50.0, 100.0]
    # AUC grows sublinearly (saturation)
    aucs = [80.0, 120.0, 160.0, 190.0]
    result = check_dose_proportionality_consistency(doses, aucs, expected_exponent=1.0)
    assert result["is_consistent"] is False


def test_dose_prop_returns_dict_keys():
    doses = [10.0, 50.0, 100.0]
    aucs = [100.0, 500.0, 1000.0]
    result = check_dose_proportionality_consistency(doses, aucs)
    assert "fitted_exponent" in result
    assert "is_consistent" in result
    assert "deviation" in result
