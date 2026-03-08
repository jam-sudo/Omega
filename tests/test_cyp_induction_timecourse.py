"""Tests for Phase 401: cyp_induction_timecourse.py"""

import pytest

from omega_pbpk.clinical.cyp_induction_timecourse import (
    CYPInductionResult,
    simulate_cyp_induction,
    simulate_induction_washout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    """Default simulate_cyp_induction call with strong inducer."""
    defaults = dict(
        drug_name="Rifampicin",
        cyp_enzyme="CYP3A4",
        inducer_conc_uM=10.0,
        ec50_uM=1.0,
        max_fold_induction=8.0,
        k_deg_per_h=0.02,
        t_end_h=336.0,
        dt_h=1.0,
        washout_start_h=None,
    )
    defaults.update(kwargs)
    return simulate_cyp_induction(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_cyp_induction_result():
    result = _default()
    assert isinstance(result, CYPInductionResult)


def test_result_has_correct_fields():
    result = _default()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "cyp_enzyme")
    assert hasattr(result, "times_h")
    assert hasattr(result, "enzyme_activity_relative")
    assert hasattr(result, "victim_cl_relative")
    assert hasattr(result, "victim_auc_ratio")
    assert hasattr(result, "peak_induction_fold")
    assert hasattr(result, "time_to_peak_h")
    assert hasattr(result, "time_to_50pct_induction_h")
    assert hasattr(result, "washout_50pct_h")
    assert hasattr(result, "fda_induction_category")
    assert hasattr(result, "auc_ratio_at_peak")
    assert hasattr(result, "notes")


def test_list_fields_are_lists():
    result = _default()
    assert isinstance(result.times_h, list)
    assert isinstance(result.enzyme_activity_relative, list)
    assert isinstance(result.victim_cl_relative, list)
    assert isinstance(result.victim_auc_ratio, list)


def test_list_lengths_match():
    result = _default()
    n = len(result.times_h)
    assert len(result.enzyme_activity_relative) == n
    assert len(result.victim_cl_relative) == n
    assert len(result.victim_auc_ratio) == n


# ---------------------------------------------------------------------------
# Induction dynamics: enzyme activity increases above baseline
# ---------------------------------------------------------------------------


def test_induction_increases_enzyme_activity():
    result = _default()
    # Enzyme should rise above 1.0 (baseline) over 336 h
    assert max(result.enzyme_activity_relative) > 1.0


def test_baseline_starts_at_one():
    result = _default()
    # First time point is t=0, enzyme starts at 1.0
    assert abs(result.enzyme_activity_relative[0] - 1.0) < 1e-9


def test_peak_fold_below_max_fold():
    """With finite time, enzyme takes time to equilibrate — peak < max."""
    # At SS with C >> EC50, max_fold contributes but E still needs time.
    # With max_fold=8 and t=336h, E should reach near SS but not exceed max_fold.
    result = _default(max_fold_induction=8.0)
    # The SS induction factor = 1 + 8 * 10 / (10 + 1) = 1 + 7.27 = 8.27
    # Peak fold should be < 8.27 (approaches asymptotically)
    # But definitely > 1.25 to be an inducer
    assert result.peak_induction_fold > 1.25
    assert result.peak_induction_fold <= 8.27 + 0.01  # can't exceed SS value


def test_no_induction_when_conc_zero():
    """With 0 inducer concentration, enzyme stays at 1.0."""
    result = _default(inducer_conc_uM=0.0, max_fold_induction=8.0)
    assert abs(result.peak_induction_fold - 1.0) < 1e-6
    assert result.fda_induction_category == "none"


# ---------------------------------------------------------------------------
# Victim drug CL and AUC ratio
# ---------------------------------------------------------------------------


def test_victim_auc_ratio_is_inverse_of_cl():
    result = _default()
    for cl, auc in zip(result.victim_cl_relative, result.victim_auc_ratio):
        expected = 1.0 / cl
        assert abs(auc - expected) < 1e-6


def test_auc_ratio_at_peak_is_inverse_of_peak_fold():
    result = _default()
    expected = 1.0 / result.peak_induction_fold
    assert abs(result.auc_ratio_at_peak - expected) < 1e-6


def test_auc_ratio_at_peak_less_than_one():
    """Induction increases CL, so victim AUC decreases below 1."""
    result = _default()
    assert result.auc_ratio_at_peak < 1.0


def test_victim_cl_matches_enzyme_activity():
    result = _default()
    for cl, ea in zip(result.victim_cl_relative, result.enzyme_activity_relative):
        assert abs(cl - ea) < 1e-9


# ---------------------------------------------------------------------------
# FDA induction category
# ---------------------------------------------------------------------------


def test_strong_inducer_category():
    # max_fold=10, conc >> EC50 → peak_fold > 5 → strong
    result = _default(inducer_conc_uM=100.0, ec50_uM=1.0, max_fold_induction=10.0, t_end_h=500.0)
    assert result.fda_induction_category == "strong"


def test_moderate_inducer_category():
    # max_fold=4 → moderate (2-5x)
    result = _default(inducer_conc_uM=100.0, ec50_uM=1.0, max_fold_induction=4.0, t_end_h=500.0)
    assert result.fda_induction_category == "moderate"


def test_weak_inducer_category():
    # max_fold=1.5, near EC50 concentration → weak (1.25-2x)
    result = _default(inducer_conc_uM=1.0, ec50_uM=5.0, max_fold_induction=2.0, t_end_h=500.0)
    # SS factor = 1 + 2 * 1/(1+5) = 1 + 0.33 = 1.33 → weak
    assert result.fda_induction_category in ("weak", "none")


def test_none_category_when_no_induction():
    result = _default(inducer_conc_uM=0.0)
    assert result.fda_induction_category == "none"


# ---------------------------------------------------------------------------
# Time to peak and 50% induction
# ---------------------------------------------------------------------------


def test_time_to_peak_positive():
    result = _default()
    assert result.time_to_peak_h > 0.0


def test_time_to_50pct_induction_less_than_time_to_peak():
    """Half induction level should be reached before peak."""
    result = _default()
    assert result.time_to_50pct_induction_h <= result.time_to_peak_h


def test_stronger_inducer_higher_peak():
    result_weak = _default(inducer_conc_uM=0.1, ec50_uM=10.0)
    result_strong = _default(inducer_conc_uM=100.0, ec50_uM=1.0)
    assert result_strong.peak_induction_fold > result_weak.peak_induction_fold


# ---------------------------------------------------------------------------
# k_deg effect on kinetics
# ---------------------------------------------------------------------------


def test_higher_kdeg_faster_induction():
    """Higher k_deg → faster enzyme turnover → faster approach to SS."""
    result_slow = _default(k_deg_per_h=0.01, t_end_h=500.0)
    result_fast = _default(k_deg_per_h=0.1, t_end_h=500.0)
    # Fast kdeg should reach 50% induction in less time
    assert result_fast.time_to_50pct_induction_h < result_slow.time_to_50pct_induction_h


# ---------------------------------------------------------------------------
# Washout kinetics
# ---------------------------------------------------------------------------


def test_washout_returns_toward_baseline():
    """After washout, enzyme should decline back toward 1.0."""
    result = _default(
        washout_start_h=168.0,
        t_end_h=336.0,
    )
    # Enzyme at washout start should be elevated
    # Find washout start index (t=168)
    washout_idx = None
    for i, t in enumerate(result.times_h):
        if t >= 167.9:
            washout_idx = i
            break
    assert washout_idx is not None
    enzyme_at_washout = result.enzyme_activity_relative[washout_idx]
    enzyme_at_end = result.enzyme_activity_relative[-1]
    # Enzyme should decrease after washout
    assert enzyme_at_end < enzyme_at_washout


def test_washout_50pct_positive_when_washout_given():
    result = _default(washout_start_h=168.0, t_end_h=500.0)
    assert result.washout_50pct_h >= 0.0


def test_no_washout_when_not_specified():
    result = _default(washout_start_h=None)
    assert result.washout_50pct_h == 0.0


def test_simulate_induction_washout_wrapper():
    result = simulate_induction_washout(
        drug_name="Rifampicin",
        cyp_enzyme="CYP3A4",
        inducer_conc_uM=10.0,
        ec50_uM=1.0,
        max_fold_induction=8.0,
        dosing_duration_h=168.0,
        k_deg_per_h=0.02,
    )
    assert isinstance(result, CYPInductionResult)
    # t_end should be 168 + 7*24 = 336
    assert result.times_h[-1] >= 335.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_empty_cyp_enzyme_raises():
    with pytest.raises(ValueError, match="cyp_enzyme"):
        _default(cyp_enzyme="")


def test_negative_inducer_conc_raises():
    with pytest.raises(ValueError, match="inducer_conc_uM"):
        _default(inducer_conc_uM=-1.0)


def test_zero_ec50_raises():
    with pytest.raises(ValueError, match="ec50_uM"):
        _default(ec50_uM=0.0)


def test_max_fold_below_one_raises():
    with pytest.raises(ValueError, match="max_fold_induction"):
        _default(max_fold_induction=0.5)


def test_zero_kdeg_raises():
    with pytest.raises(ValueError, match="k_deg_per_h"):
        _default(k_deg_per_h=0.0)
