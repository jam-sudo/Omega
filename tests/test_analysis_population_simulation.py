"""Tests for analysis/population_simulation module — Phase 700."""

import pytest

from omega_pbpk.analysis.population_simulation import (
    PopulationPKResult,
    compute_percentiles,
    population_summary_stats,
    simulate_population_pk,
)


def test_returns_population_pk_result():
    result = simulate_population_pk(n_subjects=20)
    assert isinstance(result, PopulationPKResult)


def test_subject_aucs_length_matches_n_subjects():
    result = simulate_population_pk(n_subjects=30)
    assert len(result.subject_aucs) == 30


def test_all_aucs_positive():
    result = simulate_population_pk(n_subjects=20)
    assert all(auc > 0 for auc in result.subject_aucs)


def test_all_cmaxes_positive():
    result = simulate_population_pk(n_subjects=20)
    assert all(cmax > 0 for cmax in result.subject_cmaxes)


def test_auc_percentile_ordering():
    """auc_p5 < auc_p50 < auc_p95."""
    result = simulate_population_pk(n_subjects=100, omega_cl=0.4, omega_vd=0.3)
    assert result.auc_p5 < result.auc_p50 < result.auc_p95


def test_cmax_percentile_ordering():
    """cmax_p5 < cmax_p50 < cmax_p95."""
    result = simulate_population_pk(n_subjects=100, omega_cl=0.4)
    assert result.cmax_p5 < result.cmax_p50 < result.cmax_p95


def test_cl_cv_pct_positive_when_omega_nonzero():
    result = simulate_population_pk(n_subjects=50, omega_cl=0.4)
    assert result.cl_cv_pct > 0.0


def test_auc_cv_pct_positive():
    result = simulate_population_pk(n_subjects=50, omega_cl=0.4, omega_vd=0.3)
    assert result.auc_cv_pct > 0.0


def test_population_summary_stats_returns_dict():
    result = simulate_population_pk(n_subjects=20)
    stats = population_summary_stats(result)
    assert isinstance(stats, dict)


def test_population_summary_stats_expected_keys():
    result = simulate_population_pk(n_subjects=20)
    stats = population_summary_stats(result)
    expected_keys = {
        "auc_mean",
        "auc_median",
        "auc_cv",
        "cmax_mean",
        "cmax_median",
        "cmax_cv",
        "n_subjects",
    }
    assert expected_keys.issubset(stats.keys())


def test_population_summary_stats_n_subjects_matches():
    result = simulate_population_pk(n_subjects=25)
    stats = population_summary_stats(result)
    assert stats["n_subjects"] == 25


def test_compute_percentiles_simple():
    """[1, 2, 3, 4, 5] -> p50 = 3.0."""
    result = compute_percentiles([1.0, 2.0, 3.0, 4.0, 5.0], [50.0])
    assert abs(result[0] - 3.0) < 1e-9


def test_compute_percentiles_p0_returns_min():
    vals = [10.0, 20.0, 30.0]
    result = compute_percentiles(vals, [0.0])
    assert result[0] == 10.0


def test_compute_percentiles_p100_returns_max():
    vals = [10.0, 20.0, 30.0]
    result = compute_percentiles(vals, [100.0])
    assert result[0] == 30.0


def test_larger_omega_larger_cv():
    """Larger omega -> larger CV%."""
    low_var = simulate_population_pk(n_subjects=200, omega_cl=0.1, seed=99)
    high_var = simulate_population_pk(n_subjects=200, omega_cl=0.8, seed=99)
    assert high_var.cl_cv_pct > low_var.cl_cv_pct


def test_reproducible_with_same_seed():
    """Same seed produces identical results."""
    result1 = simulate_population_pk(n_subjects=50, seed=42)
    result2 = simulate_population_pk(n_subjects=50, seed=42)
    assert result1.subject_aucs == result2.subject_aucs
    assert result1.subject_cmaxes == result2.subject_cmaxes


def test_different_seeds_produce_different_results():
    result1 = simulate_population_pk(n_subjects=50, seed=1)
    result2 = simulate_population_pk(n_subjects=50, seed=999)
    assert result1.subject_aucs != result2.subject_aucs


def test_n_subjects_in_result_matches_input():
    result = simulate_population_pk(n_subjects=77)
    assert result.n_subjects == 77


def test_validation_n_subjects_lt_2_raises():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_population_pk(n_subjects=1)


def test_validation_cl_mean_zero_raises():
    with pytest.raises(ValueError, match="cl_mean"):
        simulate_population_pk(n_subjects=10, cl_mean=0.0)


def test_validation_cl_mean_negative_raises():
    with pytest.raises(ValueError, match="cl_mean"):
        simulate_population_pk(n_subjects=10, cl_mean=-1.0)


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_population_pk(n_subjects=10, dose_mg=0.0)


def test_fraction_above_mec_with_mec():
    """Fraction above MEC is in [0, 1]."""
    result = simulate_population_pk(n_subjects=50, mec=0.5)
    assert 0.0 <= result.fraction_above_mec <= 1.0


def test_fraction_above_mec_default_zero():
    """Without MEC, fraction_above_mec defaults to 0."""
    result = simulate_population_pk(n_subjects=20)
    assert result.fraction_above_mec == 0.0
