"""Tests for Phase 584 - Pharmacokinetic Outlier Detection."""

import math

import pytest

from omega_pbpk.clinical.pk_outlier_detection import (
    PKOutlierDetectionResult,
    detect_pk_outliers,
    simulate_population_and_detect,
)


@pytest.fixture
def normal_data():
    """50 values centered around 100 with small spread."""
    import random

    rng = random.Random(123)
    return [rng.gauss(100, 10) for _ in range(50)]


@pytest.fixture
def result_2sd(normal_data):
    return detect_pk_outliers(normal_data, method="2SD")


@pytest.fixture
def result_iqr(normal_data):
    return detect_pk_outliers(normal_data, method="IQR")


def test_return_type(result_2sd):
    assert isinstance(result_2sd, PKOutlierDetectionResult)


def test_n_subjects(result_2sd, normal_data):
    assert result_2sd.n_subjects == len(normal_data)


def test_2sd_outliers_reasonable(result_2sd):
    # For normal data, ~5% outliers expected → 0-5 for n=50
    assert 0 <= result_2sd.n_outliers <= 5


def test_iqr_method_works(result_iqr):
    assert result_iqr.method == "IQR"
    assert result_iqr.n_subjects == 50


def test_outlier_indices_in_range(result_2sd):
    for idx in result_2sd.outlier_indices:
        assert 0 <= idx < result_2sd.n_subjects


def test_cv_positive(result_2sd):
    assert result_2sd.cv_auc_pct > 0


def test_mean_calculation(normal_data):
    r = detect_pk_outliers(normal_data)
    expected_mean = sum(normal_data) / len(normal_data)
    assert abs(r.mean_auc - expected_mean) < 1e-9


def test_sd_calculation(normal_data):
    r = detect_pk_outliers(normal_data)
    mean = sum(normal_data) / len(normal_data)
    var = sum((x - mean) ** 2 for x in normal_data) / (len(normal_data) - 1)
    expected_sd = math.sqrt(var)
    assert abs(r.sd_auc - expected_sd) < 1e-9


def test_cv_calculation(result_2sd):
    expected_cv = result_2sd.sd_auc / result_2sd.mean_auc * 100.0
    assert abs(result_2sd.cv_auc_pct - expected_cv) < 1e-9


def test_2sd_thresholds(result_2sd):
    assert abs(result_2sd.threshold_lower - (result_2sd.mean_auc - 2 * result_2sd.sd_auc)) < 1e-9
    assert abs(result_2sd.threshold_upper - (result_2sd.mean_auc + 2 * result_2sd.sd_auc)) < 1e-9


def test_outlier_aucs_match_indices(normal_data):
    r = detect_pk_outliers(normal_data)
    for idx, auc in zip(r.outlier_indices, r.outlier_aucs):
        assert abs(auc - normal_data[idx]) < 1e-9


def test_n_outliers_matches_indices(result_2sd):
    assert result_2sd.n_outliers == len(result_2sd.outlier_indices)


def test_no_outliers_tight_data():
    """All identical values → no outliers."""
    data = [100.0] * 10
    r = detect_pk_outliers(data)
    assert r.n_outliers == 0
    assert r.outlier_indices == ()


def test_clear_outlier_detected():
    """One extreme value should be detected."""
    data = [100.0] * 20 + [500.0]
    r = detect_pk_outliers(data)
    assert 20 in r.outlier_indices


def test_seed_reproducibility():
    r1 = simulate_population_and_detect("Drug", 10.0, 50.0, seed=42)
    r2 = simulate_population_and_detect("Drug", 10.0, 50.0, seed=42)
    assert r1.auc_values == r2.auc_values
    assert r1.outlier_indices == r2.outlier_indices


def test_different_seeds_differ():
    r1 = simulate_population_and_detect("Drug", 10.0, 50.0, seed=42)
    r2 = simulate_population_and_detect("Drug", 10.0, 50.0, seed=99)
    assert r1.auc_values != r2.auc_values


def test_simulate_n_subjects():
    r = simulate_population_and_detect("Drug", 10.0, 50.0, n_subjects=30)
    assert r.n_subjects == 30


def test_simulate_iqr_method():
    r = simulate_population_and_detect("Drug", 10.0, 50.0, method="IQR")
    assert r.method == "IQR"


def test_validation_too_few_points():
    with pytest.raises(ValueError):
        detect_pk_outliers([1.0, 2.0])


def test_validation_invalid_method():
    with pytest.raises(ValueError):
        detect_pk_outliers([1.0, 2.0, 3.0], method="MAD")


def test_frozen_result(result_2sd):
    with pytest.raises(AttributeError):
        result_2sd.n_subjects = 999


def test_auc_values_tuple(result_2sd):
    assert isinstance(result_2sd.auc_values, tuple)
    assert isinstance(result_2sd.outlier_indices, tuple)
    assert isinstance(result_2sd.outlier_aucs, tuple)
