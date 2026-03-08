"""Tests for Phase 1000 — Thousand-compounds ADMET benchmark."""

import pytest

from omega_pbpk.prediction.thousand_compounds_benchmark import BenchmarkResult, run_benchmark

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_returns_benchmark_result():
    result = run_benchmark(n_compounds=100, seed=42)
    assert isinstance(result, BenchmarkResult)


def test_n_compounds_default():
    result = run_benchmark()
    assert result.n_compounds == 1000


def test_n_compounds_custom():
    result = run_benchmark(n_compounds=100, seed=1)
    assert result.n_compounds == 100


def test_mean_logp_in_range():
    result = run_benchmark()
    assert -2.0 < result.mean_logp < 7.0


def test_mean_mw_in_range():
    result = run_benchmark()
    assert 100.0 < result.mean_mw < 700.0


def test_mean_solubility_positive():
    result = run_benchmark()
    assert result.mean_solubility_mg_mL > 0.0


def test_mean_papp_positive():
    result = run_benchmark()
    assert result.mean_papp_cm_s > 0.0


def test_mean_t_half_positive():
    result = run_benchmark()
    assert result.mean_t_half_min > 0.0


def test_lipinski_pct_in_range():
    result = run_benchmark()
    assert 0.0 <= result.pct_lipinski_pass <= 100.0


def test_bcs_pcts_sum_to_100():
    result = run_benchmark()
    total = (
        result.pct_bcs_class_i
        + result.pct_bcs_class_ii
        + result.pct_bcs_class_iii
        + result.pct_bcs_class_iv
    )
    assert total == pytest.approx(100.0, abs=1e-6)


def test_best_score_ge_worst():
    result = run_benchmark()
    assert result.best_drug_likeness_score >= result.worst_drug_likeness_score


def test_best_name_nonempty():
    result = run_benchmark()
    assert len(result.best_drug_likeness_name) > 0


def test_worst_name_nonempty():
    result = run_benchmark()
    assert len(result.worst_drug_likeness_name) > 0


def test_runtime_nonnegative():
    result = run_benchmark()
    assert result.runtime_ms >= 0.0


def test_notes_nonempty():
    result = run_benchmark()
    assert len(result.notes) > 0


def test_deterministic_same_seed():
    """Running twice with the same seed yields identical results."""
    r1 = run_benchmark(n_compounds=200, seed=99)
    r2 = run_benchmark(n_compounds=200, seed=99)
    assert r1.mean_logp == r2.mean_logp
    assert r1.mean_mw == r2.mean_mw
    assert r1.pct_lipinski_pass == r2.pct_lipinski_pass
    assert r1.best_drug_likeness_name == r2.best_drug_likeness_name
    assert r1.worst_drug_likeness_name == r2.worst_drug_likeness_name


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different mean logP."""
    r1 = run_benchmark(n_compounds=500, seed=1)
    r2 = run_benchmark(n_compounds=500, seed=9999)
    # It is astronomically unlikely they agree to 6 decimal places.
    assert r1.mean_logp != r2.mean_logp
