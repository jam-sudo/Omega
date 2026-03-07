"""Tests for drug release from matrix tablets — Phase 267."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_release_from_matrix import (
    MatrixReleaseResult,
    compare_release_models,
    simulate_matrix_release,
)


def _run(model="first_order", **kwargs):
    defaults = dict(drug_name="TestDrug", dose_mg=100.0, release_model=model)
    defaults.update(kwargs)
    return simulate_matrix_release(**defaults)


# --- Return type ---


def test_returns_result_instance():
    r = _run()
    assert isinstance(r, MatrixReleaseResult)


def test_all_models_return_result():
    for model in ("zero_order", "first_order", "higuchi", "korsmeyer_peppas", "weibull"):
        r = _run(model=model)
        assert isinstance(r, MatrixReleaseResult)


# --- Initial condition: released starts at 0 ---


def test_released_starts_at_zero():
    r = _run()
    assert r.released_mg[0] == pytest.approx(0.0, abs=1e-9)


def test_released_pct_starts_at_zero():
    r = _run()
    assert r.released_pct[0] == pytest.approx(0.0, abs=1e-9)


# --- Complete release for long simulation ---


def test_first_order_near_complete_release():
    r = simulate_matrix_release("DrugA", 100.0, "first_order", t_end_h=100.0, k_release=0.2)
    assert r.released_pct[-1] >= 95.0


def test_zero_order_complete_release():
    # k_release=0.1 => t100=10h; t_end=20h should be ~100%
    r = simulate_matrix_release("DrugA", 100.0, "zero_order", t_end_h=20.0, k_release=0.1)
    assert r.released_pct[-1] == pytest.approx(100.0, abs=2.0)


def test_weibull_near_complete_release():
    r = simulate_matrix_release("DrugB", 50.0, "weibull", t_end_h=100.0, k_release=0.2)
    assert r.released_pct[-1] >= 95.0


# --- t50 < t80 ---


def test_t50_less_than_t80():
    # Use k_release values that ensure each model reaches both 50% and 80%
    # within t_end. For Higuchi, k_release=0.1 with dose=100 only releases ~10%
    # by t=100h, so we use k_release=1.0 for that model.
    model_krelease = {
        "zero_order": 0.1,
        "first_order": 0.1,
        "higuchi": 1.0,
        "korsmeyer_peppas": 0.1,
        "weibull": 0.1,
    }
    for model, kr in model_krelease.items():
        r = simulate_matrix_release("DrugA", 100.0, model, t_end_h=100.0, k_release=kr)
        assert r.t50_h < r.t80_h, f"t50 < t80 failed for {model}"


# --- Valid t50 for all models ---


def test_all_models_valid_t50():
    for model in ("zero_order", "first_order", "higuchi", "korsmeyer_peppas", "weibull"):
        r = simulate_matrix_release("DrugA", 100.0, model, t_end_h=100.0, k_release=0.1)
        assert r.t50_h > 0.0, f"t50 should be > 0 for {model}"


# --- Monotonically non-decreasing ---


def test_released_mg_monotonically_nondecreasing():
    for model in ("zero_order", "first_order", "higuchi", "korsmeyer_peppas", "weibull"):
        r = _run(model=model)
        for i in range(1, len(r.released_mg)):
            assert r.released_mg[i] >= r.released_mg[i - 1] - 1e-9, (
                f"Non-monotonic at step {i} for {model}"
            )


# --- compare_release_models ---


def test_compare_returns_five_results():
    results = compare_release_models("DrugA", 100.0)
    assert len(results) == 5


def test_compare_all_different_models():
    results = compare_release_models("DrugA", 100.0)
    models = {r.release_model for r in results}
    assert len(models) == 5


def test_compare_sorted_by_t50_ascending():
    results = compare_release_models("DrugA", 100.0)
    for i in range(1, len(results)):
        assert results[i].t50_h >= results[i - 1].t50_h


# --- fit_r2 always 1.0 ---


def test_fit_r2_is_1():
    r = _run()
    assert r.fit_r2 == pytest.approx(1.0)


# --- release_rate_avg positive ---


def test_release_rate_avg_positive():
    r = simulate_matrix_release("DrugA", 100.0, "first_order", t_end_h=100.0, k_release=0.1)
    assert r.release_rate_avg_mg_per_h > 0.0


# --- Validation errors ---


def test_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_matrix_release("DrugA", 0.0, "first_order")


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_matrix_release("DrugA", -10.0, "first_order")


def test_invalid_model_raises():
    with pytest.raises(ValueError, match="release_model"):
        simulate_matrix_release("DrugA", 100.0, "bogus_model")


def test_t_end_le_zero_raises():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_matrix_release("DrugA", 100.0, "first_order", t_end_h=0.0)
