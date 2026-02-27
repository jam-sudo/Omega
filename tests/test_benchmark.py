"""Tests for the multi-drug benchmark validation suite.

Covers:
  1. run_benchmark_suite returns a dict with required top-level keys
  2. At least 3 drugs are present in the suite results
  3. Each per-drug result dict has the expected AUC/Cmax/Tmax metric keys
  4. Caffeine AUC relative error is within a generous PBPK tolerance (< 80%)
  5. MetricResult fields are all floats
  6. Acceptance thresholds from acceptance.json are loaded and applied

Run: pytest tests/test_benchmark.py -v --tb=short
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUITE_DIR = Path("benchmarks")
OUT_DIR = Path("outputs/test_benchmark")

REQUIRED_SUMMARY_KEYS = {"n_drugs", "n_pass", "overall_pass", "results", "thresholds", "suite"}
REQUIRED_METRIC_KEYS = {
    "auc_observed",
    "auc_simulated",
    "auc_relative_error",
    "cmax_observed",
    "cmax_simulated",
    "cmax_relative_error",
    "tmax_observed_h",
    "tmax_simulated_h",
    "tmax_abs_error_h",
    "rmse_mg_per_L",
}


@pytest.fixture(scope="module")
def benchmark_summary():
    """Run the benchmark suite once for the entire module."""
    from omega_pbpk.validation.benchmarks import run_benchmark_suite

    return run_benchmark_suite(SUITE_DIR, OUT_DIR)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    """Integration tests for the full benchmark validation suite."""

    def test_benchmark_runs(self, benchmark_summary):
        """run_benchmark_suite returns a dict with all required top-level keys."""
        assert isinstance(benchmark_summary, dict), "Summary must be a dict"
        assert REQUIRED_SUMMARY_KEYS.issubset(benchmark_summary.keys()), (
            f"Missing keys: {REQUIRED_SUMMARY_KEYS - set(benchmark_summary.keys())}"
        )

    def test_benchmark_n_drugs(self, benchmark_summary):
        """Benchmark suite covers at least 3 drugs (we expect exactly 5)."""
        n = benchmark_summary["n_drugs"]
        assert n >= 3, f"Expected at least 3 drugs, got {n}"

    def test_benchmark_metrics_keys(self, benchmark_summary):
        """Every per-drug result dict contains all expected AUC/Cmax/Tmax metric keys."""
        results = benchmark_summary["results"]
        assert len(results) > 0, "No results found"
        for entry in results:
            drug = entry["drug"]
            assert "metrics" in entry, f"Missing 'metrics' key for drug {drug}"
            metrics = entry["metrics"]
            missing = REQUIRED_METRIC_KEYS - set(metrics.keys())
            assert not missing, f"Drug '{drug}' metrics missing keys: {missing}"

    def test_caffeine_within_tolerance(self, benchmark_summary):
        """Caffeine AUC relative error should be < 80% (generous PBPK tolerance).

        A PBPK model is not expected to achieve perfect accuracy against
        simplified 1-compartment synthetic reference data. We use 0.80 as a
        very permissive upper bound to guard against catastrophic failures.
        """
        caffeine_results = [r for r in benchmark_summary["results"] if r["drug"] == "caffeine"]
        assert caffeine_results, "No caffeine result found in benchmark summary"
        auc_re = caffeine_results[0]["metrics"]["auc_relative_error"]
        assert auc_re < 0.80, (
            f"Caffeine AUC relative error {auc_re:.3f} exceeds generous 80% tolerance"
        )

    def test_metric_result_types(self, benchmark_summary):
        """All metric values in every result entry must be floats (not None/str/etc.)."""
        results = benchmark_summary["results"]
        for entry in results:
            drug = entry["drug"]
            for key, value in entry["metrics"].items():
                assert isinstance(value, (int, float)), (
                    f"Drug '{drug}', metric '{key}': expected float, got {type(value).__name__}"
                )
                assert np.isfinite(value), (
                    f"Drug '{drug}', metric '{key}': value {value} is not finite"
                )

    def test_acceptance_json_loaded(self, benchmark_summary):
        """Acceptance thresholds from acceptance.json are loaded and present in summary.

        Verifies that 'thresholds' contains the three required acceptance
        criteria keys and that they are applied per-drug (each result has
        a 'passes' sub-dict).
        """
        thresholds = benchmark_summary["thresholds"]
        required_threshold_keys = {
            "auc_relative_error_max",
            "cmax_relative_error_max",
            "tmax_abs_error_h_max",
        }
        assert required_threshold_keys.issubset(thresholds.keys()), (
            f"Missing threshold keys: {required_threshold_keys - set(thresholds.keys())}"
        )

        # Verify each drug result has a 'passes' dict reflecting threshold checks
        for entry in benchmark_summary["results"]:
            drug = entry["drug"]
            assert "passes" in entry, f"Drug '{drug}' is missing 'passes' dict"
            passes = entry["passes"]
            assert "auc" in passes, f"Drug '{drug}' missing 'auc' in passes"
            assert "cmax" in passes, f"Drug '{drug}' missing 'cmax' in passes"
            assert "tmax" in passes, f"Drug '{drug}' missing 'tmax' in passes"

            # Verify the pass/fail logic is consistent with the thresholds
            metrics = entry["metrics"]
            auc_threshold = thresholds["auc_relative_error_max"]
            cmax_threshold = thresholds["cmax_relative_error_max"]
            tmax_threshold = thresholds["tmax_abs_error_h_max"]

            expected_auc_pass = metrics["auc_relative_error"] <= auc_threshold
            expected_cmax_pass = metrics["cmax_relative_error"] <= cmax_threshold
            expected_tmax_pass = metrics["tmax_abs_error_h"] <= tmax_threshold

            assert passes["auc"] == expected_auc_pass, (
                f"Drug '{drug}': auc pass mismatch "
                f"(RE={metrics['auc_relative_error']:.3f}, threshold={auc_threshold})"
            )
            assert passes["cmax"] == expected_cmax_pass, (
                f"Drug '{drug}': cmax pass mismatch "
                f"(RE={metrics['cmax_relative_error']:.3f}, threshold={cmax_threshold})"
            )
            assert passes["tmax"] == expected_tmax_pass, (
                f"Drug '{drug}': tmax pass mismatch "
                f"(AE={metrics['tmax_abs_error_h']:.2f}h, threshold={tmax_threshold})"
            )
