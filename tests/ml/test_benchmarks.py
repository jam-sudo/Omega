"""Tests for benchmark computation with mock data."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omega_pbpk.prediction.adme_predictor import ADMEProperties


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adme(**overrides) -> ADMEProperties:
    defaults = {
        "mw": 300.0,
        "logP": 2.0,
        "logS": -3.0,
        "peff": 1.5,
        "fup": 0.15,
        "rbp": 0.7,
        "clint_3a4": 2.0,
        "clint_2d6": 0.5,
        "herg_ic50_uM": 15.0,
        "confidence": "medium",
        "fup_lo": 0.08,
        "fup_hi": 0.22,
        "clint_3a4_lo": 1.0,
        "clint_3a4_hi": 3.0,
        "peff_lo": 0.75,
        "peff_hi": 2.25,
        "rbp_lo": 0.5,
        "rbp_hi": 0.9,
    }
    defaults.update(overrides)
    return ADMEProperties(**defaults)


@pytest.fixture
def mock_predictor():
    """Mock ADME predictor returning fixed values."""
    mock = MagicMock()
    mock.predict.return_value = _make_adme()
    return mock


# ---------------------------------------------------------------------------
# Tests: fold error computation
# ---------------------------------------------------------------------------


class TestFoldError:
    """Test fold error computation."""

    def test_perfect_prediction(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert compute_fold_error(5.0, 5.0) == 1.0

    def test_2fold_over(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert compute_fold_error(10.0, 5.0) == 2.0

    def test_2fold_under(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert compute_fold_error(5.0, 10.0) == 2.0

    def test_zero_observed_is_nan(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert np.isnan(compute_fold_error(5.0, 0.0))

    def test_zero_predicted_is_nan(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert np.isnan(compute_fold_error(0.0, 5.0))

    def test_nan_input_is_nan(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_fold_error

        assert np.isnan(compute_fold_error(float("nan"), 5.0))


class TestAAFE:
    """Test AAFE computation."""

    def test_perfect_predictions(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_aafe

        assert compute_aafe([1.0, 1.0, 1.0]) == 1.0

    def test_mixed_fold_errors(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_aafe

        # AAFE = 10^(mean(log10([2, 2]))) = 10^log10(2) = 2.0
        result = compute_aafe([2.0, 2.0])
        assert abs(result - 2.0) < 0.01

    def test_empty_list_is_nan(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_aafe

        assert np.isnan(compute_aafe([]))

    def test_skips_nans(self):
        from omega_pbpk.ml.evaluation.benchmarks import compute_aafe

        result = compute_aafe([2.0, float("nan"), 2.0])
        assert abs(result - 2.0) < 0.01


class TestBenchmarkSummary:
    """Test BenchmarkSummary."""

    def test_str_representation(self):
        from omega_pbpk.ml.evaluation.benchmarks import BenchmarkSummary

        summary = BenchmarkSummary(
            n_compounds=5,
            n_success=4,
            n_failed=1,
            aafe_per_metric={"fup": 1.5, "rbp": 1.2},
            pct_within_2fold={"fup": 80.0, "rbp": 100.0},
        )
        text = str(summary)
        assert "4/5" in text
        assert "fup" in text


class TestRunBenchmark:
    """Test the run_benchmark function with mock simulation."""

    def test_run_with_mock_data(self, mock_predictor):
        """Run benchmark with mocked predictor and simulation."""
        from omega_pbpk.ml.evaluation.benchmarks import run_benchmark

        compounds = [
            {
                "name": "test_drug",
                "smiles": "CCO",
                "fup": "0.15",
                "rbp": "0.7",
            },
        ]

        # Mock the simulation internals
        with patch("omega_pbpk.ml.evaluation.benchmarks._run_single_compound") as mock_run:
            from omega_pbpk.ml.evaluation.benchmarks import BenchmarkResult

            mock_run.return_value = BenchmarkResult(
                name="test_drug",
                smiles="CCO",
                predicted={"fup": 0.15, "rbp": 0.7, "cmax": 1.0},
                observed={"fup": 0.15, "rbp": 0.7},
                fold_errors={"fup": 1.0, "rbp": 1.0},
                success=True,
            )
            summary = run_benchmark(compounds, predictor=mock_predictor)

        assert summary.n_compounds == 1
        assert summary.n_success == 1
        assert summary.n_failed == 0

    def test_run_with_no_smiles(self, mock_predictor):
        """Compounds without SMILES should be counted as failed."""
        from omega_pbpk.ml.evaluation.benchmarks import run_benchmark

        compounds = [{"name": "no_smiles_drug"}]
        summary = run_benchmark(compounds, predictor=mock_predictor)

        assert summary.n_compounds == 1
        assert summary.n_failed == 1

    def test_run_empty_list(self, mock_predictor):
        """Empty compound list should return empty summary."""
        from omega_pbpk.ml.evaluation.benchmarks import run_benchmark

        summary = run_benchmark([], predictor=mock_predictor)
        assert summary.n_compounds == 0


class TestMetrics:
    """Test calibration metrics functions."""

    def test_picp(self):
        from omega_pbpk.ml.evaluation.metrics import picp

        # All covered
        assert picp([1.0, 2.0], [0.5, 1.5], [1.5, 2.5]) == 1.0
        # None covered
        assert picp([5.0, 6.0], [0.5, 1.5], [1.5, 2.5]) == 0.0
        # Half covered
        assert picp([1.0, 5.0], [0.5, 1.5], [1.5, 2.5]) == 0.5
        # Empty
        assert picp([], [], []) == 0.0

    def test_mpiw(self):
        from omega_pbpk.ml.evaluation.metrics import mpiw

        assert mpiw([0.0, 1.0], [1.0, 3.0]) == 1.5
        assert mpiw([], []) == 0.0

    def test_calibration_result_fields(self):
        from omega_pbpk.ml.evaluation.metrics import CalibrationResult

        r = CalibrationResult(
            property_name="fup",
            target_coverage=0.9,
            actual_coverage=0.85,
            n_samples=100,
            mean_interval_width=0.1,
            is_calibrated=False,
            adjustment_factor=1.2,
        )
        assert r.property_name == "fup"
        assert not r.is_calibrated
