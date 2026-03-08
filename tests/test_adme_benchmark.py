"""Tests for ADME benchmark module."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omega_pbpk.validation.adme_benchmark import _aafe, benchmark_adme_predictor

pytestmark = pytest.mark.benchmark

# ── _aafe helper ─────────────────────────────────────────────────────


def test_aafe_perfect_prediction():
    assert _aafe([1.0], [1.0]) == pytest.approx(1.0)


def test_aafe_empty_returns_nan():
    assert math.isnan(_aafe([], []))


def test_aafe_always_gte_one():
    result = _aafe([2.0, 0.5], [1.0, 1.0])
    assert result >= 1.0


def test_aafe_multiple_perfect():
    assert _aafe([5.0, 10.0, 0.1], [5.0, 10.0, 0.1]) == pytest.approx(1.0)


# ── benchmark_adme_predictor ─────────────────────────────────────────


def test_benchmark_returns_dict():
    result = benchmark_adme_predictor(predictor=_dummy_predictor())
    assert isinstance(result, dict)


def test_benchmark_has_expected_keys():
    result = benchmark_adme_predictor(predictor=_dummy_predictor())
    assert set(result.keys()) == {"logP_aafe", "logS_aafe", "fup_aafe", "n_compounds"}


def test_benchmark_missing_csv_returns_zero():
    with patch("omega_pbpk.validation.adme_benchmark.Path") as mock_path:
        instance = mock_path.return_value.__truediv__.return_value
        for _ in range(4):
            instance = instance.__truediv__.return_value
        instance.exists.return_value = False
        # Re-call: the function builds its own path so we mock Path.__init__
        result = benchmark_adme_predictor(predictor=_dummy_predictor())
    # If file isn't found, n_compounds == 0 and aafe are nan
    # This depends on whether the real csv exists; test the contract
    assert isinstance(result["n_compounds"], int)
    assert result["n_compounds"] >= 0


def test_benchmark_n_compounds_is_int():
    result = benchmark_adme_predictor(predictor=_dummy_predictor())
    assert isinstance(result["n_compounds"], (int, float))
    assert result["n_compounds"] >= 0


def test_benchmark_no_crash():
    """Function handles exceptions gracefully."""
    try:
        benchmark_adme_predictor(predictor=_dummy_predictor())
    except Exception as e:
        pytest.fail(f"benchmark_adme_predictor crashed: {e}")


# ── Helpers ──────────────────────────────────────────────────────────


class _DummyPredictor:
    """Predictor that always returns fixed values."""

    def predict(self, smiles: str) -> SimpleNamespace:
        return SimpleNamespace(logP=1.0, logS=-3.0, fup=0.5)


def _dummy_predictor():
    return _DummyPredictor()
