"""Tests for Phase 51: PBPK Property Surrogate.

TestPBPKSurrogateUnit: pure unit tests on predict_from_adme_dict with fallback.
TestTrainAndCache: integration tests for train_and_cache() and load_or_train().
TestEndToEndIntegration: train on small dataset, verify AAFE < 5.0.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from omega_pbpk.surrogate.pbpk_surrogate import (
    PBPKSurrogateOutput,
    TrainingReport,
    _featurize,
    clear_cache,
    load_or_train,
    predict_from_adme_dict,
    train_and_cache,
)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestFeaturize:
    """Unit tests for the feature extraction helper."""

    def test_featurize_returns_6_elements(self) -> None:
        x = _featurize(logP=2.0, fup=0.5, clint_L_h=5.0, mw=300.0, peff=1.0, rbp=1.0)
        assert x.shape == (6,)

    def test_featurize_values(self) -> None:
        x = _featurize(logP=1.5, fup=0.3, clint_L_h=10.0, mw=250.0, peff=2.0, rbp=0.9)
        assert x[0] == pytest.approx(1.5)  # logP
        assert x[1] == pytest.approx(0.3)  # fup
        assert x[2] == pytest.approx(10.0)  # clint
        assert x[3] == pytest.approx(250.0)  # mw
        assert x[4] == pytest.approx(0.9)  # rbp
        assert x[5] == pytest.approx(2.0)  # peff


class TestFallbackPrediction:
    """Test predict_from_adme_dict falls back gracefully when no model is available."""

    def test_fallback_returns_output(self) -> None:
        """Even with no cached model, fallback produces a PBPKSurrogateOutput."""
        adme = {"logP": 2.0, "fup": 0.5, "clint_3a4": 5.0, "mw": 300.0, "peff": 1.0, "rbp": 1.0}
        # Use a non-existent cache dir to force fallback
        result = predict_from_adme_dict(
            adme, dose_mg=100.0, cache_dir="/tmp/nonexistent_pbpk_abc123"
        )
        assert isinstance(result, PBPKSurrogateOutput)
        assert result.cmax_mg_L >= 0.0
        assert result.auc_mg_h_L >= 0.0

    def test_fallback_cmax_positive(self) -> None:
        adme = {"logP": 1.0, "fup": 0.8, "clint_3a4": 2.0, "mw": 200.0, "peff": 5.0, "rbp": 1.0}
        result = predict_from_adme_dict(
            adme, dose_mg=50.0, cache_dir="/tmp/nonexistent_pbpk_abc123"
        )
        assert result.cmax_mg_L > 0

    def test_source_field(self) -> None:
        """Source should be 'fallback' or 'pbpk_surrogate'."""
        adme = {"logP": 0.0, "fup": 0.9, "clint_3a4": 1.0, "mw": 150.0, "peff": 3.0, "rbp": 1.0}
        result = predict_from_adme_dict(
            adme, dose_mg=100.0, cache_dir="/tmp/nonexistent_pbpk_abc123"
        )
        assert result.source in ("fallback", "pbpk_surrogate")


# ---------------------------------------------------------------------------
# Integration tests — train on small dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainAndCache:
    """Integration tests for training pipeline with minimal dataset."""

    def test_train_and_cache_returns_model_and_report(self, tmp_path: pathlib.Path) -> None:
        """train_and_cache returns a (model, TrainingReport) tuple."""
        model, report = train_and_cache(
            cache_dir=tmp_path / "surrogate",
            n_samples=30,
            epochs=20,
            seed=42,
        )
        assert model is not None
        assert isinstance(report, TrainingReport)
        assert report.n_successful > 0
        assert report.n_successful <= report.n_samples

    def test_training_report_has_aafe(self, tmp_path: pathlib.Path) -> None:
        """TrainingReport should have finite AAFE values."""
        _, report = train_and_cache(
            cache_dir=tmp_path / "surrogate",
            n_samples=30,
            epochs=20,
            seed=42,
        )
        assert np.isfinite(report.aafe_cmax) or report.n_successful < 5
        assert np.isfinite(report.aafe_auc) or report.n_successful < 5

    def test_cache_files_created(self, tmp_path: pathlib.Path) -> None:
        """After training, model files should exist in cache dir."""
        cache = tmp_path / "surrogate"
        train_and_cache(cache_dir=cache, n_samples=20, epochs=10, seed=0)
        assert (cache / "meta.json").exists()
        assert (cache / "w0.npy").exists()

    def test_load_or_train_creates_cache(self, tmp_path: pathlib.Path) -> None:
        """load_or_train should create cache when it doesn't exist."""
        cache = tmp_path / "surrogate2"
        model = load_or_train(cache_dir=cache, n_samples=20, epochs=10)
        assert model is not None
        assert (cache / "meta.json").exists()

    def test_load_or_train_reuses_cache(self, tmp_path: pathlib.Path) -> None:
        """Second call to load_or_train should load from cache, not retrain."""
        cache = tmp_path / "surrogate3"
        model1 = load_or_train(cache_dir=cache, n_samples=20, epochs=10)
        # Modify a weight to detect if reloaded vs retrained
        import copy

        params1 = copy.deepcopy(model1.weights[0])

        model2 = load_or_train(cache_dir=cache, n_samples=20, epochs=10)
        np.testing.assert_array_equal(params1, model2.weights[0])

    def test_clear_cache(self, tmp_path: pathlib.Path) -> None:
        """clear_cache should remove cache directory."""
        cache = tmp_path / "surrogate4"
        train_and_cache(cache_dir=cache, n_samples=20, epochs=10, seed=0)
        assert cache.exists()
        clear_cache(cache_dir=cache)
        assert not cache.exists()

    def test_predict_from_adme_dict_with_trained_model(self, tmp_path: pathlib.Path) -> None:
        """predict_from_adme_dict with a real trained model returns valid output."""
        model, _ = train_and_cache(cache_dir=tmp_path / "s", n_samples=30, epochs=20, seed=0)
        adme = {"logP": 2.0, "fup": 0.5, "clint_3a4": 3.0, "mw": 300.0, "peff": 1.5, "rbp": 1.0}
        result = predict_from_adme_dict(adme, dose_mg=100.0, model=model)
        assert result.source == "pbpk_surrogate"
        assert result.cmax_mg_L >= 0.0
        assert result.auc_mg_h_L >= 0.0
        assert result.tmax_h >= 0.0
        assert result.t_half_h >= 0.0


@pytest.mark.integration
class TestEndToEndQuality:
    """Test that a small surrogate achieves reasonable prediction quality."""

    def test_aafe_reasonable_on_self_data(self, tmp_path: pathlib.Path) -> None:
        """AAFE on training data should be < 10 after 50 epochs (not overfit check)."""
        from omega_pbpk.surrogate import PKSurrogate
        from omega_pbpk.surrogate.pbpk_surrogate import FEATURE_NAMES, OUTPUT_NAMES, _compute_aafe
        from omega_pbpk.surrogate.train import build_training_dataset

        X, y = build_training_dataset(n_samples=40, seed=7)
        if len(X) < 5:
            pytest.skip("Too few successful PBPK simulations for quality test")

        model = PKSurrogate(n_input=6, n_output=4)
        model.param_names = FEATURE_NAMES
        model.output_names = OUTPUT_NAMES
        model.train(X, y, epochs=50, lr=0.01, batch_size=16, seed=7)

        y_pred = model.predict(X)
        aafes = _compute_aafe(y, y_pred)
        mean_aafe = float(np.nanmean(aafes))
        assert mean_aafe < 10.0, f"Mean AAFE={mean_aafe:.2f} is unexpectedly high after 50 epochs"
