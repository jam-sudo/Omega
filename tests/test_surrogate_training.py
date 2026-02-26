"""Tests for real-PBPK surrogate training pipeline."""
import numpy as np
import pytest


class TestBuildTrainingDataset:
    def test_returns_arrays(self):
        from omega_pbpk.surrogate.train import build_training_dataset
        X, y = build_training_dataset(n_samples=5, seed=0)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_shapes_compatible(self):
        from omega_pbpk.surrogate.train import build_training_dataset
        X, y = build_training_dataset(n_samples=5, seed=0)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 6   # logP, fup, clint, mw, rbp, peff
        assert y.shape[1] == 4   # Cmax, AUC, Tmax, t_half

    def test_outputs_positive(self):
        from omega_pbpk.surrogate.train import build_training_dataset
        X, y = build_training_dataset(n_samples=5, seed=0)
        assert np.all(y >= 0)

    def test_reproducible_with_seed(self):
        from omega_pbpk.surrogate.train import build_training_dataset
        X1, y1 = build_training_dataset(n_samples=3, seed=42)
        X2, y2 = build_training_dataset(n_samples=3, seed=42)
        np.testing.assert_array_equal(X1, X2)


class TestTrainSurrogate:
    def test_smoke(self):
        from omega_pbpk.surrogate.train import train_surrogate
        model = train_surrogate(n_samples=5, epochs=10, seed=0)
        assert model is not None

    def test_predict_shape(self):
        from omega_pbpk.surrogate.train import train_surrogate, build_training_dataset
        model = train_surrogate(n_samples=5, epochs=10, seed=0)
        X, _ = build_training_dataset(n_samples=3, seed=1)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0], 4)

    def test_surrogate_save_load(self, tmp_path):
        from omega_pbpk.surrogate.train import train_surrogate
        from omega_pbpk.surrogate import PKSurrogate
        save_path = str(tmp_path / "test_surrogate.npz")
        model = train_surrogate(n_samples=5, epochs=5, seed=0, save_path=save_path)
        # PKSurrogate.save() creates a directory; .npz suffix is stripped
        dir_path = save_path[:-4] if save_path.endswith(".npz") else save_path
        loaded = PKSurrogate.load(dir_path)
        assert loaded.n_input == model.n_input
        assert loaded.n_output == model.n_output
