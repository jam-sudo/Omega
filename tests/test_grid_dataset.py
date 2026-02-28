"""Tests for grid-based synthetic dataset generation and surrogate retraining.

All tests are marked @pytest.mark.slow and are skipped in regular CI.
Run explicitly with: pytest -m slow tests/test_grid_dataset.py
"""

import numpy as np
import pytest

from omega_pbpk.surrogate.data_generator import generate_grid_dataset
from omega_pbpk.surrogate.train import train_with_grid_data


@pytest.mark.slow
def test_grid_dataset_generation():
    data = generate_grid_dataset(n_samples=30, seed=0, verbose=False)
    assert data.X.shape[0] >= 20, f"Too few valid samples: {data.X.shape[0]}"
    assert data.X.shape[1] == 6
    assert data.y.shape[1] == 4
    assert np.all(data.y > 0), "Non-positive PK metric found"
    assert np.all(np.isfinite(data.X)), "Non-finite values in X"
    assert np.all(np.isfinite(data.y)), "Non-finite values in y"


@pytest.mark.slow
def test_grid_dataset_param_names():
    """Feature order must match PKSurrogate.EXPECTED_FEATURES."""
    from omega_pbpk.surrogate import PKSurrogate
    from omega_pbpk.surrogate.data_generator import GRID_PARAM_NAMES

    data = generate_grid_dataset(n_samples=20, seed=1, verbose=False)
    assert data.param_names == GRID_PARAM_NAMES
    assert data.param_names == PKSurrogate.EXPECTED_FEATURES


@pytest.mark.slow
def test_train_with_grid_data():
    metrics = train_with_grid_data(
        n_samples=50,
        epochs=10,
        output_dir="/tmp/omega_test_models/",
        verbose=False,
    )
    assert "test_auc_r2" in metrics
    assert "test_cmax_r2" in metrics
    assert "training_loss" in metrics
    # R² should not be completely degenerate with even minimal training
    assert metrics["test_auc_r2"] > -1.0, "AUC R² too low (complete failure)"
    assert metrics["test_cmax_r2"] > -1.0, "Cmax R² too low (complete failure)"
    assert metrics["n_train"] > 0
    assert metrics["n_val"] > 0
    assert metrics["n_test"] > 0


@pytest.mark.slow
def test_conformal_calibration_after_grid_train():
    """Grid-trained model must support conformal prediction intervals."""
    from omega_pbpk.surrogate import PKSurrogate

    data = generate_grid_dataset(n_samples=60, seed=1, verbose=False)
    model = PKSurrogate(n_input=6)
    model.train(data.X[:40], data.y[:40], epochs=5, verbose=False)
    model.calibrate(data.X[40:], data.y[40:])
    y_pred, ci_lo, ci_hi = model.predict_conformal(data.X[0])
    assert np.all(ci_lo <= ci_hi), "Conformal CI lower > upper"
    assert y_pred.shape == (4,)
    assert ci_lo.shape == (4,)
    assert ci_hi.shape == (4,)
