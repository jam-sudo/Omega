"""Tests for interpretable residual correction model."""

import tempfile
from pathlib import Path

import numpy as np


class TestResidualCorrection:
    def test_correction_reduces_residual_variance(self):
        """Correction model should reduce residual variance on training set."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        n = 20
        features = np.random.randn(n, 6)
        # True log-residual has linear relationship with feature 0
        log_residuals = 0.5 * features[:, 0] + 0.1 * np.random.randn(n)

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)
        corrections = model.predict(features)

        corrected = log_residuals - corrections
        assert np.std(corrected) < np.std(log_residuals)

    def test_correction_factor_is_bounded(self):
        """Correction factors should be bounded (no extreme adjustments)."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        features = np.random.randn(10, 6)
        log_residuals = np.random.randn(10) * 0.5

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)
        corrections = model.predict(features)

        # Capped at max_correction=1.5 by default
        assert np.all(np.abs(corrections) <= 1.5 + 1e-10)

    def test_loo_cv(self):
        """Leave-one-out CV should not be dramatically worse than train."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        n = 30
        features = np.random.randn(n, 6)
        log_residuals = 0.3 * features[:, 0] - 0.2 * features[:, 1] + 0.05 * np.random.randn(n)

        model = ResidualCorrectionModel()
        loo_residuals = model.leave_one_out_cv(features, log_residuals)

        # LOO residual std should be less than 2x original std
        assert np.std(loo_residuals) < 2.0 * np.std(log_residuals)

    def test_save_load_roundtrip(self):
        """Model should serialize and deserialize correctly."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        features = np.random.randn(15, 6)
        log_residuals = np.random.randn(15) * 0.3

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.json"
            model.save(path)
            loaded = ResidualCorrectionModel.load(path)
            np.testing.assert_allclose(
                model.predict(features),
                loaded.predict(features),
                atol=1e-10,
            )

    def test_unfitted_raises(self):
        """Predict on unfitted model should raise."""
        import pytest

        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        model = ResidualCorrectionModel()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(5, 6))
