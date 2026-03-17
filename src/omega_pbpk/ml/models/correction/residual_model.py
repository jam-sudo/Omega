"""Interpretable residual correction model for PK predictions.

Trains a Ridge regression on log(predicted/observed) residuals using
molecular features. At inference, predicts a correction factor:

    corrected_cmax = raw_cmax / exp(correction)

Features (6):
    0: logP (lipophilicity)
    1: log10(mw) (molecular size)
    2: fup (fraction unbound)
    3: log10(dose_mg) (dose)
    4: pgp_substrate (0/1 binary)
    5: log10(peff) (permeability)

Ridge regularization (alpha=1.0) prevents overfitting on ~100 training drugs.

NOTE (2026-03-17): This model is NOT loaded or used in OmegaPipeline.simulate().
Phase 0 ablation (NO_RIDGE) showed zero effect on benchmark results, confirming
the Ridge correction is dead code in the production pipeline. Retained for
potential future use and training/evaluation scripts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "logP",
    "log10_mw",
    "fup",
    "log10_dose_mg",
    "pgp_substrate",
    "log10_peff",
]


class ResidualCorrectionModel:
    def __init__(self, alpha: float = 1.0, max_correction: float = 1.5):
        self.alpha = alpha
        self.max_correction = max_correction  # max |log-correction|
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Ridge regression on features X and log-residuals y.

        Args:
            X: (n_drugs, n_features) feature matrix
            y: (n_drugs,) log(predicted/observed) residuals
        """
        n, p = X.shape

        # Standardize features
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ < 1e-8] = 1.0
        X_std = (X - self.feature_mean_) / self.feature_std_

        # Ridge closed-form: (X'X + αI)^-1 X'y
        XtX = X_std.T @ X_std + self.alpha * np.eye(p)
        Xty = X_std.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)
        self.intercept_ = float(y.mean() - X_std.mean(axis=0) @ self.coef_)

        train_rmse = np.sqrt(np.mean((self.predict(X) - y) ** 2))
        logger.info(
            "Correction model fitted: %d drugs, %d features, train RMSE=%.3f",
            n,
            p,
            train_rmse,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log-correction for feature matrix X."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        X_std = (X - self.feature_mean_) / self.feature_std_
        raw = X_std @ self.coef_ + self.intercept_
        return np.clip(raw, -self.max_correction, self.max_correction)

    def leave_one_out_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leave-one-out cross-validation. Returns corrected residuals."""
        n = X.shape[0]
        loo_residuals = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = ResidualCorrectionModel(alpha=self.alpha, max_correction=self.max_correction)
            model.fit(X[mask], y[mask])
            correction = model.predict(X[i : i + 1])[0]
            loo_residuals[i] = y[i] - correction
        return loo_residuals

    def save(self, path: Path) -> None:
        """Save model coefficients to JSON."""
        data = {
            "alpha": self.alpha,
            "max_correction": self.max_correction,
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "feature_mean": self.feature_mean_.tolist(),
            "feature_std": self.feature_std_.tolist(),
            "feature_names": FEATURE_NAMES,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ResidualCorrectionModel:
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)
        model = cls(alpha=data["alpha"], max_correction=data["max_correction"])
        model.coef_ = np.array(data["coef"])
        model.intercept_ = data["intercept"]
        model.feature_mean_ = np.array(data["feature_mean"])
        model.feature_std_ = np.array(data["feature_std"])
        return model
