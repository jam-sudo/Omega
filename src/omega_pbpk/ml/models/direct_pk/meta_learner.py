"""CmaxMetaLearner: adaptive PBPK/ML blend trained on clinical data.

Replaces fixed-weight ensemble_cmax() with a learned combiner.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_PATH = _REPO_ROOT / "models" / "meta_learner" / "xgboost_meta.json"
_META_PATH = _REPO_ROOT / "models" / "meta_learner" / "meta.json"

FEATURE_NAMES = [
    "log_cmax_pbpk",
    "log_cmax_ml",
    "log_dose_mg",
    "log_cmax_ratio",
    "logP",
    "TPSA_norm",
    "MW_norm",
    "log_fup",
    "log_clint",
    "is_acid",
    "is_base",
    "pgp_flag",
]


@dataclass(frozen=True)
class MetaFeatures:
    """Features for the meta-learner combiner."""

    cmax_pbpk: float
    cmax_ml: float
    dose_mg: float
    logP: float
    TPSA_norm: float
    MW_norm: float
    fup: float
    clint: float
    is_acid: float
    is_base: float
    pgp_flag: float

    def to_array(self) -> np.ndarray:
        """Convert to 12-element feature array for XGBoost."""
        log_pbpk = np.log10(max(self.cmax_pbpk, 1e-12))
        log_ml = np.log10(max(self.cmax_ml, 1e-12))
        return np.array(
            [
                log_pbpk,
                log_ml,
                np.log10(max(self.dose_mg, 0.1)),
                log_pbpk - log_ml,
                self.logP,
                self.TPSA_norm,
                self.MW_norm,
                np.log10(max(self.fup, 1e-4)),
                np.log10(max(self.clint, 0.01)),
                self.is_acid,
                self.is_base,
                self.pgp_flag,
            ],
            dtype=np.float32,
        )


class CmaxMetaLearner:
    """Learned combiner that blends PBPK and ML Cmax predictions.

    When a trained model is available, uses 12 features to predict
    log10(Cmax) directly. Falls back to geometric mean of PBPK and ML
    predictions when no model is loaded.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or _MODEL_PATH
        self._model: Any = None
        self._meta: dict = {}
        if self._model_path.exists():
            self._load()

    def _load(self) -> None:
        """Load XGBoost model and metadata from disk."""
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))
        meta_path = self._model_path.parent / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        logger.info("CmaxMetaLearner loaded from %s", self._model_path)

    @property
    def is_loaded(self) -> bool:
        """Whether a trained model is available."""
        return self._model is not None

    def predict(self, features: MetaFeatures) -> float:
        """Predict Cmax (ug/mL) from meta-features.

        Args:
            features: MetaFeatures with PBPK prediction, ML prediction,
                      and physicochemical properties.

        Returns:
            Predicted Cmax in ug/mL. Falls back to geometric mean
            of cmax_pbpk and cmax_ml if no model is loaded.
        """
        if not self.is_loaded:
            return float(np.sqrt(max(features.cmax_pbpk, 1e-12) * max(features.cmax_ml, 1e-12)))
        x = features.to_array().reshape(1, -1)
        log_cmax = float(self._model.predict(x)[0])
        return 10.0**log_cmax

    @property
    def meta(self) -> dict:
        """Return model metadata."""
        return dict(self._meta)
