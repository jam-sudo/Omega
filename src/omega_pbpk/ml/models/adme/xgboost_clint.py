"""XGBoost-based predictor for hepatocyte intrinsic clearance (CLint).

Predicts log10(CLint_hep) in µL/min/10^6 cells using Morgan fingerprints
+ RDKit descriptors. Training data: TDC Clearance_Hepatocyte_AZ (1,213 compounds).

This serves as a fallback when ADMET-AI is not available, ensuring the pipeline
always has hepatocyte CLint for proper IVIVE (rather than falling back to the
unreliable CYP-attributed CLint path).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "models" / "xgboost_clint"
_MODEL_PATH = _MODEL_DIR / "model.json"
_META_PATH = _MODEL_DIR / "meta.json"


def _load_tdc_clearance() -> tuple[list[str], list[float]]:
    """Load Clearance_Hepatocyte_AZ from TDC.

    Returns:
        (smiles_list, clint_list) where clint is in µL/min/10^6 cells.
    """
    try:
        from tdc.single_pred import ADME

        dataset = ADME(name="Clearance_Hepatocyte_AZ")
        df = dataset.get_data()

        smiles_list = []
        clint_list = []
        for _, row in df.iterrows():
            smi = str(row["Drug"])
            clint = float(row["Y"])
            if clint > 0:
                smiles_list.append(smi)
                clint_list.append(clint)

        logger.info("Loaded %d compounds from TDC Clearance_Hepatocyte_AZ", len(smiles_list))
        return smiles_list, clint_list
    except (ImportError, Exception) as exc:
        logger.warning("TDC Clearance_Hepatocyte_AZ not available: %s", exc)
        return [], []


class XGBoostCLintPredictor:
    """XGBoost predictor for hepatocyte intrinsic clearance.

    Predicts log10(CLint) in µL/min/10^6 cells.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        auto_load: bool = True,
    ) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise ImportError("XGBoost required. pip install xgboost") from None
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise ImportError("RDKit required. pip install rdkit-pypi") from None

        self._model_dir = model_dir or _MODEL_DIR
        self._model_path = self._model_dir / "model.json"
        self._meta_path = self._model_dir / "meta.json"
        self._model: Any = None
        self._cv_metrics: dict[str, float] = {}

        if auto_load:
            if self._model_path.exists():
                self._load_model()
            else:
                self.train()

    def train(self, n_folds: int = 5) -> dict[str, float]:
        """Train on TDC Clearance_Hepatocyte_AZ."""
        import xgboost as xgb

        from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

        tdc_smiles, tdc_clint = _load_tdc_clearance()
        if not tdc_smiles:
            raise ValueError("No training data for CLint predictor.")

        # Compute features
        valid_X = []
        valid_y = []
        for smi, clint in zip(tdc_smiles, tdc_clint):
            feat = _smiles_to_features(smi)
            if feat is not None:
                valid_X.append(feat)
                valid_y.append(np.log10(max(clint, 1.0)))

        X = np.array(valid_X)
        y = np.array(valid_y, dtype=np.float64)
        n = len(y)

        logger.info("Training XGBoost CLint on %d compounds (log10 space)", n)

        # Cross-validation
        indices = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        fold_size = n // n_folds

        all_preds = np.zeros(n)
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.6,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx])
            all_preds[val_idx] = model.predict(X[val_idx])

        # CV metrics
        residuals = all_preds - y
        mae_log = float(np.mean(np.abs(residuals)))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        self._cv_metrics = {
            "mae_log10": round(mae_log, 4),
            "r2": round(r2, 4),
            "n_compounds": n,
        }
        logger.info("XGBoost CLint CV: MAE(log10)=%.4f, R²=%.4f (n=%d)", mae_log, r2, n)

        # Train final model
        self._model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y)
        self._save_model()
        return self._cv_metrics

    def predict_clint(self, smiles: str) -> float:
        """Predict hepatocyte CLint (µL/min/10^6 cells).

        Args:
            smiles: SMILES string.

        Returns:
            Predicted CLint, clipped to [1.0, 200.0].
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

        feat = _smiles_to_features(smiles)
        if feat is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        log_clint = float(self._model.predict(feat.reshape(1, -1))[0])
        return float(np.clip(10.0**log_clint, 1.0, 200.0))

    def _save_model(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self._model_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._cv_metrics, f, indent=2)
        logger.info("XGBoost CLint model saved to %s", self._model_path)

    def _load_model(self) -> None:
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._cv_metrics = json.load(f)
        logger.info("XGBoost CLint model loaded from %s", self._model_path)

    @property
    def cv_metrics(self) -> dict[str, float]:
        return dict(self._cv_metrics)
