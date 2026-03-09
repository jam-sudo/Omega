"""XGBoost-based predictor for fraction unbound in plasma (fup).

Predicts log10(fup) using Morgan fingerprints + RDKit physicochemical descriptors.
Training data: TDC PPBR_AZ (1,614 compounds) + reference CSV (153 compounds).

Key design: predicting in log-space gives equal weight to highly-bound (fup<0.01)
and moderately-bound drugs, unlike PPBR percentage space where 99% and 99.9%
look similar but the fup differs 10-fold (0.01 vs 0.001).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Paths relative to repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
_DATA_PATH = _REPO_ROOT / "data" / "adme_reference.csv"
_MODEL_DIR = _REPO_ROOT / "models" / "xgboost_fup"
_MODEL_PATH = _MODEL_DIR / "model.json"
_META_PATH = _MODEL_DIR / "meta.json"

# Feature parameters
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
N_RDKIT_DESCRIPTORS = 9  # LogP, TPSA, MW, HBA, HBD, RotBonds, RingCount, FractionCSP3, MolMR


def _smiles_to_features(smiles: str) -> np.ndarray | None:
    """Convert SMILES to feature vector: Morgan FP + RDKit descriptors.

    Returns:
        Array of shape (FINGERPRINT_BITS + N_RDKIT_DESCRIPTORS,), or None if invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
    fp_arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    for idx in fp.GetOnBits():
        fp_arr[idx] = 1.0

    # Physicochemical descriptors (normalized)
    desc = np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 200.0,  # normalize ~0-1
            Descriptors.MolWt(mol) / 600.0,  # normalize ~0-1
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,  # normalize ~0-1
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr, desc])


def _load_tdc_ppbr() -> tuple[list[str], list[float]]:
    """Load PPBR_AZ from TDC and convert to fup values.

    Returns:
        (smiles_list, fup_list) where fup = 1 - PPBR/100.
    """
    try:
        from tdc.single_pred import ADME

        dataset = ADME(name="PPBR_AZ")
        df = dataset.get_data()

        smiles_list = []
        fup_list = []
        for _, row in df.iterrows():
            smi = str(row["Drug"])
            ppbr = float(row["Y"])
            fup = 1.0 - ppbr / 100.0
            fup = max(0.001, min(1.0, fup))
            smiles_list.append(smi)
            fup_list.append(fup)

        logger.info("Loaded %d compounds from TDC PPBR_AZ", len(smiles_list))
        return smiles_list, fup_list
    except (ImportError, Exception) as exc:
        logger.warning("TDC PPBR_AZ not available: %s", exc)
        return [], []


def _load_reference_fup(data_path: Path | None = None) -> tuple[list[str], list[float]]:
    """Load fup values from reference CSV.

    Returns:
        (smiles_list, fup_list) from adme_reference.csv.
    """
    from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

    path = data_path or _DATA_PATH
    if not path.exists():
        return [], []

    smiles_list = []
    fup_list = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            fup_str = row.get("fup")
            if fup_str is None:
                continue

            smiles = row.get("smiles", row.get("SMILES", ""))
            if not smiles:
                smiles = _name_to_smiles(name)
            if not smiles:
                continue

            fup = float(fup_str)
            fup = max(0.001, min(1.0, fup))
            smiles_list.append(smiles)
            fup_list.append(fup)

    logger.info("Loaded %d compounds from reference CSV", len(smiles_list))
    return smiles_list, fup_list


class XGBoostFupPredictor:
    """XGBoost predictor for fraction unbound in plasma (fup).

    Predicts log10(fup) using Morgan fingerprints + physicochemical descriptors.
    Training data: TDC PPBR_AZ (1,614 compounds) + reference CSV (153 compounds).
    """

    def __init__(
        self,
        data_path: Path | None = None,
        model_dir: Path | None = None,
        auto_load: bool = True,
    ) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise ImportError(
                "XGBoost required for XGBoostFupPredictor. pip install xgboost"
            ) from None
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise ImportError("RDKit required for fingerprints. pip install rdkit-pypi") from None

        self._data_path = data_path or _DATA_PATH
        self._model_dir = model_dir or _MODEL_DIR
        self._model_path = self._model_dir / "model.json"
        self._meta_path = self._model_dir / "meta.json"
        self._model: Any = None
        self._cv_metrics: dict[str, float] = {}
        self._cv_residuals: np.ndarray | None = None  # for conformal intervals

        if auto_load:
            if self._model_path.exists():
                self._load_model()
            else:
                self.train()

    def train(self, n_folds: int = 5) -> dict[str, float]:
        """Train XGBoost on TDC + reference data with cross-validation.

        Returns:
            Dict with CV metrics: mae_log, mae_fup, r2, n_compounds.
        """
        import xgboost as xgb

        # Load data from both sources
        tdc_smiles, tdc_fup = _load_tdc_ppbr()
        ref_smiles, ref_fup = _load_reference_fup(self._data_path)

        all_smiles = tdc_smiles + ref_smiles
        all_fup = tdc_fup + ref_fup

        if not all_smiles:
            raise ValueError("No training data available for fup predictor.")

        # Compute features and filter invalid SMILES
        valid_X = []
        valid_y = []
        for smi, fup in zip(all_smiles, all_fup):
            feat = _smiles_to_features(smi)
            if feat is not None:
                valid_X.append(feat)
                valid_y.append(np.log10(max(fup, 0.001)))

        X = np.array(valid_X)
        y = np.array(valid_y, dtype=np.float64)
        n = len(y)

        logger.info("Training XGBoost fup on %d compounds (log10 space)", n)

        # Cross-validation
        if n < n_folds:
            n_folds = max(2, n)

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

        # CV metrics in log space
        residuals = all_preds - y
        mae_log = float(np.mean(np.abs(residuals)))

        # CV metrics in linear space
        pred_fup = np.clip(10.0**all_preds, 0.001, 1.0)
        true_fup = np.clip(10.0**y, 0.001, 1.0)
        fold_errors = np.abs(np.log10(pred_fup / true_fup))
        mae_fup = float(np.mean(fold_errors))

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        # Store residuals for conformal intervals
        self._cv_residuals = np.sort(np.abs(residuals))

        self._cv_metrics = {
            "mae_log10": round(mae_log, 4),
            "mae_fold_error": round(mae_fup, 4),
            "r2": round(r2, 4),
            "n_compounds": n,
            "median_abs_residual": round(float(np.median(np.abs(residuals))), 4),
            "p90_abs_residual": round(float(np.percentile(np.abs(residuals), 90)), 4),
        }
        logger.info(
            "XGBoost fup CV: MAE(log10)=%.4f, MAE(fold)=%.4f, R²=%.4f (n=%d)",
            mae_log,
            mae_fup,
            r2,
            n,
        )

        # Train final model on all data
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

    def predict_fup(self, smiles: str) -> float:
        """Predict fup for a single compound.

        Args:
            smiles: SMILES string.

        Returns:
            Predicted fup, clipped to [0.001, 1.0].
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feat = _smiles_to_features(smiles)
        if feat is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        log_fup = float(self._model.predict(feat.reshape(1, -1))[0])
        return float(np.clip(10.0**log_fup, 0.001, 1.0))

    def predict_fup_with_interval(
        self, smiles: str, confidence: float = 0.9
    ) -> tuple[float, float, float]:
        """Predict fup with conformal prediction interval.

        Args:
            smiles: SMILES string.
            confidence: Confidence level (default 0.9 = 90%).

        Returns:
            (fup, fup_lo, fup_hi) tuple.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feat = _smiles_to_features(smiles)
        if feat is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        log_fup = float(self._model.predict(feat.reshape(1, -1))[0])
        fup = float(np.clip(10.0**log_fup, 0.001, 1.0))

        # Conformal interval from CV residuals
        if self._cv_residuals is not None and len(self._cv_residuals) > 0:
            idx = int(confidence * len(self._cv_residuals))
            idx = min(idx, len(self._cv_residuals) - 1)
            q = float(self._cv_residuals[idx])
        else:
            q = 0.5  # default half-decade

        fup_lo = float(np.clip(10.0 ** (log_fup - q), 0.001, 1.0))
        fup_hi = float(np.clip(10.0 ** (log_fup + q), 0.001, 1.0))

        return fup, fup_lo, fup_hi

    def _save_model(self) -> None:
        """Save model, metadata, and CV residuals."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self._model_path))

        meta = dict(self._cv_metrics)
        if self._cv_residuals is not None:
            meta["cv_residuals"] = self._cv_residuals.tolist()

        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("XGBoost fup model saved to %s", self._model_path)

    def _load_model(self) -> None:
        """Load model and metadata from disk."""
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))

        if self._meta_path.exists():
            with open(self._meta_path) as f:
                meta = json.load(f)
            residuals = meta.pop("cv_residuals", None)
            self._cv_metrics = meta
            if residuals is not None:
                self._cv_residuals = np.array(residuals)

        logger.info("XGBoost fup model loaded from %s", self._model_path)

    @property
    def cv_metrics(self) -> dict[str, float]:
        """Return cross-validation metrics."""
        return dict(self._cv_metrics)
