"""XGBoost-based direct Cmax predictor.

Predicts log(Cmax/dose_mg) from Morgan fingerprints + RDKit descriptors,
then converts back via exp(pred) * dose_mg to get Cmax in ug/mL.

Training data: clinical Cmax observations from benchmark datasets (~66 drugs).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Paths
_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_PATH = _REPO_ROOT / "models" / "direct_pk" / "xgboost_cmax.json"
_META_PATH = _REPO_ROOT / "models" / "direct_pk" / "meta.json"

# Feature parameters
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
N_RDKIT_DESCRIPTORS = 9


def smiles_to_features(smiles: str) -> np.ndarray | None:
    """Convert SMILES to feature vector: Morgan FP (2048) + RDKit descriptors (9).

    Returns:
        Array of shape (2057,), or None if SMILES is invalid.
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
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr, desc])


class DirectCmaxPredictor:
    """XGBoost predictor for Cmax from SMILES + dose.

    Predicts log(Cmax/dose_mg) using molecular features, then converts
    back to Cmax via exp(pred) * dose_mg.
    """

    def __init__(
        self,
        model_path: Path | None = None,
    ) -> None:
        self._model_path = model_path or _MODEL_PATH
        self._model: Any = None
        self._meta: dict = {}

        if self._model_path.exists():
            self._load_model()
        else:
            logger.warning(
                "DirectCmaxPredictor model not found at %s — using fallback",
                self._model_path,
            )

    def _load_model(self) -> None:
        """Load XGBoost model from disk."""
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))

        meta_path = self._model_path.parent / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)

        logger.info("DirectCmaxPredictor loaded from %s", self._model_path)

    def predict(self, smiles: str, dose_mg: float) -> float:
        """Predict Cmax (ug/mL) for a single compound.

        Args:
            smiles: SMILES string.
            dose_mg: Dose in mg.

        Returns:
            Predicted Cmax in ug/mL.
            Falls back to dose_mg * 0.5 / 100.0 if model not loaded.
        """
        if self._model is None:
            return dose_mg * 0.5 / 100.0

        feat = smiles_to_features(smiles)
        if feat is None:
            logger.warning("Invalid SMILES: %s — using fallback", smiles)
            return dose_mg * 0.5 / 100.0

        log_cmax_per_mg = float(self._model.predict(feat.reshape(1, -1))[0])
        return math.exp(log_cmax_per_mg) * dose_mg

    def predict_batch(self, smiles_list: list[str], dose_mg: float) -> list[float]:
        """Predict Cmax for multiple compounds at the same dose.

        Args:
            smiles_list: List of SMILES strings.
            dose_mg: Dose in mg (same for all).

        Returns:
            List of predicted Cmax values.
        """
        return [self.predict(smi, dose_mg) for smi in smiles_list]

    @property
    def meta(self) -> dict:
        """Return model metadata."""
        return dict(self._meta)
