"""Transporter substrate binary classifiers (XGBoost).

Predicts probabilities that a given molecule is a substrate for 6 major
drug transporters: P-gp, OATP1B1, BCRP, OCT2, OAT1/3, and PepT1.

Architecture:
    SMILES -> Morgan FP (2048 bits, radius 2) + 9 RDKit descriptors = 2057 features
           -> 6 x XGBoost binary classifier
           -> {pgp: 0.82, oatp1b1: 0.15, bcrp: 0.43, oct2: 0.08, oat: 0.31, pept1: 0.02}

When models are not trained (model files don't exist), returns 0.5 for all
transporters (no-information prior) for graceful fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Model directory
_REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = _REPO_ROOT / "models" / "corrections" / "transporters"

# Feature extraction constants
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
N_DESCRIPTORS = 9
N_FEATURES = FINGERPRINT_BITS + N_DESCRIPTORS  # 2057

# Transporter names (model file names derive from these)
TRANSPORTER_NAMES = ["pgp", "oatp1b1", "bcrp", "oct2", "oat", "pept1"]

# Default probability when untrained (no-information prior)
DEFAULT_PROB = 0.5


class TransporterClassifier:
    """Binary XGBoost classifiers for 6 drug transporters.

    Predicts the probability that a molecule is a substrate for each of
    P-gp, OATP1B1, BCRP, OCT2, OAT1/3, and PepT1.

    When not trained (model files don't exist), returns 0.5 for all
    transporters — a non-informative prior that leaves downstream
    pipeline decisions unchanged.

    Usage:
        clf = TransporterClassifier()
        probs = clf.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        # probs = {'pgp': 0.12, 'oatp1b1': 0.05, 'bcrp': 0.08, ...}
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or MODEL_DIR
        self._models: dict[str, Any] = {}  # transporter_name -> XGBClassifier
        self._loaded = False

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load trained models from disk."""
        if not self._model_dir.exists():
            logger.debug(
                "TransporterClassifier: model dir not found at %s — returning defaults.",
                self._model_dir,
            )
            return

        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("TransporterClassifier: xgboost not available")
            return

        loaded = {}
        for name in TRANSPORTER_NAMES:
            model_path = self._model_dir / f"{name}_classifier.json"
            if not model_path.exists():
                logger.debug(
                    "TransporterClassifier: model not found for %s at %s",
                    name,
                    model_path,
                )
                continue
            try:
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                loaded[name] = model
            except Exception as exc:
                logger.warning(
                    "TransporterClassifier: failed to load %s model: %s",
                    name,
                    exc,
                )

        if loaded:
            self._models = loaded
            self._loaded = True
            logger.info(
                "TransporterClassifier: loaded %d/%d models from %s",
                len(loaded),
                len(TRANSPORTER_NAMES),
                self._model_dir,
            )
        else:
            logger.debug("TransporterClassifier: no models found in %s", self._model_dir)

    @property
    def is_trained(self) -> bool:
        """Whether any trained models are loaded."""
        return self._loaded and len(self._models) > 0

    def predict(self, smiles: str) -> dict[str, float]:
        """Predict transporter substrate probabilities for a molecule.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict mapping transporter name to substrate probability [0, 1].
            Returns 0.5 for all transporters if not trained or SMILES is invalid.
        """
        defaults = {name: DEFAULT_PROB for name in TRANSPORTER_NAMES}

        features = self._extract_features(smiles)
        if features is None:
            return defaults

        if not self.is_trained:
            return defaults

        X = features.reshape(1, -1)
        result = {}
        for name in TRANSPORTER_NAMES:
            if name in self._models:
                try:
                    prob = float(self._models[name].predict_proba(X)[0, 1])
                    result[name] = prob
                except Exception as exc:
                    logger.warning(
                        "TransporterClassifier: prediction failed for %s: %s",
                        name,
                        exc,
                    )
                    result[name] = DEFAULT_PROB
            else:
                result[name] = DEFAULT_PROB

        return result

    def _extract_features(self, smiles: str) -> np.ndarray | None:
        """Extract 2057-dimensional feature vector from SMILES.

        Features:
            - 2048 Morgan fingerprint bits (radius=2)
            - 9 physicochemical descriptors (normalized)

        Args:
            smiles: SMILES string.

        Returns:
            np.ndarray of shape (2057,), or None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
        except ImportError:
            logger.warning("RDKit not available for feature extraction")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Invalid SMILES for TransporterClassifier: %s", smiles)
            return None

        # Morgan fingerprint (2048 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
        fp_arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
        for idx in fp.GetOnBits():
            fp_arr[idx] = 1.0

        # 9 physicochemical descriptors (normalized consistently with pre_ode_corrector)
        descs = np.array(
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

        return np.concatenate([fp_arr, descs])
