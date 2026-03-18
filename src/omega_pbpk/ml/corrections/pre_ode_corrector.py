"""Pre-ODE ADME Corrector — learns corrections to predicted ADME parameters.

The corrector predicts multiplicative corrections in log-space:
    fup_corrected   = fup_base   * 10^delta_log_fup
    CLint_corrected = CLint_base * 10^delta_log_clint
    peff_corrected  = peff_base  * 10^delta_log_peff

These corrections are optimized end-to-end to minimize Cmax prediction error,
explicitly managing the error cancellation problem: instead of accidental
cancellation between over/under-predictions, the ML model learns the optimal
ADME adjustments for each molecular scaffold.

Architecture:
    SMILES -> molecular_features(50) -> XGBoost -> (delta_log_fup, delta_log_clint, delta_log_peff)

Feature vector (50 dims):
    - 41 Morgan fingerprint bits (top-variance from training set, radius=2, 2048 bits)
    - 9 physicochemical descriptors (MolLogP, TPSA, MolWt, HBA, HBD, RotBonds, Rings, FracCSP3, MolMR)

Corrections are clamped to [-0.5, 0.5] log units (max ~3x correction factor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Model directory
_REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = _REPO_ROOT / "models" / "corrections" / "pre_ode"

# Feature extraction constants
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
N_FP_SELECTED = 41
N_DESCRIPTORS = 9
N_FEATURES = N_FP_SELECTED + N_DESCRIPTORS  # 50

# Correction clamp bounds (log10 units)
MAX_CORRECTION = 0.5  # max ~3.16x multiplicative correction

# Output target names
TARGET_NAMES = ["delta_log_fup", "delta_log_clint", "delta_log_peff"]


def _compute_fd_gradient(
    ode_cmax_fn: Any,
    fup: float,
    clint: float,
    epsilon: float = 0.05,
) -> tuple[float, float]:
    """Compute finite-difference gradient of Cmax w.r.t. log(fup) and log(clint).

    Uses central finite differences in log-space:
        dCmax/dlog(param) ~= (Cmax(param*10^eps) - Cmax(param*10^(-eps))) / (2*eps)

    Args:
        ode_cmax_fn: Callable(fup, clint) -> Cmax value.
        fup: Base fraction unbound in plasma (0-1).
        clint: Base intrinsic clearance (uL/min/pmol).
        epsilon: Perturbation size in log10 units. Default 0.05.

    Returns:
        Tuple of (grad_fup, grad_clint) — partial derivatives of Cmax
        with respect to log10(fup) and log10(clint).
    """
    scale_up = 10.0**epsilon
    scale_down = 10.0 ** (-epsilon)

    # Gradient w.r.t. log(fup)
    cmax_fup_up = ode_cmax_fn(fup * scale_up, clint)
    cmax_fup_down = ode_cmax_fn(fup * scale_down, clint)
    grad_fup = (cmax_fup_up - cmax_fup_down) / (2.0 * epsilon)

    # Gradient w.r.t. log(clint)
    cmax_clint_up = ode_cmax_fn(fup, clint * scale_up)
    cmax_clint_down = ode_cmax_fn(fup, clint * scale_down)
    grad_clint = (cmax_clint_up - cmax_clint_down) / (2.0 * epsilon)

    return float(grad_fup), float(grad_clint)


class PreODECorrector:
    """Pre-ODE ADME correction model.

    Predicts multiplicative corrections (in log10 space) to fup, CLint, and peff
    before they enter the ODE solver. The corrections are learned from end-to-end
    Cmax prediction error, explicitly managing error cancellation.

    When not trained (model file doesn't exist), returns zero corrections
    for graceful fallback — the pipeline runs identically to baseline.

    Usage:
        corrector = PreODECorrector()
        if corrector.is_trained:
            delta = corrector.predict(smiles)
            fup_corrected = fup_base * 10**delta["delta_log_fup"]
            clint_corrected = clint_base * 10**delta["delta_log_clint"]
            peff_corrected = peff_base * 10**delta["delta_log_peff"]
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or MODEL_DIR
        self._models: list[Any] | None = None  # list of 3 XGBRegressors
        self._fp_indices: np.ndarray | None = None
        self._loaded = False

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load trained model from disk."""
        model_path = self._model_dir / "pre_ode_xgb.json"
        indices_path = self._model_dir / "fp_indices.npy"

        if not model_path.exists():
            logger.debug(
                "PreODECorrector: model not found at %s — returning zeros.",
                model_path,
            )
            return

        try:
            import json

            import xgboost as xgb

            with open(model_path) as f:
                json.load(f)  # validate metadata is readable

            self._models = []
            for name in TARGET_NAMES:
                target_path = self._model_dir / f"pre_ode_xgb_{name}.json"
                if target_path.exists():
                    model = xgb.XGBRegressor()
                    model.load_model(str(target_path))
                    self._models.append(model)
                else:
                    logger.warning("PreODECorrector: missing model file %s", target_path)
                    self._models = None
                    return

            if indices_path.exists():
                self._fp_indices = np.load(indices_path)
            else:
                self._fp_indices = np.arange(N_FP_SELECTED)

            self._loaded = True
            logger.info(
                "PreODECorrector: loaded %d models, %d FP indices from %s",
                len(self._models),
                len(self._fp_indices),
                self._model_dir,
            )
        except Exception as exc:
            logger.warning("PreODECorrector: failed to load model: %s", exc)
            self._models = None
            self._loaded = False

    @property
    def is_trained(self) -> bool:
        """Whether the corrector has a trained model loaded."""
        return self._loaded and self._models is not None

    def predict(self, smiles: str) -> dict[str, float]:
        """Predict ADME corrections for a molecule.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict with keys: delta_log_fup, delta_log_clint, delta_log_peff.
            Values are clamped to [-0.5, 0.5] log10 units.
            Returns all zeros if not trained or SMILES is invalid.
        """
        zeros = {name: 0.0 for name in TARGET_NAMES}

        if not self.is_trained:
            return zeros

        features = self._extract_features(smiles)
        if features is None:
            return zeros

        result = {}
        X = features.reshape(1, -1)
        for i, name in enumerate(TARGET_NAMES):
            pred = float(self._models[i].predict(X)[0])
            # Clamp to [-0.5, 0.5] log units
            pred = max(-MAX_CORRECTION, min(MAX_CORRECTION, pred))
            result[name] = pred

        return result

    def _extract_features(self, smiles: str) -> np.ndarray | None:
        """Extract 50-dimensional feature vector from SMILES.

        Features:
            - 41 Morgan FP bits (selected by variance from training)
            - 9 physicochemical descriptors (normalized)

        Args:
            smiles: SMILES string.

        Returns:
            np.ndarray of shape (50,), or None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
        except ImportError:
            logger.warning("RDKit not available for feature extraction")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Invalid SMILES for PreODECorrector: %s", smiles)
            return None

        # Morgan fingerprint (2048 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
        fp_arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
        for idx in fp.GetOnBits():
            fp_arr[idx] = 1.0

        # 9 physicochemical descriptors (normalized)
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

        # Feature selection: use saved indices from training or fallback
        if self._fp_indices is not None:
            top_indices = self._fp_indices
        else:
            indices_path = self._model_dir / "fp_indices.npy"
            if indices_path.exists():
                top_indices = np.load(indices_path)
            else:
                top_indices = np.arange(N_FP_SELECTED)  # fallback: first 41 bits

        return np.concatenate([fp_arr[top_indices], descs])
