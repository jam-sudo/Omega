"""Post-ODE Residual Corrector — learns residual corrections to ODE Cmax/AUC.

The corrector predicts additive corrections in log10 space:
    Cmax_corrected = Cmax_ODE * 10^correction_log
    AUC_corrected  = AUC_ODE  * 10^correction_log

These corrections are learned from the residual between ODE predictions and
observed clinical values, using molecular features + ODE output as input.

Architecture:
    Stacking(Ridge(alpha=10), XGBoost(depth=3, min_child=10))
    Input: 28 features (molecular + ADME + ODE output + physics)
    Target: log10(Cmax_obs / Cmax_ODE)

Safety: corrections are clamped to [-1.0, 1.0] log units (max 10x).
If |raw_correction| > 1.0, the correction is forced to 0.0 (ODE fallback).

Feature vector (28 dims):
    - 9 molecular descriptors (logP, TPSA, MW, HBA, HBD, RotBonds, Rings, FrCSP3, MolMR)
    - 5 ADME predicted (fup, CLint, rbp, peff, logS)
    - 4 pKa + compound type (pka_val, is_acid, is_base, is_neutral)
    - 4 ODE output (log10(Cmax_ODE), log10(AUC_ODE), tmax, t_half)
    - 2 Fg/Fh estimates
    - 4 extra (log10(dose), MW/dose ratio, TPSA_raw, ring_count)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Model directory
_REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = _REPO_ROOT / "models" / "corrections" / "post_ode"

# Feature dimensions
N_MOLECULAR = 9
N_ADME = 5
N_PKA = 4
N_ODE = 4
N_PHYSICS = 2
N_EXTRA = 4
N_FEATURES = N_MOLECULAR + N_ADME + N_PKA + N_ODE + N_PHYSICS + N_EXTRA  # 28

# Correction clamp bounds (log10 units)
MAX_CORRECTION = 1.0  # max 10x multiplicative correction

# Output target names
TARGET_NAMES = ["cmax_correction_log", "auc_correction_log"]


class PostODECorrector:
    """Post-ODE residual correction model.

    Predicts additive corrections (in log10 space) to ODE-predicted Cmax and AUC.
    Uses a stacking ensemble of Ridge regression and XGBoost, averaging their
    predictions.

    When not trained (model files don't exist), returns zero corrections
    for graceful fallback — the pipeline runs identically to baseline.

    Usage:
        corrector = PostODECorrector()
        if corrector.is_trained:
            correction = corrector.predict(smiles, ode_cmax, ode_auc, adme_pred)
            cmax_corrected = ode_cmax * 10**correction["cmax_correction_log"]
            auc_corrected = ode_auc * 10**correction["auc_correction_log"]
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or MODEL_DIR
        self._ridge_models: dict[str, Any] | None = None  # target_name -> Ridge
        self._xgb_models: dict[str, Any] | None = None  # target_name -> XGBRegressor
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._loaded = False

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load trained models from disk."""
        meta_path = self._model_dir / "post_ode_meta.json"

        if not meta_path.exists():
            logger.debug(
                "PostODECorrector: model not found at %s — returning zeros.",
                meta_path,
            )
            return

        try:
            import json

            import joblib
            import xgboost as xgb

            with open(meta_path) as f:
                json.load(f)  # validate metadata is readable

            self._ridge_models = {}
            self._xgb_models = {}

            for target in TARGET_NAMES:
                # Load Ridge model
                ridge_path = self._model_dir / f"ridge_{target}.joblib"
                if ridge_path.exists():
                    self._ridge_models[target] = joblib.load(ridge_path)
                else:
                    logger.warning("PostODECorrector: missing Ridge model %s", ridge_path)
                    self._ridge_models = None
                    self._xgb_models = None
                    return

                # Load XGBoost model
                xgb_path = self._model_dir / f"xgb_{target}.json"
                if xgb_path.exists():
                    model = xgb.XGBRegressor()
                    model.load_model(str(xgb_path))
                    self._xgb_models[target] = model
                else:
                    logger.warning("PostODECorrector: missing XGBoost model %s", xgb_path)
                    self._ridge_models = None
                    self._xgb_models = None
                    return

            # Load feature normalization stats
            means_path = self._model_dir / "feature_means.npy"
            stds_path = self._model_dir / "feature_stds.npy"
            if means_path.exists() and stds_path.exists():
                self._feature_means = np.load(means_path)
                self._feature_stds = np.load(stds_path)

            self._loaded = True
            logger.info(
                "PostODECorrector: loaded %d Ridge + %d XGBoost models from %s",
                len(self._ridge_models),
                len(self._xgb_models),
                self._model_dir,
            )
        except Exception as exc:
            logger.warning("PostODECorrector: failed to load model: %s", exc)
            self._ridge_models = None
            self._xgb_models = None
            self._loaded = False

    @property
    def is_trained(self) -> bool:
        """Whether the corrector has trained models loaded."""
        return self._loaded and self._ridge_models is not None and self._xgb_models is not None

    def predict(
        self,
        smiles: str,
        ode_cmax: float,
        ode_auc: float,
        adme_pred: dict[str, Any],
    ) -> dict[str, float]:
        """Predict residual corrections for ODE Cmax and AUC.

        Args:
            smiles: SMILES string of the molecule.
            ode_cmax: ODE-predicted Cmax (mg/L).
            ode_auc: ODE-predicted AUC0-t (mg*h/L).
            adme_pred: Dict with ADME predictions. Expected keys:
                fup, clint, logP, peff, rbp, logS, pka, compound_type,
                fg, fh, f_oral, extraction_ratio, dose_mg, tmax, t_half.
                Missing keys use safe defaults.

        Returns:
            Dict with keys: cmax_correction_log, auc_correction_log.
            Values are clamped to [-1.0, 1.0] log10 units.
            If |raw| > 1.0, forced to 0.0 (ODE fallback).
            Returns all zeros if not trained or SMILES is invalid.
        """
        zeros = {name: 0.0 for name in TARGET_NAMES}

        if not self.is_trained:
            return zeros

        features = self._extract_features(smiles, ode_cmax, ode_auc, adme_pred)
        if features is None:
            return zeros

        result = {}
        X = features.reshape(1, -1)

        # Normalize features for Ridge (XGBoost doesn't need it but it doesn't hurt)
        X_norm = X.copy()
        if self._feature_means is not None and self._feature_stds is not None:
            X_norm = (X - self._feature_means) / np.maximum(self._feature_stds, 1e-8)

        for target in TARGET_NAMES:
            # Stacking: average of Ridge and XGBoost
            ridge_pred = float(self._ridge_models[target].predict(X_norm)[0])
            xgb_pred = float(self._xgb_models[target].predict(X)[0])
            raw_correction = (ridge_pred + xgb_pred) / 2.0

            # Safety: if |correction| > 1.0 log units (10x), force to 0
            if abs(raw_correction) > MAX_CORRECTION:
                result[target] = 0.0
            else:
                result[target] = float(np.clip(raw_correction, -MAX_CORRECTION, MAX_CORRECTION))

        return result

    def _extract_features(
        self,
        smiles: str,
        ode_cmax: float,
        ode_auc: float,
        adme_pred: dict[str, Any],
    ) -> np.ndarray | None:
        """Extract 28-dimensional feature vector.

        Features:
            [0-8]   9 molecular descriptors (logP, TPSA/200, MW/600, HBA/10,
                     HBD/5, RotBonds/15, Rings/5, FrCSP3, MolMR/150)
            [9-13]  5 ADME predicted (fup, log10(clint+1), rbp, peff, logS)
            [14-17] 4 pKa + compound type (pka/14, is_acid, is_base, is_neutral)
            [18-21] 4 ODE output (log10(cmax+1e-6), log10(auc+1e-6), tmax/24, t_half/24)
            [22-23] 2 physics (fg, fh)
            [24-27] 4 extra (log10(dose+1), MW/dose, TPSA_raw/200, rings/5)

        Args:
            smiles: SMILES string.
            ode_cmax: ODE-predicted Cmax (mg/L).
            ode_auc: ODE-predicted AUC (mg*h/L).
            adme_pred: Dict with ADME predictions (missing keys use defaults).

        Returns:
            np.ndarray of shape (28,), or None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except ImportError:
            logger.warning("RDKit not available for feature extraction")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Invalid SMILES for PostODECorrector: %s", smiles)
            return None

        # --- 9 molecular descriptors (normalized) ---
        mol_logP = Descriptors.MolLogP(mol)
        tpsa_raw = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        fr_csp3 = Descriptors.FractionCSP3(mol)
        mol_mr = Descriptors.MolMR(mol)

        molecular = np.array(
            [
                mol_logP,
                tpsa_raw / 200.0,
                mw / 600.0,
                hba / 10.0,
                hbd / 5.0,
                rot_bonds / 15.0,
                ring_count / 5.0,
                fr_csp3,
                mol_mr / 150.0,
            ],
            dtype=np.float32,
        )

        # --- 5 ADME predicted ---
        fup = float(adme_pred.get("fup", 0.1))
        clint = float(adme_pred.get("clint", 5.0))
        rbp = float(adme_pred.get("rbp", 0.55))
        peff = float(adme_pred.get("peff", 1.0))
        logS = float(adme_pred.get("logS", -3.0))

        adme = np.array(
            [
                fup,
                np.log10(max(clint, 1e-6) + 1.0),
                rbp,
                peff,
                logS,
            ],
            dtype=np.float32,
        )

        # --- 4 pKa + compound type ---
        pka_val = float(adme_pred.get("pka", 7.0))
        compound_type = str(adme_pred.get("compound_type", "neutral"))
        is_acid = 1.0 if compound_type == "acid" else 0.0
        is_base = 1.0 if compound_type == "base" else 0.0
        is_neutral = 1.0 if compound_type in ("neutral", "zwitterion") else 0.0

        pka_features = np.array(
            [pka_val / 14.0, is_acid, is_base, is_neutral],
            dtype=np.float32,
        )

        # --- 4 ODE output ---
        # Safe log transform with floor to prevent log(0)
        log_cmax = np.log10(max(ode_cmax, 1e-6))
        log_auc = np.log10(max(ode_auc, 1e-6))
        tmax = float(adme_pred.get("tmax", 2.0))
        t_half = float(adme_pred.get("t_half", 8.0))

        ode_features = np.array(
            [
                log_cmax,
                log_auc,
                tmax / 24.0,
                t_half / 24.0,
            ],
            dtype=np.float32,
        )

        # --- 2 physics (Fg, Fh) ---
        fg = float(adme_pred.get("fg", 1.0))
        fh = float(adme_pred.get("fh", 0.7))

        physics = np.array([fg, fh], dtype=np.float32)

        # --- 4 extra features ---
        dose_mg = float(adme_pred.get("dose_mg", 100.0))
        log_dose = np.log10(max(dose_mg, 1.0))
        mw_dose_ratio = mw / max(dose_mg, 1.0)

        extra = np.array(
            [
                log_dose,
                mw_dose_ratio / 10.0,  # normalize
                tpsa_raw / 200.0,
                ring_count / 5.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([molecular, adme, pka_features, ode_features, physics, extra])
