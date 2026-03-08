"""Ensemble combiner that merges predictions from multiple ADME models.

Per-property selection strategy:
  - logP, fup, clint_3a4, peff, logS, herg_ic50_uM -> ADMET-AI (primary), polynomial (fallback)
  - rbp -> XGBoost RBP (primary), polynomial (fallback)
  - clint_2d6 -> ADMET-AI categorical + heuristic
  - mw -> RDKit (always available from SMILES)

Confidence = min(backend confidences across all properties).
Conformal intervals: ADMET-AI intervals if available, else +/-50% default.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from omega_pbpk.ml.models.adme import MLADMEPredictor
from omega_pbpk.prediction.adme_predictor import ADMEProperties

logger = logging.getLogger(__name__)

# Confidence ordering for min-confidence calculation
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_CONFIDENCE_NAMES = {0: "low", 1: "medium", 2: "high"}


class EnsembleADMEPredictor(MLADMEPredictor):
    """Ensemble ADME predictor combining ADMET-AI, XGBoost RBP, and polynomial.

    Falls back gracefully: if a primary backend fails, uses the polynomial
    predictor as fallback for that property.
    """

    def __init__(
        self,
        admet_ai: Any | None = None,
        xgboost_rbp: Any | None = None,
        polynomial: Any | None = None,
    ) -> None:
        """Initialize the ensemble predictor.

        Args:
            admet_ai: ADMETAIPredictor instance (or None to auto-create).
            xgboost_rbp: XGBoostRBPPredictor instance (or None to auto-create).
            polynomial: ADMEPredictor (legacy) instance (or None to auto-create).
        """
        self._admet_ai = admet_ai
        self._xgboost_rbp = xgboost_rbp
        self._polynomial = polynomial
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize backends that weren't provided."""
        if self._initialized:
            return
        self._initialized = True

        # Try to create ADMET-AI predictor
        if self._admet_ai is None:
            try:
                from omega_pbpk.ml.models.adme.admet_ai_wrapper import ADMETAIPredictor

                self._admet_ai = ADMETAIPredictor()
                logger.info("EnsembleADME: ADMET-AI backend initialized.")
            except (ImportError, Exception) as exc:
                logger.warning("EnsembleADME: ADMET-AI not available: %s", exc)
                self._admet_ai = None

        # Try to create XGBoost RBP predictor
        if self._xgboost_rbp is None:
            try:
                from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

                self._xgboost_rbp = XGBoostRBPPredictor()
                logger.info("EnsembleADME: XGBoost RBP backend initialized.")
            except (ImportError, Exception) as exc:
                logger.warning("EnsembleADME: XGBoost RBP not available: %s", exc)
                self._xgboost_rbp = None

        # Always create polynomial fallback
        if self._polynomial is None:
            try:
                from omega_pbpk.prediction.adme_predictor import ADMEPredictor

                self._polynomial = ADMEPredictor()
                logger.info("EnsembleADME: Polynomial fallback initialized.")
            except Exception as exc:
                logger.warning("EnsembleADME: Polynomial fallback failed: %s", exc)
                self._polynomial = None

    def predict(self, smiles: str) -> ADMEProperties:
        """Predict ADME properties using ensemble strategy.

        Args:
            smiles: Canonical SMILES string.

        Returns:
            ADMEProperties with properties from best available backends.

        Raises:
            ValueError: If no backend can produce predictions.
        """
        self._ensure_initialized()

        # Track which backend was used per property and confidence levels
        sources: dict[str, str] = {}
        confidences: list[int] = []

        # -- Try ADMET-AI for primary properties --
        admet_result: ADMEProperties | None = None
        if self._admet_ai is not None:
            try:
                admet_result = self._admet_ai.predict(smiles)
                logger.debug("ADMET-AI prediction succeeded for %s", smiles[:30])
            except Exception as exc:
                logger.warning("ADMET-AI prediction failed: %s", exc)
                admet_result = None

        # -- Try polynomial fallback --
        poly_result: ADMEProperties | None = None
        if self._polynomial is not None:
            try:
                poly_result = self._polynomial.predict(smiles)
                logger.debug("Polynomial prediction succeeded for %s", smiles[:30])
            except Exception as exc:
                logger.warning("Polynomial prediction failed: %s", exc)
                poly_result = None

        # -- Try XGBoost RBP --
        xgb_rbp: float | None = None
        if self._xgboost_rbp is not None:
            try:
                xgb_rbp = self._xgboost_rbp.predict_rbp(smiles)
                logger.debug("XGBoost RBP prediction: %.3f", xgb_rbp)
            except Exception as exc:
                logger.warning("XGBoost RBP prediction failed: %s", exc)
                xgb_rbp = None

        # -- Compute MW from RDKit (always available) --
        mw = self._get_mw(smiles, admet_result, poly_result)
        sources["mw"] = "rdkit"

        # -- Property selection --
        # logP: ADMET-AI primary, polynomial fallback
        logp, logp_src = self._select_property(
            "logP", admet_result, poly_result
        )
        sources["logP"] = logp_src
        confidences.append(self._confidence_val(admet_result if logp_src == "admet-ai" else poly_result))

        # fup
        fup, fup_src = self._select_property("fup", admet_result, poly_result)
        sources["fup"] = fup_src
        confidences.append(self._confidence_val(admet_result if fup_src == "admet-ai" else poly_result))

        # clint_3a4
        clint_3a4, clint_src = self._select_property(
            "clint_3a4", admet_result, poly_result
        )
        sources["clint_3a4"] = clint_src
        confidences.append(self._confidence_val(admet_result if clint_src == "admet-ai" else poly_result))

        # peff
        peff, peff_src = self._select_property("peff", admet_result, poly_result)
        sources["peff"] = peff_src
        confidences.append(self._confidence_val(admet_result if peff_src == "admet-ai" else poly_result))

        # logS
        logs, logs_src = self._select_property("logS", admet_result, poly_result)
        sources["logS"] = logs_src
        confidences.append(self._confidence_val(admet_result if logs_src == "admet-ai" else poly_result))

        # herg_ic50_uM
        herg, herg_src = self._select_property(
            "herg_ic50_uM", admet_result, poly_result
        )
        sources["herg_ic50_uM"] = herg_src
        confidences.append(self._confidence_val(admet_result if herg_src == "admet-ai" else poly_result))

        # clint_2d6: ADMET-AI categorical, polynomial fallback
        clint_2d6, clint2d6_src = self._select_property(
            "clint_2d6", admet_result, poly_result
        )
        sources["clint_2d6"] = clint2d6_src

        # rbp: XGBoost primary, polynomial fallback
        if xgb_rbp is not None:
            rbp = xgb_rbp
            sources["rbp"] = "xgboost"
            confidences.append(1)  # medium confidence for XGBoost
        elif admet_result is not None:
            rbp = admet_result.rbp
            sources["rbp"] = "admet-ai"
            confidences.append(self._confidence_val(admet_result))
        elif poly_result is not None:
            rbp = poly_result.rbp
            sources["rbp"] = "polynomial"
            confidences.append(self._confidence_val(poly_result))
        else:
            rbp = 1.0
            sources["rbp"] = "default"
            confidences.append(0)

        # -- Confidence = min across all backends --
        overall_confidence = _CONFIDENCE_NAMES.get(
            min(confidences) if confidences else 0, "low"
        )

        # -- Conformal intervals --
        fup_lo, fup_hi = self._get_intervals("fup", admet_result, poly_result, fup, fup_src)
        clint_lo, clint_hi = self._get_intervals(
            "clint_3a4", admet_result, poly_result, clint_3a4, clint_src
        )
        peff_lo, peff_hi = self._get_intervals("peff", admet_result, poly_result, peff, peff_src)
        rbp_lo, rbp_hi = self._get_rbp_intervals(
            admet_result, poly_result, rbp, sources["rbp"]
        )

        logger.info(
            "Ensemble ADME sources: %s | confidence=%s",
            sources,
            overall_confidence,
        )

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(logp, 2),
            logS=round(logs, 2),
            peff=round(peff, 3),
            fup=round(max(0.001, min(1.0, fup)), 4),
            rbp=round(max(0.5, min(3.0, rbp)), 3),
            clint_3a4=round(max(0.01, clint_3a4), 3),
            clint_2d6=round(clint_2d6, 3),
            herg_ic50_uM=round(max(0.01, herg), 2),
            confidence=overall_confidence,
            fup_lo=round(max(0.0, fup_lo), 4),
            fup_hi=round(min(1.0, fup_hi), 4),
            clint_3a4_lo=round(max(0.0, clint_lo), 3),
            clint_3a4_hi=round(clint_hi, 3),
            peff_lo=round(max(0.0, peff_lo), 4),
            peff_hi=round(peff_hi, 4),
            rbp_lo=round(rbp_lo, 4),
            rbp_hi=round(rbp_hi, 4),
        )

    def predict_batch(self, smiles_list: list[str]) -> list[ADMEProperties]:
        """Predict ADME properties for multiple compounds.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of ADMEProperties, one per input SMILES.
        """
        return [self.predict(s) for s in smiles_list]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mw(
        smiles: str,
        admet_result: ADMEProperties | None,
        poly_result: ADMEProperties | None,
    ) -> float:
        """Get MW from RDKit, falling back to ADMET-AI or polynomial."""
        try:
            from omega_pbpk.ml.models.adme.admet_ai_wrapper import (
                _compute_mw_from_smiles,
            )

            return _compute_mw_from_smiles(smiles)
        except (ImportError, ValueError):
            pass
        if admet_result is not None:
            return admet_result.mw
        if poly_result is not None:
            return poly_result.mw
        return 300.0

    @staticmethod
    def _select_property(
        prop: str,
        admet_result: ADMEProperties | None,
        poly_result: ADMEProperties | None,
    ) -> tuple[float, str]:
        """Select a property value: ADMET-AI primary, polynomial fallback.

        Returns:
            (value, source_name) tuple.
        """
        if admet_result is not None:
            val = getattr(admet_result, prop, None)
            if val is not None:
                return float(val), "admet-ai"
        if poly_result is not None:
            val = getattr(poly_result, prop, None)
            if val is not None:
                return float(val), "polynomial"
        # Hard defaults
        defaults = {
            "logP": 2.0,
            "fup": 0.1,
            "clint_3a4": 1.0,
            "peff": 1.0,
            "logS": -3.0,
            "herg_ic50_uM": 10.0,
            "clint_2d6": 0.5,
            "rbp": 1.0,
        }
        return defaults.get(prop, 0.0), "default"

    @staticmethod
    def _confidence_val(result: ADMEProperties | None) -> int:
        """Map confidence string to integer for min-comparison."""
        if result is None:
            return 0
        return _CONFIDENCE_ORDER.get(result.confidence, 0)

    @staticmethod
    def _get_intervals(
        prop: str,
        admet_result: ADMEProperties | None,
        poly_result: ADMEProperties | None,
        point_est: float,
        source: str,
    ) -> tuple[float, float]:
        """Get conformal intervals for a property.

        Uses the source backend's intervals if available, else +/-50%.
        """
        lo_attr = f"{prop}_lo"
        hi_attr = f"{prop}_hi"

        if source == "admet-ai" and admet_result is not None:
            lo = getattr(admet_result, lo_attr, 0.0)
            hi = getattr(admet_result, hi_attr, 0.0)
            if hi > lo > 0.0:
                return float(lo), float(hi)
        if source == "polynomial" and poly_result is not None:
            lo = getattr(poly_result, lo_attr, 0.0)
            hi = getattr(poly_result, hi_attr, 0.0)
            if hi > lo > 0.0:
                return float(lo), float(hi)

        # Default: +/-50%
        return point_est * 0.5, point_est * 1.5

    @staticmethod
    def _get_rbp_intervals(
        admet_result: ADMEProperties | None,
        poly_result: ADMEProperties | None,
        rbp: float,
        source: str,
    ) -> tuple[float, float]:
        """Get RBP prediction intervals."""
        if source == "xgboost":
            # XGBoost doesn't provide intervals; use +/-20%
            return max(0.5, rbp * 0.8), min(3.0, rbp * 1.2)
        if source == "admet-ai" and admet_result is not None:
            lo = admet_result.rbp_lo
            hi = admet_result.rbp_hi
            if hi > lo > 0.0:
                return float(lo), float(hi)
        if source == "polynomial" and poly_result is not None:
            lo = poly_result.rbp_lo
            hi = poly_result.rbp_hi
            if hi > lo > 0.0:
                return float(lo), float(hi)
        return max(0.5, rbp * 0.5), min(3.0, rbp * 1.5)
