"""Interactive PK prediction interface wrapping the Level 3 foundation model.

High-level API for SMILES-to-PK with patient covariates, dosing regimens,
and few-shot adaptation.

Usage:
    predictor = InteractivePKPredictor()
    result = predictor.predict("CCO", patient={"age": 65}, dosing={"dose_mg": 500})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class InteractivePKPredictor:
    """High-level API for PK prediction with patient context and few-shot adaptation.

    Parameters
    ----------
    model_path : str or Path
        Path to saved PKFoundationModel checkpoint (.pt).
        If not found, initializes a new (untrained) model for testing.
    device : str
        Device for inference ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_path: str | Path = "models/foundation/best.pt",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model_path = Path(model_path)

        if self.model_path.exists():
            from omega_pbpk.ml.models.foundation.foundation_model import (
                PKFoundationModel,
            )

            self.model = PKFoundationModel.load(self.model_path, device=device)
            logger.info("Loaded PKFoundationModel from %s", self.model_path)
        else:
            # Initialize untrained model for API testing
            from omega_pbpk.ml.models.foundation.foundation_model import (
                PKFoundationModel,
            )

            self.model = PKFoundationModel()
            self.model.to(device)
            self.model.eval()
            logger.warning("Model file not found: %s. Using untrained model.", self.model_path)

    def predict(
        self,
        smiles: str,
        patient: dict[str, Any] | None = None,
        dosing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict PK profile for a molecule.

        Parameters
        ----------
        smiles : str
            SMILES string.
        patient : dict or None
            Patient covariates (age, weight, sex, hepatic_impairment, etc.).
            If None, uses population defaults (Level 2 mode).
        dosing : dict or None
            Dosing regimen (dose_mg, route, frequency, etc.).
            If None, uses default (100 mg oral single dose).

        Returns
        -------
        dict with:
            - predicted_params: dict of PK parameters (float values)
            - pk_profile: dict with Cmax, Tmax, AUC, t_half, CL, Vss
            - concentration_curve: list of (time_h, conc_mg_L) tuples
            - confidence: "low" / "medium" / "high"
            - uncertainty_intervals: dict of (lo, hi) tuples per param
        """
        self.model.eval()

        with torch.no_grad():
            result = self.model(
                smiles,
                covariates=patient,
                regimen=dosing,
            )

        # Extract params as plain floats
        predicted_params = {key: float(val.cpu().item()) for key, val in result["params"].items()}

        # PK metrics
        pk_metrics = result["pk_metrics"]
        pk_profile = {
            "Cmax_mg_L": float(pk_metrics["cmax"].cpu().item()),
            "Tmax_h": float(pk_metrics["tmax"].cpu().item()),
            "AUC_mg_h_L": float(pk_metrics["auc"].cpu().item()),
            "t_half_h": float(pk_metrics["t_half"].cpu().item()),
        }

        # Add derived PK params if available
        if "vd" in predicted_params and "ke" in predicted_params:
            pk_profile["CL_L_h"] = predicted_params["vd"] * predicted_params["ke"]
            pk_profile["Vss_L"] = predicted_params["vd"]

        # Concentration-time curve
        curve = result["curve"].cpu().squeeze(0).numpy()
        dt_h = self.model.dt_h
        concentration_curve = [(round(i * dt_h, 2), float(curve[i])) for i in range(len(curve))]

        # Confidence based on param plausibility
        confidence = self._assess_confidence(predicted_params)

        # Uncertainty intervals (placeholder based on param variability)
        uncertainty_intervals = self._estimate_uncertainty(predicted_params)

        return {
            "predicted_params": predicted_params,
            "pk_profile": pk_profile,
            "concentration_curve": concentration_curve,
            "confidence": confidence,
            "uncertainty_intervals": uncertainty_intervals,
        }

    def adapt(
        self,
        smiles: str,
        observations: list[tuple[float, float]],
        patient: dict[str, Any] | None = None,
        dosing: dict[str, Any] | None = None,
        n_steps: int = 10,
    ) -> InteractivePKPredictor:
        """Create an adapted predictor from observed data.

        Uses gradient-based fine-tuning on the observed concentration-time
        data to create a specialized model for this specific drug.

        Parameters
        ----------
        smiles : str
            SMILES string.
        observations : list of (time_h, conc_mg_L) tuples
            Observed concentration-time data.
        patient : dict or None
            Patient covariates.
        dosing : dict or None
            Dosing regimen.
        n_steps : int
            Number of adaptation gradient steps.

        Returns
        -------
        New InteractivePKPredictor with adapted model.
        """
        from omega_pbpk.ml.training.few_shot import adapt as reptile_adapt

        adapted_model = reptile_adapt(
            model=self.model,
            observations=observations,
            smiles=smiles,
            covariates=patient,
            regimen=dosing,
            n_steps=n_steps,
            lr=1e-3,
            device=self.device,
        )

        # Create new predictor with adapted model
        new_predictor = InteractivePKPredictor.__new__(InteractivePKPredictor)
        new_predictor.device = self.device
        new_predictor.model_path = self.model_path
        new_predictor.model = adapted_model
        return new_predictor

    def _assess_confidence(self, params: dict[str, float]) -> str:
        """Assess prediction confidence based on parameter plausibility.

        Returns
        -------
        "low", "medium", or "high"
        """
        issues = 0

        # Check plausibility ranges
        if params.get("fup", 0.5) < 0.001 or params.get("fup", 0.5) > 0.99:
            issues += 1
        if params.get("rbp", 1.0) < 0.3 or params.get("rbp", 1.0) > 3.5:
            issues += 1
        if params.get("mw", 300) < 50 or params.get("mw", 300) > 2000:
            issues += 1
        if params.get("logP", 2.0) < -5 or params.get("logP", 2.0) > 10:
            issues += 1

        if issues >= 2:
            return "low"
        elif issues >= 1:
            return "medium"
        return "high"

    def _estimate_uncertainty(self, params: dict[str, float]) -> dict[str, tuple[float, float]]:
        """Estimate uncertainty intervals for predicted parameters.

        Uses a simple heuristic: +/- 30% for well-constrained params,
        +/- 50% for less constrained ones. In production, this would
        use conformal prediction or MC dropout.

        Returns
        -------
        dict mapping param name to (lo, hi) tuple
        """
        intervals: dict[str, tuple[float, float]] = {}

        # Parameters with tighter constraints
        tight_params = {"fup", "rbp", "bioavailability"}
        # Parameters with wider uncertainty
        wide_params = {"clint_3a4", "clint_2d6", "vd", "ke", "ka"}

        for name, value in params.items():
            if name in tight_params:
                margin = 0.3
            elif name in wide_params:
                margin = 0.5
            else:
                margin = 0.4

            lo = value * (1.0 - margin)
            hi = value * (1.0 + margin)
            intervals[name] = (round(lo, 6), round(hi, 6))

        return intervals


__all__ = ["InteractivePKPredictor"]
