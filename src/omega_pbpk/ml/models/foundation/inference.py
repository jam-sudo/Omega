"""Inference pipeline: ML-predicted PK params + real ODE for final predictions.

The trained SMILESToPKModel predicts PK parameters via the GNN. At inference,
we use the REAL 35-state ODE (not the surrogate) for the final PK profile,
ensuring mechanistic accuracy.

Usage:
    predictor = MLPKPredictor("models/level2/best.pt")
    result = predictor.predict("CCO")
    print(result["pk_profile"])  # real ODE simulation result
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MLPKPredictor:
    """ML-powered PK prediction using GNN for parameters and real ODE for simulation.

    Pipeline:
    1. GNN encoder -> molecular embedding
    2. Parameter head -> named PK parameters
    3. Build Drug from predicted params
    4. Run real WholeBodyPBPK ODE for final PK profile

    Parameters
    ----------
    model_path : str or Path
        Path to saved SMILESToPKModel checkpoint (.pt).
    device : str
        Device for ML inference ('cpu' or 'cuda').
    dose_mg : float
        Default dose in mg.
    route : str
        Default administration route ('oral' or 'iv').
    body_weight : float
        Default body weight in kg.
    t_end_h : float
        Default simulation end time in hours.
    """

    def __init__(
        self,
        model_path: str | Path = "models/level2/best.pt",
        device: str = "cpu",
        dose_mg: float = 100.0,
        route: str = "oral",
        body_weight: float = 70.0,
        t_end_h: float = 24.0,
    ) -> None:
        self.device = device
        self.dose_mg = dose_mg
        self.route = route
        self.body_weight = body_weight
        self.t_end_h = t_end_h

        model_path = Path(model_path)
        if model_path.exists():
            from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel

            self.model = SMILESToPKModel.load(model_path, device=device)
            logger.info("Loaded model from %s", model_path)
            self._check_encoder_trained(model_path)
        else:
            logger.warning(
                "Model file not found: %s. predict() will fail until a model is loaded.",
                model_path,
            )
            self.model = None

    def _check_encoder_trained(self, model_path: Path) -> None:
        """Check if GNN encoder weights appear trained; warn if not.

        Uses two checks:
        1. Explicit 'trained' metadata flag in checkpoint (if present).
        2. Heuristic: compare encoder weights against a freshly initialized
           model. If keys match, computes relative L2 norm difference.
           If keys don't match (e.g. PureTorch vs PyG backend mismatch),
           checks weight statistics — untrained weights have mean ≈ 0 and
           small std, while trained weights diverge from init.
        """
        if self.model is None:
            return
        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

            # Check 1: explicit metadata flag
            if checkpoint.get("trained") is False:
                logger.warning(
                    "Level 2 GNN encoder has not been trained. "
                    "End-to-end SMILES→PK predictions will be unreliable. "
                    "Use Level 1 (omega predict) instead."
                )
                return

            # Check 2: heuristic weight comparison
            saved_state = checkpoint.get("model_state_dict", {})
            encoder_keys = [k for k in saved_state if k.startswith("encoder.")]
            if not encoder_keys:
                return

            # Try to compare against fresh model weights
            from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel

            fresh = SMILESToPKModel(
                embedding_dim=checkpoint.get("embedding_dim", 256),
                surrogate_n_output=checkpoint.get("surrogate_n_output", 241),
                surrogate_hidden=checkpoint.get("surrogate_hidden", 256),
            )
            fresh_state = fresh.state_dict()

            # Find matching encoder keys between saved and fresh
            matching_keys = [k for k in encoder_keys if k in fresh_state]

            if matching_keys:
                # Direct comparison: compute relative L2 norm difference
                total_diff = 0.0
                total_norm = 0.0
                for key in matching_keys:
                    diff = (saved_state[key].float() - fresh_state[key].float()).norm().item()
                    total_diff += diff
                    total_norm += saved_state[key].float().norm().item()

                if total_norm > 0 and total_diff / total_norm < 0.01:
                    logger.warning(
                        "Level 2 GNN encoder has not been trained. "
                        "End-to-end SMILES→PK predictions will be unreliable. "
                        "Use Level 1 (omega predict) instead."
                    )
            else:
                # Keys don't match (architecture mismatch, e.g. PureTorch vs PyG).
                # Use statistical heuristic: trained weights typically have larger
                # absolute mean and std than Kaiming/Xavier initialization.
                weight_keys = [
                    k for k in encoder_keys if "weight" in k and saved_state[k].dim() >= 2
                ]
                if weight_keys:
                    means = []
                    for k in weight_keys:
                        w = saved_state[k].float()
                        means.append(w.abs().mean().item())
                    avg_abs_mean = sum(means) / len(means)
                    # Kaiming init for ReLU: std ≈ sqrt(2/fan_in), abs_mean ≈ 0.03-0.1
                    # If abs_mean is suspiciously close to typical init, warn
                    # (but this is a weaker signal, so only warn with explicit flag)
                    logger.debug(
                        "L2 encoder weight avg abs mean: %.4f (architecture mismatch, "
                        "cannot compare directly)",
                        avg_abs_mean,
                    )

        except Exception as exc:
            logger.debug("Could not check encoder training status: %s", exc)

    def predict(
        self,
        smiles: str,
        dose_mg: float | None = None,
        route: str | None = None,
        body_weight: float | None = None,
        t_end_h: float | None = None,
    ) -> dict[str, Any]:
        """Predict PK profile for a molecule.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        dose_mg : float or None
            Dose in mg (overrides default).
        route : str or None
            Administration route (overrides default).
        body_weight : float or None
            Body weight in kg (overrides default).
        t_end_h : float or None
            Simulation end time (overrides default).

        Returns
        -------
        dict with:
            - drug: Drug object with predicted parameters
            - params: dict of predicted PK parameters (raw tensors converted to floats)
            - pk_profile: dict from WholeBodyPBPK.simulate().pk_summary()
            - surrogate_curve: numpy array of surrogate-predicted C(t)
            - simulation_result: full SimulationResult from ODE
            - warnings: list of any warnings encountered
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Provide a valid model_path to the constructor.")

        dose = dose_mg or self.dose_mg
        rt = route or self.route
        bw = body_weight or self.body_weight
        t_end = t_end_h or self.t_end_h
        warnings_list: list[str] = []

        # 1. GNN -> predicted PK params
        self.model.eval()
        with torch.no_grad():
            result = self.model(smiles)

        pk_params = result["params"]
        surrogate_curve = result["curve"].cpu().numpy().squeeze()

        # 2. Convert tensor params to plain Python floats
        params_dict = {key: float(val.cpu().item()) for key, val in pk_params.items()}
        logger.info("Predicted params for %s: %s", smiles[:20], params_dict)

        # 3. Build Drug from predicted params
        drug = self._params_to_drug(params_dict, smiles, dose, rt)

        # 4. Run REAL ODE for final prediction
        try:
            sim_result = self._run_real_ode(drug, dose, rt, bw, t_end)
            pk_profile = sim_result.pk_summary()
        except Exception as e:
            warnings_list.append(f"Real ODE simulation failed: {e}")
            logger.warning("ODE simulation failed for %s: %s", smiles[:20], e)
            sim_result = None
            pk_profile = self._fallback_pk_from_surrogate(result, params_dict)

        return {
            "drug": drug,
            "params": params_dict,
            "pk_profile": pk_profile,
            "surrogate_curve": surrogate_curve,
            "simulation_result": sim_result,
            "warnings": warnings_list,
        }

    def _params_to_drug(
        self,
        params: dict[str, float],
        smiles: str,
        dose_mg: float,
        route: str,
    ) -> Any:
        """Build a Drug dataclass from predicted parameters.

        Parameters
        ----------
        params : dict of predicted parameter values
        smiles : SMILES string
        dose_mg : dose in mg
        route : administration route

        Returns
        -------
        Drug instance
        """
        from omega_pbpk.drugs.drug import Drug

        # IVIVE scaling: total CLint (uL/min/pmol) -> hepatic clearance (L/h)
        clint_3a4 = params.get("clint_3a4", 5.0)
        clint_2d6 = params.get("clint_2d6", 1.0)
        ivive_factor = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0
        clint_hepatic = (clint_3a4 + clint_2d6) * ivive_factor

        return Drug(
            name=f"ml_pred_{smiles[:12]}",
            mw=params.get("mw", 300.0),
            logP=params.get("logP", 2.0),
            fup=max(params.get("fup", 0.1), 0.001),
            rbp=max(params.get("rbp", 1.0), 0.3),
            peff=max(params.get("peff", 1.0), 0.01),
            clint_hepatic_L_per_h=max(clint_hepatic, 0.001),
            clint={"CYP3A4": clint_3a4, "CYP2D6": clint_2d6},
            smiles=smiles,
            dose_mg=dose_mg,
            route=route,
        )

    def _run_real_ode(
        self,
        drug: Any,
        dose_mg: float,
        route: str,
        body_weight: float,
        t_end_h: float,
    ) -> Any:
        """Run the real WholeBodyPBPK ODE simulation.

        Returns
        -------
        SimulationResult from WholeBodyPBPK
        """
        from omega_pbpk.core.body import WholeBodyPBPK

        model = WholeBodyPBPK(drug, body_weight=body_weight)
        if route == "iv":
            model.setup_iv(dose_mg)
        else:
            model.setup_oral(dose_mg)

        return model.simulate(t_end_h=t_end_h)

    def _fallback_pk_from_surrogate(
        self, result: dict, params: dict[str, float]
    ) -> dict[str, float]:
        """Extract PK summary from surrogate curve when ODE fails."""
        metrics = result["pk_metrics"]
        return {
            "Cmax_mg_L": float(metrics["cmax"].cpu().item()),
            "AUC_mg_h_L": float(metrics["auc"].cpu().item()),
            "Tmax_h": float(metrics["tmax"].cpu().item()),
            "half_life_h": float(metrics["t_half"].cpu().item()),
            "source": "surrogate_fallback",
        }

    def predict_batch(self, smiles_list: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Predict PK profiles for multiple molecules.

        Parameters
        ----------
        smiles_list : list of SMILES strings
        **kwargs : passed to predict()

        Returns
        -------
        list of prediction dicts
        """
        results = []
        for smi in smiles_list:
            try:
                result = self.predict(smi, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", smi[:20], e)
                results.append({"smiles": smi, "error": str(e), "warnings": [str(e)]})
        return results


__all__ = ["MLPKPredictor"]
