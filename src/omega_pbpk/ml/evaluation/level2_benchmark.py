"""Level 2 validation and benchmarking for ML PK predictions.

Evaluates the MLPKPredictor against reference drugs with known ADME/PK values.
Reports AAFE per parameter and PK metric, fold-error distributions, and
surrogate-vs-ODE agreement.

Exit criteria for Level 2:
- AAFE < 2.0 per PK metric
- >= 70% of drugs within 2-fold for Cmax, AUC
- Surrogate vs real ODE AAFE < 1.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from Level 2 benchmarking.

    Attributes
    ----------
    n_drugs : int
        Number of drugs evaluated.
    param_aafe : dict
        AAFE per predicted parameter (vs known values).
    pk_metric_aafe : dict
        AAFE per PK metric (Cmax, AUC, Tmax, t_half).
    pct_within_2fold : dict
        % of drugs within 2-fold per metric.
    surrogate_vs_ode_aafe : float
        AAFE between surrogate curve and real ODE curve.
    drug_results : list
        Per-drug prediction details.
    summary : str
        Human-readable summary.
    passes_level2 : bool
        Whether all Level 2 exit criteria are met.
    """

    n_drugs: int = 0
    param_aafe: dict[str, float] = field(default_factory=dict)
    pk_metric_aafe: dict[str, float] = field(default_factory=dict)
    pct_within_2fold: dict[str, float] = field(default_factory=dict)
    surrogate_vs_ode_aafe: float = 0.0
    drug_results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    passes_level2: bool = False


def _compute_aafe(predicted: NDArray, observed: NDArray) -> float:
    """Compute Average Absolute Fold Error.

    AAFE = exp(mean(|log(predicted/observed)|))

    Parameters
    ----------
    predicted : array of predicted values
    observed : array of observed values

    Returns
    -------
    AAFE value (>= 1.0, lower is better)
    """
    eps = 1e-12
    pred = np.maximum(np.abs(predicted), eps)
    obs = np.maximum(np.abs(observed), eps)
    log_fold = np.abs(np.log(pred / obs))
    return float(np.exp(np.mean(log_fold)))


def _pct_within_nfold(predicted: NDArray, observed: NDArray, n: float) -> float:
    """Compute percentage of predictions within n-fold of observed."""
    eps = 1e-12
    pred = np.maximum(np.abs(predicted), eps)
    obs = np.maximum(np.abs(observed), eps)
    fold = np.maximum(pred / obs, obs / pred)
    return float(np.mean(fold <= n) * 100.0)


def run_level2_benchmark(
    predictor: Any,
    reference_drugs: list[dict[str, Any]] | None = None,
    reference_csv: str | Path | None = None,
) -> BenchmarkResult:
    """Run comprehensive Level 2 benchmark.

    Parameters
    ----------
    predictor : MLPKPredictor
        The ML predictor to evaluate.
    reference_drugs : list of dicts, optional
        Each dict should have:
        - smiles: str
        - name: str
        - known_params: dict (mw, logP, fup, rbp, etc.)
        - known_pk: dict (Cmax_mg_L, AUC_mg_h_L, Tmax_h, half_life_h), optional
    reference_csv : str or Path, optional
        Path to CSV with reference data.

    Returns
    -------
    BenchmarkResult
    """
    if reference_drugs is None:
        if reference_csv is not None:
            reference_drugs = _load_reference_csv(Path(reference_csv))
        else:
            reference_drugs = _get_builtin_reference_drugs()

    if not reference_drugs:
        logger.warning("No reference drugs available for benchmarking")
        return BenchmarkResult(summary="No reference drugs available")

    drug_results = []
    param_predictions: dict[str, list[tuple[float, float]]] = {}
    pk_predictions: dict[str, list[tuple[float, float]]] = {}
    surrogate_ode_folds: list[float] = []

    for ref in reference_drugs:
        smiles = ref["smiles"]
        name = ref.get("name", smiles[:12])
        known_params = ref.get("known_params", {})
        known_pk = ref.get("known_pk", {})

        try:
            result = predictor.predict(smiles)
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", name, e)
            drug_results.append({"name": name, "error": str(e)})
            continue

        pred_params = result["params"]
        pred_pk = result.get("pk_profile", {})
        drug_entry: dict[str, Any] = {
            "name": name,
            "smiles": smiles,
            "predicted_params": pred_params,
            "predicted_pk": pred_pk,
            "known_params": known_params,
            "known_pk": known_pk,
        }

        # Compare predicted params vs known
        for param_name, known_val in known_params.items():
            if param_name in pred_params and known_val is not None:
                pred_val = pred_params[param_name]
                if param_name not in param_predictions:
                    param_predictions[param_name] = []
                param_predictions[param_name].append((pred_val, known_val))

        # Compare PK metrics
        pk_map = {
            "Cmax_mg_L": "Cmax_mg_L",
            "AUC_mg_h_L": "AUC_mg_h_L",
            "Tmax_h": "Tmax_h",
            "half_life_h": "half_life_h",
        }
        for pk_key, profile_key in pk_map.items():
            if pk_key in known_pk and profile_key in pred_pk:
                known_val = known_pk[pk_key]
                pred_val = pred_pk[profile_key]
                if (
                    known_val is not None
                    and pred_val is not None
                    and np.isfinite(known_val)
                    and np.isfinite(pred_val)
                ):
                    if pk_key not in pk_predictions:
                        pk_predictions[pk_key] = []
                    pk_predictions[pk_key].append((pred_val, known_val))

        # Surrogate vs ODE comparison
        surrogate_curve = result.get("surrogate_curve")
        sim_result = result.get("simulation_result")
        if surrogate_curve is not None and sim_result is not None:
            try:
                ode_cp = sim_result.plasma_concentration()
                # Align lengths (take min)
                n_pts = min(len(surrogate_curve), len(ode_cp))
                if n_pts > 0:
                    s_cmax = float(np.max(surrogate_curve[:n_pts]))
                    o_cmax = float(np.max(ode_cp[:n_pts]))
                    if s_cmax > 1e-12 and o_cmax > 1e-12:
                        fold = max(s_cmax / o_cmax, o_cmax / s_cmax)
                        surrogate_ode_folds.append(fold)
                        drug_entry["surrogate_cmax"] = s_cmax
                        drug_entry["ode_cmax"] = o_cmax
            except Exception as e:
                logger.debug("Surrogate-ODE comparison failed for %s: %s", name, e)

        drug_results.append(drug_entry)

    # Aggregate metrics
    result_obj = BenchmarkResult(n_drugs=len(drug_results), drug_results=drug_results)

    # Parameter AAFE
    for param_name, pairs in param_predictions.items():
        pred_arr = np.array([p[0] for p in pairs])
        obs_arr = np.array([p[1] for p in pairs])
        if len(pred_arr) > 0:
            result_obj.param_aafe[param_name] = round(_compute_aafe(pred_arr, obs_arr), 3)

    # PK metric AAFE and within 2-fold
    for pk_key, pairs in pk_predictions.items():
        pred_arr = np.array([p[0] for p in pairs])
        obs_arr = np.array([p[1] for p in pairs])
        if len(pred_arr) > 0:
            result_obj.pk_metric_aafe[pk_key] = round(_compute_aafe(pred_arr, obs_arr), 3)
            result_obj.pct_within_2fold[pk_key] = round(
                _pct_within_nfold(pred_arr, obs_arr, 2.0), 1
            )

    # Surrogate vs ODE AAFE
    if surrogate_ode_folds:
        result_obj.surrogate_vs_ode_aafe = round(
            float(np.exp(np.mean(np.log(surrogate_ode_folds)))), 3
        )

    # Check Level 2 exit criteria
    pk_aafe_ok = all(v < 2.0 for v in result_obj.pk_metric_aafe.values())
    fold_ok = all(v >= 70.0 for v in result_obj.pct_within_2fold.values())
    surrogate_ok = (
        result_obj.surrogate_vs_ode_aafe < 1.5 if result_obj.surrogate_vs_ode_aafe > 0 else True
    )
    result_obj.passes_level2 = pk_aafe_ok and fold_ok and surrogate_ok

    # Summary
    lines = [
        f"Level 2 Benchmark: {result_obj.n_drugs} drugs evaluated",
        "",
        "Parameter AAFE:",
    ]
    for k, v in sorted(result_obj.param_aafe.items()):
        lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("PK Metric AAFE (target < 2.0):")
    for k, v in sorted(result_obj.pk_metric_aafe.items()):
        status = "PASS" if v < 2.0 else "FAIL"
        lines.append(f"  {k}: {v:.3f} [{status}]")
    lines.append("")
    lines.append("% Within 2-fold (target >= 70%):")
    for k, v in sorted(result_obj.pct_within_2fold.items()):
        status = "PASS" if v >= 70.0 else "FAIL"
        lines.append(f"  {k}: {v:.1f}% [{status}]")
    if result_obj.surrogate_vs_ode_aafe > 0:
        surr_status = "PASS" if surrogate_ok else "FAIL"
        lines.append(
            f"\nSurrogate vs ODE AAFE: {result_obj.surrogate_vs_ode_aafe:.3f} [{surr_status}]"
        )
    lines.append(f"\nOverall Level 2: {'PASS' if result_obj.passes_level2 else 'FAIL'}")

    result_obj.summary = "\n".join(lines)
    return result_obj


def _load_reference_csv(path: Path) -> list[dict[str, Any]]:
    """Load reference drugs from a CSV file."""
    import csv

    drugs = []
    if not path.exists():
        logger.warning("Reference CSV not found: %s", path)
        return drugs

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("smiles", "")
            if not smiles:
                continue

            known_params: dict[str, float] = {}
            for key in ("mw", "logP", "fup", "rbp", "peff", "clint_3a4"):
                if key in row and row[key]:
                    try:
                        known_params[key] = float(row[key])
                    except ValueError:
                        pass

            known_pk: dict[str, float] = {}
            for key in ("Cmax_mg_L", "AUC_mg_h_L", "Tmax_h", "half_life_h"):
                if key in row and row[key]:
                    try:
                        known_pk[key] = float(row[key])
                    except ValueError:
                        pass

            drugs.append(
                {
                    "smiles": smiles,
                    "name": row.get("name", smiles[:12]),
                    "known_params": known_params,
                    "known_pk": known_pk,
                }
            )

    return drugs


def _get_builtin_reference_drugs() -> list[dict[str, Any]]:
    """Return a small set of well-known reference drugs for quick benchmarking.

    These are drugs with well-established PK parameters from literature.
    """
    return [
        {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "name": "Aspirin",
            "known_params": {
                "mw": 180.16,
                "logP": 1.2,
                "fup": 0.5,
            },
            "known_pk": {},
        },
        {
            "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "name": "Ibuprofen",
            "known_params": {
                "mw": 206.28,
                "logP": 3.97,
                "fup": 0.01,
            },
            "known_pk": {},
        },
        {
            "smiles": "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
            "name": "Acetaminophen",
            "known_params": {
                "mw": 151.16,
                "logP": 0.46,
                "fup": 0.8,
            },
            "known_pk": {},
        },
        {
            "smiles": "Clc1ccc(cc1)C(c1ccccc1)n1ccnc1",  # Clotrimazole
            "name": "Clotrimazole",
            "known_params": {
                "mw": 344.84,
                "logP": 5.0,
                "fup": 0.02,
            },
            "known_pk": {},
        },
        {
            "smiles": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # Salbutamol
            "name": "Salbutamol",
            "known_params": {
                "mw": 239.31,
                "logP": 0.64,
                "fup": 0.92,
            },
            "known_pk": {},
        },
    ]


__all__ = ["BenchmarkResult", "run_level2_benchmark"]
