"""Benchmark suites for ADME prediction validation.

Runs end-to-end: SMILES -> ensemble.predict() -> Drug -> WholeBodyPBPK -> PK metrics.
Computes fold-error vs known clinical values and reports AAFE per metric.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent  # evaluation -> ml -> omega_pbpk -> src -> Omega


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single compound benchmark."""

    name: str
    smiles: str
    predicted: dict[str, float]
    observed: dict[str, float]
    fold_errors: dict[str, float]
    success: bool
    error_msg: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of benchmark suite results."""

    n_compounds: int = 0
    n_success: int = 0
    n_failed: int = 0
    aafe_per_metric: dict[str, float] = field(default_factory=dict)
    pct_within_2fold: dict[str, float] = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Benchmark Summary: {self.n_success}/{self.n_compounds} compounds succeeded",
            "",
            "AAFE per metric (Average Absolute Fold Error):",
        ]
        for metric, aafe in sorted(self.aafe_per_metric.items()):
            pct = self.pct_within_2fold.get(metric, 0.0)
            lines.append(f"  {metric:<15}: AAFE={aafe:.2f}, within 2-fold={pct:.0f}%")
        return "\n".join(lines)


def compute_fold_error(predicted: float, observed: float) -> float:
    """Compute absolute fold error between predicted and observed values.

    fold_error = max(predicted/observed, observed/predicted)

    Args:
        predicted: Predicted value.
        observed: Observed (true) value.

    Returns:
        Fold error (>= 1.0). Returns NaN if either value is zero or NaN.
    """
    if np.isnan(predicted) or np.isnan(observed) or abs(predicted) < 1e-12 or abs(observed) < 1e-12:
        return float("nan")
    ratio = predicted / observed
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors: list[float]) -> float:
    """Compute Average Absolute Fold Error.

    AAFE = 10^(mean(log10(fold_errors)))

    Args:
        fold_errors: List of fold error values (each >= 1.0).

    Returns:
        AAFE value, or NaN if no valid entries.
    """
    valid = [fe for fe in fold_errors if not np.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    log_errors = [np.log10(fe) for fe in valid]
    return float(10.0 ** np.mean(log_errors))


def run_benchmark(
    compounds: list[dict[str, Any]],
    predictor: Any | None = None,
    dose_mg: float = 100.0,
    route: str = "oral",
    duration_h: float = 24.0,
) -> BenchmarkSummary:
    """Run end-to-end PK benchmark on a list of compounds.

    Each compound dict should have:
      - 'name': str
      - 'smiles': str
      - Optionally: 'observed_cmax', 'observed_auc', 'observed_thalf' (clinical values)
      - Optionally: 'observed_fup', 'observed_rbp', etc. (ADME values)

    Args:
        compounds: List of compound dicts with SMILES and optional observed values.
        predictor: MLADMEPredictor to use (default: EnsembleADMEPredictor).
        dose_mg: Dose for simulation.
        route: Administration route.
        duration_h: Simulation duration.

    Returns:
        BenchmarkSummary with per-compound results and aggregate metrics.
    """
    if predictor is None:
        try:
            from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

            predictor = EnsembleADMEPredictor()
        except Exception as exc:
            logger.error("Failed to create EnsembleADMEPredictor: %s", exc)
            return BenchmarkSummary()

    summary = BenchmarkSummary()
    summary.n_compounds = len(compounds)

    # Collect fold errors per metric across compounds
    all_fold_errors: dict[str, list[float]] = {}

    for comp in compounds:
        name = comp.get("name", "unknown")
        smiles = comp.get("smiles", "")
        if not smiles:
            summary.n_failed += 1
            summary.results.append(
                BenchmarkResult(
                    name=name,
                    smiles=smiles,
                    predicted={},
                    observed={},
                    fold_errors={},
                    success=False,
                    error_msg="No SMILES provided",
                )
            )
            continue

        try:
            result = _run_single_compound(name, smiles, comp, predictor, dose_mg, route, duration_h)
            summary.results.append(result)
            if result.success:
                summary.n_success += 1
                for metric, fe in result.fold_errors.items():
                    all_fold_errors.setdefault(metric, []).append(fe)
            else:
                summary.n_failed += 1
        except Exception as exc:
            summary.n_failed += 1
            summary.results.append(
                BenchmarkResult(
                    name=name,
                    smiles=smiles,
                    predicted={},
                    observed={},
                    fold_errors={},
                    success=False,
                    error_msg=str(exc),
                )
            )

    # Compute aggregate metrics
    for metric, errors in all_fold_errors.items():
        summary.aafe_per_metric[metric] = compute_aafe(errors)
        valid = [fe for fe in errors if not np.isnan(fe)]
        if valid:
            within_2fold = sum(1 for fe in valid if fe <= 2.0) / len(valid) * 100.0
            summary.pct_within_2fold[metric] = round(within_2fold, 1)

    return summary


def _run_single_compound(
    name: str,
    smiles: str,
    comp: dict[str, Any],
    predictor: Any,
    dose_mg: float,
    route: str,
    duration_h: float,
) -> BenchmarkResult:
    """Run benchmark for a single compound.

    Args:
        name: Compound name.
        smiles: SMILES string.
        comp: Full compound dict with observed values.
        predictor: ADME predictor instance.
        dose_mg: Dose in mg.
        route: Administration route.
        duration_h: Duration in hours.

    Returns:
        BenchmarkResult for this compound.
    """
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    # Predict ADME
    adme = predictor.predict(smiles)
    predicted_adme = {
        "mw": adme.mw,
        "logP": adme.logP,
        "logS": adme.logS,
        "fup": adme.fup,
        "rbp": adme.rbp,
        "clint_3a4": adme.clint_3a4,
        "peff": adme.peff,
    }

    # Build Drug and run simulation
    # IVIVE: µL/min/pmol × MPPGL(40) × mg/g(45) × liver(1800g) / 1e6 / 60 ≈ ×0.054
    clint_L_per_h = adme.clint_3a4 * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0

    # Solubility: convert logS (log10 mol/L) → mg/mL
    #   S_mol_L = 10^logS; S_mg_mL = S_mol_L × MW / 1000
    solubility_mg_mL = max((10 ** adme.logS) * adme.mw / 1000.0, 1e-6)

    # Gut-wall CYP3A4 extraction: enterocyte CYP3A4 activity is ~1-10% of hepatic.
    # For CYP3A4-metabolized drugs, use a gut_clint_multiplier to model gut first-pass.
    # Default multiplier of 0.1 means gut CYP3A4 = 10% of liver CYP3A4 activity.
    # This is applied to CYP3A4 CLint only — non-CYP3A4 drugs get no gut extraction.
    gut_clint_multiplier = 0.1  # standard value from literature

    # Estimate renal clearance from physicochemical properties.
    # Without this, renally-cleared drugs (metformin, atenolol, gabapentin)
    # have no renal elimination in benchmarks and massively over-predict AUC.
    from omega_pbpk.pipeline import OmegaPipeline

    cl_renal = OmegaPipeline._estimate_renal_clearance(
        adme.logP, max(adme.fup, 0.001), adme.mw, smiles=smiles
    )

    drug = Drug(
        name=name,
        mw=adme.mw,
        logP=adme.logP,
        fup=max(adme.fup, 0.001),
        rbp=adme.rbp,
        clint_hepatic_L_per_h=clint_L_per_h,
        clint={"CYP3A4": adme.clint_3a4},
        clr_L_per_h=cl_renal,
        peff=adme.peff,
        solubility_mg_mL=solubility_mg_mL,
        gut_clint_multiplier=gut_clint_multiplier,
    )

    model = WholeBodyPBPK(drug=drug, body_weight=70.0)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    result = model.simulate(t_end_h=duration_h)
    pk = result.pk_summary()

    predicted_pk = {
        "cmax": pk.get("Cmax_mg_L", float("nan")),
        "auc": pk.get("AUC0t_mg_h_L", float("nan")),
        "thalf": pk.get("t_half_h", float("nan")),
    }
    predicted = {**predicted_adme, **predicted_pk}

    # Collect observed values
    observed: dict[str, float] = {}
    fold_errors: dict[str, float] = {}

    obs_map = {
        "observed_cmax": "cmax",
        "observed_auc": "auc",
        "observed_thalf": "thalf",
        "fup": "fup",
        "rbp": "rbp",
        "clint_3a4_uL_min_pmol": "clint_3a4",
        "peff_cm_s": "peff",
    }

    for obs_key, metric_key in obs_map.items():
        obs_val = comp.get(obs_key)
        if obs_val is not None:
            try:
                obs_float = float(obs_val)
                observed[metric_key] = obs_float
                pred_val = predicted.get(metric_key, float("nan"))
                # For peff, reference is in cm/s but our predicted is x10^-4 cm/s
                if metric_key == "peff" and obs_key == "peff_cm_s":
                    pred_val = pred_val * 1e-4  # convert to cm/s for comparison
                fold_errors[metric_key] = compute_fold_error(pred_val, obs_float)
            except (ValueError, TypeError):
                pass

    return BenchmarkResult(
        name=name,
        smiles=smiles,
        predicted=predicted,
        observed=observed,
        fold_errors=fold_errors,
        success=True,
    )


def load_reference_compounds(
    data_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load reference compounds from adme_reference.csv for benchmarking.

    Args:
        data_path: Path to CSV file. Uses default if None.

    Returns:
        List of compound dicts suitable for run_benchmark().
    """
    from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

    path = data_path or (_REPO_ROOT / "data" / "adme_reference.csv")
    if not path.exists():
        logger.warning("Reference data not found at %s", path)
        return []

    compounds: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            smiles = row.get("smiles", row.get("SMILES", ""))
            if not smiles:
                smiles = _name_to_smiles(name)
            if not smiles:
                continue
            compounds.append(
                {
                    "name": name,
                    "smiles": smiles,
                    **{k: v for k, v in row.items() if k != "name"},
                }
            )

    return compounds
