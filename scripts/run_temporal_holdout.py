"""Temporal holdout validation — post-2022 FDA-approved drugs.

Validates Omega predictions on drugs approved after ADMET-AI's training
cutoff (~2022), ensuring the model generalizes to truly unseen compounds.

Usage:
    python scripts/run_temporal_holdout.py [--verbose]
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temporal_holdout")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HOLDOUT_CSV = REPO_ROOT / "data" / "ml" / "clinical" / "temporal_holdout.csv"
OUTPUT_JSON = REPO_ROOT / "outputs" / "temporal_holdout_results.json"


def load_holdout() -> list[dict]:
    """Load temporal holdout CSV."""
    rows = []
    with open(HOLDOUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_aafe(observed: list[float], predicted: list[float]) -> float:
    """Average Absolute Fold Error."""
    if not observed:
        return float("nan")
    fold_errors = []
    for o, p in zip(observed, predicted):
        if o > 0 and p > 0:
            fold_errors.append(max(o / p, p / o))
    if not fold_errors:
        return float("nan")
    return sum(fold_errors) / len(fold_errors)


def pct_within_twofold(observed: list[float], predicted: list[float]) -> float:
    """Percentage of predictions within 2-fold of observed."""
    if not observed:
        return float("nan")
    n_good = 0
    n_total = 0
    for o, p in zip(observed, predicted):
        if o > 0 and p > 0:
            n_total += 1
            fe = max(o / p, p / o)
            if fe <= 2.0:
                n_good += 1
    if n_total == 0:
        return float("nan")
    return 100.0 * n_good / n_total


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Temporal holdout validation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()

    drugs = load_holdout()
    logger.info("Loaded %d temporal holdout drugs", len(drugs))

    results = []
    obs_thalf_list: list[float] = []
    pred_thalf_list: list[float] = []
    obs_cmax_list: list[float] = []
    pred_cmax_list: list[float] = []
    obs_auc_list: list[float] = []
    pred_auc_list: list[float] = []

    for drug_row in drugs:
        name = drug_row["drug_name"]
        smiles = drug_row["smiles"]
        dose_mg = float(drug_row["dose_mg"])
        route = drug_row["route"]

        # Parse observed values (may be empty)
        obs_thalf = float(drug_row["obs_thalf_h"]) if drug_row["obs_thalf_h"] else None
        obs_cmax = float(drug_row["obs_cmax_mg_L"]) if drug_row["obs_cmax_mg_L"] else None
        obs_auc = float(drug_row["obs_auc_mg_h_L"]) if drug_row["obs_auc_mg_h_L"] else None

        # Determine simulation duration: 5x observed t_half or at least 48h
        if obs_thalf:
            duration_h = max(obs_thalf * 5.0, 48.0)
        else:
            duration_h = 48.0

        logger.info(
            "Running %s: SMILES=%s..., dose=%gmg, duration=%.0fh",
            name,
            smiles[:40],
            dose_mg,
            duration_h,
        )

        try:
            request = SimulationRequest(
                smiles=smiles,
                dose_mg=dose_mg,
                route=route,
                duration_h=duration_h,
            )
            result = pipeline.simulate(request)

            pred_thalf = result.t_half_h
            pred_cmax = result.cmax_mg_L
            pred_auc = result.auc0t_mg_h_L

            # Compute fold errors where observed data exists
            thalf_fe = None
            if obs_thalf and pred_thalf and not math.isnan(pred_thalf) and pred_thalf > 0:
                thalf_fe = round(max(obs_thalf / pred_thalf, pred_thalf / obs_thalf), 2)
                obs_thalf_list.append(obs_thalf)
                pred_thalf_list.append(pred_thalf)

            cmax_fe = None
            if obs_cmax and pred_cmax and pred_cmax > 0:
                cmax_fe = round(max(obs_cmax / pred_cmax, pred_cmax / obs_cmax), 2)
                obs_cmax_list.append(obs_cmax)
                pred_cmax_list.append(pred_cmax)

            auc_fe = None
            if obs_auc and pred_auc and pred_auc > 0:
                auc_fe = round(max(obs_auc / pred_auc, pred_auc / obs_auc), 2)
                obs_auc_list.append(obs_auc)
                pred_auc_list.append(pred_auc)

            drug_result = {
                "drug_name": name,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "obs_thalf_h": obs_thalf,
                "pred_thalf_h": round(pred_thalf, 2)
                if pred_thalf and not math.isnan(pred_thalf)
                else None,
                "thalf_fold_error": thalf_fe,
                "obs_cmax_mg_L": obs_cmax,
                "pred_cmax_mg_L": round(pred_cmax, 4) if pred_cmax else None,
                "cmax_fold_error": cmax_fe,
                "obs_auc_mg_h_L": obs_auc,
                "pred_auc_mg_h_L": round(pred_auc, 2) if pred_auc else None,
                "auc_fold_error": auc_fe,
                "confidence": result.confidence,
                "warnings": result.warnings,
                "status": "ok",
            }

            logger.info(
                "  %s: pred_t½=%.1fh (obs=%.1fh, FE=%.1fx), pred_Cmax=%.4f, confidence=%s",
                name,
                pred_thalf if pred_thalf and not math.isnan(pred_thalf) else 0,
                obs_thalf or 0,
                thalf_fe or 0,
                pred_cmax or 0,
                result.confidence,
            )

        except Exception as exc:
            logger.error("  %s FAILED: %s", name, exc)
            drug_result = {
                "drug_name": name,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "status": "error",
                "error": str(exc),
            }

        results.append(drug_result)

    # Aggregate metrics
    thalf_aafe = compute_aafe(obs_thalf_list, pred_thalf_list)
    thalf_pct2fold = pct_within_twofold(obs_thalf_list, pred_thalf_list)
    cmax_aafe = compute_aafe(obs_cmax_list, pred_cmax_list)
    cmax_pct2fold = pct_within_twofold(obs_cmax_list, pred_cmax_list)
    auc_aafe = compute_aafe(obs_auc_list, pred_auc_list)
    auc_pct2fold = pct_within_twofold(obs_auc_list, pred_auc_list)

    summary = {
        "n_drugs": len(drugs),
        "n_success": sum(1 for r in results if r.get("status") == "ok"),
        "n_failed": sum(1 for r in results if r.get("status") == "error"),
        "thalf": {
            "n_compared": len(obs_thalf_list),
            "AAFE": round(thalf_aafe, 2) if not math.isnan(thalf_aafe) else None,
            "pct_within_2fold": round(thalf_pct2fold, 1)
            if not math.isnan(thalf_pct2fold)
            else None,
        },
        "cmax": {
            "n_compared": len(obs_cmax_list),
            "AAFE": round(cmax_aafe, 2) if not math.isnan(cmax_aafe) else None,
            "pct_within_2fold": round(cmax_pct2fold, 1) if not math.isnan(cmax_pct2fold) else None,
        },
        "auc": {
            "n_compared": len(obs_auc_list),
            "AAFE": round(auc_aafe, 2) if not math.isnan(auc_aafe) else None,
            "pct_within_2fold": round(auc_pct2fold, 1) if not math.isnan(auc_pct2fold) else None,
        },
    }

    output = {
        "description": "Temporal holdout validation: post-2022 FDA-approved drugs",
        "summary": summary,
        "drugs": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("TEMPORAL HOLDOUT RESULTS")
    logger.info("=" * 60)
    logger.info("Drugs: %d success, %d failed", summary["n_success"], summary["n_failed"])
    if summary["thalf"]["n_compared"] > 0:
        logger.info(
            "t_half: AAFE=%.2f, %%2-fold=%.0f%% (n=%d)",
            summary["thalf"]["AAFE"],
            summary["thalf"]["pct_within_2fold"],
            summary["thalf"]["n_compared"],
        )
    if summary["cmax"]["n_compared"] > 0:
        logger.info(
            "Cmax:   AAFE=%.2f, %%2-fold=%.0f%% (n=%d)",
            summary["cmax"]["AAFE"],
            summary["cmax"]["pct_within_2fold"],
            summary["cmax"]["n_compared"],
        )
    if summary["auc"]["n_compared"] > 0:
        logger.info(
            "AUC:    AAFE=%.2f, %%2-fold=%.0f%% (n=%d)",
            summary["auc"]["AAFE"],
            summary["auc"]["pct_within_2fold"],
            summary["auc"]["n_compared"],
        )
    logger.info("Results saved to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
