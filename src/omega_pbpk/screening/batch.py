"""Batch screening engine for OmegaPipeline.

Runs OmegaPipeline.simulate() on multiple SMILES with error handling
and result ranking. Sequential execution (~73ms/drug → 1000 drugs in ~73s).

Usage:
    from omega_pbpk.screening.batch import batch_predict, rank_results

    results = batch_predict(smiles_list, dose_mg=100.0)
    ranked = rank_results(results, objective="cmax")
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def batch_predict(
    smiles_list: list[str],
    dose_mg: float = 100.0,
    route: str = "oral",
    duration_h: float = 24.0,
) -> list[dict[str, Any]]:
    """Run OmegaPipeline on multiple SMILES.

    Returns list of dicts, one per SMILES. Each dict has:
        smiles, cmax_mg_L, auc_mg_h_L, t_half_h, tmax_h,
        confidence, cmax_ci90, warnings, latency_ms
    On error: smiles, error
    """
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    results = []

    for i, smiles in enumerate(smiles_list):
        t0 = time.time()
        try:
            sim = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route=route,
                    duration_h=duration_h,
                )
            )
            entry: dict[str, Any] = {
                "smiles": smiles,
                "cmax_mg_L": sim.cmax_mg_L,
                "auc_mg_h_L": sim.auc0t_mg_h_L,
                "t_half_h": sim.t_half_h,
                "tmax_h": sim.tmax_h,
                "confidence": sim.confidence,
                "warnings": sim.warnings,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
            # Optional fields (may not exist if UQ/P-gp not yet integrated)
            for attr in ("cmax_ci90", "auc_ci90", "thalf_ci90"):
                val = getattr(sim, attr, None)
                if val is not None:
                    entry[attr] = val
            entry["pgp_substrate"] = sim.adme_properties.get("pgp_substrate", False)
            entry["correction_applied"] = sim.adme_properties.get("correction_applied", False)
            results.append(entry)
        except Exception as e:
            results.append({"smiles": smiles, "error": str(e)})

        if (i + 1) % 100 == 0:
            logger.info("Batch progress: %d/%d", i + 1, len(smiles_list))

    return results


def rank_results(
    results: list[dict[str, Any]],
    objective: str = "cmax",
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Rank batch results by a PK objective.

    Args:
        results: Output of batch_predict()
        objective: "cmax", "auc", "t_half", "tmax"
        ascending: If True, lower values rank higher

    Returns:
        Sorted list (errors at the end).
    """
    key_map = {
        "cmax": "cmax_mg_L",
        "auc": "auc_mg_h_L",
        "t_half": "t_half_h",
        "tmax": "tmax_h",
    }
    key = key_map.get(objective, objective)

    valid = [r for r in results if key in r]
    errors = [r for r in results if key not in r]

    valid.sort(key=lambda r: r[key], reverse=not ascending)

    for i, r in enumerate(valid):
        r["rank"] = i + 1

    return valid + errors
