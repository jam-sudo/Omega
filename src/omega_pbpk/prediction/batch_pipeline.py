"""Batch prediction pipeline: run_full_prediction over a list of SMILES."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from omega_pbpk.prediction.full_pipeline import FullPredictionResult, run_full_prediction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompoundSummary:
    """Per-compound summary extracted from a FullPredictionResult."""

    rank: int
    smiles: str
    drug_name: str
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    cmax_p50: float
    auc_p50: float
    risk_count: int
    overall_risk_level: str
    confidence: str
    adme_confidence: str
    transporter_ddi_flags: tuple[str, ...]
    composite_score: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BatchPredictionResult:
    """Aggregated result for a batch of compounds."""

    compounds: tuple[CompoundSummary, ...]
    n_total: int
    n_succeeded: int
    n_failed: int
    ranking_criterion: str
    best_compound: str
    warnings: tuple[str, ...]


def _score_compound(result: FullPredictionResult) -> float:
    """Compute composite developability score in [0, 100].

    Starts at 100 and subtracts:
    - 15 per risk flag
    - 10 if overall confidence is "low"
    - 5 per transporter DDI flag
    - 20 if t_half_h > 24 h
    - 15 if UQ band cmax_p95/cmax_p5 > 3
    """
    score = 100.0
    score -= result.risk_count * 15
    if result.confidence == "low":
        score -= 10
    score -= len(result.transporter_ddi_flags) * 5
    if result.t_half_h > 24.0:
        score -= 20
    if result.cmax_p5 > 0 and result.cmax_p95 / result.cmax_p5 > 3.0:
        score -= 15
    return max(0.0, min(100.0, score))


def run_batch_prediction(
    smiles_list: list[str],
    dose_mg: float = 100.0,
    route: str = "oral",
    duration_h: float = 24.0,
    n_uq_samples: int = 50,
    ranking: str = "score",
    max_compounds: int = 50,
) -> BatchPredictionResult:
    """Run full prediction pipeline over a list of SMILES strings.

    Args:
        smiles_list: SMILES strings to evaluate (1–max_compounds).
        dose_mg: Dose in milligrams applied to all compounds.
        route: Administration route ('oral' or 'iv').
        duration_h: Simulation duration in hours.
        n_uq_samples: LHS samples for UQ propagation per compound.
        ranking: Sort criterion — 'score' (highest), 'risk' (fewest risks),
                 'auc' (highest), or 'cmax' (lowest, safety-first).
        max_compounds: Hard cap on batch size.

    Returns:
        BatchPredictionResult with ranked CompoundSummary entries.
    """
    if not smiles_list:
        raise ValueError("smiles_list must not be empty")
    if len(smiles_list) > max_compounds:
        raise ValueError(f"Maximum {max_compounds} compounds per batch")
    if ranking not in ("score", "risk", "auc", "cmax"):
        raise ValueError(f"Invalid ranking criterion: {ranking!r}")

    batch_warnings: list[str] = []
    results: list[tuple[FullPredictionResult, float]] = []
    n_failed = 0

    for smi in smiles_list:
        try:
            r = run_full_prediction(smi, dose_mg, route, duration_h, n_uq_samples)
            score = _score_compound(r)
            results.append((r, score))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch: failed for %s: %s", smi[:30], exc)
            batch_warnings.append(f"Failed: {smi[:30]}: {exc}")
            n_failed += 1

    # Sort according to ranking criterion
    if ranking == "score":
        results.sort(key=lambda x: x[1], reverse=True)
    elif ranking == "risk":
        results.sort(key=lambda x: x[0].risk_count)
    elif ranking == "auc":
        results.sort(key=lambda x: x[0].auc0t_mg_h_L, reverse=True)
    elif ranking == "cmax":
        results.sort(key=lambda x: x[0].cmax_mg_L)

    summaries: list[CompoundSummary] = []
    for rank, (r, score) in enumerate(results, 1):
        summaries.append(
            CompoundSummary(
                rank=rank,
                smiles=r.smiles,
                drug_name=r.drug_name,
                cmax_mg_L=r.cmax_mg_L,
                tmax_h=r.tmax_h,
                auc0t_mg_h_L=r.auc0t_mg_h_L,
                t_half_h=r.t_half_h,
                cmax_p50=r.cmax_p50,
                auc_p50=r.auc_p50,
                risk_count=r.risk_count,
                overall_risk_level=r.overall_risk_level,
                confidence=r.confidence,
                adme_confidence=r.adme_confidence,
                transporter_ddi_flags=tuple(r.transporter_ddi_flags),
                composite_score=score,
                warnings=tuple(r.warnings),
            )
        )

    best = summaries[0].drug_name if summaries else "none"

    return BatchPredictionResult(
        compounds=tuple(summaries),
        n_total=len(smiles_list),
        n_succeeded=len(results),
        n_failed=n_failed,
        ranking_criterion=ranking,
        best_compound=best,
        warnings=tuple(batch_warnings),
    )


__all__ = ["BatchPredictionResult", "CompoundSummary", "run_batch_prediction"]
