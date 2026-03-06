"""Protein binding displacement DDI module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DisplacementResult",
    "assess_protein_displacement",
    "displacement_sensitivity",
]


@dataclass
class DisplacementResult:
    victim_drug: str
    perpetrator_drug: str
    baseline_fup: float
    displaced_fup: float
    fup_ratio: float
    cl_ratio: float
    auc_ratio: float
    is_clinically_significant: bool
    binding_site: str
    mechanism: str
    recommendation: str


def _competitive_displacement(
    fup_baseline: float,
    ki_app: float,
    competitor_conc_uM: float,
    total_binding_sites: float = 1.0,
) -> float:
    displaced_fup = fup_baseline * (1.0 + competitor_conc_uM / ki_app)
    return float(min(1.0, displaced_fup * total_binding_sites))


def assess_protein_displacement(
    victim_drug: str,
    perpetrator_drug: str,
    baseline_fup: float,
    ki_app_uM: float,
    competitor_conc_uM: float,
    logP: float = 2.0,
    is_nti: bool = False,
    binding_site: str = "albumin_site_I",
) -> DisplacementResult:
    if baseline_fup <= 0:
        raise ValueError("baseline_fup must be > 0")
    if ki_app_uM <= 0:
        raise ValueError("ki_app_uM must be > 0")
    if competitor_conc_uM < 0:
        raise ValueError("competitor_conc_uM must be >= 0")

    displaced_fup = _competitive_displacement(baseline_fup, ki_app_uM, competitor_conc_uM)
    fup_ratio = displaced_fup / baseline_fup
    cl_ratio = fup_ratio
    auc_ratio = 1.0 / fup_ratio if fup_ratio != 0.0 else np.inf

    is_clinically_significant = bool(fup_ratio > 1.5 and is_nti)
    mechanism = "competitive displacement from protein binding site"

    if is_clinically_significant:
        recommendation = "Monitor NTI drug levels"
    else:
        recommendation = "Unlikely clinical significance"

    return DisplacementResult(
        victim_drug=victim_drug,
        perpetrator_drug=perpetrator_drug,
        baseline_fup=baseline_fup,
        displaced_fup=displaced_fup,
        fup_ratio=fup_ratio,
        cl_ratio=cl_ratio,
        auc_ratio=auc_ratio,
        is_clinically_significant=is_clinically_significant,
        binding_site=binding_site,
        mechanism=mechanism,
        recommendation=recommendation,
    )


def displacement_sensitivity(
    victim_drug: str,
    baseline_fup: float,
    ki_app_uM: float,
    conc_range: list[float],
) -> list[DisplacementResult]:
    results = []
    for conc in conc_range:
        result = assess_protein_displacement(
            victim_drug=victim_drug,
            perpetrator_drug="perpetrator",
            baseline_fup=baseline_fup,
            ki_app_uM=ki_app_uM,
            competitor_conc_uM=conc,
        )
        results.append(result)
    return results
