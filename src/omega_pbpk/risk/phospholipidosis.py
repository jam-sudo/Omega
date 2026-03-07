"""Drug-induced phospholipidosis (DIPL) risk prediction (Phase 291).

Predicts risk of lysosomal phospholipid accumulation based on cationic
amphiphilic drug (CAD) structure and physicochemical properties.

References:
    Reasor & Kacew 2001; Hosokawa et al. 2011; Hanumegowda et al. 2010
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PhospholipidosisResult",
    "compute_cad_score",
    "assess_phospholipidosis_risk",
    "screen_phospholipidosis",
]


@dataclass(frozen=True)
class PhospholipidosisResult:
    """Result of phospholipidosis risk assessment."""

    compound_name: str
    smiles: str
    logP: float
    pka_basic: float
    mw: float
    daily_dose_mg: float
    cad_score: float
    dipl_score: float
    risk_category: str      # 'low' / 'moderate' / 'high'
    ec50_lysosome_nM: float
    notes: str


def compute_cad_score(smiles: str, logP: float, pka_basic: float, mw: float) -> float:  # noqa: ARG001
    """Compute cationic amphiphilic drug (CAD) score.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound.
    logP : float
        Lipophilicity (log P octanol/water). Can be any float.
    pka_basic : float
        Most basic pKa of the compound (>= 0).
    mw : float
        Molecular weight (g/mol), must be > 0.

    Returns
    -------
    float
        CAD score in [0, 3]. 0 indicates non-CAD character; higher values
        indicate increasing potential for phospholipidosis.
    """
    if pka_basic < 0:
        raise ValueError(f"pka_basic must be >= 0, got {pka_basic}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # Nitrogen-containing (basic amine indicator) from SMILES
    nitrogen_flag = 1.0 if "n" in smiles.lower() else 0.0

    # Weighted contribution of each CAD property
    logp_contrib = max(0.0, logP - 2.0) * 0.3
    pka_contrib = max(0.0, pka_basic - 6.0) * 0.2
    nitrogen_contrib = nitrogen_flag * 0.5

    raw_score = logp_contrib + pka_contrib + nitrogen_contrib

    # Clamp to [0, 3]
    return float(min(max(raw_score, 0.0), 3.0))


def assess_phospholipidosis_risk(
    compound_name: str,
    smiles: str,
    logP: float,
    pka_basic: float,
    mw: float,
    daily_dose_mg: float,
) -> PhospholipidosisResult:
    """Assess phospholipidosis risk for a single compound.

    Parameters
    ----------
    compound_name : str
        Human-readable compound name.
    smiles : str
        SMILES string.
    logP : float
        Lipophilicity.
    pka_basic : float
        Most basic pKa (>= 0).
    mw : float
        Molecular weight in g/mol (> 0).
    daily_dose_mg : float
        Daily clinical dose in mg (> 0).

    Returns
    -------
    PhospholipidosisResult
    """
    if pka_basic < 0:
        raise ValueError(f"pka_basic must be >= 0, got {pka_basic}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")

    cad_score = compute_cad_score(smiles, logP, pka_basic, mw)

    # Dose factor normalized to 100 mg reference
    dose_factor = math.log10(daily_dose_mg) / math.log10(100.0)

    dipl_score = cad_score * dose_factor

    # Risk classification
    if dipl_score < 0.5:
        risk_category = "low"
    elif dipl_score <= 1.5:
        risk_category = "moderate"
    else:
        risk_category = "high"

    # Lysosomal EC50 approximation (nM): higher CAD score → lower EC50 → more potent
    ec50_lysosome_nM = 10.0 ** (3.0 - cad_score)

    # Build notes
    notes_parts: list[str] = []
    if "n" in smiles.lower():
        notes_parts.append("Nitrogen detected (potential basic amine)")
    if logP > 2.0:
        notes_parts.append(f"Lipophilic (logP={logP:.1f} > 2)")
    if pka_basic > 6.0:
        notes_parts.append(f"Basic compound (pKa={pka_basic:.1f} > 6)")
    if daily_dose_mg >= 100.0:
        notes_parts.append(f"High daily dose ({daily_dose_mg:.0f} mg)")
    if not notes_parts:
        notes_parts.append("No significant CAD risk factors identified")

    notes = "; ".join(notes_parts)

    return PhospholipidosisResult(
        compound_name=compound_name,
        smiles=smiles,
        logP=logP,
        pka_basic=pka_basic,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        cad_score=cad_score,
        dipl_score=dipl_score,
        risk_category=risk_category,
        ec50_lysosome_nM=ec50_lysosome_nM,
        notes=notes,
    )


def screen_phospholipidosis(compounds: list[dict]) -> list[PhospholipidosisResult]:
    """Screen a list of compounds for phospholipidosis risk.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain:
        - "compound_name" (str)
        - "smiles" (str)
        - "logP" (float)
        - "pka_basic" (float)
        - "mw" (float)
        - "daily_dose_mg" (float)

    Returns
    -------
    list[PhospholipidosisResult]
        Results sorted by dipl_score descending (highest risk first).
    """
    results: list[PhospholipidosisResult] = []
    for c in compounds:
        result = assess_phospholipidosis_risk(
            compound_name=c["compound_name"],
            smiles=c["smiles"],
            logP=c["logP"],
            pka_basic=c["pka_basic"],
            mw=c["mw"],
            daily_dose_mg=c["daily_dose_mg"],
        )
        results.append(result)

    results.sort(key=lambda r: r.dipl_score, reverse=True)
    return results
