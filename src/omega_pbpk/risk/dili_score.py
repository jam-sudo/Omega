"""Drug-Induced Liver Injury (DILI) risk scoring.

Mechanistic DILI risk scoring using physicochemical and structural
properties to generate a composite score (0–100).

References
----------
- Chen M et al., Drug Discov Today. 2016;21:648–653
- Xu JJ et al., Chem Biol Interact. 2008;176:69–76
- Stepan AF et al., Chem Res Toxicol. 2011;24:1345–1410
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DILIResult",
    "score_dili",
    "screen_dili",
]


@dataclass(frozen=True)
class DILIResult:
    """DILI risk assessment result."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    daily_dose_mg: float
    dili_score: float
    dili_category: str
    dose_dependency: str
    reactive_metabolite_risk: bool
    mitochondrial_liability: bool
    bile_salt_inhibition: bool
    physicochemical_risk_factors: list[str]
    notes: str


def score_dili(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    daily_dose_mg: float,
    clint_uL_per_min_per_mg: float = 10.0,
    fup: float = 0.1,
    is_mitochondrial_toxin: bool = False,
    bile_salt_export_inhibitor: bool = False,
) -> DILIResult:
    """Score DILI risk for a compound.

    Parameters
    ----------
    compound_name : str
        Compound identifier.
    smiles : str
        SMILES string.
    mw : float
        Molecular weight (Da).
    logP : float
        Octanol-water partition coefficient.
    daily_dose_mg : float
        Daily dose in mg.
    clint_uL_per_min_per_mg : float
        Intrinsic clearance (default 10.0).
    fup : float
        Fraction unbound in plasma (default 0.1).
    is_mitochondrial_toxin : bool
        Whether compound is a mitochondrial toxin.
    bile_salt_export_inhibitor : bool
        Whether compound inhibits bile salt export pump.

    Returns
    -------
    DILIResult
    """
    if mw <= 0:
        raise ValueError("mw must be positive")
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be positive")
    if not smiles or not smiles.strip():
        raise ValueError("smiles must not be empty")

    score = 0.0
    risk_factors: list[str] = []

    # Dose-based scoring
    if daily_dose_mg > 100:
        score += 20
        risk_factors.append(f"High daily dose (>{daily_dose_mg:.0f} mg)")
    elif daily_dose_mg >= 10:
        score += 10
        risk_factors.append(f"Moderate daily dose ({daily_dose_mg:.0f} mg)")

    # logP scoring
    if logP > 3:
        score += 15
        risk_factors.append(f"High lipophilicity (logP={logP:.1f})")

    # MW scoring
    if mw < 300:
        score += 10
        risk_factors.append(f"Small molecule (MW={mw:.0f} Da)")

    # Reactive metabolite risk
    reactive_metabolite_risk = logP > 2 and mw < 400 and ("N" in smiles or "S" in smiles)
    if reactive_metabolite_risk:
        score += 20
        risk_factors.append("Reactive metabolite risk (logP>2, MW<400, N/S in structure)")

    # Mitochondrial toxicity
    if is_mitochondrial_toxin:
        score += 25
        risk_factors.append("Mitochondrial toxin")

    # Bile salt export pump inhibition
    if bile_salt_export_inhibitor:
        score += 20
        risk_factors.append("Bile salt export pump inhibitor")

    # Clamp score
    score = max(0.0, min(100.0, score))

    # Category
    if score > 75:
        dili_category = "high"
    elif score > 50:
        dili_category = "moderate"
    elif score >= 25:
        dili_category = "low"
    else:
        dili_category = "no_concern"

    # Dose dependency
    dose_dependency = "high_dose_only" if daily_dose_mg > 100 else "all_doses"

    # Notes
    note_parts: list[str] = []
    if risk_factors:
        note_parts.append(f"{len(risk_factors)} risk factor(s) identified")
    if dili_category in ("high", "moderate"):
        note_parts.append("Consider hepatotoxicity monitoring in clinical trials")
    notes = "; ".join(note_parts) if note_parts else "No significant DILI risk identified"

    return DILIResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        daily_dose_mg=daily_dose_mg,
        dili_score=score,
        dili_category=dili_category,
        dose_dependency=dose_dependency,
        reactive_metabolite_risk=reactive_metabolite_risk,
        mitochondrial_liability=is_mitochondrial_toxin,
        bile_salt_inhibition=bile_salt_export_inhibitor,
        physicochemical_risk_factors=risk_factors,
        notes=notes,
    )


def screen_dili(compounds: list[dict]) -> list[DILIResult]:
    """Screen multiple compounds for DILI risk.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain: compound_name, smiles, mw, logP, daily_dose_mg.
        Optional keys: clint_uL_per_min_per_mg, fup, is_mitochondrial_toxin,
        bile_salt_export_inhibitor.

    Returns
    -------
    list[DILIResult]
        Sorted by dili_score descending.
    """
    if not compounds:
        raise ValueError("compounds list must not be empty")
    results = [
        score_dili(
            compound_name=c["compound_name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            daily_dose_mg=c["daily_dose_mg"],
            clint_uL_per_min_per_mg=c.get("clint_uL_per_min_per_mg", 10.0),
            fup=c.get("fup", 0.1),
            is_mitochondrial_toxin=c.get("is_mitochondrial_toxin", False),
            bile_salt_export_inhibitor=c.get("bile_salt_export_inhibitor", False),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.dili_score, reverse=True)
