"""
Drug Protein Adduct Risk Assessment
====================================
Assesses risk of covalent protein adduct formation from reactive metabolites.
Combines structural alerts with dose and metabolic activation burden.

References:
    Nakayama et al. Drug Metab Dispos 2009
    Thompson et al. 2016
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Alert weights
_ALERT_WEIGHTS: dict[str, float] = {
    "furan": 2.0,
    "aniline": 1.5,
    "michael_acceptor": 1.5,
    "quinone_methide": 1.5,
    "epoxide_risk": 0.5,
}

# Mitigation recommendations per alert
_MITIGATIONS: dict[str, str] = {
    "furan": "Avoid furan-containing scaffolds; consider bioisosteric replacement.",
    "aniline": "Block para-position of aniline; evaluate N-acetylation metabolite.",
    "michael_acceptor": "Reduce electrophilicity; add steric protection to Michael acceptor.",
    "quinone_methide": "Block ortho/para phenol oxidation; consider fluorine substitution.",
    "epoxide_risk": (
        "Consider epoxide hydrolase stability assay; avoid aromatic epoxide precursors."
    ),
}


@dataclass
class ProteinAdductResult:
    """Result of protein adduct risk assessment."""

    compound_name: str
    smiles: str
    mw: float
    daily_dose_mg: float
    fu: float
    fm_cyp: float
    logP: float
    alerts_triggered: list[str] = field(default_factory=list)
    n_alerts: int = 0
    alert_score: float = 0.0
    adduct_burden_score: float = 0.0
    risk_category: str = "low"
    mitigation: list[str] = field(default_factory=list)
    notes: str = ""


def _detect_alerts(smiles: str, mw: float, logP: float) -> list[str]:
    """Detect structural alerts via SMILES substring matching."""
    triggered: list[str] = []
    s = smiles.lower()

    # Furan ring
    if "c1ccoc1" in s or "o1cccc1" in s:
        triggered.append("furan")

    # Aniline (aromatic amine)
    if "nc1ccc" in s or "cn" in s:
        triggered.append("aniline")

    # Michael acceptor (alpha,beta-unsaturated carbonyl or amino)
    if "c=cc=o" in s or "c=ccn" in s:
        triggered.append("michael_acceptor")

    # Quinone methide precursor
    if "c(=o)c=c" in s:
        triggered.append("quinone_methide")

    # Epoxide risk (low MW, high logP, aromatic carbon present)
    if mw < 300 and logP > 2 and "c" in s:
        triggered.append("epoxide_risk")

    return triggered


def assess_protein_adduct_risk(
    compound_name: str,
    smiles: str,
    mw: float,
    daily_dose_mg: float,
    fu: float,
    fm_cyp: float,
    logP: float,
) -> ProteinAdductResult:
    """
    Assess the risk of covalent protein adduct formation.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    smiles : str
        SMILES string of the compound.
    mw : float
        Molecular weight (g/mol). Must be > 0.
    daily_dose_mg : float
        Total daily dose in mg. Must be > 0.
    fu : float
        Unbound fraction in plasma (0, 1]. Must be in (0, 1].
    fm_cyp : float
        Fraction metabolized by CYP enzymes [0, 1]. Must be in [0, 1].
    logP : float
        Logarithm of octanol-water partition coefficient.

    Returns
    -------
    ProteinAdductResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    if not (0 < fu <= 1):
        raise ValueError("fu must be in (0, 1]")
    if not (0 <= fm_cyp <= 1):
        raise ValueError("fm_cyp must be in [0, 1]")

    alerts = _detect_alerts(smiles, mw, logP)
    n_alerts = len(alerts)
    alert_score = sum(_ALERT_WEIGHTS[a] for a in alerts)

    # Adduct burden score = alert_score * log10(daily_dose_mg) * fu * fm_cyp
    if alert_score > 0:
        adduct_burden_score = alert_score * math.log10(daily_dose_mg) * fu * fm_cyp
    else:
        adduct_burden_score = 0.0

    # Clamp to 0 in case log10 yields negative for dose < 1 mg
    adduct_burden_score = max(0.0, adduct_burden_score)

    # Risk classification
    if adduct_burden_score < 1.0:
        risk_category = "low"
    elif adduct_burden_score <= 3.0:
        risk_category = "moderate"
    else:
        risk_category = "high"

    mitigation = [_MITIGATIONS[a] for a in alerts]

    notes_parts: list[str] = []
    if n_alerts == 0:
        notes_parts.append("No structural alerts detected.")
    else:
        notes_parts.append(f"{n_alerts} structural alert(s) detected: {', '.join(alerts)}.")
    notes_parts.append(f"Adduct burden score = {adduct_burden_score:.3f} ({risk_category} risk).")

    return ProteinAdductResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        fu=fu,
        fm_cyp=fm_cyp,
        logP=logP,
        alerts_triggered=alerts,
        n_alerts=n_alerts,
        alert_score=alert_score,
        adduct_burden_score=adduct_burden_score,
        risk_category=risk_category,
        mitigation=mitigation,
        notes=" ".join(notes_parts),
    )


def rank_compounds(compounds: list[dict]) -> list[ProteinAdductResult]:
    """
    Rank a list of compounds by adduct burden score (descending).

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain: name, smiles, mw, daily_dose_mg, fu, fm_cyp, logP.

    Returns
    -------
    list[ProteinAdductResult]
        Sorted by adduct_burden_score descending.
    """
    results: list[ProteinAdductResult] = []
    for c in compounds:
        result = assess_protein_adduct_risk(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            daily_dose_mg=c["daily_dose_mg"],
            fu=c["fu"],
            fm_cyp=c["fm_cyp"],
            logP=c["logP"],
        )
        results.append(result)

    results.sort(key=lambda r: r.adduct_burden_score, reverse=True)
    return results
