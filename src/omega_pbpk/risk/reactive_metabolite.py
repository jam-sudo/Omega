"""Reactive metabolite prediction and bioactivation risk assessment.

References
----------
Thompson 2016 (Drug Metab Dispos), Stepan 2011 (J Med Chem),
FDA 2020 MIST guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ReactiveAlert",
    "ReactiveMetaboliteResult",
    "assess_reactive_metabolite",
    "screen_reactive_metabolites",
]

# ---------------------------------------------------------------------------
# Structural alert definitions
# ---------------------------------------------------------------------------

ALERT_PATTERNS: dict[str, dict] = {
    "furan": {
        "description": "Furan ring: CYP-mediated epoxidation to cis-enedial",
        "weight": 2.0,
        "mechanism": "CYP-mediated",
    },
    "thiophene": {
        "description": "Thiophene ring: CYP-mediated S-oxidation to reactive epoxide",
        "weight": 1.5,
        "mechanism": "CYP-mediated",
    },
    "aniline": {
        "description": "Aromatic amine: N-hydroxylation to reactive nitrenium ion",
        "weight": 1.5,
        "mechanism": "CYP-mediated",
    },
    "hydrazine": {
        "description": "Hydrazine (N-N bond): oxidation to diazonium reactive species",
        "weight": 2.0,
        "mechanism": "CYP-mediated",
    },
    "nitro_aromatic": {
        "description": "Nitroaromatic: nitroreduction to reactive nitroso and hydroxylamine",
        "weight": 2.0,
        "mechanism": "CYP-mediated",
    },
    "michael_acceptor": {
        "description": "Michael acceptor: electrophilic addition to protein thiols",
        "weight": 1.5,
        "mechanism": "direct",
    },
    "quinone_methide": {
        "description": "Quinone methide precursor: redox cycling and protein adducts",
        "weight": 1.5,
        "mechanism": "CYP-mediated",
    },
    "acyl_halide": {
        "description": "Acyl halide: direct acylation of nucleophilic residues",
        "weight": 2.0,
        "mechanism": "direct",
    },
    "terminal_alkyne": {
        "description": "Terminal alkyne: CYP-mediated oxidation to ketene intermediate",
        "weight": 1.0,
        "mechanism": "CYP-mediated",
    },
    "alpha_beta_unsaturated_ester": {
        "description": "Alpha-beta unsaturated ester: Michael addition to GSH",
        "weight": 1.2,
        "mechanism": "direct",
    },
    "chloroethyl": {
        "description": "Chloroethyl group: potential aziridinium formation",
        "weight": 1.0,
        "mechanism": "direct",
    },
    "epoxide_forming": {
        "description": "Low-MW lipophilic aromatic: CYP-mediated epoxide formation risk",
        "weight": 0.5,
        "mechanism": "CYP-mediated",
    },
    "high_dose": {
        "description": "High daily dose (>100 mg): increased reactive metabolite burden",
        "weight": 0.5,
        "mechanism": "unknown",
    },
}

# Mitigation suggestion map
_MITIGATION_MAP: dict[str, str] = {
    "furan": "Consider bioisosteric replacement of furan (e.g., cyclopropyl or pyrazole)",
    "aniline": "Mask aniline with electron-withdrawing group or para-substitution",
    "michael_acceptor": "Reduce electrophilicity with steric shielding",
    "high_dose": "Dose reduction would decrease DILI risk",
}

_SMILES_ALERTS = {
    "furan",
    "thiophene",
    "aniline",
    "hydrazine",
    "nitro_aromatic",
    "michael_acceptor",
    "quinone_methide",
    "acyl_halide",
    "terminal_alkyne",
    "alpha_beta_unsaturated_ester",
    "chloroethyl",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactiveAlert:
    """Single reactive metabolite structural alert result."""

    name: str
    triggered: bool
    weight: float
    description: str
    mechanism: str


@dataclass(frozen=True)
class ReactiveMetaboliteResult:
    """Complete reactive metabolite assessment result."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    daily_dose_mg: float
    alerts: list[ReactiveAlert]
    n_alerts: int
    alert_score: float
    bioactivation_prediction: str
    dili_risk: str
    idiosyncratic_risk: str
    dose_burden_score: float
    covalent_binding_prediction: str
    structural_class: str
    mitigation_suggestions: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_smiles_alert(smiles: str, alert_name: str) -> bool:  # noqa: C901
    """Check whether a SMILES string triggers a given structural alert."""
    s = smiles

    if alert_name == "furan":
        return "c1ccoc1" in s or ("o" in s and "c" in s and len(s) < 30)

    if alert_name == "thiophene":
        return "c1ccsc1" in s or ("S" in s and "c" in s)

    if alert_name == "aniline":
        return "Nc1ccccc1" in s or "cN" in s

    if alert_name == "hydrazine":
        return "NN" in s

    if alert_name == "nitro_aromatic":
        return "cN(=O)=O" in s or "c[N+](=O)[O-]" in s or ("[N+](=O)[O-]" in s and "c" in s)

    if alert_name == "michael_acceptor":
        return "C=CC=O" in s or "C=CCN" in s

    if alert_name == "quinone_methide":
        return "C(=O)c=C" in s

    if alert_name == "acyl_halide":
        return "C(=O)F" in s or "C(=O)Cl" in s

    if alert_name == "terminal_alkyne":
        return "C#CH" in s

    if alert_name == "alpha_beta_unsaturated_ester":
        return "C=CC(=O)O" in s

    if alert_name == "chloroethyl":
        return "CCl" in s

    return False


def _check_physchem_alert(
    mw: float, logP: float, daily_dose_mg: float, smiles: str, alert_name: str
) -> bool:
    """Check physicochemical property-based alerts."""
    if alert_name == "epoxide_forming":
        return mw < 300 and logP > 2 and "c" in smiles
    if alert_name == "high_dose":
        return daily_dose_mg > 100
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_reactive_metabolite(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    daily_dose_mg: float = 50.0,
) -> ReactiveMetaboliteResult:
    """Assess reactive metabolite risk using structural alerts and physicochemical properties.

    Parameters
    ----------
    compound_name : str
        Name or identifier of the compound.
    smiles : str
        SMILES representation of the compound.
    mw : float
        Molecular weight (Da).
    logP : float
        Octanol-water partition coefficient.
    daily_dose_mg : float
        Daily dose in mg (default 50.0).

    Returns
    -------
    ReactiveMetaboliteResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string must not be empty")
    if mw <= 0:
        raise ValueError("Molecular weight must be positive")

    alerts: list[ReactiveAlert] = []

    for name, info in ALERT_PATTERNS.items():
        if name in _SMILES_ALERTS:
            triggered = _check_smiles_alert(smiles, name)
        elif name == "epoxide_forming":
            triggered = _check_physchem_alert(mw, logP, daily_dose_mg, smiles, name)
        elif name == "high_dose":
            triggered = _check_physchem_alert(mw, logP, daily_dose_mg, smiles, name)
        else:
            triggered = False

        alerts.append(
            ReactiveAlert(
                name=name,
                triggered=triggered,
                weight=info["weight"],
                description=info["description"],
                mechanism=info["mechanism"],
            )
        )

    triggered_alerts = [a for a in alerts if a.triggered]
    n_alerts = len(triggered_alerts)
    alert_score = sum(a.weight for a in triggered_alerts)

    # Bioactivation prediction
    if alert_score >= 2.0:
        bioactivation_prediction = "likely"
    elif alert_score >= 1.0:
        bioactivation_prediction = "possible"
    else:
        bioactivation_prediction = "unlikely"

    # DILI risk
    if alert_score >= 2 and daily_dose_mg >= 100:
        dili_risk = "high"
    elif alert_score >= 1:
        dili_risk = "moderate"
    else:
        dili_risk = "low"

    # Idiosyncratic risk
    if alert_score >= 3:
        idiosyncratic_risk = "high"
    elif alert_score >= 1.5:
        idiosyncratic_risk = "moderate"
    else:
        idiosyncratic_risk = "low"

    # Dose burden score
    dose_burden_score = alert_score * math.log10(max(1, daily_dose_mg))

    # Covalent binding prediction
    covalent_binding_prediction = bioactivation_prediction

    # Structural class
    if alert_score >= 2.5:
        structural_class = "high_concern"
    elif alert_score >= 1.0:
        structural_class = "moderate_concern"
    else:
        structural_class = "low_concern"

    # Mitigation suggestions
    mitigation_suggestions: list[str] = []
    triggered_names = {a.name for a in triggered_alerts}
    for alert_name, suggestion in _MITIGATION_MAP.items():
        if alert_name in triggered_names:
            mitigation_suggestions.append(suggestion)
    if not mitigation_suggestions:
        mitigation_suggestions = ["No structural interventions needed"]

    # Notes
    note_parts: list[str] = []
    if n_alerts > 0:
        note_parts.append(f"Triggered alerts: {', '.join(sorted(triggered_names))}")
    if bioactivation_prediction == "likely":
        note_parts.append("Recommend GSH trapping studies and covalent binding assessment")
    notes = "; ".join(note_parts) if note_parts else "No reactive metabolite alerts detected"

    return ReactiveMetaboliteResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        daily_dose_mg=daily_dose_mg,
        alerts=alerts,
        n_alerts=n_alerts,
        alert_score=alert_score,
        bioactivation_prediction=bioactivation_prediction,
        dili_risk=dili_risk,
        idiosyncratic_risk=idiosyncratic_risk,
        dose_burden_score=dose_burden_score,
        covalent_binding_prediction=covalent_binding_prediction,
        structural_class=structural_class,
        mitigation_suggestions=mitigation_suggestions,
        notes=notes,
    )


def screen_reactive_metabolites(
    compounds: list[dict],
) -> list[ReactiveMetaboliteResult]:
    """Screen a list of compounds for reactive metabolite risk.

    Parameters
    ----------
    compounds : list[dict]
        Each dict: {"name": ..., "smiles": ..., "mw": ..., "logP": ..., "daily_dose_mg": ...}

    Returns
    -------
    list[ReactiveMetaboliteResult]
        Sorted by dose_burden_score descending.
    """
    results = [
        assess_reactive_metabolite(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            daily_dose_mg=c.get("daily_dose_mg", 50.0),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.dose_burden_score, reverse=True)
