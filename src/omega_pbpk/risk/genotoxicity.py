"""Genotoxicity prediction: Ames test surrogate, micronucleus risk, and structural alerts.

ICH S2(R1) compliant assessment framework.

References
----------
Kazius 2005 (J Med Chem), Ashby & Tennant 1988 (Mutat Res),
ICH S2(R1) 2011, OECD TG 471 (Ames test).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GenotoxAlert",
    "GenotoxResult",
    "assess_genotoxicity",
    "screen_library",
    "get_genotoxic_alerts_summary",
]

# ---------------------------------------------------------------------------
# Structural alert definitions (Ashby-Tennant + extended Kazius 2005)
# ---------------------------------------------------------------------------

ALERT_PATTERNS: dict[str, dict] = {
    "nitro_group": {
        "description": "Nitro group (-NO2): strong DNA-reactive alert",
        "weight": 2.0,
    },
    "nitroso": {
        "description": "Nitroso group (N=O): direct-acting mutagen alert",
        "weight": 1.5,
    },
    "epoxide": {
        "description": "Epoxide (3-membered oxirane ring): alkylating agent",
        "weight": 1.5,
    },
    "aziridine": {
        "description": "Aziridine (3-membered nitrogen ring): alkylating agent",
        "weight": 1.5,
    },
    "aliphatic_halide": {
        "description": "Aliphatic halide (C-Br, C-Cl, C-I): alkylating potential",
        "weight": 1.5,
    },
    "primary_amine_aromatic": {
        "description": "Aromatic amine: metabolic activation to reactive nitrenium ion",
        "weight": 1.0,
    },
    "quinone": {
        "description": "Quinone moiety: redox cycling and oxidative DNA damage",
        "weight": 0.8,
    },
    "michael_acceptor": {
        "description": "Michael acceptor (alpha,beta-unsaturated carbonyl): thiol-reactive",
        "weight": 1.0,
    },
    "aldehyde": {
        "description": "Aldehyde group: Schiff-base formation with DNA bases",
        "weight": 0.8,
    },
    "hydrazine": {
        "description": "Hydrazine (N-N bond): metabolic activation to diazonium",
        "weight": 1.2,
    },
    "low_mw_reactive": {
        "description": "Low molecular weight (<100): small reactive molecule concern",
        "weight": 0.5,
    },
    "high_logP_lipophilic": {
        "description": "High lipophilicity (logP>5): membrane accumulation concern",
        "weight": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenotoxAlert:
    """Single genotoxicity structural alert result."""

    name: str
    triggered: bool
    weight: float
    description: str


@dataclass(frozen=True)
class GenotoxResult:
    """Complete genotoxicity assessment result."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    tpsa: float
    alerts: list[GenotoxAlert]
    n_alerts: int
    alert_score: float
    ames_prediction: str
    ames_probability: float
    micronucleus_risk: str
    ichs2r1_flag: str
    structural_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_smiles_alert(smiles: str, alert_name: str) -> bool:  # noqa: C901
    """Check whether a SMILES string triggers a given structural alert."""
    s = smiles  # keep original case for pattern matching
    su = smiles.upper()

    if alert_name == "nitro_group":
        return "NO2" in su or "[N+](=O)[O-]" in s

    if alert_name == "nitroso":
        # N=O present but NOT part of a nitro group
        if "N=O" in s:
            # Exclude if it's part of [N+](=O)[O-] or adjacent to another O
            idx = s.find("N=O")
            while idx != -1:
                # Check it's not part of NO2 pattern
                after = s[idx + 3 :] if idx + 3 < len(s) else ""
                if not (after.startswith("[O-]") or after.startswith("O")):
                    return True
                idx = s.find("N=O", idx + 1)
        return False

    if alert_name == "epoxide":
        return "C1OC1" in s

    if alert_name == "aziridine":
        return "C1NC1" in s

    if alert_name == "aliphatic_halide":
        return "CBr" in s or "CCl" in s or "CI" in s

    if alert_name == "primary_amine_aromatic":
        return "cN" in s or "c-N" in s

    if alert_name == "quinone":
        return "C(=O)c" in s or "O=Cc" in s

    if alert_name == "michael_acceptor":
        return "C=CC=O" in s or "C=CN" in s

    if alert_name == "aldehyde":
        return "CHO" in s or "[CH]=O" in s

    if alert_name == "hydrazine":
        return "NN" in s

    return False


def _check_physchem_alert(mw: float, logP: float, alert_name: str) -> bool:
    """Check physicochemical property-based alerts."""
    if alert_name == "low_mw_reactive":
        return mw < 100
    if alert_name == "high_logP_lipophilic":
        return logP > 5
    return False


_SMILES_ALERTS = {
    "nitro_group",
    "nitroso",
    "epoxide",
    "aziridine",
    "aliphatic_halide",
    "primary_amine_aromatic",
    "quinone",
    "michael_acceptor",
    "aldehyde",
    "hydrazine",
}

_PHYSCHEM_ALERTS = {"low_mw_reactive", "high_logP_lipophilic"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_genotoxicity(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    tpsa: float = 50.0,
) -> GenotoxResult:
    """Assess genotoxicity risk using structural alerts and physicochemical properties.

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
    tpsa : float
        Topological polar surface area (default 50).

    Returns
    -------
    GenotoxResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string must not be empty")
    if mw < 0:
        raise ValueError("Molecular weight must be non-negative")

    alerts: list[GenotoxAlert] = []

    for name, info in ALERT_PATTERNS.items():
        if name in _SMILES_ALERTS:
            triggered = _check_smiles_alert(smiles, name)
        elif name in _PHYSCHEM_ALERTS:
            triggered = _check_physchem_alert(mw, logP, name)
        else:
            triggered = False

        alerts.append(
            GenotoxAlert(
                name=name,
                triggered=triggered,
                weight=info["weight"],
                description=info["description"],
            )
        )

    triggered_alerts = [a for a in alerts if a.triggered]
    n_alerts = len(triggered_alerts)
    alert_score = sum(a.weight for a in triggered_alerts)

    # Ames prediction
    if alert_score >= 2.0:
        ames_prediction = "positive"
        structural_class = "genotoxic"
    elif alert_score >= 1.0:
        ames_prediction = "inconclusive"
        structural_class = "borderline"
    else:
        ames_prediction = "negative"
        structural_class = "non-genotoxic"

    ames_probability = min(0.95, alert_score / 4.0)

    # Micronucleus risk
    if alert_score >= 2.0 and mw > 150:
        micronucleus_risk = "high"
    elif alert_score >= 1.0:
        micronucleus_risk = "moderate"
    else:
        micronucleus_risk = "low"

    # ICH S2(R1) flag
    if structural_class == "genotoxic":
        ichs2r1_flag = "concern"
    elif structural_class == "non-genotoxic":
        ichs2r1_flag = "no_concern"
    else:
        ichs2r1_flag = "inconclusive"

    # Notes
    note_parts: list[str] = []
    if n_alerts > 0:
        triggered_names = [a.name for a in triggered_alerts]
        note_parts.append(f"Triggered alerts: {', '.join(triggered_names)}")
    if ames_prediction == "positive":
        note_parts.append("ICH S2(R1) recommends in-vitro and in-vivo follow-up")
    notes = "; ".join(note_parts) if note_parts else "No structural alerts detected"

    return GenotoxResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        tpsa=tpsa,
        alerts=alerts,
        n_alerts=n_alerts,
        alert_score=alert_score,
        ames_prediction=ames_prediction,
        ames_probability=ames_probability,
        micronucleus_risk=micronucleus_risk,
        ichs2r1_flag=ichs2r1_flag,
        structural_class=structural_class,
        notes=notes,
    )


def screen_library(compounds: list[dict]) -> list[GenotoxResult]:
    """Screen a library of compounds for genotoxicity.

    Parameters
    ----------
    compounds : list[dict]
        Each dict: {"name": ..., "smiles": ..., "mw": ..., "logP": ..., "tpsa": ...}

    Returns
    -------
    list[GenotoxResult]
        Sorted by alert_score descending.
    """
    results = [
        assess_genotoxicity(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            tpsa=c.get("tpsa", 50.0),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.alert_score, reverse=True)


def get_genotoxic_alerts_summary() -> dict[str, str]:
    """Return dict of alert_name -> description for all implemented alerts."""
    return {name: info["description"] for name, info in ALERT_PATTERNS.items()}
