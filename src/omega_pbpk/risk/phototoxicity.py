"""Phototoxicity prediction and assessment for drug photosensitivity.

References
----------
ICH S10 (2014), EMA guideline on photosafety,
Onoue 2011 (DMPK), Monteiro 2020 (Toxicol In Vitro).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PhototoxAlert",
    "PhototoxResult",
    "assess_phototoxicity",
    "screen_phototoxicity",
    "get_phototoxic_alerts_summary",
]

# ---------------------------------------------------------------------------
# Structural / physicochemical alert definitions
# ---------------------------------------------------------------------------

ALERT_PATTERNS: dict[str, dict] = {
    "quinolone": {
        "description": "Quinolone / fluoroquinolone core: strong photosensitiser",
        "weight": 2.0,
    },
    "phenothiazine": {
        "description": "Phenothiazine tricyclic (S + N + aromatic): photoreactive",
        "weight": 1.5,
    },
    "tetracycline_core": {
        "description": "Tetracycline-like polycyclic (MW>400, multiple O): phototoxic",
        "weight": 1.5,
    },
    "psoralen": {
        "description": "Psoralen / furanocoumarin: potent photosensitiser (PUVA therapy)",
        "weight": 2.0,
    },
    "anthracene": {
        "description": "Anthracene triple-ring system: UV-absorbing photosensitiser",
        "weight": 1.5,
    },
    "porphyrin": {
        "description": "Porphyrin-like N-rich aromatic (>=4 N): photodynamic agent",
        "weight": 1.5,
    },
    "naphthalene": {
        "description": "Naphthalene bicyclic aromatic: weak UV absorber",
        "weight": 1.0,
    },
    "ketone_aromatic": {
        "description": "Aromatic ketone (c-C(=O)): triplet-state photoreactivity",
        "weight": 0.8,
    },
    "halogenated_aromatic": {
        "description": "Halogenated aromatic (cF, cCl, cBr): heavy-atom effect enhances ISC",
        "weight": 0.8,
    },
    "furan_aromatic": {
        "description": "Furan in aromatic context: photooxidation potential",
        "weight": 1.0,
    },
    "high_logP": {
        "description": "High lipophilicity (logP>4): skin/tissue accumulation",
        "weight": 0.5,
    },
    "low_mw_aromatic": {
        "description": "Low-MW aromatic (150<MW<400): skin-penetrant chromophore",
        "weight": 0.3,
    },
}

# Partition alerts into SMILES-based vs physicochemical
_SMILES_ALERTS = {
    "quinolone",
    "phenothiazine",
    "tetracycline_core",
    "psoralen",
    "anthracene",
    "porphyrin",
    "naphthalene",
    "ketone_aromatic",
    "halogenated_aromatic",
    "furan_aromatic",
}

_PHYSCHEM_ALERTS = {"high_logP", "low_mw_aromatic"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhototoxAlert:
    """Single phototoxicity structural alert result."""

    name: str
    triggered: bool
    weight: float
    description: str


@dataclass(frozen=True)
class PhototoxResult:
    """Complete phototoxicity assessment result."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    alerts: list[PhototoxAlert]
    n_alerts: int
    alert_score: float
    molar_absorptivity_proxy: float
    phototoxicity_prediction: str
    phototoxicity_probability: float
    skin_risk: str
    eye_risk: str
    ich_s10_flag: str
    photo_dri: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_smiles_alert(smiles: str, alert_name: str, mw: float) -> bool:  # noqa: C901
    """Check whether a SMILES string triggers a given phototoxicity alert."""
    s = smiles

    if alert_name == "quinolone":
        return ("c1ccc2ncccc2c1" in s) or ("C(=O)Nc" in s)

    if alert_name == "phenothiazine":
        return ("c1ccc2Sc3ccccc3Nc2c1" in s) or ("S" in s and "N" in s and "c" in s)

    if alert_name == "tetracycline_core":
        o_count = s.count("O") + s.count("o")
        return mw > 400 and o_count >= 3

    if alert_name == "psoralen":
        return ("c1coc2ccoc2c1" in s) or ("o" in s and "c" in s)

    if alert_name == "anthracene":
        return "c1ccc2cc3ccccc3cc2c1" in s

    if alert_name == "porphyrin":
        n_aromatic = sum(1 for i, ch in enumerate(s) if ch in ("n",) and i > 0)
        n_upper = s.count("N")
        return (n_aromatic + n_upper) >= 4 and "c" in s

    if alert_name == "naphthalene":
        return "c1ccc2ccccc2c1" in s

    if alert_name == "ketone_aromatic":
        return "c-C(=O)" in s or ("C(=O)" in s and "c" in s)

    if alert_name == "halogenated_aromatic":
        return ("cF" in s or "cCl" in s or "cBr" in s) or (
            "c" in s and ("F" in s or "Cl" in s or "Br" in s)
        )

    if alert_name == "furan_aromatic":
        return "c1ccoc1" in s or ("o" in s and "c" in s)

    return False


def _check_physchem_alert(mw: float, logP: float, alert_name: str, smiles: str) -> bool:
    """Check physicochemical property-based alerts."""
    if alert_name == "high_logP":
        return logP > 4
    if alert_name == "low_mw_aromatic":
        return 150 < mw < 400 and "c" in smiles
    return False


def _estimate_molar_absorptivity(smiles: str) -> float:
    """Estimate UV molar absorptivity proxy from aromatic ring count."""
    n_aromatic_rings = smiles.count("c1")
    return max(0.0, n_aromatic_rings * 20000.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_phototoxicity(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    tpsa: float = 50.0,
) -> PhototoxResult:
    """Assess phototoxicity risk using structural alerts and physicochemical properties.

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
    PhototoxResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string must not be empty")
    if mw <= 0:
        raise ValueError("Molecular weight must be positive")

    alerts: list[PhototoxAlert] = []

    for name, info in ALERT_PATTERNS.items():
        if name in _SMILES_ALERTS:
            triggered = _check_smiles_alert(smiles, name, mw)
        elif name in _PHYSCHEM_ALERTS:
            triggered = _check_physchem_alert(mw, logP, name, smiles)
        else:
            triggered = False

        alerts.append(
            PhototoxAlert(
                name=name,
                triggered=triggered,
                weight=info["weight"],
                description=info["description"],
            )
        )

    triggered_alerts = [a for a in alerts if a.triggered]
    n_alerts = len(triggered_alerts)
    alert_score = sum(a.weight for a in triggered_alerts)

    # Molar absorptivity proxy
    molar_absorptivity_proxy = _estimate_molar_absorptivity(smiles)

    # Classification
    if alert_score >= 2.0:
        phototoxicity_prediction = "positive"
        ich_s10_flag = "concern"
    elif alert_score >= 1.0:
        phototoxicity_prediction = "inconclusive"
        ich_s10_flag = "inconclusive"
    else:
        phototoxicity_prediction = "negative"
        ich_s10_flag = "no_concern"

    phototoxicity_probability = min(0.95, alert_score / 4.0)

    # Skin risk
    if logP > 3 and alert_score >= 1.5:
        skin_risk = "high"
    elif alert_score >= 1.0:
        skin_risk = "moderate"
    else:
        skin_risk = "low"

    # Eye risk
    if alert_score >= 2.0 and mw < 400:
        eye_risk = "high"
    elif alert_score >= 1.0:
        eye_risk = "moderate"
    else:
        eye_risk = "low"

    # Photo Drug Risk Index
    photo_dri = alert_score * max(1.0, logP / 2.0)

    # Notes
    note_parts: list[str] = []
    if n_alerts > 0:
        triggered_names = [a.name for a in triggered_alerts]
        note_parts.append(f"Triggered alerts: {', '.join(triggered_names)}")
    if phototoxicity_prediction == "positive":
        note_parts.append("ICH S10 recommends photosafety testing")
    notes = "; ".join(note_parts) if note_parts else "No phototoxicity alerts detected"

    return PhototoxResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        alerts=alerts,
        n_alerts=n_alerts,
        alert_score=alert_score,
        molar_absorptivity_proxy=molar_absorptivity_proxy,
        phototoxicity_prediction=phototoxicity_prediction,
        phototoxicity_probability=phototoxicity_probability,
        skin_risk=skin_risk,
        eye_risk=eye_risk,
        ich_s10_flag=ich_s10_flag,
        photo_dri=photo_dri,
        notes=notes,
    )


def screen_phototoxicity(compounds: list[dict]) -> list[PhototoxResult]:
    """Screen a list of compounds for phototoxicity risk.

    Parameters
    ----------
    compounds : list[dict]
        Each dict: {"name": ..., "smiles": ..., "mw": ..., "logP": ..., "tpsa": ...}

    Returns
    -------
    list[PhototoxResult]
        Sorted by photo_dri descending.
    """
    results = [
        assess_phototoxicity(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            tpsa=c.get("tpsa", 50.0),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.photo_dri, reverse=True)


def get_phototoxic_alerts_summary() -> dict[str, str]:
    """Return dict of alert_name -> description for all implemented alerts."""
    return {name: info["description"] for name, info in ALERT_PATTERNS.items()}
