"""Phase 182 — Ames Mutagenicity Prediction

Predict Ames test mutagenicity from SMILES structural alerts and physicochemical
properties.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AmesAlert",
    "AmesMutagenicityResult",
    "predict_ames_mutagenicity",
    "screen_ames",
]

# ---------------------------------------------------------------------------
# Alert definitions
# ---------------------------------------------------------------------------

_ALERT_DEFS: dict[str, dict[str, float | str]] = {
    "nitro_aromatic": {
        "weight": 3.0,
        "desc": "Aromatic nitro group: reductive activation to nitroso",
    },
    "primary_aromatic_amine": {
        "weight": 2.5,
        "desc": "Primary aromatic amine: N-hydroxylation",
    },
    "hydrazine": {
        "weight": 2.5,
        "desc": "Hydrazine: oxidation to diazonium",
    },
    "aziridine": {
        "weight": 2.0,
        "desc": "Aziridine: direct alkylation of DNA",
    },
    "alkyl_halide": {
        "weight": 1.5,
        "desc": "Alkyl halide: SN2 DNA alkylation",
    },
    "epoxide": {
        "weight": 2.0,
        "desc": "Epoxide: direct DNA adduct formation",
    },
    "acylating_agent": {
        "weight": 1.5,
        "desc": "Acyl halide or anhydride: acylation",
    },
    "michael_acceptor": {
        "weight": 1.0,
        "desc": "Michael acceptor: thiol/amine adducts",
    },
    "quinone": {
        "weight": 1.5,
        "desc": "Quinone: redox cycling, oxidative DNA damage",
    },
    "furan": {
        "weight": 1.0,
        "desc": "Furan: CYP-mediated epoxide",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmesAlert:
    name: str
    triggered: bool
    weight: float
    description: str


@dataclass(frozen=True)
class AmesMutagenicityResult:
    compound_name: str
    smiles: str
    mw: float
    logP: float
    alerts: list[AmesAlert]
    n_alerts: int
    mutagenicity_score: float    # 0-100
    ames_prediction: str         # "positive", "negative", "borderline"
    genotoxicity_risk: str       # "low", "moderate", "high"
    primary_concern: str         # top alert name or "none"
    mitigation: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_alert(smiles: str, name: str) -> bool:  # noqa: PLR0911
    if name == "nitro_aromatic":
        return "[N+](=O)[O-]" in smiles and "c" in smiles
    if name == "primary_aromatic_amine":
        return "Nc1" in smiles or "cN" in smiles
    if name == "hydrazine":
        return "NN" in smiles
    if name == "aziridine":
        return "C1CN1" in smiles or "N1CC1" in smiles
    if name == "alkyl_halide":
        return "CCl" in smiles or "CBr" in smiles or "CI" in smiles
    if name == "epoxide":
        return "C1OC1" in smiles
    if name == "acylating_agent":
        return "C(=O)Cl" in smiles or "C(=O)F" in smiles
    if name == "michael_acceptor":
        return "C=CC=O" in smiles or "C=CC(=O)" in smiles
    if name == "quinone":
        return "O=C1C=CC(=O)" in smiles or "C1(=O)C=CC(=O)" in smiles
    if name == "furan":
        return "c1ccoc1" in smiles or (
            "o" in smiles and "c" in smiles and len(smiles) < 30
        )
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_ames_mutagenicity(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
) -> AmesMutagenicityResult:
    """Predict Ames mutagenicity from SMILES structural alerts.

    Parameters
    ----------
    compound_name:
        Human-readable identifier for the compound.
    smiles:
        SMILES string for structural alert matching.
    mw:
        Molecular weight in g/mol.
    logP:
        Lipophilicity (log octanol-water partition coefficient).

    Returns
    -------
    AmesMutagenicityResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must not be empty")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")

    alerts: list[AmesAlert] = []
    triggered_alerts: list[AmesAlert] = []

    for alert_name, info in _ALERT_DEFS.items():
        triggered = _check_alert(smiles, alert_name)
        alert = AmesAlert(
            name=alert_name,
            triggered=triggered,
            weight=float(info["weight"]),  # type: ignore[arg-type]
            description=str(info["desc"]),
        )
        alerts.append(alert)
        if triggered:
            triggered_alerts.append(alert)

    n_alerts = len(triggered_alerts)
    alert_score = sum(a.weight for a in triggered_alerts)
    mutagenicity_score = min(100.0, alert_score * 15.0 + logP * 2.0)

    if mutagenicity_score >= 60.0:
        ames_prediction = "positive"
    elif mutagenicity_score >= 30.0:
        ames_prediction = "borderline"
    else:
        ames_prediction = "negative"

    if mutagenicity_score >= 60.0:
        genotoxicity_risk = "high"
    elif mutagenicity_score >= 30.0:
        genotoxicity_risk = "moderate"
    else:
        genotoxicity_risk = "low"

    # Primary concern: triggered alert with highest weight
    if triggered_alerts:
        primary_alert = max(triggered_alerts, key=lambda a: a.weight)
        primary_concern = primary_alert.name
    else:
        primary_concern = "none"

    # Mitigation suggestions
    mitigation: list[str] = []
    triggered_names = {a.name for a in triggered_alerts}
    if "nitro_aromatic" in triggered_names:
        mitigation.append("Replace nitro with non-reducible bioisostere")
    if "primary_aromatic_amine" in triggered_names:
        mitigation.append("Mask amine or use secondary amine alternative")
    if "hydrazine" in triggered_names:
        mitigation.append("Replace hydrazine with less reactive linker")
    if "aziridine" in triggered_names:
        mitigation.append("Avoid strained three-membered nitrogen ring")
    if "alkyl_halide" in triggered_names:
        mitigation.append("Replace alkyl halide with less electrophilic group")
    if "epoxide" in triggered_names:
        mitigation.append("Avoid epoxide functionality; use stable analogue")
    if "acylating_agent" in triggered_names:
        mitigation.append("Replace acylating agent with stable bioisostere")
    if "michael_acceptor" in triggered_names:
        mitigation.append("Saturate Michael acceptor or introduce steric shield")
    if "quinone" in triggered_names:
        mitigation.append("Replace quinone with non-redox-cycling moiety")
    if "furan" in triggered_names:
        mitigation.append("Replace furan with thiophene or other bioisostere")

    if not mitigation:
        mitigation = ["No mutagenicity concerns identified"]

    notes = (
        f"Score {mutagenicity_score:.1f}/100 based on {n_alerts} structural alert(s). "
        f"logP contribution: {logP * 2.0:.1f}."
    )

    return AmesMutagenicityResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        alerts=alerts,
        n_alerts=n_alerts,
        mutagenicity_score=mutagenicity_score,
        ames_prediction=ames_prediction,
        genotoxicity_risk=genotoxicity_risk,
        primary_concern=primary_concern,
        mitigation=mitigation,
        notes=notes,
    )


def screen_ames(compounds: list[dict]) -> list[AmesMutagenicityResult]:
    """Screen a list of compounds for Ames mutagenicity.

    Parameters
    ----------
    compounds:
        Each dict must contain keys: ``name``, ``smiles``, ``mw``, ``logP``.

    Returns
    -------
    List of AmesMutagenicityResult sorted by mutagenicity_score descending.
    """
    results: list[AmesMutagenicityResult] = []
    for cpd in compounds:
        result = predict_ames_mutagenicity(
            compound_name=cpd["name"],
            smiles=cpd["smiles"],
            mw=cpd["mw"],
            logP=cpd["logP"],
        )
        results.append(result)

    results.sort(key=lambda r: r.mutagenicity_score, reverse=True)
    return results
