"""Metabolic soft spot prediction from SMILES structural patterns.

Identifies sites most likely to be oxidized/metabolized and estimates
intrinsic clearance based on physicochemical properties and structural alerts.

Reference: Smith D.A. et al., J Med Chem 1996; Wienkers L.C. & Heath T.G.,
Nature Rev Drug Discov 2005.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SoftSpotAlert",
    "MetabolicSoftSpotResult",
    "predict_soft_spots",
    "rank_compounds_by_metabolic_stability",
]


@dataclass(frozen=True)
class SoftSpotAlert:
    """A single metabolic soft-spot structural alert."""

    motif: str  # e.g. "methyl_aromatic", "n_dealkylation"
    risk: str  # "low" / "moderate" / "high"
    cyp_enzyme: str  # e.g. "CYP3A4", "CYP2D6"
    description: str


@dataclass(frozen=True)
class MetabolicSoftSpotResult:
    """Full metabolic soft-spot prediction for one compound."""

    compound_name: str
    smiles: str
    logP: float
    mw: float
    alerts: list[SoftSpotAlert]
    n_high_risk: int
    predicted_clint_uL_per_min_per_mg: float  # intrinsic clearance estimate
    metabolic_stability: str  # "high" (stable) / "moderate" / "low" (unstable)
    primary_cyp: str  # most likely CYP enzyme
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_alerts(smiles: str, logP: float) -> list[SoftSpotAlert]:
    """Return list of SoftSpotAlert based on SMILES pattern checks."""
    alerts: list[SoftSpotAlert] = []

    # N-dealkylation: N-CH3 or N-CH2
    if "NCH3" in smiles or "N(C)" in smiles or "NCc" in smiles:
        alerts.append(
            SoftSpotAlert(
                "n_dealkylation",
                "high",
                "CYP3A4",
                "N-dealkylation site",
            )
        )

    # O-dealkylation: O-CH3 in aromatic context
    if "OC" in smiles and "c" in smiles:
        alerts.append(
            SoftSpotAlert(
                "o_dealkylation",
                "moderate",
                "CYP2D6",
                "O-dealkylation of aryl methoxy",
            )
        )

    # Aromatic hydroxylation: unsubstituted aromatic
    if smiles.count("c") >= 4:
        alerts.append(
            SoftSpotAlert(
                "aromatic_hydroxylation",
                "moderate",
                "CYP3A4",
                "Aromatic ring hydroxylation",
            )
        )

    # Aliphatic hydroxylation: methyl on aromatic
    if "Cc" in smiles or "cC" in smiles:
        alerts.append(
            SoftSpotAlert(
                "methyl_hydroxylation",
                "low",
                "CYP1A2",
                "Aromatic methyl oxidation",
            )
        )

    # Sulfoxidation: S present
    if "S" in smiles and "c" not in smiles[: smiles.find("S") + 2]:
        alerts.append(
            SoftSpotAlert(
                "sulfoxidation",
                "moderate",
                "CYP3A4",
                "Thioether sulfoxidation",
            )
        )

    # High logP: lipophilic → CYP3A4 substrate
    if logP > 3:
        alerts.append(
            SoftSpotAlert(
                "lipophilicity",
                "moderate",
                "CYP3A4",
                "High logP favors CYP3A4 metabolism",
            )
        )

    return alerts


def _estimate_clint(alerts: list[SoftSpotAlert], logP: float, mw: float) -> float:
    """Estimate intrinsic clearance (µL/min/mg protein) from alert counts."""
    n_high = sum(1 for a in alerts if a.risk == "high")
    n_mod = sum(1 for a in alerts if a.risk == "moderate")
    base_clint = 10.0 + n_high * 30.0 + n_mod * 10.0 + logP * 5.0 - mw * 0.05
    return max(1.0, min(300.0, base_clint))


def _metabolic_stability_label(clint: float) -> str:
    if clint < 30:
        return "high"
    if clint < 100:
        return "moderate"
    return "low"


def _primary_cyp(alerts: list[SoftSpotAlert]) -> str:
    """Return most frequent CYP among high-risk alerts, then moderate, then default."""
    for risk_level in ("high", "moderate"):
        cyps = [a.cyp_enzyme for a in alerts if a.risk == risk_level]
        if cyps:
            return max(set(cyps), key=cyps.count)
    return "CYP3A4"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_soft_spots(
    compound_name: str,
    smiles: str,
    logP: float,
    mw: float,
    n_aromatic_rings: int = 1,
) -> MetabolicSoftSpotResult:
    """Predict metabolic soft spots for a single compound.

    Parameters
    ----------
    compound_name : str
        Human-readable compound identifier.
    smiles : str
        SMILES string for the compound.
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight in Da.
    n_aromatic_rings : int
        Number of aromatic rings (reserved for future use).

    Returns
    -------
    MetabolicSoftSpotResult

    Raises
    ------
    ValueError
        If smiles is empty or mw is non-positive.
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must not be empty")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")

    alerts = _detect_alerts(smiles, logP)
    n_high = sum(1 for a in alerts if a.risk == "high")
    clint = _estimate_clint(alerts, logP, mw)
    stability = _metabolic_stability_label(clint)
    cyp = _primary_cyp(alerts)

    n_alerts = len(alerts)
    notes = (
        f"{n_alerts} soft-spot alert(s) detected; "
        f"CLint estimate {clint:.1f} µL/min/mg; "
        f"primary CYP: {cyp}"
    )

    return MetabolicSoftSpotResult(
        compound_name=compound_name,
        smiles=smiles,
        logP=logP,
        mw=mw,
        alerts=alerts,
        n_high_risk=n_high,
        predicted_clint_uL_per_min_per_mg=clint,
        metabolic_stability=stability,
        primary_cyp=cyp,
        notes=notes,
    )


def rank_compounds_by_metabolic_stability(
    compounds: list[dict],
) -> list[MetabolicSoftSpotResult]:
    """Screen a library and rank by metabolic stability (most stable first).

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain keys: ``name``, ``smiles``, ``logP``, ``mw``.
        Optional: ``n_aromatic_rings``.

    Returns
    -------
    list[MetabolicSoftSpotResult]
        Sorted ascending by ``predicted_clint_uL_per_min_per_mg``
        (lowest CLint = most stable appears first).
    """
    results: list[MetabolicSoftSpotResult] = []
    for cmpd in compounds:
        result = predict_soft_spots(
            compound_name=cmpd["name"],
            smiles=cmpd["smiles"],
            logP=float(cmpd["logP"]),
            mw=float(cmpd["mw"]),
            n_aromatic_rings=int(cmpd.get("n_aromatic_rings", 1)),
        )
        results.append(result)

    return sorted(results, key=lambda r: r.predicted_clint_uL_per_min_per_mg)
