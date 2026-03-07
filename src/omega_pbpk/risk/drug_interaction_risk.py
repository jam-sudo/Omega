"""DDI risk assessment based on physicochemical properties and metabolic pathways."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DDIRiskResult",
    "assess_ddi_risk",
    "screen_ddi_risk",
    "estimate_aucr",
]


@dataclass(frozen=True)
class DDIRiskResult:
    """Result of DDI risk assessment for a compound."""

    compound_name: str
    cyp3a4_risk_score: float
    cyp2d6_risk_score: float
    cyp2c9_risk_score: float
    pgp_risk: float
    protein_binding_risk: float
    overall_ddi_score: float
    risk_level: str
    primary_concern: str
    notes: str


def _cyp_score(ki_uM: float | None) -> float:
    """Compute CYP inhibition risk score (0-30) from Ki in µM."""
    if ki_uM is None:
        return 0.0
    if ki_uM < 1.0:
        return 30.0
    if ki_uM < 10.0:
        return 20.0
    if ki_uM < 100.0:
        return 10.0
    return 0.0


def assess_ddi_risk(
    compound_name: str,
    logP: float,
    mw: float,
    fup: float,
    ki_cyp3a4: float | None = None,
    ki_cyp2d6: float | None = None,
    ki_cyp2c9: float | None = None,
    is_p450_substrate: bool = True,
    is_pgp_inhibitor: bool = False,
    is_pgp_substrate: bool = False,
) -> DDIRiskResult:
    """Assess DDI risk based on physicochemical properties and metabolic pathways.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da). Must be > 0.
    fup:
        Fraction unbound in plasma. Must be in (0, 1].
    ki_cyp3a4:
        CYP3A4 inhibition constant (µM). None if not an inhibitor.
    ki_cyp2d6:
        CYP2D6 inhibition constant (µM). None if not an inhibitor.
    ki_cyp2c9:
        CYP2C9 inhibition constant (µM). None if not an inhibitor.
    is_p450_substrate:
        Whether the compound is a CYP450 substrate.
    is_pgp_inhibitor:
        Whether the compound inhibits P-gp.
    is_pgp_substrate:
        Whether the compound is a P-gp substrate.

    Returns
    -------
    DDIRiskResult with overall DDI score and risk classification.
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # CYP inhibition risk scores (0-30 each)
    s3a4 = _cyp_score(ki_cyp3a4)
    s2d6 = _cyp_score(ki_cyp2d6)
    s2c9 = _cyp_score(ki_cyp2c9)

    # P-gp risk: +20 if both inhibitor and substrate
    pgp_risk = 20.0 if (is_pgp_inhibitor and is_pgp_substrate) else 0.0

    # Protein binding displacement risk
    protein_binding_risk = 15.0 if fup < 0.05 else 0.0

    # Overall score (raw sum, capped at 100)
    raw_score = s3a4 + s2d6 + s2c9 + pgp_risk + protein_binding_risk
    overall_score = min(100.0, raw_score)

    # Risk level
    if overall_score < 30:
        risk_level = "low"
    elif overall_score <= 60:
        risk_level = "moderate"
    else:
        risk_level = "high"

    # Primary concern: identify the largest contributor
    concerns: list[tuple[float, str]] = [
        (s3a4, "CYP3A4"),
        (s2d6, "CYP2D6"),
        (s2c9, "CYP2C9"),
        (pgp_risk, "P-gp"),
        (protein_binding_risk, "protein binding"),
    ]
    concerns.sort(key=lambda c: c[0], reverse=True)
    top_score, top_concern = concerns[0]
    if top_score == 0.0:
        primary_concern = "low"
    else:
        primary_concern = top_concern

    note_parts = [
        f"CYP3A4 score={s3a4:.0f}",
        f"CYP2D6 score={s2d6:.0f}",
        f"CYP2C9 score={s2c9:.0f}",
        f"P-gp risk={pgp_risk:.0f}",
        f"Protein binding risk={protein_binding_risk:.0f}",
        f"Overall DDI score={overall_score:.1f} ({risk_level})",
        f"Primary concern: {primary_concern}",
    ]
    notes = ". ".join(note_parts) + "."

    return DDIRiskResult(
        compound_name=compound_name,
        cyp3a4_risk_score=s3a4,
        cyp2d6_risk_score=s2d6,
        cyp2c9_risk_score=s2c9,
        pgp_risk=pgp_risk,
        protein_binding_risk=protein_binding_risk,
        overall_ddi_score=overall_score,
        risk_level=risk_level,
        primary_concern=primary_concern,
        notes=notes,
    )


def screen_ddi_risk(compounds: list[dict]) -> list[DDIRiskResult]:
    """Screen multiple compounds for DDI risk, sorted by overall_ddi_score descending.

    Each dict in compounds should contain keys accepted by assess_ddi_risk.
    Required keys: compound_name, logP, mw, fup.
    Optional: ki_cyp3a4, ki_cyp2d6, ki_cyp2c9, is_p450_substrate, is_pgp_inhibitor, is_pgp_substrate.
    """
    results: list[DDIRiskResult] = []
    for cpd in compounds:
        result = assess_ddi_risk(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            fup=cpd["fup"],
            ki_cyp3a4=cpd.get("ki_cyp3a4"),
            ki_cyp2d6=cpd.get("ki_cyp2d6"),
            ki_cyp2c9=cpd.get("ki_cyp2c9"),
            is_p450_substrate=cpd.get("is_p450_substrate", True),
            is_pgp_inhibitor=cpd.get("is_pgp_inhibitor", False),
            is_pgp_substrate=cpd.get("is_pgp_substrate", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_ddi_score, reverse=True)
    return results


def estimate_aucr(ki_uM: float, i_max_uM: float, fm_cyp: float = 1.0) -> float:
    """Estimate AUCR using static basic model.

    AUCR = 1 / (1 - fm_cyp + fm_cyp / (1 + I_max / Ki))

    Parameters
    ----------
    ki_uM:
        Inhibition constant of the perpetrator (µM).
    i_max_uM:
        Maximum inhibitor concentration at the enzyme site (µM).
    fm_cyp:
        Fraction metabolized by the inhibited CYP (0-1).

    Returns
    -------
    AUCR (ratio of victim AUC in presence vs absence of inhibitor).
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if fm_cyp < 0 or fm_cyp > 1:
        raise ValueError(f"fm_cyp must be in [0, 1], got {fm_cyp}")

    denominator = 1.0 - fm_cyp + fm_cyp / (1.0 + i_max_uM / ki_uM)
    if abs(denominator) < 1e-15:
        return float("inf")
    return 1.0 / denominator
