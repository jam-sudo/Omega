"""Cardiovascular safety risk assessment — hERG inhibition, QTc prolongation, cardiac ion channel effects."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CardiovascularRiskResult",
    "assess_cardiovascular_risk",
    "screen_cardiovascular_risk",
    "estimate_herg_pIC50",
]


@dataclass(frozen=True)
class CardiovascularRiskResult:
    """Result of cardiovascular risk assessment."""

    compound_name: str
    therapeutic_conc_uM: float
    herg_ic50_uM: float
    sr_hERG: float
    qtc_risk: str
    nav15_ic50_uM: float | None
    sr_nav15: float | None
    overall_cardiac_risk: str
    requires_cardiac_safety_study: bool
    tqt_study_recommended: bool
    notes: str


def estimate_herg_pIC50(logP: float, mw: float, psa: float, pka_base: float | None = None) -> float:
    """Predict hERG pIC50 from physicochemical properties.

    Returns predicted pIC50 in range [3.0, 8.0].
    (pIC50=6 corresponds to IC50=1 µM; pIC50=7 to IC50=0.1 µM)
    """
    base_score = logP * 2.5 + (10 - mw / 100) * 0.5
    if pka_base is not None and pka_base > 7:
        base_score += 3.0
    if psa > 100:
        base_score -= 5.0
    return max(3.0, min(8.0, base_score))


def _qtc_risk_from_sr(sr: float) -> str:
    if sr > 30:
        return "low"
    elif sr > 10:
        return "moderate"
    else:
        return "high"


def _overall_cardiac_risk(sr_herg: float) -> str:
    if sr_herg <= 3:
        return "very_high"
    elif sr_herg <= 10:
        return "high"
    elif sr_herg <= 30:
        return "moderate"
    else:
        return "low"


def assess_cardiovascular_risk(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    pka_base: float | None = None,
    herg_ic50_uM: float | None = None,
    nav15_ic50_uM: float | None = None,
    cav12_ic50_uM: float | None = None,
    therapeutic_conc_uM: float = 1.0,
) -> CardiovascularRiskResult:
    """Assess cardiovascular safety risk for a compound.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).
    psa:
        Polar surface area (Å²).
    pka_base:
        Basic pKa (if applicable).
    herg_ic50_uM:
        Experimentally determined hERG IC50 (µM). If None, QSAR prediction is used.
    nav15_ic50_uM:
        Nav1.5 IC50 (µM), optional.
    cav12_ic50_uM:
        Cav1.2 IC50 (µM), optional (used for notes only).
    therapeutic_conc_uM:
        Expected therapeutic free concentration (µM).

    Returns
    -------
    CardiovascularRiskResult
    """
    if therapeutic_conc_uM <= 0:
        raise ValueError("therapeutic_conc_uM must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")

    # hERG IC50 — use QSAR if not provided
    if herg_ic50_uM is None:
        pic50 = estimate_herg_pIC50(logP, mw, psa, pka_base)
        # pIC50 = 6 - log10(IC50_uM)  => IC50_uM = 10^(6 - pIC50)
        herg_ic50_computed = 10 ** (6.0 - pic50)
    else:
        herg_ic50_computed = herg_ic50_uM

    sr_herg = herg_ic50_computed / therapeutic_conc_uM
    qtc_risk = _qtc_risk_from_sr(sr_herg)
    overall_risk = _overall_cardiac_risk(sr_herg)

    # Nav1.5
    sr_nav15: float | None = None
    nav15_flag = False
    if nav15_ic50_uM is not None:
        sr_nav15 = nav15_ic50_uM / therapeutic_conc_uM
        if sr_nav15 < 10:
            nav15_flag = True

    requires_study = sr_herg < 30
    tqt_recommended = sr_herg < 10

    # Build notes
    note_parts = [
        f"hERG IC50={herg_ic50_computed:.3g} µM, SR={sr_herg:.2f}x, QTc risk={qtc_risk}",
        f"Overall cardiac risk: {overall_risk}",
    ]
    if herg_ic50_uM is None:
        note_parts.append("hERG IC50 estimated by QSAR (logP/MW/PSA/pKa model)")
    else:
        note_parts.append("hERG IC50 provided experimentally")
    if nav15_flag:
        note_parts.append(f"Nav1.5 block concern: SR_nav15={sr_nav15:.2f}x (bradycardia risk)")
    if cav12_ic50_uM is not None:
        sr_cav12 = cav12_ic50_uM / therapeutic_conc_uM
        note_parts.append(f"Cav1.2 SR={sr_cav12:.2f}x")
    if tqt_recommended:
        note_parts.append("Thorough QT study (TQT) strongly recommended per ICH E14")
    elif requires_study:
        note_parts.append("Cardiac safety study recommended (hERG SR < 30)")
    notes = "; ".join(note_parts)

    return CardiovascularRiskResult(
        compound_name=compound_name,
        therapeutic_conc_uM=therapeutic_conc_uM,
        herg_ic50_uM=herg_ic50_computed,
        sr_hERG=sr_herg,
        qtc_risk=qtc_risk,
        nav15_ic50_uM=nav15_ic50_uM,
        sr_nav15=sr_nav15,
        overall_cardiac_risk=overall_risk,
        requires_cardiac_safety_study=requires_study,
        tqt_study_recommended=tqt_recommended,
        notes=notes,
    )


def screen_cardiovascular_risk(compounds: list[dict]) -> list[CardiovascularRiskResult]:
    """Screen a list of compounds for cardiovascular risk.

    Each dict must contain at minimum: compound_name, logP, mw, psa.
    Optional keys: pka_base, herg_ic50_uM, nav15_ic50_uM, cav12_ic50_uM, therapeutic_conc_uM.

    Returns list sorted by sr_hERG ascending (highest risk first).
    """
    results = []
    for c in compounds:
        result = assess_cardiovascular_risk(
            compound_name=c["compound_name"],
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            pka_base=c.get("pka_base"),
            herg_ic50_uM=c.get("herg_ic50_uM"),
            nav15_ic50_uM=c.get("nav15_ic50_uM"),
            cav12_ic50_uM=c.get("cav12_ic50_uM"),
            therapeutic_conc_uM=c.get("therapeutic_conc_uM", 1.0),
        )
        results.append(result)
    results.sort(key=lambda r: r.sr_hERG)
    return results
