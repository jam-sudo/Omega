"""Phase 747 — Drug Metabolite Toxicity Predictor.

Assesses reactive metabolite burden, organ-specific accumulation,
and phase I vs phase II metabolic fate to predict organ toxicity risk.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaboliteToxicityResult:
    """Result of metabolite toxicity assessment."""

    compound_name: str
    daily_dose_mg: float
    mw_da: float
    fm_cyp3a4: float
    fe_renal: float
    hepatic_risk_score: float
    renal_risk_score: float
    dili_risk_level: str
    renal_risk_level: str
    phase2_protection_factor: float
    overall_safety_concern: str
    recommendation: str
    notes: str


def _compute_hepatic_risk_score(
    daily_dose_mg: float,
    has_quinone_forming: bool,
    has_epoxide_forming: bool,
    fm_cyp3a4: float,
) -> float:
    score = 0.0
    if daily_dose_mg > 100:
        score += 20
    if daily_dose_mg > 500:
        score += 20
    if has_quinone_forming:
        score += 40
    if has_epoxide_forming:
        score += 30
    if fm_cyp3a4 > 0.7 and daily_dose_mg > 100:
        score += 15
    return min(100.0, score)


def _compute_renal_risk_score(
    fe_renal: float,
    mw_da: float,
    has_nephrotoxic_metabolite_flag: bool,
) -> float:
    score = 0.0
    if fe_renal > 0.5:
        score += 25
    if mw_da < 300 and fe_renal > 0.3:
        score += 15
    if has_nephrotoxic_metabolite_flag:
        score += 40
    return min(100.0, score)


def _dili_risk_level(hepatic_risk_score: float) -> str:
    if hepatic_risk_score > 60:
        return "high"
    if hepatic_risk_score >= 30:
        return "moderate"
    return "low"


def _renal_risk_level(renal_risk_score: float) -> str:
    if renal_risk_score > 60:
        return "high"
    if renal_risk_score >= 30:
        return "moderate"
    return "low"


def _overall_safety_concern(hepatic_risk_score: float, renal_risk_score: float) -> str:
    if hepatic_risk_score > 60 or renal_risk_score > 60:
        return "significant concern"
    if hepatic_risk_score > 30 or renal_risk_score > 30:
        return "caution"
    return "acceptable"


def _build_recommendation(
    hepatic_risk_score: float,
    renal_risk_score: float,
    has_quinone_forming: bool,
    has_epoxide_forming: bool,
    has_nephrotoxic_metabolite_flag: bool,
    phase2_coverage: float,
) -> str:
    parts = []
    if hepatic_risk_score > 60:
        parts.append("High hepatic risk: monitor liver enzymes closely")
    elif hepatic_risk_score > 30:
        parts.append("Moderate hepatic risk: periodic liver function tests recommended")
    if renal_risk_score > 60:
        parts.append("High renal risk: monitor renal function and adjust dose in impairment")
    elif renal_risk_score > 30:
        parts.append("Moderate renal risk: consider dose adjustment in renal impairment")
    if has_quinone_forming:
        parts.append(
            "Quinone/quinoneimine reactive metabolite detected: consider structural modification"
        )
    if has_epoxide_forming:
        parts.append("Arene epoxide risk: assess bioactivation pathway")
    if has_nephrotoxic_metabolite_flag:
        parts.append("Nephrotoxic metabolite flag: avoid in renal insufficiency")
    if phase2_coverage < 0.5:
        parts.append("Low phase II coverage: increased reactive metabolite burden expected")
    if not parts:
        parts.append("No major concerns identified; routine monitoring applies")
    return "; ".join(parts)


def _build_notes(
    daily_dose_mg: float,
    fm_cyp3a4: float,
    fe_renal: float,
    phase2_coverage: float,
    hepatic_risk_score: float,
    renal_risk_score: float,
) -> str:
    note_parts = [
        f"Daily dose {daily_dose_mg:.1f} mg",
        f"fm_CYP3A4={fm_cyp3a4:.2f}",
        f"fe_renal={fe_renal:.2f}",
        f"Phase II coverage={phase2_coverage:.2f}",
        f"Hepatic score={hepatic_risk_score:.0f}",
        f"Renal score={renal_risk_score:.0f}",
    ]
    return "; ".join(note_parts)


def assess_metabolite_toxicity(
    compound_name: str,
    daily_dose_mg: float,
    mw_da: float = 300.0,
    fm_cyp3a4: float = 0.3,
    fe_renal: float = 0.3,
    has_quinone_forming: bool = False,
    has_epoxide_forming: bool = False,
    has_nephrotoxic_metabolite_flag: bool = False,
    phase2_coverage: float = 0.7,
) -> MetaboliteToxicityResult:
    """Assess drug metabolite toxicity risk for organ-specific endpoints.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    daily_dose_mg:
        Total daily dose in milligrams (must be > 0).
    mw_da:
        Molecular weight in Daltons (must be > 0).
    fm_cyp3a4:
        Fraction metabolized by CYP3A4 (0–1).
    fe_renal:
        Fraction eliminated renally (0–1).
    has_quinone_forming:
        True if the compound forms quinone/quinoneimine reactive metabolites.
    has_epoxide_forming:
        True if the compound forms arene epoxide reactive metabolites.
    has_nephrotoxic_metabolite_flag:
        True if a known nephrotoxic metabolite is present.
    phase2_coverage:
        Fraction conjugated by phase II enzymes (higher → safer).

    Returns
    -------
    MetaboliteToxicityResult
    """
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if not (0.0 <= fm_cyp3a4 <= 1.0):
        raise ValueError(f"fm_cyp3a4 must be in [0, 1], got {fm_cyp3a4}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")

    hepatic_risk_score = _compute_hepatic_risk_score(
        daily_dose_mg, has_quinone_forming, has_epoxide_forming, fm_cyp3a4
    )
    renal_risk_score = _compute_renal_risk_score(fe_renal, mw_da, has_nephrotoxic_metabolite_flag)

    dili_lvl = _dili_risk_level(hepatic_risk_score)
    renal_lvl = _renal_risk_level(renal_risk_score)
    concern = _overall_safety_concern(hepatic_risk_score, renal_risk_score)
    recommendation = _build_recommendation(
        hepatic_risk_score,
        renal_risk_score,
        has_quinone_forming,
        has_epoxide_forming,
        has_nephrotoxic_metabolite_flag,
        phase2_coverage,
    )
    notes = _build_notes(
        daily_dose_mg, fm_cyp3a4, fe_renal, phase2_coverage, hepatic_risk_score, renal_risk_score
    )

    return MetaboliteToxicityResult(
        compound_name=compound_name,
        daily_dose_mg=daily_dose_mg,
        mw_da=mw_da,
        fm_cyp3a4=fm_cyp3a4,
        fe_renal=fe_renal,
        hepatic_risk_score=hepatic_risk_score,
        renal_risk_score=renal_risk_score,
        dili_risk_level=dili_lvl,
        renal_risk_level=renal_lvl,
        phase2_protection_factor=phase2_coverage,
        overall_safety_concern=concern,
        recommendation=recommendation,
        notes=notes,
    )


def screen_metabolite_toxicity(
    compounds: list,
) -> list:
    """Screen a list of compound parameter dicts and return sorted results.

    Parameters
    ----------
    compounds:
        List of dicts, each containing keyword arguments for
        `assess_metabolite_toxicity`.

    Returns
    -------
    list[MetaboliteToxicityResult] sorted by hepatic_risk_score descending.
    """
    results = [assess_metabolite_toxicity(**c) for c in compounds]
    results.sort(key=lambda r: r.hepatic_risk_score, reverse=True)
    return results
