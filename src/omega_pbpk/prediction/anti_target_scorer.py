"""Phase 774 — Drug Anti-Target Avoidance Scorer.

Score compounds for predicted selectivity away from anti-targets (hERG, MAO-A, COX-1,
muscarinic M1) using physicochemical QSAR features.

Anti-targets are receptors whose activation causes adverse effects.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntiTargetScoreResult:
    """Anti-target avoidance scoring result for a single compound."""

    compound_name: str
    herg_risk_score: float
    mao_risk_score: float
    cox1_risk_score: float
    m1_risk_score: float
    overall_anti_target_risk: float
    herg_risk_level: str
    mao_risk_level: str
    cox1_risk_level: str
    m1_risk_level: str
    overall_risk_level: str
    selectivity_score: float
    recommendations: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _risk_level(score: float) -> str:
    """Map a 0-100 risk score to a risk level label."""
    if score < 30.0:
        return "low"
    elif score <= 60.0:
        return "moderate"
    else:
        return "high"


def _compute_herg_risk(
    logp: float,
    pka_basic: float,
    psa: float,
    mw_da: float,
) -> float:
    """hERG channel blockade risk (cardiac arrhythmia)."""
    score = logp * 8.0 + pka_basic * 3.0 - psa * 0.2 - 20.0
    score = max(0.0, score)
    if mw_da > 600.0:
        score -= 10.0
    return min(100.0, max(0.0, score))


def _compute_mao_risk(
    has_hydrazine: bool,
    has_propargylamine: bool,
    aromatic_amine_count: int,
) -> float:
    """MAO-A inhibition risk (hypertensive crisis from tyramine)."""
    score = (
        (50.0 if has_hydrazine else 0.0)
        + (40.0 if has_propargylamine else 0.0)
        + aromatic_amine_count * 15.0
    )
    score = max(0.0, score)
    return min(100.0, score)


def _compute_cox1_risk(
    has_carboxylic_acid: bool,
    has_aryl_acetic_acid: bool,
    has_aryl_propionic_acid: bool,
) -> float:
    """COX-1 inhibition risk (GI bleeding)."""
    score = (
        (25.0 if has_carboxylic_acid else 0.0)
        + (35.0 if has_aryl_acetic_acid else 0.0)
        + (35.0 if has_aryl_propionic_acid else 0.0)
    )
    score = max(0.0, score)
    return min(100.0, score)


def _compute_m1_risk(
    has_tertiary_amine: bool,
    logp: float,
    psa: float,
) -> float:
    """Muscarinic M1 anticholinergic risk."""
    score = (30.0 if has_tertiary_amine else 0.0) + logp * 5.0 - psa * 0.3
    return min(100.0, max(0.0, score))


def _build_recommendations(
    herg_risk: float,
    mao_risk: float,
    cox1_risk: float,
    m1_risk: float,
    logp: float,
    pka_basic: float,
    has_hydrazine: bool,
    has_propargylamine: bool,
    has_carboxylic_acid: bool,
    has_aryl_acetic_acid: bool,
    has_aryl_propionic_acid: bool,
    has_tertiary_amine: bool,
) -> str:
    """Build comma-joined structural modification recommendations."""
    recs: list[str] = []

    if herg_risk >= 30.0:
        if logp > 3.0:
            recs.append("reduce lipophilicity (lower logP) to decrease hERG risk")
        if pka_basic > 8.0:
            recs.append("lower basic pKa to decrease hERG binding affinity")
        recs.append("add polar groups to increase PSA and reduce hERG risk")

    if mao_risk >= 30.0:
        if has_hydrazine:
            recs.append("replace hydrazine moiety to reduce MAO-A inhibition risk")
        if has_propargylamine:
            recs.append("remove propargylamine group to reduce MAO-A inhibition")
        if has_hydrazine or has_propargylamine:
            recs.append("consider non-hydrazine/propargylamine alternatives")

    if cox1_risk >= 30.0:
        if has_aryl_acetic_acid or has_aryl_propionic_acid:
            recs.append("consider prodrug approach to reduce COX-1 GI effects")
        if has_carboxylic_acid:
            recs.append(
                "replace carboxylic acid with bioisostere (e.g., tetrazole) for COX-1 selectivity"
            )

    if m1_risk >= 30.0:
        if has_tertiary_amine:
            recs.append(
                "replace tertiary amine with secondary amine or quaternary N to reduce M1 anticholinergic risk"
            )
        recs.append("reduce lipophilicity to decrease CNS M1 exposure")

    if not recs:
        recs.append("no major structural modifications required; maintain current scaffold")

    return ", ".join(recs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_anti_target_avoidance(
    compound_name: str,
    logp: float = 2.0,
    mw_da: float = 300.0,
    psa: float = 60.0,
    pka_basic: float | None = None,
    has_hydrazine: bool = False,
    has_propargylamine: bool = False,
    has_carboxylic_acid: bool = False,
    has_aryl_acetic_acid: bool = False,
    has_aryl_propionic_acid: bool = False,
    has_tertiary_amine: bool = False,
    aromatic_amine_count: int = 0,
) -> AntiTargetScoreResult:
    """Score a compound's anti-target avoidance profile.

    Parameters
    ----------
    compound_name : str
    logp : float
        Lipophilicity (octanol-water partition coefficient).
    mw_da : float
        Molecular weight in Daltons.
    psa : float
        Polar surface area in Angstrom^2.
    pka_basic : float | None
        Basic pKa. Defaults to -1.0 (no basic center) when None.
    has_hydrazine : bool
        Structural alert for MAO-A inhibition.
    has_propargylamine : bool
        Structural alert for MAO-A inhibition.
    has_carboxylic_acid : bool
        Structural alert for COX-1 inhibition.
    has_aryl_acetic_acid : bool
        Structural alert for COX-1 inhibition (aryl acetic acid NSAIDs).
    has_aryl_propionic_acid : bool
        Structural alert for COX-1 inhibition (aryl propionic acid NSAIDs).
    has_tertiary_amine : bool
        Structural alert for M1 anticholinergic activity.
    aromatic_amine_count : int
        Number of aromatic amine groups (for MAO-A risk).

    Returns
    -------
    AntiTargetScoreResult
    """
    # Validation
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if aromatic_amine_count < 0:
        raise ValueError(f"aromatic_amine_count must be >= 0, got {aromatic_amine_count}")

    # Sentinel for missing pKa basic
    effective_pka = pka_basic if pka_basic is not None else -1.0

    herg_risk = _compute_herg_risk(logp, effective_pka, psa, mw_da)
    mao_risk = _compute_mao_risk(has_hydrazine, has_propargylamine, aromatic_amine_count)
    cox1_risk = _compute_cox1_risk(
        has_carboxylic_acid, has_aryl_acetic_acid, has_aryl_propionic_acid
    )
    m1_risk = _compute_m1_risk(has_tertiary_amine, logp, psa)

    overall_risk = max(herg_risk, mao_risk, cox1_risk, m1_risk)
    selectivity_score = 100.0 - overall_risk

    recommendations = _build_recommendations(
        herg_risk=herg_risk,
        mao_risk=mao_risk,
        cox1_risk=cox1_risk,
        m1_risk=m1_risk,
        logp=logp,
        pka_basic=effective_pka,
        has_hydrazine=has_hydrazine,
        has_propargylamine=has_propargylamine,
        has_carboxylic_acid=has_carboxylic_acid,
        has_aryl_acetic_acid=has_aryl_acetic_acid,
        has_aryl_propionic_acid=has_aryl_propionic_acid,
        has_tertiary_amine=has_tertiary_amine,
    )

    # Determine overall risk level
    overall_level = _risk_level(overall_risk)
    notes = (
        f"hERG risk: {herg_risk:.1f} ({_risk_level(herg_risk)}), "
        f"MAO-A risk: {mao_risk:.1f} ({_risk_level(mao_risk)}), "
        f"COX-1 risk: {cox1_risk:.1f} ({_risk_level(cox1_risk)}), "
        f"M1 risk: {m1_risk:.1f} ({_risk_level(m1_risk)}). "
        f"Selectivity score {selectivity_score:.1f}/100 "
        f"(higher = better anti-target avoidance)."
    )

    return AntiTargetScoreResult(
        compound_name=compound_name,
        herg_risk_score=herg_risk,
        mao_risk_score=mao_risk,
        cox1_risk_score=cox1_risk,
        m1_risk_score=m1_risk,
        overall_anti_target_risk=overall_risk,
        herg_risk_level=_risk_level(herg_risk),
        mao_risk_level=_risk_level(mao_risk),
        cox1_risk_level=_risk_level(cox1_risk),
        m1_risk_level=_risk_level(m1_risk),
        overall_risk_level=overall_level,
        selectivity_score=selectivity_score,
        recommendations=recommendations,
        notes=notes,
    )


def screen_anti_target(
    compounds: list[dict],
) -> list[AntiTargetScoreResult]:
    """Screen a list of compounds for anti-target avoidance.

    Each element of compounds is a dict of kwargs for score_anti_target_avoidance.

    Returns list sorted by selectivity_score descending.
    """
    results: list[AntiTargetScoreResult] = []
    for cmpd in compounds:
        result = score_anti_target_avoidance(**cmpd)
        results.append(result)
    results.sort(key=lambda r: r.selectivity_score, reverse=True)
    return results
