"""Drug repurposing similarity scoring based on PK/PD profile matching.

Given a target disease indication and a library of existing drugs,
score each drug for repurposing suitability based on physicochemical
similarity, PK target attainment, and safety flags.

References
----------
Pushpakom 2018 (Nat Rev Drug Discov), Barabasi 2011 (Nat Rev Drug Discov).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugProfile",
    "RepurposingTarget",
    "RepurposingScore",
    "score_repurposing",
    "screen_library_repurposing",
    "get_default_drug_library",
    "create_target",
]


@dataclass(frozen=True)
class DrugProfile:
    """Known drug profile for repurposing analysis."""

    name: str
    mw: float
    logP: float
    fup: float
    t_half_h: float
    f_oral: float
    cmax_mg_L: float
    herg_ic50_uM: float
    target_indication: str


@dataclass(frozen=True)
class RepurposingTarget:
    """Target specification for a repurposing screen."""

    indication: str
    required_t_half_range_h: tuple[float, float]
    required_f_oral: float
    target_cmax_mg_L: float
    mw_range: tuple[float, float]
    logP_range: tuple[float, float]


@dataclass(frozen=True)
class RepurposingScore:
    """Repurposing score for a single drug-target pair."""

    drug_name: str
    target_indication: str
    current_indication: str
    pk_match_score: float
    safety_score: float
    similarity_score: float
    composite_score: float
    recommendation: str
    pk_gaps: list[str]
    notes: str


def _compute_pk_match(drug: DrugProfile, target: RepurposingTarget) -> tuple[float, list[str]]:
    """Compute PK match score and identify gaps."""
    score = 0.0
    gaps: list[str] = []

    t_min, t_max = target.required_t_half_range_h
    if t_min <= drug.t_half_h <= t_max:
        score += 0.4
    else:
        gaps.append(f"t1/2 {drug.t_half_h}h outside [{t_min}-{t_max}h]")

    if drug.f_oral >= target.required_f_oral:
        score += 0.3
    else:
        gaps.append(f"F_oral {drug.f_oral:.2f} < required {target.required_f_oral:.2f}")

    if target.target_cmax_mg_L > 0:
        ratio = drug.cmax_mg_L / target.target_cmax_mg_L
        if 0.2 <= ratio <= 5.0:
            score += 0.3
        else:
            gaps.append(f"Cmax {drug.cmax_mg_L} outside 5x of target {target.target_cmax_mg_L}")

    return score, gaps


def _compute_safety(drug: DrugProfile) -> float:
    """Compute safety score (0-1)."""
    score = 0.0

    # hERG safety margin
    if drug.fup > 0 and drug.cmax_mg_L > 0:
        # Convert cmax to approximate uM (assume mw available)
        cmax_free_uM = (drug.cmax_mg_L / drug.fup) / (drug.mw / 1000) if drug.mw > 0 else 0
        if cmax_free_uM > 0 and drug.herg_ic50_uM / cmax_free_uM > 30:
            score += 0.5
    else:
        score += 0.5  # No data, assume safe

    if drug.logP < 5:
        score += 0.25
    if drug.mw < 500:
        score += 0.25

    return score


def _compute_similarity(drug: DrugProfile, target: RepurposingTarget) -> float:
    """Compute physicochemical similarity score (0-1)."""
    score = 0.0

    mw_min, mw_max = target.mw_range
    if mw_min <= drug.mw <= mw_max:
        score += 0.33

    lp_min, lp_max = target.logP_range
    if lp_min <= drug.logP <= lp_max:
        score += 0.33

    if drug.f_oral >= 0.2:
        score += 0.34

    return score


def score_repurposing(
    drug: DrugProfile,
    target: RepurposingTarget,
) -> RepurposingScore:
    """Score a single drug for repurposing to a target indication.

    Parameters
    ----------
    drug : Drug profile with known PK.
    target : Target indication requirements.

    Returns
    -------
    RepurposingScore
    """
    pk_score, pk_gaps = _compute_pk_match(drug, target)
    safety = _compute_safety(drug)
    similarity = _compute_similarity(drug, target)

    composite = pk_score * 40 + safety * 30 + similarity * 30

    # Determine recommendation
    if safety < 0.5:
        recommendation = "contraindicated"
    elif composite >= 70:
        recommendation = "strong"
    elif composite >= 50:
        recommendation = "moderate"
    elif composite >= 30:
        recommendation = "weak"
    else:
        recommendation = "weak"

    notes = (
        f"Repurposing analysis: {drug.name} ({drug.target_indication}) "
        f"-> {target.indication}. Composite={composite:.1f}."
    )

    return RepurposingScore(
        drug_name=drug.name,
        target_indication=target.indication,
        current_indication=drug.target_indication,
        pk_match_score=pk_score,
        safety_score=safety,
        similarity_score=similarity,
        composite_score=composite,
        recommendation=recommendation,
        pk_gaps=pk_gaps,
        notes=notes,
    )


def screen_library_repurposing(
    target: RepurposingTarget,
    library: list[DrugProfile] | None = None,
) -> list[RepurposingScore]:
    """Screen a drug library for repurposing candidates.

    Parameters
    ----------
    target : Target indication requirements.
    library : Drug profiles to screen. Uses default library if None.

    Returns
    -------
    list[RepurposingScore]
        Sorted by composite_score descending.
    """
    if library is not None and len(library) == 0:
        return []
    if library is None:
        library = get_default_drug_library()
    results = [score_repurposing(drug, target) for drug in library]
    return sorted(results, key=lambda r: r.composite_score, reverse=True)


def get_default_drug_library() -> list[DrugProfile]:
    """Return a built-in library of 10 representative drugs with known PK."""
    return [
        DrugProfile("aspirin", 180.2, 1.2, 0.50, 0.3, 0.68, 30.0, 1000.0, "pain"),
        DrugProfile("metformin", 129.2, -1.4, 1.00, 5.0, 0.55, 1.0, 1000.0, "diabetes"),
        DrugProfile("ibuprofen", 206.3, 3.5, 0.01, 2.0, 0.80, 30.0, 300.0, "pain"),
        DrugProfile("atorvastatin", 558.6, 4.5, 0.02, 14.0, 0.14, 0.04, 30.0, "hyperlipidemia"),
        DrugProfile("amoxicillin", 365.4, 0.9, 0.80, 1.5, 0.74, 8.0, 1000.0, "infection"),
        DrugProfile("metoprolol", 267.4, 1.9, 0.88, 4.0, 0.50, 0.1, 50.0, "hypertension"),
        DrugProfile("sertraline", 306.2, 5.1, 0.02, 26.0, 0.44, 0.15, 10.0, "depression"),
        DrugProfile("tamoxifen", 371.5, 6.3, 0.02, 120.0, 0.50, 0.04, 5.0, "breast_cancer"),
        DrugProfile("dexamethasone", 392.5, 1.8, 0.23, 4.0, 0.80, 0.2, 1000.0, "inflammation"),
        DrugProfile("ciprofloxacin", 331.3, 0.3, 0.60, 4.0, 0.70, 3.0, 100.0, "infection"),
    ]


def create_target(
    indication: str,
    required_t_half_range_h: tuple[float, float] = (4, 24),
    required_f_oral: float = 0.5,
    target_cmax_mg_L: float = 1.0,
    mw_range: tuple[float, float] = (150, 500),
    logP_range: tuple[float, float] = (-1, 5),
) -> RepurposingTarget:
    """Create a RepurposingTarget with sensible defaults.

    Parameters
    ----------
    indication : Target disease indication.
    required_t_half_range_h : Acceptable half-life range (h).
    required_f_oral : Minimum oral bioavailability.
    target_cmax_mg_L : Optimal peak concentration (mg/L).
    mw_range : Acceptable molecular weight range.
    logP_range : Acceptable logP range.

    Returns
    -------
    RepurposingTarget
    """
    return RepurposingTarget(
        indication=indication,
        required_t_half_range_h=required_t_half_range_h,
        required_f_oral=required_f_oral,
        target_cmax_mg_L=target_cmax_mg_L,
        mw_range=mw_range,
        logP_range=logP_range,
    )
