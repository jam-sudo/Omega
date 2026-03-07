"""Formulation strategy screening -- Phase 460.

Screen and rank pharmaceutical formulation approaches for a drug candidate
based on BCS classification and physicochemical properties.

BCS class determines the primary challenge:
    Class I  : good solubility & permeability -> simple tablet/capsule
    Class II : low solubility                -> solubility enhancement
    Class III: low permeability              -> permeation enhancement
    Class IV : poor solubility & permeability -> combination strategies required

Scoring is deterministic and rule-based; each strategy receives a score 0-100
reflecting its expected benefit for the given BCS class and molecular profile.

References
----------
Amidon GL et al. Pharm Res. 1995;12(3):413-20.  (BCS)
Savjani KT et al. ISRN Pharm. 2012;2012:195727.  (solubility enhancement)
FDA Guidance: Waiver of In Vivo BA/BE Studies. 2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_BCS_CLASSES: set[int] = {1, 2, 3, 4}

# Maximum predicted bioavailability cap
_MAX_BIOAVAILABILITY: float = 0.95

# logP thresholds for lipophilic formulation decisions
_HIGH_LOGP: float = 3.0
_VERY_HIGH_LOGP: float = 5.0

# MW threshold for permeation enhancer suitability
_HIGH_MW: float = 500.0

# Default base bioavailability by BCS class (conservative literature estimates)
_BASE_F_BY_CLASS: dict[int, float] = {
    1: 0.80,
    2: 0.35,
    3: 0.25,
    4: 0.10,
}


# ---------------------------------------------------------------------------
# Formulation strategy definitions
# ---------------------------------------------------------------------------
# Each entry: (strategy_name, {bcs_class: base_score}, description)

_STRATEGIES: list[tuple[str, dict[int, float], str]] = [
    (
        "simple_tablet_capsule",
        {1: 90.0, 2: 20.0, 3: 40.0, 4: 15.0},
        "Conventional immediate-release solid oral dosage form",
    ),
    (
        "wet_granulation",
        {1: 75.0, 2: 30.0, 3: 35.0, 4: 20.0},
        "Wet granulation improves content uniformity and dissolution",
    ),
    (
        "amorphous_solid_dispersion",
        {1: 40.0, 2: 85.0, 3: 30.0, 4: 55.0},
        "ASD: drug dispersed in polymer matrix; supersaturation enables absorption",
    ),
    (
        "nanosuspension",
        {1: 35.0, 2: 78.0, 3: 25.0, 4: 50.0},
        "Particle size reduction to nano range increases dissolution rate",
    ),
    (
        "SEDDS_SMEDDS",
        {1: 30.0, 2: 80.0, 3: 20.0, 4: 48.0},
        "Self-emulsifying systems; ideal for lipophilic BCS II drugs",
    ),
    (
        "cyclodextrin_complex",
        {1: 45.0, 2: 72.0, 3: 35.0, 4: 45.0},
        "Cyclodextrin complexation improves apparent aqueous solubility",
    ),
    (
        "permeation_enhancers",
        {1: 20.0, 2: 15.0, 3: 78.0, 4: 55.0},
        "Chemical or physical permeation enhancers for BCS III/IV drugs",
    ),
    (
        "bioadhesive_system",
        {1: 25.0, 2: 20.0, 3: 70.0, 4: 50.0},
        "Bioadhesive polymers increase GI residence time and permeation",
    ),
    (
        "lipid_based_formulation",
        {1: 30.0, 2: 75.0, 3: 20.0, 4: 45.0},
        "Lipid-based drug delivery for lipophilic molecules",
    ),
    (
        "prodrug_strategy",
        {1: 10.0, 2: 40.0, 3: 60.0, 4: 58.0},
        "Chemical modification to improve solubility or permeability",
    ),
    (
        "pH_adjusted_formulation",
        {1: 50.0, 2: 55.0, 3: 45.0, 4: 40.0},
        "pH adjustment / salt form to maximise solubility at GI pH",
    ),
    (
        "combination_approach",
        {1: 15.0, 2: 45.0, 3: 50.0, 4: 60.0},
        "Dual strategy targeting both solubility and permeability limitations",
    ),
]

# Risk level by BCS class
_RISK_BY_CLASS: dict[int, str] = {
    1: "low",
    2: "medium",
    3: "medium",
    4: "high",
}

# Bioavailability improvement factors by strategy and BCS class (multiplicative)
_F_IMPROVEMENT_FACTOR: dict[str, dict[int, float]] = {
    "simple_tablet_capsule": {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00},
    "wet_granulation": {1: 1.05, 2: 1.10, 3: 1.05, 4: 1.05},
    "amorphous_solid_dispersion": {1: 1.05, 2: 1.80, 3: 1.10, 4: 1.40},
    "nanosuspension": {1: 1.05, 2: 1.60, 3: 1.05, 4: 1.30},
    "SEDDS_SMEDDS": {1: 1.05, 2: 1.70, 3: 1.05, 4: 1.35},
    "cyclodextrin_complex": {1: 1.05, 2: 1.50, 3: 1.10, 4: 1.25},
    "permeation_enhancers": {1: 1.05, 2: 1.05, 3: 1.70, 4: 1.50},
    "bioadhesive_system": {1: 1.05, 2: 1.05, 3: 1.60, 4: 1.45},
    "lipid_based_formulation": {1: 1.05, 2: 1.65, 3: 1.05, 4: 1.30},
    "prodrug_strategy": {1: 1.05, 2: 1.30, 3: 1.50, 4: 1.55},
    "pH_adjusted_formulation": {1: 1.10, 2: 1.30, 3: 1.15, 4: 1.20},
    "combination_approach": {1: 1.05, 2: 1.40, 3: 1.45, 4: 1.60},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FormulationScreenResult:
    """Results from formulation strategy screening.

    Attributes
    ----------
    drug_name:
        Compound identifier.
    bcs_class:
        BCS class (1-4) used for scoring.
    strategies:
        Ranked list of formulation strategy names (highest score first).
    scores:
        Corresponding scores (0-100) for each strategy.
    primary_recommendation:
        Top-ranked strategy name.
    expected_f_improvement:
        Predicted bioavailability fraction after applying the primary strategy.
    risk_level:
        Formulation development risk: 'low', 'medium', or 'high'.
    notes:
        Human-readable summary.
    """

    drug_name: str
    bcs_class: int
    strategies: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    primary_recommendation: str = ""
    expected_f_improvement: float = 0.0
    risk_level: str = "medium"
    notes: str = ""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def screen_formulations(
    drug_name: str,
    bcs_class: int,
    logP: float,
    mw: float,
    pka: float | None = None,
    solubility_mg_mL: float | None = None,
    target_dose_mg: float = 100.0,
) -> FormulationScreenResult:
    """Screen and rank formulation strategies for a drug candidate.

    Scores are derived from BCS-class base scores adjusted for
    physicochemical properties (logP, MW, solubility, pKa).

    Parameters
    ----------
    drug_name:
        Compound identifier.
    bcs_class:
        BCS class (1, 2, 3, or 4).
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).
    pka:
        Most acidic/basic pKa (optional).  Used for pH-adjusted formulation
        scoring.
    solubility_mg_mL:
        Aqueous solubility (mg/mL, optional).  Low values boost solubility-
        enhancement strategies.
    target_dose_mg:
        Target oral dose (mg).  Higher doses worsen solubility challenge.

    Returns
    -------
    FormulationScreenResult

    Raises
    ------
    ValueError
        If bcs_class is not in {1, 2, 3, 4}.
    """
    _validate_bcs_class(bcs_class)

    scored: list[tuple[float, str]] = []
    for name, class_scores, _desc in _STRATEGIES:
        base = class_scores[bcs_class]
        adjusted = _adjust_score(
            base, name, bcs_class, logP, mw, pka, solubility_mg_mL, target_dose_mg
        )
        scored.append((adjusted, name))

    # Sort descending by score, then alphabetically for stability
    scored.sort(key=lambda x: (-x[0], x[1]))

    strategies = [name for _, name in scored]
    scores = [round(score, 2) for score, _ in scored]

    primary = strategies[0] if strategies else "simple_tablet_capsule"
    base_f = _BASE_F_BY_CLASS[bcs_class]
    expected_f = estimate_bioavailability_improvement(base_f, primary, bcs_class, logP)

    risk = _RISK_BY_CLASS[bcs_class]
    notes = _build_notes(drug_name, bcs_class, primary, expected_f, risk, logP, mw)

    return FormulationScreenResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        strategies=strategies,
        scores=scores,
        primary_recommendation=primary,
        expected_f_improvement=expected_f,
        risk_level=risk,
        notes=notes,
    )


def get_formulation_recommendation(
    bcs_class: int,
    logP: float,
    mw: float,
) -> dict:
    """Return primary and alternative formulation recommendations.

    Parameters
    ----------
    bcs_class:
        BCS class (1-4).
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).

    Returns
    -------
    dict with keys:
        'primary'      : str  -- top recommended strategy
        'alternatives' : list[str]  -- next 2-3 alternatives
        'rationale'    : str  -- explanation of the choice

    Raises
    ------
    ValueError
        If bcs_class is not in {1, 2, 3, 4}.
    """
    _validate_bcs_class(bcs_class)

    result = screen_formulations(
        drug_name="query",
        bcs_class=bcs_class,
        logP=logP,
        mw=mw,
    )

    alternatives = result.strategies[1:4] if len(result.strategies) > 1 else []
    rationale = _build_rationale(bcs_class, result.primary_recommendation, logP, mw)

    return {
        "primary": result.primary_recommendation,
        "alternatives": alternatives,
        "rationale": rationale,
    }


def estimate_bioavailability_improvement(
    current_f: float,
    formulation_strategy: str,
    bcs_class: int,
    logP: float,
) -> float:
    """Predict new bioavailability fraction after applying a formulation strategy.

    Parameters
    ----------
    current_f:
        Current oral bioavailability fraction (0-1).
    formulation_strategy:
        Strategy name (must match keys in _F_IMPROVEMENT_FACTOR).
    bcs_class:
        BCS class (1-4).
    logP:
        logP value (used for SEDDS/lipid formulation bonus).

    Returns
    -------
    float
        Predicted bioavailability fraction after formulation, capped at 0.95.

    Raises
    ------
    ValueError
        If current_f is outside [0, 1], bcs_class invalid, or strategy unknown.
    """
    _validate_bcs_class(bcs_class)
    if not (0.0 <= current_f <= 1.0):
        raise ValueError(f"current_f must be in [0, 1], got {current_f}")

    factor_map = _F_IMPROVEMENT_FACTOR.get(formulation_strategy)
    if factor_map is None:
        raise ValueError(
            f"Unknown formulation_strategy '{formulation_strategy}'. "
            f"Valid options: {sorted(_F_IMPROVEMENT_FACTOR)}"
        )

    factor = factor_map[bcs_class]

    # Bonus for lipophilic compounds with lipid-based / SEDDS strategies
    if formulation_strategy in {"SEDDS_SMEDDS", "lipid_based_formulation"} and logP >= _HIGH_LOGP:
        factor = factor * 1.10

    new_f = current_f * factor
    return float(round(min(new_f, _MAX_BIOAVAILABILITY), 4))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_bcs_class(bcs_class: int) -> None:
    if bcs_class not in _VALID_BCS_CLASSES:
        raise ValueError(f"bcs_class must be one of {sorted(_VALID_BCS_CLASSES)}, got {bcs_class}")


def _adjust_score(
    base_score: float,
    strategy_name: str,
    bcs_class: int,
    logP: float,
    mw: float,
    pka: float | None,
    solubility_mg_mL: float | None,
    target_dose_mg: float,
) -> float:
    """Apply physicochemical modifier to a strategy's base score."""
    score = base_score

    # SEDDS/lipid bonuses for lipophilic BCS II drugs
    if strategy_name in {"SEDDS_SMEDDS", "lipid_based_formulation"} and bcs_class in {2, 4}:
        if logP >= _VERY_HIGH_LOGP:
            score = min(score + 10.0, 100.0)
        elif logP >= _HIGH_LOGP:
            score = min(score + 5.0, 100.0)

    # ASD bonus when solubility is very low
    if strategy_name == "amorphous_solid_dispersion" and bcs_class in {2, 4}:
        if solubility_mg_mL is not None and solubility_mg_mL < 0.01:
            score = min(score + 8.0, 100.0)

    # pH formulation boost when pKa in physiological range
    if strategy_name == "pH_adjusted_formulation" and pka is not None:
        if 4.0 <= pka <= 9.0:
            score = min(score + 5.0, 100.0)

    # Permeation enhancer penalty for large MW molecules
    if strategy_name in {"permeation_enhancers", "bioadhesive_system"} and mw > _HIGH_MW:
        score = max(score - 10.0, 0.0)

    # Simple tablet penalty for high-dose, low-solubility drugs
    if strategy_name == "simple_tablet_capsule" and bcs_class in {2, 4}:
        if solubility_mg_mL is not None and target_dose_mg > 0:
            dose_number = (target_dose_mg / 250.0) / max(solubility_mg_mL, 1e-9)
            if dose_number > 10.0:
                score = max(score - 15.0, 0.0)

    # Clamp
    return float(max(0.0, min(score, 100.0)))


def _build_notes(
    drug_name: str,
    bcs_class: int,
    primary: str,
    expected_f: float,
    risk: str,
    logP: float,
    mw: float,
) -> str:
    """Build a human-readable notes string."""
    parts = [
        f"{drug_name}: BCS Class {bcs_class}.",
        f"Primary recommendation: {primary}.",
        f"Expected bioavailability: {expected_f * 100:.1f}%.",
        f"Development risk: {risk}.",
    ]
    if logP >= _VERY_HIGH_LOGP:
        parts.append("High logP suggests lipid-based formulations preferred.")
    if mw > _HIGH_MW:
        parts.append("High MW may limit permeation enhancer efficacy.")
    return " ".join(parts)


def _build_rationale(
    bcs_class: int,
    primary: str,
    logP: float,
    mw: float,
) -> str:
    """Build a rationale string for get_formulation_recommendation."""
    rationale_map: dict[int, str] = {
        1: "BCS Class I has both high solubility and permeability; "
        "simple conventional formulations are preferred to minimise complexity.",
        2: "BCS Class II is dissolution-limited; solubility enhancement "
        "strategies (ASD, SEDDS, nanosuspension) are the primary focus.",
        3: "BCS Class III is permeability-limited; permeation enhancers and "
        "bioadhesive systems can extend GI residence and increase absorption.",
        4: "BCS Class IV requires combination approaches to address both "
        "low solubility and low permeability; development risk is high.",
    }
    base = rationale_map[bcs_class]
    if logP >= _VERY_HIGH_LOGP and bcs_class in {2, 4}:
        base += f" High logP ({logP:.1f}) further supports lipid-based delivery."
    return base


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "FormulationScreenResult",
    "screen_formulations",
    "get_formulation_recommendation",
    "estimate_bioavailability_improvement",
]
