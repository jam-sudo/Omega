"""Phase 227: Oxidative stability predictor for drug candidates.

Predicts susceptibility of drug candidates to oxidative degradation based on
structural features and physicochemical properties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OxidativeStabilityResult:
    """Result of oxidative stability prediction."""

    drug_name: str
    has_phenol: bool
    has_catechol: bool
    has_thioether: bool
    has_aldehyde: bool
    logP: float
    mw_Da: float
    risk_score: float
    stability_class: str
    predicted_shelf_life_months: int
    antioxidant_recommendation: str
    notes: str


def predict_oxidative_stability(
    drug_name: str,
    has_phenol: bool = False,
    has_catechol: bool = False,
    has_thioether: bool = False,
    has_aldehyde: bool = False,
    logP: float = 0.0,
    mw_Da: float = 300.0,
) -> OxidativeStabilityResult:
    """Predict oxidative stability of a drug candidate.

    Args:
        drug_name: Name of the drug compound.
        has_phenol: Whether the drug contains a phenol group (+25 points).
        has_catechol: Whether the drug contains a catechol group (+40 points).
        has_thioether: Whether the drug contains a thioether group (+20 points).
        has_aldehyde: Whether the drug contains an aldehyde group (+30 points).
        logP: Lipophilicity; logP > 3 adds +10 points.
        mw_Da: Molecular weight in Da; mw > 500 adds +5 points.

    Returns:
        OxidativeStabilityResult with risk score, class, shelf life, and recommendations.

    Raises:
        ValueError: If mw_Da <= 0.
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive, got {mw_Da}")

    # Compute risk score
    score = 0.0
    if has_phenol:
        score += 25.0
    if has_catechol:
        score += 40.0
    if has_thioether:
        score += 20.0
    if has_aldehyde:
        score += 30.0
    if logP > 3.0:
        score += 10.0
    if mw_Da > 500.0:
        score += 5.0

    # Cap at 100
    risk_score = min(score, 100.0)

    # Determine stability class
    if risk_score <= 20.0:
        stability_class = "stable"
        predicted_shelf_life_months = 24
        antioxidant_recommendation = (
            "No antioxidant required; standard inert atmosphere storage sufficient."
        )
    elif risk_score <= 50.0:
        stability_class = "moderate_risk"
        predicted_shelf_life_months = 12
        antioxidant_recommendation = (
            "Consider BHA/BHT (0.01-0.1%) or ascorbic acid; store under nitrogen."
        )
    else:
        stability_class = "high_risk"
        predicted_shelf_life_months = 3
        antioxidant_recommendation = (
            "Use EDTA chelator + alpha-tocopherol + ascorbic acid; "
            "nitrogen purge; light-protected amber vials."
        )

    # Build notes
    flags = []
    if has_catechol:
        flags.append("catechol → prone to quinone formation")
    if has_phenol:
        flags.append("phenol → easily oxidized")
    if has_aldehyde:
        flags.append("aldehyde → readily oxidized to carboxylic acid")
    if has_thioether:
        flags.append("thioether → sulfoxide/sulfone formation risk")
    if logP > 3.0:
        flags.append("high logP → mitochondrial/microsomal exposure")
    if mw_Da > 500.0:
        flags.append("high MW → complex molecule; elevated reactivity")
    notes = "; ".join(flags) if flags else "No major oxidative liabilities identified."

    return OxidativeStabilityResult(
        drug_name=drug_name,
        has_phenol=has_phenol,
        has_catechol=has_catechol,
        has_thioether=has_thioether,
        has_aldehyde=has_aldehyde,
        logP=logP,
        mw_Da=mw_Da,
        risk_score=risk_score,
        stability_class=stability_class,
        predicted_shelf_life_months=predicted_shelf_life_months,
        antioxidant_recommendation=antioxidant_recommendation,
        notes=notes,
    )


def screen_oxidative_stability(
    compounds: list[dict],
) -> list[OxidativeStabilityResult]:
    """Screen a list of compounds for oxidative stability.

    Each compound dict may contain keys: drug_name, has_phenol, has_catechol,
    has_thioether, has_aldehyde, logP, mw_Da. Missing keys use defaults.

    Returns:
        List of OxidativeStabilityResult sorted by risk_score ascending
        (most stable first).

    Raises:
        ValueError: If any compound has invalid mw_Da.
    """
    results = []
    for compound in compounds:
        result = predict_oxidative_stability(
            drug_name=compound.get("drug_name", "unknown"),
            has_phenol=compound.get("has_phenol", False),
            has_catechol=compound.get("has_catechol", False),
            has_thioether=compound.get("has_thioether", False),
            has_aldehyde=compound.get("has_aldehyde", False),
            logP=compound.get("logP", 0.0),
            mw_Da=compound.get("mw_Da", 300.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.risk_score)
    return results
