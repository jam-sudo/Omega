"""Skin sensitization and contact allergy risk assessment."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SkinSensitizationResult",
    "assess_skin_sensitization",
    "screen_skin_sensitization",
    "estimate_ec3_value",
]


@dataclass(frozen=True)
class SkinSensitizationResult:
    """Result of skin sensitization risk assessment."""

    compound_name: str
    skin_sensitization_score: float  # 0-100
    sensitization_class: str  # non_sensitizer / low / moderate / strong / extreme
    n_structural_alerts: int
    dpra_category: str  # non_reactive / low_moderate / moderate_high / high_extreme
    requires_llna_testing: bool  # score > 20
    occupational_hazard: bool  # isocyanates > 0 or score > 60
    notes: str


def _classify_sensitization(score: float) -> str:
    if score <= 0:
        return "non_sensitizer"
    if score <= 20:
        return "low"
    if score <= 50:
        return "moderate"
    if score <= 80:
        return "strong"
    return "extreme"


def _dpra_category(score: float) -> str:
    if score < 10:
        return "non_reactive"
    if score < 40:
        return "low_moderate"
    if score <= 70:
        return "moderate_high"
    return "high_extreme"


def assess_skin_sensitization(
    compound_name: str,
    logP: float,
    mw: float,
    smiles_features: dict | None = None,
) -> SkinSensitizationResult:
    """Assess skin sensitization risk.

    Args:
        compound_name: Name of the compound.
        logP: Octanol-water partition coefficient.
        mw: Molecular weight (g/mol).
        smiles_features: Dict of structural alert counts/flags.

    Returns:
        SkinSensitizationResult with score, classification, DPRA category, and flags.
    """
    if not compound_name:
        raise ValueError("compound_name must be nonempty.")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}.")

    if smiles_features is None:
        smiles_features = {}

    # Extract features with defaults
    n_michael = int(smiles_features.get("n_michael_acceptors", 0))
    n_isocyanates = int(smiles_features.get("n_isocyanates", 0))
    n_epoxides = int(smiles_features.get("n_epoxides_skin", 0))
    n_aldehydes = int(smiles_features.get("n_aldehydes_skin", 0))
    n_acrylates = int(smiles_features.get("n_acrylates", 0))
    n_quinones = int(smiles_features.get("n_quinones", 0))
    n_aromatic_amines = int(smiles_features.get("n_aromatic_amines_skin", 0))
    has_protein_reactive = bool(smiles_features.get("has_protein_reactive_group", False))

    # Validate counts
    for name, val in [
        ("n_michael_acceptors", n_michael),
        ("n_isocyanates", n_isocyanates),
        ("n_epoxides_skin", n_epoxides),
        ("n_aldehydes_skin", n_aldehydes),
        ("n_acrylates", n_acrylates),
        ("n_quinones", n_quinones),
        ("n_aromatic_amines_skin", n_aromatic_amines),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}.")

    # Base score from structural alerts
    score = 0.0
    score += 20.0 * n_michael
    score += 40.0 * n_isocyanates
    score += 25.0 * n_epoxides
    score += 15.0 * n_aldehydes
    score += 20.0 * n_acrylates
    score += 18.0 * n_quinones
    score += 12.0 * n_aromatic_amines
    if has_protein_reactive:
        score += 30.0

    # Physicochemical modifiers
    modifier = 1.0
    if 1.0 <= logP <= 4.0:
        modifier = 1.2  # optimal skin penetration
    elif logP < -1.0 or logP > 6.0:
        modifier = 0.7  # poor skin penetration

    if mw > 1000:
        modifier *= 0.5  # large molecules don't penetrate skin

    score = min(100.0, score * modifier)

    # Count structural alerts
    n_alerts = (
        n_michael
        + n_isocyanates
        + n_epoxides
        + n_aldehydes
        + n_acrylates
        + n_quinones
        + n_aromatic_amines
        + (1 if has_protein_reactive else 0)
    )

    sensitization_class = _classify_sensitization(score)
    dpra_cat = _dpra_category(score)
    requires_llna = score > 20.0
    occ_hazard = n_isocyanates > 0 or score > 60.0

    # Build notes
    notes_parts = [f"logP={logP:.2f}, MW={mw:.1f}"]
    if n_isocyanates > 0:
        notes_parts.append("isocyanate group detected — occupational sensitizer")
    if score >= 80:
        notes_parts.append("extreme sensitization potential")
    elif score >= 51:
        notes_parts.append("strong sensitization potential")
    elif score >= 21:
        notes_parts.append("moderate sensitization potential")
    elif score >= 1:
        notes_parts.append("low sensitization potential")
    else:
        notes_parts.append("no sensitization alerts detected")

    notes = "; ".join(notes_parts)

    return SkinSensitizationResult(
        compound_name=compound_name,
        skin_sensitization_score=round(score, 2),
        sensitization_class=sensitization_class,
        n_structural_alerts=n_alerts,
        dpra_category=dpra_cat,
        requires_llna_testing=requires_llna,
        occupational_hazard=occ_hazard,
        notes=notes,
    )


def screen_skin_sensitization(
    compounds: list[dict],
) -> list[SkinSensitizationResult]:
    """Screen a list of compounds for skin sensitization.

    Each dict must contain: compound_name, logP, mw.
    Optional key: smiles_features (dict).

    Returns list sorted by skin_sensitization_score descending.
    """
    results = []
    for comp in compounds:
        result = assess_skin_sensitization(
            compound_name=comp["compound_name"],
            logP=comp["logP"],
            mw=comp["mw"],
            smiles_features=comp.get("smiles_features"),
        )
        results.append(result)

    results.sort(key=lambda r: r.skin_sensitization_score, reverse=True)
    return results


def estimate_ec3_value(sensitization_score: float, logP: float) -> float:
    """Estimate EC3 value (% w/v) for LLNA test.

    EC3 = effective concentration causing 3% stimulation index.
    log_EC3 = 3.0 - 0.03 * score - 0.1 * max(0, logP - 2)
    EC3 = clamp(10^log_EC3, 0.001, 100.0)
    """
    log_ec3 = 3.0 - 0.03 * sensitization_score - 0.1 * max(0.0, logP - 2.0)
    ec3 = 10.0**log_ec3
    return max(0.001, min(100.0, ec3))
