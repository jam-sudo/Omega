"""Reproductive toxicity risk assessment based on structural alerts and physicochemical properties."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ReproductiveToxicityResult",
    "assess_reproductive_toxicity",
    "screen_reproductive_toxicity",
]


@dataclass(frozen=True)
class ReproductiveToxicityResult:
    """Result of reproductive toxicity assessment for a single compound."""

    compound_name: str
    reproductive_toxicity_score: float  # 0-100
    risk_class: str  # none / low / moderate / high / very_high
    n_structural_alerts: int
    pregnancy_category: str  # A / B / C / D / X
    placental_transfer_concern: bool
    teratogenicity_flag: bool
    requires_reproductive_study: bool
    notes: str


def _classify_risk(score: float) -> str:
    if score <= 0:
        return "none"
    elif score <= 20:
        return "low"
    elif score <= 50:
        return "moderate"
    elif score <= 80:
        return "high"
    else:
        return "very_high"


def _pregnancy_category(score: float) -> str:
    if score <= 10:
        return "A"
    elif score <= 25:
        return "B"
    elif score <= 50:
        return "C"
    elif score <= 75:
        return "D"
    else:
        return "X"


def assess_reproductive_toxicity(
    compound_name: str,
    mw: float,
    logP: float,
    smiles_features: dict,
) -> ReproductiveToxicityResult:
    """Assess reproductive toxicity risk for a compound.

    Parameters
    ----------
    compound_name : str
        Identifier for the compound.
    mw : float
        Molecular weight in g/mol.
    logP : float
        Lipophilicity (log octanol-water partition coefficient).
    smiles_features : dict
        Structural feature counts/flags (all optional):
        n_nitrosamines, n_estrogen_like, n_retinoid_like, n_alkylating,
        n_heavy_metals, is_planar_aromatic, n_thalidomide_like
    """
    if not compound_name:
        raise ValueError("compound_name must be nonempty")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")

    n_nitrosamines = int(smiles_features.get("n_nitrosamines", 0))
    n_estrogen_like = int(smiles_features.get("n_estrogen_like", 0))
    n_retinoid_like = int(smiles_features.get("n_retinoid_like", 0))
    n_alkylating = int(smiles_features.get("n_alkylating", 0))
    n_heavy_metals = int(smiles_features.get("n_heavy_metals", 0))
    is_planar_aromatic = bool(smiles_features.get("is_planar_aromatic", False))
    n_thalidomide_like = int(smiles_features.get("n_thalidomide_like", 0))

    # Validate counts are non-negative
    for field_name, val in [
        ("n_nitrosamines", n_nitrosamines),
        ("n_estrogen_like", n_estrogen_like),
        ("n_retinoid_like", n_retinoid_like),
        ("n_alkylating", n_alkylating),
        ("n_heavy_metals", n_heavy_metals),
        ("n_thalidomide_like", n_thalidomide_like),
    ]:
        if val < 0:
            raise ValueError(f"{field_name} must be >= 0, got {val}")

    # Base score from structural alerts
    score = 0.0
    score += n_nitrosamines * 35.0
    score += n_estrogen_like * 20.0
    score += n_retinoid_like * 30.0
    score += n_alkylating * 25.0
    score += n_heavy_metals * 40.0
    score += 10.0 if is_planar_aromatic else 0.0
    score += n_thalidomide_like * 45.0

    # MW adjustment for placental transfer
    if mw > 1000:
        score *= 0.4
    elif mw > 800:
        score *= 0.6

    # logP adjustment for membrane penetration
    if logP < -2:
        score *= 0.7

    # Clamp to 0-100
    score = min(100.0, max(0.0, score))

    # Count structural alerts
    n_structural_alerts = (
        n_nitrosamines
        + n_estrogen_like
        + n_retinoid_like
        + n_alkylating
        + n_heavy_metals
        + (1 if is_planar_aromatic else 0)
        + n_thalidomide_like
    )

    risk_class = _classify_risk(score)
    pregnancy_category = _pregnancy_category(score)

    # Placental transfer concern: small (mw < 600) and lipophilic (logP > -1)
    placental_transfer_concern = mw < 600 and logP > -1

    # Teratogenicity flag
    teratogenicity_flag = n_retinoid_like > 0 or n_thalidomide_like > 0 or score > 50

    # Requires reproductive study
    requires_reproductive_study = score > 20

    # Build notes
    note_parts = []
    if n_nitrosamines > 0:
        note_parts.append(f"{n_nitrosamines} N-nitroso group(s) (IARC Group 1)")
    if n_retinoid_like > 0:
        note_parts.append(f"{n_retinoid_like} retinoid-like structure(s) (teratogen)")
    if n_thalidomide_like > 0:
        note_parts.append(f"{n_thalidomide_like} thalidomide-like ring(s)")
    if n_heavy_metals > 0:
        note_parts.append(f"{n_heavy_metals} heavy metal chelate(s)")
    if n_estrogen_like > 0:
        note_parts.append(f"{n_estrogen_like} estrogen-like group(s) (endocrine disruption)")
    if n_alkylating > 0:
        note_parts.append(f"{n_alkylating} alkylating group(s) (gonadotoxic)")
    if is_planar_aromatic:
        note_parts.append("planar aromatic (intercalation risk)")
    if mw > 800:
        note_parts.append(f"high MW ({mw:.0f} g/mol) reduces placental transfer")
    if logP < -2:
        note_parts.append(f"low logP ({logP:.1f}) reduces membrane penetration")
    if not note_parts:
        note_parts.append("No structural alerts detected")

    notes = (
        "; ".join(note_parts)
        + f". Risk class: {risk_class}. Pregnancy category: {pregnancy_category}."
    )

    return ReproductiveToxicityResult(
        compound_name=compound_name,
        reproductive_toxicity_score=round(score, 2),
        risk_class=risk_class,
        n_structural_alerts=n_structural_alerts,
        pregnancy_category=pregnancy_category,
        placental_transfer_concern=placental_transfer_concern,
        teratogenicity_flag=teratogenicity_flag,
        requires_reproductive_study=requires_reproductive_study,
        notes=notes,
    )


def screen_reproductive_toxicity(compounds: list[dict]) -> list[ReproductiveToxicityResult]:
    """Screen a list of compounds for reproductive toxicity.

    Each compound dict must have: compound_name, mw, logP
    Optional: smiles_features dict

    Returns results sorted by reproductive_toxicity_score descending.
    """
    results = []
    for compound in compounds:
        name = compound["compound_name"]
        mw = float(compound["mw"])
        logP = float(compound["logP"])
        smiles_features = compound.get("smiles_features", {})
        result = assess_reproductive_toxicity(name, mw, logP, smiles_features)
        results.append(result)

    return sorted(results, key=lambda r: r.reproductive_toxicity_score, reverse=True)
