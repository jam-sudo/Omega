"""Phase 466 — Drug Crystal Polymorphism PK Impact.

Models the impact of crystal polymorphism on drug dissolution rate and
oral bioavailability.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Polymorph database
# ---------------------------------------------------------------------------

_POLYMORPH_DB: dict[str, dict] = {
    "Form_I": {
        "solubility_ratio": 1.0,
        "dissolution_rate_ratio": 1.0,
        "conversion_risk": "none",
    },
    "Form_II": {
        "solubility_ratio": 1.5,
        "dissolution_rate_ratio": 2.0,
        "conversion_risk": "low",
    },
    "Form_III": {
        "solubility_ratio": 3.0,
        "dissolution_rate_ratio": 4.0,
        "conversion_risk": "high",
    },
    "Amorphous": {
        "solubility_ratio": 10.0,
        "dissolution_rate_ratio": 15.0,
        "conversion_risk": "very_high",
    },
}

_VALID_FORMS: set[str] = set(_POLYMORPH_DB.keys())

_STABILITY_RISK_MAP: dict[str, float] = {
    "none": 0.0,
    "low": 25.0,
    "high": 75.0,
    "very_high": 95.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolymorphismPKResult:
    """Result of crystal polymorphism PK prediction."""

    drug_name: str
    polymorph_form: str
    solubility_ratio: float
    dissolution_rate_ratio: float
    conversion_risk: str
    intrinsic_solubility_mg_mL: float
    predicted_fa: float
    predicted_cmax_ratio: float
    bioavailability_advantage: bool
    stability_risk_score: float
    recommended_for_development: bool
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_polymorphism_pk(
    drug_name: str,
    polymorph_form: str,
    base_solubility_mg_mL: float = 0.1,
    base_fa: float = 0.3,
) -> PolymorphismPKResult:
    """Predict PK impact of a drug crystal polymorph.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    polymorph_form:
        One of "Form_I", "Form_II", "Form_III", "Amorphous".
    base_solubility_mg_mL:
        Intrinsic solubility of Form_I in mg/mL (> 0).
    base_fa:
        Baseline fraction absorbed for Form_I (0 < base_fa <= 1).

    Returns
    -------
    PolymorphismPKResult
    """
    # --- Validation ---
    if polymorph_form not in _VALID_FORMS:
        raise ValueError(
            f"polymorph_form must be one of {sorted(_VALID_FORMS)}, got '{polymorph_form}'"
        )
    if base_solubility_mg_mL <= 0:
        raise ValueError(f"base_solubility_mg_mL must be > 0, got {base_solubility_mg_mL}")
    if not (0 < base_fa <= 1.0):
        raise ValueError(f"base_fa must be in (0, 1], got {base_fa}")

    entry = _POLYMORPH_DB[polymorph_form]
    solubility_ratio: float = entry["solubility_ratio"]
    dissolution_rate_ratio: float = entry["dissolution_rate_ratio"]
    conversion_risk: str = entry["conversion_risk"]

    # --- Derived values ---
    intrinsic_solubility_mg_mL = base_solubility_mg_mL * solubility_ratio

    # BCS-like fa model: higher solubility → higher absorption, capped at 1.0
    predicted_fa = min(1.0, solubility_ratio * base_fa)

    # Cmax ratio relative to Form_I (cap at 5.0)
    predicted_cmax_ratio = min(5.0, dissolution_rate_ratio)

    bioavailability_advantage = predicted_fa > base_fa

    stability_risk_score = _STABILITY_RISK_MAP[conversion_risk]

    recommended_for_development = stability_risk_score < 50.0 and bioavailability_advantage

    notes = _build_notes(
        polymorph_form,
        solubility_ratio,
        dissolution_rate_ratio,
        conversion_risk,
        predicted_fa,
        base_fa,
        stability_risk_score,
        recommended_for_development,
    )

    return PolymorphismPKResult(
        drug_name=drug_name,
        polymorph_form=polymorph_form,
        solubility_ratio=solubility_ratio,
        dissolution_rate_ratio=dissolution_rate_ratio,
        conversion_risk=conversion_risk,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        predicted_fa=predicted_fa,
        predicted_cmax_ratio=predicted_cmax_ratio,
        bioavailability_advantage=bioavailability_advantage,
        stability_risk_score=stability_risk_score,
        recommended_for_development=recommended_for_development,
        notes=notes,
    )


def _build_notes(
    polymorph_form: str,
    solubility_ratio: float,
    dissolution_rate_ratio: float,
    conversion_risk: str,
    predicted_fa: float,
    base_fa: float,
    stability_risk_score: float,
    recommended: bool,
) -> str:
    """Build a human-readable notes string."""
    parts: list[str] = []
    parts.append(
        f"{polymorph_form}: solubility ratio {solubility_ratio:.1f}x Form_I, "
        f"dissolution rate {dissolution_rate_ratio:.1f}x Form_I."
    )
    parts.append(
        f"Conversion risk: {conversion_risk}. Stability risk score: {stability_risk_score:.0f}/100."
    )
    if predicted_fa > base_fa:
        parts.append(
            f"Bioavailability advantage over Form_I (fa {base_fa:.2f} -> {predicted_fa:.2f})."
        )
    else:
        parts.append("No bioavailability advantage over Form_I.")
    if recommended:
        parts.append("Recommended for development: favorable balance of efficacy and stability.")
    else:
        parts.append("Not recommended for development as primary solid form.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_polymorphs(
    drug_name: str,
    base_solubility_mg_mL: float = 0.1,
    base_fa: float = 0.3,
) -> list[PolymorphismPKResult]:
    """Compare all polymorph forms for a drug, sorted by predicted_fa descending."""
    results = [
        predict_polymorphism_pk(
            drug_name=drug_name,
            polymorph_form=form,
            base_solubility_mg_mL=base_solubility_mg_mL,
            base_fa=base_fa,
        )
        for form in _VALID_FORMS
    ]
    results.sort(key=lambda r: r.predicted_fa, reverse=True)
    return results
