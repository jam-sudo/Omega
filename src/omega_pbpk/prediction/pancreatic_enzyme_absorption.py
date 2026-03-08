"""Phase 465 — Pancreatic Enzyme Effect on Drug Absorption.

Models the effect of pancreatic enzyme activity on drug absorption for
pro-drugs and enzyme-labile compounds.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_DRUG_TYPES: set[str] = {"prodrug", "stable", "labile"}

_VALID_CONDITIONS: set[str] = {
    "normal",
    "pancreatitis",
    "pancreatic_insufficiency",
    "post_surgical",
}

# Enzyme activity relative to normal (1.0)
_ENZYME_FACTORS: dict[str, float] = {
    "normal": 1.0,
    "pancreatitis": 0.3,
    "pancreatic_insufficiency": 0.1,
    "post_surgical": 0.5,
}

_RISK_SCORE_NORMAL: dict[str, float] = {
    "normal": 0.0,
    "pancreatitis": 30.0,
    "pancreatic_insufficiency": 70.0,
    "post_surgical": 40.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PancreaticEnzymeAbsorptionResult:
    """Result of pancreatic enzyme absorption prediction."""

    drug_name: str
    drug_type: str
    pancreatic_condition: str
    base_fa: float
    enzyme_factor: float
    fa_adjusted: float
    dose_absorbed_mg: float
    bioavailability_change_pct: float
    dose_recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_pancreatic_enzyme_absorption(
    drug_name: str,
    drug_type: str,
    pancreatic_condition: str,
    dose_mg: float,
    base_fa: float = 0.8,
) -> PancreaticEnzymeAbsorptionResult:
    """Predict drug absorption adjusted for pancreatic enzyme activity.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_type:
        One of "prodrug", "stable", or "labile".
    pancreatic_condition:
        One of "normal", "pancreatitis", "pancreatic_insufficiency",
        "post_surgical".
    dose_mg:
        Administered dose in mg.
    base_fa:
        Baseline fraction absorbed under normal enzyme conditions (0 < base_fa <= 1).

    Returns
    -------
    PancreaticEnzymeAbsorptionResult
    """
    # --- Validation ---
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got '{drug_type}'")
    if pancreatic_condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"pancreatic_condition must be one of {sorted(_VALID_CONDITIONS)}, "
            f"got '{pancreatic_condition}'"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < base_fa <= 1.0):
        raise ValueError(f"base_fa must be in (0, 1], got {base_fa}")

    enzyme_factor = _ENZYME_FACTORS[pancreatic_condition]

    # --- Adjusted fa ---
    if drug_type == "prodrug":
        fa_adjusted = min(1.0, base_fa * enzyme_factor)
    elif drug_type == "stable":
        fa_adjusted = base_fa
    else:  # labile
        fa_adjusted = base_fa * max(0.1, 1.0 - 0.5 * enzyme_factor)

    # --- fa under normal for this drug type (used for bioavailability_change_pct) ---
    if drug_type == "prodrug":
        fa_normal = min(1.0, base_fa * 1.0)
    elif drug_type == "stable":
        fa_normal = base_fa
    else:
        fa_normal = base_fa * max(0.1, 1.0 - 0.5 * 1.0)

    # Bioavailability change relative to normal condition
    if fa_normal > 0:
        bioavailability_change_pct = (fa_adjusted - fa_normal) / fa_normal * 100.0
    else:
        bioavailability_change_pct = 0.0

    dose_absorbed_mg = fa_adjusted * dose_mg

    # --- Dose recommendation ---
    dose_recommendation = _determine_recommendation(
        drug_type, pancreatic_condition, fa_adjusted, base_fa
    )

    # --- Notes ---
    notes = _build_notes(drug_type, pancreatic_condition, enzyme_factor, fa_adjusted, base_fa)

    return PancreaticEnzymeAbsorptionResult(
        drug_name=drug_name,
        drug_type=drug_type,
        pancreatic_condition=pancreatic_condition,
        base_fa=base_fa,
        enzyme_factor=enzyme_factor,
        fa_adjusted=fa_adjusted,
        dose_absorbed_mg=dose_absorbed_mg,
        bioavailability_change_pct=round(bioavailability_change_pct, 4),
        dose_recommendation=dose_recommendation,
        notes=notes,
    )


def _determine_recommendation(
    drug_type: str,
    pancreatic_condition: str,
    fa_adjusted: float,
    base_fa: float,
) -> str:
    """Determine dose recommendation based on drug type and condition."""
    # Override for prodrug with impaired enzyme
    if drug_type == "prodrug" and pancreatic_condition in {
        "pancreatitis",
        "pancreatic_insufficiency",
        "post_surgical",
    }:
        return "enzyme_supplement"

    # For prodrug with normal condition
    if drug_type == "prodrug":
        return "standard"

    # For stable drugs
    if drug_type == "stable":
        return "standard"

    # For labile drugs: check relative to base_fa
    ratio = fa_adjusted / base_fa if base_fa > 0 else 1.0
    if ratio > 1.5:
        return "decrease_dose"
    if fa_adjusted < 0.5 * base_fa:
        return "increase_dose"
    return "standard"


def _build_notes(
    drug_type: str,
    pancreatic_condition: str,
    enzyme_factor: float,
    fa_adjusted: float,
    base_fa: float,
) -> str:
    """Build a human-readable notes string."""
    parts: list[str] = []
    parts.append(f"Enzyme activity factor {enzyme_factor:.2f} relative to normal.")
    if drug_type == "prodrug":
        parts.append(
            "Prodrug requires enzymatic activation; reduced enzyme activity "
            "decreases bioavailability."
        )
        if pancreatic_condition != "normal":
            parts.append("Pancreatic enzyme supplementation may restore activation.")
    elif drug_type == "labile":
        parts.append(
            "Labile compound is degraded by pancreatic enzymes; lower enzyme "
            "activity may increase bioavailability."
        )
    else:
        parts.append("Stable compound is not significantly affected by pancreatic enzymes.")
    parts.append(f"Adjusted fa = {fa_adjusted:.4f} (base fa = {base_fa:.4f}).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


def screen_pancreatic_conditions(
    drug_name: str,
    drug_type: str,
    dose_mg: float,
    base_fa: float = 0.8,
) -> list[PancreaticEnzymeAbsorptionResult]:
    """Screen all pancreatic conditions for a given drug.

    Returns a list of results sorted by fa_adjusted descending.
    """
    results = [
        predict_pancreatic_enzyme_absorption(
            drug_name=drug_name,
            drug_type=drug_type,
            pancreatic_condition=condition,
            dose_mg=dose_mg,
            base_fa=base_fa,
        )
        for condition in _VALID_CONDITIONS
    ]
    results.sort(key=lambda r: r.fa_adjusted, reverse=True)
    return results
