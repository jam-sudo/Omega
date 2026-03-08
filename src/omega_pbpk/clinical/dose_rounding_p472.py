"""Phase 472 — Drug Dose Rounding and Formulation Selection.

Utility for clinical dose rounding to available tablet/capsule strengths.
Rounds computed doses to nearest available strength, checks if rounded dose
stays within acceptable range (+/-10%), and handles tablet splitting for
scored tablets.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DoseRoundingResult",
    "round_dose",
    "find_optimal_regimen",
    "AVAILABLE_STRENGTHS",
    "SCORED_TABLETS",
]

# Available tablet/capsule strengths in mg
AVAILABLE_STRENGTHS: dict[str, list[float]] = {
    "metformin": [500.0, 750.0, 850.0, 1000.0],
    "atorvastatin": [10.0, 20.0, 40.0, 80.0],
    "lisinopril": [2.5, 5.0, 10.0, 20.0, 40.0],
    "omeprazole": [10.0, 20.0, 40.0],
    "amlodipine": [2.5, 5.0, 10.0],
    "generic": [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0],
}

# Scored tablets that can be split in half
SCORED_TABLETS: set[str] = {"metformin", "atorvastatin", "lisinopril"}


@dataclass(frozen=True)
class DoseRoundingResult:
    """Result of a dose rounding calculation."""

    drug_name: str
    computed_dose_mg: float
    rounded_dose_mg: float
    available_strengths_mg: tuple
    rounding_error_pct: float
    within_acceptable_range: bool
    tablet_count: float
    splitting_required: bool
    splitting_allowed: bool
    recommendation: str
    notes: str


def round_dose(drug_name: str, computed_dose_mg: float) -> DoseRoundingResult:
    """Round a computed dose to the nearest available tablet/capsule strength.

    Parameters
    ----------
    drug_name:
        Name of the drug. Used to look up available strengths.
        Falls back to "generic" strengths if drug not found.
    computed_dose_mg:
        Computed target dose in milligrams. Must be > 0.

    Returns
    -------
    DoseRoundingResult
    """
    if computed_dose_mg <= 0:
        raise ValueError(f"computed_dose_mg must be > 0, got {computed_dose_mg}.")

    drug_key = drug_name.lower()
    if drug_key not in AVAILABLE_STRENGTHS:
        drug_key = "generic"

    strengths = AVAILABLE_STRENGTHS[drug_key]
    strengths_tuple = tuple(float(s) for s in strengths)
    is_scored = drug_name.lower() in SCORED_TABLETS

    # Build candidate options: full tablets + half tablets if scored
    candidates: list[
        tuple[float, float, float]
    ] = []  # (candidate_dose, tablet_strength, tablet_count)
    for strength in strengths:
        # Full tablet
        candidates.append((strength, strength, 1.0))
        # Multiple tablets (up to 4)
        for n in [2, 3, 4]:
            candidates.append((strength * n, strength, float(n)))
        # Half tablet if scored
        if is_scored:
            half = strength / 2.0
            candidates.append((half, strength, 0.5))

    # Find nearest candidate by absolute difference
    best_candidate = min(candidates, key=lambda c: abs(c[0] - computed_dose_mg))
    rounded_dose, tablet_strength, tablet_count = best_candidate

    rounding_error_pct = abs(rounded_dose - computed_dose_mg) / computed_dose_mg * 100.0
    within_acceptable_range = rounding_error_pct <= 10.0
    splitting_required = tablet_count == 0.5
    splitting_allowed = drug_name.lower() in SCORED_TABLETS

    # Build recommendation string
    if tablet_count == 0.5:
        rec = f"Take 0.5 tablet(s) of {tablet_strength:.4g} mg (split tablet)"
    elif tablet_count == 1.0:
        rec = f"Take 1.0 tablet(s) of {rounded_dose:.4g} mg"
    else:
        rec = f"Take {tablet_count:.4g} tablet(s) of {tablet_strength:.4g} mg"

    # Build notes
    notes_parts = []
    if not within_acceptable_range:
        notes_parts.append(
            f"Rounding error ({rounding_error_pct:.1f}%) exceeds 10% threshold; "
            "consider alternate dosing strategy."
        )
    if splitting_required and not splitting_allowed:
        notes_parts.append(
            "Splitting required but this tablet is not scored; "
            "consult pharmacist for alternative formulation."
        )
    if drug_key == "generic":
        notes_parts.append(f"Drug '{drug_name}' not in database; using generic strength table.")
    notes = " ".join(notes_parts) if notes_parts else "Dose rounding within acceptable limits."

    return DoseRoundingResult(
        drug_name=drug_name,
        computed_dose_mg=computed_dose_mg,
        rounded_dose_mg=rounded_dose,
        available_strengths_mg=strengths_tuple,
        rounding_error_pct=rounding_error_pct,
        within_acceptable_range=within_acceptable_range,
        tablet_count=tablet_count,
        splitting_required=splitting_required,
        splitting_allowed=splitting_allowed,
        recommendation=rec,
        notes=notes,
    )


def find_optimal_regimen(
    drug_name: str,
    daily_dose_mg: float,
    doses_per_day: int,
) -> DoseRoundingResult:
    """Find the optimal dosing regimen given a daily dose and frequency.

    Divides the daily dose by doses_per_day and rounds the per-dose amount
    to the nearest available tablet strength.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    daily_dose_mg:
        Total daily dose in milligrams. Must be > 0.
    doses_per_day:
        Number of doses per day. Must be >= 1.

    Returns
    -------
    DoseRoundingResult
        Result for a single dose (per-dose amount rounded).
    """
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}.")
    if doses_per_day < 1:
        raise ValueError(f"doses_per_day must be >= 1, got {doses_per_day}.")

    per_dose_mg = daily_dose_mg / doses_per_day
    return round_dose(drug_name, per_dose_mg)
