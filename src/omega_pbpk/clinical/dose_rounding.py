"""Phase 181 / Phase 558 — Dose Rounding

Round continuous pharmacokinetic-optimised doses to available tablet/capsule
strengths. Also includes weight-based, BSA-based, and pediatric dose rounding
utilities added in Phase 558.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DoseRoundingResult",
    "round_dose",
    "select_optimal_regimen",
    # Phase 558 additions
    "round_to_available_tablet",
    "weight_based_dose_rounding",
    "bsa_based_dose_rounding",
    "pediatric_dose_by_age",
    "dose_rounding_report",
]


@dataclass(frozen=True)
class DoseRoundingResult:
    drug_name: str
    target_dose_mg: float
    rounded_dose_mg: float
    tablet_strength_mg: float
    n_tablets: int
    dose_error_pct: float  # (rounded - target) / target * 100
    regimen_description: str  # e.g. "2 x 50 mg tablets"
    within_10pct: bool
    notes: str


def round_dose(
    drug_name: str,
    target_dose_mg: float,
    available_strengths_mg: list[float],
    max_tablets: int = 4,
    prefer_fewer_tablets: bool = True,
) -> DoseRoundingResult:
    """Find the best tablet combination (whole numbers only) closest to target.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    target_dose_mg:
        Continuous PK-optimal dose in mg.
    available_strengths_mg:
        List of commercially available tablet/capsule strengths in mg.
    max_tablets:
        Maximum number of tablets allowed per dose.
    prefer_fewer_tablets:
        When two combinations produce identical fractional error, prefer the one
        with fewer tablets.

    Returns
    -------
    DoseRoundingResult
    """
    if target_dose_mg <= 0:
        raise ValueError(f"target_dose_mg must be positive, got {target_dose_mg}")
    if not available_strengths_mg:
        raise ValueError("available_strengths_mg must not be empty")
    for s in available_strengths_mg:
        if s <= 0:
            raise ValueError(f"All tablet strengths must be positive, got {s}")
    if max_tablets < 1:
        raise ValueError(f"max_tablets must be >= 1, got {max_tablets}")

    best_err: float = float("inf")
    best: tuple[float, int, float] | None = None

    for strength in available_strengths_mg:
        for n in range(1, max_tablets + 1):
            candidate = strength * n
            err = abs(candidate - target_dose_mg) / target_dose_mg
            if err < best_err or (
                prefer_fewer_tablets and err == best_err and best is not None and n < best[1]
            ):
                best_err = err
                best = (strength, n, candidate)

    assert best is not None  # guaranteed: at least one iteration
    strength, n_tablets, rounded = best

    dose_error_pct = (rounded - target_dose_mg) / target_dose_mg * 100.0
    within_10pct = abs(dose_error_pct) <= 10.0

    tablet_word = "tablet" if n_tablets == 1 else "tablets"
    regimen_description = f"{n_tablets} x {strength:.0f} mg {tablet_word}"

    notes_parts: list[str] = []
    if abs(dose_error_pct) > 20:
        notes_parts.append(
            f"Dose error ({dose_error_pct:+.1f}%) exceeds 20% — consider additional strengths."
        )
    notes = " ".join(notes_parts) if notes_parts else "Dose rounding successful."

    return DoseRoundingResult(
        drug_name=drug_name,
        target_dose_mg=target_dose_mg,
        rounded_dose_mg=rounded,
        tablet_strength_mg=strength,
        n_tablets=n_tablets,
        dose_error_pct=dose_error_pct,
        regimen_description=regimen_description,
        within_10pct=within_10pct,
        notes=notes,
    )


def select_optimal_regimen(
    drug_name: str,
    target_daily_dose_mg: float,
    available_strengths_mg: list[float],
    frequencies: list[int] | None = None,
    max_tablets_per_dose: int = 4,
) -> list[DoseRoundingResult]:
    """Try each dosing frequency and return results sorted by |dose_error_pct| ascending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    target_daily_dose_mg:
        Total daily dose in mg.
    available_strengths_mg:
        Available tablet/capsule strengths in mg.
    frequencies:
        Doses per day to evaluate (default [1, 2, 3, 4]).
    max_tablets_per_dose:
        Maximum tablets allowed per individual dose.

    Returns
    -------
    List of DoseRoundingResult sorted by ascending absolute dose error.
    """
    if frequencies is None:
        frequencies = [1, 2, 3, 4]

    results: list[DoseRoundingResult] = []
    for freq in frequencies:
        per_dose_target = target_daily_dose_mg / freq
        result = round_dose(
            drug_name=f"{drug_name} ({freq}x/day)",
            target_dose_mg=per_dose_target,
            available_strengths_mg=available_strengths_mg,
            max_tablets=max_tablets_per_dose,
        )
        results.append(result)

    results.sort(key=lambda r: abs(r.dose_error_pct))
    return results


# ---------------------------------------------------------------------------
# Phase 558 additions
# ---------------------------------------------------------------------------


def _round_to_half(value: float) -> float:
    """Round to nearest 0.5, minimum 0.5."""
    rounded = round(value * 2.0) / 2.0
    return max(0.5, rounded)


def round_to_available_tablet(
    dose_mg: float,
    available_tablets_mg: list[float],
) -> dict:
    """Find the closest available tablet strength and compute rounding details.

    Parameters
    ----------
    dose_mg:
        Target dose in mg. Must be > 0.
    available_tablets_mg:
        Non-empty list of available tablet strengths (mg). All must be > 0.

    Returns
    -------
    dict with keys:
        recommended_dose_mg, tablet_strength_mg, n_tablets, deviation_pct
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not available_tablets_mg:
        raise ValueError("available_tablets_mg must not be empty")
    if any(t <= 0 for t in available_tablets_mg):
        raise ValueError("all tablet strengths must be > 0")

    # Choose tablet strength closest to dose_mg
    best_tablet = min(available_tablets_mg, key=lambda t: abs(t - dose_mg))

    # Number of tablets, rounded to nearest 0.5, minimum 0.5
    raw_n = dose_mg / best_tablet
    n_tablets = _round_to_half(raw_n)

    recommended_dose_mg = n_tablets * best_tablet
    deviation_pct = abs(recommended_dose_mg - dose_mg) / dose_mg * 100.0

    return {
        "recommended_dose_mg": recommended_dose_mg,
        "tablet_strength_mg": best_tablet,
        "n_tablets": n_tablets,
        "deviation_pct": deviation_pct,
    }


def weight_based_dose_rounding(
    weight_kg: float,
    dose_mg_per_kg: float,
    available_tablets_mg: list[float],
    max_dose_mg: float | None = None,
) -> dict:
    """Compute a weight-based dose and round to available tablet strengths.

    Parameters
    ----------
    weight_kg:
        Patient weight in kg. Must be > 0.
    dose_mg_per_kg:
        Dose per kg body weight. Must be > 0.
    available_tablets_mg:
        Available tablet strengths (mg). All must be > 0.
    max_dose_mg:
        Optional maximum dose cap (mg). Must be > 0 if provided.

    Returns
    -------
    dict with keys:
        target_dose_mg, weight_kg, dose_mg_per_kg,
        recommended_dose_mg, tablet_strength_mg, n_tablets, deviation_pct
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if max_dose_mg is not None and max_dose_mg <= 0:
        raise ValueError("max_dose_mg must be > 0 if provided")

    target = weight_kg * dose_mg_per_kg
    if max_dose_mg is not None:
        target = min(target, max_dose_mg)

    rounding = round_to_available_tablet(target, available_tablets_mg)

    return {
        "target_dose_mg": target,
        "weight_kg": weight_kg,
        "dose_mg_per_kg": dose_mg_per_kg,
        **rounding,
    }


def bsa_based_dose_rounding(
    height_cm: float,
    weight_kg: float,
    dose_mg_per_m2: float,
    available_tablets_mg: list[float],
) -> dict:
    """Compute a BSA-based dose using the Mosteller formula and round to tablets.

    BSA (m²) = sqrt(height_cm * weight_kg / 3600)

    Parameters
    ----------
    height_cm:
        Patient height in cm. Must be > 0.
    weight_kg:
        Patient weight in kg. Must be > 0.
    dose_mg_per_m2:
        Dose per m² BSA. Must be > 0.
    available_tablets_mg:
        Available tablet strengths (mg). All must be > 0.

    Returns
    -------
    dict with keys:
        bsa_m2, target_dose_mg, recommended_dose_mg,
        tablet_strength_mg, n_tablets, deviation_pct
    """
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if dose_mg_per_m2 <= 0:
        raise ValueError("dose_mg_per_m2 must be > 0")

    bsa_m2 = math.sqrt(height_cm * weight_kg / 3600.0)
    target = bsa_m2 * dose_mg_per_m2

    rounding = round_to_available_tablet(target, available_tablets_mg)

    return {
        "bsa_m2": bsa_m2,
        "target_dose_mg": target,
        **rounding,
    }


def pediatric_dose_by_age(
    age_years: float,
    adult_dose_mg: float,
    method: str = "clark",
) -> float:
    """Estimate pediatric dose from age using established empirical rules.

    Methods
    -------
    clark:
        weight_kg = 2.5 * age_years + 8
        dose = (weight_kg / 68) * adult_dose_mg

    young:
        dose = age / (age + 12) * adult_dose_mg

    diling:
        age >= 12: dose = age / 20 * adult_dose_mg
        age <  12: dose = age / (age + 12) * adult_dose_mg  (Young's rule)

    Parameters
    ----------
    age_years:
        Patient age in years. Must be >= 0.
    adult_dose_mg:
        Standard adult dose in mg. Must be > 0.
    method:
        One of 'clark', 'young', 'diling'.

    Returns
    -------
    float
        Estimated pediatric dose in mg.
    """
    if age_years < 0:
        raise ValueError("age_years must be >= 0")
    if adult_dose_mg <= 0:
        raise ValueError("adult_dose_mg must be > 0")
    valid_methods = {"clark", "young", "diling"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    if method == "clark":
        weight_kg = 2.5 * age_years + 8.0
        return (weight_kg / 68.0) * adult_dose_mg

    if method == "young":
        return age_years / (age_years + 12.0) * adult_dose_mg

    # diling
    if age_years >= 12.0:
        return age_years / 20.0 * adult_dose_mg
    else:
        return age_years / (age_years + 12.0) * adult_dose_mg


def dose_rounding_report(
    drug_name: str,
    target_dose_mg: float,
    available_tablets_mg: list[float],
    tolerance_pct: float = 20.0,
) -> dict:
    """Generate a clinical dose rounding report.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_dose_mg:
        Target dose in mg. Must be > 0.
    available_tablets_mg:
        Available tablet strengths (mg). All must be > 0.
    tolerance_pct:
        Acceptable deviation tolerance in percent. Must be >= 0.

    Returns
    -------
    dict with keys:
        drug_name, target_dose_mg, recommended_dose_mg,
        tablet_strength_mg, n_tablets, deviation_pct,
        within_tolerance, notes
    """
    if tolerance_pct < 0:
        raise ValueError("tolerance_pct must be >= 0")

    rounding = round_to_available_tablet(target_dose_mg, available_tablets_mg)
    within_tolerance = rounding["deviation_pct"] <= tolerance_pct

    if within_tolerance:
        notes = (
            f"Recommended dose is within {tolerance_pct:.0f}% of target. "
            f"Administer {rounding['n_tablets']} x {rounding['tablet_strength_mg']} mg tablet(s)."
        )
    else:
        notes = (
            f"Deviation ({rounding['deviation_pct']:.1f}%) exceeds tolerance ({tolerance_pct:.0f}%). "
            "Consider alternative tablet strengths or liquid formulation."
        )

    return {
        "drug_name": drug_name,
        "target_dose_mg": target_dose_mg,
        "recommended_dose_mg": rounding["recommended_dose_mg"],
        "tablet_strength_mg": rounding["tablet_strength_mg"],
        "n_tablets": rounding["n_tablets"],
        "deviation_pct": rounding["deviation_pct"],
        "within_tolerance": within_tolerance,
        "notes": notes,
    }
