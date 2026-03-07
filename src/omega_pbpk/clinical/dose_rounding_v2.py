"""Dose rounding v2 — Phase 780.

Practical dose selection for clinical use with half-tablet support
and comprehensive practicality assessment.
Pure Python only (stdlib math + dataclasses, no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ClinicalDoseResult",
    "select_clinical_dose",
    "find_best_tablet_combination",
    "batch_select_clinical_doses",
]

_DEFAULT_STRENGTHS: list[float] = [25.0, 50.0, 100.0, 200.0, 250.0, 500.0]


@dataclass
class ClinicalDoseResult:
    """Result of a clinical dose rounding / selection calculation."""

    drug_name: str
    target_dose_mg: float
    rounded_dose_mg: float
    available_strengths: list[float]
    selected_strength_mg: float
    n_tablets: float
    dose_error_pct: float
    is_practical: bool
    notes: str


def _half_tablet_candidates(target: float, strength: float) -> list[float]:
    """Return n_tablet candidates in 0.5 increments up to a reasonable ceiling."""
    if strength <= 0.0:
        return []
    approx = target / strength
    ceiling = math.ceil(approx) + 1
    candidates: list[float] = []
    n = 0.5
    while n <= ceiling + 0.5:
        candidates.append(n)
        n += 0.5
    return candidates


def find_best_tablet_combination(
    target_dose_mg: float,
    available_strengths_mg: list[float],
) -> tuple[float, float, float]:
    """Find tablet strength and count minimising dose error.

    Parameters
    ----------
    target_dose_mg:
        Target dose in mg (must be > 0).
    available_strengths_mg:
        Non-empty list of tablet strengths in mg.

    Returns
    -------
    tuple (selected_strength_mg, n_tablets, actual_dose_mg)
    """
    if target_dose_mg <= 0.0:
        raise ValueError("target_dose_mg must be positive")
    if not available_strengths_mg:
        raise ValueError("available_strengths_mg must not be empty")

    best_strength = available_strengths_mg[0]
    best_n = 1.0
    best_actual = best_strength
    best_error = abs(best_actual - target_dose_mg)

    for strength in available_strengths_mg:
        if strength <= 0.0:
            continue
        for n in _half_tablet_candidates(target_dose_mg, strength):
            if n <= 0.0:
                continue
            actual = strength * n
            error = abs(actual - target_dose_mg)
            if error < best_error:
                best_error = error
                best_strength = strength
                best_n = n
                best_actual = actual

    return (best_strength, best_n, best_actual)


def select_clinical_dose(
    drug_name: str,
    target_dose_mg: float,
    available_strengths_mg: list[float] | None = None,
    max_tablets: int = 3,
    max_dose_error_pct: float = 15.0,
) -> ClinicalDoseResult:
    """Select the nearest practical tablet combination for a target dose.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_dose_mg:
        Desired dose in mg (must be > 0).
    available_strengths_mg:
        Available tablet strengths. Defaults to [25, 50, 100, 200, 250, 500].
    max_tablets:
        Maximum number of tablets per dose.
    max_dose_error_pct:
        Maximum acceptable dose error (%) for the result to be practical.

    Returns
    -------
    ClinicalDoseResult
    """
    if target_dose_mg <= 0.0:
        raise ValueError("target_dose_mg must be positive")

    strengths: list[float] = (
        list(available_strengths_mg)
        if available_strengths_mg is not None
        else list(_DEFAULT_STRENGTHS)
    )

    if not strengths:
        raise ValueError("available_strengths_mg must not be empty")

    selected_strength, n_tablets, rounded_dose_mg = find_best_tablet_combination(
        target_dose_mg, strengths
    )

    dose_error_pct = abs(rounded_dose_mg - target_dose_mg) / target_dose_mg * 100.0

    is_practical = dose_error_pct <= max_dose_error_pct and n_tablets <= float(max_tablets)

    notes_parts: list[str] = [
        f"Selected: {n_tablets:.1f} x {selected_strength:.0f} mg = {rounded_dose_mg:.1f} mg"
    ]
    if is_practical:
        notes_parts.append(
            f"Dose error {dose_error_pct:.1f}% within {max_dose_error_pct:.0f}% threshold."
        )
    else:
        if dose_error_pct > max_dose_error_pct:
            notes_parts.append(
                f"Dose error {dose_error_pct:.1f}% exceeds {max_dose_error_pct:.0f}% threshold."
            )
        if n_tablets > float(max_tablets):
            notes_parts.append(f"Tablets required ({n_tablets:.1f}) exceeds max ({max_tablets}).")
        notes_parts.append("Not practical; consider additional strengths.")

    notes = " ".join(notes_parts)

    return ClinicalDoseResult(
        drug_name=drug_name,
        target_dose_mg=target_dose_mg,
        rounded_dose_mg=rounded_dose_mg,
        available_strengths=strengths,
        selected_strength_mg=selected_strength,
        n_tablets=n_tablets,
        dose_error_pct=dose_error_pct,
        is_practical=is_practical,
        notes=notes,
    )


def batch_select_clinical_doses(params_list: list[dict]) -> list[ClinicalDoseResult]:
    """Run select_clinical_dose for each parameter dict in params_list.

    Parameters
    ----------
    params_list:
        List of dicts; each dict is passed as kwargs to select_clinical_dose.

    Returns
    -------
    list[ClinicalDoseResult]
    """
    return [select_clinical_dose(**p) for p in params_list]
