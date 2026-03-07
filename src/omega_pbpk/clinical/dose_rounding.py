"""Phase 181 — Dose Rounding

Round continuous pharmacokinetic-optimised doses to available tablet/capsule
strengths.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DoseRoundingResult", "round_dose", "select_optimal_regimen"]


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
