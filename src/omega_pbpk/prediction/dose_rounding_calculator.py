"""
Phase 206 — Dose Rounding Calculator (prediction/dose_rounding_calculator.py)

Calculate practical dose rounding to available tablet/capsule strengths,
minimizing dose error while preferring simpler regimens (fewer tablets).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoseRoundingResult:
    """Result of a dose rounding calculation."""

    drug_name: str
    calculated_dose_mg: float
    rounded_dose_mg: float
    tablet_combination: tuple
    n_tablets: int
    dose_error_pct: float
    acceptable: bool
    available_strengths_mg: tuple
    notes: str


def _generate_combinations(
    available_strengths_mg: tuple,
    max_tablets: int,
) -> list:
    """Generate all possible tablet combinations up to max_tablets total.

    Returns list of tuples (sorted counts per strength), each representing
    a combination.
    """
    strengths = list(available_strengths_mg)
    n_strengths = len(strengths)

    combinations = []
    # counts[i] = number of tablets of strengths[i], total sum <= max_tablets
    # Iterate via bounded loops using recursion-free approach
    # max_tablets is typically small (1–6), so exhaustive search is fine

    def _recurse(idx: int, remaining: int, current: list) -> None:
        if idx == n_strengths:
            if sum(current) >= 1:
                combinations.append(tuple(current))
            return
        for count in range(0, remaining + 1):
            current.append(count)
            _recurse(idx + 1, remaining - count, current)
            current.pop()

    _recurse(0, max_tablets, [])
    return combinations


def round_to_available_dose(
    drug_name: str,
    calculated_dose_mg: float,
    available_strengths_mg: tuple,
    max_tablets: int = 4,
    max_error_pct: float = 20.0,
) -> DoseRoundingResult:
    """Find the best tablet combination closest to the calculated dose.

    Selection criteria:
    1. Prefer fewest tablets (simplest regimen).
    2. Among combinations with fewest tablets, pick lowest |error%|.
    3. Acceptable if |error%| <= max_error_pct (default 20%).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    calculated_dose_mg:
        Target dose calculated (e.g., from mg/kg × weight).
    available_strengths_mg:
        Tuple of available tablet strengths in mg (e.g., (25.0, 50.0, 100.0)).
    max_tablets:
        Maximum number of tablets allowed per dose. Default 4.
    max_error_pct:
        Acceptable dose error threshold (%). Default 20.0.

    Returns
    -------
    DoseRoundingResult
    """
    if calculated_dose_mg <= 0:
        raise ValueError("calculated_dose_mg must be > 0")
    if not available_strengths_mg:
        raise ValueError("available_strengths_mg must not be empty")

    strengths = tuple(float(s) for s in available_strengths_mg)

    # Enumerate all valid combinations
    count_combos = _generate_combinations(strengths, max_tablets)

    # Compute rounded dose and error for each combination
    candidates = []
    for counts in count_combos:
        rounded = sum(cnt * s for cnt, s in zip(counts, strengths))
        n_total = sum(counts)
        error_pct = (rounded - calculated_dose_mg) / calculated_dose_mg * 100.0
        candidates.append((n_total, abs(error_pct), error_pct, rounded, counts))

    # Sort: primary = n_tablets ASC, secondary = |error_pct| ASC
    candidates.sort(key=lambda x: (x[0], x[1]))

    # Pick best
    best_n, best_abs_err, best_err_pct, best_rounded, best_counts = candidates[0]

    # Build tablet_combination tuple
    tablet_combo = tuple(s for cnt, s in zip(best_counts, strengths) for _ in range(cnt))

    acceptable = abs(best_err_pct) <= max_error_pct

    notes = (
        f"Calculated: {calculated_dose_mg:.1f} mg, "
        f"Rounded: {best_rounded:.1f} mg, "
        f"Error: {best_err_pct:.1f}%, "
        f"Tablets: {best_n}, "
        f"Acceptable: {acceptable}."
    )

    return DoseRoundingResult(
        drug_name=drug_name,
        calculated_dose_mg=calculated_dose_mg,
        rounded_dose_mg=best_rounded,
        tablet_combination=tablet_combo,
        n_tablets=best_n,
        dose_error_pct=best_err_pct,
        acceptable=acceptable,
        available_strengths_mg=strengths,
        notes=notes,
    )


def screen_dose_adjustments(
    drug_name: str,
    weight_kg: float,
    mg_per_kg: float,
    available_strengths_mg: tuple,
    weight_range_kg: list,
    max_tablets: int = 4,
    max_error_pct: float = 20.0,
) -> list:
    """Screen dose rounding across a range of patient weights.

    For each weight in weight_range_kg, calculates the target dose
    (weight * mg_per_kg) and finds the best rounding.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    weight_kg:
        Reference weight (unused in iteration; kept for API symmetry).
    mg_per_kg:
        Dose in mg/kg.
    available_strengths_mg:
        Tuple of available tablet strengths in mg.
    weight_range_kg:
        List of patient weights to screen (kg).
    max_tablets:
        Maximum tablets per dose.
    max_error_pct:
        Acceptable error threshold (%).

    Returns
    -------
    List of DoseRoundingResult, one per weight in weight_range_kg.
    """
    if not weight_range_kg:
        return []

    results = []
    for w in weight_range_kg:
        calc_dose = w * mg_per_kg
        result = round_to_available_dose(
            drug_name=drug_name,
            calculated_dose_mg=calc_dose,
            available_strengths_mg=available_strengths_mg,
            max_tablets=max_tablets,
            max_error_pct=max_error_pct,
        )
        results.append(result)
    return results
