"""
Phase 953 — Dose Rounding Optimization
Optimize dose rounding to available tablet/capsule strengths while maintaining
therapeutic adequacy using 1-compartment steady-state analytical model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DoseRoundingResult", "optimize_dose_rounding", "compare_dosing_regimens"]


@dataclass(frozen=True)
class DoseRoundingResult:
    drug_name: str
    ideal_dose_mg: float
    recommended_dose_mg: float
    recommended_strength_mg: float
    n_units: float
    css_max_mg_L: float
    css_min_mg_L: float
    within_therapeutic_window: bool
    deviation_from_ideal_pct: float
    alternative_doses: tuple
    notes: str


def _compute_css(dose_mg: float, cl_L_per_h: float, vd_L: float, dosing_interval_h: float) -> tuple:
    """Compute Css_max and Css_min at steady state for 1-cpt model."""
    ke = cl_L_per_h / vd_L
    exp_term = math.exp(-ke * dosing_interval_h)
    # Css_max (just after dose absorption at t=0+)
    css_max = dose_mg / (vd_L * (1.0 - exp_term))
    css_min = css_max * exp_term
    return css_max, css_min


def _generate_candidates(
    available_strengths_mg: list[float], allow_multiple_tablets: bool, max_tablets: int
) -> list[tuple]:
    """
    Generate (dose_mg, strength_mg, n_units) candidate tuples.
    n_units values: 0.5, 1, 1.5, 2, 3, 4 (limited by max_tablets).
    """
    if allow_multiple_tablets:
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        multipliers = [m for m in multipliers if m <= max_tablets]
    else:
        multipliers = [1.0]

    seen = set()
    candidates = []
    for strength in available_strengths_mg:
        for n in multipliers:
            dose = strength * n
            key = round(dose, 6)
            if key not in seen:
                seen.add(key)
                candidates.append((dose, strength, n))
    return candidates


def optimize_dose_rounding(
    drug_name: str,
    ideal_dose_mg: float,
    available_strengths_mg: list[float],
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    dosing_interval_h: float = 12.0,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    allow_multiple_tablets: bool = True,
    max_tablets: int = 4,
) -> DoseRoundingResult:
    """
    Find the best commercially available dose within the therapeutic window.
    """
    # Validation
    if ideal_dose_mg <= 0:
        raise ValueError("ideal_dose_mg must be > 0")
    if len(available_strengths_mg) < 1:
        raise ValueError("available_strengths_mg must not be empty")
    if not all(s > 0 for s in available_strengths_mg):
        raise ValueError("all strengths must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mec_mg_L >= mtc_mg_L:
        raise ValueError("mec_mg_L must be < mtc_mg_L")

    candidates = _generate_candidates(available_strengths_mg, allow_multiple_tablets, max_tablets)

    # Evaluate each candidate
    evaluated = []
    for dose_mg, strength_mg, n_units in candidates:
        css_max, css_min = _compute_css(dose_mg, cl_L_per_h, vd_L, dosing_interval_h)
        in_window = (css_min >= mec_mg_L) and (css_max <= mtc_mg_L)
        deviation = abs(dose_mg - ideal_dose_mg) / ideal_dose_mg
        evaluated.append((dose_mg, strength_mg, n_units, css_max, css_min, in_window, deviation))

    # Prefer: in-window first, then by deviation
    in_window_cands = [e for e in evaluated if e[5]]
    out_window_cands = [e for e in evaluated if not e[5]]

    if in_window_cands:
        # Among in-window, pick lowest deviation
        in_window_cands.sort(key=lambda e: e[6])
        best = in_window_cands[0]
        # Alternatives: next up to 3 in-window
        alts = in_window_cands[1:4]
    else:
        # No dose in window — pick closest to ideal
        out_window_cands.sort(key=lambda e: e[6])
        best = out_window_cands[0]
        alts = out_window_cands[1:4]

    dose_mg, strength_mg, n_units, css_max, css_min, in_window, deviation = best
    deviation_pct = deviation * 100.0

    alt_doses = tuple(round(e[0], 4) for e in alts)

    # Build notes
    if in_window:
        notes = (
            f"Dose {dose_mg:.1f} mg ({n_units}x {strength_mg:.1f} mg) "
            f"is within therapeutic window. "
            f"Css_max={css_max:.3f} mg/L, Css_min={css_min:.3f} mg/L."
        )
    else:
        notes = (
            f"No available dose combination falls within the therapeutic window. "
            f"Closest dose {dose_mg:.1f} mg selected (deviation "
            f"{deviation_pct:.1f}% from ideal). "
            f"Css_max={css_max:.3f} mg/L, Css_min={css_min:.3f} mg/L."
        )

    return DoseRoundingResult(
        drug_name=drug_name,
        ideal_dose_mg=ideal_dose_mg,
        recommended_dose_mg=round(dose_mg, 6),
        recommended_strength_mg=strength_mg,
        n_units=n_units,
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        within_therapeutic_window=bool(in_window),
        deviation_from_ideal_pct=deviation_pct,
        alternative_doses=alt_doses,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    ideal_dose_mg: float,
    available_strengths_mg: list[float],
    intervals_h: list[float] | None = None,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    allow_multiple_tablets: bool = True,
    max_tablets: int = 4,
) -> list[DoseRoundingResult]:
    """
    Compare dose rounding across multiple dosing intervals.
    Returns results sorted by within_therapeutic_window (True first),
    then by deviation_from_ideal_pct (ascending).
    """
    if intervals_h is None:
        intervals_h = [8.0, 12.0, 24.0]

    results = []
    for interval in intervals_h:
        result = optimize_dose_rounding(
            drug_name=drug_name,
            ideal_dose_mg=ideal_dose_mg,
            available_strengths_mg=available_strengths_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            dosing_interval_h=interval,
            mec_mg_L=mec_mg_L,
            mtc_mg_L=mtc_mg_L,
            allow_multiple_tablets=allow_multiple_tablets,
            max_tablets=max_tablets,
        )
        results.append(result)

    # Sort: within_therapeutic_window=True first, then by deviation ascending
    results.sort(key=lambda r: (not r.within_therapeutic_window, r.deviation_from_ideal_pct))
    return results
