"""Phase 754 — Drug Half-Life Optimization for Ideal Dosing Regimens.

Given therapeutic targets (Css_max, Css_min), finds the optimal elimination half-life
that minimizes fluctuation while maintaining the therapeutic range. Useful for drug
design and formulation development.

Note: half_life_calculator.py (Phase 96) simply calculates t½ from Vd/CL.
This module OPTIMIZES the target t½ for an ideal dosing regimen via grid search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HalfLifeOptimizationResult:
    """Result from half-life optimization grid search."""

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    optimal_t_half_h: float
    optimal_css_max: float
    optimal_css_min: float
    optimal_css_avg: float
    optimal_fluctuation_pct: float
    optimal_composite_score: float
    in_therapeutic_range: bool
    all_t_halfs: str  # comma-separated 5 best t_half values
    notes: str


def _log_space(start: float, stop: float, n: int) -> list:
    """Generate n log-spaced values between start and stop."""
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    step = (log_stop - log_start) / (n - 1) if n > 1 else 0.0
    return [10.0 ** (log_start + i * step) for i in range(n)]


def _compute_css(
    t_half_h: float,
    dose_mg: float,
    dosing_interval_h: float,
    vd_L: float,
    f_oral: float,
) -> tuple:
    """Compute steady-state Css_max, Css_min, Css_avg for 1-compartment model.

    Returns (css_max, css_min, css_avg).
    """
    ke = math.log(2.0) / t_half_h
    tau = dosing_interval_h

    # Analytical steady-state: Css_max = F*Dose/Vd / (1 - exp(-ke*tau))
    exp_neg = math.exp(-ke * tau)
    if abs(1.0 - exp_neg) < 1e-15:
        # Very long t_half relative to tau: Css ≈ F*Dose/(Vd*ke*tau)
        css_max = f_oral * dose_mg / (vd_L * ke * tau)
    else:
        css_max = f_oral * dose_mg / (vd_L * (1.0 - exp_neg))

    css_min = css_max * exp_neg

    # Css_avg = F*Dose / (Vd * ke * tau) = F*Dose*t_half / (Vd * ln2 * tau)
    css_avg = f_oral * dose_mg * t_half_h / (vd_L * math.log(2.0) * tau)

    return css_max, css_min, css_avg


def _composite_score(
    css_max: float,
    css_min: float,
    css_avg: float,
    mec_mg_L: float,
    mtc_mg_L: float,
) -> tuple:
    """Compute composite score (0–1) from fluctuation and TIR.

    Returns (score, fluctuation_pct, tir_score).
    """
    if css_avg > 0:
        fluctuation_pct = (css_max - css_min) / css_avg * 100.0
    else:
        fluctuation_pct = 200.0

    tir_score = 1.0 if (css_min >= mec_mg_L and css_max <= mtc_mg_L) else 0.0

    # Cap fluctuation at 200 for scoring purposes
    fluct_capped = min(fluctuation_pct, 200.0)
    composite = (1.0 - fluct_capped / 200.0) * 0.5 + tir_score * 0.5

    return composite, fluctuation_pct, tir_score


def scan_t_half_landscape(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    mec_mg_L: float,
    mtc_mg_L: float,
    vd_L: float = 50.0,
    f_oral: float = 1.0,
    t_half_range: tuple = (0.5, 200.0),
    n_candidates: int = 100,
) -> list:
    """Evaluate composite score across a grid of t_half values.

    Returns list of dicts, each with:
        t_half_h, css_max, css_min, css_avg, fluctuation_pct, tir_score, composite_score
    """
    candidates = _log_space(t_half_range[0], t_half_range[1], n_candidates)
    results = []
    for t_half in candidates:
        css_max, css_min, css_avg = _compute_css(t_half, dose_mg, dosing_interval_h, vd_L, f_oral)
        composite, fluct_pct, tir = _composite_score(css_max, css_min, css_avg, mec_mg_L, mtc_mg_L)
        results.append(
            {
                "t_half_h": t_half,
                "css_max": css_max,
                "css_min": css_min,
                "css_avg": css_avg,
                "fluctuation_pct": fluct_pct,
                "tir_score": tir,
                "composite_score": composite,
            }
        )
    return results


def optimize_half_life(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    mec_mg_L: float,
    mtc_mg_L: float,
    vd_L: float = 50.0,
    f_oral: float = 1.0,
    t_half_range: tuple = (0.5, 200.0),
    n_candidates: int = 100,
) -> HalfLifeOptimizationResult:
    """Find optimal elimination half-life via grid search.

    Maximises a composite score balancing low fluctuation and therapeutic range coverage.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be > mec_mg_L")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    landscape = scan_t_half_landscape(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        vd_L=vd_L,
        f_oral=f_oral,
        t_half_range=t_half_range,
        n_candidates=n_candidates,
    )

    # Sort by composite score descending, then pick best
    sorted_landscape = sorted(landscape, key=lambda x: x["composite_score"], reverse=True)
    best = sorted_landscape[0]

    # Top-5 t_half values for the result string
    top5 = sorted_landscape[:5]
    all_t_halfs_str = ", ".join(f"{r['t_half_h']:.2f}" for r in top5)

    in_range = best["css_min"] >= mec_mg_L and best["css_max"] <= mtc_mg_L

    notes = (
        f"Optimal t½={best['t_half_h']:.2f} h; "
        f"Css_max={best['css_max']:.3f}, Css_min={best['css_min']:.3f} mg/L; "
        f"Fluctuation={best['fluctuation_pct']:.1f}%; "
        f"TIR={'yes' if in_range else 'no'}; "
        f"Composite={best['composite_score']:.3f}"
    )

    return HalfLifeOptimizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        optimal_t_half_h=best["t_half_h"],
        optimal_css_max=best["css_max"],
        optimal_css_min=best["css_min"],
        optimal_css_avg=best["css_avg"],
        optimal_fluctuation_pct=best["fluctuation_pct"],
        optimal_composite_score=best["composite_score"],
        in_therapeutic_range=in_range,
        all_t_halfs=all_t_halfs_str,
        notes=notes,
    )
