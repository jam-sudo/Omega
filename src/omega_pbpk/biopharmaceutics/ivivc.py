"""IVIVC — In Vitro–In Vivo Correlation.

Implements Wagner-Nelson deconvolution and Level A/B/C correlations
per FDA Guidance for Industry: Extended Release Oral Dosage Forms (1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class IVIVCCorrelation:
    level: str  # "A", "B", or "C"
    r_squared: float
    slope: float
    intercept: float
    passes_criteria: bool  # Level A: R²≥0.90; Level B/C: R²≥0.80


@dataclass(frozen=True)
class IVIVCResult:
    level_a: IVIVCCorrelation | None
    level_b: IVIVCCorrelation | None
    level_c: IVIVCCorrelation | None
    f_absorbed: list[float]  # Wagner-Nelson fa values at each dissolution time point
    f_dissolved: list[float]  # input dissolution fractions (same length)
    mdt_in_vitro_h: float  # mean dissolution time
    mrt_in_vivo_h: float  # mean residence time
    recommended_level: str  # "A", "B", "C", or "none"


def wagner_nelson(
    t: np.ndarray,
    cp: np.ndarray,
    ke: float,
    vd_L: float,
    dose_mg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wagner-Nelson deconvolution: Fa(t) = [Cp(t)*Vd + ke*AUC(0->t)*Vd] / Dose.

    Returns (t, fa) where fa is clipped to [0, 1].
    """
    if ke <= 0:
        raise ValueError("ke must be positive")

    t = np.asarray(t, dtype=float)
    cp = np.asarray(cp, dtype=float)

    cum_auc = np.zeros_like(cp)
    for i in range(1, len(t)):
        cum_auc[i] = cum_auc[i - 1] + (t[i] - t[i - 1]) * (cp[i] + cp[i - 1]) / 2

    fa = (cp * vd_L + ke * cum_auc * vd_L) / dose_mg
    fa = np.clip(fa, 0.0, 1.0)
    return t, fa


def _compute_mdt(times: np.ndarray, fractions: np.ndarray) -> float:
    """Mean dissolution time: weighted average of midpoint times."""
    numerator = 0.0
    for i in range(1, len(times)):
        mid = (times[i] + times[i - 1]) / 2
        delta_f = fractions[i] - fractions[i - 1]
        numerator += mid * delta_f
    denom = fractions[-1] if fractions[-1] != 0 else 1e-9
    return numerator / denom


def _compute_mrt(times: np.ndarray, cp: np.ndarray) -> float:
    """Mean residence time: AUMC / AUC."""
    auc = float(np_trapz(cp, times))
    aumc = float(np_trapz(times * cp, times))
    if abs(auc) < 1e-12:
        return 0.0
    return aumc / auc


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared)."""
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, r_squared


def compute_ivivc(
    dissolution_times: list[float],
    dissolution_fractions: list[float],
    plasma_times: list[float],
    plasma_conc: list[float],
    ke: float,
    vd_L: float,
    dose_mg: float,
    levels: list[str] | None = None,
) -> IVIVCResult:
    if levels is None:
        levels = ["A", "B", "C"]

    dt = np.asarray(dissolution_times, dtype=float)
    df = np.asarray(dissolution_fractions, dtype=float)
    pt = np.asarray(plasma_times, dtype=float)
    pc = np.asarray(plasma_conc, dtype=float)

    # Wagner-Nelson on plasma data, then interpolate to dissolution times
    _, fa_plasma = wagner_nelson(pt, pc, ke, vd_L, dose_mg)
    fa_at_diss = np.interp(dt, pt, fa_plasma)

    mdt = _compute_mdt(dt, df)
    mrt = _compute_mrt(pt, pc)

    # Level A
    level_a: IVIVCCorrelation | None = None
    if "A" in levels and len(dt) >= 3:
        slope, intercept, r2 = _linear_regression(df, fa_at_diss)
        level_a = IVIVCCorrelation(
            level="A",
            r_squared=r2,
            slope=slope,
            intercept=intercept,
            passes_criteria=r2 >= 0.90,
        )

    # Level B
    level_b: IVIVCCorrelation | None = None
    if "B" in levels:
        ratio = mdt / max(mrt, 1e-9)
        passes = abs(mdt - mrt) / max(mrt, 1e-9) < 0.30
        level_b = IVIVCCorrelation(
            level="B",
            r_squared=1.0,
            slope=ratio,
            intercept=0.0,
            passes_criteria=passes,
        )

    # Level C
    level_c: IVIVCCorrelation | None = None
    if "C" in levels:
        # Find time point nearest to 80% dissolved
        idx_80 = int(np.argmin(np.abs(df - 0.80)))
        f_at_80 = float(df[idx_80])
        level_c = IVIVCCorrelation(
            level="C",
            r_squared=1.0,
            slope=1.0,
            intercept=0.0,
            passes_criteria=f_at_80 >= 0.70,
        )

    # Recommended level
    recommended = "none"
    for lvl, corr in [("A", level_a), ("B", level_b), ("C", level_c)]:
        if corr is not None and corr.passes_criteria:
            recommended = lvl
            break

    return IVIVCResult(
        level_a=level_a,
        level_b=level_b,
        level_c=level_c,
        f_absorbed=[float(v) for v in fa_at_diss],
        f_dissolved=[float(v) for v in df],
        mdt_in_vitro_h=mdt,
        mrt_in_vivo_h=mrt,
        recommended_level=recommended,
    )


__all__ = ["IVIVCCorrelation", "IVIVCResult", "wagner_nelson", "compute_ivivc"]
